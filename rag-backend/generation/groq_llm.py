# generation/groq_llm.py
#
# CHANGES vs original:
#   - EntityMemory REMOVED — regex NER was capturing "January", "He", "The"
#     as "entities" and polluting every system prompt with noise.
#     Replaced with a cleaner RollingSummary: every 5 turns, a 2-sentence
#     summary is generated (by the LLM, not regex) and injected into context.
#     This is optional — defaults off for speed, can be enabled per-instance.
#   - ChatHistory.clear_turns_only() kept (UI uses it for "Clear Chat (keep facts)")
#   - All class names, method signatures, and __all__ unchanged
#   - LLMFactory unchanged — OllamaLLM still registers itself via ollama_llm.py

import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, MAX_TURNS


# ─────────────────────────────────────────────────────────
# ROLLING SUMMARY  (replaces EntityMemory)
# ─────────────────────────────────────────────────────────

class RollingSummary:
    """
    Lightweight session context that summarises the conversation every
    N turns so long-running sessions don't lose early context.

    WHY NOT ENTITY MEMORY?
    ──────────────────────
    The original EntityMemory used regex to extract "named entities".
    In practice it captured stop words, dates, and generic phrases —
    then injected all of them into every system prompt as "facts".
    This added noise, not signal.

    THIS APPROACH:
    ──────────────
    Every summarize_every turns, we ask the LLM to write a 2-sentence
    summary of what's been discussed so far. This summary is:
      - Injected into the system prompt on every call
      - Replaced with a new summary after the next N turns

    This is opt-in (disabled by default for speed). Enable it in GroqLLM
    via use_rolling_summary=True when lower latency matters less.

    BACKWARD COMPATIBILITY:
    ────────────────────────
    The old EntityMemory methods are stubbed so code that calls
    entity_memory.update_from_text() or len(entity_memory) won't crash.
    """

    def __init__(self, summarize_every: int = 5):
        self.summarize_every = summarize_every
        self._summary: str = ""
        self._turn_count   = 0

    # ── stubs for backward compat ─────────────────────────

    def update_from_text(self, text: str) -> None:
        """No-op — kept so old code calling this doesn't crash."""
        pass

    def to_prompt_block(self) -> str:
        if not self._summary:
            return ""
        return f"\n\n[Conversation summary: {self._summary}]"

    def clear(self) -> None:
        self._summary    = ""
        self._turn_count = 0

    def get_all(self) -> dict:
        return {"summary": self._summary}

    def __len__(self) -> int:
        """Returns 1 if a summary exists (for stats display compatibility)."""
        return 1 if self._summary else 0

    # ── real logic ────────────────────────────────────────

    def maybe_update(self, turns: list[dict], client: "Groq", model: str) -> None:
        """
        Called after every assistant reply.
        Generates a new summary every summarize_every turns.
        """
        self._turn_count += 1
        if self._turn_count % self.summarize_every != 0:
            return
        if len(turns) < 4:
            return

        # Use last N turns as input — avoid sending entire history
        recent = turns[-min(len(turns), self.summarize_every * 2):]
        convo  = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        )
        prompt = (
            f"Summarise this conversation in exactly 2 sentences, "
            f"focusing on key facts and topics discussed:\n\n{convo}"
        )

        try:
            resp = client.chat.completions.create(
                model      = model,
                messages   = [{"role": "user", "content": prompt}],
                max_tokens = 100,
                temperature= 0.2,
            )
            self._summary = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [SUMMARY] Could not generate summary: {e}")


# ─────────────────────────────────────────────────────────
# CHAT HISTORY  (sliding window + optional rolling summary)
# ─────────────────────────────────────────────────────────

class ChatHistory:
    """
    Manages conversation memory with optional rolling summary.

    Layer 1 — Sliding Window
        Keeps only the last max_turns user+assistant pairs.
        Prevents the context window from overflowing in long sessions.

    Layer 2 — Rolling Summary (optional, replaces EntityMemory)
        Every N turns, generates a 2-sentence LLM summary and injects
        it into the system prompt so early context isn't lost.

    BACKWARD COMPATIBILITY:
        entity_memory attribute still exists (is a RollingSummary stub)
        clear_turns_only() still works
        set_system() still works
    """

    def __init__(
        self,
        system_prompt      : str  = None,
        max_turns          : int  = MAX_TURNS,
        use_rolling_summary: bool = False,
    ):
        self.max_turns     = max_turns
        self._system       = system_prompt
        self._turns: list[dict]  = []

        # entity_memory kept as attribute for backward compat
        # (RollingSummary stubs update_from_text + __len__)
        self.entity_memory = RollingSummary()
        self._rolling      = self.entity_memory  # alias
        self._use_rolling  = use_rolling_summary

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def add_user(self, content: str) -> None:
        self._turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str, client=None, model: str = None) -> None:
        self._turns.append({"role": "assistant", "content": content})
        if self._use_rolling and client and model:
            self._rolling.maybe_update(self._turns, client, model)

    def _trim(self) -> None:
        """Sliding window — drop oldest pairs when limit exceeded."""
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def to_messages(self) -> list[dict]:
        """Return full message list ready for the LLM API."""
        messages: list[dict] = []
        if self._system:
            system_content = self._system + self._rolling.to_prompt_block()
            messages.append({"role": "system", "content": system_content})
        messages.extend(self._turns)
        return messages

    def clear(self) -> None:
        """Wipe turn history AND rolling summary (full reset)."""
        self._turns = []
        self._rolling.clear()

    def clear_turns_only(self) -> None:
        """Wipe only turn history, preserve rolling summary."""
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)


# ─────────────────────────────────────────────────────────
# BASE LLM
# ─────────────────────────────────────────────────────────

class BaseLLM:
    """
    Abstract base class for all LLM providers.
    Every LLM must implement generate() and stream().
    """

    def __init__(self):
        self.model_name = "base"

    def generate(self, prompt, system_prompt=None, history=None, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement generate()")

    def stream(self, prompt, system_prompt=None, history=None, **kwargs):
        raise NotImplementedError("Subclasses must implement stream()")

    def get_info(self) -> dict:
        return {"model": self.model_name, "provider": "base"}


# ─────────────────────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────────────────────

class GroqLLM(BaseLLM):
    """
    Groq Cloud LLM — extremely fast inference via GroqChip.
    Requires GROQ_API_KEY in .env

    ✅ Very fast (LPU inference)  ✅ OpenAI-compatible API
    ✅ Free tier available        ❌ Needs internet + API key

    Default model : llama-3.3-70b-versatile
    Memory        : sliding window (max_turns=8) + optional rolling summary
    """

    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma2-9b-it",
    ]

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. Answer questions clearly and concisely "
        "based on the context provided. If you don't know something, say so."
    )

    def __init__(
        self,
        model_name         : str   = GROQ_MODEL,
        system_prompt      : str   = None,
        temperature        : float = 0.3,   # lower = more factual (was 0.7)
        max_tokens         : int   = 1024,
        max_turns          : int   = MAX_TURNS,
        use_rolling_summary: bool  = False,
    ):
        super().__init__()
        self.model_name    = model_name
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

        print(f"  [GROQ] Initializing model: {model_name}")
        self.client  = Groq(api_key=GROQ_API_KEY)
        self.history = ChatHistory(
            system_prompt       = self.system_prompt,
            max_turns           = max_turns,
            use_rolling_summary = use_rolling_summary,
        )
        print(f"  [GROQ] ✅ Ready! Memory: sliding window ({max_turns} turns)")

    # ── HELPERS ───────────────────────────────────────────

    def _build_messages(
        self,
        prompt        : str,
        system_prompt : str  = None,
        history             = None,
        store_as      : str  = None,
    ) -> list[dict]:
        """
        Build the messages list for the API call.

        store_as — if provided, THIS string is stored in history (the raw
                   question), while `prompt` (the full context+question block)
                   is sent to the API only for the current turn.

        Why: we want history to contain clean Q->A pairs so the LLM can
        follow the conversation. Storing the full "Context:[...]\nQuestion:X"
        prompt in history adds noise and wastes context window.
        """
        active_history = history or self.history
        if system_prompt:
            active_history.set_system(system_prompt)

        # Store the clean question in history (not the full prompt+context)
        active_history.add_user(store_as if store_as else prompt)

        messages = active_history.to_messages()

        # For the current turn, replace the stored message with the full
        # prompt (with context) — only the LLM API call sees this, not history
        if store_as:
            messages[-1] = {"role": "user", "content": prompt}

        return messages

    # ── GENERATE (blocking) ───────────────────────────────

    def generate(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
        store_as      : str         = None,   # raw question to store in history
    ) -> dict:
        messages = self._build_messages(prompt, system_prompt, history, store_as=store_as)

        response = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = False,
        )

        content        = response.choices[0].message.content
        active_history = history or self.history
        # Pass client + model so rolling summary can self-trigger if enabled
        active_history.add_assistant(
            content, client=self.client, model=self.model_name
        )

        usage = {
            "prompt_tokens"    : response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens"     : response.usage.total_tokens,
        }
        return {"content": content, "model": self.model_name, "usage": usage}

    # ── STREAM (generator) ────────────────────────────────

    def stream(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
        store_as      : str         = None,   # raw question to store in history
    ):
        """
        Yields:
            str  — text tokens as they stream
            dict — final item: {"model": ..., "usage": {...}}
        """
        messages = self._build_messages(prompt, system_prompt, history, store_as=store_as)

        response_stream = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = True,
        )

        full_reply: list[str] = []
        usage_data: dict      = {}

        for chunk in response_stream:
            # Some SDK versions surface usage on the final chunk
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_data = {
                    "prompt_tokens"    : chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens"     : chunk.usage.total_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_reply.append(delta.content)
                    yield delta.content

        complete_reply = "".join(full_reply)
        active_history = history or self.history
        active_history.add_assistant(
            complete_reply, client=self.client, model=self.model_name
        )

        yield {"model": self.model_name, "usage": usage_data}

    # ── CONVENIENCE ───────────────────────────────────────

    def chat(self, message: str, **kwargs) -> str:
        return self.generate(message, **kwargs)["content"]

    def reset_history(self) -> None:
        """Full reset — clears turns AND rolling summary."""
        self.history.clear()
        print(f"  [GROQ] Conversation history cleared.")

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.history.set_system(prompt)

    def get_info(self) -> dict:
        return {
            "model"       : self.model_name,
            "provider"    : "groq-cloud",
            "temperature" : self.temperature,
            "max_tokens"  : self.max_tokens,
            "history_len" : len(self.history),
            "entity_facts": len(self.history.entity_memory),  # compat
        }


# ─────────────────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────────────────

class LLMFactory:
    PROVIDERS: dict = {"groq": GroqLLM}

    @staticmethod
    def get(provider: str = "groq", **kwargs) -> BaseLLM:
        provider = provider.lower().strip()
        if provider not in LLMFactory.PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(LLMFactory.PROVIDERS.keys())}"
            )
        return LLMFactory.PROVIDERS[provider](**kwargs)

    @staticmethod
    def available_providers() -> list[str]:
        return list(LLMFactory.PROVIDERS.keys())


__all__ = [
    "RollingSummary", "ChatHistory",
    "BaseLLM", "GroqLLM", "LLMFactory",
]