# generation/ollama_llm.py
#
# CHANGES vs original:
#   - store_as param added to generate() and stream() — matches GroqLLM interface
#   - _build_messages() updated to support store_as (same logic as GroqLLM)
#     Keeps history clean: raw question stored, full context+question sent to API

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from generation.groq_llm import BaseLLM, ChatHistory, LLMFactory
from config import OLLAMA_MODEL


class OllamaLLM(BaseLLM):
    """
    Local Ollama LLM — fully offline via Ollama daemon.
    Requires: `ollama serve` running + model pulled locally.

    ✅ Free  ✅ Private  ✅ No internet
    ❌ Slower than Groq  ❌ Needs Ollama running
    """

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. Answer questions clearly and concisely "
        "based on the context provided. If you don't know something, say so."
    )

    def __init__(
        self,
        model_name    : str   = OLLAMA_MODEL,
        system_prompt : str   = None,
        temperature   : float = 0.7,
        max_tokens    : int   = 1024,
        max_turns     : int   = 20,
    ):
        super().__init__()
        self.model_name    = model_name
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

        print(f"  [OLLAMA] Initializing model: {model_name}")
        self._check_model()
        self.history = ChatHistory(
            system_prompt = self.system_prompt,
            max_turns     = max_turns
        )
        print(f"  [OLLAMA] Ready!")

    def _check_model(self) -> None:
        try:
            local_models = [m.model for m in ollama.list().models]
            names = [m.split(":")[0] for m in local_models]
            if self.model_name.split(":")[0] not in names:
                print(f"  [OLLAMA]  Model '{self.model_name}' not found. Run: ollama pull {self.model_name}")
            else:
                print(f"  [OLLAMA] Model found locally")
        except Exception as e:
            print(f"  [OLLAMA]  Could not connect to Ollama daemon: {e}")

    def _build_messages(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
        store_as      : str         = None,
    ) -> list[dict]:
        """
        store_as — raw question stored in history (not the full context+question prompt).
        This keeps conversation history clean across turns, same as GroqLLM.
        """
        active_history = history or self.history
        if system_prompt:
            active_history.set_system(system_prompt)

        active_history.add_user(store_as if store_as else prompt)
        messages = active_history.to_messages()

        # For this API call only, replace last message with the full prompt
        if store_as:
            messages[-1] = {"role": "user", "content": prompt}

        return messages

    def generate(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
        store_as      : str         = None,   # ← matches GroqLLM
    ) -> dict:
        messages = self._build_messages(prompt, system_prompt, history, store_as=store_as)

        response = ollama.chat(
            model    = self.model_name,
            messages = messages,
            options  = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens  or self.max_tokens,
            },
            stream   = False,
        )

        content        = response.message.content
        active_history = history or self.history
        active_history.add_assistant(content)

        usage = {
            "prompt_tokens"    : response.prompt_eval_count or 0,
            "completion_tokens": response.eval_count        or 0,
            "total_tokens"     : (response.prompt_eval_count or 0) + (response.eval_count or 0),
        }
        return {"content": content, "model": self.model_name, "usage": usage}

    def stream(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
        store_as      : str         = None,   # ← matches GroqLLM
    ):
        """
        Yields str tokens then final dict {"model": ..., "usage": {...}}
        """
        messages = self._build_messages(prompt, system_prompt, history, store_as=store_as)

        response_stream = ollama.chat(
            model    = self.model_name,
            messages = messages,
            options  = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens  or self.max_tokens,
            },
            stream   = True,
        )

        full_reply = []
        usage_data = {}

        for chunk in response_stream:
            text = chunk.message.content
            if text:
                full_reply.append(text)
                yield text
            if chunk.done:
                usage_data = {
                    "prompt_tokens"    : chunk.prompt_eval_count or 0,
                    "completion_tokens": chunk.eval_count        or 0,
                    "total_tokens"     : (chunk.prompt_eval_count or 0) + (chunk.eval_count or 0),
                }

        active_history = history or self.history
        active_history.add_assistant("".join(full_reply))
        yield {"model": self.model_name, "usage": usage_data}

    def chat(self, message: str, **kwargs) -> str:
        return self.generate(message, **kwargs)["content"]

    def reset_history(self) -> None:
        self.history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.history.set_system(prompt)

    def get_info(self) -> dict:
        return {
            "model"      : self.model_name,
            "provider"   : "ollama-local",
            "temperature": self.temperature,
            "max_tokens" : self.max_tokens,
            "history_len": len(self.history),
        }


# Register into shared factory
LLMFactory.PROVIDERS["ollama"] = OllamaLLM

__all__ = ["OllamaLLM"]