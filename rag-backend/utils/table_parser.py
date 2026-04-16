# utils/table_parser.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd


# ─────────────────────────────────────────
# PARSED TABLE
# ─────────────────────────────────────────

class ParsedTable:
    """
    Wraps a parsed table and provides multiple output formats.
    All formats are derived from a single pandas DataFrame.

    Formats available:
        .df             → pandas DataFrame
        .to_markdown()  → Markdown table string (for LLM context)
        .to_json_rows() → list of dicts (one per row)
        .get_summary()  → shape, columns, numeric stats
        .to_chunk()     → RAG-ready chunk dict
    """

    def __init__(self, df: pd.DataFrame, source: str = "unknown", page: int = None):
        self.df     = df
        self.source = source
        self.page   = page

    # ── FORMATS ──────────────────────────────

    def to_markdown(self, max_rows: int = 50) -> str:
        """
        Convert table to Markdown string — best format for LLM context.

        Args:
            max_rows : cap rows to avoid flooding the LLM context window

        Returns:
            Markdown table string e.g.:
            | Name  | Revenue |
            |-------|---------|
            | AAPL  | 1.2M    |
        """
        df = self.df.head(max_rows)
        return df.to_markdown(index=False)

    def to_json_rows(self) -> list[dict]:
        """
        Return table as a list of row dicts.
        Good for structured processing or storage.

        Example:
            [{"Name": "AAPL", "Revenue": "1.2M"}, ...]
        """
        return self.df.to_dict(orient="records")

    def to_json_string(self) -> str:
        """JSON-serialized rows as a string."""
        return json.dumps(self.to_json_rows(), indent=2, default=str)

    def get_summary(self) -> dict:
        """
        Return summary statistics about the table.

        Returns:
            {
                "rows"        : int,
                "columns"     : int,
                "column_names": list[str],
                "numeric_cols": dict,     ← {col: {min, max, mean}} for numeric cols
                "null_counts" : dict,     ← {col: null_count}
            }
        """
        numeric_summary = {}
        for col in self.df.select_dtypes(include="number").columns:
            numeric_summary[col] = {
                "min" : round(float(self.df[col].min()), 4),
                "max" : round(float(self.df[col].max()), 4),
                "mean": round(float(self.df[col].mean()), 4),
            }

        return {
            "rows"        : len(self.df),
            "columns"     : len(self.df.columns),
            "column_names": list(self.df.columns),
            "numeric_cols": numeric_summary,
            "null_counts" : self.df.isnull().sum().to_dict(),
        }

    def to_chunk(self, format: str = "markdown", max_rows: int = 50) -> dict:
        """
        Convert table to a RAG-ready chunk dict for ingestion.

        Args:
            format   : 'markdown' or 'json' — content format for embedding
            max_rows : max rows to include in the chunk

        Returns:
            {
                "content": str,      ← table as markdown or JSON
                "source" : str,
                "page"   : int|None,
                "type"   : "table",
                "rows"   : int,
                "columns": int,
            }
        """
        if format == "json":
            content = self.to_json_string()
        else:
            content = self.to_markdown(max_rows=max_rows)

        return {
            "content": content,
            "source" : self.source,
            "page"   : self.page,
            "type"   : "table",
            "rows"   : len(self.df),
            "columns": len(self.df.columns),
        }

    def __repr__(self) -> str:
        return f"ParsedTable({len(self.df)} rows × {len(self.df.columns)} cols | source={self.source})"


# ─────────────────────────────────────────
# TABLE PARSER
# ─────────────────────────────────────────

class TableParser:
    """
    Parses tables from CSV, Excel (.xlsx), and raw HTML strings.
    Returns ParsedTable objects with multiple output formats.

    Supported inputs:
        - .csv  files  → parse_csv()
        - .xlsx files  → parse_excel()  (supports multiple sheets)
        - HTML string  → parse_html()   (e.g. scraped web tables)
        - Any file     → parse_file()   (auto-detects type)

    All methods return ParsedTable or list[ParsedTable].
    """

    def __init__(self):
        print(f"  [TABLE PARSER] Ready.")

    # ── CSV ──────────────────────────────────

    def parse_csv(self, file_path: str, **kwargs) -> "ParsedTable":
        """
        Parse a CSV file into a ParsedTable.

        Args:
            file_path : path to .csv file
            **kwargs  : passed to pd.read_csv (encoding, sep, etc.)

        Returns:
            ParsedTable
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV not found: {file_path}")

        df = pd.read_csv(file_path, **kwargs)
        df = self._clean(df)
        print(f"  [TABLE PARSER] CSV parsed: {len(df)} rows × {len(df.columns)} cols")
        return ParsedTable(df, source=os.path.basename(file_path))

    # ── EXCEL ────────────────────────────────

    def parse_excel(
        self,
        file_path  : str,
        sheet_name : str | int | None = None,
        **kwargs
    ) -> list["ParsedTable"]:
        """
        Parse an Excel file — handles single or multiple sheets.

        Args:
            file_path  : path to .xlsx file
            sheet_name : specific sheet name/index, or None for all sheets
            **kwargs   : passed to pd.read_excel

        Returns:
            list[ParsedTable] — one per sheet
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        base = os.path.basename(file_path)

        if sheet_name is not None:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            df = self._clean(df)
            print(f"  [TABLE PARSER] Excel sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} cols")
            return [ParsedTable(df, source=f"{base}::{sheet_name}")]

        # Load all sheets
        sheets = pd.read_excel(file_path, sheet_name=None, **kwargs)
        tables = []
        for name, df in sheets.items():
            df = self._clean(df)
            if df.empty:
                continue
            print(f"  [TABLE PARSER] Sheet '{name}': {len(df)} rows × {len(df.columns)} cols")
            tables.append(ParsedTable(df, source=f"{base}::{name}"))

        print(f"  [TABLE PARSER] Excel parsed: {len(tables)} non-empty sheets")
        return tables

    # ── HTML ─────────────────────────────────

    def parse_html(self, html: str, source: str = "html") -> list["ParsedTable"]:
        """
        Extract all tables from an HTML string.

        Args:
            html   : raw HTML string containing <table> elements
            source : label for the source field

        Returns:
            list[ParsedTable] — one per <table> found
        """
        try:
            dfs    = pd.read_html(html)
            tables = []
            for i, df in enumerate(dfs):
                df = self._clean(df)
                if df.empty:
                    continue
                tables.append(ParsedTable(df, source=f"{source}::table_{i}"))
            print(f"  [TABLE PARSER] HTML parsed: {len(tables)} tables found")
            return tables
        except Exception as e:
            print(f"  [TABLE PARSER] ⚠️  HTML parse error: {e}")
            return []

    # ── AUTO-DETECT ──────────────────────────

    def parse_file(self, file_path: str, **kwargs) -> list["ParsedTable"]:
        """
        Auto-detect file type and parse accordingly.
        Always returns a list for consistency.

        Supports: .csv, .xlsx, .xls
        """
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".csv":
            return [self.parse_csv(file_path, **kwargs)]
        elif ext in (".xlsx", ".xls"):
            return self.parse_excel(file_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: .csv, .xlsx, .xls"
            )

    def to_chunks(
        self,
        file_path : str,
        format    : str = "markdown",
        max_rows  : int = 50,
        **kwargs
    ) -> list[dict]:
        """
        Parse a file and return all tables as RAG-ready chunk dicts.
        One-liner for the ingestion pipeline.

        Args:
            file_path : path to .csv or .xlsx
            format    : 'markdown' or 'json'
            max_rows  : max rows per chunk

        Returns:
            list[dict] — ready to pass to vector_store.add_documents()
        """
        tables = self.parse_file(file_path, **kwargs)
        chunks = [t.to_chunk(format=format, max_rows=max_rows) for t in tables]
        print(f"  [TABLE PARSER] {len(chunks)} table chunk(s) ready for ingestion")
        return chunks

    # ── HELPERS ──────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning applied to all parsed tables:
        - Drop fully empty rows and columns
        - Strip whitespace from string columns
        - Reset index
        """
        df = df.dropna(how="all").dropna(axis=1, how="all")
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
        df = df.reset_index(drop=True)
        return df

    def get_info(self) -> dict:
        return {
            "type"     : "TableParser",
            "supported": [".csv", ".xlsx", ".xls", "html_string"],
            "formats"  : ["markdown", "json"],
        }


__all__ = ["ParsedTable", "TableParser"]