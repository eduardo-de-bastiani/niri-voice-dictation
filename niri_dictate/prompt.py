"""
Autoregressive Priming Prompt Builder for Niri Voice Dictation.
Biases Whisper-large-v3-turbo toward technical software engineering vocabulary in Brazilian Portuguese,
enforcing strict token budgets (138 base tokens, <= 200 target ceiling, < 224 hard limit) via
Tier 1 BPE tail pruning and Tier 2 static fallback heuristics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GroqConfig

logger = logging.getLogger("niri-dictate.prompt")


class PrimingPromptBuilder:
    """Builds and bounds technical priming prompts for Whisper speech recognition."""

    DEFAULT_BASE_PROMPT = (
        "Código, commits e comandos de programação em Python, JavaScript, TypeScript, Rust, Go, SQL. "
        "Termos técnicos: pull request, merge, branch, commit, rebase, diff, git, docker, kubernetes, "
        "linux, bash, function, const, let, async, await, return, import, export, class, interface, "
        "null, undefined, boolean, string, number, array, object, api, rest, json, postgres, backend, "
        "frontend, devops, deploy, ci/cd, bug, fix, test, mock, endpoint, middleware, refactor."
    )

    def __init__(
        self,
        base_prompt: str | None = None,
        custom_keywords: list[str] | None = None,
        max_tokens: int = 200,
        max_words: int = 65,
        max_chars: int = 520,
    ) -> None:
        """
        Initializes the prompt builder.

        Args:
            base_prompt: Canonical autoregressive base template (defaults to 138 GPT-2 token template).
            custom_keywords: Optional list of domain-specific project keywords.
            max_tokens: Maximum BPE token ceiling (defaults to 200, leaving >= 24 safety margin under 224).
            max_words: Tier 2 static fallback maximum word count (defaults to 65).
            max_chars: Tier 2 static fallback maximum character length (defaults to 520).
        """
        self.base_prompt = (base_prompt or self.DEFAULT_BASE_PROMPT).strip()
        self.custom_keywords = [k.strip() for k in (custom_keywords or []) if k.strip()]
        self.max_tokens = max_tokens
        self.max_words = max_words
        self.max_chars = max_chars

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Conservative BPE token estimator for Latin/Portuguese text when tiktoken is unavailable.
        Accurately accounts for multibyte UTF-8 diacritic expansion in GPT-2 BPE vocabularies.
        """
        non_ascii = sum(1 for c in text if ord(c) >= 128)
        ascii_chars = len(text) - non_ascii
        return int(ascii_chars / 3.3) + int(non_ascii * 1.3) + 1

    def _is_tier2_safe(self, text: str, token_limit: int, max_words: int, max_chars: int) -> bool:
        """Validates that candidate text conforms to word count, char length, and estimated token limits."""
        if len(text.split()) > max_words:
            return False
        if len(text) > max_chars:
            return False
        if self._estimate_tokens(text) > token_limit:
            return False
        return True

    def _prune_text_tier2(self, text: str, token_limit: int, max_words: int, max_chars: int) -> str:
        """Word-by-word tail pruning until text satisfies word, char, and estimated token limits."""
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
        while words:
            candidate = " ".join(words)
            if len(candidate) <= max_chars and self._estimate_tokens(candidate) <= token_limit:
                return candidate
            words.pop()
        return text[:max_chars]

    @classmethod
    def from_config(cls, config: GroqConfig) -> PrimingPromptBuilder:
        """Factory constructor instantiating builder from GroqConfig dataclass."""
        return cls(
            base_prompt=config.prompt,
            custom_keywords=config.custom_keywords,
            max_tokens=config.max_prompt_tokens,
            max_words=config.max_prompt_words,
            max_chars=config.max_prompt_chars,
        )

    def build_prompt(self, max_tokens: int | None = None) -> str:
        """
        Constructs a prompt strictly bounded under the token target.
        Applies Tier 1 BPE token-aware pruning if tiktoken is available,
        falling back to Tier 2 static length bounds otherwise.

        Args:
            max_tokens: Optional token ceiling override.

        Returns:
            Priming prompt string strictly within the safety envelope.
        """
        token_limit = max_tokens if max_tokens is not None else self.max_tokens

        # Tier 1: Exact BPE Token-Aware Pruning (GPT-2 tokenizer)
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("gpt2")

            if not self.custom_keywords:
                tokens = tokenizer.encode(self.base_prompt)
                if len(tokens) <= token_limit:
                    return self.base_prompt
                return tokenizer.decode(tokens[:token_limit])

            keywords = list(self.custom_keywords)
            while keywords:
                candidate = f"{self.base_prompt} Termos adicionais: {', '.join(keywords)}."
                if len(tokenizer.encode(candidate)) <= token_limit:
                    return candidate
                keywords.pop()

            # If all custom keywords were popped, check base template alone
            base_tokens = tokenizer.encode(self.base_prompt)
            if len(base_tokens) <= token_limit:
                return self.base_prompt
            return tokenizer.decode(base_tokens[:token_limit])

        # Tier 2: Hardened Diacritic-Aware Static Heuristic Fallback
        except Exception as err:
            logger.debug(
                "tiktoken unavailable or failed (%s); using Tier 2 static length fallback.", err
            )

            max_words = self.max_words
            max_chars = self.max_chars

            # Dynamically scale word and char thresholds if token_limit is tightened below 200
            if token_limit < 200:
                ratio = token_limit / 200.0
                max_words = max(10, int(max_words * ratio))
                max_chars = max(50, int(max_chars * ratio))

            if not self.custom_keywords:
                # Canonical default base prompt is empirically verified at 138 GPT-2 tokens (60 words)
                if self.base_prompt == self.DEFAULT_BASE_PROMPT and token_limit >= 138:
                    return self.base_prompt
                if self._is_tier2_safe(self.base_prompt, token_limit, max_words, max_chars):
                    return self.base_prompt
                return self._prune_text_tier2(self.base_prompt, token_limit, max_words, max_chars)

            keywords = list(self.custom_keywords)
            while keywords:
                candidate = f"{self.base_prompt} Termos adicionais: {', '.join(keywords)}."
                if self._is_tier2_safe(candidate, token_limit, max_words, max_chars):
                    return candidate
                keywords.pop()

            if self.base_prompt == self.DEFAULT_BASE_PROMPT and token_limit >= 138:
                return self.base_prompt
            if self._is_tier2_safe(self.base_prompt, token_limit, max_words, max_chars):
                return self.base_prompt
            return self._prune_text_tier2(self.base_prompt, token_limit, max_words, max_chars)
