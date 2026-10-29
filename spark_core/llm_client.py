"""LLM client for Spark Engine creative reasoning. Gated by SPARK_LLM env var."""
import os


def _is_enabled() -> bool:
    return os.environ.get("SPARK_LLM", "").lower() in ("1", "true", "enabled", "yes", "on")


def _get_api_key() -> str:
    """Get API key from env, checking both Anthropic and third-party formats."""
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")


def _get_base_url() -> str:
    """Get base URL, supporting third-party proxies like DeepSeek."""
    return os.environ.get("ANTHROPIC_BASE_URL", "")


def _get_model() -> str:
    """Get the model name, defaulting to sonnet if not set."""
    return os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6-20250514")


def llm_reason(system_prompt: str, user_prompt: str, timeout: int = 30,
               max_tokens: int = 3000, temperature: float = 0.7) -> str:
    """Call Claude-compatible API for creative reasoning. Returns empty string on failure."""
    if not _is_enabled():
        return ""

    api_key = _get_api_key()
    if not api_key:
        return ""

    try:
        from anthropic import Anthropic
        kwargs = {"api_key": api_key}
        base_url = _get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        client = Anthropic(**kwargs)
        response = client.messages.create(
            model=_get_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract text: prefer TextBlock (actual response) over ThinkingBlock (reasoning).
        # DeepSeek returns ThinkingBlock first, then TextBlock — so we must scan ALL blocks
        # and prefer the TextBlock. Only fall back to ThinkingBlock if no TextBlock exists.
        thinking_text = ""
        for block in response.content:
            if hasattr(block, 'text') and block.text:
                return block.text.strip()
            if hasattr(block, 'thinking') and block.thinking:
                thinking_text = block.thinking.strip()
        if thinking_text:
            return thinking_text
        return ""
    except Exception:
        return ""


def llm_is_available() -> bool:
    """Check if LLM is configured and enabled."""
    return _is_enabled() and bool(_get_api_key())
