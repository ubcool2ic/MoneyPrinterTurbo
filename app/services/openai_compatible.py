"""OpenAI-compatible client helpers used by app.services.llm."""

from time import sleep

from loguru import logger
from openai import OpenAI


def openai_compatible_client_kwargs(
    llm_provider: str, api_key: str, base_url: str
) -> dict:
    """Build OpenAI client kwargs; OpenRouter needs app attribution headers."""
    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
    }
    if llm_provider == "openrouter":
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/ubcool2ic/MoneyPrinterTurbo",
            "X-Title": "MoneyPrinterTurbo",
        }
    return kwargs


def is_transient_openai_http_error(error: BaseException) -> bool:
    """Retry OpenAI-compatible calls on HTTP 429 and 5xx only."""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    return isinstance(status_code, int) and status_code >= 500


def create_openai_compatible_completion(
    llm_provider: str,
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    openai_client=None,
    sleeper=None,
):
    """Run chat.completions.create with OpenRouter headers and 429/5xx retries.

    openai_client and sleeper are injectable so llm.py tests can patch llm.OpenAI
    and llm.sleep without importing this module in tests.
    """
    client_factory = openai_client if openai_client is not None else OpenAI
    sleep_fn = sleeper if sleeper is not None else sleep
    client = client_factory(
        **openai_compatible_client_kwargs(llm_provider, api_key, base_url)
    )
    response = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            break
        except Exception as e:
            if not is_transient_openai_http_error(e) or attempt == 2:
                raise
            logger.warning(
                f"[{llm_provider}] transient HTTP error "
                f"{getattr(e, 'status_code', None)}, retrying ({attempt + 1}/3)"
            )
            sleep_fn(1 if attempt == 0 else 2)
    return response
