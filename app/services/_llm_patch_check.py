import json
import logging
import re
from time import perf_counter, sleep
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\\s?#@]*:[^/\\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, \"welcome to this video\".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include \"voiceover\", \"narrator\" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()


def _normalize_text_response(content, llm_provider: str) -> str:
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content


def _sanitize_error_message(error: object) -> str:
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _openai_compatible_client_kwargs(
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


def _is_transient_openai_http_error(error: BaseException) -> bool:
    """Retry OpenAI-compatible calls on HTTP 429 and 5xx only."""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    return isinstance(status_code, int) and status_code >= 500
