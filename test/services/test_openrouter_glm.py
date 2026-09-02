import types
import unittest
from unittest.mock import patch

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.services import llm


class TestOpenRouterGlmImprovements(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_default_provider_stays_moonshot(self):
        self.assertEqual(DEFAULT_LLM_PROVIDER_ID, "moonshot")

    def test_openrouter_default_model_is_glm_flash(self):
        openrouter = get_llm_provider("openrouter")
        self.assertEqual(openrouter.default_model, "z-ai/glm-5.3-flash")
        self.assertEqual(
            openrouter.resolve_model_name("minimax/minimax-m3:free"),
            "z-ai/glm-5.3-flash",
        )

    def test_openrouter_sends_attribution_headers(self):
        config.app["llm_provider"] = "openrouter"
        config.app["openrouter_api_key"] = "openrouter-key"
        config.app["openrouter_base_url"] = ""
        config.app["openrouter_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nopenrouter")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/ubcool2ic/MoneyPrinterTurbo",
                "X-Title": "MoneyPrinterTurbo",
            },
        )
        self.assertEqual(fake_completions.kwargs["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(result, "hello\nopenrouter")

    def test_openai_compatible_retries_transient_429(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = "openai-key"
        config.app["openai_base_url"] = ""
        config.app["openai_model_name"] = ""

        rate_limited = Exception("rate limited")
        rate_limited.status_code = 429

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise rate_limited
                message = types.SimpleNamespace(content="hello\nretry")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
            patch.object(llm, "sleep") as mock_sleep,
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="openai-key",
            base_url="https://api.openai.com/v1",
        )
        self.assertEqual(fake_completions.calls, 2)
        mock_sleep.assert_called_once_with(1)
        self.assertEqual(result, "hello\nretry")


if __name__ == "__main__":
    unittest.main()
