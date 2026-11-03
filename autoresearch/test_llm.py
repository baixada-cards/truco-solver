"""Exercise the installed SDK request builders without network access."""

import json

import anthropic
import httpx2
import openai
import pytest
from llm import LLMConfig, call_llm


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_call_llm_with_installed_sdk(provider, monkeypatch):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(request)
        assert payload["model"] == "offline-model"
        assert payload["messages"] == [{"role": "user", "content": "offline prompt"}]
        assert payload["max_tokens"] == 8192
        if provider == "anthropic":
            assert request.url.path == "/v1/messages"
            assert "temperature" not in payload
            body = {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "offline-model",
                "content": [{"type": "text", "text": "offline response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            assert payload["temperature"] == 0.7
            body = {
                "id": "chatcmpl_offline",
                "object": "chat.completion",
                "created": 1,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "offline response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        return httpx2.Response(200, json=body)

    sdk = anthropic if provider == "anthropic" else openai
    constructor = "Anthropic" if provider == "anthropic" else "OpenAI"
    with getattr(sdk, constructor)(
        api_key="offline-placeholder",
        base_url=(
            "https://offline.invalid"
            if provider == "anthropic"
            else "https://offline.invalid/v1"
        ),
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        max_retries=0,
    ) as client:
        monkeypatch.setattr(sdk, constructor, lambda **kwargs: client)
        config = LLMConfig(provider, "offline-placeholder", "offline-model")
        assert call_llm(config, "offline prompt") == "offline response"
    assert len(requests) == 1
