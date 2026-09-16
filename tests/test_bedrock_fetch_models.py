import json
from io import BytesIO

from toki.bedrock import fetch_models


def _response(metadata: dict) -> BytesIO:
    return BytesIO(json.dumps(metadata).encode())


def test_fetch_models_uses_public_converse_metadata(monkeypatch):
    metadata = {
        "amazon.nova-micro-v1:0": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "max_input_tokens": 128_000,
            "supports_function_calling": True,
            "supports_prompt_caching": True,
        },
        "global.anthropic.claude-sonnet-4-6": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "supports_function_calling": True,
            "supports_reasoning": True,
            "supports_prompt_caching": True,
            "supports_adaptive_thinking": True,
            "cache_creation_input_token_cost_above_1hr": 0.000001,
            "prompt_cache_min_tokens": 1_024,
        },
        "bedrock/converse/us.example.chat-v1:0": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "max_input_tokens": 64_000,
        },
        "legacy.direct-v1": {
            "litellm_provider": "bedrock",
            "mode": "chat",
        },
        "amazon.embedding-v1": {
            "litellm_provider": "bedrock_converse",
            "mode": "embedding",
        },
        "anthropic.alias@20260101": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
        },
    }
    monkeypatch.setattr(
        fetch_models,
        "urlopen",
        lambda url: _response(metadata),
    )

    models = fetch_models._fetch_models()

    assert list(models) == [
        "amazon.nova-micro-v1:0",
        "global.anthropic.claude-sonnet-4-6",
        "us.example.chat-v1:0",
    ]
    nova = models["amazon.nova-micro-v1:0"]
    assert nova.context_size == 128_000
    assert nova.supports_explicit_caching
    assert nova.cache_min_tokens == 1_000
    assert nova.cache_locations == ("system", "messages")

    claude = models["global.anthropic.claude-sonnet-4-6"]
    assert claude.reasoning_family == "claude_adaptive"
    assert claude.cache_ttls == ("5m", "1h")


def test_create_models_file_needs_no_aws_credentials(monkeypatch, tmp_path):
    metadata = {
        "bedrock/converse/us.example.chat-v1:0": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "max_input_tokens": 64_000,
            "supports_function_calling": True,
        }
    }
    monkeypatch.setattr(
        fetch_models,
        "urlopen",
        lambda url: _response(metadata),
    )
    output = tmp_path / "models.py"

    fetch_models._create_models_types_file(output)

    generated = output.read_text()
    assert '"us.example.chat-v1:0"' in generated
    assert "discover_bedrock_models" not in generated
    compile(generated, output, "exec")
