import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def discovery(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class Client:
        def list_foundation_models(self):
            calls.append(("foundation", {}))
            return {
                "modelSummaries": [
                    {
                        "modelId": "amazon.nova-micro-v1:0",
                        "modelArn": (
                            "arn:aws:bedrock:us-east-1::foundation-model/"
                            "amazon.nova-micro-v1:0"
                        ),
                        "modelName": "Nova Micro",
                        "providerName": "Amazon",
                        "inputModalities": ["TEXT"],
                        "outputModalities": ["TEXT"],
                        "responseStreamingSupported": True,
                        "inferenceTypesSupported": ["ON_DEMAND"],
                        "modelLifecycle": {"status": "ACTIVE"},
                    }
                ]
            }

        def list_inference_profiles(self, **kwargs):
            calls.append(("profiles", kwargs))
            if "nextToken" not in kwargs:
                return {
                    "inferenceProfileSummaries": [
                        {
                            "inferenceProfileId": "us.amazon.nova-micro-v1:0",
                            "inferenceProfileArn": "arn:profile/nova",
                            "inferenceProfileName": "US Nova",
                            "models": [
                                {
                                    "modelArn": (
                                        "arn:aws:bedrock:us-east-1::foundation-model/"
                                        "amazon.nova-micro-v1:0"
                                    )
                                }
                            ],
                            "status": "ACTIVE",
                        }
                    ],
                    "nextToken": "next",
                }
            return {"inferenceProfileSummaries": []}

    client = Client()

    def Session(**kwargs):
        return SimpleNamespace(client=lambda service, **client_kwargs: client)

    sys.modules.pop("toki.bedrock.discovery", None)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=Session))
    module = importlib.import_module("toki.bedrock.discovery")
    yield module, calls
    sys.modules.pop("toki.bedrock.discovery", None)


def test_discovery_resolves_profiles_and_caches(discovery):
    module, calls = discovery

    first = module.discover_bedrock_models(region_name="us-east-1")
    second = module.discover_bedrock_models(region_name="us-east-1")

    assert first == second
    assert len(first) == 2
    profile = first[1]
    assert profile.model_id == "us.amazon.nova-micro-v1:0"
    assert profile.provider == "Amazon"
    assert profile.supports_streaming is True
    assert profile.attributes.supports_explicit_caching is True
    assert [name for name, _ in calls].count("foundation") == 1
    assert [kwargs for name, kwargs in calls if name == "profiles"] == [
        {"maxResults": 1000},
        {"maxResults": 1000, "nextToken": "next"},
    ]


async def test_async_discovery_refreshes(discovery):
    module, calls = discovery

    await module.adiscover_bedrock_models(
        region_name="us-east-1",
        refresh=True,
    )

    assert [name for name, _ in calls].count("foundation") == 1
