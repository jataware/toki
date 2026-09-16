import asyncio
import os
from dataclasses import dataclass
from typing import Literal

import boto3
from botocore.config import Config

from .models import Attr, get_bedrock_model_attributes


@dataclass(frozen=True)
class BedrockModelInfo:
    model_id: str
    model_arn: str
    name: str
    provider: str
    kind: Literal["foundation_model", "inference_profile"]
    status: str | None
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supports_streaming: bool | None
    inference_types: tuple[str, ...]
    source_model_arns: tuple[str, ...] = ()
    attributes: Attr | None = None


_discovery_cache: dict[tuple[str | None, str | None], tuple[BedrockModelInfo, ...]] = {}


def _client(profile_name: str | None, region_name: str | None):
    session = boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )
    kwargs: dict = {}
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK") is not None:
        kwargs["config"] = Config(auth_scheme_preference="httpBearerAuth")
    return session.client("bedrock", **kwargs)


def _foundation_info(summary: dict) -> BedrockModelInfo:
    model_id = summary["modelId"]
    lifecycle = summary.get("modelLifecycle") or {}
    return BedrockModelInfo(
        model_id=model_id,
        model_arn=summary["modelArn"],
        name=summary.get("modelName", model_id),
        provider=summary.get("providerName", ""),
        kind="foundation_model",
        status=lifecycle.get("status"),
        input_modalities=tuple(summary.get("inputModalities", ())),
        output_modalities=tuple(summary.get("outputModalities", ())),
        supports_streaming=summary.get("responseStreamingSupported"),
        inference_types=tuple(summary.get("inferenceTypesSupported", ())),
        attributes=get_bedrock_model_attributes(model_id),
    )


def _profile_info(
    summary: dict,
    foundation_by_arn: dict[str, BedrockModelInfo],
) -> BedrockModelInfo:
    source_arns = tuple(model["modelArn"] for model in summary.get("models", ()))
    source = next(
        (foundation_by_arn[arn] for arn in source_arns if arn in foundation_by_arn),
        None,
    )
    if source is None:
        source_ids = {arn.rsplit("/", 1)[-1] for arn in source_arns}
        source = next(
            (
                model
                for model in foundation_by_arn.values()
                if model.model_id in source_ids
            ),
            None,
        )
    model_id = summary["inferenceProfileId"]
    return BedrockModelInfo(
        model_id=model_id,
        model_arn=summary["inferenceProfileArn"],
        name=summary.get("inferenceProfileName", model_id),
        provider=source.provider if source is not None else "",
        kind="inference_profile",
        status=summary.get("status"),
        input_modalities=source.input_modalities if source is not None else (),
        output_modalities=source.output_modalities if source is not None else (),
        supports_streaming=source.supports_streaming if source is not None else None,
        inference_types=("INFERENCE_PROFILE",),
        source_model_arns=source_arns,
        attributes=(
            get_bedrock_model_attributes(model_id)
            or (source.attributes if source is not None else None)
        ),
    )


def _discover(
    profile_name: str | None,
    region_name: str | None,
) -> tuple[BedrockModelInfo, ...]:
    client = _client(profile_name, region_name)
    foundation = [
        _foundation_info(summary)
        for summary in client.list_foundation_models()["modelSummaries"]
    ]
    foundation_by_arn = {model.model_arn: model for model in foundation}

    profiles: list[BedrockModelInfo] = []
    kwargs: dict = {"maxResults": 1000}
    while True:
        response = client.list_inference_profiles(**kwargs)
        profiles.extend(
            _profile_info(summary, foundation_by_arn)
            for summary in response.get("inferenceProfileSummaries", ())
        )
        next_token = response.get("nextToken")
        if next_token is None:
            break
        kwargs["nextToken"] = next_token
    return tuple(foundation + profiles)


def discover_bedrock_models(
    *,
    profile_name: str | None = None,
    region_name: str | None = None,
    refresh: bool = False,
) -> list[BedrockModelInfo]:
    """Discover foundation models and inference profiles for an AWS region."""
    key = (profile_name, region_name)
    if refresh or key not in _discovery_cache:
        _discovery_cache[key] = _discover(profile_name, region_name)
    return list(_discovery_cache[key])


async def adiscover_bedrock_models(
    *,
    profile_name: str | None = None,
    region_name: str | None = None,
    refresh: bool = False,
) -> list[BedrockModelInfo]:
    """Async sibling of `discover_bedrock_models`."""
    return await asyncio.to_thread(
        discover_bedrock_models,
        profile_name=profile_name,
        region_name=region_name,
        refresh=refresh,
    )
