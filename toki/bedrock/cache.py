from .models import CacheLocation, CacheTTL


def cache_point(ttl: CacheTTL) -> dict:
    point: dict = {"type": "default"}
    if ttl == "1h":
        point["ttl"] = ttl
    return {"cachePoint": point}


def apply_cache_point(
    *,
    system: list[dict] | None,
    messages: list[dict],
    tool_config: dict | None,
    boundary_index: int,
    ttl: CacheTTL,
    locations: tuple[CacheLocation, ...],
) -> tuple[list[dict] | None, list[dict], dict | None]:
    """Place one Bedrock cache checkpoint at the latest supported boundary."""
    marker = cache_point(ttl)
    new_system = list(system) if system is not None else None
    new_messages = list(messages)
    new_tool_config = dict(tool_config) if tool_config is not None else None

    if "messages" in locations and boundary_index >= 0:
        message = dict(new_messages[boundary_index])
        message["content"] = [*message["content"], marker]
        new_messages[boundary_index] = message
        return new_system, new_messages, new_tool_config

    if "system" in locations and new_system:
        new_system.append(marker)
        return new_system, new_messages, new_tool_config

    if "tools" in locations and new_tool_config is not None:
        new_tool_config["tools"] = [*new_tool_config["tools"], marker]
    return new_system, new_messages, new_tool_config
