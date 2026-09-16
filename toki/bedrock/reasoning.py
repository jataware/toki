from dataclasses import dataclass
from typing import Literal

from ..model import ReasoningEffort
from .models import ReasoningFamily


@dataclass(frozen=True)
class ClaudeAdaptiveReasoning:
    effort: ReasoningEffort = "medium"


@dataclass(frozen=True)
class ClaudeBudgetReasoning:
    budget_tokens: int


@dataclass(frozen=True)
class NovaReasoning:
    effort: Literal["low", "medium", "high"] = "medium"


@dataclass(frozen=True)
class OpenAIReasoning:
    effort: ReasoningEffort = "medium"


BedrockReasoningConfig = (
    ClaudeAdaptiveReasoning | ClaudeBudgetReasoning | NovaReasoning | OpenAIReasoning
)


_CLAUDE_BUDGETS: dict[ReasoningEffort, int] = {
    "minimal": 1_024,
    "low": 2_048,
    "medium": 4_096,
    "high": 8_192,
    "xhigh": 16_384,
}


def default_reasoning_config(
    family: ReasoningFamily,
    effort: ReasoningEffort,
) -> BedrockReasoningConfig:
    if family == "claude_adaptive":
        return ClaudeAdaptiveReasoning(effort=effort)
    if family == "claude_budget":
        return ClaudeBudgetReasoning(budget_tokens=_CLAUDE_BUDGETS[effort])
    if family == "nova":
        nova_effort: Literal["low", "medium", "high"]
        if effort in {"minimal", "low"}:
            nova_effort = "low"
        elif effort == "xhigh":
            nova_effort = "high"
        else:
            nova_effort = effort
        return NovaReasoning(effort=nova_effort)
    return OpenAIReasoning(effort=effort)


def reasoning_family(
    config: BedrockReasoningConfig,
) -> ReasoningFamily:
    if isinstance(config, ClaudeAdaptiveReasoning):
        return "claude_adaptive"
    if isinstance(config, ClaudeBudgetReasoning):
        return "claude_budget"
    if isinstance(config, NovaReasoning):
        return "nova"
    return "openai"


def reasoning_request_fields(
    config: BedrockReasoningConfig,
) -> dict:
    if isinstance(config, ClaudeAdaptiveReasoning):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": config.effort},
        }
    if isinstance(config, ClaudeBudgetReasoning):
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": config.budget_tokens,
            }
        }
    if isinstance(config, NovaReasoning):
        return {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": config.effort,
            }
        }
    return {"reasoning": {"effort": config.effort}}
