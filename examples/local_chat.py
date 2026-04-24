"""
Interactive REPL against a local `transformers` model via toki.

Extra deps (beyond `toki[local]`):
- easyrepl
"""

import json
import logging
import warnings
from collections.abc import Callable
from datetime import UTC, datetime

from easyrepl import REPL

from toki import Agent, pretty_tool_call
from toki.local import Model
from toki.model import TokiToolCall, TokiToolResponse


LOG_LEVEL = logging.INFO
logger = logging.getLogger(__name__)
logger.propagate = False


def get_current_time() -> dict[str, str]:
    return {"utc_time": datetime.now(UTC).isoformat()}


def add_numbers(a: float, b: float) -> dict[str, float]:
    return {"sum": a + b}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current UTC time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                },
                "required": ["a", "b"],
            },
        },
    },
]


TOOLS_BY_NAME: dict[str, Callable[..., object]] = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}


def execute_tool_call(tool_call: TokiToolCall) -> str:
    function_name = tool_call["function"]["name"]
    function = TOOLS_BY_NAME[function_name]
    arguments = json.loads(tool_call["function"]["arguments"])
    result = function(**arguments)
    result_json = json.dumps(result)
    logger.info("Tool result for %s: %s", pretty_tool_call(tool_call), result_json)
    return result_json


def run_agent_turn(agent: Agent) -> str:
    while True:
        result_chunks: list[str] = []
        tool_response: TokiToolResponse | None = None

        if agent.tools is None:
            for chunk in agent.execute(stream=True):
                print(chunk, end="", flush=True)
                result_chunks.append(chunk)
            print()
            return "".join(result_chunks)

        for chunk in agent.model.complete(agent.messages, stream=True, tools=agent.tools):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                result_chunks.append(chunk)
            else:
                tool_response = chunk

        if result_chunks:
            print()

        assistant_text = "".join(result_chunks)
        if tool_response is None:
            agent.add_assistant_message(assistant_text)
            return assistant_text

        agent.add_assistant_tool_calls(tool_response["thought"], tool_response["tool_calls"])

        for tool_call in tool_response["tool_calls"]:
            logger.info("Tool call: %s", pretty_tool_call(tool_call))
            tool_result = execute_tool_call(tool_call)
            agent.add_tool_message(tool_call["id"], tool_result)


def main() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

    for noisy_logger in [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="Warning: You are sending unauthenticated requests to the HF Hub.*",
    )

    model = Model("Qwen/Qwen3-0.6B")
    agent = Agent(model, tools=TOOL_SCHEMAS)
    agent.add_system_message("You are a concise, helpful assistant. Use tools when they are useful.")

    for query in REPL(history=".chat"):
        agent.add_user_message(query)
        run_agent_turn(agent)


if __name__ == "__main__":
    main()
