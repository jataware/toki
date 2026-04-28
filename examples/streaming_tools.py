"""
Demo of streaming tool calls.

Streaming-flagged tools (declared with `StreamingToolSchema(...)`) surface as
`TokiToolCallStream` objects when consumed in streaming mode. Each argument
value can be consumed character-by-character via `expect_arg(name)` (returns a
`TokiArgStream`) or by iterating `items()` for `(name, TokiArgStream)` pairs.

In this demo the model is asked to propose a small file-edit patch via a
streaming `propose_patch(target, replacement)` tool. As the model generates the
arguments, we render a side-by-side preview that updates live as `replacement`
arrives.
"""

import os

from toki import (
    Agent,
    OpenRouterModel,
    StreamingToolSchema,
    TokiToolCall,
    TokiToolCallStream,
)


PROPOSE_PATCH_SCHEMA = StreamingToolSchema({
    "type": "function",
    "function": {
        "name": "propose_patch",
        "description": "Propose replacing one block of text in a file with another.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "The exact text to find and replace."},
                "replacement": {"type": "string", "description": "The new text to substitute in."},
            },
            "required": ["target", "replacement"],
        },
    },
})


class _PatchViz:
    """Tiny live preview that prints the new line each time text is appended."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.replacement = ""
        print(f"--- target ---\n{target}\n--- replacement (streaming) ---")

    def push(self, fragment: str) -> None:
        self.replacement += fragment
        # naive live render: just stream chars to stdout
        print(fragment, end="", flush=True)

    def finalize(self) -> None:
        print()  # newline


def handle_propose_patch(stream: TokiToolCallStream) -> dict:
    """Render the patch live and return the parsed arguments."""
    target_stream = stream.expect_arg("target")
    target = "".join(target_stream)

    viz = _PatchViz(target)
    replacement_stream = stream.expect_arg("replacement")
    for fragment in replacement_stream:
        viz.push(fragment)
    viz.finalize()

    return stream.arguments  # parsed dict, drained


HANDLERS = {
    "propose_patch": handle_propose_patch,
}


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY first")

    model = OpenRouterModel("openai/gpt-4o-mini", api_key=api_key)
    agent = Agent(model, tools=[PROPOSE_PATCH_SCHEMA])
    agent.add_system_message(
        "You help refactor code. When asked to modify a file, propose changes via the propose_patch tool."
    )

    file_text = "def greet():\n    print('hello')\n"
    agent.add_user_message(
        f"Here is a file:\n```\n{file_text}```\n"
        "Please make the greeting more enthusiastic by changing the print statement."
    )

    for chunk in agent.execute(stream=True):
        if isinstance(chunk, TokiToolCallStream):
            handler = HANDLERS.get(chunk.name)
            if handler is None:
                raise ValueError(f"no handler for tool {chunk.name!r}")
            args = handler(chunk)
            print(f"[parsed args: {args}]")
        elif isinstance(chunk, TokiToolCall):
            # any non-streaming tool would land here; not used in this demo
            print(f"[static tool call: {chunk.function.name}({chunk.function.arguments})]")
        else:
            print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
