# Toki

[![PyPI version](https://img.shields.io/pypi/v/toki.svg)](https://pypi.org/project/toki/)

Minimal, universal Python interface for talking to LLMs across backends.

Pick a backend (e.g. OpenRouter, local `transformers`) and keep the same code path. Toki provides a tiny surface:
- `<Provider>Model` for direct chat completions (blocking and streaming) — one concrete class per backend (e.g. `OpenRouterModel`, `LocalModel`), all sharing the `BaseModel` interface
- `Agent` for conversation history (with optional tool-calling), works with any model
- `StateMachine` and `ClassStateMachine` for simple agentic flows

Toki targets instruction-tuned chat models — anything that ships a tokenizer `chat_template` (Qwen-Instruct, Llama-Instruct, Gemma-`-it`, etc.). Base / pretrained-only checkpoints aren't supported; for raw text continuation, use `transformers` directly.

Browse all OpenRouter models: [openrouter.ai/models](https://openrouter.ai/models).

## Install
Backend deps are split into extras. Install only what you need:

```bash
pip install 'toki[openrouter]'     # OpenRouter HTTP API
pip install 'toki[local]'          # local models via transformers + torch
pip install 'toki[all]'            # everything
```

Plain `pip install toki` installs only the backend-agnostic core (`BaseModel`, `Agent`, types, state machines).

## Configure
The OpenRouter backend needs an API key:

```bash
export OPENROUTER_API_KEY=...  # https://openrouter.ai/
```

Or retrieve it in code:

```python
from toki import get_openrouter_api_key
api_key = get_openrouter_api_key()  # raises if not set
```

## Quickstart

### Blocking completion (OpenRouter)
```python
from toki import Agent, OpenRouterModel, get_openrouter_api_key

model = OpenRouterModel('openai/gpt-5', api_key=get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("Say hello in 5 words")
result = agent.execute()            # returns str
print(result)
```

### Streaming completion (OpenRouter)
```python
from toki import Agent, OpenRouterModel, get_openrouter_api_key

model = OpenRouterModel('google/gemini-2.5-pro', api_key=get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("Explain diffusion models in 2 sentences.")
for chunk in agent.execute(stream=True):  # yields str chunks
    print(chunk, end='', flush=True)
print()
```

### Local model (transformers)
```python
from toki import Agent, LocalModel

model = LocalModel('Qwen/Qwen3-0.6B')    # any HF causal-LM repo id or local path
agent = Agent(model)

agent.add_user_message("Say hello in 5 words")
print(agent.execute())
```

## Tools (function calling)
Toki can pass OpenRouter-compatible tool schemas to the model. When a tool call is returned, you execute your function(s), then send tool responses back to the model via the `Agent`.

See OpenRouter’s tool-calling docs for the official schema and flow: [Tool & Function Calling](https://openrouter.ai/docs/features/tool-calling).

Tool schemas can be passed as raw dicts or wrapped in `ToolSchema(...)` (synonymous; the wrapper is purely for typing).

Blocking example:
```python
from toki import Agent, OpenRouterModel, ToolSchema, TokiToolsResponse, get_openrouter_api_key

tools = [
    ToolSchema({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }),
]

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 25C"  # demo

model = OpenRouterModel('openai/gpt-5', api_key=get_openrouter_api_key(), allow_parallel_tool_calls=True)
agent = Agent(model, tools=tools)

agent.add_user_message("What's the weather in Paris?")
result = agent.execute()  # str or TokiToolsResponse[TokiToolCall]

if isinstance(result, str):
    print(result)
else:
    assert isinstance(result, TokiToolsResponse)
    for call in result.tool_calls:
        if call.function.name == "get_weather":
            tool_output = get_weather(call.function.arguments["city"])  # run your function
            agent.add_tool_message(call.id, tool_output)

    final = agent.execute()
    print(final)
```

Streaming mode yields each tool call as a bare `TokiToolCall` object as soon as the model finishes producing it:

```python
from toki import TokiToolCall

for chunk in agent.execute(stream=True):
    if isinstance(chunk, TokiToolCall):
        print(f"[tool: {chunk.function.name}({chunk.function.arguments})]")
    else:
        print(chunk, end="", flush=True)
```

Notes:
- `allow_parallel_tool_calls=True` lets the model request multiple tools at once when supported.
- WIP: utilities to auto-generate tool schemas from Python callables.

## Streaming tool calls
For tools whose argument values you want to consume *as they arrive* (rather than waiting for the whole call to land), declare them with `StreamingToolSchema(...)`. The schema dict is identical to the static case; the wrapper only changes how the call is surfaced.

In streaming mode, each invocation of a streaming-flagged tool yields a `TokiToolCallStream` once the model has emitted the tool's id and name. Argument values can be consumed via:

- `expect_arg(name)` returns a `TokiArgStream` for that one argument. Iterating it yields decoded characters (for string args) or raw JSON-text fragments (for numbers, booleans, null, arrays, objects). Order-independent: claim args in any order, claim already-completed args as a single-shot replay, and `expect_arg` raises if the argument never appears.
- `items()` iterates `(name, TokiArgStream)` pairs in the order the model emits them.
- `arguments` (after the stream has been drained) returns the parsed args dict.

`expect_arg` and `items()` are mutually exclusive and one-shot per `TokiToolCallStream`.

```python
from toki import Agent, OpenRouterModel, StreamingToolSchema, TokiToolCallStream, get_openrouter_api_key

PROPOSE_PATCH = StreamingToolSchema({
    "type": "function",
    "function": {
        "name": "propose_patch",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "required": ["target", "replacement"],
        },
    },
})

def handle_propose_patch(stream: TokiToolCallStream) -> None:
    target = "".join(stream.expect_arg("target"))
    print(f"--- target ---\n{target}\n--- replacement ---")
    for chunk in stream.expect_arg("replacement"):
        print(chunk, end="", flush=True)
    print()

agent = Agent(OpenRouterModel('openai/gpt-4o-mini', api_key=get_openrouter_api_key()), tools=[PROPOSE_PATCH])
agent.add_user_message("Propose a small patch to make `print('hi')` more enthusiastic.")
for chunk in agent.execute(stream=True):
    if isinstance(chunk, TokiToolCallStream):
        handle_propose_patch(chunk)
    else:
        print(chunk, end="", flush=True)
```

In blocking mode (`stream=False`), streaming-flagged tools still come back as `TokiToolCallStream` objects (pre-drained, so the only liveness is lost) for API symmetry — the same handler code works either way.

Mixing static and streaming tools in the same `Agent` is fine; static tools yield as `TokiToolCall`, streaming tools as `TokiToolCallStream`.

## Capturing thinking
Reasoning models (OpenAI o-series, Anthropic Claude with thinking, DeepSeek-R1, QwQ, Qwen3 thinking variants, etc.) produce internal "thinking" before their final answer. By default toki strips this — your stream stays a clean stream of answer text. Pass `capture_thinking=True` to surface it as `TokiThinking` chunks (streaming) or as a `thought` field on the response object (blocking).

Streaming:
```python
from toki import Agent, OpenRouterModel, TokiThinking, get_openrouter_api_key

model = OpenRouterModel('openai/o4-mini', api_key=get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("If a train travels 60 mph for 2.5 hours, how far does it go?")
for chunk in agent.execute(stream=True, capture_thinking=True):
    if isinstance(chunk, TokiThinking):
        print(f"\033[2m{chunk.text}\033[0m", end='', flush=True)  # dim
    elif isinstance(chunk, str):
        print(chunk, end='', flush=True)
print()
```


Blocking:
```python
from toki import Agent, OpenRouterModel, TokiThoughtResponse, get_openrouter_api_key

model = OpenRouterModel('openai/o4-mini', api_key=get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("Solve: 9.9 vs 9.11, which is larger?")
result = agent.execute(capture_thinking=True)  # TokiThoughtResponse
assert isinstance(result, TokiThoughtResponse)
print("thought:", result.thought)
print("answer:", result.content)
```

When tools are configured, blocking mode returns `TokiToolsThoughtResponse[T]` (which also carries a `thought` field) whenever the model invoked a tool.

Notes:
- Thinking text is *not* added back to message history; round-tripping reasoning context across turns is not yet supported.
- For OpenRouter, toki sets `reasoning: {enabled: true}` in the request when `capture_thinking=True`. For the local backend, toki parses `<think>...</think>` tags inline.

## Agentic flows with Implicit State Machines
Toki includes lightweight state machines to structure multi-step interactions. State machines are implicit as state transitions are controlled solely by the return value(s) of each state handler function, as opposed to a more global description of the graph.

Function + context version:
```python
from enum import Enum, auto
from dataclasses import dataclass
from toki import StateMachine, on, EndState, END_STATE

class State(Enum):
    A = auto()
    B = auto()
    C = auto()

@dataclass
class Context:
    name: str

def a(ctx: Context):
    print(f"{ctx.name} handling A")
    return State.B

def b(ctx: Context):
    print(f"{ctx.name} handling B")
    return State.C

def c(ctx: Context):
    print(f"{ctx.name} handling C")
    return END_STATE

sm = StateMachine(State, {State.A: a, State.B: b, State.C: c})
for s in sm.run(State.A, context=Context("Alice")):
    ...
```

Class-based version:
```python
from enum import Enum, auto
from toki import ClassStateMachine, on, END_STATE

class State(Enum):
    A = auto(); B = auto(); C = auto()

class Scenario:
    def __init__(self, name: str):
        self.name = name

    @on(State.A)
    def a(self):
        print(f"{self.name} handling A")
        return State.B

    @on(State.B)
    def b(self):
        print(f"{self.name} handling B")
        return State.C

    @on(State.C)
    def c(self):
        print(f"{self.name} handling C")
        return END_STATE

sm = ClassStateMachine(Scenario("Bob"))
for s in sm.run(State.A):
    ...
```

## Models and Types
- OpenRouter model names are strongly typed via `OpenRouterModelName` (generated from the live OpenRouter model list).
- To view available models at runtime:

```python
from toki.openrouter import list_openrouter_models

models = list_openrouter_models()
print(len(models), "models")
print(models[:10])
```

### Getting model attributes
Each generated OpenRouter model has metadata in `toki.openrouter.attributes_map`, including context window and whether the model supports tools (as reported by OpenRouter):

```python
from toki.openrouter import attributes_map

attr = attributes_map['google/gemini-2.5-pro']
print(attr.context_size, attr.supports_tools)
```

## Backends
Each backend lives under its own submodule and exposes a `<Provider>Model` class that implements `toki.BaseModel` (also re-exported at the top level). Pick the one you want:

- `toki.OpenRouterModel` — hosted models via the OpenRouter HTTP API (install `toki[openrouter]`)
- `toki.LocalModel` — local models via HuggingFace `transformers` + `torch` (install `toki[local]`)
- `toki.openai`, `toki.google`, `toki.anthropic`, `toki.ollama` — not yet implemented

`Agent` is backend-agnostic and accepts any `BaseModel`. If you're writing a new backend, subclass `toki.BaseModel` and implement just two methods — `_raw_blocking` (returns a `_RawTurn`) and `_raw_streaming` (yields `_RawContentChunk` / `_RawThoughtChunk` / `_RawToolCallChunk` / `_RawUsage`). All public-facing types, schema unwrapping, and streaming JSON parsing happen in the base class.

## Development
- Python ≥ 3.10
- Optional dev deps: `pip install 'toki[dev]'`
- Useful scripts:
  - `toki-fetch-models` – regenerate model types from OpenRouter
  - `uv version --bump <level>` where `<level>` is one of `major`, `minor`, or `patch`