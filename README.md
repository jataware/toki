# Toki

[![PyPI version](https://img.shields.io/pypi/v/toki.svg)](https://pypi.org/project/toki/)

Minimal, universal Python interface for talking to LLMs across backends.

Pick a backend (e.g. OpenRouter, local `transformers`) and keep the same code path. Toki provides a tiny surface:
- `Model` for direct chat completions (blocking and streaming) — one concrete class per backend, all sharing the `BaseModel` interface
- `Agent` for conversation history (with optional tool-calling), works with any `Model`
- `StateMachine` and `ClassStateMachine` for simple agentic flows

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
from toki.openrouter import get_openrouter_api_key
api_key = get_openrouter_api_key()  # raises if not set
```

## Quickstart

### Blocking completion (OpenRouter)
```python
from toki import Agent
from toki.openrouter import Model, get_openrouter_api_key

model = Model('openai/gpt-5', get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("Say hello in 5 words")
result = agent.execute()            # returns str
print(result)
```

### Streaming completion (OpenRouter)
```python
from toki import Agent
from toki.openrouter import Model, get_openrouter_api_key

model = Model('google/gemini-2.5-pro', get_openrouter_api_key())
agent = Agent(model)

agent.add_user_message("Explain diffusion models in 2 sentences.")
for chunk in agent.execute(stream=True):  # yields str chunks
    print(chunk, end='', flush=True)
print()
```

### Local model (transformers)
```python
from toki import Agent
from toki.local import Model

model = Model('Qwen/Qwen3-0.6B')    # any HF causal-LM repo id or local path
agent = Agent(model)

agent.add_user_message("Say hello in 5 words")
print(agent.execute())
```

## Tools (function calling)
Toki can pass OpenRouter-compatible tool schemas to the model. When a tool call is returned, you execute your function(s), then send tool responses back to the model via the `Agent`.

See OpenRouter’s tool-calling docs for the official schema and flow: [Tool & Function Calling](https://openrouter.ai/docs/features/tool-calling).

Blocking example:
```python
import json
from toki import Agent
from toki.openrouter import Model, get_openrouter_api_key

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 25C"  # demo

model = Model('openai/gpt-5', get_openrouter_api_key(), allow_parallel_tool_calls=True)
agent = Agent(model, tools=tools)

agent.add_user_message("What's the weather in Paris?")
result = agent.execute()  # str or {thought:str, tool_calls:list}

if isinstance(result, dict):
    # Execute tool calls and send results back
    for call in result["tool_calls"]:
        args = json.loads(call["function"]["arguments"])  # dict
        if call["function"]["name"] == "get_weather":
            tool_output = get_weather(args["city"])  # run your function
            agent.add_tool_message(call["id"], tool_output)

    # Ask model to continue after tool outputs
    final = agent.execute()
    print(final)
else:
    print(result)
```

Notes:
- In streaming mode, `Agent.execute(stream=True)` yields `str` chunks and may also yield tool-call payloads. The streaming API is designed to be straightforward; use it for responsive UIs/logging. The blocking pattern above is still the simplest entry point when first wiring up tools.
- `allow_parallel_tool_calls=True` lets the model request multiple tools at once when supported.
- WIP: We plan to add utilities to auto-generate tool schemas from Python callables for faster integrations.

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
- OpenRouter model names are strongly typed via `ModelName` (generated from the live OpenRouter model list).
- To view available models at runtime:

```python
from toki.openrouter import list_openrouter_models, get_openrouter_api_key

models = list_openrouter_models(get_openrouter_api_key())
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
Each backend lives under its own submodule and exposes a `Model` class that implements `toki.BaseModel`. Pick the one you want:

- `toki.openrouter` — hosted models via the OpenRouter HTTP API (install `toki[openrouter]`)
- `toki.local` — local models via HuggingFace `transformers` + `torch` (install `toki[local]`)
- `toki.openai`, `toki.google`, `toki.anthropic`, `toki.ollama` — not yet implemented

`Agent` is backend-agnostic and accepts any `Model`. If you're writing a new backend, subclass `toki.BaseModel` and implement `_blocking_complete` and `_streaming_complete`.

## Development
- Python ≥ 3.10
- Optional dev deps: `pip install 'toki[dev]'`
- Useful scripts:
  - `toki-fetch-models` – regenerate model types from OpenRouter
  - `uv version --bump <level>` where `<level>` is one of `major`, `minor`, or `patch`