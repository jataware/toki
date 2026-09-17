# Building agents with toki

Toki provides the model-facing parts of an agent:

- one interface across hosted and local model backends
- conversation history in `Agent.messages`
- static and streaming tool calls
- sync and async execution
- reasoning capture, token counting, and prompt caching

The rest of an agent is ordinary Python. Schema libraries turn callables into
tools, your dispatch loop executes them, and focused libraries can supply MCP,
retrieval, workflows, sandboxes, context management, and observability.

This cookbook shows how those pieces fit together. The
[main README](../README.md) remains the API and backend reference.

## What toki leaves to other libraries

Toki deliberately stops at the model boundary. Schema generation from Python
callables, argument validation, MCP clients, retrieval, durable workflows,
sandboxes, context compaction, and tracing are all ordinary integrations: pass
toki the resulting tool schemas and text, and consume `Agent.messages` and the
streamed chunks on the way out. The recipes below name a library where one is
needed for the example; none of them is a toki dependency, and `pip install
toki` keeps the core small.

## The core recipe: typed functions as tools

Most agent applications reduce to four steps:

1. describe Python functions to the model
2. let the model request calls
3. execute those calls in application code
4. return each result and continue until the model answers

Install a backend and the schema helper:

```bash
pip install 'toki[openrouter]' function-schema
export OPENROUTER_API_KEY=...
```

```python
from collections.abc import Callable
from typing import Annotated

from function_schema import Doc, get_function_schema
from toki import (
    Agent,
    OpenRouterModel,
    TokiToolsResponse,
    ToolSchema,
    get_openrouter_api_key,
)


def get_weather(
    city: Annotated[str, Doc("City name, such as Paris")],
) -> str:
    """Get the current weather."""
    return f"{city}: sunny, 25°C"


def as_tool(fn: Callable) -> ToolSchema:
    return ToolSchema({
        "type": "function",
        "function": get_function_schema(fn),
    })


def run_until_answer(
    agent: Agent,
    dispatch: dict[str, Callable[..., object]],
) -> str:
    while True:
        response = agent.execute()
        if not isinstance(response, TokiToolsResponse):
            return response

        for call in response.tool_calls:
            fn = dispatch[call.function.name]
            result = fn(**call.function.arguments)
            agent.add_tool_message(call.id, str(result))


tools = [get_weather]
agent = Agent(
    OpenRouterModel(
        "google/gemini-2.5-flash",
        api_key=get_openrouter_api_key(),
    ),
    tools=[as_tool(fn) for fn in tools],
)
agent.add_user_message("Should I bring a jacket in Paris today?")
print(run_until_answer(agent, {fn.__name__: fn for fn in tools}))
```

This loop is the agent runtime. Tools can query a database, call an API, ask a
human, search documents, delegate to another agent, or run code. Toki keeps the
loop explicit so the application controls authorization, retries, logging, and
side effects.

`function-schema` handles the Python-to-JSON-schema direction. JSON-native
arguments (`str`, `int`, `float`, `bool`, lists, and dictionaries) can go
directly into the function as above.

### Bind structured arguments with Pydantic

JSON does not contain `datetime`, `Path`, `UUID`, `Enum`, dataclass, or Pydantic
instances. When tools accept those types, use Pydantic for both schema generation
and argument binding:

```python
import inspect
from collections.abc import Callable
from functools import cache
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model
from toki import ToolSchema


@cache
def argument_model(fn: Callable) -> type[BaseModel]:
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(fn).parameters.items():
        annotation = hints.get(name, Any)
        default = (
            ...
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        fields[name] = (annotation, default)
    return create_model(f"{fn.__name__}Arguments", **fields)


def as_typed_tool(fn: Callable) -> ToolSchema:
    return ToolSchema({
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": argument_model(fn).model_json_schema(),
        },
    })


def bind(fn: Callable, arguments: dict) -> dict:
    parsed = argument_model(fn).model_validate(arguments)
    return {
        name: getattr(parsed, name)
        for name in type(parsed).model_fields
    }
```

Dispatch with `fn(**bind(fn, call.function.arguments))`. Validation failures can
be returned as the tool result so the model can correct its call.

[`typed_react.py`](typed_react.py) is the complete runnable version. It covers
nested dataclasses, Pydantic models, `datetime`, `Enum`, `Path`, `UUID`,
`list[T]`, and `T | None`.

If the application already uses LangChain tools, do not rebuild their schemas:
`langchain_core.utils.function_calling.convert_to_openai_tool(tool)` produces the
dictionary toki expects, and `tool.invoke(arguments)` executes it.

### Stateful and class-based tools

A tool is a callable plus a schema. Bound methods work naturally and keep state
on their instance:

```python
class ShoppingCart:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, item: str) -> str:
        """Add an item to the shopping cart."""
        self.items.append(item)
        return f"cart: {', '.join(self.items)}"

    def view(self) -> str:
        """Show the shopping cart."""
        return f"cart: {', '.join(self.items)}"


cart = ShoppingCart()
tools = [cart.add, cart.view]
agent = Agent(model, tools=[as_tool(fn) for fn in tools])
dispatch = {fn.__name__: fn for fn in tools}
```

Use explicit names when methods from several objects would collide.

## Common tool patterns

### Human questions and approval

A tool may pause for information:

```python
def ask_user(question: str) -> str:
    """Ask the user for information needed to continue."""
    return input(f"{question}\n> ")
```

For sensitive tools, approve the model's requested arguments before dispatch:

```python
from toki import pretty_tool_call

for call in response.tool_calls:
    print(pretty_tool_call(call))
    if input("Run this action? [y/N] ").lower() == "y":
        output = dispatch[call.function.name](**call.function.arguments)
    else:
        output = "The user denied this action."
    agent.add_tool_message(call.id, str(output))
```

Returning a denial as a tool result lets the model explain or propose another
approach without corrupting the tool-call sequence.

### Structured output

To extract or generate validated data, expose the output model as a tool:

```python
from pydantic import BaseModel, Field
from toki import Agent, TokiToolsResponse, ToolSchema


class Invoice(BaseModel):
    vendor: str
    total: float
    line_items: list[str] = Field(default_factory=list)


SUBMIT_INVOICE = ToolSchema({
    "type": "function",
    "function": {
        "name": "submit_invoice",
        "description": "Submit the extracted invoice. Always call this tool.",
        "parameters": Invoice.model_json_schema(),
    },
})

agent = Agent(model, tools=[SUBMIT_INVOICE])
agent.add_user_message(
    "Extract: Acme Hosting — compute $20, storage $5, total $25."
)
response = agent.execute()
assert isinstance(response, TokiToolsResponse)
invoice = Invoice.model_validate(
    response.tool_calls[0].function.arguments
)
```

This is portable across tool-capable backends. When tool calling is unavailable,
ask for JSON and use `toki.helpers.streaming_parse_json` with `trash_skipper`.

### Sandboxed Python and data analysis

Code execution is also a tool. Use an isolated environment for model-generated
code rather than `exec()` in the application process. For example, E2B Code
Interpreter provides a persistent Python session with files and streaming output:

```python
from e2b_code_interpreter import Sandbox

sandbox = Sandbox.create()


def run_python(code: str) -> str | None:
    """Run Python in an isolated, persistent sandbox."""
    return sandbox.run_code(code).text
```

Register `run_python` with `as_tool`. A persistent sandbox lets the model load a
dataset in one call and analyze or plot it in later calls. Give the sandbox only
the files, credentials, network access, and lifetime required by the task.
Close it when the agent session ends.

The same pattern works with a container service, a notebook kernel, a SQL engine,
or an internal job runner: wrap its execution API in a typed function and return
the useful output.

### Coding agents and live artifact generation

A coding agent is a tool loop with filesystem, search, patch, and test tools.
Keep file mutation and command execution in application-owned functions so policy
and workspace boundaries remain visible.

`StreamingToolSchema` is useful when a tool has a long string argument such as a
patch, SQL query, report, or generated file. A `TokiToolCallStream` exposes that
argument while the model is still producing it, allowing the UI to render a live
preview before execution. See [`streaming_tools.py`](streaming_tools.py).

## External tools through MCP

There are two useful MCP integration levels.

### Minimal: the official MCP client

Use the official Python SDK when the application needs direct control over one
session, including tools, resources, prompts, or server instructions:

```python
from mcp import Client
from toki import ToolSchema

async with Client("http://localhost:8000/mcp") as mcp:
    listed = await mcp.list_tools()
    schemas = [
        ToolSchema({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })
        for tool in listed.tools
    ]
```

Create an `Agent(model, tools=schemas)`, then dispatch each call with
`await mcp.call_tool(call.function.name, call.function.arguments)`. Keep the
client session open for the full agent run.

### Multi-server: `langchain-mcp-adapters`

If the application already uses LangChain tools or needs a convenient
multi-server client, adapt its `BaseTool` objects at the boundary:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from toki import Agent, TokiToolsResponse, ToolSchema


client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],
    },
    "docs": {
        "transport": "http",
        "url": "https://example.com/mcp",
    },
})
tools = await client.get_tools()
dispatch = {tool.name: tool for tool in tools}
agent = Agent(
    model,
    tools=[
        ToolSchema(convert_to_openai_tool(tool))
        for tool in tools
    ],
)

while True:
    response = await agent.aexecute()
    if not isinstance(response, TokiToolsResponse):
        print(response)
        break
    for call in response.tool_calls:
        output = await dispatch[call.function.name].ainvoke(
            call.function.arguments
        )
        agent.add_tool_message(call.id, str(output))
```

Do not bind every tool from every server by default. Tool schemas consume tokens
on every model call and a large undifferentiated toolset reduces selection
quality. Select tools for the task or expose a small discovery layer.

## Retrieval and long-term knowledge

Retrieval output is text. Fetch it before the model call when every request needs
context, or make retrieval a tool when the agent should decide whether and how to
search.

With a LlamaIndex retriever:

```python
def search_docs(query: str) -> str:
    """Search the documentation for passages relevant to a query."""
    nodes = retriever.retrieve(query)
    return "\n\n".join(node.node.get_content() for node in nodes)
```

Register `search_docs` like any other tool. Alternatively, fetch first:

```python
def answer(question: str) -> str:
    context = search_docs(question)
    agent = Agent(model)
    agent.add_system_message(
        "Answer from the supplied context. Say when it is insufficient."
    )
    agent.add_user_message(
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return agent.execute()
```

For a small local corpus, `bm25s` provides fast lexical retrieval without an
embedding service. `qdrant-client[fastembed]` provides local embeddings plus
in-memory, on-disk, or hosted semantic search. LlamaIndex supplies a broad index
and connector ecosystem; Haystack supplies composable ingestion and retrieval
pipelines. Qdrant, LanceDB, Chroma, and other stores can also be queried directly.
Toki only needs the resulting text and optional metadata.

Use retrieval for durable facts and corpora. Use conversation compaction for what
happened during the current task. They solve different problems.

## Delegation and multi-agent systems

A subagent is an `Agent` exposed as a tool to another agent:

```python
def research(topic: str) -> str:
    """Research a topic and return a concise evidence summary."""
    child = Agent(
        make_model(),
        tools=[as_tool(search_web), as_tool(read_page)],
    )
    child.add_system_message(
        "Research the assigned topic. Cite sources and return only a summary."
    )
    child.add_user_message(topic)
    return run_until_answer(
        child,
        {"search_web": search_web, "read_page": read_page},
    )
```

Register `research` on the parent. Give each subagent:

- its own `Agent` and message history
- the narrow tool subset needed for its role
- a task-specific system prompt
- only the parent context required to complete the delegated task

Run independent subagents with `asyncio.gather`. Do not share one `Agent` across
concurrent jobs. For models using explicit prompt caching, separate model
instances also prevent diverging histories from invalidating each other's cache.

## Workflows and durable execution

Use the smallest orchestration mechanism that matches the job:

- a `while` loop for model → tools → model
- toki's `StateMachine` or `ClassStateMachine` for a few explicit in-process
  stages
- `pydantic-graph` for a small independently installable typed graph
- LangGraph when execution needs persistence, resumability, human interrupts,
  conditional branches, fan-out, or time travel
- [Temporal](https://docs.temporal.io/develop/python) when a production workflow
  must survive process failure or run for days

When using LangGraph, keep toki as a node rather than wrapping it as a LangChain
chat model:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from toki import Agent, TokiMessage


class State(TypedDict):
    messages: list[TokiMessage]


def call_model(state: State) -> dict[str, list[TokiMessage]]:
    agent = Agent(model)
    agent.messages = list(state["messages"])
    agent.execute()
    return {"messages": agent.messages}


graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
app = graph.compile()
```

For graphs where nodes append messages, configure an append reducer and return
only each node's new messages. Keep `TokiMessage` objects intact:
`provider_state` can contain Bedrock or OpenAI Responses reasoning data required
by the next tool turn.

## Skills and progressive disclosure

[Agent Skills](https://agentskills.io/specification) package reusable
instructions as a `SKILL.md` plus optional references, scripts, and assets. A
simple loader is enough:

1. put each skill's name and description in the system prompt
2. expose `load_skill(name)` or a general file-reading tool
3. load the full instructions only when the task matches
4. load referenced files only when the instructions require them

```python
from pathlib import Path

skills_dir = Path("skills")
skill_files = {
    path.parent.name: path
    for path in skills_dir.glob("*/SKILL.md")
}
catalog = "\n".join(f"- {name}" for name in skill_files)


def load_skill(name: str) -> str:
    """Load the full instructions for a named skill."""
    return skill_files[name].read_text()


agent.add_system_message(
    "Load a relevant skill before following it.\n"
    f"Available skills:\n{catalog}"
)
```

Register `load_skill` as a tool. This keeps large procedures and manuals out of
the initial prompt while making them discoverable. If the agent already has
filesystem tools, it can read `SKILL.md` directly.

## Context management

`Agent.messages` is the live prompt. Toki exposes it so the application can
choose a policy instead of inheriting one.

The usual order is:

1. cap large tool output before adding it to history
2. replace old tool results with short outcome summaries
3. call `model.count_tokens(...)` to decide when to compact
4. summarize older turns with a separate, cheaper agent
5. retain a fresh tail and complete assistant/tool-result groups
6. call `model.invalidate_cache()` after rewriting a static cached prefix

[`compact.py`](compact.py) implements this policy in a runnable coding-agent
example. It preserves tool-call groups and retains the recent conversation.

Dedicated context-management libraries follow the same contract: whichever
policy produces the shorter history, assign the resulting messages to
`agent.messages`. Do not split an assistant tool call from its matching tool
results. Preserve `provider_state` on retained assistant messages.

## Interactive applications

For a terminal chat, `easyrepl` adds persistent history, reverse search, and
multiline editing without imposing an application framework:

```python
from easyrepl import REPL
from toki import Agent, TokiThinking

agent = Agent(model)
for query in REPL(history=".chat"):
    agent.add_user_message(query)
    for chunk in agent.execute(
        stream=True,
        capture_thinking=True,
    ):
        if isinstance(chunk, TokiThinking):
            print(f"\033[2m{chunk.text}\033[0m", end="", flush=True)
        else:
            print(chunk, end="", flush=True)
    print()
```

[`harness/`](harness/README.md) is the full version of this idea: an engine
that yields typed events, consumed by both a prompt_toolkit CLI and a Textual
UI. For a web or TUI frontend, consume the same stream and translate each chunk
into UI events. Static tools arrive as completed `TokiToolCall` objects; streaming
tools arrive early as `TokiToolCallStream` objects whose arguments can drive live
previews. `prompt_toolkit` supplies async input, completion, and key bindings for
larger terminal applications; Rich supplies Markdown rendering and live display.

## Observability and evals

The tool loop is an effective trace boundary:

- trace each `execute` / `aexecute` call
- record tool name, arguments, output, latency, and approval
- attach `model.usage_metadata` after each model response
- redact prompts, credentials, and sensitive tool output before export

OpenTelemetry with OpenInference is the portable foundation when traces may move
between backends. Langfuse and Braintrust provide Python tracing, datasets,
scoring, and experiment workflows on top of a hosted product. Wrap the model
call, tool dispatch, retrieval, and delegation functions in spans rather than
depending on provider-specific instrumentation. The same fixed `TokiMessage`
history can be replayed against different model backends for regression tests.
For a dedicated model-neutral eval runner with datasets, scorers, sandboxes, and
resumable eval sets, see [Inspect AI](https://inspect.aisi.org.uk/).

For lightweight local evals, ordinary pytest is often sufficient:

```python
def test_invoice_extraction() -> None:
    agent = Agent(model, tools=[SUBMIT_INVOICE])
    agent.add_user_message("Invoice from Acme, total $25")
    response = agent.execute()
    assert isinstance(response, TokiToolsResponse)
    invoice = Invoice.model_validate(
        response.tool_calls[0].function.arguments
    )
    assert invoice.vendor == "Acme"
    assert invoice.total == 25
```

## Runnable examples

- [`local_stream_repl.py`](local_stream_repl.py) — local streaming chat and
  reasoning display
- [`local_chat.py`](local_chat.py) — a local model running a tool loop
- [`typed_react.py`](typed_react.py) — Pydantic schemas and typed argument
  binding
- [`streaming_tools.py`](streaming_tools.py) — consume tool arguments while they
  are generated
- [`compact.py`](compact.py) — prune tool output and summarize conversation
  history
- [`harness/`](harness/README.md) — a Cursor-style coding agent: agent/ask/plan
  modes, workspace tools, shell permission levels, file-backed plans, web
  search, sessions, undo; one engine with a plain CLI and a Textual UI
- [`test_thinking.py`](test_thinking.py) — stream Anthropic reasoning

The common architecture is deliberately small:

```text
user or workflow
      ↓
Agent.messages → toki model → provider
      ↑                ↓
 tool results ← application dispatch
      ↑
Python functions, MCP, retrieval, sandboxes, subagents
```

Toki owns the provider boundary and conversation representation. The application
owns capabilities and policy.
