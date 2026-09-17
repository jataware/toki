"""
Token-budget compaction for a live `Agent.messages` list.

Cheapest first: cap tool output as it enters history, stub old tool results,
then (if still over budget) summarize the middle with a *separate* cheaper
Agent and keep a fresh tail. Assistant+tool groups are dropped or kept as a
unit so the next `execute()` doesn't see an orphaned tool result. After
rewriting history under `cache='static'`, `invalidate_cache()`.

This is a policy, not a toki feature. `count_tokens` decides *when*; you
mutate `agent.messages` in the open, not inside `execute()`.

Extra deps (beyond a hosted toki backend):
- easyrepl
- litellm  (`count_tokens(kind="offline")`)
"""

from __future__ import annotations

from collections.abc import Callable

from easyrepl import REPL

from toki import (
    Agent,
    OpenRouterModel,
    TokenCountEstimate,
    TokiMessage,
    TokiToolCall,
    ToolSchema,
    get_openrouter_api_key,
    pretty_tool_call,
)

# --- candidate utility -------------------------------------------------------


def prompt_tokens(agent: Agent) -> int:
    n = agent.model.count_tokens(agent.messages, tools=agent.tools, kind="offline")
    return n.prompt_tokens if isinstance(n, TokenCountEstimate) else n


def compact(agent: Agent, summarizer: Agent, *, budget: int, keep_last: int = 6) -> bool:
    """Prune, then maybe summarize. Returns True if `agent.messages` changed."""
    before = prompt_tokens(agent)
    if before <= budget:
        return False

    pruned = prune_tool_results(agent.messages, keep_last=2)
    after_prune = prompt_tokens(agent)
    if after_prune <= budget:
        agent.model.invalidate_cache()
        print(f"compact: pruned {pruned} tool results  {before} -> {after_prune} tokens")
        return True

    sys, rest = _split_system(agent.messages)
    groups = _groups(rest)
    if len(groups) <= keep_last:
        if pruned:
            agent.model.invalidate_cache()
            print(f"compact: pruned {pruned} tool results  {before} -> {after_prune} tokens")
        return bool(pruned)

    head = [m for g in groups[:-keep_last] for m in g]
    tail = [m for g in groups[-keep_last:] for m in g]
    summary = _summarize(summarizer, head)
    agent.messages = [
        *sys,
        TokiMessage(role="user", content=f"[conversation so far]\n{summary}"),
        *tail,
    ]
    agent.model.invalidate_cache()
    after = prompt_tokens(agent)
    print(
        f"compact: pruned {pruned} tool results, summarized {len(head)} messages  "
        f"{before} -> {after} tokens"
    )
    return True


def prune_tool_results(messages: list[TokiMessage], *, keep_last: int) -> int:
    """Stub every tool result except the most recent `keep_last` ones."""
    tools = [m for m in messages if m.role == "tool"]
    n = 0
    for m in tools[:-keep_last]:
        if m.content and not m.content.startswith("(truncated;"):
            m.content = f"(truncated; {len(m.content)} chars)"
            n += 1
    return n


def _split_system(messages: list[TokiMessage]) -> tuple[list[TokiMessage], list[TokiMessage]]:
    sys = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return sys, rest


def _groups(messages: list[TokiMessage]) -> list[list[TokiMessage]]:
    groups: list[list[TokiMessage]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.tool_calls:
            ids = {tc.id for tc in m.tool_calls}
            group = [m]
            i += 1
            while i < len(messages) and messages[i].role == "tool" and messages[i].tool_call_id in ids:
                group.append(messages[i])
                i += 1
            groups.append(group)
        else:
            groups.append([m])
            i += 1
    return groups


def _summarize(summarizer: Agent, head: list[TokiMessage]) -> str:
    summarizer.messages = []
    summarizer.add_user_message(
        "Summarize this conversation for a future assistant. Keep decisions, "
        "file paths, test failures, and open todos. Drop chatter and raw logs.\n\n"
        + "\n".join(_render(m) for m in head)
    )
    return summarizer.execute()


def _render(m: TokiMessage, limit: int = 1500) -> str:
    body = m.content[:limit]
    if m.tool_calls:
        names = ", ".join(tc.function.name for tc in m.tool_calls)
        return f"assistant called {names}\n{body}".rstrip()
    if m.role == "tool":
        return f"tool result: {body}"
    return f"{m.role}: {body}"


# --- demo workspace + tools --------------------------------------------------

FILES = {
    "src/app.py": '''\
def login(user, password):
    # TODO: hash the password; currently compared in plaintext
    if user is None:
        return None
    return user if password == user.password else None


def process(n):
    """Double n. Off-by-one when n is even — see test_batch_*."""
    if n % 2 == 0:
        return n * 2 - 1
    return n * 2


def greet(name):
    return f"hello, {name}"
''',
    "src/utils.py": '''\
def load_users(path):
    users = []
    with open(path) as f:
        for line in f:
            name, password = line.strip().split(":")
            users.append(User(name, password))
    return users


class User:
    def __init__(self, name, password):
        self.name = name
        self.password = password
''',
    "tests/test_app.py": '''\
from src.app import login, process, greet

def test_login():
    assert login(None, "x") is not None  # currently fails: user is None

def test_greet():
    assert greet("ada") == "hello, ada"
''',
}


def list_files() -> str:
    """List files in the workspace."""
    return "\n".join(sorted(FILES))


def read_file(path: str) -> str:
    """Read a workspace file. `path` is relative, e.g. src/app.py."""
    if path not in FILES:
        return f"no such file: {path}. known: {', '.join(sorted(FILES))}"
    return FILES[path]


def run_tests() -> str:
    """Run the test suite. Returns the full log."""
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.0",
        "collected 40 items",
        "",
        "tests/test_app.py::test_login FAILED",
        "tests/test_app.py::test_greet PASSED",
    ]
    for i in range(38):
        lines += [
            f"tests/test_app.py::test_batch_{i} FAILED",
            f"    def test_batch_{i}():",
            f"        assert process({i}) == {i * 2}",
            f"E       AssertionError: {process_wrong(i)} != {i * 2}",
            "",
        ]
    lines += [
        "=========================== short test summary info ============================",
        "FAILED tests/test_app.py::test_login - AssertionError: user is None",
        "FAILED tests/test_app.py::test_batch_0 - AssertionError: off-by-one on evens",
        "======================== 39 failed, 1 passed in 4.21s =========================",
    ]
    return "\n".join(lines)


def process_wrong(n: int) -> int:
    return n * 2 - 1 if n % 2 == 0 else n * 2


TOOLS = [list_files, read_file, run_tests]
DISPATCH: dict[str, Callable[..., str]] = {fn.__name__: fn for fn in TOOLS}

TOOL_SCHEMAS = [
    ToolSchema({
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }),
    ToolSchema({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a workspace file. path is relative, e.g. src/app.py.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }),
    ToolSchema({
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite. Returns the full log.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }),
]

BUDGET = 2500
MAX_TOOL_CHARS = 2500

PROMPTS = [
    "what files are in the workspace?",
    "read src/app.py",
    "read src/utils.py",
    "run the tests",
    "what's still failing and where?",
]


def react(query: str, agent: Agent, summarizer: Agent, dispatch: dict[str, Callable[..., str]]) -> str:
    agent.add_user_message(query)
    while True:
        compact(agent, summarizer, budget=BUDGET)
        print(f"[{prompt_tokens(agent)} tokens]")
        chunks: list[str] = []
        tool_calls: list[TokiToolCall] = []
        for chunk in agent.execute(stream=True):
            if isinstance(chunk, TokiToolCall):
                tool_calls.append(chunk)
            else:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        if chunks:
            print()
        if not tool_calls:
            return "".join(chunks)
        for call in tool_calls:
            print(f"tool: {pretty_tool_call(call)}")
            fn = dispatch[call.function.name]
            output = fn(**call.function.arguments)
            if len(output) > MAX_TOOL_CHARS:
                output = output[:MAX_TOOL_CHARS] + f"\n(truncated; {len(output)} chars total)"
            agent.add_tool_message(call.id, output)


def main() -> None:
    key = get_openrouter_api_key()
    agent = Agent(
        OpenRouterModel("anthropic/claude-haiku-4.5", api_key=key, cache="static"),
        tools=TOOL_SCHEMAS,
    )
    summarizer = Agent(OpenRouterModel("anthropic/claude-haiku-4.5", api_key=key))
    agent.add_system_message(
        "You are a coding assistant for a tiny workspace. Use tools; do not invent file contents."
    )
    print(f"budget {BUDGET} tokens (offline). compact runs before each model call.")
    print("try:")
    for prompt in PROMPTS:
        print(f"  {prompt}")
    for query in REPL(history=".chat"):
        react(query, agent, summarizer, DISPATCH)


if __name__ == "__main__":
    main()
