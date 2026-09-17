"""
ReAct agent whose tools take the useful structured-arg set:

- dataclass / nested dataclass
- pydantic BaseModel
- datetime, Enum, Path, UUID
- list[T], T | None

Schema generation and JSON→typed-kwargs both go through pydantic
(`create_model` + `model_validate`). That block is the candidate utility;
the rest is a normal toki tool loop.

Extra deps (beyond a hosted toki backend):
- pydantic
- easyrepl
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, get_type_hints
from uuid import UUID, uuid4

from easyrepl import REPL
from pydantic import BaseModel, Field, ValidationError, create_model

from toki import (
    Agent,
    OpenRouterModel,
    TokiToolCall,
    ToolSchema,
    get_openrouter_api_key,
    pretty_tool_call,
)

# --- candidate utility -------------------------------------------------------

@cache
def _arg_model(fn: Callable) -> type[BaseModel]:
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        ann = hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (ann, default)
    return create_model(f"{fn.__name__}Args", **fields)


def openai_tool(fn: Callable) -> ToolSchema:
    parameters = _arg_model(fn).model_json_schema()
    parameters.pop("title", None)
    return ToolSchema({
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": parameters,
        },
    })


def bind(fn: Callable, raw: dict) -> dict:
    model = _arg_model(fn)
    parsed = model.model_validate(raw)
    return {name: getattr(parsed, name) for name in model.model_fields}


def react(query: str, agent: Agent, dispatch: dict[str, Callable]) -> str:
    agent.add_user_message(query)
    while True:
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
            try:
                kwargs = bind(fn, call.function.arguments)
                for name, value in kwargs.items():
                    print(f"  {name}: {type(value).__name__} = {value!r}")
                output = fn(**kwargs)
            except ValidationError as e:
                output = f"invalid arguments:\n{e}"
            agent.add_tool_message(call.id, str(output))


# --- types + tools -----------------------------------------------------------

@dataclass
class Address:
    street: str
    city: str
    zip: str = "00000"


@dataclass
class Venue:
    name: str
    address: Address


class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Ticket(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list)
    id: UUID = Field(default_factory=uuid4)


ADA = UUID("550e8400-e29b-41d4-a716-446655440000")
ALAN = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
PEOPLE = {ADA: "Ada Lovelace", ALAN: "Alan Turing"}
ORDERS: dict[UUID, str] = {}


def directory() -> str:
    """Who you can book a table for. Call this to get ids; do not invent them."""
    return "\n".join(f"{name}  id={uid}" for uid, name in PEOPLE.items())


def book_table(
    when: datetime,
    where: Venue,
    guests: list[UUID] | None = None,
    notes: str | None = None,
) -> str:
    """Reserve a restaurant table.
    `when` is ISO-8601. `where` is a restaurant name plus address (street, city, zip).
    `guests` are directory ids if you want to name who is coming.
    """
    assert isinstance(when, datetime)
    assert isinstance(where, Venue)
    assert isinstance(where.address, Address)
    assert guests is None or all(isinstance(g, UUID) for g in guests)
    assert notes is None or isinstance(notes, str)
    who = ", ".join(PEOPLE.get(g, str(g)) for g in guests) if guests else "unspecified party"
    note = f" notes={notes!r}" if notes else ""
    addr = where.address
    return (
        f"reserved {where.name} @ {addr.street}, {addr.city} {addr.zip} "
        f"on {when.isoformat()} for {who}{note}"
    )


def order_pizza(size: Size, toppings: list[str], dropoff: Address, when: datetime) -> str:
    """Place a pizza delivery. `when` is the ISO-8601 dropoff time. Returns an order id."""
    assert isinstance(size, Size)
    assert isinstance(toppings, list) and all(isinstance(t, str) for t in toppings)
    assert isinstance(dropoff, Address)
    assert isinstance(when, datetime)
    order_id = uuid4()
    topping = ", ".join(toppings) if toppings else "cheese"
    summary = (
        f"{size.value} pizza ({topping}) to {dropoff.street}, {dropoff.city} "
        f"{dropoff.zip} at {when.isoformat()}"
    )
    ORDERS[order_id] = summary
    return f"order {order_id}: {summary}"


def check_order(order_id: UUID) -> str:
    """Look up a pizza order. Use an id returned by order_pizza."""
    assert isinstance(order_id, UUID)
    if order_id not in ORDERS:
        return f"no order {order_id}"
    return f"order {order_id}: {ORDERS[order_id]}"


def save_note(path: Path, priority: Priority, ticket: Ticket | None = None) -> str:
    """Write a note to a file path. Optionally attach a ticket (title, tags, optional id)."""
    assert isinstance(path, Path)
    assert isinstance(priority, Priority)
    assert ticket is None or isinstance(ticket, Ticket)
    extra = f" ticket={ticket.id} {ticket.title!r} tags={ticket.tags}" if ticket else ""
    return f"wrote {path} priority={priority.value}{extra}"


TOOLS = [directory, book_table, order_pizza, check_order, save_note]
DISPATCH = {fn.__name__: fn for fn in TOOLS}

PROMPTS = [
    "who's in the directory?",
    "book a table at Harbor Cafe, 1 Long Wharf, Boston, tomorrow at 7pm",
    "book that table for Ada, notes: window if you can",
    "order a large pizza with mushrooms and olives to 12 Main Street, Boston at 6:30pm",
    "what's the status of that order?",
    "save a high-priority note to /tmp/tonight.md",
]


def main() -> None:
    agent = Agent(
        OpenRouterModel("openai/gpt-5.4-nano", api_key=get_openrouter_api_key()),
        tools=[openai_tool(fn) for fn in TOOLS],
    )
    agent.add_system_message(
        "You are a neighborhood concierge. Use tools; do not invent results. "
        "Call directory() for user ids; use order ids returned by order_pizza."
    )
    print("--- react ---")
    print("try:")
    for prompt in PROMPTS:
        print(f"  {prompt}")
    for query in REPL(history=".chat"):
        react(query, agent, DISPATCH)


if __name__ == "__main__":
    main()
