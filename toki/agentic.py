"""
agentic state machine support tools

[NOTES]
Currently have 2 interfaces for state machines:
- function+context based
- class based
TBD which is better, both seem to have pros and cons:
- class based looks cleaner, though is less flexible. E.g. say you want to have many versions of the same state machine,
  but each with slightly different initializations, it gets messy since there's only one __init__ function.
  more concrete example: for the market scenario, the original version ran several benign buyer rounds before getting to
  the illegal buyer round. Then I made the bootstrap version which skips immedately to the illegal buyer round.
  Using the class based version, it's not straightforward to swap between those two use cases without having a convoluted
  __init__ function.
- function+context based is more flexible, but a bit less cohesive. Basically you're just constructing all the parts used
  in the class version, but you construct them in the global scope and pass them in as arguments

Almost seems like the ideal would be class based, but with an option to keep state/context as a separate object that gets passed in on runtime?
Or just keep both versions. default to the class version, and switch to the function version when needed?
"""

from abc import ABC
from typing import Protocol, TypeVar, Mapping, Generic, Final, Type
from itertools import count
from enum import Enum
from typing import Generator


class Singleton(ABC):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def __repr__(self):
        return f"<{self.__class__.__name__}(Singleton)>"




class EndState(Singleton): ...
END_STATE: Final[EndState] = EndState()

E = TypeVar("E", bound=Enum)  # user's enum of states
C = TypeVar("C")              # context/meta-state object


# signature for state handler functions
class StateHandler(Protocol[E, C]):
    def __call__(self, context: C) -> E | EndState: ...


class StateMachine(Generic[E, C]):
    def __init__(self, states: Type[E], handlers: Mapping[E, StateHandler[E, C]]) -> None:
        self._states = states
        self._handlers = dict(handlers)

        # determine if any handlers are missing
        missing = set(states) - self._handlers.keys()
        if missing:
            names = ", ".join(m.name for m in sorted(missing, key=lambda x: x.value))
            raise ValueError(f"Missing handlers for: {names}")
        extra = set(self._handlers.keys()) - set(states)
        if extra:
            names = ", ".join(m.name for m in sorted(extra, key=lambda x: x.value))
            raise ValueError(f"Extra handlers found for: {names}")

    def step(self, state: E, context: C|None) -> E | EndState:
        # get the handler for the current state
        handler = self._handlers.get(state)
        if handler is None:
            raise ValueError(f"INTERNAL ERROR: No handler defined for state: '{state.name}'. This shouldn't be possible since total coverage is checked on init...")

        # run the handler with or without context if it was defined
        # if context is None: return handler()
        return handler(context)

    def run(self, start_state:E, *, context:C|None=None, max_steps: int|None = None) -> Generator[E|EndState, None, None]:
        current_state = start_state
        iterator = range(max_steps) if isinstance(max_steps, int) else count()
        for _ in iterator:
            yield current_state
            current_state = self.step(current_state, context)
            if current_state is END_STATE:
                break

        yield current_state
        if current_state is not END_STATE:
            print(f'WARNING: state machine exited without reaching END_STATE. Final state: "{current_state}"')


# decorator to mark a method as being for a specific state
def on(state:E):
    def decorator(func:Callable[[C], E|EndState]) -> Callable[[C], E|EndState]:
        func._handler_state = state
        return func
    return decorator


from inspect import ismethod, getmembers
from typing import Callable
class ClassStateMachine(Generic[E, C]):
    """A state machine that operates on a class or instance"""
    def __init__(self, cls_or_instance:Type[C]|C):
        # instantiate the class if it isn't already instantiated. Otherwise use as is.
        self._context = cls_or_instance() if isinstance(cls_or_instance, type) else cls_or_instance
        
        # collect handlers from the class methods, which should be marked with the `on` decorator
        # also save the enum class while collecting handlers
        enum_cls = None
        self._handlers: dict[E, Callable[[C], E|EndState]] = {}
        for name, method in getmembers(self._context, predicate=ismethod):
            if hasattr(method, '_handler_state'):
                if enum_cls is None:
                    enum_cls = method._handler_state.__class__
                elif enum_cls != method._handler_state.__class__:
                    raise ValueError(f"Inconsistent enum classes: {enum_cls} != {method._handler_state.__class__}")
                self._handlers[method._handler_state] = method

        if enum_cls is None:
            raise ValueError("No state class enum found in the context class. Mark handlers with `@on(State.STATE_NAME)`")

        # check that all states are covered
        missing = set(enum_cls) - set(self._handlers.keys())
        if missing:
            raise ValueError(f"Missing handlers for: {missing}")
        extra = set(self._handlers.keys()) - set(enum_cls)
        if extra:
            raise ValueError(f"Extra handlers found for: {extra}")

    # TODO: if keeping both class and function versions, consider
    # reusing these between the two versions rather than duplicating code
    def step(self, state:E) -> E | EndState:
        # get the handler for the current state
        handler = self._handlers.get(state)
        if handler is None:
            raise ValueError(f"INTERNAL ERROR: No handler defined for state: '{state.name}'. This shouldn't be possible since total coverage is checked on init...")

        # handler is already bound to the instance/context, so just call it
        return handler()

    def run(self, start_state:E, *, max_steps: int|None = None) -> Generator[E|EndState, None, None]:
        current_state = start_state
        iterator = range(max_steps) if isinstance(max_steps, int) else count()
        for _ in iterator:
            yield current_state
            current_state = self.step(current_state)
            if current_state is END_STATE:
                break

        yield current_state
        if current_state is not END_STATE:
            print(f'WARNING: state machine exited without reaching END_STATE. Final state: "{current_state}"')


    

######################### example usage of function state machine #########################
if __name__ == "__main__":
    from enum import Enum, auto
    from dataclasses import dataclass
    # from .agentic import on, StateMachine, EndState, END_STATE

    class State(Enum):
        A = auto()
        B = auto()
        C = auto()

    @dataclass
    class Context:
        name: str

    def a(context:Context) -> State|EndState:
        print(f"{context.name} is handling state A")
        return State.B

    def b(context:Context) -> State|EndState:
        print(f"{context.name} is handling state B")
        return State.C

    def c(context:Context) -> State|EndState:
        print(f"{context.name} is handling state C")
        return END_STATE

    # run the state machine
    context = Context("fn Bob")
    handlers = {State.A: a, State.B: b, State.C: c} 
    sm = StateMachine(State, handlers)
    for state in sm.run(State.A, context=context):
        ...
    print("done")


######################### example usage of class state machine #########################
if __name__ == "__main__":
    from enum import Enum, auto
    # from .agentic import on, ClassStateMachine, EndState, END_STATE
    
    class State(Enum):
        A = auto()
        B = auto()
        C = auto()

    class MyScenario:
        def __init__(self, name:str):
            # whatever setup is needed
            self.name = name
        
        @on(State.A)
        def a(self) -> State|EndState:
            #handle state A
            print(f"{self.name} is handling state A")
            return State.B
        
        @on(State.B)
        def b(self) -> State|EndState:
            #handle state B
            print(f"{self.name} is handling state B")
            return State.C
        
        @on(State.C)
        def c(self) -> State|EndState:
            #handle state C
            print(f"{self.name} is handling state C")
            return END_STATE


    scenario = MyScenario("class Bob")
    sm = ClassStateMachine(scenario)
    for state in sm.run(State.A): ...
    print("done")
