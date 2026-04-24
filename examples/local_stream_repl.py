from easyrepl import REPL

from toki import Agent
from toki.local import Model


model = Model("Qwen/Qwen3-1.7B")
agent = Agent(model)

for query in REPL(history=".chat"):
    agent.add_user_message(query)
    for chunk in agent.execute(stream=True):
        print(chunk, end="", flush=True)
    print()
