from easyrepl import REPL

from toki import Agent, LocalModel


model = LocalModel("Qwen/Qwen3-0.6B")
agent = Agent(model)

for query in REPL(history=".chat"):
    agent.add_user_message(query)
    for chunk in agent.execute(stream=True):
        print(chunk, end="", flush=True)
    print()
