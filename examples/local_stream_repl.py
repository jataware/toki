from easyrepl import REPL

from toki import Agent, LocalModel, TokiThinking

model = LocalModel("Qwen/Qwen3-0.6B")
agent = Agent(model)

for query in REPL(history=".chat"):
    agent.add_user_message(query)
    for chunk in agent.execute(stream=True, capture_thinking=True):
        if isinstance(chunk, TokiThinking):
            print(f"\033[2;32m{chunk.text}\033[0m", end="", flush=True)
        else:
            print(chunk, end="", flush=True)
    print()
