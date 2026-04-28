from easyrepl import REPL

from toki import Agent, AnthropicModel, get_anthropic_api_key, TokiThinking

model = AnthropicModel("claude-sonnet-4-5", api_key=get_anthropic_api_key(), reasoning_effort="medium")
agent = Agent(model)

for query in REPL(history=".chat"):
    agent.add_user_message(query)
    for chunk in agent.execute(stream=True, capture_thinking=True):
        if isinstance(chunk, TokiThinking):
            print(f"\033[2;32m{chunk.text}\033[0m", end="", flush=True)
        else:
            print(chunk, end="", flush=True)
    print()
