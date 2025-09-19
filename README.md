# Toki
A minimal python library for talking to LLMs.

When using this library, in most cases one can simply switch the model name (e.g. `'openai/gpt-5'` -> `'google/gemini-2.5-pro'`) and everything will just work.

## Getting Started
```bash
pip install toki
```

```python
from toki import Model, Agent, get_openrouter_api_key
from easyrepl import REPL

agent = Agent(
    Model('openai/gpt-5', get_openrouter_api_key())
)

for query in REPL():
    agent.add_user_message(query)
    for chunk in agent.execute(stream=True):
        print(chunk, end='', flush=True)
    print()
```

## Features
- `Model`: stateless class for talking to LLMs. includes blocking and streaming
- `Agent`: class for maintaining conversation history when talking with a model
- `StateMachine`/`ClassStateMachine`: tools for easily implementing complex agentic interaction scenarios

## Backends
The primary backend is OpenRouter which provides REST API access to all major LLM models on the market


## Issues
OpenAI models fail to work with tools
- [ ] TODO: consider adding a separate backend just for openai. also consider adding a text only tools interface to skip around the issue


## Dev
### Rebuild models list file
```bash
toki-fetch-models
```
- [ ] TODO: look into making this step part of the github action for publishing new version (or periodically checking the models list and auto updating/building new versions)