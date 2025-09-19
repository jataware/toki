# Toki
A simple interface for talking to LLMs.

## Features
- `Model`: class for talking to LLMs
- `Agent`: class for maintaining conversation history when talking with a model
- `StateMachine`/`ClassStateMachine`: tools for easily implementing complex agentic interaction scenarios

## Backends
The primary backend is OpenRouter which provides REST API access to all LLMs available on the internet


## Issues
OpenAI models fail to work with tools
- [ ] TODO: consider adding a separate backend just for openai. also consider adding a text only tools interface to skip around the issue


## Dev
### Rebuild models list file
```bash
toki-fetch-models
```
- [ ] TODO: look into making this step part of the github action for publishing new version (or periodically checking the models list and auto updating/building new versions)