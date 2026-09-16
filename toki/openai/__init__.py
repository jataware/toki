from .models import OpenAIModelName, attributes_map
from .utils import get_openai_api_key, list_openai_models

__all__ = [
    'OpenAIModel',
    'OpenAIModelName',
    'OpenAIResponsesModel',
    'attributes_map',
    'get_openai_api_key',
    'list_openai_models',
]


def __getattr__(name: str):
    if name == 'OpenAIModel':
        from .model import OpenAIModel
        return OpenAIModel
    if name == 'OpenAIResponsesModel':
        from .responses import OpenAIResponsesModel
        return OpenAIResponsesModel
    raise AttributeError(f"module 'toki.openai' has no attribute {name!r}")
