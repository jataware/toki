"""Dev tooling for the OpenAI backend.

`_create_models_types_file` regenerates `toki/openai/models.py` from litellm's
bundled metadata, filtering for chat models in the `openai` provider.
"""

from os import PathLike
from pathlib import Path

from ..litellm.fetch_models import fetch_provider_models, write_models_file


here = Path(__file__).parent


def _create_models_types_file(file: PathLike = here / 'models.py'):
    """Dev helper: create `toki/openai/models.py` with `OpenAIModelName` and an attributes map."""
    ids_with_attrs = fetch_provider_models("openai")
    write_models_file(
        file=file,
        name_alias="OpenAIModelName",
        fetch_script="toki-fetch-openai-models",
        ids_with_attrs=ids_with_attrs,
    )
