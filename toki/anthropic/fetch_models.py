"""Dev tooling for the Anthropic backend.

`_create_models_types_file` regenerates `toki/anthropic/models.py` from
litellm's bundled metadata, filtering for chat models in the `anthropic`
provider.
"""

from os import PathLike
from pathlib import Path

from ..litellm.fetch_models import fetch_provider_models, write_models_file


here = Path(__file__).parent


def _create_models_types_file(file: PathLike = here / 'models.py'):
    """Dev helper: create `toki/anthropic/models.py` with `AnthropicModelName` and an attributes map."""
    ids_with_attrs = fetch_provider_models("anthropic")
    write_models_file(
        file=file,
        name_alias="AnthropicModelName",
        fetch_script="toki-fetch-anthropic-models",
        ids_with_attrs=ids_with_attrs,
    )
