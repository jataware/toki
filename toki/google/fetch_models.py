"""Dev tooling for the Google (Gemini AI Studio) backend.

`_create_models_types_file` regenerates `toki/google/models.py` from litellm's
bundled metadata, filtering for chat models in the `gemini` provider.
"""

from os import PathLike
from pathlib import Path

from ..litellm.fetch_models import fetch_provider_models, write_models_file


here = Path(__file__).parent


def _create_models_types_file(file: PathLike = here / 'models.py'):
    """Dev helper: create `toki/google/models.py` with `GoogleModelName` and an attributes map."""
    ids_with_attrs = fetch_provider_models("gemini")
    write_models_file(
        file=file,
        name_alias="GoogleModelName",
        fetch_script="toki-fetch-google-models",
        ids_with_attrs=ids_with_attrs,
    )
