import os

def get_openrouter_api_key() -> str:
    """Retrieve the OpenRouter API key from environment variables."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")
    return api_key


from datetime import datetime
def get_timestamp_string() -> str:
    """Get the current timestamp as a string in the format YYYYMMDD-HHMMSS."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")