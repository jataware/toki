import os


def get_bedrock_api_key() -> str:
    """Retrieve the Amazon Bedrock bearer API key from the environment."""
    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not api_key:
        raise ValueError(
            "Please set the AWS_BEARER_TOKEN_BEDROCK environment variable."
        )
    return api_key
