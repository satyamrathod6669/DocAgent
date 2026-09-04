import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads the .env file and loads its values

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file locally, or set it "
        "as an environment variable in your deployment platform."
    )

client = Groq(api_key=_api_key)


def call_groq(prompt: str, max_retries: int = 2) -> str:
    """
    Calls the Groq chat completion endpoint with a small retry loop for
    transient failures (rate limits, timeouts, flaky connections).
    Raises the underlying exception if all retries are exhausted, so callers
    can decide how to handle a genuine failure.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_error
