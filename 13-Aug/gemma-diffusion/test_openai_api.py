#!/usr/bin/env python3
import os
import sys

from openai import OpenAI


def main() -> int:
    """
    Can be run with the following command:
    curl http://selab.nhtlongcs.com:20571/v1/models -H "Authorization: Bearer 14June2026"
    """
    base_url = os.getenv("OPENAI_BASE_URL", "http://selab.nhtlongcs.com:20571/v1")
    api_key = os.getenv("OPENAI_API_KEY", "14June2026")
    prompt = " ".join(sys.argv[1:]) or "Say hello in one sentence."

    client = OpenAI(base_url=base_url, api_key=api_key)
    models = client.models.list()
    if not models.data:
        raise RuntimeError("The server returned no models")

    model = os.getenv("OPENAI_MODEL", models.data[0].id)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The server returned an empty completion")

    print(f"Model: {model}")
    print(f"Response: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
