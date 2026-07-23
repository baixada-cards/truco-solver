"""LLM client for proposing experiments. Supports Anthropic and OpenAI APIs."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str  # "anthropic" or "openai"
    api_key: str
    model: str


ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL_PREFERENCES = [
    "claude-opus-4-6",
    "claude-opus-4-1",
    "claude-opus-4-1-20250805",
    "claude-opus-4",
    "claude-opus-4-20250514",
    "claude-sonnet-4-6",
    "claude-sonnet-4",
]


def _fetch_anthropic_model_ids(api_key: str) -> list[str]:
    """Fetch available Anthropic model IDs from the official models endpoint."""
    request = urllib.request.Request(
        ANTHROPIC_MODELS_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    return [item["id"] for item in payload.get("data", []) if item.get("id")]


def _resolve_anthropic_model(api_key: str) -> str:
    """Choose the best available Anthropic model, preferring Opus when possible."""
    try:
        model_ids = _fetch_anthropic_model_ids(api_key)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return "claude-opus-4-6"

    for preferred in ANTHROPIC_MODEL_PREFERENCES:
        if preferred in model_ids:
            return preferred

    # Fall back to the newest listed model if our preferred set is missing.
    if model_ids:
        return model_ids[0]

    return "claude-opus-4-6"


def detect_config(model_override: str | None = None) -> LLMConfig:
    """Auto-detect which API to use based on available env vars.

    Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    env_model = os.environ.get("AUTORESEARCH_MODEL")

    if anthropic_key:
        return LLMConfig(
            provider="anthropic",
            api_key=anthropic_key,
            model=model_override
            or env_model
            or _resolve_anthropic_model(anthropic_key),
        )
    elif openai_key:
        return LLMConfig(
            provider="openai",
            api_key=openai_key,
            model=model_override or env_model or "o3-mini",
        )
    else:
        raise RuntimeError(
            "No API key found. Set one of:\n"
            "  ANTHROPIC_API_KEY\n"
            "  OPENAI_API_KEY\n"
            "Use `op run --env-file=.env -- ...` when .env contains op:// references."
        )


def call_llm(config: LLMConfig, prompt: str) -> str:
    """Call the LLM and return the response text."""
    if config.provider == "anthropic":
        return _call_anthropic(config, prompt)
    elif config.provider == "openai":
        return _call_openai(config, prompt)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _call_anthropic(config: LLMConfig, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=8192,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(config: LLMConfig, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        temperature=0.7,
    )
    return response.choices[0].message.content


def propose_experiment(
    config: LLMConfig,
    program_md: str,
    current_code: str,
    results_tsv: str,
    git_history: str = "",
) -> dict:
    """Ask the LLM to propose a new experiment.

    Returns dict with keys: 'description', 'code', 'reasoning'.
    """
    git_section = ""
    if git_history:
        git_section = f"""
## Recent git history (diffs of previous experiments)
```
{git_history}
```
"""

    prompt = f"""You are an autonomous AI researcher optimizing a CFR algorithm.

## Your instructions
{program_md}

## Current experiment file (cfr_experiment.rs)
```rust
{current_code}
```

## Experiment history (results.tsv)
```
{results_tsv}
```
{git_section}
## Your task

Propose the next experiment. You must return:
1. A short description (one line) of what you're trying.
2. Your reasoning (2-3 sentences on why this might work).
3. The complete new contents of `cfr_experiment.rs`.

Format your response exactly as:

DESCRIPTION: <one-line description>

REASONING: <2-3 sentences>

CODE:
```rust
<complete file contents>
```
"""

    text = call_llm(config, prompt)
    return parse_proposal(text)


def parse_proposal(text: str) -> dict:
    """Parse the LLM's response into structured fields."""
    result = {"description": "", "reasoning": "", "code": "", "raw": text}

    # Extract description
    for line in text.split("\n"):
        if line.startswith("DESCRIPTION:"):
            result["description"] = line[len("DESCRIPTION:") :].strip()
            break

    # Extract reasoning
    in_reasoning = False
    reasoning_lines = []
    for line in text.split("\n"):
        if line.startswith("REASONING:"):
            in_reasoning = True
            reasoning_lines.append(line[len("REASONING:") :].strip())
        elif in_reasoning and line.startswith("CODE:"):
            break
        elif in_reasoning:
            reasoning_lines.append(line)
    result["reasoning"] = " ".join(reasoning_lines).strip()

    # Extract code block
    if "```rust" in text:
        code_start = text.index("```rust") + len("```rust")
        code_end = text.index("```", code_start)
        result["code"] = text[code_start:code_end].strip()

    return result
