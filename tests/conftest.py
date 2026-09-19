"""Shared test setup.

pytest loads this before any test module, so the dummy key is in place
before agent.nodes builds its OpenAI client at import time. load_dotenv()
does not override an existing environment variable, so a real key in .env
cannot leak into a test run: no test makes a real API call.
"""
import os

os.environ["OPENAI_API_KEY"] = "test-key-never-used"

import pytest

import agent.nodes as nodes


@pytest.fixture
def captured(monkeypatch):
    """Stub the LLM classifier and the capture tool.

    Returns the list of leads the tool was called with, so a test can assert
    both that it fired and what it received.
    """
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(nodes, "classify_intent", lambda msg: "high_intent")
    monkeypatch.setattr(
        nodes,
        "mock_lead_capture",
        lambda name, email, platform: calls.append((name, email, platform)),
    )
    return calls
