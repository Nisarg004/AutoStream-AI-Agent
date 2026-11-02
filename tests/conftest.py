"""Shared test setup.

pytest loads this before any test module, so the dummy key is in place
before agent.nodes builds its OpenAI client at import time. load_dotenv()
does not override an existing environment variable, so a real key in .env
cannot leak into a test run: no test makes a real API call.
"""
import os
from unittest.mock import MagicMock

os.environ["OPENAI_API_KEY"] = "test-key-never-used"

import pytest

import agent.nodes as nodes
import tools.lead_capture as lead_capture
import utils.intent_classifier as ic


@pytest.fixture(autouse=True)
def no_network_and_no_real_leads(monkeypatch, tmp_path):
    """A test that forgets to stub the LLM gets a failure, not a network call."""
    offline = MagicMock()
    offline.chat.completions.create.side_effect = RuntimeError("real OpenAI call in a test")
    monkeypatch.setattr(ic, "client", offline)
    monkeypatch.setattr(nodes, "client", offline)
    monkeypatch.setattr(lead_capture, "LEADS_FILE", str(tmp_path / "leads.csv"))


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
