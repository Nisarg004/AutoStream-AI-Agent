"""mock_lead_capture writes real CSV rows; conftest points it at tmp_path."""
import csv

import agent.nodes as nodes
import tools.lead_capture as lead_capture
from tools.lead_capture import mock_lead_capture


def _rows() -> list[dict]:
    with open(lead_capture.LEADS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_first_capture_writes_a_header_and_a_row():
    mock_lead_capture("Alice", "alice@example.com", "YouTube")

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["Name"] == "Alice"
    assert rows[0]["Email"] == "alice@example.com"
    assert rows[0]["Platform"] == "YouTube"
    assert rows[0]["Timestamp"]


def test_later_captures_append_without_repeating_the_header():
    mock_lead_capture("Alice", "alice@example.com", "YouTube")
    mock_lead_capture("Bob", "bob@example.com", "TikTok")

    assert [row["Name"] for row in _rows()] == ["Alice", "Bob"]


def test_values_with_commas_and_unicode_round_trip():
    mock_lead_capture("Núñez, José", "jose@example.com", "YouTube, TikTok")

    row = _rows()[0]
    assert row["Name"] == "Núñez, José"
    assert row["Platform"] == "YouTube, TikTok"


def test_tool_node_writes_to_the_csv_once_validated():
    update = nodes.tool_execution_node(
        {"lead_name": "Alice", "lead_email": "alice@example.com", "lead_platform": "YouTube"}
    )

    assert update["lead_captured"] is True
    assert len(_rows()) == 1


def test_tool_node_writes_nothing_when_a_field_is_blank(tmp_path):
    update = nodes.tool_execution_node(
        {"lead_name": "   ", "lead_email": "alice@example.com", "lead_platform": "YouTube"}
    )

    assert update["lead_captured"] is False
    assert not (tmp_path / "leads.csv").exists()
