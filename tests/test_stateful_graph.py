"""State persistence through the checkpointer, and history reaching the LLM."""
import uuid
from unittest.mock import MagicMock

import agent.nodes as nodes
from agent.graph import build_graph, graph
from langgraph.checkpoint.memory import MemorySaver


def _config(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _scripted_client(*replies: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=reply))]) for reply in replies
    ]
    return client


def _sent_messages(client: MagicMock, call: int) -> list[dict]:
    return client.chat.completions.create.call_args_list[call].kwargs["messages"]


def test_lead_fields_persist_when_each_turn_sends_only_the_new_message(captured):
    config = _config()
    graph.invoke({"user_message": "sign me up"}, config)
    graph.invoke({"user_message": "Alice"}, config)

    saved = graph.get_state(config).values
    assert saved["lead_name"] == "Alice"
    assert saved["lead_collection_step"] == "ask_email"


def test_a_full_lead_completes_across_four_invokes(captured):
    config = _config()
    for message in ("sign me up", "Alice", "alice@example.com", "YouTube"):
        result = graph.invoke({"user_message": message}, config)

    assert captured == [("Alice", "alice@example.com", "YouTube")]
    assert result["lead_captured"] is True


def test_threads_do_not_share_state(captured):
    first, second = _config(), _config()
    graph.invoke({"user_message": "sign me up"}, first)
    graph.invoke({"user_message": "Alice"}, first)
    graph.invoke({"user_message": "sign me up"}, second)

    assert graph.get_state(first).values["lead_name"] == "Alice"
    assert graph.get_state(second).values.get("lead_name") is None
    assert graph.get_state(second).values["lead_collection_step"] == "ask_name"


def test_state_survives_rebuilding_the_graph_on_the_same_checkpointer(captured):
    saver = MemorySaver()
    config = _config()
    build_graph(saver).invoke({"user_message": "sign me up"}, config)

    result = build_graph(saver).invoke({"user_message": "Alice"}, config)

    assert result["lead_name"] == "Alice"


def test_rag_call_receives_the_previous_exchange(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "product_query")
    monkeypatch.setattr(nodes, "retrieve", lambda q: "Pro Plan: $79/month.")
    client = _scripted_client("Pro is $79/month.", "Yes, it has AI captions.")
    monkeypatch.setattr(nodes, "client", client)
    config = _config()

    graph.invoke({"user_message": "How much is Pro?"}, config)
    graph.invoke({"user_message": "Does it have captions?"}, config)

    sent = _sent_messages(client, 1)
    assert {"role": "user", "content": "How much is Pro?"} in sent
    assert {"role": "assistant", "content": "Pro is $79/month."} in sent
    assert sent[-1]["content"].endswith("Question: Does it have captions?")


def test_greeting_call_receives_the_previous_exchange(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "greeting")
    client = _scripted_client("Hi Sam!", "Bye Sam!")
    monkeypatch.setattr(nodes, "client", client)
    config = _config()

    graph.invoke({"user_message": "hi, I'm Sam"}, config)
    graph.invoke({"user_message": "bye"}, config)

    assert {"role": "assistant", "content": "Hi Sam!"} in _sent_messages(client, 1)


def test_history_sent_to_the_llm_is_capped(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "greeting")
    client = _scripted_client(*[f"reply {i}" for i in range(8)])
    monkeypatch.setattr(nodes, "client", client)
    config = _config()

    for i in range(8):
        graph.invoke({"user_message": f"message {i}"}, config)

    sent = _sent_messages(client, 7)
    assert len(sent) == 1 + nodes._HISTORY_WINDOW + 1  # system + window + new message
    assert {"role": "user", "content": "message 0"} not in sent


def test_lead_answers_never_reach_the_llm_history(monkeypatch, captured):
    monkeypatch.setattr(nodes, "retrieve", lambda q: "context")
    client = _scripted_client("It is $79/month.")
    monkeypatch.setattr(nodes, "client", client)
    config = _config()
    for message in ("sign me up", "Alice", "alice@example.com", "YouTube"):
        graph.invoke({"user_message": message}, config)
    monkeypatch.setattr(nodes, "classify_intent", lambda m: "product_query")

    graph.invoke({"user_message": "How much is Pro?"}, config)

    assert "alice@example.com" not in str(_sent_messages(client, 0))
    assert "alice@example.com" not in str(graph.get_state(config).values["chat_history"])
