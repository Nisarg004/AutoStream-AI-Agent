"""Knowledge base loading and FAISS retrieval.

The retrieve() tests run the real all-MiniLM-L6-v2 model, which is cached
locally after the first run. The _load_knowledge() tests are pure.
"""
import pytest

from rag.retriever import _load_knowledge, retrieve


def test_knowledge_base_flattens_to_one_chunk_per_entry():
    assert len(_load_knowledge()) == 4  # 2 plans + 2 policies


def test_every_chunk_is_non_empty_text():
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in _load_knowledge())


def test_plan_chunks_carry_price_and_features():
    pro = next(c for c in _load_knowledge() if c.startswith("Pro Plan"))

    assert "$79/month" in pro
    assert "AI captions" in pro


def test_retrieve_ranks_the_pro_plan_first_for_a_pricing_query():
    result = retrieve("How much does the Pro plan cost?")

    assert result.splitlines()[0].startswith("Pro Plan")
    assert "$79/month" in result


def test_retrieve_honours_top_k():
    assert len(retrieve("refund policy", top_k=2).splitlines()) == 2


@pytest.mark.parametrize(
    "query,expected_start",
    [
        ("What does the Basic plan cost?", "Basic Plan"),
        ("Can I get a refund after a week?", "Refund Policy"),
        ("What is your refund policy?", "Refund Policy"),
        ("Who gets 24/7 support?", "Support"),
    ],
)
def test_retrieve_ranks_the_right_chunk_first(query, expected_start):
    assert retrieve(query).splitlines()[0].startswith(expected_start)


def test_default_top_k_returns_three_of_the_four_chunks():
    """Pins the limitation stated in the README: retrieval barely filters."""
    assert len(retrieve("anything").splitlines()) == 3


@pytest.mark.xfail(strict=True, reason="Known weakness: Basic outranks Pro for this feature query")
def test_feature_query_ranks_the_pro_plan_first():
    assert retrieve("Which plan gives me 4K and unlimited videos?").startswith("Pro Plan")


def test_feature_query_still_gets_the_pro_plan_into_the_context():
    assert "Pro Plan" in retrieve("Which plan gives me 4K and unlimited videos?")
