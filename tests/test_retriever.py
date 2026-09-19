"""Knowledge base loading and FAISS retrieval.

The retrieve() tests run the real all-MiniLM-L6-v2 model, which is cached
locally after the first run. The _load_knowledge() tests are pure.
"""
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
