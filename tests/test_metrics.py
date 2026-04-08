from faar.metrics import exact_match, ndcg_at_k, normalize_text, recall_at_k, token_f1


def test_normalize_text_basic() -> None:
    assert normalize_text("  Hello,   WORLD! ") == "hello world"


def test_exact_match_and_f1() -> None:
    assert exact_match("Amount 100", "amount 100") == 1.0
    assert exact_match("Amount 100", "amount 101") == 0.0
    assert token_f1("amount 100 usd", "100 amount") > 0.6


def test_retrieval_metrics() -> None:
    hits = ["alpha", "the answer is 42", "beta"]
    assert recall_at_k(hits, "42", k=2) == 1.0
    assert ndcg_at_k(hits, "42", k=3) > 0.0
