"""Evaluation harness: 10 test questions across countries and languages."""
import os
import time
import logging
import httpx
import sys

logging.basicConfig(level=logging.WARNING)

API_BASE = os.environ.get("EVAL_API_BASE", "http://localhost:8000")
# Gemini free tier is often ~5 requests/minute per model; space calls to avoid 429s.
EVAL_SLEEP_SECONDS = float(os.environ.get("EVAL_SLEEP_SECONDS", "20"))
REQUEST_TIMEOUT = float(os.environ.get("EVAL_REQUEST_TIMEOUT", "120"))

TEST_CASES = [
    {
        "name": "Country A - Return policy (en)",
        "question": "What is your return policy?",
        "country": "A",
        "language": "en",
        "expected_content_ids": ["a_faq_returns_en"],
        "expected_keywords": ["48 hours", "defective"],
        "should_have_answer": True,
    },
    {
        "name": "Country B - Return policy (en) - different window than A",
        "question": "What is your return policy?",
        "country": "B",
        "language": "en",
        "expected_content_ids": ["b_faq_returns_en"],
        "expected_keywords": ["7 days"],
        "should_have_answer": True,
    },
    {
        "name": "Country B - Return policy (es)",
        "question": "¿Cuál es su política de devoluciones?",
        "country": "B",
        "language": "es",
        "expected_content_ids": ["b_faq_returns_es"],
        "expected_keywords": ["7 días"],
        "should_have_answer": True,
    },
    {
        "name": "Country A - Account closure (hi)",
        "question": "मैं अपना खाता कैसे बंद करूं?",
        "country": "A",
        "language": "hi",
        "expected_content_ids": ["a_faq_account_hi"],
        "expected_keywords": ["समर्थन", "बंद"],
        "should_have_answer": True,
    },
    {
        "name": "Country C - Return policy (fr_CA)",
        "question": "Quelle est votre politique de retour?",
        "country": "C",
        "language": "fr_CA",
        "expected_content_ids": ["c_faq_returns_fr"],
        "expected_keywords": ["14 jours"],
        "should_have_answer": True,
    },
    {
        "name": "Country D - Delivery time (en)",
        "question": "When will my order be delivered?",
        "country": "D",
        "language": "en",
        "expected_content_ids": ["d_faq_delivery_en"],
        "expected_keywords": ["1 business day"],
        "should_have_answer": True,
    },
    {
        "name": "Multi-tenant isolation: A returns != B returns",
        "question": "How many days do I have to return an item?",
        "country": "A",
        "language": "en",
        "must_not_contain": ["7 days", "30 days"],
        "expected_keywords": ["48 hours"],
        "should_have_answer": True,
    },
    {
        "name": "Language fallback: Spanish for Country A (should fallback to en/hi)",
        "question": "What payment methods do you accept?",
        "country": "A",
        "language": "es",
        "should_have_answer": True,
        "expect_fallback": True,
    },
    {
        "name": "Country C - Account closure (en)",
        "question": "How do I close my account?",
        "country": "C",
        "language": "en",
        "expected_content_ids": ["c_faq_account_en"],
        "expected_keywords": ["close", "account"],
        "should_have_answer": True,
    },
    {
        "name": "Invalid country handling",
        "question": "What is your return policy?",
        "country": "X",
        "language": "en",
        "should_have_answer": False,
    },
]


def run_test(case: dict, client: httpx.Client) -> dict:
    name = case["name"]
    try:
        resp = client.post(
            f"{API_BASE}/ask",
            json={
                "question": case["question"],
                "country": case["country"],
                "language": case["language"],
            },
            timeout=REQUEST_TIMEOUT,
        )

        if case["country"] == "X":
            if resp.status_code == 422:
                return {"name": name, "passed": True, "reason": "Correctly rejected invalid country (422)"}
            data = resp.json()
            if not data.get("citations"):
                return {"name": name, "passed": True, "reason": "No citations for invalid input"}
            return {"name": name, "passed": False, "reason": f"Should not have answered for invalid country"}

        if resp.status_code != 200:
            return {"name": name, "passed": False, "reason": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        trace = data.get("trace", {})
        failures = []

        if case.get("should_have_answer") and not answer.strip():
            failures.append("Empty answer")

        if case.get("expected_content_ids"):
            cited_ids = {c["content_id"] for c in citations}
            for expected_id in case["expected_content_ids"]:
                if expected_id not in cited_ids:
                    failures.append(f"Missing expected citation: {expected_id}")

        if case.get("expected_keywords"):
            for kw in case["expected_keywords"]:
                if kw.lower() not in answer.lower():
                    failures.append(f"Missing keyword in answer: '{kw}'")

        if case.get("expected_keywords_any_one_of"):
            lowered = answer.lower()
            if not any(kw.lower() in lowered for kw in case["expected_keywords_any_one_of"]):
                failures.append(
                    f"Answer missing any of: {case['expected_keywords_any_one_of']}"
                )

        if "could not be generated right now" in answer.lower():
            failures.append("LLM returned error message (rate limit or provider error)")

        if case.get("must_not_contain"):
            for bad_kw in case["must_not_contain"]:
                if bad_kw.lower() in answer.lower():
                    failures.append(f"LEAKAGE: answer contains '{bad_kw}' (should not)")

        if case.get("expect_fallback"):
            if not trace.get("fallback_used"):
                failures.append("Expected fallback but none was used")

        if failures:
            return {"name": name, "passed": False, "reason": "; ".join(failures)}
        return {
            "name": name,
            "passed": True,
            "reason": f"OK ({len(citations)} citations, {trace.get('latency_ms', '?')}ms)",
        }

    except Exception as e:
        return {"name": name, "passed": False, "reason": f"Exception: {e}"}


def main():
    print("=" * 70)
    print("EVALUATION HARNESS — Multi-Country Content Q&A")
    print("=" * 70)

    client = httpx.Client()

    print(f"Sleep between requests: {EVAL_SLEEP_SECONDS}s (set EVAL_SLEEP_SECONDS to change)")

    try:
        health = client.get(f"{API_BASE}/health", timeout=5.0)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy: {health.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach server at {API_BASE}: {e}")
        print("Make sure the server is running: uv run uvicorn app.api.main:app")
        sys.exit(1)

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        if i > 1:
            time.sleep(EVAL_SLEEP_SECONDS)
        print(f"\n[{i}/{len(TEST_CASES)}] {case['name']}...")
        result = run_test(case, client)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status}: {result['reason']}")

    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"RESULTS: {passed}/{total} passed")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")

    if passed < total:
        print(f"\n{total - passed} test(s) failed.")
    else:
        print("\nAll tests passed!")

    client.close()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
