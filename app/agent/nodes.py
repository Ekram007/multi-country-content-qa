import logging
from difflib import SequenceMatcher

from app.agent.state import AgentState
from app.ingestion.retriever import retrieve
from app.agent.llm import get_llm

logger = logging.getLogger(__name__)

VALID_COUNTRIES = {"A", "B", "C", "D"}
COUNTRY_LANGUAGES = {
    "A": ["en", "hi"],
    "B": ["en", "es"],
    "C": ["en", "fr_CA"],
    "D": ["en"],
}


def validate_input(state: AgentState) -> AgentState:
    country = state["country"].upper()
    language = state["language"]

    if country not in VALID_COUNTRIES:
        return {
            "error": f"Invalid country '{country}'. Must be one of: {', '.join(sorted(VALID_COUNTRIES))}",
            "route": "error",
        }

    return {"error": None, "route": "retrieve"}


def retrieve_chunks(state: AgentState) -> AgentState:
    chunks = retrieve(
        query=state["question"],
        country=state["country"],
        language=state["language"],
        top_k=5,
    )

    if chunks:
        return {
            "retrieved_chunks": chunks,
            "fallback_used": False,
            "fallback_language": None,
            "route": "synthesize",
        }

    return {
        "retrieved_chunks": [],
        "route": "fallback",
    }


def fallback_retrieve(state: AgentState) -> AgentState:
    """Try retrieving in other languages available for the same country."""
    country = state["country"]
    requested_lang = state["language"]
    available_langs = COUNTRY_LANGUAGES.get(country, [])

    for lang in available_langs:
        if lang == requested_lang:
            continue
        chunks = retrieve(
            query=state["question"],
            country=country,
            language=lang,
            top_k=5,
        )
        if chunks:
            logger.info(
                f"Fallback: found {len(chunks)} chunks in {lang} "
                f"for country {country} (requested {requested_lang})"
            )
            return {
                "retrieved_chunks": chunks,
                "fallback_used": True,
                "fallback_language": lang,
                "route": "synthesize",
            }

    return {
        "retrieved_chunks": [],
        "fallback_used": False,
        "fallback_language": None,
        "route": "no_answer",
    }


def synthesize_answer(state: AgentState) -> AgentState:
    chunks = state["retrieved_chunks"]
    question = state["question"]
    language = state["language"]
    fallback_used = state.get("fallback_used", False)
    fallback_language = state.get("fallback_language")

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] (ID: {chunk['content_id']}, Type: {chunk['type']}, "
            f"Title: {chunk['title']})\n{chunk['body']}"
        )
    context = "\n\n".join(context_parts)

    translation_instruction = ""
    if fallback_used and fallback_language:
        translation_instruction = (
            f"\nIMPORTANT: The source content is in '{fallback_language}' but the user "
            f"asked in '{language}'. Translate your answer to '{language}'. "
            f"Keep the citation references [1], [2], etc. as-is."
        )

    prompt = f"""You are a customer support assistant. Answer the question using ONLY the provided content.
Do NOT use any outside knowledge. If the content does not contain the answer, say so.

Rules:
- Ground every claim in the provided content
- Use inline citations like [1], [2] to reference the source documents
- Be concise and direct
- Answer in the language: {language}{translation_instruction}

Content:
{context}

Question: {question}

Answer:"""

    llm = get_llm()
    try:
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer_text, "llm_failed": False}
    except Exception as e:
        logger.exception("LLM synthesis failed: %s", e)
        return {
            "answer": (
                "The answer could not be generated right now due to a temporary "
                "provider limit or error. Please wait a minute and try again."
            ),
            "llm_failed": True,
        }


def extract_citations(state: AgentState) -> AgentState:
    if state.get("llm_failed"):
        return {"citations": [], "route": "done"}

    answer = state["answer"]
    chunks = state["retrieved_chunks"]

    citations = []
    for i, chunk in enumerate(chunks, 1):
        ref_marker = f"[{i}]"
        if ref_marker in answer:
            excerpt = _extract_relevant_excerpt(answer, chunk["body"], ref_marker)
            score = _compute_match_score(excerpt, chunk["body"])
            citations.append({
                "content_id": chunk["content_id"],
                "type": chunk["type"],
                "title": chunk["title"],
                "excerpt": excerpt,
                "match_score": round(score, 2),
            })

    return {"citations": citations, "route": "done"}


def no_answer_response(state: AgentState) -> AgentState:
    language = state["language"]
    country = state["country"]
    return {
        "answer": (
            f"No content found for country '{country}' in language '{language}'. "
            f"Please try a different language or contact support."
        ),
        "citations": [],
        "route": "done",
    }


def _extract_relevant_excerpt(answer: str, body: str, ref_marker: str) -> str:
    """Extract a short excerpt from the source body that best matches the claim near the citation."""
    ref_pos = answer.find(ref_marker)
    if ref_pos == -1:
        return body[:200]

    start = max(0, ref_pos - 150)
    end = min(len(answer), ref_pos + 50)
    claim_text = answer[start:end].replace(ref_marker, "").strip()

    sentences = [s.strip() for s in body.replace(". ", ".\n").split("\n") if s.strip()]
    if not sentences:
        return body[:200]

    best_sentence = max(sentences, key=lambda s: SequenceMatcher(None, claim_text.lower(), s.lower()).ratio())
    return best_sentence[:300]


def _compute_match_score(excerpt: str, body: str) -> float:
    if not excerpt or not body:
        return 0.0
    return SequenceMatcher(None, excerpt.lower(), body.lower()).ratio()
