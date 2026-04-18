"""Tools and workflow functions for the content Q&A agent."""

import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.content_qa.schema import AgentState, ContentChunk, Citation
from src.ingestion.retriever import retrieve
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

# Configuration constants
VALID_COUNTRIES = {"A", "B", "C", "D"}
COUNTRY_LANGUAGES: Dict[str, List[str]] = {
    "A": ["en", "hi"],
    "B": ["en", "es"], 
    "C": ["en", "fr_CA"],
    "D": ["en"],
}

# Default retrieval parameters
DEFAULT_TOP_K = 5
MAX_EXCERPT_LENGTH = 300
CLAIM_CONTEXT_WINDOW = 150


def _load_prompt(prompt_name: str) -> str:
    """Load prompt template from text file."""
    prompt_file = Path(__file__).parent.parent.parent / "prompts" / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    return prompt_file.read_text(encoding='utf-8').strip()


def validate_input(state: AgentState) -> AgentState:
    """Validate the input parameters for the Q&A request.
    
    Args:
        state: Current agent state containing question, country, language
        
    Returns:
        Updated state with error info and routing decision
    """
    country = state["country"].upper()
    language = state["language"]
    question = state.get("question", "").strip()

    # Validate required fields
    if not question:
        return {
            "error": "Question cannot be empty",
            "route": "error",
        }

    # Validate country
    if country not in VALID_COUNTRIES:
        available = ", ".join(sorted(VALID_COUNTRIES))
        return {
            "error": f"Invalid country '{country}'. Must be one of: {available}",
            "route": "error",
        }

    # Validate language format (basic check)
    if not language or len(language) < 2:
        return {
            "error": f"Invalid language format '{language}'",
            "route": "error",
        }

    logger.debug(f"Input validated: country={country}, language={language}")
    return {"error": None, "route": "retrieve"}


def retrieve_chunks(state: AgentState) -> AgentState:
    """Retrieve content chunks for the given query and filters.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved chunks and routing decision
    """
    try:
        chunks = retrieve(
            query=state["question"],
            country=state["country"],
            language=state["language"],
            top_k=DEFAULT_TOP_K,
        )

        if chunks:
            logger.info(
                f"Retrieved {len(chunks)} chunks for country={state['country']}, "
                f"language={state['language']}"
            )
            return {
                "retrieved_chunks": chunks,
                "fallback_used": False,
                "fallback_language": None,
                "route": "synthesize",
            }

        logger.info(
            f"No chunks found for country={state['country']}, "
            f"language={state['language']}, trying fallback"
        )
        return {
            "retrieved_chunks": [],
            "route": "fallback",
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "retrieved_chunks": [],
            "error": f"Content retrieval failed: {str(e)}",
            "route": "no_answer",
        }


def fallback_retrieve(state: AgentState) -> AgentState:
    """Attempt fallback retrieval using other available languages for the country.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with fallback retrieval results
    """
    country = state["country"]
    requested_lang = state["language"]
    available_langs = COUNTRY_LANGUAGES.get(country, [])

    logger.debug(f"Attempting fallback for country {country}, available languages: {available_langs}")

    for lang in available_langs:
        if lang == requested_lang:
            continue
            
        try:
            chunks = retrieve(
                query=state["question"],
                country=country,
                language=lang,
                top_k=DEFAULT_TOP_K,
            )
            
            if chunks:
                logger.info(
                    f"Fallback successful: found {len(chunks)} chunks in '{lang}' "
                    f"for country {country} (originally requested '{requested_lang}')"
                )
                return {
                    "retrieved_chunks": chunks,
                    "fallback_used": True,
                    "fallback_language": lang,
                    "route": "synthesize",
                }
                
        except Exception as e:
            logger.warning(f"Fallback retrieval failed for language {lang}: {e}")
            continue

    logger.info(f"Fallback failed: no content found in any language for country {country}")
    return {
        "retrieved_chunks": [],
        "fallback_used": False,
        "fallback_language": None,
        "route": "no_answer",
    }


def synthesize_answer(state: AgentState) -> AgentState:
    """Synthesize an answer using the LLM based on retrieved content.
    
    Args:
        state: Current agent state with retrieved chunks
        
    Returns:
        Updated state with synthesized answer
    """
    chunks = state["retrieved_chunks"]
    question = state["question"]
    language = state["language"]
    fallback_used = state.get("fallback_used", False)
    fallback_language = state.get("fallback_language")

    if not chunks:
        logger.warning("No chunks available for synthesis")
        return {
            "answer": "No relevant content found to answer the question.",
            "llm_failed": False,
        }

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] (ID: {chunk['content_id']}, Type: {chunk['type']}, "
            f"Title: {chunk['title']})\n{chunk['body']}"
        )
    context = "\n\n".join(context_parts)

    # Build synthesis prompt from text file
    try:
        if fallback_used and fallback_language:
            prompt_template = _load_prompt("synthesis_fallback")
            prompt = prompt_template.format(
                question=question,
                context=context,
                content_language=language,
                fallback_language=fallback_language
            )
        else:
            prompt_template = _load_prompt("synthesis")
            prompt = prompt_template.format(
                question=question,
                context=context
            )
    except Exception as e:
        logger.error(f"Failed to load prompt: {e}")
        return {
            "answer": "Unable to load prompt template.",
            "llm_failed": True,
        }

    # Call LLM
    llm = get_llm()
    try:
        logger.debug("Calling LLM for answer synthesis")
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        logger.info("LLM synthesis completed successfully")
        return {"answer": answer_text, "llm_failed": False}
        
    except Exception as e:
        logger.exception(f"LLM synthesis failed: {e}")
        return {
            "answer": (
                "The answer could not be generated right now due to a temporary "
                "service issue. Please wait a moment and try again."
            ),
            "llm_failed": True,
        }


def extract_citations(state: AgentState) -> AgentState:
    """Extract citations from the synthesized answer.
    
    Args:
        state: Current agent state with answer and chunks
        
    Returns:
        Updated state with extracted citations
    """
    if state.get("llm_failed"):
        logger.debug("Skipping citation extraction due to LLM failure")
        return {"citations": [], "route": "done"}

    answer = state["answer"]
    chunks = state["retrieved_chunks"]

    if not answer or not chunks:
        logger.debug("No answer or chunks available for citation extraction")
        return {"citations": [], "route": "done"}

    citations = []
    for i, chunk in enumerate(chunks, 1):
        ref_marker = f"[{i}]"
        if ref_marker in answer:
            excerpt = _extract_relevant_excerpt(answer, chunk["body"], ref_marker)
            score = _compute_match_score(excerpt, chunk["body"])
            citations.append({
                "content_id": chunk["content_id"],
                "type": chunk["type"],
                "excerpt": excerpt,
                "match_score": round(score, 2),
            })

    logger.debug(f"Extracted {len(citations)} citations")
    return {"citations": citations, "route": "done"}


def no_answer_response(state: AgentState) -> AgentState:
    """Generate a fallback response when no answer can be provided.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with no-answer response
    """
    language = state["language"]
    country = state["country"]
    error = state.get("error")

    if error:
        answer = f"Unable to process request: {error}"
    else:
        answer = (
            f"No relevant content found for country '{country}' in language '{language}'. "
            f"Please try a different language or contact customer support."
        )

    logger.info(f"Returning no-answer response for country={country}, language={language}")
    return {
        "answer": answer,
        "citations": [],
        "route": "done",
    }


# Helper functions

def _extract_relevant_excerpt(answer: str, body: str, ref_marker: str) -> str:
    """Extract a relevant excerpt from the source body.
    
    Args:
        answer: The synthesized answer containing citations
        body: The source content body
        ref_marker: The citation reference marker (e.g., "[1]")
        
    Returns:
        Relevant excerpt from the source body
    """
    ref_pos = answer.find(ref_marker)
    if ref_pos == -1:
        return body[:200] + "..." if len(body) > 200 else body

    # Extract claim text around the citation
    start = max(0, ref_pos - CLAIM_CONTEXT_WINDOW)
    end = min(len(answer), ref_pos + 50)
    claim_text = answer[start:end].replace(ref_marker, "").strip()

    # Split body into sentences for matching
    sentences = [
        s.strip() 
        for s in body.replace(". ", ".\n").split("\n") 
        if s.strip()
    ]
    
    if not sentences:
        return body[:200] + "..." if len(body) > 200 else body

    # Find best matching sentence
    best_sentence = max(
        sentences, 
        key=lambda s: SequenceMatcher(None, claim_text.lower(), s.lower()).ratio()
    )
    
    # Truncate if too long
    if len(best_sentence) > MAX_EXCERPT_LENGTH:
        return best_sentence[:MAX_EXCERPT_LENGTH] + "..."
    
    return best_sentence


def _compute_match_score(excerpt: str, body: str) -> float:
    """Compute similarity score between excerpt and source body.
    
    Args:
        excerpt: Extracted excerpt
        body: Full source body
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not excerpt or not body:
        return 0.0
    
    return SequenceMatcher(
        None, 
        excerpt.lower().strip(), 
        body.lower().strip()
    ).ratio()