"""
Content search tools for multi-country Q&A agent.

Simple @tool decorated functions following Agent-Service-Toolkit pattern.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.tools import tool
from src.ingestion.retriever import retrieve

logger = logging.getLogger(__name__)

# Configuration constants
VALID_COUNTRIES = {"A", "B", "C", "D"}
COUNTRY_LANGUAGES: Dict[str, List[str]] = {
    "A": ["en", "hi"],
    "B": ["en", "es"], 
    "C": ["en", "fr_CA"],
    "D": ["en"],
}


@tool
def search_content(query: str, country: str, language: str = "en", top_k: int = 5) -> str:
    """
    Search for relevant content in a specific country and language.
    
    Use this tool when the user asks questions about policies, procedures, or information
    that varies by country. The tool will search for content in the requested language,
    and fallback to English if no content is found.
    
    Args:
        query: The user's question or search terms
        country: Country code (A, B, C, or D)  
        language: Preferred content language (e.g., "en", "es", "hi", "fr_CA")
        top_k: Number of results to retrieve (default: 5)
        
    Returns:
        Formatted content with source information for answering the question
    """
    logger.info(f"Content search: country={country}, language={language}, query='{query[:50]}...'")
    
    # Validate country
    country = country.upper()
    if country not in VALID_COUNTRIES:
        available = ", ".join(sorted(VALID_COUNTRIES))
        return f"Error: Invalid country '{country}'. Available countries: {available}"
    
    available_languages = COUNTRY_LANGUAGES[country]
    search_languages = []
    
    # Build language search order
    if language in available_languages:
        search_languages.append(language)
    if language != "en" and "en" in available_languages:
        search_languages.append("en")  # Fallback to English
    
    if not search_languages:
        return f"Error: No supported languages found for country {country}"
    
    # Try each language until we find results
    for search_lang in search_languages:
        try:
            logger.debug(f"Searching in language: {search_lang}")
            results = retrieve(
                query=query,
                country=country,
                language=search_lang,
                top_k=top_k
            )
            
            if results:
                # Format results for LLM context
                content_parts = []
                content_parts.append(f"Content found in {search_lang} for country {country}:")
                content_parts.append(f"Search query: {query}")
                content_parts.append("")
                
                for i, doc in enumerate(results, 1):
                    content_parts.append(
                        f"[{i}] {doc['type']} - {doc['title']}\n"
                        f"Content ID: {doc['content_id']}\n"
                        f"Language: {doc['language']}\n"
                        f"Content: {doc['body']}\n"
                    )
                
                logger.info(f"Found {len(results)} documents in {search_lang}")
                return "\n".join(content_parts)
                
        except Exception as e:
            logger.error(f"Search failed for {search_lang}: {e}")
            continue
    
    # No results found in any language
    logger.warning(f"No content found for country {country} in any available language")
    return f"No relevant content found for country {country}. Please try rephrasing your question."


@tool  
def get_supported_countries() -> str:
    """
    Get list of supported countries and their available languages.
    
    Use this tool when the user asks about which countries or languages are supported.
    
    Returns:
        Information about supported countries and languages
    """
    info_parts = ["Supported countries and languages:"]
    
    for country, languages in COUNTRY_LANGUAGES.items():
        lang_list = ", ".join(languages)
        info_parts.append(f"• Country {country}: {lang_list}")
    
    info_parts.append("")
    info_parts.append("Country codes: A, B, C, D")
    info_parts.append("Common languages: en (English), es (Spanish), hi (Hindi), fr_CA (French Canadian)")
    
    return "\n".join(info_parts)


# Export tools list for agent binding
content_qa_tools = [search_content, get_supported_countries]