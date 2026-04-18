"""Interactive CLI for the Multi-Country Content Q&A Agent.

This script provides a command-line interface to interact with the agent directly.
Agent-Service-Toolkit style runner.
"""

import json
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

from src.agents import get_content_qa_agent
from src.core.settings import settings
from src.schema.models import AskRequest

# Load environment variables
load_dotenv()


def print_banner():
    """Print application banner."""
    print("🤖 Multi-Country Content Q&A Agent")
    print("Interactive CLI Mode")
    print(f"🔍 Embedding: {settings.embedding_model}")
    print(f"🧠 LLM: {settings.llm_provider} ({settings.llm_model})")
    print("=" * 50)


def print_help():
    """Print help information."""
    print("\nCommands:")
    print("  ask <question> <country> <language>  - Ask a question")
    print("  health                              - Check agent health") 
    print("  help                                - Show this help")
    print("  exit                                - Exit the CLI")
    print("\nExample:")
    print('  ask "What is your return policy?" A en')
    print('  ask "¿Cuál es su política de devoluciones?" B es')


def ask_question(question: str, country: str, language: str) -> None:
    """Ask the agent a question and display the response."""
    try:
        # Get agent instance
        agent = get_content_qa_agent()
        
        # Create request
        request = AskRequest(
            question=question,
            country=country,
            language=language
        )
        
        print(f"\n🔍 Processing question for country {country} in {language}...")
        print("-" * 50)
        
        # Get response from agent
        response = agent.ask(request)
        
        # Display response
        print(f"📝 Answer: {response.answer}")
        print(f"🌐 Language used: {response.language_used}")
        print(f"⏱️  Latency: {response.trace.latency_ms}ms")
        print(f"📊 Retrieved: {response.trace.retrieval_count} chunks")
        print(f"🤖 Model: {response.trace.model}")
        
        if response.citations:
            print(f"\n📚 Citations ({len(response.citations)}):")
            for i, citation in enumerate(response.citations, 1):
                print(f"  [{i}] {citation.content_id} ({citation.type})")
                print(f"      Score: {citation.match_score:.2f}")
                print(f"      \"{citation.excerpt[:100]}{'...' if len(citation.excerpt) > 100 else ''}\"")
        else:
            print("\n📚 No citations found")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def check_health() -> None:
    """Check agent health and display status."""
    try:
        agent = get_content_qa_agent()
        health = agent.health_check()
        
        print(f"\n🏥 Agent Health Check:")
        print(f"  Status: {'✅' if health['status'] == 'healthy' else '❌'} {health['status']}")
        print(f"  Agent: {health['agent']}")
        print(f"  Version: {health['version']}")
        
        if 'error' in health:
            print(f"  Error: {health['error']}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")


def interactive_mode():
    """Run interactive CLI mode."""
    print_banner()
    print_help()
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            parts = user_input.split()
            command = parts[0].lower()
            
            if command == "exit":
                print("👋 Goodbye!")
                break
                
            elif command == "help":
                print_help()
                
            elif command == "health":
                check_health()
                
            elif command == "ask":
                if len(parts) < 4:
                    print("❌ Usage: ask <question> <country> <language>")
                    print('   Example: ask "What is your return policy?" A en')
                    continue
                    
                # Parse arguments - question might be multiple words
                country = parts[-2]
                language = parts[-1]
                question = " ".join(parts[1:-2])
                
                # Remove quotes if present
                question = question.strip('"\'')
                
                ask_question(question, country, language)
                
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point."""
    # Set up logging (less verbose for CLI)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s"
    )
    
    # Check command line arguments for direct execution
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            check_health()
        elif sys.argv[1] == "ask" and len(sys.argv) >= 5:
            question = " ".join(sys.argv[2:-2])
            country = sys.argv[-2]
            language = sys.argv[-1]
            ask_question(question, country, language)
        else:
            print("Usage:")
            print("  python -m src.run_agent                     # Interactive mode")
            print("  python -m src.run_agent health              # Health check")
            print('  python -m src.run_agent ask "question" A en # Direct question')
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()