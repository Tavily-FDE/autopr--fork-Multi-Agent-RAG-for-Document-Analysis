"""
Utility functions for topic relevance checking and routing logic.
"""


def check_topic_relevance(question: str) -> bool:
    """
    Quick heuristic-based check: Is the question related to the knowledge base topic?
    
    Args:
        question: User's question
        
    Returns:
        bool: True if on-topic, False if off-topic
    """
    try:
        # Off-topic keywords
        off_topic_keywords = [
            "weather", "hava durumu", "joke", "şaka", "funny", "komik",
            "cook recipe", "tarif", "sports score", "spor", "movie", "film",
            "song", "şarkı", "chat", "sohbet", "personal advice", "tavsiye"
        ]
        
        question_lower = question.lower()
        
        # If any off-topic keyword found, it's off-topic
        for keyword in off_topic_keywords:
            if keyword in question_lower:
                return False
        
        # Technical keywords - if found, question is on-topic
        technical_keywords = [
            "how", "nasıl", "what", "ne", "why", "neden", "api", "code",
            "kod", "function", "fonksiyon", "library", "kütüphane", "framework",
            "tutorial", "documentation", "dokümantasyon", "error", "hata",
            "implement", "use", "configure", "fit", "train", "model", "neural",
            "algorithm", "data", "veri", "python", "javascript", "java",
            "database", "veri tabanı", "sql", "api", "rest"
        ]
        
        for keyword in technical_keywords:
            if keyword in question_lower:
                return True
        
        # Default: allow if not obviously off-topic
        return True
        
    except Exception as e:
        print(f"   ⚠️ [Topic Check Error]: {e}. Allowing question.")
        return True


def should_search_web(
    is_topic_related: bool, 
    found_in_db: bool, 
    rag_context: str
) -> str:
    """
    Determine routing decision based on topic relevance and DB findings.
    
    Args:
        is_topic_related: Is question on-topic?
        found_in_db: Was relevant content found in database?
        rag_context: Retrieved context from database
        
    Returns:
        str: Routing decision ("end_workflow", "go_to_writer", or "go_to_web")
    """
    # If question is off-topic, stop immediately
    if not is_topic_related:
        print("🛑 [Router] Question is off-topic -> Stopping workflow.")
        return "end_workflow"
    
    # If content found in database, use it (don't search web)
    if found_in_db and rag_context and len(rag_context.strip()) >= 100:
        print("✅ [Router] Relevant content found in database -> Using local data.")
        return "go_to_writer"
    
    # If topic related but not in database, search web
    if is_topic_related and not found_in_db:
        print("🔍 [Router] Question is on-topic but not in database -> Searching web.")
        return "go_to_web"
    
    # Fallback to writer if any content exists
    if rag_context and len(rag_context.strip()) >= 100:
        print("🔍 [Router] Using available content.")
        return "go_to_writer"
    
    print("🔍 [Router] No content available -> Triggering web search.")
    return "go_to_web"
