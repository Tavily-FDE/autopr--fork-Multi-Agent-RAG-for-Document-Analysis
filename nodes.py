"""
Workflow node implementations for the RAG Multi-Agent system.
"""
from typing import Any, Dict
from langchain_ollama import ChatOllama
from langchain_core.retrievers import BaseRetriever

from models import AgentState, GraderOutput, CriticOutput
from utils import check_topic_relevance


class WorkflowNodes:
    """Encapsulates all workflow node functions"""

    def __init__(
        self,
        config: dict,
        llm: ChatOllama,
        grader_llm: Any,
        critic_llm: Any,
        retriever: BaseRetriever,
        web_search_tool: Any
    ):
        self.config = config
        self.llm = llm
        self.grader_llm = grader_llm
        self.critic_llm = critic_llm
        self.retriever = retriever
        self.web_search_tool = web_search_tool

    def pdf_researcher_node(self, state: AgentState) -> Dict[str, Any]:
        """PDF Researcher: Check topic relevance and search database"""
        question = state["question"]
        print("📚 [PDF Researcher] Checking topic relevance and scanning documents...")
        
        # Step 1: Check if question is on-topic
        print("   🔎 Checking if question is on-topic...")
        is_topic_related = check_topic_relevance(question)
        
        if not is_topic_related:
            print("   ❌ Question is off-topic. Stopping workflow.")
            # Return response for off-topic question
            return {
                "rag_context": "",
                "is_relevant": False,
                "is_topic_related": False,
                "found_in_db": False,
                "report": self.config["prompts"]["irrelevant_question"]
            }
        
        print("   ✅ Question is on-topic. Searching database...")
        
        # Step 2: Search database if retriever available
        if not self.retriever:
            print("   ℹ️  No database available. Will search web.")
            return {
                "rag_context": "",
                "is_relevant": False,
                "is_topic_related": True,
                "found_in_db": False
            }
        
        docs = self.retriever.invoke(question)
        max_context_len = self.config["context_limits"]["grader_context"]
        raw_context = "\n\n".join([d.page_content for d in docs])[:max_context_len]
        
        if not raw_context.strip():
            print("   ℹ️  No documents found in database.")
            return {
                "rag_context": "",
                "is_relevant": False,
                "is_topic_related": True,
                "found_in_db": False
            }
        
        # Step 3: Check if found content is relevant
        grading_prompt = self.config["prompts"]["grader"].format(
            question=question,
            context=raw_context
        )
        
        try:
            result = self.grader_llm.invoke(grading_prompt)
            is_relevant = result.is_relevant
            is_topic_related = getattr(result, 'is_topic_related', True)
        except Exception as e:
            print(f"   ⚠️ [Grader Error]: LLM could not provide structured output. {e}")
            is_relevant = False
        
        if is_relevant:
            print("   ✅ Relevant content found in database.")
            return {
                "rag_context": raw_context,
                "is_relevant": True,
                "is_topic_related": is_topic_related,
                "found_in_db": True
            }
        else:
            print("   ⚠️ Content in database not relevant, but question is on-topic.")
            return {
                "rag_context": "",
                "is_relevant": False,
                "is_topic_related": is_topic_related,
                "found_in_db": False
            }

    def web_searcher_node(self, state: AgentState) -> Dict[str, Any]:
        """Web Searcher: Search the internet for information"""
        question = state["question"]
        print("🌐 [Web Researcher] Searching the internet...")
        try:
            web_results = self.web_search_tool.invoke(question)
        except Exception as e:
            web_results = f"Web search failed: {str(e)}"
        return {"web_context": web_results}

    def writer_node(self, state: AgentState) -> Dict[str, Any]:
        """Writer: Generate or revise report"""
        print("📝 [Writer] Creating/revising report...")
        question = state["question"]
        
        # Context window management
        max_context = self.config["context_limits"]["writer_context"]
        rag_context = state.get("rag_context", "")[:max_context]
        web_context = state.get("web_context", "")[:max_context]
        
        if len(state.get("rag_context", "")) > max_context:
            print(f"   ⚠️ [Writer] RAG content truncated to {max_context} characters.")
        if len(state.get("web_context", "")) > max_context:
            print(f"   ⚠️ [Writer] Web content truncated to {max_context} characters.")
        
        feedback = state.get("feedback", "")
        
        if feedback:
            instruction = self.config["prompts"]["writer_feedback"].format(feedback=feedback)
        else:
            instruction = self.config["prompts"]["writer_initial"]
        
        sections = self.config["prompts"]["writer_sections"]
        prompt = self.config["prompts"]["writer_template"].format(
            question=question,
            instruction=instruction,
            rag_context=rag_context,
            web_context=web_context,
            sections=sections
        )
        
        response = self.llm.invoke(prompt)
        return {"report": response.content}

    def critic_node(self, state: AgentState) -> Dict[str, Any]:
        """Critic: Evaluate report quality"""
        print("⚖️ [Critic] Evaluating report...")
        
        question = state["question"]
        report = state["report"]
        iterations = state.get("iterations", 0) + 1
        
        prompt = self.config["prompts"]["critic"].format(
            question=question,
            report=report
        )
        
        try:
            result = self.critic_llm.invoke(prompt)
            # Use quality_score for flexible evaluation
            quality_score = getattr(result, 'quality_score', 0.7)
            approval_threshold = self.config["critic"]["approval_threshold"]
            is_ok = quality_score >= approval_threshold
            feedback = result.feedback
        except Exception as e:
            print(f"   ⚠️ [Critic Error]: {e}")
            is_ok = False
            feedback = "System could not evaluate the report. Please revise."
        
        if is_ok:
            print(f"   ✅ Critic approved the report (quality score: {quality_score:.2f}).")
        else:
            print(f"   ❌ Critic rejected the report (quality score: {quality_score:.2f}). Revising...")
            if feedback:
                print(f"   💡 Feedback: {feedback[:150]}...")

        return {
            "is_satisfactory": is_ok,
            "feedback": feedback,
            "iterations": iterations
        }
