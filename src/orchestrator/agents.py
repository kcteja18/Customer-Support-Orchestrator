from typing import Optional

try:
    from langchain.chains import RetrievalQA
except Exception:
    RetrievalQA = None
try:
    from langchain.llms import HuggingFaceHub
except Exception:
    HuggingFaceHub = None
import os


class MockLLM:
    """Enhanced LLM that generates query-specific answers from retrieved documents.

    Uses keyword matching, document analysis, and relevance detection to provide
    appropriate responses. Detects out-of-scope queries and prompts users accordingly.
    """

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
        
        # Define customer support domain keywords
        self.support_keywords = [
            # Account related
            'account', 'login', 'password', 'register', 'signup', 'sign up', 'username',
            'email', 'profile', 'settings', 'authentication', '2fa', 'security',
            
            # Billing related
            'billing', 'payment', 'subscription', 'plan', 'invoice', 'refund', 'charge',
            'credit card', 'upgrade', 'downgrade', 'price', 'cost', 'cancel', 'renew',
            
            # Technical support
            'error', 'bug', 'issue', 'problem', 'not working', 'broken', 'crash', 'slow',
            'troubleshoot', 'fix', 'help', 'support', 'technical', 'connection', 'browser',
            
            # Features and product
            'feature', 'how to', 'tutorial', 'guide', 'documentation', 'api', 'integration',
            'mobile app', 'chatbot', 'ticket', 'automation', 'report', 'export',
            
            # Data and privacy
            'data', 'privacy', 'gdpr', 'ccpa', 'security', 'encryption', 'delete',
            'download', 'export data', 'compliance', 'breach',
            
            # Contact and business
            'contact', 'phone', 'email us', 'chat', 'hours', 'business hours', 'support team',
            'address', 'office', 'response time'
        ]
        
    def is_relevant_query(self, query: str) -> bool:
        """Check if query is relevant to customer support domain.
        
        Returns True if query appears to be support-related, False otherwise.
        """
        query_lower = query.lower()
        
        # Check if query contains any support-related keywords
        keyword_matches = sum(1 for keyword in self.support_keywords if keyword in query_lower)
        
        # Check for common question words that might indicate a support query
        question_patterns = [
            'how do i', 'how can i', 'how to', 'what is', 'where is', 'when',
            'why', 'can i', 'do you', 'does', 'is there', 'are there'
        ]
        has_question_pattern = any(pattern in query_lower for pattern in question_patterns)
        
        # Out-of-scope indicators (queries clearly not about customer support)
        out_of_scope_indicators = [
            'weather', 'joke', 'story', 'recipe', 'restaurant', 'movie', 'music',
            'sports', 'news', 'stock', 'celebrity', 'game', 'what time is it',
            'temperature', 'forecast', 'capital of', 'who is', 'when was',
            'calculate', 'math', 'translate', 'definition of'
        ]
        has_out_of_scope = any(indicator in query_lower for indicator in out_of_scope_indicators)
        
        # Query is relevant if:
        # - Has at least 2 keyword matches, OR
        # - Has 1 keyword match and a question pattern, AND
        # - Doesn't have clear out-of-scope indicators
        is_relevant = ((keyword_matches >= 2) or (keyword_matches >= 1 and has_question_pattern)) and not has_out_of_scope
        
        return is_relevant

    def generate_answer(self, query: str, docs: list) -> str:
        # First, check if query is relevant to customer support
        if not self.is_relevant_query(query):
            return ("I'm a customer support assistant and can only help with questions related to our service. "
                    "This query appears to be outside my area of expertise. "
                    "Please ask questions about:\n"
                    "- Account management and login issues\n"
                    "- Billing, payments, and subscriptions\n"
                    "- Technical problems and troubleshooting\n"
                    "- Product features and how to use them\n"
                    "- Data privacy and security\n"
                    "- Contacting our support team\n\n"
                    "How can I help you with your account or our services today?")
        
        if not docs:
            return "I don't have information about that in my knowledge base. Please contact our support team for assistance."
        
        query_lower = query.lower()
        
        # Find most relevant document based on query keywords
        best_doc = docs[0]
        best_score = 0
        
        for doc in docs[:3]:
            content_lower = doc.page_content.lower()
            keywords = [w for w in query_lower.split() if len(w) > 3]
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_doc = doc
        
        # Get the content
        content = best_doc.page_content.strip()
        
        # Clean up the content - remove excessive newlines but keep structure
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n\n'.join(lines)
        
        # Limit length but try to keep complete sentences
        if len(content) > 600:
            content = content[:600]
            # Try to end at a sentence
            last_period = content.rfind('.')
            if last_period > 300:
                content = content[:last_period + 1]
        
        return content


class SupportOrchestrator:
    """A small orchestrator that uses a retriever + LLM to answer and optionally escalate.

    When `mode=='local'` the orchestrator will use the `MockLLM`. When `mode=='hf'`
    it will use a Hugging Face Hub model via LangChain's `HuggingFaceHub`.
    """

    def __init__(self, retriever, llm: Optional[object] = None, mode: str = "local", escalate_phrases=None):
        self.retriever = retriever
        self.mode = mode
        self.llm = llm
        if self.mode == "local":
            self.llm = MockLLM()
        elif self.mode == "hf":
            if HuggingFaceHub is not None:
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
                model = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-small")
                if not hf_token:
                    raise RuntimeError("Hugging Face API token not set. Set HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_API_KEY in env")
                # LangChain's HuggingFaceHub expects `repo_id` and `huggingfacehub_api_token`
                self.llm = HuggingFaceHub(repo_id=model, huggingfacehub_api_token=hf_token)
            else:
                raise RuntimeError("HuggingFaceHub LLM not available in this environment. Install compatible langchain or run in local mode.")

        self.escalate_phrases = escalate_phrases or ["i don't know", "i am not sure", "i'm not sure"]

    def answer(self, query: str) -> str:
        """Generate an answer to the query.
        
        Returns just the answer text string for compatibility with backend.
        """
        if self.mode == "local":
            docs = self.retriever.get_relevant_documents(query)
            answer = self.llm.generate_answer(query, docs)
        else:
            if RetrievalQA is None:
                raise RuntimeError("RetrievalQA not available — ensure langchain and LLM are installed or use --mode local")
            qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)
            answer = qa.run(query)
        
        return answer
    
    def classify_confidence(self, query: str, answer: str) -> float:
        """Estimate confidence in the answer based on heuristics.
        
        Returns a float between 0 and 1.
        """
        # Simple heuristics for confidence scoring
        confidence = 0.5  # baseline
        
        answer_lower = answer.lower()
        
        # Reduce confidence if answer contains uncertainty phrases
        uncertainty_phrases = ["i don't know", "i'm not sure", "unclear", "not able to find"]
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Increase confidence if answer is detailed (longer)
        if len(answer) > 200:
            confidence += 0.2
        
        # Increase confidence if answer references documents/sources
        if "based on" in answer_lower or "according to" in answer_lower:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def should_escalate(self, query: str, answer: str) -> bool:
        """Determine if query should be escalated to human agent.
        
        Returns True if escalation is recommended.
        """
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Check for escalation phrases in answer
        if any(phrase in answer_lower for phrase in self.escalate_phrases):
            return True
        
        # Check for urgent keywords in query
        urgent_keywords = ["manager", "supervisor", "urgent", "complaint", "legal", "lawsuit"]
        if any(keyword in query_lower for keyword in urgent_keywords):
            return True
        
        # Low confidence answers should escalate
        confidence = self.classify_confidence(query, answer)
        if confidence < 0.4:
            return True
        
        return False


class EscalationAgent:
    """Simple escalation agent to format a human-handoff message."""

    @staticmethod
    def create_ticket(query: str, context: str) -> dict:
        # In a real system this would call a ticketing API (Zendesk, Freshdesk, etc.)
        return {"ticket_subject": f"Escalation: {query[:120]}", "context": context}
