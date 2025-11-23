"""Conversation memory management for multi-turn conversations."""
from typing import List, Dict, Optional
from datetime import datetime
import json


class ConversationMemory:
    """Manages conversation history and context for multi-turn interactions."""
    
    def __init__(self, max_messages: int = 10, session_id: Optional[str] = None):
        """Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
            session_id: Unique identifier for this conversation session
        """
        self.messages: List[Dict] = []
        self.max_messages = max_messages
        self.session_id = session_id or self._generate_session_id()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_messages': 0
        }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (confidence, documents, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        self.metadata['total_messages'] += 1
        
        # Keep only recent messages to avoid memory issues
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self, num_messages: int = 5) -> str:
        """Get conversation context as formatted string.
        
        Args:
            num_messages: Number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        recent_messages = self.messages[-num_messages:]
        
        context_parts = []
        for msg in recent_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_recent_topics(self) -> List[str]:
        """Extract recent topics from conversation.
        
        Returns:
            List of topics discussed (from metadata)
        """
        topics = set()
        for msg in self.messages[-5:]:
            if 'category' in msg.get('metadata', {}):
                topics.add(msg['metadata']['category'])
        return list(topics)
    
    def get_history(self) -> List[Dict]:
        """Get full conversation history.
        
        Returns:
            List of all messages
        """
        return self.messages.copy()
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.metadata['total_messages'] = 0
    
    def to_dict(self) -> Dict:
        """Export conversation to dictionary.
        
        Returns:
            Dictionary representation of conversation
        """
        return {
            'session_id': self.session_id,
            'messages': self.messages,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Export conversation to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMemory':
        """Create ConversationMemory from dictionary.
        
        Args:
            data: Dictionary with session data
            
        Returns:
            ConversationMemory instance
        """
        memory = cls(session_id=data.get('session_id'))
        memory.messages = data.get('messages', [])
        memory.metadata = data.get('metadata', {})
        return memory
    
    def has_context(self) -> bool:
        """Check if conversation has any history.
        
        Returns:
            True if there are messages in history
        """
        return len(self.messages) > 0
    
    def get_last_user_query(self) -> Optional[str]:
        """Get the last query from user.
        
        Returns:
            Last user message content or None
        """
        for msg in reversed(self.messages):
            if msg['role'] == 'user':
                return msg['content']
        return None
    
    def is_follow_up_question(self, query: str) -> bool:
        """Detect if current query is a follow-up.
        
        Args:
            query: Current query text
            
        Returns:
            True if query appears to be a follow-up
        """
        query_lower = query.lower()
        
        # Follow-up indicators
        follow_up_phrases = [
            'what about', 'how about', 'and', 'also', 'too',
            'what if', 'can i also', 'do you also', 'is there',
            'another question', 'one more', 'additionally'
        ]
        
        # Short queries often reference previous context
        if len(query.split()) <= 3:
            return True
        
        # Check for follow-up phrases
        for phrase in follow_up_phrases:
            if phrase in query_lower:
                return True
        
        # Check if query references recent topics
        recent_topics = self.get_recent_topics()
        for topic in recent_topics:
            if topic.lower() in query_lower:
                return True
        
        return False
