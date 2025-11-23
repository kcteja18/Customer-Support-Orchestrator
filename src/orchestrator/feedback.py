"""User feedback collection and analysis."""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class FeedbackCollector:
    """Collect and analyze user feedback for continuous improvement."""
    
    def __init__(self, feedback_file: str = "data/feedback.jsonl"):
        """Initialize feedback collector.
        
        Args:
            feedback_file: Path to feedback storage file (JSONL format)
        """
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file if it doesn't exist
        if not self.feedback_file.exists():
            self.feedback_file.touch()
    
    def record_feedback(self, 
                       query: str, 
                       answer: str, 
                       rating: int,
                       comment: str = "",
                       session_id: Optional[str] = None,
                       metadata: Optional[Dict] = None):
        """Record user feedback.
        
        Args:
            query: Original user query
            answer: System's answer
            rating: User rating (1-5 stars)
            comment: Optional text comment
            session_id: Session identifier
            metadata: Additional metadata (confidence, documents, etc.)
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'query': query,
            'answer': answer,
            'rating': rating,
            'comment': comment,
            'metadata': metadata or {}
        }
        
        # Append to JSONL file
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback) + '\n')
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback entries.
        
        Returns:
            List of all feedback entries
        """
        feedback_list = []
        
        if not self.feedback_file.exists() or self.feedback_file.stat().st_size == 0:
            return feedback_list
        
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    feedback_list.append(json.loads(line))
        
        return feedback_list
    
    def get_low_rated_queries(self, threshold: int = 3) -> List[Dict]:
        """Get queries with low ratings for improvement.
        
        Args:
            threshold: Rating threshold (queries <= threshold)
            
        Returns:
            List of low-rated feedback entries
        """
        all_feedback = self.get_all_feedback()
        return [
            fb for fb in all_feedback
            if fb['rating'] <= threshold
        ]
    
    def get_high_rated_queries(self, threshold: int = 4) -> List[Dict]:
        """Get queries with high ratings (working well).
        
        Args:
            threshold: Rating threshold (queries >= threshold)
            
        Returns:
            List of high-rated feedback entries
        """
        all_feedback = self.get_all_feedback()
        return [
            fb for fb in all_feedback
            if fb['rating'] >= threshold
        ]
    
    def get_average_rating(self) -> float:
        """Calculate average rating across all feedback.
        
        Returns:
            Average rating (0.0 if no feedback)
        """
        all_feedback = self.get_all_feedback()
        
        if not all_feedback:
            return 0.0
        
        ratings = [fb['rating'] for fb in all_feedback]
        return sum(ratings) / len(ratings)
    
    def get_rating_distribution(self) -> Dict[int, int]:
        """Get distribution of ratings.
        
        Returns:
            Dictionary mapping rating (1-5) to count
        """
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for fb in self.get_all_feedback():
            rating = fb['rating']
            distribution[rating] += 1
        
        return distribution
    
    def get_feedback_stats(self) -> Dict:
        """Get comprehensive feedback statistics.
        
        Returns:
            Dictionary with various statistics
        """
        all_feedback = self.get_all_feedback()
        
        if not all_feedback:
            return {
                'total_feedback': 0,
                'average_rating': 0.0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'positive_rate': 0.0,
                'negative_rate': 0.0
            }
        
        ratings = [fb['rating'] for fb in all_feedback]
        positive = sum(1 for r in ratings if r >= 4)
        negative = sum(1 for r in ratings if r <= 2)
        
        return {
            'total_feedback': len(all_feedback),
            'average_rating': round(sum(ratings) / len(ratings), 2),
            'rating_distribution': self.get_rating_distribution(),
            'positive_rate': round(positive / len(all_feedback) * 100, 2),
            'negative_rate': round(negative / len(all_feedback) * 100, 2),
            'with_comments': sum(1 for fb in all_feedback if fb.get('comment'))
        }
    
    def get_common_issues(self, min_occurrences: int = 2) -> List[Tuple[str, int]]:
        """Identify common issues from low-rated queries.
        
        Args:
            min_occurrences: Minimum number of occurrences to include
            
        Returns:
            List of tuples (query pattern, count) sorted by frequency
        """
        low_rated = self.get_low_rated_queries()
        
        # Simple keyword extraction from queries
        query_keywords = defaultdict(int)
        
        for fb in low_rated:
            # Extract significant words (length > 3)
            words = [
                word.lower() 
                for word in fb['query'].split() 
                if len(word) > 3
            ]
            for word in words:
                query_keywords[word] += 1
        
        # Filter and sort
        common = [
            (word, count) 
            for word, count in query_keywords.items()
            if count >= min_occurrences
        ]
        
        return sorted(common, key=lambda x: x[1], reverse=True)
    
    def get_improvement_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on feedback.
        
        Returns:
            List of suggested improvements
        """
        suggestions = []
        stats = self.get_feedback_stats()
        
        # Low average rating
        if stats['average_rating'] < 3.5:
            suggestions.append(
                f"⚠️ Average rating is low ({stats['average_rating']:.2f}/5.0). "
                "Review answer quality and relevance."
            )
        
        # High negative rate
        if stats['negative_rate'] > 30:
            suggestions.append(
                f"⚠️ {stats['negative_rate']:.1f}% of feedback is negative. "
                "Analyze low-rated queries for common patterns."
            )
        
        # Common issues
        common_issues = self.get_common_issues()
        if common_issues:
            top_issues = ', '.join([word for word, _ in common_issues[:3]])
            suggestions.append(
                f"🔍 Common topics in low-rated queries: {top_issues}. "
                "Consider improving documentation in these areas."
            )
        
        # Low feedback volume
        if stats['total_feedback'] < 10:
            suggestions.append(
                "💡 Collect more feedback to get better insights. "
                "Encourage users to rate responses."
            )
        
        # Good performance
        if stats['average_rating'] >= 4.0 and stats['positive_rate'] >= 70:
            suggestions.append(
                f"✅ Great performance! {stats['positive_rate']:.1f}% positive feedback. "
                "Keep up the good work."
            )
        
        return suggestions
    
    def export_report(self, output_file: str):
        """Export detailed feedback report.
        
        Args:
            output_file: Path to output report file
        """
        stats = self.get_feedback_stats()
        low_rated = self.get_low_rated_queries(threshold=2)
        common_issues = self.get_common_issues()
        suggestions = self.get_improvement_suggestions()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'low_rated_examples': [
                {
                    'query': fb['query'],
                    'rating': fb['rating'],
                    'comment': fb.get('comment', '')
                }
                for fb in low_rated[:10]  # Top 10 worst
            ],
            'common_issues': common_issues,
            'suggestions': suggestions
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    
    def clear_old_feedback(self, days: int = 90):
        """Remove feedback older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        all_feedback = self.get_all_feedback()
        recent_feedback = [
            fb for fb in all_feedback
            if datetime.fromisoformat(fb['timestamp']) > cutoff_date
        ]
        
        # Rewrite file with recent feedback only
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            for fb in recent_feedback:
                f.write(json.dumps(fb) + '\n')
