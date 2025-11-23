"""Query caching for improved performance."""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


class QueryCache:
    """Cache for frequently asked questions to improve response time."""
    
    def __init__(self, ttl_minutes: int = 60, max_size: int = 1000):
        """Initialize query cache.
        
        Args:
            ttl_minutes: Time-to-live for cache entries in minutes
            max_size: Maximum number of entries to store
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent cache keys.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(query.lower().strip().split())
        
        # Remove common punctuation that doesn't change meaning
        for char in ['?', '!', '.', ',']:
            normalized = normalized.replace(char, '')
        
        return normalized
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query.
        
        Args:
            query: Query string
            
        Returns:
            Cache key (MD5 hash)
        """
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for query.
        
        Args:
            query: Query string
            
        Returns:
            Cached response or None if not found/expired
        """
        key = self._get_cache_key(query)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if entry is still valid
            if datetime.now() - entry['cached_at'] < self.ttl:
                self.hits += 1
                entry['hit_count'] += 1
                entry['last_accessed'] = datetime.now()
                return entry['response']
            else:
                # Entry expired, remove it
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: Dict[str, Any]):
        """Cache a response.
        
        Args:
            query: Query string
            response: Response to cache
        """
        key = self._get_cache_key(query)
        
        # If cache is full, remove least recently used entry
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'query': query,
            'response': response,
            'cached_at': datetime.now(),
            'last_accessed': datetime.now(),
            'hit_count': 0
        }
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find entry with oldest last_accessed time
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_accessed']
        )
        del self.cache[lru_key]
    
    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'ttl_minutes': self.ttl.total_seconds() / 60
        }
    
    def get_popular_queries(self, top_n: int = 10) -> list:
        """Get most frequently accessed queries.
        
        Args:
            top_n: Number of top queries to return
            
        Returns:
            List of tuples (query, hit_count)
        """
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda x: x['hit_count'],
            reverse=True
        )
        
        return [
            (entry['query'], entry['hit_count'])
            for entry in sorted_entries[:top_n]
        ]
    
    def invalidate(self, query: str):
        """Invalidate specific cache entry.
        
        Args:
            query: Query string to invalidate
        """
        key = self._get_cache_key(query)
        if key in self.cache:
            del self.cache[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all entries matching pattern.
        
        Args:
            pattern: Pattern to match in queries (case-insensitive)
        """
        pattern_lower = pattern.lower()
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if pattern_lower in entry['query'].lower()
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def save_to_disk(self, filepath: str):
        """Save cache to disk.
        
        Args:
            filepath: Path to save cache file
        """
        cache_data = {
            'cache': {
                key: {
                    **entry,
                    'cached_at': entry['cached_at'].isoformat(),
                    'last_accessed': entry['last_accessed'].isoformat()
                }
                for key, entry in self.cache.items()
            },
            'stats': {
                'hits': self.hits,
                'misses': self.misses
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_from_disk(self, filepath: str):
        """Load cache from disk.
        
        Args:
            filepath: Path to cache file
        """
        if not Path(filepath).exists():
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        self.cache = {}
        for key, entry in cache_data['cache'].items():
            # Convert ISO format strings back to datetime
            entry['cached_at'] = datetime.fromisoformat(entry['cached_at'])
            entry['last_accessed'] = datetime.fromisoformat(entry['last_accessed'])
            
            # Only restore non-expired entries
            if datetime.now() - entry['cached_at'] < self.ttl:
                self.cache[key] = entry
        
        stats = cache_data.get('stats', {})
        self.hits = stats.get('hits', 0)
        self.misses = stats.get('misses', 0)
