"""
Retriever Module for AI Policy Query System

This module handles semantic search and retrieval of relevant document chunks
using embeddings and vector similarity search.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .embedder import EmbeddingManager


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result.
    
    Attributes:
        text: The retrieved text chunk
        score: Similarity score (0-1, higher is better)
        metadata: Associated metadata
        chunk_id: Unique identifier for the chunk
    """
    text: str
    score: float
    metadata: Dict
    chunk_id: int


class PolicyRetriever:
    """
    Handles retrieval of relevant policy document chunks based on user queries.
    
    This class provides semantic search capabilities with query preprocessing
    and result post-processing for optimal retrieval performance.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever with an embedding manager.
        
        Args:
            embedding_manager: Configured EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        
        # Query preprocessing patterns
        self.query_patterns = {
            'coverage_keywords': ['covered', 'coverage', 'benefit', 'included', 'eligible'],
            'exclusion_keywords': ['excluded', 'not covered', 'limitation', 'restriction'],
            'time_keywords': ['waiting period', 'duration', 'term', 'period'],
            'amount_keywords': ['amount', 'limit', 'maximum', 'minimum', 'deductible']
        }
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.1,
        preprocess_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant document chunks for a given query.
        
        Args:
            query: User's natural language question
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
            preprocess_query: Whether to apply query preprocessing
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        # Validate input
        if not query or not query.strip():
            return []
        
        if top_k <= 0:
            return []
        
        # Preprocess query if enabled
        if preprocess_query:
            processed_query = self._preprocess_query(query)
        else:
            processed_query = query
        
        # Perform semantic search
        try:
            raw_results = self.embedding_manager.search(processed_query, top_k)
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        # Convert to RetrievalResult objects and filter by score
        results = []
        for i, (text, score, metadata) in enumerate(raw_results):
            if score >= min_score:
                result = RetrievalResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                    chunk_id=metadata.get('chunk_id', i)
                )
                results.append(result)
        
        # Post-process results
        results = self._post_process_results(results, query)
        
        return results
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve retrieval performance.
        
        Args:
            query: Original user query
            
        Returns:
            Preprocessed query string
        """
        # Convert to lowercase for processing
        processed = query.lower().strip()
        
        # Expand common insurance terms
        expansions = {
            'maternity': 'maternity pregnancy childbirth delivery',
            'dental': 'dental teeth oral mouth',
            'vision': 'vision eye optical glasses contacts',
            'prescription': 'prescription medication drugs pharmacy',
            'emergency': 'emergency urgent care immediate',
            'surgery': 'surgery surgical operation procedure',
            'mental health': 'mental health psychiatric therapy counseling'
        }
        
        # Apply expansions
        for term, expansion in expansions.items():
            if term in processed:
                processed = processed.replace(term, expansion)
        
        # Add question context for better matching
        question_words = ['what', 'when', 'where', 'how', 'why', 'is', 'are', 'does', 'do']
        if not any(qw in processed.split()[:3] for qw in question_words):
            # If query doesn't start with question word, add context
            processed = f"information about {processed}"
        
        return processed
    
    def _post_process_results(
        self, 
        results: List[RetrievalResult], 
        original_query: str
    ) -> List[RetrievalResult]:
        """
        Post-process retrieval results to improve relevance.
        
        Args:
            results: Raw retrieval results
            original_query: Original user query
            
        Returns:
            Post-processed and re-ranked results
        """
        if not results:
            return results
        
        # Apply keyword-based boosting
        boosted_results = self._apply_keyword_boosting(results, original_query)
        
        # Remove duplicates based on text similarity
        deduplicated_results = self._remove_similar_chunks(boosted_results)
        
        # Re-sort by adjusted score
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)
        
        return deduplicated_results
    
    def _apply_keyword_boosting(
        self, 
        results: List[RetrievalResult], 
        query: str
    ) -> List[RetrievalResult]:
        """
        Apply keyword-based score boosting to results.
        
        Args:
            results: Original results
            query: User query
            
        Returns:
            Results with adjusted scores
        """
        query_lower = query.lower()
        
        for result in results:
            text_lower = result.text.lower()
            boost_factor = 1.0
            
            # Boost for exact keyword matches
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            text_words = set(re.findall(r'\b\w+\b', text_lower))
            
            # Calculate word overlap
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                boost_factor += 0.1 * overlap
            
            # Special boosts for insurance-specific patterns
            if 'coverage' in query_lower and any(word in text_lower for word in ['covered', 'coverage', 'benefit']):
                boost_factor += 0.2
            
            if 'excluded' in query_lower and any(word in text_lower for word in ['excluded', 'not covered', 'limitation']):
                boost_factor += 0.2
            
            if any(word in query_lower for word in ['amount', 'cost', '$']) and '$' in result.text:
                boost_factor += 0.15
            
            # Apply boost
            result.score = min(1.0, result.score * boost_factor)
        
        return results
    
    def _remove_similar_chunks(
        self, 
        results: List[RetrievalResult], 
        similarity_threshold: float = 0.8
    ) -> List[RetrievalResult]:
        """
        Remove chunks that are too similar to avoid redundancy.
        
        Args:
            results: List of results to deduplicate
            similarity_threshold: Threshold for considering chunks similar
            
        Returns:
            Deduplicated results
        """
        if len(results) <= 1:
            return results
        
        filtered_results = []
        
        for result in results:
            is_duplicate = False
            
            for existing in filtered_results:
                # Simple similarity check based on word overlap
                words1 = set(result.text.lower().split())
                words2 = set(existing.text.lower().split())
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    if union > 0:  # Prevent division by zero
                        jaccard_similarity = overlap / union
                        
                        if jaccard_similarity > similarity_threshold:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_context_for_query(
        self, 
        query: str, 
        max_context_length: int = 2000,
        top_k: int = 3
    ) -> str:
        """
        Get concatenated context from top relevant chunks for LLM processing.
        
        Args:
            query: User query
            max_context_length: Maximum character length of context
            top_k: Number of top chunks to consider
            
        Returns:
            Concatenated context string
        """
        results = self.retrieve(query, top_k=top_k)
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Add section header
            section_header = f"\n--- Relevant Section {i+1} (Score: {result.score:.2f}) ---\n"
            chunk_text = result.text
            
            # Check if adding this chunk would exceed limit
            addition_length = len(section_header) + len(chunk_text)
            if current_length + addition_length > max_context_length and context_parts:
                break
            
            context_parts.append(section_header)
            context_parts.append(chunk_text)
            current_length += addition_length
        
        if not context_parts:
            return "No relevant context found for the query."
        
        return "".join(context_parts)
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with retrieval system statistics
        """
        embedding_stats = self.embedding_manager.get_stats()
        
        return {
            'embedding_stats': embedding_stats,
            'preprocessing_enabled': True,
            'supported_file_types': ['pdf', 'docx'],
            'default_top_k': 5,
            'min_score_threshold': 0.1
        }


def create_retriever_for_documents(
    text_chunks: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    use_openai: bool = False
) -> PolicyRetriever:
    """
    Convenience function to create a retriever for document chunks.
    
    Args:
        text_chunks: List of text chunks from documents
        model_name: Embedding model name
        use_openai: Whether to use OpenAI embeddings
        
    Returns:
        PolicyRetriever instance ready for use
    """
    from .embedder import create_embeddings_for_document
    
    embedding_manager = create_embeddings_for_document(
        text_chunks, 
        model_name=model_name, 
        use_openai=use_openai
    )
    
    return PolicyRetriever(embedding_manager)


# Example usage and testing
if __name__ == "__main__":
    # Sample policy chunks for testing
    sample_chunks = [
        "Section 4.2 - Maternity Coverage: Maternity expenses including prenatal care, delivery, and postnatal care are covered after a 2-year waiting period. Maximum benefit is $10,000 per pregnancy.",
        "Section 3.1 - Dental Benefits: Routine dental care including cleanings, fillings, and extractions are covered up to $1,000 per calendar year. Orthodontic treatment requires pre-authorization.",
        "Section 5.3 - Emergency Services: Emergency room visits and urgent care are covered 24/7 worldwide. No prior authorization required for true emergencies.",
        "Section 2.4 - Exclusions: The following are not covered under this policy: cosmetic surgery, experimental treatments, and pre-existing conditions not disclosed at enrollment.",
        "Section 6.1 - Prescription Drug Coverage: Generic medications are covered at 90%, brand-name drugs at 70%. Specialty drugs require prior authorization and have a $500 monthly maximum."
    ]
    
    try:
        print("Creating retriever for sample policy chunks...")
        retriever = create_retriever_for_documents(sample_chunks, use_openai=False)
        
        # Test various queries
        test_queries = [
            "Is maternity covered?",
            "What about dental care limits?",
            "Are emergency services available overseas?",
            "What medications are excluded?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = retriever.retrieve(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     Text: {result.text[:100]}...")
        
        # Test context generation
        context = retriever.get_context_for_query("maternity benefits and waiting period")
        print(f"\n📄 Generated context length: {len(context)} characters")
        
        # Show stats
        stats = retriever.get_retrieval_stats()
        print(f"\n📊 Retrieval stats: {stats['embedding_stats']}")
        
    except Exception as e:
        print(f"Error during testing: {e}") 