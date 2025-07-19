"""
Embedder Module for AI Policy Query System

This module handles the generation of text embeddings and management of FAISS vector stores
for efficient similarity search in policy documents.
"""

import os
import pickle
from typing import List, Optional, Tuple, Union
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
import openai


class EmbeddingManager:
    """
    Manages text embeddings and FAISS vector store operations.
    
    This class provides functionality to generate embeddings using either
    SentenceTransformers or OpenAI, and manage FAISS indexes for fast similarity search.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the SentenceTransformer model or OpenAI model
            use_openai: Whether to use OpenAI embeddings instead of SentenceTransformers
            openai_api_key: OpenAI API key (if using OpenAI embeddings)
        """
        self.use_openai = use_openai
        self.model_name = model_name
        
        if use_openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError("OpenAI API key required when use_openai=True")
            
            # Set default OpenAI embedding model
            if model_name == "all-MiniLM-L6-v2":
                self.model_name = "text-embedding-ada-002"
            
            self.model = None  # OpenAI doesn't need model loading
            self.embedding_dim = 1536  # Default for text-embedding-ada-002
        else:
            # Load SentenceTransformer model
            print(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []  # Store original text chunks
        self.metadata = []   # Store metadata for each chunk
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([])
        
        if self.use_openai:
            return self._generate_openai_embeddings(valid_texts)
        else:
            return self._generate_sentence_transformer_embeddings(valid_texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = []
        
        # Process texts in batches to avoid API limits
        batch_size = 100
        max_retries = 3
        
        # Use the new OpenAI API format for v1.0.0+
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise RuntimeError(f"Error generating OpenAI embeddings after {max_retries} attempts: {str(e)}")
                    else:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformers.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 50
            )
            return embeddings.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Error generating SentenceTransformer embeddings: {str(e)}")
    
    def create_index(self, texts: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Create FAISS index from text chunks.
        
        Args:
            texts: List of text chunks to index
            metadata: Optional metadata for each text chunk
        """
        if not texts:
            raise ValueError("Cannot create index from empty text list")
        
        print(f"Creating embeddings for {len(texts)} text chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents = texts.copy()
        self.metadata = metadata if metadata else [{"chunk_id": i} for i in range(len(texts))]
        
        print(f"FAISS index created successfully with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar text chunks using the query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (text_chunk, similarity_score, metadata)
        """
        if not self.index:
            raise ValueError("No index created. Call create_index() first.")
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def save_index(self, file_path: str) -> None:
        """
        Save the FAISS index and associated data to disk.
        
        Args:
            file_path: Path to save the index (without extension)
        """
        if not self.index:
            raise ValueError("No index to save. Create an index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{file_path}.faiss")
        
        # Save documents and metadata
        with open(f"{file_path}_data.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "use_openai": self.use_openai
            }, f)
        
        print(f"Index saved to {file_path}.faiss and {file_path}_data.pkl")
    
    def load_index(self, file_path: str) -> None:
        """
        Load a saved FAISS index and associated data from disk.
        
        Args:
            file_path: Path to the saved index (without extension)
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{file_path}.faiss")
        
        # Load documents and metadata
        with open(f"{file_path}_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.embedding_dim = data["embedding_dim"]
            
            # Verify model compatibility
            saved_model = data["model_name"]
            saved_use_openai = data["use_openai"]
            
            if saved_use_openai != self.use_openai or saved_model != self.model_name:
                print(f"Warning: Loaded index was created with different model settings")
                print(f"  Saved: {saved_model} (OpenAI: {saved_use_openai})")
                print(f"  Current: {self.model_name} (OpenAI: {self.use_openai})")
        
        print(f"Index loaded successfully with {self.index.ntotal} vectors")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.index:
            return {"status": "No index created"}
        
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "using_openai": self.use_openai,
            "total_documents": len(self.documents)
        }


def create_embeddings_for_document(
    text_chunks: List[str], 
    model_name: str = "all-MiniLM-L6-v2",
    use_openai: bool = False
) -> EmbeddingManager:
    """
    Convenience function to create embeddings for document chunks.
    
    Args:
        text_chunks: List of text chunks from document parsing
        model_name: Name of the embedding model to use
        use_openai: Whether to use OpenAI embeddings
        
    Returns:
        EmbeddingManager instance with created index
    """
    embedder = EmbeddingManager(model_name=model_name, use_openai=use_openai)
    
    # Create metadata for chunks
    metadata = [{"chunk_id": i, "chunk_text": chunk[:100] + "..."} 
                for i, chunk in enumerate(text_chunks)]
    
    embedder.create_index(text_chunks, metadata)
    return embedder


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Insurance coverage includes medical expenses and hospitalization.",
        "Maternity benefits are available after a 2-year waiting period.",
        "Dental care is covered up to $1000 per year.",
        "Emergency services are covered 24/7 worldwide."
    ]
    
    try:
        # Test with SentenceTransformers
        print("Testing with SentenceTransformers...")
        embedder = EmbeddingManager(use_openai=False)
        embedder.create_index(sample_texts)
        
        # Test search
        query = "What about pregnancy coverage?"
        results = embedder.search(query, top_k=2)
        
        print(f"\nSearch results for: '{query}'")
        for text, score, metadata in results:
            print(f"Score: {score:.3f} - {text[:80]}...")
        
        # Show stats
        stats = embedder.get_stats()
        print(f"\nIndex stats: {stats}")
        
    except Exception as e:
        print(f"Error during testing: {e}") 