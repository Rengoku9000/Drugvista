"""
Vector Store Implementation using FAISS
AWS Mapping: This would use SageMaker for embeddings + S3 for document storage
"""
import os
import json
import pickle
import numpy as np
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store
        AWS Mapping: SageMaker endpoint for embeddings
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # FAISS index (AWS: would be managed vector DB)
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load pre-built index and metadata"""
        index_path = "vector_index.faiss"
        metadata_path = "vector_metadata.pkl"
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                logger.info(f"Loaded index with {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._create_empty_index()
        else:
            logger.info("No existing index found")
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create empty FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to vector store
        AWS Mapping: Documents would be stored in S3, embeddings in SageMaker
        """
        if not documents:
            return
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        """
        if self.index.ntotal == 0:
            logger.warning("No documents in vector store")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['content'] = self.documents[idx]
                results.append(result)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, "vector_index.faiss")
            with open("vector_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name
        }