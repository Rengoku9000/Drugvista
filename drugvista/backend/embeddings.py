"""
Document Embedding and Index Creation
AWS Mapping: This would be a Lambda function triggered by S3 uploads
"""
import os
import glob
import logging
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_data_folder():
    """
    Load all documents from the data folder
    AWS Mapping: This would read from S3 buckets
    """
    documents = []
    data_path = "../data"
    
    # Load papers
    papers_path = os.path.join(data_path, "papers", "*.txt")
    for file_path in glob.glob(papers_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'filename': os.path.basename(file_path),
                'type': 'paper',
                'content': content,
                'source_path': file_path
            })
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    # Load clinical trials
    trials_path = os.path.join(data_path, "clinical_trials", "*.txt")
    for file_path in glob.glob(trials_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'filename': os.path.basename(file_path),
                'type': 'clinical_trial',
                'content': content,
                'source_path': file_path
            })
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    # Load market intelligence
    market_path = os.path.join(data_path, "market", "*.txt")
    for file_path in glob.glob(market_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'filename': os.path.basename(file_path),
                'type': 'market',
                'content': content,
                'source_path': file_path
            })
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def create_vector_index():
    """
    Create vector index from documents
    AWS Mapping: Lambda function for batch processing
    """
    logger.info("Starting vector index creation")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Load documents
    documents = load_documents_from_data_folder()
    
    if not documents:
        logger.error("No documents found! Make sure data folder exists with content.")
        return False
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Save index
    vector_store.save_index()
    
    # Print statistics
    stats = vector_store.get_stats()
    logger.info(f"Vector index created successfully:")
    logger.info(f"- Total documents: {stats['total_documents']}")
    logger.info(f"- Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"- Model: {stats['model_name']}")
    
    return True

def test_search():
    """Test the search functionality"""
    logger.info("Testing search functionality")
    
    vector_store = VectorStore()
    
    test_queries = [
        "Alzheimer's disease treatment",
        "cancer immunotherapy",
        "drug toxicity",
        "clinical trial results",
        "market analysis"
    ]
    
    for query in test_queries:
        results = vector_store.search(query, top_k=3)
        logger.info(f"Query: '{query}' -> {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['filename']} (score: {result['similarity_score']:.3f})")

if __name__ == "__main__":
    # Create vector index
    success = create_vector_index()
    
    if success:
        # Test search
        test_search()
        logger.info("Setup complete! You can now start the backend server.")
    else:
        logger.error("Setup failed. Please check the data folder and try again.")