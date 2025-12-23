# DRUGVISTA - AI Co-pilot for Molecular, Clinical, and Market Intelligence

## What It Does
DRUGVISTA is an AI-powered system that analyzes diseases, molecules, or research abstracts to provide structured clinical insights, risk assessments, and market intelligence for pharmaceutical decision-making.

## Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │     Backend      │    │   Data Layer    │
│   (Streamlit)   │───▶│   (FastAPI)      │───▶│   (FAISS +      │
│                 │    │                  │    │    Documents)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   GenAI Pipeline │
                       │   (Multi-step    │
                       │    Reasoning)    │
                       └──────────────────┘
```

## GenAI Core Features
- **Multi-step Reasoning**: 4-stage analysis (Context → Clinical → Market → Decision)
- **RAG Implementation**: Vector search over biomedical documents with embeddings
- **Structured Output**: JSON + human-readable explanations
- **Evidence Citation**: Links recommendations to source documents

## RAG Pipeline
1. **Document Ingestion**: Biomedical papers, clinical trials, market news
2. **Embedding Generation**: sentence-transformers for semantic search
3. **Vector Storage**: FAISS for efficient similarity search
4. **Context Retrieval**: Top-K relevant documents for each query
5. **LLM Reasoning**: Multi-step analysis with retrieved context

## AWS Alignment (ImpactX Criteria)
- **S3**: Document storage (currently local files)
- **Bedrock**: LLM inference (currently OpenAI-compatible)
- **Lambda**: API orchestration (currently FastAPI)
- **SageMaker**: Embedding models (currently sentence-transformers)
- **Healthcare Impact**: Accelerates drug discovery and clinical decision-making

## Quick Start (< 5 minutes)

### Prerequisites
```bash
pip install fastapi uvicorn streamlit sentence-transformers faiss-cpu openai python-multipart
```

### 1. Setup Data & Embeddings
```bash
cd drugvista
python backend/embeddings.py  # Creates vector store
```

### 2. Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 3. Start Frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

### 4. Demo
- Open http://localhost:8501
- Try queries like:
  - "Alzheimer's disease treatment"
  - "Aspirin molecular structure"
  - "Phase II clinical trial results for cancer immunotherapy"

## Sample Output
```json
{
  "clinical_viability": "High",
  "key_evidence": ["alzheimer_paper_1.txt", "clinical_trial_3.txt"],
  "major_risks": ["blood-brain barrier penetration", "dosage optimization"],
  "market_signal": "Strong",
  "recommendation": "Proceed",
  "confidence_score": 0.85
}
```

## Technology Stack
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **GenAI**: OpenAI GPT (configurable)
- **Embeddings**: sentence-transformers
- **Vector DB**: FAISS
- **Data**: Curated biomedical documents