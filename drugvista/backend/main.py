"""
FastAPI Backend for DRUGVISTA
AWS Mapping: This would be deployed as Lambda + API Gateway
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from typing import Optional
import logging
import os
import csv
import json
from io import StringIO, BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DRUGVISTA API")

# Supported file types
ALLOWED_EXTENSIONS = {'.txt', '.csv', '.json', '.pdf', '.docx'}


def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except ImportError:
        raise HTTPException(status_code=500, detail="PyPDF2 not installed. Run: pip install PyPDF2")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")


def extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(BytesIO(content))
        text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text_parts)
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {str(e)}")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
try:
    rag = RAGPipeline()
    logger.info("RAG Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")
    rag = None

class AnalysisRequest(BaseModel):
    query: str

class AnalysisResponse(BaseModel):
    clinical_viability: str
    key_evidence: list[str]
    major_risks: list[str]
    market_signal: str
    recommendation: str
    confidence_score: float
    explanation: str

@app.get("/")
def root():
    return {
        "service": "DRUGVISTA API",
        "version": "1.0",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Main analysis endpoint
    AWS Mapping: Lambda function triggered by API Gateway
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}")
        result = rag.analyze(request.query)
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_added: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form(default="patient_data"),
    description: Optional[str] = Form(default=None)
):
    """
    Ingest new patient data or documents into the vector store
    AWS Mapping: S3 upload + Lambda trigger for indexing
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file_ext == '.pdf':
            text_content = extract_pdf_text(content)
        elif file_ext == '.docx':
            text_content = extract_docx_text(content)
        else:
            text_content = content.decode('utf-8')
        
        if len(text_content.strip()) < 10:
            raise HTTPException(status_code=400, detail="File content too short")
        
        documents = []
        
        if file_ext == '.csv':
            # Parse CSV - each row becomes a document
            reader = csv.DictReader(StringIO(text_content))
            for i, row in enumerate(reader):
                # Convert row to readable text
                row_text = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
                if len(row_text.strip()) >= 10:
                    documents.append({
                        'content': row_text,
                        'filename': f"{file.filename}_row_{i+1}",
                        'type': doc_type,
                        'description': description or f"CSV row {i+1} from {file.filename}"
                    })
        
        elif file_ext == '.json':
            # Parse JSON - handle array of records or single object
            data = json.loads(text_content)
            
            if isinstance(data, list):
                # Array of records
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        item_text = "\n".join([f"{k}: {v}" for k, v in item.items()])
                    else:
                        item_text = str(item)
                    
                    if len(item_text.strip()) >= 10:
                        documents.append({
                            'content': item_text,
                            'filename': f"{file.filename}_item_{i+1}",
                            'type': doc_type,
                            'description': description or f"JSON item {i+1} from {file.filename}"
                        })
            else:
                # Single object
                item_text = "\n".join([f"{k}: {v}" for k, v in data.items()])
                documents.append({
                    'content': item_text,
                    'filename': file.filename,
                    'type': doc_type,
                    'description': description or f"User uploaded: {file.filename}"
                })
        
        else:
            # Plain text file - single document
            documents.append({
                'content': text_content,
                'filename': file.filename,
                'type': doc_type,
                'description': description or f"User uploaded: {file.filename}"
            })
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid content found in file")
        
        # Add to vector store
        rag.vector_store.add_documents(documents)
        rag.vector_store.save_index()
        
        logger.info(f"Ingested {len(documents)} documents from: {file.filename}")
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(documents)} record(s) from {file.filename}",
            documents_added=len(documents)
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TextIngestRequest(BaseModel):
    content: str
    doc_type: str = "patient_data"
    title: Optional[str] = None


@app.post("/ingest-text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """
    Ingest patient data as plain text
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if len(request.content.strip()) < 10:
        raise HTTPException(status_code=400, detail="Content too short (min 10 characters)")
    
    try:
        document = {
            'content': request.content,
            'filename': request.title or "user_text_input",
            'type': request.doc_type,
            'description': f"Text input: {request.title or 'Patient data'}"
        }
        
        rag.vector_store.add_documents([document])
        rag.vector_store.save_index()
        
        logger.info(f"Ingested text: {request.title or 'user input'}")
        
        return IngestResponse(
            success=True,
            message="Successfully added text data",
            documents_added=1
        )
    except Exception as e:
        logger.error(f"Text ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get vector store statistics"""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    return rag.vector_store.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
