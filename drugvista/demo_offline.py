"""
Offline Demo for DRUGVISTA
Works without OpenAI API key for hackathon demonstration
"""
import sys
import os
sys.path.append("backend")

from vector_store import VectorStore
import json

class OfflineRAGPipeline:
    def __init__(self):
        """Initialize offline RAG pipeline"""
        # Change to backend directory to find the index files
        original_dir = os.getcwd()
        try:
            os.chdir("backend")
            self.vector_store = VectorStore()
            os.chdir(original_dir)
        except:
            # If already in backend directory
            self.vector_store = VectorStore()
        
        stats = self.vector_store.get_stats()
        print(f"✅ Loaded vector store with {stats['total_documents']} documents")
    
    def analyze(self, query: str) -> dict:
        """Analyze query using retrieved documents and rule-based reasoning"""
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, top_k=5)
        
        if not retrieved_docs:
            return self._fallback_response("No relevant documents found")
        
        # Step 2: Rule-based analysis (simulating LLM reasoning)
        analysis = self._analyze_documents(query, retrieved_docs)
        
        return analysis
    
    def _analyze_documents(self, query: str, docs: list) -> dict:
        """Rule-based document analysis"""
        
        query_lower = query.lower()
        doc_contents = [doc['content'].lower() for doc in docs]
        all_content = ' '.join(doc_contents)
        
        # Clinical Viability Assessment
        viability = "Medium"
        if any(word in all_content for word in ['effective', 'successful', 'promising', 'approved']):
            viability = "High"
        elif any(word in all_content for word in ['failed', 'ineffective', 'toxic', 'discontinued']):
            viability = "Low"
        
        # Risk Assessment
        risks = []
        if 'toxicity' in all_content or 'toxic' in all_content:
            risks.append('toxicity concerns')
        if 'side effect' in all_content or 'adverse' in all_content:
            risks.append('adverse effects')
        if 'trial' in all_content and 'fail' in all_content:
            risks.append('clinical trial failures')
        if 'bleeding' in all_content:
            risks.append('bleeding risk')
        if not risks:
            risks = ['standard development risks']
        
        # Market Signal
        market_signal = "Moderate"
        if any(word in all_content for word in ['billion', 'growing', 'strong', 'demand']):
            market_signal = "Strong"
        elif any(word in all_content for word in ['declining', 'weak', 'saturated']):
            market_signal = "Weak"
        
        # Recommendation Logic
        recommendation = "Investigate Further"
        if viability == "High" and market_signal == "Strong":
            recommendation = "Proceed"
        elif viability == "Low" or market_signal == "Weak":
            recommendation = "Drop"
        
        # Confidence based on document relevance
        confidence = min(0.6 + (len(docs) * 0.05), 0.95)
        
        # Evidence files
        evidence_files = [doc.get('filename', f'doc_{i}') for i, doc in enumerate(docs[:3])]
        
        # Generate explanation
        explanation = self._generate_explanation(query, docs, viability, risks, market_signal, recommendation)
        
        return {
            "clinical_viability": viability,
            "key_evidence": evidence_files,
            "major_risks": risks,
            "market_signal": market_signal,
            "recommendation": recommendation,
            "confidence_score": round(confidence, 2),
            "explanation": explanation
        }
    
    def _generate_explanation(self, query, docs, viability, risks, market_signal, recommendation):
        """Generate human-readable explanation"""
        
        doc_types = [doc.get('type', 'unknown') for doc in docs]
        
        explanation = f"""
**Query Analysis**: {query}

**Clinical Assessment**: Based on analysis of {len(docs)} relevant documents, the clinical viability is assessed as {viability}. 
The evidence includes {doc_types.count('paper')} research papers, {doc_types.count('clinical_trial')} clinical trials, 
and {doc_types.count('market')} market reports.

**Risk Profile**: Key risks identified include {', '.join(risks[:3])}. These factors should be carefully monitored 
in any development program.

**Market Intelligence**: Market signal strength is {market_signal} based on available market data and competitive landscape analysis.

**Final Recommendation**: {recommendation} - This recommendation is based on the balance of clinical evidence, 
risk factors, and market opportunity assessment.

**Evidence Base**: Analysis based on {len(docs)} documents with {confidence*100:.0f}% confidence level.
        """.strip()
        
        return explanation
    
    def _fallback_response(self, error: str) -> dict:
        """Fallback response for errors"""
        return {
            "clinical_viability": "Unknown",
            "key_evidence": [],
            "major_risks": ["insufficient data"],
            "market_signal": "Unknown", 
            "recommendation": "Gather More Data",
            "confidence_score": 0.0,
            "explanation": f"Analysis could not be completed: {error}. Please try a different query or check the data sources."
        }

def demo_queries():
    """Run demo with sample queries"""
    
    print("🧬 DRUGVISTA Offline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = OfflineRAGPipeline()
    
    # Sample queries
    queries = [
        "Alzheimer's disease treatment options",
        "Cancer immunotherapy safety profile", 
        "Aspirin molecular mechanism",
        "Drug toxicity prediction models",
        "Diabetes GLP-1 receptor agonists"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        result = pipeline.analyze(query)
        
        print(f"Clinical Viability: {result['clinical_viability']}")
        print(f"Market Signal: {result['market_signal']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence_score']*100:.0f}%")
        print(f"Key Evidence: {', '.join(result['key_evidence'][:2])}")
        print(f"Major Risks: {', '.join(result['major_risks'][:2])}")
        
        print("\nDetailed Analysis:")
        print(result['explanation'][:300] + "...")

def interactive_demo():
    """Interactive demo mode"""
    
    print("🧬 DRUGVISTA Interactive Demo")
    print("=" * 50)
    print("Enter queries about diseases, molecules, or research topics.")
    print("Type 'quit' to exit.\n")
    
    pipeline = OfflineRAGPipeline()
    
    while True:
        query = input("🔍 Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using DRUGVISTA!")
            break
        
        if len(query) < 3:
            print("❌ Please enter a longer query (at least 3 characters)")
            continue
        
        print("\n🔄 Analyzing...")
        result = pipeline.analyze(query)
        
        print(f"\n📊 Results for: '{query}'")
        print("-" * 50)
        print(f"🔬 Clinical Viability: {result['clinical_viability']}")
        print(f"📈 Market Signal: {result['market_signal']}")
        print(f"🎯 Recommendation: {result['recommendation']}")
        print(f"📊 Confidence: {result['confidence_score']*100:.0f}%")
        print(f"📄 Evidence: {', '.join(result['key_evidence'][:2])}")
        print(f"⚠️  Risks: {', '.join(result['major_risks'][:2])}")
        
        show_details = input("\n📋 Show detailed analysis? (y/n): ").lower().startswith('y')
        if show_details:
            print("\n" + result['explanation'])
        
        print("\n" + "="*50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        demo_queries()