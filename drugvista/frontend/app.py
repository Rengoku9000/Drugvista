"""
Streamlit Frontend for DRUGVISTA
Simple, hackathon-friendly UI
"""
import streamlit as st
import requests
import json
import time

# Page config
st.set_page_config(
    page_title="DRUGVISTA - AI Co-pilot",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.tagline {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.result-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.recommendation-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-weight: bold;
}
.proceed { background-color: #d4edda; color: #155724; }
.investigate { background-color: #fff3cd; color: #856404; }
.drop { background-color: #f8d7da; color: #721c24; }
.confidence-high {
    background: linear-gradient(90deg, #28a745, #20c997);
    color: white;
    padding: 1rem 2rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(40, 167, 69, 0.3);
}
.confidence-medium {
    background: linear-gradient(90deg, #ffc107, #fd7e14);
    color: #212529;
    padding: 1rem 2rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(255, 193, 7, 0.3);
}
.confidence-low {
    background: linear-gradient(90deg, #dc3545, #e83e8c);
    color: white;
    padding: 1rem 2rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(220, 53, 69, 0.3);
}
.confidence-label {
    font-size: 0.9rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Backend URL
BACKEND_URL = "http://localhost:8001"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_patient_data(file, doc_type, description):
    """Upload patient data to the backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "text/plain")}
        data = {"doc_type": doc_type, "description": description}
        
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Upload Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as e:
        return None, f"Upload error: {str(e)}"

def get_vector_stats():
    """Get vector store statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def submit_text_data(content, doc_type, title):
    """Submit patient data as text"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest-text",
            json={"content": content, "doc_type": doc_type, "title": title},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as e:
        return None, f"Error: {str(e)}"

def call_analyze_api(query):
    """Call the backend analyze endpoint"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return None, "Request timed out. The analysis is taking longer than expected."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Make sure the server is running on port 8000."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def display_results(result):
    """Display analysis results in structured format"""
    
    # Confidence Score - Highlighted at the top, outside analysis box
    confidence = result["confidence_score"]
    confidence_pct = confidence * 100
    
    if confidence >= 0.7:
        confidence_class = "confidence-high"
        confidence_label = "HIGH CONFIDENCE"
        confidence_icon = "✅"
    elif confidence >= 0.5:
        confidence_class = "confidence-medium"
        confidence_label = "MEDIUM CONFIDENCE"
        confidence_icon = "⚠️"
    else:
        confidence_class = "confidence-low"
        confidence_label = "LOW CONFIDENCE"
        confidence_icon = "❗"
    
    st.markdown(f"""
    <div class="{confidence_class}">
        {confidence_icon} {confidence_pct:.0f}% CONFIDENCE
        <div class="confidence-label">{confidence_label} - {"Based on internal knowledge base" if confidence >= 0.5 else "Based on general AI knowledge"}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Viability
    st.markdown("### 🔬 Clinical Insight")
    viability_color = {
        "High": "🟢", "Medium": "🟡", "Low": "🔴"
    }.get(result["clinical_viability"], "⚪")
    
    st.markdown(f"**Viability**: {viability_color} {result['clinical_viability']}")
    
    # Key Evidence
    if result["key_evidence"]:
        st.markdown("**Key Evidence Sources**:")
        for evidence in result["key_evidence"]:
            st.markdown(f"- 📄 {evidence}")
    
    # Risk Flags
    st.markdown("### ⚠️ Risk Flags")
    if result["major_risks"]:
        for risk in result["major_risks"]:
            st.markdown(f"- 🚨 {risk}")
    else:
        st.markdown("- ✅ No major risks identified")
    
    # Market Signal
    st.markdown("### 📈 Market Signal")
    market_color = {
        "Strong": "🟢", "Moderate": "🟡", "Weak": "🔴"
    }.get(result["market_signal"], "⚪")
    
    st.markdown(f"**Market Strength**: {market_color} {result['market_signal']}")
    
    # Final Recommendation
    st.markdown("### 🎯 Final Recommendation")
    
    rec_class = {
        "Proceed": "proceed",
        "Investigate Further": "investigate", 
        "Drop": "drop"
    }.get(result["recommendation"], "investigate")
    
    st.markdown(f"""
    <div class="recommendation-box {rec_class}">
        {result["recommendation"]}
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Explanation
    with st.expander("📋 Detailed Analysis", expanded=False):
        st.markdown(result["explanation"])

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 DRUGVISTA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">AI Co-pilot for Molecular, Clinical, and Market Intelligence</p>', unsafe_allow_html=True)
    
    # Backend status check
    if not check_backend_health():
        st.error("⚠️ Backend server is not running. Please start the backend first:")
        st.code("cd backend && uvicorn main:app --reload --port 8000")
        st.stop()
    
    st.success("✅ Backend connected")
    
    # Sidebar for data upload
    with st.sidebar:
        st.markdown("### 📤 Add Patient Data")
        st.markdown("Add your own data to enhance analysis")
        
        input_method = st.radio(
            "Input Method",
            ["📁 File Upload", "✏️ Text Input"],
            horizontal=True
        )
        
        doc_type = st.selectbox(
            "Document Type",
            ["patient_data", "clinical_trial", "paper", "market"],
            help="Categorize your document for better analysis"
        )
        
        if input_method == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["txt", "csv", "json", "pdf", "docx"],
                help="Upload clinical notes, patient records, or research documents"
            )
            
            description = st.text_input(
                "Description (optional)",
                placeholder="Brief description of the data..."
            )
            
            if st.button("📥 Upload & Index", use_container_width=True):
                if uploaded_file is None:
                    st.warning("Please select a file first")
                else:
                    with st.spinner("Uploading and indexing..."):
                        result, error = upload_patient_data(uploaded_file, doc_type, description)
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif result:
                        st.success(f"✅ {result['message']}")
        
        else:  # Text Input
            title = st.text_input(
                "Title (optional)",
                placeholder="e.g., Patient Case #123"
            )
            
            text_content = st.text_area(
                "Patient Data",
                placeholder="Paste patient notes, clinical observations, or research text here...",
                height=150
            )
            
            if st.button("📥 Add Text Data", use_container_width=True):
                if not text_content or len(text_content.strip()) < 10:
                    st.warning("Please enter at least 10 characters")
                else:
                    with st.spinner("Indexing text..."):
                        result, error = submit_text_data(text_content, doc_type, title)
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif result:
                        st.success(f"✅ {result['message']}")
        
        # Show vector store stats
        st.markdown("---")
        st.markdown("### 📊 Knowledge Base")
        stats = get_vector_stats()
        if stats:
            st.metric("Total Documents", stats.get('total_documents', 0))
        else:
            st.caption("Stats unavailable")
    
    # Input section
    st.markdown("### 🔍 Enter Your Query")
    st.markdown("You can analyze:")
    st.markdown("- **Disease names** (e.g., 'Alzheimer's disease treatment')")
    st.markdown("- **Molecule names** (e.g., 'Aspirin molecular structure')")
    st.markdown("- **Research abstracts** (paste full text)")
    
    # Query input
    query = st.text_area(
        "Query",
        placeholder="Enter disease name, molecule, or paste research abstract...",
        height=100,
        label_visibility="collapsed"
    )
    
    # Example queries
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Alzheimer's Example"):
            query = "Alzheimer's disease treatment options and clinical trial outcomes"
    
    with col2:
        if st.button("💊 Cancer Example"):
            query = "Cancer immunotherapy checkpoint inhibitors market analysis"
    
    with col3:
        if st.button("🔬 Toxicity Example"):
            query = "Drug toxicity screening and safety assessment protocols"
    
    # Analyze button
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if not query or len(query.strip()) < 3:
            st.error("Please enter a query with at least 3 characters.")
            return
        
        # Show loading
        with st.spinner("🔄 Analyzing... This may take 10-30 seconds"):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
            
            # Call API
            result, error = call_analyze_api(query.strip())
        
        # Clear progress bar
        progress_bar.empty()
        
        if error:
            st.error(f"Analysis failed: {error}")
        elif result:
            st.success("✅ Analysis complete!")
            display_results(result)
        else:
            st.error("Unknown error occurred during analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("**DRUGVISTA** - Built for AWS ImpactX Challenge | Powered by GenAI + RAG")

if __name__ == "__main__":
    main()