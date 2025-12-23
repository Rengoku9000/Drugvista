"""
System test for DRUGVISTA
Verifies all components work correctly
"""
import os
import sys
import requests
import time
import subprocess
import threading

def test_data_files():
    """Test that all data files exist"""
    print("📁 Testing data files...")
    
    required_files = [
        "data/papers/alzheimer_paper_1.txt",
        "data/papers/cancer_immunotherapy_1.txt", 
        "data/papers/drug_toxicity_1.txt",
        "data/clinical_trials/alzheimer_trial_1.txt",
        "data/clinical_trials/cancer_trial_1.txt",
        "data/market/alzheimer_market_1.txt",
        "data/market/immunotherapy_market_1.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All data files present")
    return True

def test_vector_store():
    """Test vector store creation"""
    print("🔍 Testing vector store...")
    
    try:
        # Change to backend directory to find the index files
        original_dir = os.getcwd()
        os.chdir("backend")
        
        sys.path.append(".")
        from vector_store import VectorStore
        
        vs = VectorStore()
        stats = vs.get_stats()
        
        if stats['total_documents'] == 0:
            print("❌ No documents in vector store")
            os.chdir(original_dir)
            return False
        
        # Test search
        results = vs.search("Alzheimer's disease", top_k=3)
        if len(results) == 0:
            print("❌ Search returned no results")
            os.chdir(original_dir)
            return False
        
        print(f"✅ Vector store working ({stats['total_documents']} documents)")
        os.chdir(original_dir)
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def test_backend_api():
    """Test backend API"""
    print("🔧 Testing backend API...")
    
    # Start backend in background
    backend_process = None
    try:
        os.chdir("backend")
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir("..")
        
        # Wait for startup
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code != 200:
            print("❌ Health check failed")
            return False
        
        # Test analyze endpoint
        test_query = {"query": "Alzheimer's disease treatment"}
        response = requests.post("http://localhost:8000/analyze", json=test_query, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Analyze endpoint failed: {response.status_code}")
            return False
        
        result = response.json()
        required_fields = ["clinical_viability", "recommendation", "confidence_score"]
        
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field in response: {field}")
                return False
        
        print("✅ Backend API working")
        return True
        
    except Exception as e:
        print(f"❌ Backend API error: {e}")
        return False
    
    finally:
        if backend_process:
            backend_process.terminate()
            backend_process.wait()

def main():
    print("🧬 DRUGVISTA System Test")
    print("========================")
    
    tests = [
        ("Data Files", test_data_files),
        ("Vector Store", test_vector_store),
        ("Backend API", test_backend_api)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for demo.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)