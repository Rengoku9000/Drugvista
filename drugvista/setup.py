"""
Setup script for DRUGVISTA
Run this to initialize the system
"""
import os
import subprocess
import sys

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_embeddings():
    """Create vector embeddings"""
    print("Creating vector embeddings...")
    try:
        os.chdir("backend")
        subprocess.check_call([sys.executable, "embeddings.py"])
        os.chdir("..")
        print("✅ Vector embeddings created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create embeddings: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🧬 DRUGVISTA Setup")
    print("==================")
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the drugvista directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create embeddings
    if not create_embeddings():
        return
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Start backend: cd backend && uvicorn main:app --reload --port 8000")
    print("2. Start frontend: cd frontend && streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main()