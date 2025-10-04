#!/usr/bin/env python3
"""
Setup script for HMPI Calculator Application
This script helps set up the environment and install dependencies
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages from requirements.txt"""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_streamlit_installation():
    """Check if Streamlit is properly installed"""
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def create_sample_data():
    """Create sample data file if it doesn't exist"""
    sample_file = Path(__file__).parent / "sample_data.csv"
    
    if sample_file.exists():
        print("✅ Sample data file already exists")
        return True
    
    print("📄 Creating sample data file...")
    
    sample_data = """Sample_ID,latitude,longitude,Pb,Cd,Cr,Ni,Cu,Zn,Fe,Mn
GW_001,23.2599,77.4126,0.008,0.002,0.025,0.015,0.8,1.2,0.15,0.08
GW_002,23.2845,77.3256,0.015,0.004,0.048,0.032,1.5,2.8,0.28,0.15
GW_003,23.1976,77.5123,0.032,0.008,0.095,0.068,3.2,5.2,0.65,0.45
GW_004,23.3421,77.2987,0.005,0.001,0.018,0.012,0.5,0.8,0.12,0.06
GW_005,23.2156,77.4567,0.025,0.006,0.075,0.055,2.1,3.5,0.42,0.28"""
    
    try:
        with open(sample_file, 'w') as f:
            f.write(sample_data)
        print("✅ Sample data file created")
        return True
    
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def run_application():
    """Run the Streamlit application"""
    print("\n🚀 Starting HMPI Calculator Application...")
    print("The application will open in your default web browser")
    print("Press Ctrl+C to stop the application")
    
    app_file = Path(__file__).parent / "app.py"
    
    if not app_file.exists():
        print("❌ Error: app.py not found")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file)
        ])
        return True
    
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False

def main():
    """Main setup function"""
    print("🌊 Heavy Metal Pollution Index Calculator - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check Streamlit installation
    if not check_streamlit_installation():
        return False
    
    # Create sample data
    create_sample_data()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python setup.py --run")
    print("2. Or run: streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")
    
    # Ask if user wants to run the application now
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_application()
    else:
        response = input("\nWould you like to start the application now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_application()
    
    return True

def show_help():
    """Show help information"""
    help_text = """
HMPI Calculator Setup Script

Usage:
    python setup.py              # Run setup only
    python setup.py --run        # Run setup and start application
    python setup.py --help       # Show this help

What this script does:
1. Checks Python version compatibility (3.8+)
2. Installs required packages from requirements.txt
3. Verifies Streamlit installation
4. Creates sample data file if needed
5. Optionally starts the application

Requirements:
- Python 3.8 or higher
- pip package manager
- Internet connection for package installation

Files created/modified:
- Installs packages listed in requirements.txt
- Creates sample_data.csv if not present

Troubleshooting:
- If packages fail to install, try: pip install --upgrade pip
- For permission issues, try: pip install --user -r requirements.txt
- Make sure you're in the correct directory with all project files

For more information, see README.md
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        main()