# check_dependencies.py
import importlib
import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and get its version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def main():
    print("🔍 Checking Disaster Prediction System Dependencies...")
    print("=" * 50)
    
    packages = [
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("joblib", "joblib"),
        ("streamlit", "streamlit"),
        ("requests", "requests"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print("=" * 50)
    if all_installed:
        print("🎉 All dependencies are installed correctly!")
        print("\n🚀 You can now run:")
        print("   python train_model.py    # To train the model")
        print("   streamlit run disaster_app.py  # To launch the app")
    else:
        print("❌ Some dependencies are missing.")
        print("💡 Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()