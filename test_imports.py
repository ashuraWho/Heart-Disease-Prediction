# Test Script to verify Refactoring

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
sys.path.append(str(NOTEBOOKS_DIR))

def test_imports():
    print("Testing Imports...")
    try:
        import shared_utils
        print("✅ shared_utils imported")
        from shared_utils import setup_environment, console, ARTIFACTS_DIR
        print(f"✅ shared_utils attributes accessed (Artifacts: {ARTIFACTS_DIR})")
        setup_environment()
        print("✅ setup_environment() called")
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    print("All basic import tests passed.")
