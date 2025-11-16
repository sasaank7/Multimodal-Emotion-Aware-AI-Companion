"""
Main entry point for the Emotion-Aware AI Companion.
Run this script to start the Streamlit application.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

# Import and run
if __name__ == "__main__":
    from streamlit.web import cli as stcli

    # Path to streamlit app
    app_path = str(Path(__file__).parent / "src" / "ui" / "streamlit_app.py")

    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
