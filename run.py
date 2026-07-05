#!/usr/bin/env python3
"""
Launcher script for Medico-Legal Assistant.
Double-click this file or run: python run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bundle_rag.app import run_app

if __name__ == "__main__":
    print("Starting Medico-Legal Assistant...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    run_app()
