"""
Entry point for Document Intelligence Assistant
"""
import sys

# Force utf-8 for Windows console - must be done before any other imports
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from cli import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
