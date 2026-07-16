import hashlib
import logging
from config import DEBUG_MODE

def get_file_hash(filepath: str) -> str:
    """Returns SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Failed to hash {filepath}: {e}")
        return ""

def setup_logging():
    level = logging.INFO if DEBUG_MODE else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Suppress verbose httpx logs from ollama even in DEBUG_MODE unless explicitly needed
    if DEBUG_MODE:
        logging.getLogger("httpx").setLevel(logging.WARNING)
