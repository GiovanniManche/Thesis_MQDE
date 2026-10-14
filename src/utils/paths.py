from pathlib import Path
from utils.config import PROJECT_ROOT

def project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
