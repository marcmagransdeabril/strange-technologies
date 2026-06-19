"""Configuración de pytest para tests del capítulo tda."""

import os
import sys
from pathlib import Path

# Directorio con los scripts del capítulo (code/tda/)
CODE_DIR = str(Path(__file__).resolve().parent.parent.parent / "code" / "tda")
sys.path.insert(0, CODE_DIR)
