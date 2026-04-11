"""Configuración compartida de pytest para los ejemplos de código."""

import subprocess
import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-marca tests que requieren dependencias opcionales."""
    pass


@pytest.fixture(autouse=True)
def _clean_quick_start_cache():
    """Evita que quick_start quede cacheado entre directorios distintos."""
    sys.modules.pop("quick_start", None)
    yield
    sys.modules.pop("quick_start", None)
