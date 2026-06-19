"""Tests para code/tda/umap_reduccion.py (Receta 7)."""

import subprocess
import sys
import os

import pytest
import numpy as np

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "code", "tda")


def test_script_ejecuta_o_falta_dependencia():
    """El script debe ejecutar sin error (o saltar si falta umap)."""
    result = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "umap_reduccion.py")],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        if "No module named" in result.stderr or "ModuleNotFoundError" in result.stderr:
            pytest.skip("dependencia no instalada (umap-learn, ripser, etc.)")
        pytest.fail(f"Script falló:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    assert "Receta 7" in result.stdout


def test_generar_toro():
    """Genera un toro con la forma correcta."""
    sys.path.insert(0, CODE_DIR)
    from umap_reduccion import generar_toro

    puntos, toro_3d, theta, phi = generar_toro(n=100, dim_ambiente=10, seed=0)
    assert puntos.shape == (100, 10)
    assert theta.shape == (100,)
    # Todos los ángulos en [0, 2π)
    assert np.all(theta >= 0) and np.all(theta < 2 * np.pi)


def test_reducir_pca():
    """PCA reduce correctamente a n_components."""
    pytest.importorskip("sklearn")
    sys.path.insert(0, CODE_DIR)
    from umap_reduccion import generar_toro, reducir_pca

    puntos, toro_3d, theta, phi = generar_toro(n=50, dim_ambiente=10, seed=0)
    red = reducir_pca(puntos, n_components=3)
    assert red.shape == (50, 3)


def test_reducir_umap():
    """UMAP reduce correctamente (o se salta si no instalado)."""
    pytest.importorskip("umap", reason="umap-learn no instalado")
    sys.path.insert(0, CODE_DIR)
    from umap_reduccion import generar_toro, reducir_umap

    puntos, toro_3d, theta, phi = generar_toro(n=50, dim_ambiente=10, seed=0)
    red = reducir_umap(puntos, n_components=3)
    assert red.shape == (50, 3)


def test_calcular_h1():
    """Calcula H1 sobre un círculo simple."""
    pytest.importorskip("ripser", reason="ripser no instalado")
    sys.path.insert(0, CODE_DIR)
    from umap_reduccion import calcular_h1

    # Círculo: debe tener 1 ciclo persistente en H1
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    puntos = np.column_stack([np.cos(theta), np.sin(theta)])
    dgms = calcular_h1(puntos, maxdim=1)
    assert len(dgms) >= 2  # H0 y H1
    h1 = dgms[1]
    assert len(h1) > 0
    pers = h1[:, 1] - h1[:, 0]
    assert pers.max() > 0.5  # ciclo persistente


def test_contar_ciclos_persistentes():
    """Cuenta ciclos correctamente con umbral relativo."""
    sys.path.insert(0, CODE_DIR)
    from umap_reduccion import contar_ciclos_persistentes

    # Diagrama fake con 1 ciclo persistente (pers=1.4) y 3 de ruido
    dgms = [
        None,  # H0 placeholder
        np.array([[0.1, 1.5], [0.2, 0.3], [0.1, 0.25], [0.3, 0.4]]),
    ]
    n, pers = contar_ciclos_persistentes(dgms)
    assert n == 1  # solo el de persistencia 1.4 supera 0.3 * 1.4 = 0.42
