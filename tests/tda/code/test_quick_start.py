"""Tests para el ejemplo de TDA."""

import subprocess
import sys
import os

import pytest
import numpy as np

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "code", "tda")
DATA_DIR = os.path.join(CODE_DIR, "data")


def test_script_ejecuta_o_falta_dependencia():
    result = subprocess.run(
        [sys.executable, os.path.join(CODE_DIR, "quick_start.py")],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        if "No module named" in result.stderr or "ModuleNotFoundError" in result.stderr:
            pytest.skip("dependencia no instalada")
        if "FileNotFoundError" in result.stderr:
            pytest.skip("datos no generados (ejecutar collect_*.py)")
    if "no instalado" in result.stdout:
        pytest.skip("ripser no instalado")
    assert "Receta 1" in result.stdout or "Receta 2" in result.stdout


# ── Receta 1: Tres islas (H0) ────────────────────────────────────────


def test_generar_clusters():
    sys.path.insert(0, CODE_DIR)
    from quick_start import generar_clusters

    puntos = generar_clusters(30)
    assert puntos.shape == (90, 2)  # 3 clusters × 30


def test_clusters_tiene_tres_componentes():
    pytest.importorskip("ripser", reason="ripser no instalado")
    sys.path.insert(0, CODE_DIR)
    from quick_start import generar_clusters, homologia_persistente

    puntos = generar_clusters(50, separacion=5.0)
    dgms = homologia_persistente(puntos, maxdim=0)
    # Con separacion=5.0, debe haber al menos 2 muertes tardías (ε > 1.0)
    finitos = dgms[0][np.isfinite(dgms[0][:, 1])]
    muertes = np.sort(finitos[:, 1])[::-1]
    assert len(muertes) >= 2
    assert muertes[0] > 1.0, "Los clusters bien separados deben fusionarse tarde"


# ── Receta 2: Un anillo (H1) ─────────────────────────────────────────


def test_generar_circulo():
    sys.path.insert(0, CODE_DIR)
    from quick_start import generar_circulo

    puntos = generar_circulo(50)
    assert puntos.shape == (50, 2)
    # Los puntos deben estar aproximadamente en el círculo unitario
    distancias = np.sqrt(puntos[:, 0]**2 + puntos[:, 1]**2)
    assert np.mean(np.abs(distancias - 1.0)) < 0.3


def test_circulo_tiene_un_agujero():
    pytest.importorskip("ripser", reason="ripser no instalado")
    sys.path.insert(0, CODE_DIR)
    from quick_start import generar_circulo, homologia_persistente, agujero_principal

    puntos = generar_circulo(200, ruido=0.05)
    diagramas = homologia_persistente(puntos)
    agujero = agujero_principal(diagramas)
    assert agujero is not None
    assert agujero["persistencia"] > 0.5, "El agujero del círculo debería ser muy persistente"


# ── Receta 3: Señal y ruido ──────────────────────────────────────────


def test_persistencia_decrece_con_ruido():
    pytest.importorskip("ripser", reason="ripser no instalado")
    sys.path.insert(0, CODE_DIR)
    from quick_start import persistencia_vs_ruido

    resultados = persistencia_vs_ruido(niveles_ruido=(0.05, 0.3), n=150)
    # Con poco ruido hay más persistencia que con mucho
    assert resultados[0][1] > resultados[1][1], \
        "La persistencia debería decrecer al aumentar el ruido"


# ── Receta 4: Conectoma ──────────────────────────────────────────────


def test_cargar_conectoma():
    pytest.importorskip("networkx", reason="networkx no instalado")
    pytest.importorskip("ripser", reason="ripser no instalado")
    csv_path = os.path.join(DATA_DIR, "elegans_connectome.csv")
    if not os.path.exists(csv_path):
        pytest.skip("datos no generados")

    sys.path.insert(0, CODE_DIR)
    from quick_start import cargar_conectoma
    D, nodes = cargar_conectoma(csv_path)
    assert D.shape[0] == D.shape[1]
    assert D.shape[0] > 200  # ~281 neuronas en la componente principal
    assert len(nodes) == D.shape[0]


def test_conectoma_tiene_ciclos():
    pytest.importorskip("networkx", reason="networkx no instalado")
    pytest.importorskip("ripser", reason="ripser no instalado")
    csv_path = os.path.join(DATA_DIR, "elegans_connectome.csv")
    if not os.path.exists(csv_path):
        pytest.skip("datos no generados")

    sys.path.insert(0, CODE_DIR)
    from quick_start import cargar_conectoma, homologia_persistente
    from ripser import ripser

    D, _ = cargar_conectoma(csv_path)
    resultado = ripser(D, maxdim=1, distance_matrix=True)
    dgms = resultado["dgms"]
    assert len(dgms[1]) > 0, "El conectoma debería tener ciclos"


# ── Receta 5: Libros (viaje del héroe) ───────────────────────────────


def test_cargar_libros():
    npz_path = os.path.join(DATA_DIR, "hero_books.npz")
    if not os.path.exists(npz_path):
        pytest.skip("datos no generados")

    sys.path.insert(0, CODE_DIR)
    from quick_start import cargar_libros
    data = cargar_libros(npz_path)
    assert data["embeddings"].shape[1] == 384
    assert len(data["book_names"]) >= 4
    assert len(data["positions"]) == len(data["book_ids"])


def test_persistencia_libro():
    pytest.importorskip("ripser", reason="ripser no instalado")
    pytest.importorskip("sklearn", reason="scikit-learn no instalado")
    npz_path = os.path.join(DATA_DIR, "hero_books.npz")
    if not os.path.exists(npz_path):
        pytest.skip("datos no generados")

    sys.path.insert(0, CODE_DIR)
    from quick_start import cargar_libros, persistencia_libro

    data = cargar_libros(npz_path)
    # Usar solo el primer libro, submuestreado
    mask = data["book_ids"] == 0
    emb = data["embeddings"][mask][:50]
    pos = data["positions"][mask][:50]

    dgms = persistencia_libro(emb, pos)
    assert len(dgms) >= 2  # Al menos H0 y H1


# ── Receta 6: arXiv ──────────────────────────────────────────────────


def test_cargar_arxiv():
    npz_path = os.path.join(DATA_DIR, "arxiv_tda.npz")
    if not os.path.exists(npz_path):
        pytest.skip("datos no generados")

    sys.path.insert(0, CODE_DIR)
    from quick_start import cargar_arxiv
    data = cargar_arxiv(npz_path)
    assert data["embeddings"].shape[1] == 384
    assert len(data["years"]) == len(data["embeddings"])
    assert data["years"].min() > 2000
