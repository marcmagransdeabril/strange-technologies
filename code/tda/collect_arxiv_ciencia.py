"""
Recolección estratificada de arXiv para el estudio multi-campo.

Pregunta: ¿es toda la ciencia publicada en arXiv una única componente
conexa topológica, o hay islas entre disciplinas?

Estrategia de muestreo:
  1. Definir ~28 categorías representativas de toda la ciencia arXiv.
  2. Asignar pesos proporcionales al volumen de submissions 2024-2025
     (fuente: arxiv.org/stats, redondeo de magnitudes relativas).
  3. Con un presupuesto total BUDGET, calcular n_cat = round(BUDGET * w/W).
  4. Para cada categoría, descargar n_cat papers de 2025 con backoff.
  5. Embeber títulos + abstracts con sentence-transformers.
  6. Guardar en data/arxiv_ciencia.npz con etiquetas de campo.

Salida: data/arxiv_ciencia.npz
  - embeddings:  (N, 384) float32
  - titles:      (N,) object
  - categories:  (N,) object   — categoría arXiv (e.g. "cs.LG")
  - fields:      (N,) object   — campo macro (e.g. "Computer Science")
  - years:       (N,) float32
  - months:      (N,) float32
  - published:   (N,) object

Requisitos: pip install arxiv sentence-transformers numpy
"""

import os
import ssl
import sys
import time

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "arxiv_ciencia.npz")

# ─────────────────────────────────────────────────────────────────────
# Categorías y pesos
#
# Pesos basados en las estadísticas públicas de arXiv (submissions 2024):
# cs.LG ~21k/mes, cs.CV ~13k, hep-ph ~5k, quant-ph ~4.5k, etc.
# Las magnitudes son relativas; se normalizarán a la suma.
# Cada categoría representa un subcampo distinto para maximizar la
# diversidad semántica del corpus.
# ─────────────────────────────────────────────────────────────────────
CATEGORIES = {
    # ── Computer Science ──────────────────────────────────────────
    "cs.LG":  9,   # Machine Learning
    "cs.CV":  7,   # Computer Vision
    "cs.CL":  6,   # Computation and Language (NLP)
    "cs.AI":  4,   # Artificial Intelligence
    "cs.RO":  3,   # Robotics
    "cs.SE":  2,   # Software Engineering
    "cs.CR":  2,   # Cryptography & Security
    "cs.DC":  2,   # Distributed Computing
    # ── Physics ───────────────────────────────────────────────────
    "hep-th": 4,   # High Energy Physics — Theory
    "hep-ph": 4,   # High Energy Physics — Phenomenology
    "quant-ph": 5, # Quantum Physics (incl. quantum computing)
    "cond-mat.mes-hall": 3,  # Mesoscale & Nanoscale Physics
    "cond-mat.str-el": 3,    # Strongly Correlated Electrons
    "astro-ph.GA": 3,        # Astrophysics — Galaxies
    "astro-ph.CO": 3,        # Astrophysics — Cosmology
    "gr-qc": 2,   # General Relativity & Quantum Cosmology
    # ── Mathematics ───────────────────────────────────────────────
    "math.AG": 2,  # Algebraic Geometry
    "math.ST": 2,  # Statistics Theory
    "math.OC": 2,  # Optimization & Control
    "math.CO": 2,  # Combinatorics
    "math.NT": 2,  # Number Theory
    # ── Statistics ────────────────────────────────────────────────
    "stat.ML": 4,  # Machine Learning (statistics)
    "stat.ME": 2,  # Methodology
    # ── Biology & Life Sciences ───────────────────────────────────
    "q-bio.QM": 2, # Quantitative Methods
    "q-bio.GN": 2, # Genomics
    # ── Economics & Social Sciences ───────────────────────────────
    "econ.GN": 1,  # General Economics
    "econ.EM": 2,  # Econometrics
    # ── Other ─────────────────────────────────────────────────────
    "physics.soc-ph": 2,  # Physics and Society
    "nlin.CD": 1,          # Nonlinear Sciences — Chaotic Dynamics
}

# Budget total de papers; ajusta según tiempo disponible
BUDGET = 10_000

# Años a cubrir (fecha exacta: 2025-01-01 → 2025-12-31)
YEAR = 2025
DELAY = 5.0   # segundos entre requests (evita 429)


def compute_allocation(categories: dict, budget: int) -> dict:
    """Calcula n_papers por categoría con muestreo proporcional."""
    total_weight = sum(categories.values())
    allocation = {}
    remaining = budget
    cats = list(categories.items())
    for i, (cat, w) in enumerate(cats):
        if i == len(cats) - 1:
            # Última categoría: asignar el resto para que sumen exacto
            allocation[cat] = remaining
        else:
            n = max(30, round(budget * w / total_weight))
            allocation[cat] = n
            remaining -= n
    return allocation


def fetch_category(client, cat: str, n: int, year: int,
                   seen_ids: set) -> list:
    """Descarga hasta n papers de una categoría en un año dado."""
    import arxiv

    date_range = f"{year}01010000 TO {year}12312359"
    query = f"cat:{cat} AND submittedDate:[{date_range}]"

    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for attempt in range(4):
        try:
            for result in client.results(search):
                aid = result.entry_id.split("/")[-1]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                papers.append({
                    "title":    result.title.replace("\n", " "),
                    "abstract": result.summary.replace("\n", " "),
                    "year":     (result.published.year
                                 + result.published.month / 12.0),
                    "month":    float(result.published.month),
                    "category": cat,
                    "published": result.published.isoformat(),
                })
            break
        except Exception as e:
            wait = DELAY * (2 ** attempt)
            print(f"    [intento {attempt+1}/4] {e}. "
                  f"Esperando {wait:.0f}s...")
            time.sleep(wait)

    return papers


def compute_embeddings(texts: list, model_name="all-MiniLM-L6-v2",
                       batch_size=128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  Calculando embeddings ({len(texts)} textos)...")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size,
                       show_progress_bar=True,
                       normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def main():
    try:
        import arxiv  # noqa: F401
    except ImportError:
        print("ERROR: pip install arxiv")
        sys.exit(1)
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print("ERROR: pip install sentence-transformers")
        sys.exit(1)

    allocation = compute_allocation(CATEGORIES, BUDGET)
    total_alloc = sum(allocation.values())

    print("=" * 60)
    print(f"  Recolección estratificada arXiv {YEAR}")
    print(f"  Budget: {BUDGET}  |  Categorías: {len(CATEGORIES)}")
    print(f"  Total asignado: {total_alloc}")
    print("=" * 60)
    print()
    print("Asignación por categoría:")
    for cat, n in allocation.items():
        w = CATEGORIES[cat]
        print(f"  {cat:25s}  peso={w:2d}  n={n:4d}")
    print()

    import arxiv
    client = arxiv.Client(
        page_size=100,
        delay_seconds=DELAY,
        num_retries=3,
    )

    seen_ids: set = set()
    all_papers: list = []

    for cat, n in allocation.items():
        print(f"  [{cat}] descargando {n} papers de {YEAR}...")
        papers = fetch_category(client, cat, n, YEAR, seen_ids)
        all_papers.extend(papers)
        print(f"    → {len(papers)} nuevos  (total: {len(all_papers)})")
        time.sleep(DELAY)

    print(f"\nTotal papers únicos recolectados: {len(all_papers)}")
    print()

    # ── Embeddings ──────────────────────────────────────────────
    texts = [f"{p['title']}. {p['abstract']}" for p in all_papers]
    embeddings = compute_embeddings(texts)

    # ── Guardar ─────────────────────────────────────────────────
    titles    = np.array([p["title"]    for p in all_papers], dtype=object)
    categories = np.array([p["category"] for p in all_papers], dtype=object)
    years     = np.array([p["year"]     for p in all_papers], dtype=np.float32)
    months    = np.array([p["month"]    for p in all_papers], dtype=np.float32)
    published = np.array([p["published"] for p in all_papers], dtype=object)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        titles=titles,
        categories=categories,
        years=years,
        months=months,
        published=published,
    )

    size_mb = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    print(f"\nGuardado: {OUTPUT_NPZ}")
    print(f"  Forma embeddings: {embeddings.shape}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print()

    # ── Resumen por categoría ────────────────────────────────────
    unique_cats, counts = np.unique(categories, return_counts=True)
    print("Papers por categoría (real vs. asignado):")
    for c, n_real in zip(unique_cats, counts):
        n_plan = allocation.get(c, "?")
        print(f"  {c:25s}  real={n_real:4d}  plan={n_plan:4d}")


if __name__ == "__main__":
    main()
