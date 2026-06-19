"""
Descarga artículos de arXiv sobre TDA, calcula embeddings
de sus abstracts con sentence-transformers, y guarda el resultado.

Genera un archivo NPZ con:
  - embeddings: (N, 384) float32
  - years:      (N,) float32   — año de publicación
  - categories: (N,) object    — categoría primaria
  - titles:     (N,) object    — título del artículo

Requisitos: pip install arxiv sentence-transformers
Salida: data/arxiv_tda.npz
"""

import os
import ssl
import sys
import time

import numpy as np

# Corporate proxy: disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "arxiv_tda.npz")

# Queries para capturar el campo de TDA
QUERIES = [
    '"topological data analysis"',
    '"persistent homology"',
    '"persistence diagram"',
    '"topological machine learning"',
]

MAX_RESULTS_PER_QUERY = 2000


def fetch_arxiv_papers(queries=QUERIES, max_per_query=MAX_RESULTS_PER_QUERY):
    """Busca artículos en arXiv y devuelve metadatos únicos."""
    import arxiv

    seen_ids = set()
    papers = []

    client = arxiv.Client(
        page_size=200,
        delay_seconds=3.0,
        num_retries=3,
    )

    for query in queries:
        print(f"  Buscando: {query}")
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        count = 0
        for result in client.results(search):
            aid = result.entry_id.split("/")[-1]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            papers.append({
                "title": result.title.replace("\n", " "),
                "abstract": result.summary.replace("\n", " "),
                "year": result.published.year + result.published.month / 12.0,
                "category": result.primary_category,
            })
            count += 1
        print(f"    → {count} nuevos ({len(papers)} total)")

    print(f"  Total artículos únicos: {len(papers)}")
    return papers


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=128):
    """Calcula embeddings con sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"  Calculando embeddings ({len(texts)} textos) ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def main():
    try:
        import arxiv  # noqa: F401
    except ImportError:
        print("ERROR: arxiv no instalado (pip install arxiv)")
        sys.exit(1)
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print("ERROR: sentence-transformers no instalado")
        sys.exit(1)

    print("Recopilando artículos de arXiv sobre TDA...")
    papers = fetch_arxiv_papers()

    if not papers:
        print("No se encontraron artículos.")
        sys.exit(1)

    # Embed title + abstract
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = compute_embeddings(texts)

    years = np.array([p["year"] for p in papers], dtype=np.float32)
    categories = np.array([p["category"] for p in papers], dtype=object)
    titles = np.array([p["title"] for p in papers], dtype=object)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        years=years,
        categories=categories,
        titles=titles,
    )
    size_mb = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    print(f"\nGuardado: {OUTPUT_NPZ}")
    print(f"  Forma embeddings: {embeddings.shape}")
    print(f"  Rango años: {years.min():.1f} - {years.max():.1f}")
    print(f"  Categorías: {len(set(categories))} únicas")
    print(f"  Tamaño: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
