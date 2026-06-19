"""
Descarga artículos de arXiv sobre Quantum Computing de los últimos 5 años,
calcula embeddings con sentence-transformers y guarda el resultado.

Estrategia: una query general por año (2021-2026), 300 artículos por año,
con backoff exponencial ante rate-limits.  Esto produce ~1500 artículos
distribuidos uniformemente en el tiempo, lo que garantiza diversidad
semántica inter-anual sin saturar la API.

Salida: data/arxiv_qc_5y.npz
"""

import os
import ssl
import sys
import time
from datetime import datetime

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_NPZ = os.path.join(OUTPUT_DIR, "arxiv_qc_5y.npz")

# Una query amplia por año: filtra por fecha directamente en la API
# formato arXiv: submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
BASE_QUERY = 'cat:quant-ph AND ("quantum computing" OR "quantum algorithm" OR "quantum error correction")'
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
MAX_PER_YEAR = 350   # 6 años × 350 ≈ 2100 artículos únicos
DELAY = 5.0          # segundos entre requests (seguro contra 429)


def fetch_year(client, year, max_results, seen_ids):
    """Descarga hasta max_results artículos de un año concreto."""
    import arxiv

    # Rango de fechas para el año
    if year == 2026:
        date_range = "202601010000 TO 202612312359"
    else:
        date_range = f"{year}01010000 TO {year}12312359"

    query = f'{BASE_QUERY} AND submittedDate:[{date_range}]'

    search = arxiv.Search(
        query=query,
        max_results=max_results,
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
                first_author = (
                    result.authors[0].name if result.authors else ""
                )
                papers.append({
                    "title": result.title.replace("\n", " "),
                    "abstract": result.summary.replace("\n", " "),
                    "year": (result.published.year
                             + result.published.month / 12.0),
                    "month": result.published.month,
                    "category": result.primary_category,
                    "published": result.published.isoformat(),
                    "first_author": first_author,
                })
            break  # éxito
        except Exception as e:
            wait = DELAY * (2 ** attempt)
            print(f"    [intento {attempt+1}/4] error: {e}. "
                  f"Esperando {wait:.0f}s...")
            time.sleep(wait)

    return papers


def fetch_arxiv_papers():
    """Descarga artículos año a año para asegurar diversidad temporal."""
    import arxiv

    client = arxiv.Client(
        page_size=100,
        delay_seconds=DELAY,
        num_retries=2,
    )

    seen_ids = set()
    all_papers = []

    for year in YEARS:
        print(f"  Año {year}...")
        papers = fetch_year(client, year, MAX_PER_YEAR, seen_ids)
        all_papers.extend(papers)
        print(f"    → {len(papers)} artículos  "
              f"(total acumulado: {len(all_papers)})")
        time.sleep(DELAY)   # pausa entre años

    print(f"\n  Total artículos únicos: {len(all_papers)}")
    return all_papers

    print(f"  Total artículos únicos: {len(papers)}")
    return papers


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2",
                       batch_size=128):
    """Calcula embeddings con sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"  Calculando embeddings ({len(texts)} textos)...")
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

    print(f"Recopilando artículos de QC en arXiv (años {YEARS[0]}–{YEARS[-1]})...")
    papers = fetch_arxiv_papers()

    if not papers:
        print("No se encontraron artículos.")
        sys.exit(1)

    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = compute_embeddings(texts)

    years         = np.array([p["year"]         for p in papers], dtype=np.float32)
    months        = np.array([p["month"]        for p in papers], dtype=np.float32)
    categories    = np.array([p["category"]     for p in papers], dtype=object)
    titles        = np.array([p["title"]        for p in papers], dtype=object)
    abstracts     = np.array([p["abstract"]     for p in papers], dtype=object)
    published     = np.array([p["published"]    for p in papers], dtype=object)
    first_authors = np.array([p.get("first_author", "") for p in papers], dtype=object)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        years=years,
        months=months,
        categories=categories,
        titles=titles,
        abstracts=abstracts,
        published=published,
        first_authors=first_authors,
    )
    size_mb = os.path.getsize(OUTPUT_NPZ) / 1024 / 1024
    print(f"\nGuardado: {OUTPUT_NPZ}")
    print(f"  Forma embeddings: {embeddings.shape}")
    print(f"  Rango años: {years.min():.2f} – {years.max():.2f}")
    print(f"  Tamaño: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
