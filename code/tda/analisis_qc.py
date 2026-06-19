"""
Análisis topológico del campo de Quantum Computing en arXiv.

Tres análisis:
  1. Agujeros (H1): ¿hay zonas no estudiadas?
  2. Ramas (H0 / clustering): ¿cuáles son las subramas y su centro?
  3. Frontera (edge): artículos del último mes más alejados del núcleo.

Requisitos: pip install ripser numpy scikit-learn
Datos: data/arxiv_qc.npz (generado por collect_arxiv_qc.py)
"""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
QC_NPZ = os.path.join(DATA_DIR, "arxiv_qc.npz")


def load_data():
    """Carga los datos de arXiv QC."""
    data = np.load(QC_NPZ, allow_pickle=True)
    # first_authors is optional (added in a later version of the collection script)
    first_authors = (
        data["first_authors"]
        if "first_authors" in data
        else np.array([""] * len(data["titles"]), dtype=object)
    )
    return {
        "embeddings": data["embeddings"],
        "years": data["years"],
        "months": data["months"],
        "categories": data["categories"],
        "titles": data["titles"],
        "published": data["published"],
        "first_authors": first_authors,
    }


def reducir_dimensiones(embeddings, n_components=50):
    """Reduce dimensionalidad con PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)


def analisis_agujeros(emb_red, n_sample=800, seed=42):
    """
    Detecta agujeros (H1) en el paisaje de QC.
    Submuestrea para que ripser sea tratable.
    """
    from ripser import ripser

    rng = np.random.default_rng(seed)
    n = len(emb_red)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        puntos = emb_red[idx]
    else:
        idx = np.arange(n)
        puntos = emb_red

    resultado = ripser(puntos, maxdim=1)
    dgms = resultado["dgms"]

    h0 = dgms[0]
    h1 = dgms[1]

    # Persistencias H1
    if len(h1) > 0:
        pers_h1 = h1[:, 1] - h1[:, 0]
        top_h1 = np.argsort(pers_h1)[::-1][:10]
    else:
        pers_h1 = np.array([])
        top_h1 = np.array([])

    return {
        "h0": h0,
        "h1": h1,
        "pers_h1": pers_h1,
        "top_h1_indices": top_h1,
        "sample_idx": idx,
    }


def analisis_ramas(embeddings, titles, first_authors=None, n_clusters=8):
    """
    Identifica ramas del campo con clustering jerárquico.
    Devuelve clusters, centroide y paper más central de cada rama.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances

    if first_authors is None:
        first_authors = np.array([""] * len(titles), dtype=object)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward")
    labels = clustering.fit_predict(embeddings)

    ramas = []
    for k in range(n_clusters):
        mask = labels == k
        cluster_emb = embeddings[mask]
        cluster_titles = titles[mask]
        cluster_authors = first_authors[mask]

        # Centroide y paper más cercano al centroide
        centroid = cluster_emb.mean(axis=0)
        dists = pairwise_distances(
            centroid.reshape(1, -1), cluster_emb)[0]
        idx_central = np.argmin(dists)

        ramas.append({
            "label": k,
            "size": int(mask.sum()),
            "paper_central": str(cluster_titles[idx_central]),
            "first_author": str(cluster_authors[idx_central]),
            "dist_central": float(dists[idx_central]),
        })

    return {"labels": labels, "ramas": ramas}


def analisis_frontera(embeddings, titles, published, months,
                      n_recent=None):
    """
    Identifica artículos del último mes en la frontera.
    Frontera = mayor distancia media al resto del corpus.
    """
    from sklearn.metrics import pairwise_distances

    # Último mes: mes máximo presente en los datos
    max_month = months.max()
    max_year = None
    # Usar published para ser más preciso
    recent_mask = months == max_month

    if recent_mask.sum() == 0:
        # Fallback: último mes disponible
        unique_months = np.unique(months)
        if len(unique_months) > 1:
            max_month = unique_months[-1]
            recent_mask = months == max_month

    recent_emb = embeddings[recent_mask]
    recent_titles = titles[recent_mask]

    if len(recent_emb) == 0:
        return {"edge_papers": [], "recent_count": 0}

    # Distancia media de cada paper reciente al corpus completo
    dists = pairwise_distances(recent_emb, embeddings)
    mean_dists = dists.mean(axis=1)

    # Top 10 más alejados = en la frontera
    n_edge = min(10, len(mean_dists))
    edge_idx = np.argsort(mean_dists)[::-1][:n_edge]

    edge_papers = []
    for i in edge_idx:
        edge_papers.append({
            "title": str(recent_titles[i]),
            "mean_dist": float(mean_dists[i]),
        })

    return {
        "edge_papers": edge_papers,
        "recent_count": int(recent_mask.sum()),
        "month": float(max_month),
    }


def main():
    """Ejecuta los tres análisis."""
    print("=== Estructura del campo de Quantum Computing ===\n")
    data = load_data()
    emb = data["embeddings"]
    titles = data["titles"]
    months = data["months"]
    published = data["published"]

    print(f"Artículos cargados: {len(emb)}")
    print(f"Dimensión embeddings: {emb.shape[1]}\n")

    # Reducir dimensiones
    emb_red = reducir_dimensiones(emb, n_components=50)

    # ── 1. Agujeros ──
    print("─── Análisis 1: Agujeros (H1) ───")
    res_h = analisis_agujeros(emb_red)
    h1 = res_h["h1"]
    pers = res_h["pers_h1"]
    print(f"  Características H1: {len(h1)}")
    if len(pers) > 0:
        print(f"  Top 5 persistencias H1: "
              f"{pers[res_h['top_h1_indices'][:5]]}")
        # Ratio señal/ruido: max persistencia vs mediana
        median_p = np.median(pers)
        print(f"  Ratio max/mediana: "
              f"{pers.max() / median_p:.1f}x")
    print()

    # ── 2. Ramas ──
    print("─── Análisis 2: Ramas del campo ───")
    res_r = analisis_ramas(emb_red, titles,
                           first_authors=data.get("first_authors"))
    for rama in sorted(res_r["ramas"], key=lambda x: -x["size"]):
        author = rama["first_author"]
        author_str = f" [{author}]" if author else ""
        print(f"  Rama {rama['label']} "
              f"({rama['size']} papers){author_str}: "
              f"{rama['paper_central'][:70]}...")
    print()

    # ── 3. Frontera ──
    print("─── Análisis 3: Artículos en la frontera ───")
    res_f = analisis_frontera(emb_red, titles, published, months)
    print(f"  Artículos del último mes: {res_f['recent_count']}")
    print(f"  Top papers en el edge:")
    for i, p in enumerate(res_f["edge_papers"][:5], 1):
        print(f"    {i}. {p['title'][:70]}... "
              f"(dist={p['mean_dist']:.3f})")

    return {"agujeros": res_h, "ramas": res_r, "frontera": res_f}


if __name__ == "__main__":
    main()
