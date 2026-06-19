"""
Robustez de las ramas de Quantum Computing bajo submuestreo.

Ejecuta N_RUNS subsamples del corpus de 5 años, calcula H0 + enlace simple
en cada uno y mide:
  1. Estabilidad de N_RAMAS (media ± desv. típica).
  2. Coherencia semántica: para cada rama aparecida, calcula el centroide
     medio de todos los runs y la similitud coseno con el centroide de cada
     run individual.  Una similitud alta (> 0.85) indica que la rama es
     estable.
  3. Tabla de los artículos más centrales por rama, con su frecuencia de
     aparición (en qué % de runs ese artículo es el más central de su rama).

Uso:
  python robustez_ramas_qc.py [--npz PATH] [--runs N] [--sample M]

Parámetros por defecto:
  --npz    data/arxiv_qc_5y.npz   (fallback: arxiv_qc.npz si no existe)
  --runs   20
  --sample 2000
"""

import argparse
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"], data["titles"]


def pca_reduce(emb, n_components=50):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(emb)


def bootstrap_threshold(emb_red, n_bootstrap=200, subsample=500, seed=0):
    """Calcula el umbral del 95% de persistencia bajo hipótesis nula."""
    from ripser import ripser
    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(emb_red), min(subsample, len(emb_red)),
                         replace=False)
        h = ripser(emb_red[idx], maxdim=0)["dgms"][0]
        fin = h[np.isfinite(h[:, 1])]
        null_max.append((fin[:, 1] - fin[:, 0]).max() if len(fin) else 0.0)
    return float(np.percentile(null_max, 95))


def rama_clustering(emb_red, thresh):
    """Calcula N_RAMAS y asigna labels con enlace simple."""
    from ripser import ripser
    from sklearn.cluster import AgglomerativeClustering

    dgms = ripser(emb_red, maxdim=0)["dgms"]
    h0 = dgms[0]
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    pers_h0 = finite_h0[:, 1] - finite_h0[:, 0]
    n_ramas = int((pers_h0 > thresh).sum()) + 1

    labels = AgglomerativeClustering(
        n_clusters=n_ramas, linkage="single"
    ).fit_predict(emb_red)
    return n_ramas, labels


def centroides(emb_red, labels, titles, n_ramas):
    """Devuelve (centroide_array, titulo_mas_central) por rama."""
    from sklearn.metrics import pairwise_distances
    cents = []
    cent_titles = []
    for k in range(n_ramas):
        mask = labels == k
        sub = emb_red[mask]
        c = sub.mean(axis=0)
        dists = pairwise_distances(c.reshape(1, -1), sub)[0]
        cents.append(c)
        cent_titles.append(titles[mask][dists.argmin()])
    return np.array(cents), cent_titles


def cosine_sim(a, b):
    """Similitud coseno entre dos vectores."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def hungarian_match(cents_ref, cents_new):
    """
    Asigna cada rama de cents_new a la rama más similar de cents_ref.
    Devuelve permutación: match[i] = índice en cents_ref más cercano a cents_new[i].
    Usa similitud coseno.
    """
    from scipy.optimize import linear_sum_assignment
    sim = np.array([
        [cosine_sim(cents_ref[i], cents_new[j])
         for j in range(len(cents_new))]
        for i in range(len(cents_ref))
    ])
    row_ind, col_ind = linear_sum_assignment(-sim)  # maximizar similitud
    # match[j] = i: rama j de new se asigna a rama i de ref
    match = {}
    for i, j in zip(row_ind, col_ind):
        match[j] = i
    return match, sim[row_ind, col_ind].mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    # Seleccionar fichero de datos
    if args.npz:
        npz_path = args.npz
    else:
        p5y = os.path.join(DATA_DIR, "arxiv_qc_5y.npz")
        p1y = os.path.join(DATA_DIR, "arxiv_qc.npz")
        npz_path = p5y if os.path.exists(p5y) else p1y

    print(f"Cargando datos: {npz_path}")
    emb_raw, titles = load_data(npz_path)
    print(f"  {len(emb_raw)} artículos, dim={emb_raw.shape[1]}")

    print("Reduciendo dimensión (PCA 50)...")
    emb_red = pca_reduce(emb_raw, n_components=50)

    # Umbral bootstrap sobre muestra representativa (una sola vez)
    print("Calculando umbral H0 (bootstrap 200 iteraciones)...")
    thresh = bootstrap_threshold(emb_red, n_bootstrap=200,
                                 subsample=500, seed=0)
    print(f"  Umbral 95% = {thresh:.4f}")

    # Primera pasada para establecer referencia de ramas
    rng0 = np.random.default_rng(0)
    idx0 = rng0.choice(len(emb_red), min(args.sample, len(emb_red)),
                       replace=False)
    n_ramas_ref, labels_ref = rama_clustering(emb_red[idx0], thresh)
    cents_ref, titles_ref = centroides(
        emb_red[idx0], labels_ref, titles[idx0], n_ramas_ref
    )
    print(f"\nReferencia (seed 0): N_RAMAS = {n_ramas_ref}")
    for k, t in enumerate(titles_ref):
        print(f"  Rama {k}: {t[:70]}")

    # N runs de robustez
    print(f"\nEjecutando {args.runs} subsamples (n={args.sample})...")
    all_n_ramas = []
    # acumular sumas de centroides alineados con la referencia
    cent_sum = np.zeros_like(cents_ref)
    cent_count = np.zeros(n_ramas_ref, dtype=int)
    # similitudes de alineación
    align_sims = []
    # contador de artículos más centrales por rama de referencia
    title_counter = [{} for _ in range(n_ramas_ref)]

    for run in range(args.runs):
        rng = np.random.default_rng(run + 1)
        idx = rng.choice(len(emb_red), min(args.sample, len(emb_red)),
                         replace=False)
        sub_emb = emb_red[idx]
        sub_titles = titles[idx]

        n_r, lbl = rama_clustering(sub_emb, thresh)
        all_n_ramas.append(n_r)
        cents_new, titles_new = centroides(sub_emb, lbl, sub_titles, n_r)

        # Alinear con referencia usando Hungarian si mismo n_ramas,
        # si difiere, usar solo las ramas comunes
        if n_r == n_ramas_ref:
            match, mean_sim = hungarian_match(cents_ref, cents_new)
            align_sims.append(mean_sim)
            for j, i in match.items():
                cent_sum[i] += cents_new[j]
                cent_count[i] += 1
                t = titles_new[j]
                title_counter[i][t] = title_counter[i].get(t, 0) + 1
        else:
            # Alineación parcial: asignar cada rama nueva a la ref más similar
            for j in range(n_r):
                sims = [cosine_sim(cents_ref[i], cents_new[j])
                        for i in range(n_ramas_ref)]
                best = int(np.argmax(sims))
                if sims[best] > 0.5:
                    cent_sum[best] += cents_new[j]
                    cent_count[best] += 1
                    t = titles_new[j]
                    title_counter[best][t] = title_counter[best].get(t, 0) + 1

        if (run + 1) % 5 == 0:
            print(f"  run {run+1}/{args.runs}  "
                  f"N_RAMAS={n_r}  "
                  f"(media hasta ahora: "
                  f"{np.mean(all_n_ramas):.1f})")

    # ---------------------------------------------------------------------------
    # Resultados
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTADOS DE ROBUSTEZ")
    print("=" * 70)

    arr = np.array(all_n_ramas)
    print(f"\nN_RAMAS a lo largo de {args.runs} runs:")
    print(f"  Media: {arr.mean():.2f}  ±  {arr.std():.2f}")
    vals, cnts = np.unique(arr, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  N={v}: {c}/{args.runs} runs ({100*c/args.runs:.0f}%)")

    if align_sims:
        print(f"\nSimilitud coseno de alineación Hungarian "
              f"(runs con N_RAMAS={n_ramas_ref}): "
              f"{np.mean(align_sims):.3f} ± {np.std(align_sims):.3f}")

    print(f"\nEstabilidad semántica por rama (similitud coseno "
          f"centroide_run vs centroide_referencia):")

    # Centroide promedio por rama
    for i in range(n_ramas_ref):
        if cent_count[i] == 0:
            print(f"  Rama {i}: sin datos suficientes")
            continue
        cent_avg = cent_sum[i] / cent_count[i]
        sim = cosine_sim(cents_ref[i], cent_avg)
        # artículo más frecuente como centro
        top_title = max(title_counter[i], key=title_counter[i].get)
        freq = title_counter[i][top_title]
        n_runs_with_this_rama = cent_count[i]
        print(f"\n  Rama {i}  (aparece en {n_runs_with_this_rama}/{args.runs} runs)")
        print(f"    Similitud coseno promedio:  {sim:.3f}"
              f"  {'[ESTABLE]' if sim > 0.85 else '[INESTABLE]'}")
        print(f"    Artículo más central (ref): {titles_ref[i][:72]}")
        print(f"    Artículo más frecuente:     {top_title[:72]}")
        print(f"      → aparece como centro en {freq}/{args.runs} runs "
              f"({100*freq/args.runs:.0f}%)")

    print("\n" + "=" * 70)
    print("Conclusión:")
    stable = sum(
        1 for i in range(n_ramas_ref)
        if cent_count[i] > 0
        and cosine_sim(cents_ref[i], cent_sum[i] / cent_count[i]) > 0.85
    )
    print(f"  {stable}/{n_ramas_ref} ramas son semánticamente estables "
          f"(similitud > 0.85) bajo submuestreo.")
    pct_same_n = 100 * (arr == n_ramas_ref).sum() / args.runs
    print(f"  N_RAMAS={n_ramas_ref} en el {pct_same_n:.0f}% de los runs.")


if __name__ == "__main__":
    main()
