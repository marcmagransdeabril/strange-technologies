"""
Análisis Topológico de Datos: homología persistente.

Seis recetas que demuestran TDA de lo simple a lo complejo:
  1. Tres islas (clusters) — H0, componentes conexas
  2. Un anillo — H1, detección de agujeros
  3. Señal y ruido — estabilidad frente a perturbaciones
  4. Conectoma de C. elegans — topología de un cerebro
  5. Viaje del héroe — arcos narrativos circulares en libros clásicos
  6. Paisaje de investigación — artículos de arXiv sobre TDA

Requisitos: pip install ripser persim networkx numpy matplotlib scikit-learn gudhi
Datos: code/tda/data/ (generados por collect_*.py)
"""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Receta 1: Tres islas (H0) ────────────────────────────────────────


def generar_clusters(n_por_cluster=50, separacion=3.0, seed=42):
    """Genera 3 clusters gaussianos en 2D."""
    rng = np.random.default_rng(seed)
    centros = np.array([[0, 0], [separacion, 0], [separacion / 2, separacion]])
    puntos = np.vstack([
        rng.normal(loc=c, scale=0.3, size=(n_por_cluster, 2))
        for c in centros
    ])
    return puntos


def clusters_main():
    """Receta 1: componentes conexas con tres clusters."""
    from ripser import ripser

    print("=== Receta 1: Tres islas (H0) ===")
    puntos = generar_clusters()
    print(f"Puntos: {len(puntos)} (3 clusters de 50)")

    resultado = ripser(puntos, maxdim=0)
    dgms = resultado["dgms"]
    print(f"H0 (componentes): {len(dgms[0])}")

    # Las dos muertes más tardías: cuándo se fusionan los clusters
    finitos = dgms[0][np.isfinite(dgms[0][:, 1])]
    if len(finitos) > 0:
        muertes = np.sort(finitos[:, 1])[::-1]
        print(f"Las 2 fusiones más tardías: ε = {muertes[0]:.2f}, {muertes[1]:.2f}")


# ── Receta 2: Un anillo (H1) ─────────────────────────────────────────


def generar_circulo(n=100, ruido=0.1, seed=42):
    """Genera puntos sobre un círculo con ruido gaussiano."""
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.normal(0, ruido, (n, 2))
    return np.column_stack([np.cos(theta), np.sin(theta)]) + r


def homologia_persistente(puntos, maxdim=1):
    """Calcula la homología persistente con ripser."""
    from ripser import ripser
    resultado = ripser(puntos, maxdim=maxdim)
    return resultado["dgms"]


def agujero_principal(diagramas):
    """Encuentra la característica H1 más persistente."""
    if len(diagramas) < 2 or len(diagramas[1]) == 0:
        return None
    persistencias = diagramas[1][:, 1] - diagramas[1][:, 0]
    idx = np.argmax(persistencias)
    return {
        "nacimiento": float(diagramas[1][idx, 0]),
        "muerte": float(diagramas[1][idx, 1]),
        "persistencia": float(persistencias[idx]),
    }


def main():
    try:
        from ripser import ripser  # noqa: F401
    except ImportError:
        print("ripser no instalado (pip install ripser)")
        return

    clusters_main()
    print()

    puntos = generar_circulo()
    diagramas = homologia_persistente(puntos)

    print("=== Receta 2: Un anillo (H1) ===")
    print(f"Puntos: {len(puntos)}")
    print(f"H0 (componentes): {len(diagramas[0])}")
    print(f"H1 (agujeros):    {len(diagramas[1])}")

    agujero = agujero_principal(diagramas)
    if agujero:
        print(f"\nAgujero principal:")
        print(f"  Nace en:      {agujero['nacimiento']:.3f}")
        print(f"  Muere en:     {agujero['muerte']:.3f}")
        print(f"  Persistencia: {agujero['persistencia']:.3f}")

    print()
    ruido_main()
    print()
    conectoma_main()
    print()
    heroes_main()
    print()
    arxiv_main()


# ── Receta 3: Señal y ruido ──────────────────────────────────────────


def persistencia_vs_ruido(niveles_ruido=(0.05, 0.15, 0.3, 0.5), n=200, seed=42):
    """Calcula la persistencia máxima en H1 del círculo a varios niveles de ruido."""
    resultados = []
    for sigma in niveles_ruido:
        puntos = generar_circulo(n=n, ruido=sigma, seed=seed)
        dgms = homologia_persistente(puntos, maxdim=1)
        agujero = agujero_principal(dgms)
        pers = agujero["persistencia"] if agujero else 0.0
        resultados.append((sigma, pers))
    return resultados


def generar_bucle_fourier(n=200, dim=2, K=5, ruido=0.05, seed=42):
    """Genera un bucle cerrado aleatorio en ℝ^dim mediante series de Fourier.

    Cada coordenada es una combinación de K armónicos con coeficientes
    aleatorios decayendo como 1/k (suavidad). El resultado es una curva
    cerrada (periódica) con forma aleatoria distinta en cada dimensión.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    puntos = np.zeros((n, dim))
    for d in range(dim):
        for k in range(1, K + 1):
            a = rng.standard_normal() / k
            b = rng.standard_normal() / k
            puntos[:, d] += a * np.cos(k * t) + b * np.sin(k * t)
    # Añadir ruido gaussiano
    puntos += rng.normal(0, ruido, (n, dim))
    return puntos


def ruido_main():
    """Receta 3: efecto del ruido y de la dimensión sobre la persistencia."""
    print("=== Receta 3: Señal, ruido e invarianza ===")
    print("-- Efecto del ruido --")
    resultados = persistencia_vs_ruido()
    for sigma, pers in resultados:
        barra = "█" * int(pers * 30)
        print(f"  σ = {sigma:.2f}  →  persistencia H1 = {pers:.3f}  {barra}")
    print("\n-- Bucles aleatorios en distintas dimensiones --")
    for dim in [2, 3, 4, 5]:
        pts = generar_bucle_fourier(n=200, dim=dim, ruido=0.05, seed=100 + dim * 7)
        dgms = homologia_persistente(pts, maxdim=1)
        ag = agujero_principal(dgms)
        pers = ag["persistencia"] if ag else 0.0
        print(f"  ℝ^{dim}  →  persistencia H1 = {pers:.3f}")


# ── Receta 4: Conectoma de C. elegans ────────────────────────────────



def cargar_conectoma(path=None):
    """Carga el conectoma y devuelve una matriz de distancia."""
    import csv
    import networkx as nx

    if path is None:
        path = os.path.join(DATA_DIR, "elegans_connectome.csv")

    G = nx.Graph()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n1, n2 = row["neuron1"], row["neuron2"]
            w = int(row["weight"])
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += w
            else:
                G.add_edge(n1, n2, weight=w)

    # Distancia = inversa del peso (más sinapsis → más cerca)
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]

    # Componente conexa más grande
    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()

    # Matriz de distancias (camino más corto)
    nodes = sorted(G.nodes())
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    n = len(nodes)
    D = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            D[i, j] = dist[u].get(v, np.inf)

    return D, nodes


def conectoma_main():
    """Receta 4: topología del conectoma de C. elegans."""
    from ripser import ripser

    print("=== Receta 4: Conectoma de C. elegans ===")
    D, nodes = cargar_conectoma()
    print(f"Neuronas (componente principal): {len(nodes)}")

    resultado = ripser(D, maxdim=1, distance_matrix=True)
    diagramas = resultado["dgms"]
    print(f"H0 (componentes): {len(diagramas[0])}")
    print(f"H1 (ciclos):      {len(diagramas[1])}")

    # Top 5 ciclos más persistentes
    if len(diagramas[1]) > 0:
        pers = diagramas[1][:, 1] - diagramas[1][:, 0]
        top = np.argsort(pers)[::-1][:5]
        print("\nTop 5 ciclos más persistentes:")
        for rank, idx in enumerate(top, 1):
            print(f"  {rank}. nace={diagramas[1][idx,0]:.3f}, "
                  f"muere={diagramas[1][idx,1]:.3f}, "
                  f"persistencia={pers[idx]:.3f}")


# ── Receta 5: El viaje del héroe ─────────────────────────────────────


def cargar_libros(path=None):
    """Carga embeddings pre-calculados de libros clásicos."""
    if path is None:
        path = os.path.join(DATA_DIR, "hero_books.npz")

    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "positions": data["positions"],
        "book_ids": data["book_ids"],
        "book_names": [str(n) for n in data["book_names"]],
    }


def persistencia_libro(embeddings, positions, maxdim=1, n_components=50):
    """Calcula persistencia de la trayectoria narrativa de un libro."""
    from ripser import ripser
    from sklearn.decomposition import PCA

    # Reducir dimensión con PCA antes de Rips
    n_comp = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    emb_red = PCA(n_components=n_comp).fit_transform(embeddings)

    # Añadir posición narrativa como dimensión extra (escalada)
    pos_col = positions.reshape(-1, 1) * 2.0
    puntos = np.hstack([emb_red, pos_col])

    resultado = ripser(puntos, maxdim=maxdim)
    return resultado["dgms"]


def heroes_main():
    """Receta 5: detectar arcos narrativos circulares."""
    print("=== Receta 5: El viaje del héroe ===")
    data = cargar_libros()
    names = data["book_names"]

    print(f"Libros: {names}")
    print(f"Total ventanas de texto: {len(data['embeddings'])}")
    print()

    for i, name in enumerate(names):
        mask = data["book_ids"] == i
        emb = data["embeddings"][mask]
        pos = data["positions"][mask]

        # Submuestrear si hay demasiados puntos
        if len(emb) > 300:
            idx = np.linspace(0, len(emb) - 1, 300, dtype=int)
            emb, pos = emb[idx], pos[idx]

        dgms = persistencia_libro(emb, pos)

        max_pers = 0.0
        if len(dgms) > 1 and len(dgms[1]) > 0:
            pers = dgms[1][:, 1] - dgms[1][:, 0]
            max_pers = float(np.max(pers))

        print(f"  {name:20s}  H1={len(dgms[1]):3d} features, "
              f"max persistencia={max_pers:.3f}")


# ── Receta 6: Paisaje de investigación en arXiv ──────────────────────


def cargar_arxiv(path=None):
    """Carga embeddings pre-calculados de artículos de arXiv."""
    if path is None:
        path = os.path.join(DATA_DIR, "arxiv_tda.npz")

    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "years": data["years"],
        "categories": data["categories"],
        "titles": data["titles"],
    }


def arxiv_main():
    """Receta 6: topología del paisaje de investigación en TDA."""
    from ripser import ripser

    print("=== Receta 6: Paisaje de investigación ===")
    data = cargar_arxiv()
    emb = data["embeddings"]
    years = data["years"]
    print(f"Artículos: {len(emb)}")
    print(f"Rango: {years.min():.0f} - {years.max():.0f}")

    # Submuestrear para que Rips sea tratable
    rng = np.random.default_rng(42)
    n_sample = min(500, len(emb))
    idx = rng.choice(len(emb), n_sample, replace=False)
    emb_sub = emb[idx]
    years_sub = years[idx]

    # Reducir dimensión con PCA
    from sklearn.decomposition import PCA
    n_comp = min(50, emb_sub.shape[0] - 1)
    emb_red = PCA(n_components=n_comp).fit_transform(emb_sub)

    # Añadir año normalizado como dimensión extra
    year_norm = (years_sub - years_sub.min()) / (years_sub.max() - years_sub.min())
    puntos = np.hstack([emb_red, year_norm.reshape(-1, 1)])

    resultado = ripser(puntos, maxdim=1)
    dgms = resultado["dgms"]
    print(f"H0 (clusters): {len(dgms[0])}")
    print(f"H1 (ciclos):   {len(dgms[1])}")

    if len(dgms[1]) > 0:
        pers = dgms[1][:, 1] - dgms[1][:, 0]
        top = np.argsort(pers)[::-1][:3]
        print("\nTop 3 ciclos en el paisaje de investigación:")
        for rank, i in enumerate(top, 1):
            print(f"  {rank}. nace={dgms[1][i,0]:.3f}, "
                  f"muere={dgms[1][i,1]:.3f}, "
                  f"persistencia={pers[i]:.3f}")


# ── Generación de figuras ────────────────────────────────────────────

DIAGRAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "diagrams", "tda")


def _setup_style():
    """Estilo limpio para figuras del libro (6×9 pulgadas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def _null_landscape_95(puntos, lim, eps_range, n_perm=100, dim=0):
    """Calcula el percentil 95 de λ_1 bajo H0 (puntos uniformes)."""
    from ripser import ripser as _ripser
    rng_perm = np.random.default_rng(0)
    bbox_min = puntos.min(axis=0)
    bbox_max = puntos.max(axis=0)
    null_landscapes = np.zeros((n_perm, len(eps_range)))
    for p_idx in range(n_perm):
        pts_null = rng_perm.uniform(bbox_min, bbox_max, size=(len(puntos), 2))
        dgms_null = _ripser(pts_null, maxdim=dim)["dgms"][dim]
        fin_null = dgms_null[np.isfinite(dgms_null[:, 1])]
        if len(fin_null) == 0:
            continue
        tiendas_null = np.zeros((len(fin_null), len(eps_range)))
        for j in range(len(fin_null)):
            bn, dn = fin_null[j]
            mid_n = (bn + dn) / 2.0
            half_n = (dn - bn) / 2.0
            tiendas_null[j] = np.maximum(0, half_n - np.abs(eps_range - mid_n))
        null_landscapes[p_idx] = np.sort(tiendas_null, axis=0)[-1]
    return np.percentile(null_landscapes, 95, axis=0)


def fig_tres_islas(out_dir=None):
    """Genera la figura de Receta 1: datos + diagrama + barcode + landscape."""
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    puntos = generar_clusters()
    dgms = homologia_persistente(puntos, maxdim=0)

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.2))
    ax_pts, ax_dgm = axes[0]
    ax_bar, ax_land = axes[1]

    # Panel superior izquierdo: nube de puntos
    n = 50
    colores = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, c in enumerate(colores):
        ax_pts.scatter(puntos[i * n:(i + 1) * n, 0],
                       puntos[i * n:(i + 1) * n, 1],
                       s=12, c=c, alpha=0.7, edgecolors="none")
    ax_pts.set_title("Nube de puntos")
    ax_pts.set_aspect("equal")
    ax_pts.set_xlabel("$x$")
    ax_pts.set_ylabel("$y$")

    # Panel superior derecho: diagrama de persistencia H0
    h0 = dgms[0]
    finitos = h0[np.isfinite(h0[:, 1])]
    infinitos = h0[~np.isfinite(h0[:, 1])]
    lim = max(finitos[:, 1].max() * 1.1, 0.5)
    ax_dgm.scatter(finitos[:, 0], finitos[:, 1],
                   s=12, c="#888888", alpha=0.5,
                   edgecolors="none", label="intra-cluster")
    # Marcar las 2 fusiones más tardías (inter-cluster)
    muertes = np.sort(finitos[:, 1])[::-1]
    for k in range(min(2, len(muertes))):
        mask = np.isclose(finitos[:, 1], muertes[k])
        ax_dgm.scatter(finitos[mask, 0], finitos[mask, 1],
                       s=50, c="red", marker="*", zorder=5,
                       label=f"fusión {k + 1} (ε={muertes[k]:.1f})")
    # Marcar la componente inmortal (muerte=∞) en el borde superior
    if len(infinitos) > 0:
        y_inf = lim * 0.97
        ax_dgm.scatter(infinitos[:, 0], [y_inf] * len(infinitos),
                       s=50, c="red", marker="*", zorder=5,
                       label="∞ (nunca muere)")
        ax_dgm.annotate("", xy=(infinitos[0, 0], lim),
                        xytext=(infinitos[0, 0], y_inf),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax_dgm.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
    ax_dgm.set_xlim(-0.05, lim)
    ax_dgm.set_ylim(-0.05, lim)
    ax_dgm.set_xlabel("Nacimiento")
    ax_dgm.set_ylabel("Muerte")
    ax_dgm.set_title("Diagrama $H_0$")
    ax_dgm.legend(fontsize=5, loc="lower right")

    # Panel inferior izquierdo: código de barras
    # Ordenar por muerte descendente para visualización clara
    orden = np.argsort(finitos[:, 1])[::-1]
    n_barras = min(25, len(orden))  # Mostrar las 25 más persistentes
    # Añadir la barra infinita primero
    if len(infinitos) > 0:
        ax_bar.plot([infinitos[0, 0], lim], [0, 0], c="red", lw=1.5,
                    solid_capstyle="butt")
        ax_bar.annotate("→∞", xy=(lim, 0), fontsize=5, color="red",
                        va="center")
        offset = 1
    else:
        offset = 0
    for i in range(n_barras):
        idx = orden[i]
        nac, mue = finitos[idx]
        color = "red" if i < 2 else "#888888"
        lw = 1.5 if i < 2 else 0.7
        ax_bar.plot([nac, mue], [i + offset, i + offset],
                    c=color, lw=lw, solid_capstyle="butt")
    ax_bar.set_xlabel("$\\varepsilon$")
    ax_bar.set_ylabel("Componente")
    ax_bar.set_title("Código de barras $H_0$")
    ax_bar.set_yticks([])
    ax_bar.invert_yaxis()

    # Panel inferior derecho: persistence landscape (λ_k)
    eps_range = np.linspace(0, lim, 300)
    # Incluir todos los pares: finitos + inmortales (con muerte truncada a lim)
    todos = np.vstack([finitos, np.column_stack([
        infinitos[:, 0], np.full(len(infinitos), lim)
    ])]) if len(infinitos) > 0 else finitos
    tiendas = np.zeros((len(todos), len(eps_range)))
    for i in range(len(todos)):
        b, d = todos[i]
        mid = (b + d) / 2.0
        half = (d - b) / 2.0
        tent = np.maximum(0, half - np.abs(eps_range - mid))
        tiendas[i] = tent
    # λ_k es el k-ésimo máximo pointwise
    tiendas_sorted = np.sort(tiendas, axis=0)[::-1]
    colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
    for k in range(min(3, len(tiendas_sorted))):
        ax_land.plot(eps_range, tiendas_sorted[k],
                     c=colores_land[k], lw=1.0,
                     label=f"$\\lambda_{{{k + 1}}}$")
    # Umbral 95% bajo H0
    ci_95 = _null_landscape_95(puntos, lim, eps_range)
    ax_land.plot(eps_range, ci_95, c="#888888", lw=0.8, ls="--",
                 label="umbral 95%")
    ax_land.set_xlabel("$\\varepsilon$")
    ax_land.set_ylabel("$\\lambda_k(\\varepsilon)$")
    ax_land.set_title("Paisaje $H_0$")
    ax_land.legend(fontsize=5, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "tres-islas.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_tres_islas_null(out_dir=None):
    """Genera la figura de control: nube uniforme + diagrama + barcode + landscape."""
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    from ripser import ripser as _ripser

    # Nube uniforme con el mismo número de puntos y bounding box
    # que los tres clusters originales
    puntos_orig = generar_clusters()
    bbox_min = puntos_orig.min(axis=0)
    bbox_max = puntos_orig.max(axis=0)
    rng = np.random.default_rng(42)
    puntos = rng.uniform(bbox_min, bbox_max, size=(len(puntos_orig), 2))

    dgms = _ripser(puntos, maxdim=0)["dgms"]

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.2))
    ax_pts, ax_dgm = axes[0]
    ax_bar, ax_land = axes[1]

    # Panel superior izquierdo: nube de puntos uniforme
    ax_pts.scatter(puntos[:, 0], puntos[:, 1],
                   s=12, c="#888888", alpha=0.7, edgecolors="none")
    ax_pts.set_title("Nube uniforme (sin estructura)")
    ax_pts.set_aspect("equal")
    ax_pts.set_xlabel("$x$")
    ax_pts.set_ylabel("$y$")

    # Panel superior derecho: diagrama de persistencia H0
    h0 = dgms[0]
    finitos = h0[np.isfinite(h0[:, 1])]
    infinitos = h0[~np.isfinite(h0[:, 1])]
    lim = max(finitos[:, 1].max() * 1.1, 0.5)
    ax_dgm.scatter(finitos[:, 0], finitos[:, 1],
                   s=12, c="#888888", alpha=0.5, edgecolors="none")
    # Marcar la componente inmortal (muerte=∞)
    if len(infinitos) > 0:
        y_inf = lim * 0.97
        ax_dgm.scatter(infinitos[:, 0], [y_inf] * len(infinitos),
                       s=50, c="red", marker="*", zorder=5,
                       label="∞ (nunca muere)")
        ax_dgm.annotate("", xy=(infinitos[0, 0], lim),
                        xytext=(infinitos[0, 0], y_inf),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
        ax_dgm.legend(fontsize=5, loc="lower right")
    ax_dgm.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
    ax_dgm.set_xlim(-0.05, lim)
    ax_dgm.set_ylim(-0.05, lim)
    ax_dgm.set_xlabel("Nacimiento")
    ax_dgm.set_ylabel("Muerte")
    ax_dgm.set_title("Diagrama $H_0$")

    # Panel inferior izquierdo: código de barras
    # Añadir la barra infinita primero
    if len(infinitos) > 0:
        ax_bar.plot([infinitos[0, 0], lim], [0, 0], c="red", lw=1.5,
                    solid_capstyle="butt")
        ax_bar.annotate("→∞", xy=(lim, 0), fontsize=5, color="red",
                        va="center")
        offset = 1
    else:
        offset = 0
    orden = np.argsort(finitos[:, 1])[::-1]
    n_barras = min(25, len(orden))
    for i in range(n_barras):
        idx = orden[i]
        nac, mue = finitos[idx]
        ax_bar.plot([nac, mue], [i + offset, i + offset], c="#888888", lw=0.7,
                    solid_capstyle="butt")
    ax_bar.set_xlabel("$\\varepsilon$")
    ax_bar.set_ylabel("Componente")
    ax_bar.set_title("Código de barras $H_0$")
    ax_bar.set_yticks([])
    ax_bar.invert_yaxis()

    # Panel inferior derecho: persistence landscape
    eps_range = np.linspace(0, lim, 300)
    # Incluir inmortales con muerte truncada a lim
    todos = np.vstack([finitos, np.column_stack([
        infinitos[:, 0], np.full(len(infinitos), lim)
    ])]) if len(infinitos) > 0 else finitos
    tiendas = np.zeros((len(todos), len(eps_range)))
    for i in range(len(todos)):
        b, d = todos[i]
        mid = (b + d) / 2.0
        half = (d - b) / 2.0
        tent = np.maximum(0, half - np.abs(eps_range - mid))
        tiendas[i] = tent
    tiendas_sorted = np.sort(tiendas, axis=0)[::-1]
    colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
    for k in range(min(3, len(tiendas_sorted))):
        ax_land.plot(eps_range, tiendas_sorted[k],
                     c=colores_land[k], lw=1.0,
                     label=f"$\\lambda_{{{k + 1}}}$")
    # Umbral 95% bajo H0 (mismo null model)
    ci_95 = _null_landscape_95(puntos, lim, eps_range)
    ax_land.plot(eps_range, ci_95, c="#888888", lw=0.8, ls="--",
                 label="umbral 95%")
    ax_land.set_xlabel("$\\varepsilon$")
    ax_land.set_ylabel("$\\lambda_k(\\varepsilon)$")
    ax_land.set_title("Paisaje $H_0$")
    ax_land.legend(fontsize=5, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "tres-islas-null.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_anillo(out_dir=None):
    """Genera la figura de Receta 2: datos + diagramas H0/H1 + landscapes."""
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    puntos = generar_circulo(n=100, ruido=0.1)
    dgms = homologia_persistente(puntos, maxdim=1)

    fig, axes = plt.subplots(2, 3, figsize=(5.0, 3.8))
    ax_pts, ax_h0, ax_h1 = axes[0]
    ax_empty, ax_land0, ax_land1 = axes[1]

    # Panel superior izquierdo: nube de puntos
    ax_pts.scatter(puntos[:, 0], puntos[:, 1],
                   s=12, c="#2196F3", alpha=0.7, edgecolors="none")
    ax_pts.set_title("Nube de puntos")
    ax_pts.set_aspect("equal")
    ax_pts.set_xlabel("$x$")
    ax_pts.set_ylabel("$y$")

    # Panel superior central: diagrama H0
    h0 = dgms[0]
    h0_fin = h0[np.isfinite(h0[:, 1])]
    h0_inf = h0[~np.isfinite(h0[:, 1])]
    lim = 2.5
    if len(h0_fin) > 0:
        ax_h0.scatter(h0_fin[:, 0], h0_fin[:, 1],
                      s=12, c="#888888", alpha=0.6,
                      edgecolors="none", label="$H_0$")
    # Marcar la componente inmortal (muerte=∞) en el borde superior
    if len(h0_inf) > 0:
        y_inf = lim * 0.97
        ax_h0.scatter(h0_inf[:, 0], [y_inf] * len(h0_inf),
                      s=60, c="red", marker="*", zorder=5,
                      label="∞ (nunca muere)")
        ax_h0.annotate("", xy=(h0_inf[0, 0], lim),
                       xytext=(h0_inf[0, 0], y_inf),
                       arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax_h0.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
    ax_h0.set_xlim(-0.05, lim)
    ax_h0.set_ylim(-0.05, lim)
    ax_h0.set_xlabel("Nacimiento")
    ax_h0.set_ylabel("Muerte")
    ax_h0.set_title("Diagrama $H_0$")
    ax_h0.legend(fontsize=6, loc="lower right")

    # Panel superior derecho: diagrama H1
    h1 = dgms[1]
    h1_fin = h1[np.isfinite(h1[:, 1])]
    if len(h1_fin) > 0:
        ax_h1.scatter(h1_fin[:, 0], h1_fin[:, 1],
                      s=12, c="#E91E63", alpha=0.6,
                      edgecolors="none", label="$H_1$")
    # Marcar el agujero principal
    agujero = agujero_principal(dgms)
    if agujero:
        ax_h1.scatter(agujero["nacimiento"], agujero["muerte"],
                      s=60, c="#E91E63", marker="*", zorder=5,
                      label=f"agujero (p={agujero['persistencia']:.2f})")
    ax_h1.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
    ax_h1.set_xlabel("Nacimiento")
    ax_h1.set_ylabel("Muerte")
    ax_h1.set_title("Diagrama $H_1$")
    ax_h1.legend(fontsize=6, loc="lower right")

    # Panel inferior izquierdo: vacío (simetría visual)
    ax_empty.axis("off")

    # Panel inferior central: landscape H0 con CI 95%
    eps_range = np.linspace(0, lim, 300)
    # Incluir todos los pares: finitos + inmortales (con muerte truncada a lim)
    h0_all = h0_fin
    if len(h0_inf) > 0:
        h0_all = np.vstack([h0_fin, np.column_stack([
            h0_inf[:, 0], np.full(len(h0_inf), lim)
        ])])
    if len(h0_all) > 0:
        tiendas_h0 = np.zeros((len(h0_all), len(eps_range)))
        for i in range(len(h0_all)):
            b, d = h0_all[i]
            mid = (b + d) / 2.0
            half = (d - b) / 2.0
            tiendas_h0[i] = np.maximum(0, half - np.abs(eps_range - mid))
        tiendas_h0_sorted = np.sort(tiendas_h0, axis=0)[::-1]
        colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
        for k in range(min(3, len(tiendas_h0_sorted))):
            ax_land0.plot(eps_range, tiendas_h0_sorted[k],
                          c=colores_land[k], lw=1.0,
                          label=f"$\\lambda_{{{k + 1}}}$")
    ci_95_h0 = _null_landscape_95(puntos, lim, eps_range, dim=0)
    ax_land0.plot(eps_range, ci_95_h0, c="#888888", lw=0.8, ls="--",
                  label="umbral 95%")
    ax_land0.set_xlabel("$\\varepsilon$")
    ax_land0.set_ylabel("$\\lambda_k(\\varepsilon)$")
    ax_land0.set_title("Paisaje $H_0$")
    ax_land0.legend(fontsize=5, loc="upper right")

    # Panel inferior derecho: landscape H1 con CI 95%
    if len(h1_fin) > 0:
        tiendas_h1 = np.zeros((len(h1_fin), len(eps_range)))
        for i in range(len(h1_fin)):
            b, d = h1_fin[i]
            mid = (b + d) / 2.0
            half = (d - b) / 2.0
            tiendas_h1[i] = np.maximum(0, half - np.abs(eps_range - mid))
        tiendas_h1_sorted = np.sort(tiendas_h1, axis=0)[::-1]
        colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
        for k in range(min(3, len(tiendas_h1_sorted))):
            ax_land1.plot(eps_range, tiendas_h1_sorted[k],
                          c=colores_land[k], lw=1.0,
                          label=f"$\\lambda_{{{k + 1}}}$")
    ci_95_h1 = _null_landscape_95(puntos, lim, eps_range, dim=1)
    ax_land1.plot(eps_range, ci_95_h1, c="#888888", lw=0.8, ls="--",
                  label="umbral 95%")
    ax_land1.set_xlabel("$\\varepsilon$")
    ax_land1.set_ylabel("$\\lambda_k(\\varepsilon)$")
    ax_land1.set_title("Paisaje $H_1$")
    ax_land1.legend(fontsize=5, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "anillo.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_anillo_invarianza(out_dir=None):
    """Genera la figura de invarianza topológica: óvalo, cuadrado, curva aleatoria.

    Muestra que formas geométricamente distintas pero topológicamente
    equivalentes (todas tienen un agujero) producen los mismos diagramas
    de persistencia: 1 componente en H0, 1 ciclo persistente en H1.
    Grid: 3 columnas (formas) × 5 filas (puntos, dgm H0, land H0, dgm H1, land H1).
    """
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    n = 150
    ruido = 0.08
    rng = np.random.default_rng(42)

    # Generar las tres formas
    def _ovalo():
        theta = rng.uniform(0, 2 * np.pi, n)
        pts = np.column_stack([1.5 * np.cos(theta), 0.7 * np.sin(theta)])
        return pts + rng.normal(0, ruido, (n, 2))

    def _cuadrado():
        # Puntos uniformes sobre el perímetro de un cuadrado [-1,1]×[-1,1]
        perimetro = 8.0  # 4 lados × 2
        t = rng.uniform(0, perimetro, n)
        pts = np.zeros((n, 2))
        for i, ti in enumerate(t):
            if ti < 2:
                pts[i] = [ti - 1, -1]       # lado inferior
            elif ti < 4:
                pts[i] = [1, (ti - 2) - 1]  # lado derecho
            elif ti < 6:
                pts[i] = [1 - (ti - 4), 1]  # lado superior
            else:
                pts[i] = [-1, 1 - (ti - 6)] # lado izquierdo
        return pts + rng.normal(0, ruido, (n, 2))

    def _curva_aleatoria():
        # Curva cerrada suave aleatoria (perturbación radial del círculo)
        theta = np.sort(rng.uniform(0, 2 * np.pi, n))
        # Radio variable: 1 + suma de armónicos
        r = 1.0 + 0.3 * np.sin(2 * theta) + 0.2 * np.cos(3 * theta) \
            + 0.15 * np.sin(5 * theta)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return pts + rng.normal(0, ruido, (n, 2))

    formas = [
        ("Óvalo", _ovalo()),
        ("Cuadrado", _cuadrado()),
        ("Curva aleatoria", _curva_aleatoria()),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(5.0, 7.5))
    lim = 3.0

    for col, (nombre, puntos) in enumerate(formas):
        dgms = homologia_persistente(puntos, maxdim=1)
        h0 = dgms[0]
        h0_fin = h0[np.isfinite(h0[:, 1])]
        h0_inf = h0[~np.isfinite(h0[:, 1])]
        h1 = dgms[1]
        h1_fin = h1[np.isfinite(h1[:, 1])]

        eps_range = np.linspace(0, lim, 300)

        # Fila 0: nube de puntos
        ax = axes[0, col]
        ax.scatter(puntos[:, 0], puntos[:, 1],
                   s=8, c="#2196F3", alpha=0.7, edgecolors="none")
        ax.set_title(nombre, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Fila 1: diagrama H0
        ax = axes[1, col]
        if len(h0_fin) > 0:
            ax.scatter(h0_fin[:, 0], h0_fin[:, 1],
                       s=8, c="#888888", alpha=0.5, edgecolors="none")
        if len(h0_inf) > 0:
            y_inf = lim * 0.95
            ax.scatter(h0_inf[:, 0], [y_inf] * len(h0_inf),
                       s=50, c="red", marker="*", zorder=5)
            ax.annotate("", xy=(h0_inf[0, 0], lim),
                        xytext=(h0_inf[0, 0], y_inf),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.0))
        ax.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
        ax.set_xlim(-0.05, lim)
        ax.set_ylim(-0.05, lim)
        if col == 0:
            ax.set_ylabel("Muerte", fontsize=7)
        ax.set_xlabel("Nacimiento", fontsize=6)
        ax.set_title("$H_0$", fontsize=8) if col > 0 else ax.set_title(
            "Diagrama $H_0$", fontsize=8)
        ax.tick_params(labelsize=5)

        # Fila 2: landscape H0
        ax = axes[2, col]
        h0_all = h0_fin
        if len(h0_inf) > 0:
            h0_all = np.vstack([h0_fin, np.column_stack([
                h0_inf[:, 0], np.full(len(h0_inf), lim)
            ])])
        if len(h0_all) > 0:
            tiendas_h0 = np.zeros((len(h0_all), len(eps_range)))
            for i in range(len(h0_all)):
                b, d = h0_all[i]
                mid = (b + d) / 2.0
                half = (d - b) / 2.0
                tiendas_h0[i] = np.maximum(0, half - np.abs(eps_range - mid))
            tiendas_h0_sorted = np.sort(tiendas_h0, axis=0)[::-1]
            colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
            for k in range(min(3, len(tiendas_h0_sorted))):
                ax.plot(eps_range, tiendas_h0_sorted[k],
                        c=colores_land[k], lw=0.8,
                        label=f"$\\lambda_{{{k + 1}}}$")
        ci_95_h0 = _null_landscape_95(puntos, lim, eps_range, dim=0)
        ax.plot(eps_range, ci_95_h0, c="#888888", lw=0.7, ls="--",
                label="umbral 95%")
        if col == 0:
            ax.set_ylabel("$\\lambda_k$", fontsize=7)
        ax.set_xlabel("$\\varepsilon$", fontsize=6)
        ax.set_title("Paisaje $H_0$", fontsize=8) if col == 0 else ax.set_title(
            "$H_0$", fontsize=8)
        ax.legend(fontsize=4, loc="upper right")
        ax.tick_params(labelsize=5)

        # Fila 3: diagrama H1
        ax = axes[3, col]
        if len(h1_fin) > 0:
            ax.scatter(h1_fin[:, 0], h1_fin[:, 1],
                       s=8, c="#E91E63", alpha=0.6, edgecolors="none")
            # Marcar el agujero principal
            pers_h1 = h1_fin[:, 1] - h1_fin[:, 0]
            idx_max = np.argmax(pers_h1)
            ax.scatter(h1_fin[idx_max, 0], h1_fin[idx_max, 1],
                       s=50, c="#E91E63", marker="*", zorder=5)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
        ax.set_xlim(-0.05, lim)
        ax.set_ylim(-0.05, lim)
        if col == 0:
            ax.set_ylabel("Muerte", fontsize=7)
        ax.set_xlabel("Nacimiento", fontsize=6)
        ax.set_title("Diagrama $H_1$", fontsize=8) if col == 0 else ax.set_title(
            "$H_1$", fontsize=8)
        ax.tick_params(labelsize=5)

        # Fila 4: landscape H1
        ax = axes[4, col]
        if len(h1_fin) > 0:
            tiendas_h1 = np.zeros((len(h1_fin), len(eps_range)))
            for i in range(len(h1_fin)):
                b, d = h1_fin[i]
                mid = (b + d) / 2.0
                half = (d - b) / 2.0
                tiendas_h1[i] = np.maximum(0, half - np.abs(eps_range - mid))
            tiendas_h1_sorted = np.sort(tiendas_h1, axis=0)[::-1]
            colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
            for k in range(min(3, len(tiendas_h1_sorted))):
                ax.plot(eps_range, tiendas_h1_sorted[k],
                        c=colores_land[k], lw=0.8,
                        label=f"$\\lambda_{{{k + 1}}}$")
        ci_95_h1 = _null_landscape_95(puntos, lim, eps_range, dim=1)
        ax.plot(eps_range, ci_95_h1, c="#888888", lw=0.7, ls="--",
                label="umbral 95%")
        if col == 0:
            ax.set_ylabel("$\\lambda_k$", fontsize=7)
        ax.set_xlabel("$\\varepsilon$", fontsize=6)
        ax.set_title("Paisaje $H_1$", fontsize=8) if col == 0 else ax.set_title(
            "$H_1$", fontsize=8)
        ax.legend(fontsize=4, loc="upper right")
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    path = os.path.join(out_dir, "anillo-invarianza.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def _compute_landscape(dgm_fin, eps_range):
    """Calcula las funciones λ_k del paisaje de persistencia."""
    if len(dgm_fin) == 0:
        return np.zeros((1, len(eps_range)))
    tiendas = np.zeros((len(dgm_fin), len(eps_range)))
    for i in range(len(dgm_fin)):
        b, d = dgm_fin[i]
        mid = (b + d) / 2.0
        half = (d - b) / 2.0
        tiendas[i] = np.maximum(0, half - np.abs(eps_range - mid))
    return np.sort(tiendas, axis=0)[::-1]


def fig_ruido(out_dir=None):
    """Genera la figura de Receta 3: 4 niveles de ruido con diagrama H1 + landscape H1."""
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    niveles = [0.05, 0.15, 0.3, 0.5]
    fig, axes = plt.subplots(3, 4, figsize=(5.0, 4.5))

    lim = 2.8
    eps_range = np.linspace(0, lim, 300)

    for col, sigma in enumerate(niveles):
        pts = generar_circulo(n=200, ruido=sigma)
        dgms = homologia_persistente(pts, maxdim=1)
        ag = agujero_principal(dgms)
        pers = ag["persistencia"] if ag else 0.0

        # Fila 0: nube de puntos
        ax_pts = axes[0, col]
        ax_pts.scatter(pts[:, 0], pts[:, 1],
                       s=3, c="#2196F3", alpha=0.5, edgecolors="none")
        ax_pts.set_title(f"$\\sigma$ = {sigma}", fontsize=7)
        ax_pts.set_aspect("equal")
        ax_pts.set_xticks([])
        ax_pts.set_yticks([])

        # Fila 1: diagrama de persistencia H1
        ax_dg = axes[1, col]
        h1 = dgms[1]
        h1_fin = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else h1
        if len(h1_fin) > 0:
            ax_dg.scatter(h1_fin[:, 0], h1_fin[:, 1],
                          s=8, c="#E91E63", alpha=0.6, edgecolors="none")
            # Marcar el agujero principal
            if ag:
                p = h1_fin[:, 1] - h1_fin[:, 0]
                idx_max = np.argmax(p)
                ax_dg.scatter(h1_fin[idx_max, 0], h1_fin[idx_max, 1],
                              s=40, c="#E91E63", marker="*", zorder=5)
        ax_dg.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.5)
        ax_dg.set_xlim(-0.05, lim)
        ax_dg.set_ylim(-0.05, lim)
        ax_dg.set_title(f"$H_1$  p = {pers:.2f}", fontsize=6)
        ax_dg.set_aspect("equal")
        if col == 0:
            ax_dg.set_xlabel("Nacimiento", fontsize=6)
            ax_dg.set_ylabel("Muerte", fontsize=6)
        ax_dg.tick_params(labelsize=5)

        # Fila 2: landscape H1
        ax_land = axes[2, col]
        if len(h1_fin) > 0:
            land = _compute_landscape(h1_fin, eps_range)
            colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
            for k in range(min(3, len(land))):
                ax_land.plot(eps_range, land[k],
                             c=colores_land[k], lw=0.8,
                             label=f"$\\lambda_{{{k + 1}}}$")
        # Umbral 95%
        ci_95 = _null_landscape_95(pts, lim, eps_range, dim=1)
        ax_land.plot(eps_range, ci_95, c="#888888", lw=0.7, ls="--",
                     label="umbral 95%")
        ax_land.set_xlabel("$\\varepsilon$", fontsize=6)
        if col == 0:
            ax_land.set_ylabel("$\\lambda_k$", fontsize=6)
        ax_land.set_title(f"Paisaje $H_1$", fontsize=6)
        ax_land.legend(fontsize=4, loc="upper right")
        ax_land.tick_params(labelsize=5)

    plt.tight_layout()
    path = os.path.join(out_dir, "ruido.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_invarianza_dimension(out_dir=None):
    """Genera la figura de invarianza: bucles aleatorios en ℝ², ℝ³, ℝ⁴, ℝ⁵.

    Fila 0: proyección en las primeras 2 coordenadas (formas distorsionadas).
    Fila 1: proyección PCA a 2D (recupera la forma de bucle).
    Fila 2: diagrama de persistencia H1 (todos muestran un ciclo persistente).
    Fila 3: paisaje de persistencia H1 (todos con λ_1 dominante).
    """
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)
    from sklearn.decomposition import PCA

    dimensiones = [2, 3, 4, 5]
    fig, axes = plt.subplots(4, 4, figsize=(5.0, 5.8))

    lim = 2.8
    eps_range = np.linspace(0, lim, 300)

    for col, dim in enumerate(dimensiones):
        pts = generar_bucle_fourier(n=200, dim=dim, ruido=0.05, seed=100 + col * 7)
        dgms = homologia_persistente(pts, maxdim=1)
        ag = agujero_principal(dgms)
        pers = ag["persistencia"] if ag else 0.0

        # Fila 0: proyección en las primeras 2 coordenadas
        ax_pts = axes[0, col]
        ax_pts.scatter(pts[:, 0], pts[:, 1],
                       s=3, c="#2196F3", alpha=0.6, edgecolors="none")
        ax_pts.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    c="#2196F3", alpha=0.3, lw=0.5)
        ax_pts.set_title(f"$\\mathbb{{R}}^{{{dim}}}$ → $(x_1, x_2)$", fontsize=7)
        ax_pts.set_aspect("equal")
        ax_pts.set_xticks([])
        ax_pts.set_yticks([])

        # Fila 1: proyección PCA a 2D
        ax_pca = axes[1, col]
        if dim > 2:
            pts_2d = PCA(n_components=2).fit_transform(pts)
        else:
            pts_2d = pts
        ax_pca.scatter(pts_2d[:, 0], pts_2d[:, 1],
                       s=3, c="#FF9800", alpha=0.6, edgecolors="none")
        ax_pca.plot(np.append(pts_2d[:, 0], pts_2d[0, 0]),
                    np.append(pts_2d[:, 1], pts_2d[0, 1]),
                    c="#FF9800", alpha=0.3, lw=0.5)
        ax_pca.set_title(f"PCA 2D", fontsize=7)
        ax_pca.set_aspect("equal")
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])

        # Fila 2: diagrama H1
        ax_dg = axes[2, col]
        h1 = dgms[1]
        h1_fin = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else h1
        if len(h1_fin) > 0:
            ax_dg.scatter(h1_fin[:, 0], h1_fin[:, 1],
                          s=8, c="#E91E63", alpha=0.6, edgecolors="none")
            if ag:
                p = h1_fin[:, 1] - h1_fin[:, 0]
                idx_max = np.argmax(p)
                ax_dg.scatter(h1_fin[idx_max, 0], h1_fin[idx_max, 1],
                              s=40, c="#E91E63", marker="*", zorder=5)
        ax_dg.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.5)
        ax_dg.set_xlim(-0.05, lim)
        ax_dg.set_ylim(-0.05, lim)
        ax_dg.set_title(f"$H_1$  p = {pers:.2f}", fontsize=6)
        ax_dg.set_aspect("equal")
        if col == 0:
            ax_dg.set_xlabel("Nacimiento", fontsize=6)
            ax_dg.set_ylabel("Muerte", fontsize=6)
        ax_dg.tick_params(labelsize=5)

        # Fila 3: landscape H1
        ax_land = axes[3, col]
        if len(h1_fin) > 0:
            land = _compute_landscape(h1_fin, eps_range)
            colores_land = ["#D32F2F", "#1976D2", "#388E3C"]
            for k in range(min(3, len(land))):
                ax_land.plot(eps_range, land[k],
                             c=colores_land[k], lw=0.8,
                             label=f"$\\lambda_{{{k + 1}}}$")
        ci_95 = _null_landscape_95(pts[:, :2], lim, eps_range, dim=1)
        ax_land.plot(eps_range, ci_95, c="#888888", lw=0.7, ls="--",
                     label="umbral 95%")
        ax_land.set_xlabel("$\\varepsilon$", fontsize=6)
        if col == 0:
            ax_land.set_ylabel("$\\lambda_k$", fontsize=6)
        ax_land.set_title(f"Paisaje $H_1$", fontsize=6)
        ax_land.legend(fontsize=4, loc="upper right")
        ax_land.tick_params(labelsize=5)

    plt.tight_layout()
    path = os.path.join(out_dir, "invarianza-dimension.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_ciclo_representativo(out_dir=None):
    """Genera la figura de Receta 4: ciclo representativo sobre el anillo.

    Usa GUDHI para extraer el ciclo homológico del H1 más persistente.
    El ciclo es una cadena cerrada de aristas (camino más corto entre los
    extremos de la arista de nacimiento) que encierra el agujero detectado.
    """
    import gudhi
    import networkx as nx

    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    puntos = generar_circulo(n=100, ruido=0.1, seed=42)

    # --- Homología persistente con GUDHI ---
    rips = gudhi.RipsComplex(points=puntos, max_edge_length=2.5)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2)

    # Generadores H1: shape (m, 4) = [birth_v1, birth_v2, death_v1, death_v2]
    h1_gens = st.flag_persistence_generators()[1][0]

    # Seleccionar el más persistente
    def _filt_edge(row):
        return st.filtration(sorted([int(row[0]), int(row[1])]))

    def _filt_death(row):
        return st.filtration(sorted([int(row[2]), int(row[3])]))

    pers = np.array([_filt_death(r) - _filt_edge(r) for r in h1_gens])
    h1_gens = h1_gens[np.argsort(pers)[::-1]]
    u, v = int(h1_gens[0, 0]), int(h1_gens[0, 1])
    birth_scale = st.filtration([u, v])
    persistence = pers.max()

    # Diagrama H1
    dgms_h1 = np.column_stack([
        [_filt_edge(r) for r in h1_gens],
        [_filt_death(r) for r in h1_gens],
    ])

    # --- Reconstruir ciclo: camino más corto u→v con aristas anteriores ---
    G = nx.Graph()
    for simplex, filt in st.get_filtration():
        if len(simplex) == 2 and filt < birth_scale:
            G.add_edge(simplex[0], simplex[1])
    try:
        ciclo = nx.shortest_path(G, u, v)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        for simplex, filt in st.get_filtration():
            if len(simplex) == 2 and filt <= birth_scale and set(simplex) != {u, v}:
                G.add_edge(simplex[0], simplex[1])
        ciclo = nx.shortest_path(G, u, v)

    # --- Figura: 2 paneles ---
    fig, (ax_all, ax_dgm) = plt.subplots(1, 2, figsize=(5.0, 2.6))

    # Panel izquierdo: puntos + ciclo
    mask_ciclo = np.zeros(len(puntos), dtype=bool)
    mask_ciclo[ciclo] = True
    ax_all.scatter(puntos[~mask_ciclo, 0], puntos[~mask_ciclo, 1],
                   s=12, c="#BDBDBD", alpha=0.6, edgecolors="none",
                   zorder=2, label="Otros puntos")
    for k in range(len(ciclo)):
        i, j = ciclo[k], ciclo[(k + 1) % len(ciclo)]
        ax_all.plot([puntos[i, 0], puntos[j, 0]],
                    [puntos[i, 1], puntos[j, 1]],
                    c="#D32F2F", lw=1.5, alpha=0.8, zorder=3)
    ax_all.scatter(puntos[ciclo, 0], puntos[ciclo, 1],
                   s=18, c="#D32F2F", alpha=0.9, edgecolors="none",
                   zorder=4, label=f"Ciclo ({len(ciclo)} vértices)")
    ax_all.set_aspect("equal")
    ax_all.set_title("Ciclo representativo de $H_1$")
    ax_all.set_xlabel("$x$")
    ax_all.set_ylabel("$y$")
    ax_all.legend(fontsize=6, loc="lower left")

    # Panel derecho: diagrama de persistencia H1
    lim = 2.5
    ax_dgm.scatter(dgms_h1[:, 0], dgms_h1[:, 1],
                   s=12, c="#E91E63", alpha=0.5, edgecolors="none", label="$H_1$")
    ax_dgm.scatter(dgms_h1[0, 0], dgms_h1[0, 1],
                   s=80, c="#D32F2F", marker="*", zorder=5,
                   label=f"Ciclo principal\n(pers={persistence:.2f})")
    ax_dgm.plot([0, lim], [0, lim], "k--", alpha=0.2, lw=0.8)
    ax_dgm.set_xlim(-0.05, lim)
    ax_dgm.set_ylim(-0.05, lim)
    ax_dgm.set_xlabel("Nacimiento")
    ax_dgm.set_ylabel("Muerte")
    ax_dgm.set_title("Diagrama $H_1$")
    ax_dgm.legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    path = os.path.join(out_dir, "ciclo-representativo.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def fig_mapper_heroe(out_dir=None):
    """Genera la figura de Receta 5: hipergrafo en espiral para 9 novelas (3×3).

    Cada nodo se coloca en una espiral de Arquímedes ordenada por posición
    narrativa (centro = inicio, exterior = final). Aristas grises = conexiones
    semánticas del Mapper. Flechas azules = recorrido cronológico. Flechas rojas
    = camino de retorno (shortest path end→start). El ratio d/trav se muestra
    como subtítulo.
    """
    import kmapper as km
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch

    plt = _setup_style()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if out_dir is None:
        out_dir = DIAGRAMS_DIR

    data = cargar_libros()
    embeddings = data["embeddings"]
    positions = data["positions"]
    book_ids = data["book_ids"]

    titles_es = [
        "La Odisea",
        "Beowulf",
        "Divina Comedia",
        "Don Quijote",
        "Moby Dick",
        "El Conde de\nMonte Cristo",
        "El Mago de Oz",
        "La vuelta al mundo\nen 80 días",
        "Una Princesa\nde Marte",
    ]

    mapper = km.KeplerMapper(verbose=0)

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 6.2))
    axes = axes.flatten()

    # Color map: red (start) → orange → yellow → green → dark blue (end)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [
        "#d62728",  # red (start)
        "#ff7f0e",  # orange
        "#e8c838",  # yellow
        "#2ca02c",  # green
        "#1a6b3a",  # dark green
        "#1f4e79",  # blue
        "#1f3b73",  # dark blue (end)
    ]
    cmap = LinearSegmentedColormap.from_list("heroe", colors_list, N=256)
    norm = Normalize(vmin=0, vmax=1)

    for book_idx in range(9):
        ax = axes[book_idx]

        mask = book_ids == book_idx
        emb = embeddings[mask]
        pos = positions[mask]

        if len(emb) > 300:
            idx = np.linspace(0, len(emb) - 1, 300, dtype=int)
            emb, pos = emb[idx], pos[idx]

        # Lente semántica: PCA 2D independiente (organiza la cuadrícula de cobertura)
        lens = PCA(n_components=2).fit_transform(emb)

        # Clustering: PCA 20D independiente — cubre el rango medio de la
        # dimensión intrínseca de embeddings de frases (~10-30D), con un
        # coeficiente de variación de distancias ~32% que hace DBSCAN discriminativo
        emb_clust = PCA(n_components=20).fit_transform(emb)

        graph = mapper.map(
            lens, emb_clust,
            cover=km.Cover(n_cubes=7, perc_overlap=0.35),
            clusterer=DBSCAN(eps=1.5, min_samples=3))

        # Build NetworkX graph with average position per node
        G = nx.Graph()
        node_positions_avg = {}
        node_sizes = {}

        for node_id, members in graph["nodes"].items():
            G.add_node(node_id)
            node_positions_avg[node_id] = np.mean(pos[members])
            node_sizes[node_id] = len(members)

        for src, targets in graph["links"].items():
            for tgt in targets:
                G.add_edge(src, tgt)

        if len(G.nodes()) == 0:
            ax.set_title(titles_es[book_idx], fontsize=7)
            ax.axis("off")
            continue

        # Sort nodes by narrative position for spiral placement
        node_list = sorted(G.nodes(),
                           key=lambda n: node_positions_avg[n])

        # Archimedean spiral layout: r = a + b*theta
        n_nodes = len(node_list)
        max_turns = 2.5  # number of spiral turns
        thetas = np.linspace(0, max_turns * 2 * np.pi, n_nodes)
        a, b = 0.15, 0.12  # spiral parameters
        radii = a + b * thetas
        spiral_layout = {}
        for i, node_id in enumerate(node_list):
            spiral_layout[node_id] = (
                radii[i] * np.cos(thetas[i]),
                radii[i] * np.sin(thetas[i])
            )

        # Draw semantic edges (grey, thin)
        nx.draw_networkx_edges(
            G, spiral_layout, ax=ax, alpha=0.2, width=0.4,
            edge_color="#999999", style="solid")

        # Draw chronological traversal (blue arrows along spiral)
        for i in range(n_nodes - 1):
            src_pos = spiral_layout[node_list[i]]
            tgt_pos = spiral_layout[node_list[i + 1]]
            arrow = FancyArrowPatch(
                src_pos, tgt_pos,
                arrowstyle="->,head_width=2,head_length=1.5",
                color="#4c78a8", linewidth=0.4, alpha=0.6,
                connectionstyle="arc3,rad=0.0")
            ax.add_patch(arrow)

        # Compute d/trav metric
        start_node = node_list[0]
        end_node = node_list[-1]
        # trav = sum of shortest-path distances between consecutive nodes
        trav = 0
        for i in range(n_nodes - 1):
            try:
                trav += nx.shortest_path_length(
                    G, node_list[i], node_list[i + 1])
            except nx.NetworkXNoPath:
                trav += n_nodes  # penalty for disconnected
        # d = shortest path from end to start
        try:
            d = nx.shortest_path_length(G, end_node, start_node)
            return_path = nx.shortest_path(G, end_node, start_node)
        except nx.NetworkXNoPath:
            d = -1
            return_path = []

        # Draw return path (red arrows, straight and thin)
        if len(return_path) > 1:
            for i in range(len(return_path) - 1):
                src_pos = spiral_layout[return_path[i]]
                tgt_pos = spiral_layout[return_path[i + 1]]
                arrow = FancyArrowPatch(
                    src_pos, tgt_pos,
                    arrowstyle="->,head_width=2,head_length=1.5",
                    color="#d62728", linewidth=0.5, alpha=0.85,
                    connectionstyle="arc3,rad=0.0")
                ax.add_patch(arrow)

        # Draw nodes colored by narrative position
        node_colors = [cmap(norm(node_positions_avg[n]))
                       for n in node_list]
        sizes = [max(15, min(60, (node_sizes[n] / mask.sum()) * 500))
                 for n in node_list]

        nx.draw_networkx_nodes(
            G, spiral_layout, nodelist=node_list, node_color=node_colors,
            node_size=sizes, ax=ax, edgecolors="k", linewidths=0.2)

        # Title with d/trav ratio
        if d >= 0 and trav > 0:
            ratio = d / trav
            ax.set_title(
                f"{titles_es[book_idx]}\n"
                f"$d/\\text{{trav}} = {ratio:.3f}$",
                fontsize=6.5)
        else:
            ax.set_title(titles_es[book_idx], fontsize=7)
        ax.axis("off")
        ax.set_aspect("equal")

    # Colorbar
    fig.subplots_adjust(bottom=0.07, hspace=0.45, wspace=0.15)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.012])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Posición narrativa", fontsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Inicio", "Medio", "Final"])

    path = os.path.join(out_dir, "mapper-heroe.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_qc_ramas(out_dir=None):
    """Diagrama H0 + paisaje con umbral 95% (nulo Gaussiano) para Receta 6."""
    plt = _setup_style()
    out_dir = out_dir or DIAGRAMS_DIR
    os.makedirs(out_dir, exist_ok=True)

    from ripser import ripser as _ripser
    from sklearn.decomposition import PCA
    import matplotlib.cm as cm

    qc_npz = os.path.join(DATA_DIR, "arxiv_qc.npz")
    if not os.path.exists(qc_npz):
        print(f"  ⚠ {qc_npz} no encontrado — omitiendo fig_qc_ramas")
        return

    data = np.load(qc_npz, allow_pickle=True)
    emb = data["embeddings"]
    titles = data["titles"]

    emb_red = PCA(n_components=50).fit_transform(emb)

    rng = np.random.default_rng(42)
    idx800 = rng.choice(len(emb_red), 800, replace=False)
    puntos = emb_red[idx800]

    # Ambos paneles en la proyección 2D (PC1, PC2): el nulo uniforme sólo
    # es calibrable en baja dimensión — en 50D la concentración de la medida
    # hace que las distancias del nulo y la señal sean indistinguibles.
    puntos_2d = puntos[:, :2]

    dgms_2d = _ripser(puntos_2d, maxdim=0)["dgms"]
    h0_2d = dgms_2d[0]
    fin_2d = h0_2d[np.isfinite(h0_2d[:, 1])]
    inf_2d = h0_2d[~np.isfinite(h0_2d[:, 1])]
    lim_2d = float(fin_2d[:, 1].max()) * 1.1

    eps_range = np.linspace(0, lim_2d, 300)
    todos_2d = fin_2d.copy()
    if len(inf_2d) > 0:
        todos_2d = np.vstack([todos_2d,
                              np.column_stack([inf_2d[:, 0],
                                               np.full(len(inf_2d), lim_2d)])])
    tiendas = np.zeros((len(todos_2d), len(eps_range)))
    for i in range(len(todos_2d)):
        b, d = todos_2d[i]
        mid = (b + d) / 2.0
        half = (d - b) / 2.0
        tiendas[i] = np.maximum(0, half - np.abs(eps_range - mid))
    tiendas_sorted = np.sort(tiendas, axis=0)[::-1]

    # Umbral 95%: nulo uniforme en 2D
    ci_95 = _null_landscape_95(puntos_2d, lim_2d, eps_range)

    # Contar cuántas λ_k superan el umbral — ese es el número de ramas
    # estadísticamente significativas en la proyección 2D
    MAX_K = min(len(tiendas_sorted), 15)
    N_RAMAS = 0
    for k in range(MAX_K):
        if np.any(tiendas_sorted[k] > ci_95):
            N_RAMAS += 1
        else:
            break
    N_RAMAS = max(N_RAMAS, 1)

    fig, (ax_dgm, ax_land) = plt.subplots(1, 2, figsize=(5.5, 2.8))

    # ── Panel izquierdo: diagrama de persistencia H0 (2D) ────────────
    ax_dgm.scatter(fin_2d[:, 0], fin_2d[:, 1],
                   s=8, c="#AAAAAA", alpha=0.4, edgecolors="none",
                   label="fusiones rápidas")
    muertes_ord = np.argsort(fin_2d[:, 1])[::-1]
    n_late = min(N_RAMAS - 1, len(muertes_ord))
    for i in range(n_late):
        mi = muertes_ord[i]
        ax_dgm.scatter(fin_2d[mi, 0], fin_2d[mi, 1],
                       s=60, c="#D32F2F", marker="*", zorder=5)
    if len(inf_2d) > 0:
        y_inf = lim_2d * 0.96
        ax_dgm.scatter(inf_2d[:, 0], [y_inf] * len(inf_2d),
                       s=60, c="#D32F2F", marker="*", zorder=5,
                       label=f"ramas ({N_RAMAS})")
        ax_dgm.annotate("", xy=(inf_2d[0, 0], lim_2d),
                        xytext=(inf_2d[0, 0], y_inf),
                        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.0))
    ax_dgm.plot([0, lim_2d], [0, lim_2d], "k--", alpha=0.2, lw=0.8)
    ax_dgm.set_xlim(-0.01 * lim_2d, lim_2d)
    ax_dgm.set_ylim(-0.01 * lim_2d, lim_2d * 1.05)
    ax_dgm.set_xlabel("Nacimiento")
    ax_dgm.set_ylabel("Muerte")
    ax_dgm.set_title("Diagrama $H_0$ (proyección 2D)")
    ax_dgm.legend(fontsize=6, loc="lower right")

    # ── Panel derecho: paisaje de persistencia H0 (2D) ───────────────
    for k in range(N_RAMAS):
        lk = tiendas_sorted[k]
        color = cm.plasma(k / max(N_RAMAS - 1, 1))
        ax_land.plot(eps_range, lk, c=color, lw=1.0,
                     label=f"$\\lambda_{{{k + 1}}}$")

    ax_land.plot(eps_range, ci_95, c="#555555", lw=0.9, ls="--",
                 label="umbral 95%")
    ax_land.set_xlabel("$\\varepsilon$")
    ax_land.set_ylabel("$\\lambda_k(\\varepsilon)$")
    ax_land.set_title("Paisaje $H_0$ (proyección 2D)")
    ax_land.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "qc-ramas.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


def generar_figuras(out_dir=None):
    """Genera todas las figuras del capítulo TDA."""
    try:
        from ripser import ripser  # noqa: F401
    except ImportError:
        print("ripser no instalado — no se generan figuras")
        return

    print("Generando figuras TDA...")
    fig_tres_islas(out_dir)
    fig_tres_islas_null(out_dir)
    fig_anillo(out_dir)
    fig_anillo_invarianza(out_dir)
    fig_ruido(out_dir)
    fig_invarianza_dimension(out_dir)
    fig_ciclo_representativo(out_dir)
    fig_mapper_heroe(out_dir)
    fig_qc_ramas(out_dir)
    print("Figuras generadas.")


if __name__ == "__main__":
    import sys
    if "--figuras" in sys.argv:
        generar_figuras()
    else:
        main()
