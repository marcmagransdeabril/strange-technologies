"""
Genera los diagramas para la Receta 8 del capítulo TDA:
  «UMAP y Mapper aplicado al Camino del Héroe y al árbol de la ciencia»

Comparación PCA 2D vs UMAP 2D como función lente del Mapper, en los mismos
corpus de las Recetas 5 y 6. Sin reducción de dimensión adicional.

Salidas:
  ../../diagrams/tda/mapper-heroe-umap.{pdf,png}   — 3×3 espiral, lente UMAP
  ../../diagrams/tda/mapper-ciencia-umap.{pdf,png} — comparación PCA 2D vs UMAP 2D
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# ── Rutas ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "..", "diagrams", "tda")

# ── Paleta por campo macro (igual que diagrama_ciencia.py) ────────────
FIELD_COLOR = {
    "Informática":  "#1f77b4",
    "Física":       "#d62728",
    "Matemáticas":  "#2ca02c",
    "Estadística":  "#ff7f0e",
    "Biología":     "#9467bd",
    "Economía":     "#8c564b",
    "Otros":        "#7f7f7f",
}
MACRO_FIELD = {
    "cs.LG": "Informática", "cs.CV": "Informática", "cs.CL": "Informática",
    "cs.AI": "Informática", "cs.RO": "Informática", "cs.SE": "Informática",
    "cs.CR": "Informática", "cs.DC": "Informática",
    "hep-th": "Física", "hep-ph": "Física", "quant-ph": "Física",
    "cond-mat.mes-hall": "Física", "cond-mat.str-el": "Física",
    "astro-ph.GA": "Física", "astro-ph.CO": "Física", "gr-qc": "Física",
    "math.AG": "Matemáticas", "math.ST": "Matemáticas", "math.OC": "Matemáticas",
    "math.CO": "Matemáticas", "math.NT": "Matemáticas",
    "stat.ML": "Estadística", "stat.ME": "Estadística",
    "q-bio.QM": "Biología", "q-bio.GN": "Biología",
    "econ.GN": "Economía", "econ.EM": "Economía",
    "physics.soc-ph": "Otros", "nlin.CD": "Otros",
}


# ─────────────────────────────────────────────────────────────────────
# Receta 8a: Camino del Héroe con lente UMAP
# ─────────────────────────────────────────────────────────────────────

def _plot_heroe_grid(axes, embeddings, positions, book_ids, titles_es,
                     use_umap: bool, mapper: km.KeplerMapper):
    """Rellena la cuadrícula 3×3 con el hipergrafo en espiral de 9 novelas.

    Si use_umap=True usa UMAP directamente sobre los embeddings (sin PCA previa).
    Si use_umap=False usa las 2 primeras componentes PCA(20) como lente.
    """
    colors_list = [
        "#d62728", "#ff7f0e", "#e8c838",
        "#2ca02c", "#1a6b3a", "#1f4e79", "#1f3b73",
    ]
    cmap = LinearSegmentedColormap.from_list("heroe", colors_list, N=256)
    norm = Normalize(vmin=0, vmax=1)

    reducer_lens  = umap.UMAP(n_components=2, n_neighbors=15, random_state=42) \
        if use_umap else None
    reducer_clust = umap.UMAP(n_components=10, n_neighbors=15, random_state=42) \
        if use_umap else None

    for book_idx in range(9):
        ax = axes[book_idx]

        mask = book_ids == book_idx
        emb  = embeddings[mask]
        pos  = positions[mask]

        if len(emb) > 300:
            idx = np.linspace(0, len(emb) - 1, 300, dtype=int)
            emb, pos = emb[idx], pos[idx]

        if use_umap:
            # Lente 2D: UMAP 2D independiente — captura la variedad semántica
            # no lineal de los pasajes (análogo a PCA 2D en Receta 5)
            lens       = reducer_lens.fit_transform(emb)   # (n, 2)
            # Clustering: UMAP 10D independiente
            # (cubre la dimensión intrínseca ~10-30D de embeddings de frases,
            # CV de distancias ~45% — más discriminativo que PCA 20D)
            emb_graph  = reducer_clust.fit_transform(emb)
        else:
            # PCA branch: dos reducciones independientes consistentes con Receta 5
            lens      = PCA(n_components=2).fit_transform(emb)   # lente
            emb_graph = PCA(n_components=20).fit_transform(emb)  # clustering

        graph = mapper.map(
            lens, emb_graph,
            cover=km.Cover(n_cubes=7, perc_overlap=0.5),
            clusterer=DBSCAN(eps=0.8, min_samples=3),
        )

        G = nx.Graph()
        node_positions_avg = {}
        node_sizes = {}

        for node_id, members in graph["nodes"].items():
            G.add_node(node_id)
            node_positions_avg[node_id] = float(np.mean(pos[members]))
            node_sizes[node_id] = len(members)

        for src, targets in graph["links"].items():
            for tgt in targets:
                G.add_edge(src, tgt)

        if len(G.nodes()) == 0:
            ax.set_title(titles_es[book_idx], fontsize=7)
            ax.axis("off")
            continue

        node_list = sorted(G.nodes(), key=lambda n: node_positions_avg[n])
        n_nodes = len(node_list)

        # Espiral de Arquímedes
        max_turns = 2.5
        thetas = np.linspace(0, max_turns * 2 * np.pi, n_nodes)
        a, b = 0.15, 0.12
        radii = a + b * thetas
        spiral_layout = {
            node_list[i]: (radii[i] * np.cos(thetas[i]),
                           radii[i] * np.sin(thetas[i]))
            for i in range(n_nodes)
        }

        # Aristas semánticas (gris)
        nx.draw_networkx_edges(
            G, spiral_layout, ax=ax, alpha=0.2, width=0.4,
            edge_color="#999999")

        # Flechas azules: recorrido cronológico
        for i in range(n_nodes - 1):
            ax.add_patch(FancyArrowPatch(
                spiral_layout[node_list[i]], spiral_layout[node_list[i + 1]],
                arrowstyle="->,head_width=2,head_length=1.5",
                color="#4c78a8", linewidth=0.4, alpha=0.6,
                connectionstyle="arc3,rad=0.0"))

        # Métrica d/trav
        start_node, end_node = node_list[0], node_list[-1]
        trav = 0
        for i in range(n_nodes - 1):
            try:
                trav += nx.shortest_path_length(G, node_list[i], node_list[i + 1])
            except nx.NetworkXNoPath:
                trav += n_nodes
        try:
            d = nx.shortest_path_length(G, end_node, start_node)
            return_path = nx.shortest_path(G, end_node, start_node)
        except nx.NetworkXNoPath:
            d = -1
            return_path = []

        # Flechas rojas: camino de retorno
        if len(return_path) > 1:
            for i in range(len(return_path) - 1):
                ax.add_patch(FancyArrowPatch(
                    spiral_layout[return_path[i]],
                    spiral_layout[return_path[i + 1]],
                    arrowstyle="->,head_width=2,head_length=1.5",
                    color="#d62728", linewidth=0.5, alpha=0.85,
                    connectionstyle="arc3,rad=0.0"))

        # Nodos coloreados por posición narrativa
        node_colors = [cmap(norm(node_positions_avg[n])) for n in node_list]
        sizes = [max(15, min(60, (node_sizes[n] / mask.sum()) * 500))
                 for n in node_list]
        nx.draw_networkx_nodes(
            G, spiral_layout, nodelist=node_list, node_color=node_colors,
            node_size=sizes, ax=ax, edgecolors="k", linewidths=0.2)

        n_edges = sum(len(v) for v in graph["links"].values())
        label = ("UMAP" if use_umap else "PCA")
        if d >= 0 and trav > 0:
            ratio = d / trav
            ax.set_title(
                f"{titles_es[book_idx]}\n$d/\\text{{trav}}={ratio:.3f}$  [{label}]",
                fontsize=6.0)
        else:
            ax.set_title(f"{titles_es[book_idx]}  [{label}]", fontsize=7)

        ax.axis("off")
        ax.set_aspect("equal")
        print(f"    {titles_es[book_idx]:25s}  nodos={n_nodes}, aristas={n_edges}")


def hacer_diagrama_heroe_umap():
    """Genera mapper-heroe-umap.pdf: lente UMAP para las 9 novelas."""
    npz_path = os.path.join(DATA_DIR, "hero_books.npz")
    if not os.path.exists(npz_path):
        print(f"  ⚠ {npz_path} no encontrado — omitiendo mapper-heroe-umap")
        return

    data       = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    positions  = data["positions"]
    book_ids   = data["book_ids"]

    titles_es = [
        "La Odisea", "Beowulf", "Divina Comedia",
        "Don Quijote", "Moby Dick", "El Conde de\nMonte Cristo",
        "El Mago de Oz", "La vuelta al mundo\nen 80 días", "Una Princesa\nde Marte",
    ]

    mapper = km.KeplerMapper(verbose=0)
    colors_list = ["#d62728", "#ff7f0e", "#e8c838",
                   "#2ca02c", "#1a6b3a", "#1f4e79", "#1f3b73"]
    cmap = LinearSegmentedColormap.from_list("heroe", colors_list, N=256)
    norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 6.2))
    axes_flat = axes.flatten()

    print("  Héroe (lente UMAP):")
    _plot_heroe_grid(axes_flat, embeddings, positions, book_ids,
                     titles_es, use_umap=True, mapper=mapper)

    fig.subplots_adjust(bottom=0.07, hspace=0.50, wspace=0.15)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.012])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Posición narrativa", fontsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Inicio", "Medio", "Final"])

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"mapper-heroe-umap.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  Guardado: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Receta 8b: Árbol de la ciencia — comparación PCA 2D vs UMAP 2D
# ─────────────────────────────────────────────────────────────────────

def _composicion_campos(idx_list, fields):
    if not idx_list:
        return [("Otros", 1.0)]
    flds = fields[idx_list]
    vals, counts = np.unique(flds, return_counts=True)
    total = len(idx_list)
    return sorted(zip(vals, counts / total), key=lambda x: -x[1])


def _dibujar_grafo_ciencia(ax, graph, fields, title):
    nodes = graph["nodes"]
    links = graph["links"]
    n_nodes = len(nodes)
    n_edges = sum(len(v) for v in links.values())

    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    for src, dsts in links.items():
        for d in dsts:
            G.add_edge(src, d)
    pos = nx.kamada_kawai_layout(G)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(xs) - min(xs) if len(xs) > 1 else 1.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    base_r = min(x_span, y_span) * 0.004

    for src, dsts in links.items():
        for d in dsts:
            x0, y0 = pos[src]
            x1, y1 = pos[d]
            ax.plot([x0, x1], [y0, y1], "k-", lw=0.8, alpha=0.35, zorder=1)

    for n in nodes:
        x, y = pos[n]
        r = base_r * max(1.0, len(nodes[n]) ** 0.25)
        comp = _composicion_campos(nodes[n], fields)
        start = 90.0
        for campo, frac in comp:
            angle = frac * 360.0
            wedge = Wedge(
                (x, y), r, start, start + angle,
                facecolor=FIELD_COLOR.get(campo, "#7f7f7f"),
                edgecolor="none", linewidth=0, alpha=0.9, zorder=3,
            )
            ax.add_patch(wedge)
            start += angle

    ax.set_title(f"{title}\n{n_nodes} nodos · {n_edges} aristas", fontsize=10)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    print(f"    {title}: {n_nodes} nodos, {n_edges} aristas")
    return n_nodes, n_edges


def hacer_diagrama_ciencia_umap():
    """Genera mapper-ciencia-umap.pdf: comparación lente PCA 2D (izq) vs UMAP 2D (der)."""
    npz_path = os.path.join(DATA_DIR, "arxiv_ciencia.npz")
    if not os.path.exists(npz_path):
        print(f"  ⚠ {npz_path} no encontrado — omitiendo mapper-ciencia-umap")
        return

    data   = np.load(npz_path, allow_pickle=True)
    emb    = data["embeddings"]
    cats   = data["categories"]
    fields = np.array([MACRO_FIELD.get(c, "Otros") for c in cats], dtype=object)

    # Reducción PCA 50 — igual que en la Receta 6; para ripser (criterio: varianza)
    print("  PCA → 50 componentes...")
    emb_red = PCA(n_components=50, random_state=42).fit_transform(emb)

    mapper = km.KeplerMapper(verbose=0)

    # Lente PCA 2D: independiente, desde emb bruto
    print("  Construyendo grafo con lente PCA 2D...")
    lens_pca  = PCA(n_components=2, random_state=42).fit_transform(emb)
    # Clustering PCA 20D: independiente (CV ~32%, rango intrínseco ~10-30D)
    emb_clust_pca = PCA(n_components=20, random_state=42).fit_transform(emb)
    graph_pca = mapper.map(
        lens_pca, emb_clust_pca,
        cover=km.Cover(n_cubes=10, perc_overlap=0.4),
        clusterer=DBSCAN(eps=2.0, min_samples=3),
    )

    # Lente UMAP 2D: independiente, desde emb_red (eficiencia computacional)
    print("  Construyendo grafo con lente UMAP 2D...")
    lens_umap = umap.UMAP(n_components=2, n_neighbors=15,
                          random_state=42).fit_transform(emb_red)
    # Clustering UMAP 10D: independiente, desde emb_red
    # (no lineal, cubre dimensión intrínseca ~10-30D)
    emb_clust_umap = umap.UMAP(n_components=10, n_neighbors=15,
                               random_state=42).fit_transform(emb_red)
    graph_umap = mapper.map(
        lens_umap, emb_clust_umap,
        cover=km.Cover(n_cubes=10, perc_overlap=0.4),
        clusterer=DBSCAN(eps=0.5, min_samples=3),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    _dibujar_grafo_ciencia(axes[0], graph_pca,  fields, "Lente PCA 2D")
    _dibujar_grafo_ciencia(axes[1], graph_umap, fields, "Lente UMAP 2D")

    legend_patches = [
        mpatches.Patch(color=c, label=f)
        for f, c in FIELD_COLOR.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Árbol de la ciencia: lente PCA 2D vs lente UMAP 2D\n"
        "Corpus multi-campo arXiv 2025 · 8.085 artículos",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"mapper-ciencia-umap.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  Guardado: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=== Receta 8: UMAP como lente Mapper ===\n")

    print("Generando mapper-heroe-umap (Camino del Héroe con lente UMAP)...")
    hacer_diagrama_heroe_umap()

    print("\nGenerando mapper-ciencia-umap (PCA 2D vs UMAP 2D)...")
    hacer_diagrama_ciencia_umap()

    print("\nListo.")


if __name__ == "__main__":
    main()
