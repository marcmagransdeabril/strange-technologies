"""
Mapper con filtro de densidad sobre el corpus de QC (arXiv).

Construye un grafo Mapper donde la función lente es la densidad local
(distancia media a los 10 vecinos más próximos, proxy de la escala de
fusión en H0).  Los nodos periféricos (grado ≤ 1) identifican las dos
clases de zonas no estudiadas:
  - Alta densidad-lente (vecinos lejanos): papers genuinamente aislados.
  - Baja densidad-lente (vecinos cercanos): bolsillos densos pero
    conectados al resto sólo por un hilo.

Ejecución:
    cd code/tda
    python mapper_zonas.py

Salida:
    ../../diagrams/tda/mapper-qc-zonas.{pdf,png}
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import kmapper as km


# --------------------------------------------------------------------------- #
# Análisis                                                                     #
# --------------------------------------------------------------------------- #

def cargar_corpus(path: str):
    """Carga embeddings, títulos y fechas de publicación del corpus."""
    data = np.load(path, allow_pickle=True)
    published = data["published"] if "published" in data.files else None
    return data["embeddings"], data["titles"], published


def densidad_local(emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Distancia media a los k vecinos más próximos (escala de fusión H0)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    dists, _ = nn.kneighbors(emb)
    return dists[:, 1:].mean(axis=1)


def construir_mapper(
    emb_red: np.ndarray,
    lens: np.ndarray,
    n_cubes: int = 20,
    perc_overlap: float = 0.3,
    eps: float = 2.0,
    min_samples: int = 3,
) -> dict:
    mapper = km.KeplerMapper(verbose=0)
    return mapper.map(
        lens.reshape(-1, 1),
        emb_red,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
    )


def grados(graph: dict) -> dict:
    """Grado de cada nodo (aristas no dirigidas)."""
    deg = {n: 0 for n in graph["nodes"]}
    for src, dsts in graph["links"].items():
        deg[src] += len(dsts)
        for d in dsts:
            deg[d] += 1
    return deg


def paper_mas_antiguo(idx_list: list, published) -> int:
    """Índice del paper más antiguo del nodo (proxy del más citado en corpus reciente)."""
    return min(idx_list, key=lambda i: published[i])


def paper_central(idx_list: list, emb_red: np.ndarray) -> int:
    """Índice del paper más cercano al centroide del nodo."""
    cluster_emb = emb_red[idx_list]
    centroid = cluster_emb.mean(axis=0)
    dists = np.linalg.norm(cluster_emb - centroid, axis=1)
    return idx_list[int(np.argmin(dists))]


def imprimir_nodos_perifericos(
    graph: dict,
    deg: dict,
    emb_red: np.ndarray,
    lens: np.ndarray,
    titles: np.ndarray,
    published=None,
    umbral_grado: int = 1,
):
    nodes = graph["nodes"]
    perifericos = [n for n, d in deg.items() if d <= umbral_grado and nodes[n]]
    perifericos.sort(key=lambda n: float(lens[nodes[n]].mean()), reverse=True)

    print(f"Nodos totales: {len(nodes)}")
    print(f"Nodos periféricos (grado ≤ {umbral_grado}): {len(perifericos)}")
    print()

    for nodo in perifericos:
        idx_list = nodes[nodo]
        dens = float(lens[idx_list].mean())
        print(f"[{nodo}]  n={len(idx_list)}  densidad={dens:.3f}  grado={deg[nodo]}")
        if published is not None:
            rep = paper_mas_antiguo(idx_list, published)
            print(f"  Más antiguo: {published[rep][:10]}  {titles[rep]}")
        else:
            rep = paper_central(idx_list, emb_red)
            print(f"  Central: {titles[rep]}")
        if len(idx_list) <= 8:
            for i in idx_list:
                print(f"    · {titles[i][:88]}")
        print()


# --------------------------------------------------------------------------- #
# Visualización                                                                #
# --------------------------------------------------------------------------- #

def _layout_vertical(nodes: dict, lens: np.ndarray, edges: dict) -> tuple:
    """
    Layout vertical: nodos ordenados por densidad-lente de abajo (baja) a arriba (alta).
    y = posición equiespaciada por rango de densidad (evita apilamientos).
    x = sinusoide suave para romper la linealidad.
    El islote aislado se desplaza a x ≈ 1.3.
    """
    G_temp = nx.Graph()
    G_temp.add_nodes_from(nodes.keys())
    for src, dsts in edges.items():
        for d in dsts:
            G_temp.add_edge(src, d)
    components = list(nx.connected_components(G_temp))
    main_comp = max(components, key=len)

    lens_mean = {n: float(lens[nodes[n]].mean()) for n in nodes}
    lmin, lmax = min(lens_mean.values()), max(lens_mean.values())

    sorted_main = sorted([n for n in nodes if n in main_comp],
                         key=lambda n: lens_mean[n])
    sorted_iso = sorted([n for n in nodes if n not in main_comp],
                        key=lambda n: lens_mean[n])

    pos = {}
    n_main = len(sorted_main)
    for i, n in enumerate(sorted_main):
        y = i / (n_main - 1) if n_main > 1 else 0.5
        x = 0.08 * np.sin(i * np.pi / 3.5)
        pos[n] = (x, y)

    # Islote: a la derecha, posición y según densidad real
    for n in sorted_iso:
        y = (lens_mean[n] - lmin) / (lmax - lmin + 1e-9)
        pos[n] = (1.3, y)

    return pos, lens_mean, lmin, lmax


def visualizar_mapper(
    graph: dict,
    deg: dict,
    lens: np.ndarray,
    titles: np.ndarray,
    emb_red: np.ndarray,
    published,
    out_prefix: str,
):
    """Genera y guarda el diagrama Mapper como PDF y PNG."""
    nodes = graph["nodes"]
    edges = graph["links"]

    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    for src, dsts in edges.items():
        for d in dsts:
            G.add_edge(src, d)

    pos, lens_mean, lmin, lmax = _layout_vertical(nodes, lens, edges)
    perifericos = {n for n, d in deg.items() if d <= 1}

    cmap = cm.YlOrRd
    norm_c = mcolors.Normalize(vmin=lmin, vmax=lmax)

    node_list = list(nodes.keys())
    colors = [cmap(norm_c(lens_mean[n])) for n in node_list]
    sizes = [max(80, len(nodes[n]) ** 0.4 * 22) for n in node_list]
    edge_colors = ["black" if n in perifericos else "#bbbbbb" for n in node_list]
    linewidths = [2.5 if n in perifericos else 0.6 for n in node_list]

    fig, ax = plt.subplots(figsize=(9, 13))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color="#888888", width=1.1)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           nodelist=node_list,
                           node_color=colors,
                           node_size=sizes,
                           edgecolors=edge_colors,
                           linewidths=linewidths)

    # Etiqueta = paper más central (más cercano al centroide) para TODOS los nodos
    for n in node_list:
        if not nodes[n]:
            continue
        idx = paper_central(nodes[n], emb_red)
        title = titles[idx]
        short = (title[:52] + "…") if len(title) > 52 else title
        n_papers = len(nodes[n])
        date = published[idx][:7] if published is not None else ""
        label = f"{short}  [{date}] ({n_papers} papers)"

        x, y = pos[n]
        # Nodos del islote: etiqueta a la derecha; resto: también a la derecha
        ha = "left"
        x_offset = 12
        ax.annotate(
            label, xy=(x, y),
            xytext=(x_offset, 0), textcoords="offset points",
            ha=ha, va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"),
        )

    # Barra de color (horizontal, en la parte inferior)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        shrink=0.45, pad=0.02, aspect=25)
    cbar.set_label("densidad-lente (media dist. a 10 vecinos)", fontsize=8)

    ax.set_title(
        "Grafo Mapper — corpus QC (arXiv)\n"
        "↑ alta densidad (ruido de corpus)      ↓ baja densidad (bolsillos aislados)",
        fontsize=9, pad=10,
    )

    # Anotación del islote
    iso_nodes = [n for n in node_list if abs(pos[n][0] - 1.3) < 0.1]
    if iso_nodes:
        ys = [pos[n][1] for n in iso_nodes]
        ax.annotate(
            "islote aislado\n(ruido de corpus)",
            xy=(1.3, sum(ys) / len(ys)),
            xytext=(1.3, sum(ys) / len(ys) - 0.12),
            ha="center", va="top", fontsize=7.5, color="#aa2222",
            arrowprops=dict(arrowstyle="-", color="#aa2222", lw=0.8),
        )

    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#f5c27a", markeredgecolor="#bbbbbb",
               markeredgewidth=0.6, markersize=10, label="Nodo core"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#c0392b", markeredgecolor="black",
               markeredgewidth=2.5, markersize=10,
               label="Nodo periférico (grado ≤ 1)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8, framealpha=0.85)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fpath = f"{out_prefix}.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {fpath}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Punto de entrada                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_here, "data", "arxiv_qc_5y.npz")
    out_prefix = os.path.join(_here, "..", "..", "diagrams", "tda", "mapper-qc-zonas")

    print("Cargando corpus…")
    emb, titles, published = cargar_corpus(data_path)

    print("Reducción dimensional (PCA 50)…")
    emb_red = PCA(n_components=50, random_state=42).fit_transform(emb)

    print("Calculando densidad local…")
    lens = densidad_local(emb_red, k=10)

    print("Construyendo grafo Mapper…")
    graph = construir_mapper(emb_red, lens)
    deg = grados(graph)

    print("\n── Nodos periféricos ──")
    imprimir_nodos_perifericos(graph, deg, emb_red, lens, titles, published)

    print("Generando diagrama…")
    visualizar_mapper(graph, deg, lens, titles, emb_red, published, out_prefix)

    print("\nListo.")
