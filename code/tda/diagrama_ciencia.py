"""
Genera los diagramas para la Receta 6 del capítulo TDA:
  «¿Es la ciencia un único continente?»

Salidas:
  ../../diagrams/tda/ciencia-h0.{pdf,png}   — H0 del corpus multi-campo
  ../../diagrams/tda/mapper-ciencia.{pdf,png} — Mapper con puentes inter-disciplina

Datos: data/arxiv_ciencia.npz (generado por collect_arxiv_ciencia.py)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from ripser import ripser
import kmapper as km

# ── Rutas ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_NPZ   = os.path.join(SCRIPT_DIR, "data", "arxiv_ciencia.npz")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "diagrams", "tda")

# ── Paleta por campo macro ────────────────────────────────────────────
FIELD_COLOR = {
    "Informática":  "#1f77b4",   # azul
    "Física":       "#d62728",   # rojo
    "Matemáticas":  "#2ca02c",   # verde
    "Estadística":  "#ff7f0e",   # naranja
    "Biología":     "#9467bd",   # violeta
    "Economía":     "#8c564b",   # marrón
    "Otros":        "#7f7f7f",   # gris
}
MACRO_FIELD = {
    "cs.LG": "Informática", "cs.CV": "Informática", "cs.CL": "Informática", "cs.AI": "Informática",
    "cs.RO": "Informática", "cs.SE": "Informática", "cs.CR": "Informática", "cs.DC": "Informática",
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


# ── Utilidades ────────────────────────────────────────────────────────

def cargar_datos():
    data = np.load(DATA_NPZ, allow_pickle=True)
    cats   = data["categories"]
    fields = np.array([MACRO_FIELD.get(c, "Other") for c in cats], dtype=object)
    return data["embeddings"], data["titles"], cats, fields


def bootstrap_umbral(emb_red, n_bootstrap=200, n_null=500, seed=42):
    rng = np.random.default_rng(seed)
    low, high = emb_red.min(axis=0), emb_red.max(axis=0)
    null_max = []
    for _ in range(n_bootstrap):
        pts = rng.uniform(low, high, size=(n_null, emb_red.shape[1]))
        h   = ripser(pts, maxdim=0)["dgms"][0]
        fin = h[np.isfinite(h[:, 1])]
        pers = fin[:, 1] - fin[:, 0]
        null_max.append(pers.max() if len(fin) > 0 else 0.0)
    return float(np.percentile(null_max, 95)), null_max


# ── Diagrama 1: H0 ────────────────────────────────────────────────────

def hacer_diagrama_h0(emb_red, thresh, null_max, pers_h0, out_prefix):
    """
    Dos paneles:
      Izq: diagrama de persistencia H0
      Der: paisaje de persistencia lambda_1 con umbral nulo
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Panel izquierdo: diagrama de persistencia ──
    ax = axes[0]
    h0 = ripser(emb_red, maxdim=0)["dgms"][0]
    births = h0[:, 0]
    deaths_raw = h0[:, 1]
    # Componente inmortal: poner muerte en max + margen
    deaths = np.where(np.isfinite(deaths_raw), deaths_raw,
                      deaths_raw[np.isfinite(deaths_raw)].max() * 1.15
                      if np.any(np.isfinite(deaths_raw)) else 2.0)

    ax.scatter(births, deaths, s=6, color="#aaaaaa", alpha=0.5, zorder=2)
    # Inmortales en rojo
    immortal_mask = ~np.isfinite(deaths_raw)
    if immortal_mask.any():
        ax.scatter(births[immortal_mask], deaths[immortal_mask],
                   s=40, color="#d62728", marker="*", zorder=4,
                   label="Componente inmortal")

    xmax = max(deaths[np.isfinite(deaths)].max() * 1.05, thresh * 1.1) if len(deaths) else thresh * 1.2
    ax.plot([0, xmax], [0, xmax], "k--", lw=0.8, alpha=0.5)
    ax.axhline(thresh, color="#e377c2", lw=1.2, ls="--",
               label=f"Umbral 95% ({thresh:.2f})")
    ax.set_xlabel("Nacimiento $\\varepsilon$", fontsize=11)
    ax.set_ylabel("Muerte $\\varepsilon$", fontsize=11)
    ax.set_title("Diagrama de persistencia $H_0$", fontsize=12)
    ax.legend(fontsize=9)

    # ── Panel derecho: paisaje λ1 ──
    ax = axes[1]

    # Construir λ1 desde las barras finitas
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    pers = finite_h0[:, 1] - finite_h0[:, 0]

    eps_range = np.linspace(0, xmax, 400)
    lambda1 = np.zeros_like(eps_range)
    for (b, d) in zip(finite_h0[:, 0], finite_h0[:, 1]):
        mid = (b + d) / 2
        half = (d - b) / 2
        tent = np.maximum(0, half - np.abs(eps_range - mid))
        lambda1 = np.maximum(lambda1, tent)

    ax.fill_between(eps_range, 0, lambda1, alpha=0.3, color="#1f77b4")
    ax.plot(eps_range, lambda1, color="#1f77b4", lw=1.5, label="$\\lambda_1$")

    # Umbral nulo: media de los null_max
    null_arr = np.array(null_max)
    umbral_landscape = float(np.percentile(null_arr, 95)) / 2
    ax.axhline(umbral_landscape, color="#e377c2", lw=1.2, ls="--",
               label=f"Umbral 95%")
    ax.set_xlabel("Escala $\\varepsilon$", fontsize=11)
    ax.set_ylabel("$\\lambda_1(\\varepsilon)$", fontsize=11)
    ax.set_title("Paisaje de persistencia $\\lambda_1$", fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle("Análisis $H_0$ — Corpus multi-campo arXiv 2025\n"
                 f"8.085 artículos · 29 categorías · N_RAMAS = 1",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(OUT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {pdf_path}")


# ── Diagrama 2: Mapper ────────────────────────────────────────────────

def hacer_diagrama_mapper(emb, titles, fields, out_prefix,
                          n_cubes=10, perc_overlap=0.4):
    """
    Grafo Mapper coloreado por campo macro dominante.
    Dos reducciones independientes desde los embeddings brutos:
      - Lente PCA 2D: organiza la cuadrícula de cobertura
      - Clustering PCA 20D: dimensión en el rango intrínseco (~10-30D),
        CV de distancias ~32% — DBSCAN discriminativo
    """
    # Lente: PCA 2D independiente (organiza la cuadrícula de cobertura)
    lens = PCA(n_components=2, random_state=42).fit_transform(emb)

    # Clustering: PCA 20D independiente
    emb_clust = PCA(n_components=20, random_state=42).fit_transform(emb)

    mapper = km.KeplerMapper(verbose=0)
    graph  = mapper.map(
        lens, emb_clust,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=DBSCAN(eps=2.0, min_samples=3),
    )

    nodes = graph["nodes"]
    links = graph["links"]

    n_nodes = len(nodes)
    n_edges = sum(len(v) for v in links.values())

    # Nodos con mezcla de campos (campo dominante < 60% del nodo)
    def campo_dominante_y_frac(idx_list):
        if not idx_list:
            return "Other", 1.0
        flds_node = fields[idx_list]
        vals, counts = np.unique(flds_node, return_counts=True)
        best = np.argmax(counts)
        return vals[best], counts[best] / len(idx_list)

    mixed_nodes = set()
    for n in nodes:
        _, frac = campo_dominante_y_frac(nodes[n])
        if frac < 0.60:
            mixed_nodes.add(n)

    print(f"  Mapper: {n_nodes} nodos, {n_edges} aristas, "
          f"{len(mixed_nodes)} nodos mixtos")

    # Campo dominante de cada nodo
    def campo_dominante(idx_list):
        if not idx_list:
            return "Other"
        flds_node = fields[idx_list]
        vals, counts = np.unique(flds_node, return_counts=True)
        return vals[np.argmax(counts)]

    node_fields = {n: campo_dominante(nodes[n]) for n in nodes}

    # Composición completa de campos por nodo (para los sectores)
    def composicion_campos(idx_list):
        """Devuelve lista de (campo, fracción) ordenada de mayor a menor."""
        if not idx_list:
            return [("Other", 1.0)]
        flds_node = fields[idx_list]
        vals, counts = np.unique(flds_node, return_counts=True)
        total = len(idx_list)
        pairs = sorted(zip(vals, counts / total), key=lambda x: -x[1])
        return pairs

    # Layout: spring con NetworkX
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    for src, dsts in links.items():
        for d in dsts:
            G.add_edge(src, d)
    pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Determinar rango de coordenadas para escalar radios
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(xs) - min(xs) if len(xs) > 1 else 1.0
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    base_radius = min(x_span, y_span) * 0.0079   # radio base en unidades de datos

    # Aristas
    for src, dsts in links.items():
        for d in dsts:
            x0, y0 = pos[src]
            x1, y1 = pos[d]
            ax.plot([x0, x1], [y0, y1], "k-", lw=0.8, alpha=0.4, zorder=1)

    # Nodos: sectores proporcionales a la composición de campos
    from matplotlib.patches import Wedge

    for n in nodes:
        x, y = pos[n]
        n_papers = len(nodes[n])
        r_data = base_radius * max(1.0, n_papers ** 0.25)

        comp = composicion_campos(nodes[n])
        start = 90.0  # empezar desde arriba
        for campo, frac in comp:
            angle = frac * 360.0
            color = FIELD_COLOR.get(campo, "#7f7f7f")
            wedge = Wedge(
                (x, y), r_data,
                start, start + angle,
                facecolor=color, edgecolor="none",
                linewidth=0, alpha=0.9, zorder=3,
            )
            ax.add_patch(wedge)
            start += angle


    # Leyenda de campos
    legend_patches = [
        mpatches.Patch(color=c, label=f)
        for f, c in FIELD_COLOR.items()
    ]
    ax.legend(handles=legend_patches,
              loc="lower right", fontsize=8, framealpha=0.9,
              ncol=2, handlelength=1)

    ax.set_title(
        f"Grafo Mapper — Corpus multi-campo arXiv 2025\n"
        f"{n_nodes} nodos · {n_edges} aristas · "
        f"Sectores proporcionales a la composición de campos",
        fontsize=11
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(OUT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {pdf_path}")


# ── Diagrama 3: Comparación Mapper 1D vs 2D ───────────────────────────

def _dibujar_grafo_mapper(ax, graph, fields, title):
    """Dibuja un grafo Mapper en los ejes dados con nodos de sectores."""
    nodes = graph["nodes"]
    links = graph["links"]
    n_nodes = len(nodes)
    n_edges = sum(len(v) for v in links.values())

    def composicion_campos(idx_list):
        if not idx_list:
            return [("Other", 1.0)]
        flds_node = fields[idx_list]
        vals, counts = np.unique(flds_node, return_counts=True)
        total = len(idx_list)
        return sorted(zip(vals, counts / total), key=lambda x: -x[1])

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
    base_radius = min(x_span, y_span) * 0.0079

    from matplotlib.patches import Wedge
    for src, dsts in links.items():
        for d in dsts:
            x0, y0 = pos[src]
            x1, y1 = pos[d]
            ax.plot([x0, x1], [y0, y1], "k-", lw=0.8, alpha=0.4, zorder=1)

    for n in nodes:
        x, y = pos[n]
        n_papers = len(nodes[n])
        r_data = base_radius * max(1.0, n_papers ** 0.25)
        comp = composicion_campos(nodes[n])
        start = 90.0
        for campo, frac in comp:
            angle = frac * 360.0
            color = FIELD_COLOR.get(campo, "#7f7f7f")
            wedge = Wedge(
                (x, y), r_data, start, start + angle,
                facecolor=color, edgecolor="none",
                linewidth=0, alpha=0.9, zorder=3,
            )
            ax.add_patch(wedge)
            start += angle

    ax.set_title(f"{title}\n{n_nodes} nodos · {n_edges} aristas",
                 fontsize=10)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    return n_nodes, n_edges


def hacer_diagrama_mapper_comparacion(emb, fields, out_prefix,
                                      n_cubes=10, perc_overlap=0.4):
    """
    Figura de dos paneles: Mapper con lente PCA 1D (izquierda) y PCA 2D (derecha).
    La lente 1D produce un árbol; la 2D produce un grafo con ciclos.
    Ambas usan PCA 20D independiente para el clustering.
    """
    mapper = km.KeplerMapper(verbose=0)

    # Clustering compartido: PCA 20D independiente
    emb_clust = PCA(n_components=20, random_state=42).fit_transform(emb)

    # Lente 1D
    lens1d = PCA(n_components=1, random_state=42).fit_transform(emb)
    graph1d = mapper.map(
        lens1d, emb_clust,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=DBSCAN(eps=2.0, min_samples=3),
    )

    # Lente 2D
    lens2d = PCA(n_components=2, random_state=42).fit_transform(emb)
    graph2d = mapper.map(
        lens2d, emb_clust,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=DBSCAN(eps=2.0, min_samples=3),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    n1, e1 = _dibujar_grafo_mapper(axes[0], graph1d, fields,
                                   "Lente PCA 1D (árbol, sin ciclos)")
    n2, e2 = _dibujar_grafo_mapper(axes[1], graph2d, fields,
                                   "Lente PCA 2D (con ciclos)")

    print(f"  Comparación — 1D: {n1} nodos, {e1} aristas | "
          f"2D: {n2} nodos, {e2} aristas")

    # Leyenda compartida abajo
    legend_patches = [
        mpatches.Patch(color=c, label=f)
        for f, c in FIELD_COLOR.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Elección de lente en Mapper — Corpus multi-campo arXiv 2025",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(OUT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Cargando datos...")
    emb, titles, cats, fields = cargar_datos()
    print(f"  {len(emb)} papers, {len(np.unique(cats))} categorías")

    print("PCA → 50 componentes...")
    emb_red = PCA(n_components=50, random_state=42).fit_transform(emb)

    print("Bootstrap umbral 95%...")
    thresh, null_max = bootstrap_umbral(emb_red)

    h0 = ripser(emb_red, maxdim=0)["dgms"][0]
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    pers_h0 = finite_h0[:, 1] - finite_h0[:, 0]
    print(f"  Umbral={thresh:.4f}, max barra={pers_h0.max():.4f}, "
          f"N_RAMAS={int((pers_h0 > thresh).sum()) + 1}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nGenerando diagrama H0...")
    hacer_diagrama_h0(emb_red, thresh, null_max, pers_h0, "ciencia-h0")

    print("\nGenerando diagrama Mapper...")
    hacer_diagrama_mapper(emb, titles, fields, "mapper-ciencia")

    print("\nGenerando comparación Mapper 1D vs 2D...")
    hacer_diagrama_mapper_comparacion(emb, fields, "mapper-ciencia-lentes")

    print("\nListo.")


if __name__ == "__main__":
    main()
