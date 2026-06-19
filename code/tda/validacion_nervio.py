"""
Receta 8 — Validación interna del nervio Mapper.

Genera diagrams/tda/validacion-nervio-ciencia.pdf:
  2 paneles (PCA-nervio | UMAP-nervio), cada uno con:
    - Paisaje medio H₀, H₁ (capas λ₁ en rojo, λ₂ en azul, λ₃ en verde)
    - Banda bootstrap 5–95% (sombreado naranja)
    - Línea nula al 95% (permutaciones de coordenadas, línea gris discontinua)

  H₂ no se calcula: con lente 2D la cubierta es una lámina y β₂=0 de forma
  estructural. Se usa maxdim=1, lo que reduce el coste de cada iteración.

El nervio Mapper se trata como complejo simplicial ponderado:
  - nodos = clusters del Mapper
  - peso de arista = 1 − Jaccard(cluster_i, cluster_j)
  - ripser opera sobre la matriz de distancias resultante con maxdim=1

Pasos del protocolo implementados:
  1. Consistencia del recuento (β₀, β₁)
  2. Banda bootstrap del paisaje (H₀, H₁)
  3. Línea nula (permutaciones)
  4. Sensibilidad a parámetros: cuadrícula (m,p) ∈ {8,10,12}×{0.3,0.4,0.5}
  5. Estabilidad de membresía: mediana J̄ de Jaccard sobre B remuestras
  6. Estabilidad del grafo (punto a punto): tasa de persistencia S̄ — sin matching de nodos
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "diagrams", "tda")
CACHE_PATH = os.path.join(DATA_DIR, "validacion_nervio_cache.npz")

# ── Parámetros del protocolo ─────────────────────────────────────────
B_BOOT  = 30    # remuestras bootstrap (80% de los papers)
B_NULL  = 200   # permutaciones para línea nula
N_PTS   = 300   # puntos en el eje ε del paisaje
N_CAPAS = 3     # λ₁, λ₂, λ₃
DIMS    = [0, 1] # H₂ es estructuralmente 0 con lente 2D; no se calcula

# Parámetros para la línea nula de estabilidad combinatoria (pasos 5-6).
# Coste: 2 × B_NULL_STAB_OUTER × B_NULL_STAB_INNER nervios extra.
B_NULL_STAB_OUTER = 10   # permutaciones de coordenadas para la nula
B_NULL_STAB_INNER = 10   # bootstrap interno sobre cada permutación


# ─────────────────────────────────────────────────────────────────────
# Primitivas topológicas
# ─────────────────────────────────────────────────────────────────────

def jaccard_dist_matrix(graph):
    """Matriz de distancias de Jaccard entre todos los clusters del nervio.

    Peso de arista = 1 − |C_i ∩ C_j| / |C_i ∪ C_j|.
    Para pares sin arista en el grafo original la distancia es 1.0
    (no hay solapamiento → máxima distancia).
    """
    nodes   = list(graph["nodes"].keys())
    n       = len(nodes)
    idx_map = {node: i for i, node in enumerate(nodes)}
    D       = np.ones((n, n))
    np.fill_diagonal(D, 0.0)

    for src, targets in graph["links"].items():
        i = idx_map[src]
        set_i = set(graph["nodes"][src])
        for tgt in targets:
            j = idx_map[tgt]
            set_j = set(graph["nodes"][tgt])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            dist  = 1.0 - inter / union if union > 0 else 1.0
            D[i, j] = dist
            D[j, i] = dist
    return D, nodes


def ripser_on_graph(graph, maxdim=1):
    """Aplica ripser sobre la matriz de distancias del nervio Mapper.

    maxdim=1: H₀ y H₁. H₂ es estructuralmente 0 con lente 2D.
    """
    from ripser import ripser
    D, _ = jaccard_dist_matrix(graph)
    return ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]


def landscape(dgms, dim=1, n_pts=N_PTS, n_capas=N_CAPAS):
    """Calcula el paisaje de persistencia para la dimensión `dim`."""
    h = dgms[dim]
    finite = h[np.isfinite(h[:, 1])] if len(h) > 0 else np.empty((0, 2))
    if len(finite) == 0:
        eps = np.linspace(0, 1, n_pts)
        return np.zeros((n_capas, n_pts)), eps

    eps_max = finite[:, 1].max() * 1.1
    eps     = np.linspace(0, eps_max, n_pts)
    tents   = np.zeros((len(finite), n_pts))
    for k, (b, d) in enumerate(finite):
        mid      = (b + d) / 2.0
        half     = (d - b) / 2.0
        tents[k] = np.maximum(half - np.abs(eps - mid), 0.0)

    tents_sorted = np.sort(tents, axis=0)[::-1]
    result       = np.zeros((n_capas, n_pts))
    n_avail      = min(n_capas, len(tents_sorted))
    result[:n_avail] = tents_sorted[:n_avail]
    return result, eps


def betti_at_epsilon(dgms, dim, eps=1.0):
    """Número de clases en H_dim vivas a escala eps.

    En una filtración por Vietoris-Rips, las barras que cumplen
    birth <= eps < death representan clases activas en esa escala.
    """
    h = dgms[dim]
    if len(h) == 0:
        return 0
    birth = h[:, 0]
    death = h[:, 1]
    alive = (birth <= eps) & (death > eps)
    return int(np.sum(alive))


# ─────────────────────────────────────────────────────────────────────
# Construcción de los nervios
# ─────────────────────────────────────────────────────────────────────

def _build_graphs(emb, emb_red):
    """Devuelve (graph_pca, graph_umap) sobre el corpus arXiv."""
    import kmapper as km
    import umap as umap_lib
    from sklearn.cluster import DBSCAN

    mapper = km.KeplerMapper(verbose=0)

    # PCA-nervio: lente PCA 2D, clustering PCA 20D
    lens_pca      = PCA(n_components=2,  random_state=42).fit_transform(emb)
    clust_pca     = PCA(n_components=20, random_state=42).fit_transform(emb)
    graph_pca = mapper.map(
        lens_pca, clust_pca,
        cover=km.Cover(n_cubes=10, perc_overlap=0.4),
        clusterer=DBSCAN(eps=2.0, min_samples=3),
    )

    # UMAP-nervio: lente UMAP 2D, clustering UMAP 10D (ambas desde emb_red)
    lens_umap  = umap_lib.UMAP(n_components=2,  n_neighbors=15,
                                random_state=42).fit_transform(emb_red)
    clust_umap = umap_lib.UMAP(n_components=10, n_neighbors=15,
                                random_state=42).fit_transform(emb_red)
    graph_umap = mapper.map(
        lens_umap, clust_umap,
        cover=km.Cover(n_cubes=10, perc_overlap=0.4),
        clusterer=DBSCAN(eps=0.5, min_samples=3),
    )
    return graph_pca, graph_umap


# ─────────────────────────────────────────────────────────────────────
# Bootstrap (pasos 2–3) — reutiliza grafos para pasos 5–6
# ─────────────────────────────────────────────────────────────────────

def bootstrap_full(emb, emb_red, build_fn, dims=None,
                   B=B_BOOT, frac=0.8, seed=0):
    """Ejecuta B remuestras bootstrap del 80% de papers.

    Devuelve:
      graphs          : lista de grafos bootstrap (reutilizados en pasos 5 y 6)
      means, los, his : dict dim → array (N_CAPAS, N_PTS)
      epss            : dict dim → array (N_PTS,)

    build_fn(emb_sub, emb_red_sub) → graph
    """
    if dims is None:
        dims = DIMS
    rng       = np.random.default_rng(seed)
    n         = len(emb)
    graphs    = []
    land_data = {dim: [] for dim in dims}
    eps_refs  = {dim: None for dim in dims}

    for _ in range(B):
        idx = rng.choice(n, size=int(n * frac), replace=False)
        g   = build_fn(emb[idx], emb_red[idx])
        if len(g["nodes"]) < 3:
            continue
        # kmapper guarda los miembros de cada nodo como índices posicionales
        # 0..N_sub-1 dentro de la submuestra. Para que el Jaccard de membresía
        # y la estabilidad de aristas comparen conjuntos de papers reales (no
        # índices locales que coinciden por casualidad numérica), remapeamos
        # a los identificadores originales del corpus.
        g["nodes"] = {k: [int(idx[i]) for i in v] for k, v in g["nodes"].items()}
        graphs.append(g)
        dgms = ripser_on_graph(g, maxdim=max(dims))
        for dim in dims:
            lam, eps = landscape(dgms, dim=dim)
            if eps_refs[dim] is None:
                eps_refs[dim] = eps
            if len(eps) != len(eps_refs[dim]):
                lam = np.array([np.interp(eps_refs[dim], eps, lam[k])
                                for k in range(N_CAPAS)])
            land_data[dim].append(lam)

    means, los, his = {}, {}, {}
    for dim in dims:
        if not land_data[dim]:
            eps_refs[dim] = np.linspace(0, 1, N_PTS)
            means[dim] = np.zeros((N_CAPAS, N_PTS))
            los[dim]   = np.zeros((N_CAPAS, N_PTS))
            his[dim]   = np.zeros((N_CAPAS, N_PTS))
        else:
            stack      = np.array(land_data[dim])  # (B_valid, N_CAPAS, N_PTS)
            means[dim] = stack.mean(axis=0)
            los[dim]   = np.percentile(stack, 5,  axis=0)
            his[dim]   = np.percentile(stack, 95, axis=0)

    return graphs, means, los, his, eps_refs


def null_line(emb, emb_red, build_fn, dim=1,
              B=B_NULL, seed=1):
    """Línea nula al 95%: permuta coordenadas de cada paper independientemente."""
    rng    = np.random.default_rng(seed)
    maxima = []

    for _ in range(B):
        # Permutación por columna: destruye geometría conjunta, conserva marginales
        emb_perm     = np.apply_along_axis(rng.permutation, 0, emb)
        emb_red_perm = np.apply_along_axis(rng.permutation, 0, emb_red)
        g = build_fn(emb_perm, emb_red_perm)
        if len(g["nodes"]) < 3:
            maxima.append(0.0)
            continue
        dgms = ripser_on_graph(g, maxdim=dim)
        lam, _ = landscape(dgms, dim=dim)
        maxima.append(float(lam[0].max()))

    return float(np.percentile(maxima, 95))


# ─────────────────────────────────────────────────────────────────────
# Línea nula para estabilidad combinatoria (pasos 5-6)
# ─────────────────────────────────────────────────────────────────────

def null_stability(emb, emb_red, build_fn,
                   B_outer=B_NULL_STAB_OUTER,
                   B_inner=B_NULL_STAB_INNER,
                   frac=0.8, seed=42):
    """Distribución nula de J̄ y núcleo robusto bajo permutación de coordenadas.

    Para cada una de B_outer permutaciones de coordenadas:
      - construye un nervio base sobre los datos permutados
      - hace bootstrap interno de B_inner remuestras del 80% sobre esos datos
      - calcula J̄ y edge_frac del base permutado frente a sus bootstrap

    El percentil 95 de cada lista define la línea nula: J̄ o edge_frac
    observados por encima de esa línea no se explican solo por las propiedades
    estructurales del Mapper bajo datos sin geometría real.

    Retorna: (j95, e95, list_j, list_e)
    """
    rng = np.random.default_rng(seed)
    n   = len(emb)
    j_bars, edge_fracs = [], []

    for outer in range(B_outer):
        emb_perm     = np.apply_along_axis(rng.permutation, 0, emb)
        emb_red_perm = np.apply_along_axis(rng.permutation, 0, emb_red)
        g_base = build_fn(emb_perm, emb_red_perm)
        if len(g_base["nodes"]) < 3:
            continue
        boot_graphs = []
        for _ in range(B_inner):
            idx = rng.choice(n, size=int(n * frac), replace=False)
            g   = build_fn(emb_perm[idx], emb_red_perm[idx])
            if len(g["nodes"]) < 3:
                continue
            g["nodes"] = {k: [int(idx[i]) for i in v]
                          for k, v in g["nodes"].items()}
            boot_graphs.append(g)
        if not boot_graphs:
            continue
        j, _      = membership_stability(g_base, boot_graphs)
        ef, _, _, _  = edge_stability_p2p(g_base, boot_graphs)
        j_bars.append(j)
        edge_fracs.append(ef)
        print(f"    permutación {outer+1}/{B_outer}: "
              f"J̄_null={j:.3f}, S̄_null={ef:.3f}", flush=True)

    j95 = float(np.percentile(j_bars, 95))     if j_bars     else 0.0
    e95 = float(np.percentile(edge_fracs, 95)) if edge_fracs else 0.0
    return j95, e95, j_bars, edge_fracs


# ─────────────────────────────────────────────────────────────────────
# Paso 4: sensibilidad a parámetros
# ─────────────────────────────────────────────────────────────────────

def sensitivity_sweep(emb, emb_red, nervio_type):
    """Cuadrícula (m, p) ∈ {8,10,12}×{0.3,0.4,0.5} — verifica (β₀,β₁) constante.

    Retorna dict {(m,p): {"n_nodes": int, "betas": {0: int, 1: int}}}
    """
    import kmapper as km
    import umap as umap_lib
    from sklearn.cluster import DBSCAN

    sweep_results = {}
    for m in [8, 10, 12]:
        for p in [0.3, 0.4, 0.5]:
            mapper = km.KeplerMapper(verbose=0)
            if nervio_type == "PCA":
                lens  = PCA(n_components=2,  random_state=42).fit_transform(emb)
                clust = PCA(n_components=20, random_state=42).fit_transform(emb)
                g = mapper.map(lens, clust,
                               cover=km.Cover(n_cubes=m, perc_overlap=p),
                               clusterer=DBSCAN(eps=2.0, min_samples=3))
            else:
                lens  = umap_lib.UMAP(n_components=2,  n_neighbors=15,
                                       random_state=42).fit_transform(emb_red)
                clust = umap_lib.UMAP(n_components=10, n_neighbors=15,
                                       random_state=42).fit_transform(emb_red)
                g = mapper.map(lens, clust,
                               cover=km.Cover(n_cubes=m, perc_overlap=p),
                               clusterer=DBSCAN(eps=0.5, min_samples=3))

            if len(g["nodes"]) < 3:
                sweep_results[(m, p)] = {"n_nodes": 0, "betas": {0: 0, 1: 0}}
                continue

            dgms  = ripser_on_graph(g, maxdim=1)
            betas = {dim: betti_at_epsilon(dgms, dim, eps=1.0) for dim in [0, 1]}
            sweep_results[(m, p)] = {"n_nodes": len(g["nodes"]), "betas": betas}
            print(f"    ({m},{p:.1f}): {len(g['nodes'])} nodos, "
                  f"β₀={betas[0]}, β₁={betas[1]}")

    return sweep_results


# ─────────────────────────────────────────────────────────────────────
# Paso 5: estabilidad de membresía
# ─────────────────────────────────────────────────────────────────────

def membership_stability(graph_base, bootstrap_graphs):
    """Mediana J̄ de Jaccard: solapamiento máximo de cada nodo base con el bootstrap.

    Para cada nodo C_i del nervio base y cada nervio bootstrap N^(b), calcula
    J_i^(b) = max_{C' ∈ N^(b)} |C_i ∩ C'|/|C_i ∪ C'|.
    J̄ = mediana de todos los (nodo, remuestra).

    Retorna: (j_bar float, scores list[float])
    """
    nodes_base = list(graph_base["nodes"].keys())
    all_scores = []
    for node_i in nodes_base:
        set_i = set(graph_base["nodes"][node_i])
        for g_boot in bootstrap_graphs:
            best_j = 0.0
            for members_j in g_boot["nodes"].values():
                set_j = set(members_j)
                union = len(set_i | set_j)
                if union > 0:
                    j = len(set_i & set_j) / union
                    if j > best_j:
                        best_j = j
            all_scores.append(best_j)

    return float(np.median(all_scores)), all_scores


# ─────────────────────────────────────────────────────────────────────
# Paso 6: reproducibilidad de pares (par de papers individual)
# ─────────────────────────────────────────────────────────────────────

def edge_stability_p2p(graph_base, bootstrap_graphs):
    """Reproducibilidad de pares S̄ y Z-score analítico (implementación numpy).

    Para cada arista base (C_u, C_v) y cada par de puntos (i,j) con i∈C_u,
    j∈C_v: testeable en b si ambos están en la submuestra; reproducido en b si
    sus nodos bootstrap están conectados.

    Implementación eficiente con multiplicación matricial:
      Sea M_boot la matriz papers×nodos_boot (M_boot[i,k]=1 si papel i en nodo k)
      y A_boot la adyacencia nodos_boot×nodos_boot.
      Para cada arista base (C_u, C_v):
        R_u = M_boot[C_u ∩ S_b, :]  →  rep_mat = R_u @ A_boot @ R_v.T
        rep_mat[i,j] > 0  ⟺  el par de papers está reproducido.
      Coste: O(n_ub × N_nb²) por arista — en lugar de O(n_pares × c²) con bucle.

    Z-score analítico por bootstrap bajo H₀ binomial:
      Z_b = (k_b − n_b·ρ_b) / √(n_b·ρ_b·(1−ρ_b)),  Z̄ = media(Z_b).

    Retorna: (s_bar float, z_bar float, rho_null float, n_pairs_base int)
    """
    # ── Aristas base y arrays de papers por lado ───────────────────────
    base_edge_papers = []   # lista de (array_C_u, array_C_v)
    visited = set()
    max_paper_id = 0
    for src, targets in graph_base["links"].items():
        pu = np.array(graph_base["nodes"][src], dtype=np.int64)
        for tgt in targets:
            key = tuple(sorted([src, tgt]))
            if key in visited:
                continue
            visited.add(key)
            pv = np.array(graph_base["nodes"][tgt], dtype=np.int64)
            base_edge_papers.append((pu, pv))
            if pu.size:
                max_paper_id = max(max_paper_id, int(pu.max()))
            if pv.size:
                max_paper_id = max(max_paper_id, int(pv.max()))

    if not base_edge_papers:
        return 0.0, 0.0, 0.0, 0

    n_pairs_base = int(sum(len(u) * len(v) for u, v in base_edge_papers))
    N_p = max_paper_id + 1

    k_per_boot = []
    n_per_boot = []
    rho_per_boot = []

    for g_boot in bootstrap_graphs:
        boot_nodes = list(g_boot["nodes"].keys())
        N_nb = len(boot_nodes)
        if N_nb == 0:
            rho_per_boot.append(0.0)
            continue
        node_to_col = {n: i for i, n in enumerate(boot_nodes)}

        # ── M_boot: papers × nodos_boot (float32) ─────────────────────
        M_boot = np.zeros((N_p, N_nb), dtype=np.float32)
        for nid, members in g_boot["nodes"].items():
            if members:
                M_boot[np.array(members, dtype=np.int64), node_to_col[nid]] = 1.0

        # ── A_boot: adyacencia nodos_boot × nodos_boot ─────────────────
        A_boot = np.zeros((N_nb, N_nb), dtype=np.float32)
        n_edges = 0
        for bsrc, btargets in g_boot["links"].items():
            u = node_to_col.get(bsrc)
            if u is None:
                continue
            for btgt in btargets:
                v = node_to_col.get(btgt)
                if v is None:
                    continue
                if A_boot[u, v] == 0:
                    n_edges += 1
                A_boot[u, v] = 1.0
                A_boot[v, u] = 1.0
        rho = (2 * n_edges / (N_nb * (N_nb - 1))) if N_nb > 1 else 0.0
        rho_per_boot.append(rho)

        # ── Máscara de papers en este bootstrap ────────────────────────
        in_boot = np.zeros(N_p, dtype=bool)
        for members in g_boot["nodes"].values():
            if members:
                in_boot[np.array(members, dtype=np.int64)] = True

        # ── Reproducibilidad por arista base (matmul) ──────────────────
        reproduced_b = 0
        testeable_b = 0
        for (pu, pv) in base_edge_papers:
            pu_in = pu[in_boot[pu]]
            pv_in = pv[in_boot[pv]]
            if len(pu_in) == 0 or len(pv_in) == 0:
                continue
            testeable_b += len(pu_in) * len(pv_in)
            R_u = M_boot[pu_in, :]          # (n_ub, N_nb)
            R_v = M_boot[pv_in, :]          # (n_vb, N_nb)
            tmp = R_u @ A_boot              # (n_ub, N_nb)
            rep_mat = tmp @ R_v.T           # (n_ub, n_vb)
            reproduced_b += int((rep_mat > 0).sum())

        if testeable_b > 0:
            k_per_boot.append(reproduced_b)
            n_per_boot.append(testeable_b)

    if not k_per_boot:
        return 0.0, 0.0, 0.0, 0

    rho_null = float(np.mean(rho_per_boot))
    s_bar = float(np.mean([k / n for k, n in zip(k_per_boot, n_per_boot)]))

    # Z̄ analítico per bootstrap bajo H₀ binomial
    z_scores = []
    for k, n, rho in zip(k_per_boot, n_per_boot, rho_per_boot):
        mu  = n * rho
        sig = np.sqrt(n * rho * (1.0 - rho))
        if sig > 0:
            z_scores.append((k - mu) / sig)
    z_bar = float(np.mean(z_scores)) if z_scores else 0.0

    return s_bar, z_bar, rho_null, n_pairs_base



# ─────────────────────────────────────────────────────────────────────
# Caché de resultados
# ─────────────────────────────────────────────────────────────────────

def save_cache(results, path=CACHE_PATH):
    """Guarda results en NPZ aplanando las claves anidadas."""
    flat = {}
    for key in results:            # "PCA", "UMAP"
        for attr in ["mean", "lo", "hi", "eps"]:
            for dim in DIMS:
                flat[f"{key}_{attr}_{dim}"] = np.asarray(results[key][attr][dim])
        for dim in DIMS:
            flat[f"{key}_null_{dim}"]  = np.array([results[key]["null"][dim]])
            flat[f"{key}_betas_{dim}"] = np.array([results[key]["betas"][dim]])
        flat[f"{key}_j_bar"]        = np.array([results[key]["j_bar"]])
        flat[f"{key}_s_bar"]        = np.array([results[key]["s_bar"]])
        flat[f"{key}_z_bar"]        = np.array([results[key].get("z_bar", float("nan"))])
        flat[f"{key}_rho_null"]     = np.array([results[key]["rho_null"]])
        flat[f"{key}_n_pairs"]      = np.array([results[key]["n_pairs"]])
        flat[f"{key}_j_bar_null95"] = np.array([results[key]["j_bar_null95"]])
        flat[f"{key}_edge_null95"]  = np.array([results[key]["edge_null95"]])
        flat[f"{key}_j_bar_null"]   = np.asarray(results[key]["j_bar_null"])
        flat[f"{key}_edge_null"]    = np.asarray(results[key]["edge_null"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **flat)
    print(f"Caché guardada en: {path}")


def load_cache(path=CACHE_PATH):
    """Carga results desde NPZ. Devuelve None si no existe o es de versión anterior."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    # Verificar claves mínimas esperadas
    required = ([f"{key}_{attr}_{dim}"
                 for key in ["PCA", "UMAP"]
                 for attr in ["mean", "lo", "hi", "eps"]
                 for dim in DIMS] +
                [f"{key}_j_bar" for key in ["PCA", "UMAP"]])
    if not all(k in data for k in required):
        print("Caché incompleta o de versión anterior — recomputando.")
        return None
    results = {}
    for key in ["PCA", "UMAP"]:
        results[key] = {}
        for attr in ["mean", "lo", "hi", "eps"]:
            results[key][attr] = {dim: data[f"{key}_{attr}_{dim}"] for dim in DIMS}
        results[key]["null"]      = {dim: float(data[f"{key}_null_{dim}"][0]) for dim in DIMS}
        results[key]["betas"]     = {dim: int(data[f"{key}_betas_{dim}"][0])  for dim in DIMS}
        results[key]["j_bar"]        = float(data[f"{key}_j_bar"][0])
        results[key]["s_bar"]     = (float(data[f"{key}_s_bar"][0])
                                       if f"{key}_s_bar" in data else float("nan"))
        results[key]["z_bar"]     = (float(data[f"{key}_z_bar"][0])
                                       if f"{key}_z_bar" in data else float("nan"))
        results[key]["rho_null"]  = (float(data[f"{key}_rho_null"][0])
                                       if f"{key}_rho_null" in data else float("nan"))
        results[key]["n_pairs"]   = (int(data[f"{key}_n_pairs"][0])
                                       if f"{key}_n_pairs" in data else 0)
        results[key]["j_bar_null95"] = (float(data[f"{key}_j_bar_null95"][0])
                                          if f"{key}_j_bar_null95" in data else float("nan"))
        results[key]["edge_null95"]  = (float(data[f"{key}_edge_null95"][0])
                                          if f"{key}_edge_null95" in data else float("nan"))
        results[key]["j_bar_null"]   = (data[f"{key}_j_bar_null"]
                                          if f"{key}_j_bar_null" in data else np.array([]))
        results[key]["edge_null"]    = (data[f"{key}_edge_null"]
                                          if f"{key}_edge_null" in data else np.array([]))
    return results


# ─────────────────────────────────────────────────────────────────────
# Figura principal
# ─────────────────────────────────────────────────────────────────────

def plot_validacion(results, output_path):
    """Genera la figura 2 paneles × 2 dimensiones homológicas (H₀, H₁).

    results: dict con claves "PCA" y "UMAP", cada una con:
      mean[dim], lo[dim], hi[dim], eps[dim], null[dim], betas[dim],
      j_bar, edge_frac, n_edges, n_robust
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Validación interna de nervios Mapper — corpus arXiv\n"
        "Banda bootstrap 5–95% (naranja) · Línea nula 95% (gris discontinua)",
        fontsize=11,
    )

    COLORES = ["#D32F2F", "#1976D2", "#388E3C"]   # λ₁, λ₂, λ₃
    LABELS  = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    NERVIOS = [("PCA", "PCA-nervio\n(lente PCA 2D, clustering PCA 20D)"),
               ("UMAP", "UMAP-nervio\n(lente UMAP 2D, clustering UMAP 10D)")]

    for row, (key, title) in enumerate(NERVIOS):
        res = results[key]
        for col, dim in enumerate(DIMS):
            ax   = axes[row, col]
            eps  = res["eps"][dim]
            mean = res["mean"][dim]
            lo   = res["lo"][dim]
            hi   = res["hi"][dim]
            null = res["null"][dim]
            beta = res["betas"][dim]

            for k in range(N_CAPAS):
                if mean[k].max() > 1e-8:
                    ax.plot(eps, mean[k], color=COLORES[k], lw=1.6,
                            label=LABELS[k])
                    ax.fill_between(eps, lo[k], hi[k],
                                    color=COLORES[k], alpha=0.15)

            ax.axhline(null, color="#888888", lw=1.0, ls="--",
                       label="nulo 95%")

            ax.set_ylim(0, 0.5)
            ax.set_xlabel(r"$\varepsilon$", fontsize=9)
            ax.set_ylabel(r"$\lambda_k(\varepsilon)$", fontsize=9)
            ax.set_title(
                f"$H_{dim}$   ($\\beta_{dim} = {beta}$)",
                fontsize=10,
            )
            ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                j_bar    = res["j_bar"]
                s_bar    = res.get("s_bar", float("nan"))
                rho_null = res.get("rho_null", float("nan"))
                ax.set_ylabel(
                    f"{title}\n"
                    r"$\lambda_k(\varepsilon)$" +
                    f"\n$\\bar{{J}}={j_bar:.2f}$  "
                    f"$\\bar{{S}}={s_bar:.2f}$  $\\bar{{\\rho}}={rho_null:.2f}$",
                    fontsize=7.5,
                )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Diagrama guardado en: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(force_recompute=False):
    npz_path = os.path.join(DATA_DIR, "arxiv_ciencia.npz")

    # ── Intentar cargar caché ──────────────────────────────────────────
    if not force_recompute:
        results = load_cache()
        if results is not None:
            print(f"Cargando resultados desde caché: {CACHE_PATH}")
            output = os.path.join(OUT_DIR, "validacion-nervio-ciencia.pdf")
            print(f"\nGenerando figura: {output}")
            plot_validacion(results, output)
            plot_validacion(results, output.replace(".pdf", ".png"))
            return

    if not os.path.exists(npz_path):
        print(f"⚠  {npz_path} no encontrado — abortando")
        return

    print("Cargando corpus arXiv...")
    data    = np.load(npz_path, allow_pickle=True)
    emb     = data["embeddings"]                         # (N, 384)
    emb_red = PCA(n_components=50, random_state=42).fit_transform(emb)

    print(f"  {emb.shape[0]} papers, embeddings {emb.shape[1]}D → PCA 50D")

    print("\nConstruyendo nervios base...")
    graph_pca, graph_umap = _build_graphs(emb, emb_red)
    n_aristas_pca  = sum(len(v) for v in graph_pca["links"].values())  // 2
    n_aristas_umap = sum(len(v) for v in graph_umap["links"].values()) // 2
    print(f"  PCA-nervio:  {len(graph_pca['nodes'])} nodos, {n_aristas_pca} aristas")
    print(f"  UMAP-nervio: {len(graph_umap['nodes'])} nodos, {n_aristas_umap} aristas")

    # Funciones de construcción para bootstrap/nulo
    def build_pca(e, er):
        import kmapper as km
        from sklearn.cluster import DBSCAN
        m = km.KeplerMapper(verbose=0)
        l = PCA(n_components=2,  random_state=42).fit_transform(e)
        c = PCA(n_components=20, random_state=42).fit_transform(e)
        return m.map(l, c,
                     cover=km.Cover(n_cubes=10, perc_overlap=0.4),
                     clusterer=DBSCAN(eps=2.0, min_samples=3))

    def build_umap(e, er):
        import kmapper as km
        import umap as umap_lib
        from sklearn.cluster import DBSCAN
        m  = km.KeplerMapper(verbose=0)
        l  = umap_lib.UMAP(n_components=2,  n_neighbors=15,
                            random_state=42).fit_transform(er)
        c  = umap_lib.UMAP(n_components=10, n_neighbors=15,
                            random_state=42).fit_transform(er)
        return m.map(l, c,
                     cover=km.Cover(n_cubes=10, perc_overlap=0.4),
                     clusterer=DBSCAN(eps=0.5, min_samples=3))

    results = {}
    for key, graph_base, build_fn in [
        ("PCA",  graph_pca,  build_pca),
        ("UMAP", graph_umap, build_umap),
    ]:
        print(f"\n{'─'*50}")
        print(f"Procesando {key}-nervio...")

        # β_k del nervio base
        dgms_base = ripser_on_graph(graph_base, maxdim=max(DIMS))
        betas = {dim: betti_at_epsilon(dgms_base, dim, eps=1.0) for dim in DIMS}
        print(f"  β₀={betas[0]}, β₁={betas[1]}")

        # Pasos 2–3: bootstrap (H0+H1 en un solo pase) + guardar grafos para 5–6
        print(f"  Pasos 2-3 — bootstrap (B={B_BOOT}, H₀+H₁)...", end=" ", flush=True)
        boot_graphs, means, los, his, epss = bootstrap_full(emb, emb_red, build_fn)
        print(f"✓  ({len(boot_graphs)} remuestras válidas)")

        nulls = {}
        for dim in DIMS:
            print(f"  Línea nula H{dim} (B={B_NULL})...", end=" ", flush=True)
            nulls[dim] = null_line(emb, emb_red, build_fn, dim=dim)
            print(f"✓  ({nulls[dim]:.4f})")

        # Paso 4: sensibilidad a parámetros
        print(f"  Paso 4 — sensibilidad a parámetros (9 configs)...")
        sweep = sensitivity_sweep(emb, emb_red, key)
        n_ok  = sum(
            1 for v in sweep.values()
            if v["betas"].get(0) == 1 and v["betas"].get(1) == 0
        )
        print(f"    {n_ok}/9 configuraciones con (β₀,β₁)=(1,0)")

        # Paso 5: estabilidad de membresía (reutiliza boot_graphs)
        print(f"  Paso 5 — estabilidad de membresía...", end=" ", flush=True)
        j_bar, _ = membership_stability(graph_base, boot_graphs)
        print(f"✓  J̄={j_bar:.3f}")

        # Paso 6: reproducibilidad de pares (reutiliza boot_graphs)
        print(f"  Paso 6 — reproducibilidad de pares...", end=" ", flush=True)
        s_bar, z_bar, rho_null, n_pairs = edge_stability_p2p(graph_base, boot_graphs)
        print(f"✓  S̄={s_bar:.3f}  ρ̅={rho_null:.3f}  ({n_pairs} pares base)")

        # Línea nula combinatoria: J̄ y S̄ bajo permutación de coordenadas
        print(f"  Línea nula combinatoria "
              f"(B_outer={B_NULL_STAB_OUTER}, B_inner={B_NULL_STAB_INNER})...",
              flush=True)
        j95, e95, j_null, e_null = null_stability(emb, emb_red, build_fn)
        print(f"    nulo 95%: J̄_null={j95:.3f}, S̄_null={e95:.3f}")

        results[key] = dict(
            mean=means, lo=los, hi=his, eps=epss,
            null=nulls, betas=betas,
            j_bar=j_bar, s_bar=s_bar, z_bar=z_bar,
            rho_null=rho_null, n_pairs=n_pairs,
            j_bar_null95=j95, edge_null95=e95,
            j_bar_null=np.asarray(j_null), edge_null=np.asarray(e_null),
        )

    save_cache(results)

    output = os.path.join(OUT_DIR, "validacion-nervio-ciencia.pdf")
    print(f"\nGenerando figura: {output}")
    plot_validacion(results, output)
    plot_validacion(results, output.replace(".pdf", ".png"))

    # ── Resumen final ──────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("RESUMEN DE VALIDACIÓN")
    print(f"{'═'*50}")
    for key in ["PCA", "UMAP"]:
        r = results[key]
        print(f"\n{key}-nervio:")
        print(f"  β₀={r['betas'][0]}, β₁={r['betas'][1]}")
        print(f"  Membresía  J̄ = {r['j_bar']:.3f}  "
              f"(nulo 95% = {r['j_bar_null95']:.3f}; "
              f"{'> nulo' if r['j_bar'] > r['j_bar_null95'] else '≤ nulo'})")
        print(f"  Grafo  S̄ = {r['s_bar']:.3f}  "
              f"(nulo permutación S_95 = {r['edge_null95']:.3f};  "
              f"{'> nulo' if r['s_bar'] > r['edge_null95'] else '≤ nulo'};  "
              f"exceso sobre ρ̅ = {r['s_bar'] - r['rho_null']:+.3f})  "
              f"({r['n_pairs']} pares base)")

if __name__ == "__main__":
    import sys
    force = "--recompute" in sys.argv
    main(force_recompute=force)
