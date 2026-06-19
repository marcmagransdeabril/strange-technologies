"""
Receta 7: UMAP y la reducción topológica.

Demuestra que UMAP preserva la topología de un toro ruidoso en R^50
mejor que PCA, usando homología persistente y paisajes de persistencia
como validación.

Salidas:
  ../../diagrams/tda/umap-toro.pdf  — comparación de reducciones, H₁ y paisaje
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Rutas ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "diagrams", "tda")


def generar_toro(n=1000, R=2.0, r=1.0, dim_ambiente=50,
                 ruido=0.08, seed=42):
    """Genera n puntos sobre un toro en R^dim_ambiente con ruido.

    Usa R=2, r=1 (radios 2:1) para que ambos ciclos del toro tengan
    persistencia similar y sean claramente detectables.
    Devuelve también el toro en R^3 (antes de la incrustación) como referencia.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)

    # Toro en R^3
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    toro_3d = np.column_stack([x, y, z])
    toro_3d += rng.normal(0, ruido, toro_3d.shape)

    # Incrustar en R^dim_ambiente con proyección aleatoria NO lineal
    # Usamos características senoidales aleatorias: puntos = sin(toro_3d @ W + fase)
    # Esto es una homeomorfía suave (UMAP preserva vecindades) pero no lineal
    # (PCA no puede invertirla y pierde el ciclo menor).
    rng2 = np.random.default_rng(seed + 1)
    if dim_ambiente > 3:
        W = rng2.standard_normal((3, dim_ambiente)) * 0.7
        fase = rng2.uniform(0, 2 * np.pi, dim_ambiente)
        puntos = np.sin(toro_3d @ W + fase)
        puntos += rng2.normal(0, ruido * 0.3, puntos.shape)
    else:
        puntos = toro_3d.copy()
    return puntos, toro_3d, theta, phi


def reducir_pca(puntos, n_components=3):
    """Reduce a n_components con PCA."""
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=42).fit_transform(puntos)


def reducir_umap(puntos, n_components=3):
    """Reduce a n_components con UMAP (n_neighbors=15, min_dist=0.1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        ).fit_transform(puntos)


def calcular_h1(puntos, maxdim=1):
    """Calcula homología persistente hasta dimensión maxdim."""
    from ripser import ripser
    return ripser(puntos, maxdim=maxdim)["dgms"]


def calcular_landscape(dgms, n_pts=500, n_capas=3):
    """Calcula las n_capas primeras funciones del paisaje de persistencia H₁.

    λ_k(ε) es el k-ésimo máximo pointwise de todas las tiendas de campaña,
    donde cada intervalo [b, d] contribuye con la tienda
        tent(ε) = max(0, (d-b)/2 - |ε - (b+d)/2|).

    Devuelve (lambdas, eps) donde lambdas[k] es λ_{k+1}:
      lambdas[0] = λ₁ (envolvente), lambdas[1] = λ₂, etc.
    """
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))

    if len(finite) == 0:
        eps = np.linspace(0, 1, n_pts)
        return np.zeros((n_capas, n_pts)), eps

    eps_max = finite[:, 1].max() * 1.1
    eps = np.linspace(0, eps_max, n_pts)
    tents = np.zeros((len(finite), n_pts))
    for i, (b, d) in enumerate(zip(finite[:, 0], finite[:, 1])):
        mid = (b + d) / 2.0
        half = (d - b) / 2.0
        tents[i] = np.maximum(half - np.abs(eps - mid), 0)

    # λ_k es el k-ésimo máximo pointwise: ordenar filas en sentido descendente
    tents_sorted = np.sort(tents, axis=0)[::-1]
    result = np.zeros((n_capas, n_pts))
    n = min(n_capas, len(tents_sorted))
    result[:n] = tents_sorted[:n]
    return result, eps


def calcular_umbral_nulo(coords, n_iter=100, n_pts_null=300, seed=0):
    """Umbral escalar al 95% bajo H₀ (puntos uniformes).

    Para cada iteración genera n_pts_null puntos uniformes en el bounding-box
    de coords, calcula el máximo de λ₁ y devuelve el percentil 95 de esos máximos
    — un escalar, para plotear con axhline.
    """
    rng = np.random.default_rng(seed)
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    maximos = []
    for _ in range(n_iter):
        pts = rng.uniform(low, high, size=(n_pts_null, coords.shape[1]))
        dgms_null = calcular_h1(pts)
        lambdas_null, _ = calcular_landscape(dgms_null)
        maximos.append(float(lambdas_null[0].max()))
    return float(np.percentile(maximos, 95))


def contar_ciclos_persistentes(dgms, umbral_relativo=0.3):
    """Cuenta ciclos en H1 con persistencia > umbral_relativo * max_persistencia."""
    h1 = dgms[1]
    if len(h1) == 0:
        return 0, np.array([])
    pers = h1[:, 1] - h1[:, 0]
    umbral = umbral_relativo * pers.max()
    return int((pers > umbral).sum()), pers


def comparar_reducciones(puntos, toro_3d):
    """Compara el toro original, PCA y UMAP en preservación topológica."""
    resultados = {}

    items = [
        ("Toro $\\mathbb{R}^3$", lambda p, **kw: toro_3d),
        ("PCA (lineal)", reducir_pca),
        ("UMAP (topológico)", reducir_umap),
    ]
    for metodo, red_fn in items:
        coords = red_fn(puntos, n_components=3)
        dgms = calcular_h1(coords)
        lambdas, eps_lam = calcular_landscape(dgms)
        umbral = calcular_umbral_nulo(coords)
        h1 = dgms[1]
        pers = (h1[:, 1] - h1[:, 0]) if len(h1) > 0 else np.array([])
        # Capas significativas: aquellas cuyo pico supera el umbral nulo
        n_ciclos = int(sum(
            lambdas[k].max() > umbral for k in range(lambdas.shape[0])
        ))
        resultados[metodo] = {
            "coords": coords, "dgms": dgms,
            "lambdas": lambdas, "eps_lam": eps_lam,
            "umbral": umbral,
            "n_ciclos": n_ciclos, "pers": pers,
        }

    return resultados


def generar_diagrama(toro_3d, theta, resultados, output_path):
    """Figura 3 columnas × 3 filas: (toro original, PCA, UMAP) × (3D, H1, paisaje)."""
    fig = plt.figure(figsize=(14, 11))
    titulos = list(resultados.keys())

    for col, metodo in enumerate(titulos):
        res = resultados[metodo]
        coords = res["coords"]
        dgms = res["dgms"]
        lambdas = res["lambdas"]
        eps_lam = res["eps_lam"]
        umbral = res["umbral"]

        # ── Fila 1: scatter 3D ──────────────────────────────────────
        ax = fig.add_subplot(3, 3, col + 1, projection="3d")
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=theta, cmap="hsv", s=5, alpha=0.7, vmin=0, vmax=2 * np.pi,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.55, aspect=15)
        cb.set_label(r"$\theta$", fontsize=7)
        cb.set_ticks([0, np.pi, 2 * np.pi])
        cb.set_ticklabels(["0", r"$\pi$", r"$2\pi$"], fontsize=6)
        ax.set_title(metodo, fontsize=11, fontweight="bold")
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=5)
        # Aspecto cúbico
        lims = np.array([
            [coords[:, 0].min(), coords[:, 0].max()],
            [coords[:, 1].min(), coords[:, 1].max()],
            [coords[:, 2].min(), coords[:, 2].max()],
        ])
        mid = lims.mean(axis=1)
        rng_ax = (lims[:, 1] - lims[:, 0]).max() / 2
        ax.set_xlim(mid[0] - rng_ax, mid[0] + rng_ax)
        ax.set_ylim(mid[1] - rng_ax, mid[1] + rng_ax)
        ax.set_zlim(mid[2] - rng_ax, mid[2] + rng_ax)

        # ── Fila 2: diagrama de persistencia H1 ─────────────────────
        ax = fig.add_subplot(3, 3, col + 4)
        h1 = dgms[1]
        if len(h1) > 0:
            pers = h1[:, 1] - h1[:, 0]
            significativos = (pers / 2) > umbral
            ax.scatter(
                h1[~significativos, 0], h1[~significativos, 1],
                c="gray", s=15, alpha=0.5, label="ruido",
            )
            ax.scatter(
                h1[significativos, 0], h1[significativos, 1],
                c="red", s=70, marker="*", zorder=5,
                label=f"ciclos ({significativos.sum()})",
            )
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.5)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Nacimiento", fontsize=8)
        ax.set_ylabel("Muerte", fontsize=8)
        ax.set_title(f"$H_1$: {res['n_ciclos']} ciclos", fontsize=9)
        ax.legend(loc="lower right", fontsize=7)

        # ── Fila 3: paisaje de persistencia λ₁, λ₂, λ₃ ─────────────
        ax = fig.add_subplot(3, 3, col + 7)
        COLORES_LAND = ["#D32F2F", "#1976D2", "#388E3C"]
        for k in range(lambdas.shape[0]):
            if lambdas[k].max() > 1e-6:
                ax.plot(eps_lam, lambdas[k],
                        c=COLORES_LAND[k], lw=1.4,
                        label=f"$\\lambda_{{{k + 1}}}$")
        ax.axhline(umbral, color="#888888", lw=0.9, ls="--",
                   label="umbral 95%")
        ax.set_xlabel(r"$\varepsilon$", fontsize=8)
        ax.set_ylabel(r"$\lambda_k(\varepsilon)$", fontsize=8)
        ax.set_title("Paisaje $H_1$", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Diagrama guardado en: {output_path}")


def main():
    """Ejecuta la receta completa."""
    print("Receta 7: UMAP y la reducción topológica")
    print("=" * 50)

    print("\n1. Generando toro (1000 puntos, R=2, r=1, R^50, ruido=0.08)...")
    puntos, toro_3d, theta, phi = generar_toro(
        n=1000, R=2.0, r=1.0, dim_ambiente=50, ruido=0.08,
    )
    print(f"   Forma: {puntos.shape}")

    print("\n2. Reduciendo, calculando H1 y paisajes...")
    resultados = comparar_reducciones(puntos, toro_3d)

    print("\n3. Resultados:")
    print(f"   {'Método':<6} {'Ciclos H1':>10} {'Max pers':>10} {'Umbral':>8}")
    print(f"   {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for metodo, res in resultados.items():
        max_p = res["pers"].max() if len(res["pers"]) > 0 else 0.0
        print(f"   {metodo:<6} {res['n_ciclos']:>10} {max_p:>10.3f}"
              f" {res['umbral']:>8.3f}")
    print(f"\n   Esperado: β₁ = 2 (el toro tiene dos ciclos independientes)")

    output_path = os.path.join(OUT_DIR, "umap-toro.pdf")
    print(f"\n4. Generando diagrama...")
    generar_diagrama(toro_3d, theta, resultados, output_path)
    output_png = output_path.replace(".pdf", ".png")
    generar_diagrama(toro_3d, theta, resultados, output_png)


if __name__ == "__main__":
    main()
