"""Figures for module key 'ml3' (clustering and mixture models).

Figure names start with 'ml3_'. All data are synthetic (fixed seeds) unless a
caption says the figure was computed from a supplied course file.

A synthetic well log, "well S-1" (synth_well below), is the running example of
the day: four lithologies (shale, brine sand, gas sand, limestone) with GR,
RHOB, NPHI and resistivity values in plausible ranges, blocky beds, a
gradational (coarsening-upward) unit and a short tool-averaging window that
creates intermediate readings at bed boundaries.
"""
from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

import warnings
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.sparse import diags
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (KMeans, DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering, kmeans_plusplus)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

KEY = 'ml3'
fig = register(KEY)

GREY = '#56687A'
NOISE = '#8C99A6'
DATA = ROOT / 'machine_learning_course' / 'Module_3_Unsupervised_Learning_1' / 'exercises'


# ----------------------------------------------------------------- helpers
def match_labels(labels, ref):
    """Relabel cluster ids to agree as far as possible with ref (Hungarian
    matching) so that the same group keeps the same colour across panels.
    Noise (-1) is kept. Unmatched clusters get the next free ids."""
    labels = np.asarray(labels)
    ref = np.asarray(ref)
    labs = [l for l in np.unique(labels) if l >= 0]
    refs = [r for r in np.unique(ref) if r >= 0]
    n = max(len(labs), len(refs))
    C = np.zeros((len(labs), n))
    for i, l in enumerate(labs):
        for j, r in enumerate(refs):
            C[i, j] = np.sum((labels == l) & (ref == r))
    rows, cols = linear_sum_assignment(-C)
    targets = list(refs) + [max(refs + [-1]) + 1 + k for k in range(n - len(refs))]
    mapping = {labs[r]: targets[c] for r, c in zip(rows, cols)}
    out = labels.copy()
    for l, t in mapping.items():
        out[labels == l] = t
    return out


def cluster_tints(k, amount=.80):
    return ListedColormap([tint(SERIES[i % 6], amount) for i in range(k)])


def shade_regions(ax, predict, xlim, ylim, k, res=300, amount=.80, lines=True):
    """Shade the plane by predicted cluster (predict: (n, 2) array -> labels)."""
    xx, yy = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, int(res * .8)))
    zz = np.asarray(predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.arange(k + 1) - .5, cmap=cluster_tints(k, amount), zorder=0)
    if lines:
        ax.contour(xx, yy, zz, levels=np.arange(k) + .5, colors=['#8FA1B3'], linewidths=.9, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def centers(ax, C, s=190, colors=None, z=6):
    colors = colors or [SERIES[i % 6] for i in range(len(C))]
    ax.scatter(C[:, 0], C[:, 1], marker='X', s=s, c=colors, edgecolors=INK, linewidths=1.1, zorder=z)


def panel(ax, title, fs=14):
    ax.set_title(title, fontsize=fs, loc='left')


def titled(ax, title, sub=None, fs=14, sfs=12.5, color=INK):
    """Panel title with an optional second line in regular weight, so that
    numbers about a panel sit above it instead of on top of the data."""
    ax.set_title(title, fontsize=fs, loc='left', color=color, pad=10 if sub is None else 12 + sfs * 1.45)
    if sub:
        ax.annotate(sub, (0, 1), xycoords='axes fraction', xytext=(0, 7), textcoords='offset points',
                    ha='left', va='bottom', fontsize=sfs, color=INK)


HALO = [pe.withStroke(linewidth=3.5, foreground='white')]


def free_gap(dn, y0, y1, xmax):
    """Widest horizontal interval of a scipy dendrogram plot with no vertical
    link crossing the height band [y0, y1]. Returns (x_left, x_right)."""
    xs = [0.0, float(xmax)]
    for ic, dc in zip(dn['icoord'], dn['dcoord']):
        for x, lo, hi in ((ic[0], dc[0], dc[1]), (ic[3], dc[3], dc[2])):
            if lo < y1 and hi > y0:
                xs.append(x)
    xs = np.sort(xs)
    k = int(np.argmax(np.diff(xs)))
    return xs[k], xs[k + 1]


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def ellipse_from_cov(ax, mean, cov, color, n_std=(1, 2), alpha=.14, lw=1.6, z=2):
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for k in n_std:
        w, h = 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0])
        ax.add_patch(Ellipse(mean, w, h, angle=angle, fc=color if k == n_std[0] else 'none',
                             alpha=alpha if k == n_std[0] else 1, ec=color, lw=lw, zorder=z))
        if k == n_std[0]:
            ax.add_patch(Ellipse(mean, w, h, angle=angle, fc='none', ec=color, lw=lw, zorder=z))


def gmm_covs(m):
    """Full 2x2 covariance matrices for any sklearn covariance_type."""
    k, d = m.means_.shape
    if m.covariance_type == 'full':
        return m.covariances_
    if m.covariance_type == 'tied':
        return np.repeat(m.covariances_[None], k, 0)
    if m.covariance_type == 'diag':
        return np.array([np.diag(c) for c in m.covariances_])
    return np.array([np.eye(d) * c for c in m.covariances_])


def rounded_hull(P, r, n=24):
    """Convex hull of points P buffered by radius r (closed polygon)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(t), np.sin(t)] * r
    Q = (np.asarray(P)[:, None, :] + ring[None]).reshape(-1, 2)
    h = ConvexHull(Q)
    return Q[h.vertices]


# ------------------------------------------------------- synthetic well S-1
LITH = ['Shale', 'Brine sand', 'Gas sand', 'Limestone']
LOGS = ['GR', 'RHOB', 'NPHI', 'LRT']
#                   GR    RHOB  NPHI  log10(RT)
LITH_MEAN = np.array([[105, 2.52, 0.32, 0.40],
                      [45, 2.26, 0.25, 0.10],
                      [40, 2.20, 0.16, 1.55],
                      [22, 2.64, 0.06, 2.10]])
LITH_SD = np.array([[10, .035, .025, .10],
                    [10, .035, .025, .10],
                    [10, .035, .025, .15],
                    [7, .030, .020, .15]])


def synth_well(seed=7, top=2000.0, step=0.5):
    """Synthetic well S-1: layered lithologies with tool averaging and noise.

    Beds 7-24 m thick, one coarsening-upward unit (bed 7) and one thin-bedded
    gas-sand/shale unit (bed 11, beds 1.0-1.5 m, below the 2.5 m averaging
    window). Returns DEPTH (m), LITH (0-3), BED, GR (API), RHOB (g/cm3),
    NPHI (v/v), LRT (log10 ohm.m) and RT (ohm.m)."""
    rng = np.random.default_rng(seed)
    seq = [0, 1, 0, 2, 0, 3, 0, 'cu', 0, 2, 0, 'tb', 0, 3, 0, 1, 0, 2, 0, 3, 0]
    rows, z = [], top
    for bed, lith in enumerate(seq):
        if lith == 'tb':                       # thin beds below tool resolution
            for k in range(14):
                l = 2 if k % 2 == 0 else 0
                shift = rng.normal(0, .35, 4) * LITH_SD[l]
                for _ in range(int(rng.integers(2, 4))):
                    rows.append((z, l, bed, *(LITH_MEAN[l] + shift)))
                    z += step
            continue
        n = int(round(rng.uniform(7, 24) / step))
        if lith == 'cu':                       # coarsening-upward: sand on top, shale at base
            f = np.linspace(0, 1, n)
            for k in range(n):
                m = (1 - f[k]) * LITH_MEAN[1] + f[k] * LITH_MEAN[0]
                rows.append((z, 1 if f[k] < .5 else 0, bed, *m))
                z += step
            continue
        shift = rng.normal(0, .35, 4) * LITH_SD[lith]
        for _ in range(n):
            rows.append((z, lith, bed, *(LITH_MEAN[lith] + shift)))
            z += step
    d = pd.DataFrame(rows, columns=['DEPTH', 'LITH', 'BED'] + LOGS)
    sd = LITH_SD[d.LITH.values]
    for j, c in enumerate(LOGS):
        v = d[c].values
        w = np.ones(5) / 5                     # 2.5 m tool-averaging window
        vs = np.convolve(np.pad(v, 2, mode='edge'), w, mode='valid')
        d[c] = vs + rng.normal(0, 1, len(v)) * sd[:, j] * .75
    d['RT'] = 10 ** d['LRT']
    return d


def well_features(d):
    return d[LOGS].values


# =================================================================== OPENING
@fig
def energy_uses():
    """Four energy clustering tasks, each computed on synthetic data."""
    f, axs = sub(4, height=4.3)
    # (a) electrofacies: k-means on standardized logs of well S-1
    d = synth_well()
    Z = StandardScaler().fit_transform(well_features(d))
    lab = match_labels(KMeans(4, n_init=10, random_state=0).fit_predict(Z), d.LITH.values)
    ax = axs[0]
    points(ax, np.c_[d.NPHI, d.RHOB], lab, s=13)
    ax.set_ylim(2.78, 2.05)
    ax.set_xlim(-.02, .42)
    clean(ax, 'NPHI (v/v)', 'RHOB (g/cm³)')
    panel(ax, 'Electrofacies')
    # (b) production-profile typing: k-means on peak-normalized decline curves
    rng = np.random.default_rng(3)
    t = np.arange(36)
    curves, kinds = [], []
    for i in range(36):
        kind = i % 3
        if kind == 0:     # steep hyperbolic decline
            q = 1 / (1 + .8 * rng.uniform(.25, .35) * t) ** (1 / .8)
        elif kind == 1:   # shallow decline
            q = 1 / (1 + .5 * rng.uniform(.04, .07) * t) ** (1 / .5)
        else:             # plateau then decline
            p = rng.integers(9, 14)
            q = np.where(t < p, 1, np.exp(-rng.uniform(.08, .12) * (t - p)))
        q = q * np.exp(rng.normal(0, .05, len(t)))
        curves.append(q / q.max())
        kinds.append(kind)
    Q = np.array(curves)
    labq = match_labels(KMeans(3, n_init=10, random_state=0).fit_predict(Q), np.array(kinds))
    ax = axs[1]
    for q, l in zip(Q, labq):
        ax.plot(t, q, color=SERIES[l], lw=.9, alpha=.45)
    for l in range(3):
        ax.plot(t, Q[labq == l].mean(0), color=SERIES[l], lw=3)
    clean(ax, 'Months on production', 'Rate / peak rate')
    ax.set_ylim(0, 1.08)
    panel(ax, 'Production types')
    # (c) operating regimes: Gaussian mixture on ESP drive frequency and intake pressure
    rng = np.random.default_rng(5)
    regimes = [((50, 95), (1.0, 5.0), 170), ((57, 70), (1.2, 6.0), 120), ((40, 126), (1.5, 6.5), 70)]
    X, yr = [], []
    for r, ((mf, mp), (sf, sp), n) in enumerate(regimes):
        fq = rng.normal(mf, sf, n)
        X.append(np.c_[fq, mp - 1.6 * (fq - mf) + rng.normal(0, sp, n)])
        yr += [r] * n
    X = np.vstack(X)
    g = GaussianMixture(3, n_init=5, random_state=0).fit(StandardScaler().fit_transform(X))
    labg = match_labels(g.predict(StandardScaler().fit_transform(X)), np.array(yr))
    ax = axs[2]
    points(ax, X, labg, s=13)
    clean(ax, 'ESP drive frequency (Hz)', 'Intake pressure (bar)')
    panel(ax, 'Operating regimes')
    # (d) zonation: depth-constrained Ward clustering of well S-1
    n = len(d)
    A = diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1])
    zl = AgglomerativeClustering(8, linkage='ward', connectivity=A).fit_predict(Z)
    ax = axs[3]
    dep = d.DEPTH.values
    for z, (a, b, _) in enumerate(runs(zl)):          # same style as the zonation figure
        ax.axhspan(dep[a] - .25, dep[b - 1] + .25, color='#E3EBF3' if z % 2 == 0 else 'white', lw=0, zorder=0)
        if z:
            ax.axhline(dep[a] - .25, color=TEAL, lw=1.8, zorder=2)
    ax.plot(d.GR, dep, color=INK, lw=1.0, zorder=3)
    ax.set_ylim(d.DEPTH.max(), d.DEPTH.min())
    ax.set_xlim(0, 150)
    clean(ax, 'GR (API)', 'Depth (m)', grid=False)
    panel(ax, 'Log zonation')
    save('ml3_energy_uses', f, 'Synthetic data: well S-1 (seed 7) k-means K=4 and depth-constrained Ward; '
         'Arps-type decline curves (seed 3) k-means K=3; ESP regimes (seed 5) Gaussian mixture K=3.',
         kind='computed')


def warmup_profiles():
    t = np.arange(0, 37)
    A = 300 / (1 + .8 * .25 * t) ** (1 / .8)
    B = 3 * A
    C = np.where(t < 14, 170.0, 170 * np.exp(-.10 * (t - 14)))
    return t, A, B, C


@fig
def profiles_warmup():
    t, A, B, C = warmup_profiles()
    f, axs = half()
    ax = axs[0]
    for q, name, col in [(A, 'Well A', BLUE), (B, 'Well B', ORANGE), (C, 'Well C', TEAL)]:
        ax.plot(t, q, color=col, lw=2.8)
    ax.text(6.5, 3 * 300 / (1 + .2 * 6.5) ** 1.25 + 25, 'Well B', color=ORANGE, fontsize=14, fontweight='medium')
    ax.text(7.2, 186, 'Well C', color=TEAL, fontsize=14, fontweight='medium')
    ax.text(8.5, 300 / (1 + .2 * 8.5) ** 1.25 - 52, 'Well A', color=BLUE, fontsize=14, fontweight='medium')
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 950)
    clean(ax, 'Months on production', 'Gas rate (10³ m³/month)')
    save('ml3_profiles_warmup', f, 'Synthetic decline curves: A hyperbolic (qi=300, Di=0.25/month, b=0.8); '
         'B = 3 x A; C plateau at 170 for 14 months then exponential decline 0.10/month.', kind='computed')


# ==================================================================== PART 1
@fig
def structureless():
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (500, 2))
    km = KMeans(4, n_init=10, random_state=0).fit(X)
    f, axs = half(2)
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, c=GREY, alpha=.75, edgecolors='white', linewidths=.3)
    titled(ax, 'One Gaussian cloud', '500 draws, no groups')
    ax2 = axs[1]
    shade_regions(ax2, km.predict, (-3.9, 3.9), (-3.9, 3.9), 4, amount=.84)
    points(ax2, X, km.labels_, s=12)
    centers(ax2, km.cluster_centers_, s=150)
    sil = silhouette_score(X, km.labels_)
    titled(ax2, 'K-means, K = 4', f'mean silhouette {sil:.2f}')
    for a in (ax, ax2):
        a.set_xlim(-3.9, 3.9)
        a.set_ylim(-3.9, 3.9)
        a.set_aspect('equal')
        clean(a, 'Feature 1', None, grid=False)
    ax.set_ylabel('Feature 2')
    ax.grid(alpha=1)
    save('ml3_structureless', f, 'Synthetic: 500 draws from one standard bivariate normal, seed 11; '
         'KMeans(K=4, n_init=10, random_state=0).', kind='computed')


@fig
def workflow():
    f, ax = canvas(11.8, 3.5)
    labels = ['Features', 'Scale and\ntransform', 'Method and\nsettings', 'Validate', 'Interpret', 'Act']
    subs = ['what should\ncount as similar', 'units, logs,\nweights', 'K-means, DBSCAN,\nhierarchy, GMM',
            'internal, stability,\nexternal', 'profiles in original\nunits, depth, time',
            'decision,\nmonitoring']
    cols = [GREY, GREY, BLUE, TEAL, ORANGE, ORANGE]
    w, h, gap, x0, y = 1.62, 1.55, .36, .22, 1.35
    bx = []
    for i, (lab, s, col) in enumerate(zip(labels, subs, cols)):
        fill = 'solid' if i == 5 else 'tint'
        b = box(ax, x0 + i * (w + gap), y, w, h, '', col, fill)
        tc = 'white' if fill == 'solid' else INK
        ax.text(b.c[0], y + h - .42, lab, ha='center', va='center', fontsize=14.5, fontweight='semibold',
                color=tc, linespacing=1.1)
        ax.text(b.c[0], y + .42, s, ha='center', va='center', fontsize=12, color='white' if fill == 'solid' else MUTED,
                linespacing=1.15)
        if i:
            arrow(ax, bx[-1].r, b.l, color=INK, shrink=2)
        bx.append(b)
    # feedback loop from validate / interpret back to features
    yb = .62
    ax.plot([bx[4].c[0], bx[4].c[0]], [y, yb], color=TEAL, lw=1.8)
    ax.plot([bx[3].c[0], bx[3].c[0]], [y, yb], color=TEAL, lw=1.8)
    ax.plot([bx[4].c[0], bx[0].c[0]], [yb, yb], color=TEAL, lw=1.8)
    arrow(ax, (bx[0].c[0], yb), (bx[0].c[0], y - .02), color=TEAL, lw=1.8)
    ax.text((bx[0].c[0] + bx[4].c[0]) / 2, yb - .22, 'revise features, scaling or method when the evidence is weak',
            ha='center', va='center', fontsize=13, color=TEAL)
    bracket(ax, bx[0].x, bx[1].x + w, y + h + .22, 'decide what similarity means', color=MUTED, below=False, fs=12.5)
    save('ml3_workflow', f, 'Schematic clustering workflow.', kind='schematic')


def scaling_data(seed=4):
    """GR and RHOB samples of three lithologies (subset of the S-1 parameters).
    Gamma ray cannot be negative, so GR draws below 5 API are redrawn."""
    rng = np.random.default_rng(seed)
    spec = [(0, 150), (1, 110), (3, 90)]           # shale, sand, limestone
    X, y = [], []
    for lith, n in spec:
        m, s = LITH_MEAN[lith, :2], LITH_SD[lith, :2] * np.array([1.6, 1.0])
        P = rng.normal(m, s, (n, 2))
        low = P[:, 0] < 5
        while low.any():
            P[low] = rng.normal(m, s, (int(low.sum()), 2))
            low = P[:, 0] < 5
        X.append(P)
        y += [lith] * n
    return np.vstack(X), np.array(y)


@fig
def scaling_effect():
    X, y = scaling_data()
    sc = StandardScaler().fit(X)
    raw = KMeans(3, n_init=10, random_state=0).fit(X)
    std = KMeans(3, n_init=10, random_state=0).fit(sc.transform(X))
    lab_raw = match_labels(raw.labels_, y)
    lab_std = match_labels(std.labels_, y)

    def remap(fitted, labels_before, labels_after):
        m = {a: b for a, b in zip(labels_before, labels_after)}
        return lambda G: np.vectorize(m.get)(fitted(G))
    xl, yl = (0, 158), (2.05, 2.78)
    f, axs = sub(3, height=4.3)
    for ax, lab, title in [(axs[0], y, 'Reference lithology'), (axs[1], lab_raw, 'K-means on raw units'),
                           (axs[2], lab_std, 'K-means on standardized units')]:
        if ax is axs[1]:
            shade_regions(ax, remap(raw.predict, raw.labels_, lab_raw), xl, yl, 4, amount=.86)
        if ax is axs[2]:
            shade_regions(ax, remap(lambda G: std.predict(sc.transform(G)), std.labels_, lab_std), xl, yl, 4,
                          amount=.86)
        points(ax, X, lab, s=16)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        titled(ax, title, 'boundaries follow GR only' if ax is axs[1] else None)
        clean(ax, 'GR (API)', 'RHOB (g/cm³)' if ax is axs[0] else None, grid=ax is axs[0])
    for k, name, dy in [(0, 'shale', .115), (1, 'sand', -.105), (3, 'limestone', .095)]:
        c = X[y == k].mean(0)
        axs[0].text(c[0], c[1] + dy, name, color=shade(SERIES[k], .25),
                    fontsize=13.5, ha='center', va='center', fontweight='semibold')
    save('ml3_scaling_effect', f, 'Synthetic GR and RHOB samples of three lithologies, seed 4; KMeans(K=3, n_init=10, '
         'random_state=0) on raw and on standardized features; cluster colours matched to the reference.',
         kind='computed')


@fig
def feature_weights():
    rng = np.random.default_rng(8)
    cen = np.array([[-1.25, -1.0], [-1.25, 1.0], [1.25, -1.0], [1.25, 1.0]])
    X = np.vstack([rng.normal(c, .42, (70, 2)) for c in cen])
    X = StandardScaler().fit_transform(X)
    f, axs = half(2)
    for ax, w in zip(axs, [0.5, 2.0]):
        W = np.array([1.0, w])
        km = KMeans(2, n_init=10, random_state=0).fit(X * W)
        # colour convention: cluster 0 is the left (w < 1) or lower (w > 1) group
        axis = 0 if w < 1 else 1
        flip = X[km.labels_ == 0, axis].mean() > X[km.labels_ == 1, axis].mean()
        lab = 1 - km.labels_ if flip else km.labels_
        shade_regions(ax, lambda G: (1 - km.predict(G * W)) if flip else km.predict(G * W),
                      (-2.4, 2.4), (-2.4, 2.4), 2, amount=.86, res=900)
        points(ax, X, lab, s=13)
        ax.set_aspect('equal')
        panel(ax, f'Weight on feature 2 = {w:g}')
        clean(ax, 'Feature 1', None, grid=False)
    axs[0].set_ylabel('Feature 2')
    save('ml3_feature_weights', f, 'Synthetic four-group data, seed 8, standardized; KMeans(K=2) after multiplying '
         'feature 2 by the stated weight; plotted in unweighted coordinates.', kind='computed')


# ==================================================================== PART 2
@fig
def kmeans_steps():
    """One Lloyd iteration from three fixed starting centres (module variant of the
    shared kmeans_steps figure: same data and centres)."""
    X, _ = make_blobs(n_samples=150, centers=[[-2, 0], [1, 2], [2, -1]], cluster_std=.55, random_state=42)
    init = np.array([[-2.5, 1.8], [.2, .5], [2.8, 1.6]])
    lab0 = ((X[:, None] - init[None]) ** 2).sum(2).argmin(1)
    means = np.array([X[lab0 == k].mean(0) for k in range(3)])
    lab1 = ((X[:, None] - means[None]) ** 2).sum(2).argmin(1)
    f, axs = sub(3, height=4.3)
    steps = [(lab0, init, 'Assign to initial centres'), (lab0, means, 'Move centres to the means'),
             (lab1, means, 'Reassign to nearest centre')]
    for i, (ax, (lab, C, title)) in enumerate(zip(axs, steps)):
        points(ax, X, lab, s=22)
        if i == 1:                              # where each centre came from
            ax.scatter(init[:, 0], init[:, 1], marker='X', s=150, facecolors='white',
                       edgecolors=[SERIES[k] for k in range(3)], linewidths=1.6, zorder=5)
            for a, b in zip(init, means):
                ax.annotate('', b, a, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.5,
                                                      shrinkA=8, shrinkB=9), zorder=6)
        centers(ax, C, s=160)
        panel(ax, title)
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-2.5, 3.9)
        clean(ax, 'Feature 1', 'Feature 2' if i == 0 else None)
    save('ml3_kmeans_steps', f, 'Synthetic three blobs (make_blobs, 150 points, random_state=42); one Lloyd '
         'iteration from fixed starting centres (-2.5, 1.8), (0.2, 0.5), (2.8, 1.6). Hollow crosses in the '
         'middle panel mark the starting centres.', kind='computed')


@fig
def voronoi():
    X, _ = make_blobs(320, centers=5, cluster_std=.95, random_state=6)
    X = StandardScaler().fit_transform(X)
    km = KMeans(5, n_init=10, random_state=0).fit(X)
    C = km.cluster_centers_
    f, axs = half()
    ax = axs[0]
    xl, yl = (-2.6, 2.6), (-2.4, 2.4)
    shade_regions(ax, km.predict, xl, yl, 5, amount=.84, res=700)
    points(ax, X, km.labels_, s=16)
    centers(ax, C)
    # the closest pair of centroids always shares a cell boundary at its midpoint
    D = ((C[:, None] - C[None]) ** 2).sum(2) + np.eye(5) * 1e9
    i, j = np.unravel_index(np.argmin(D), D.shape)
    ax.plot(*zip(C[i], C[j]), color=INK, lw=1.6, ls=(0, (4, 3)), zorder=5)
    m = (C[i] + C[j]) / 2
    u = (C[j] - C[i]) / np.linalg.norm(C[j] - C[i])
    v = np.array([-u[1], u[0]])
    sq = .12
    ax.plot([m[0] + sq * u[0], m[0] + sq * u[0] + sq * v[0], m[0] + sq * v[0]],
            [m[1] + sq * u[1], m[1] + sq * u[1] + sq * v[1], m[1] + sq * v[1]], color=INK, lw=1.3, zorder=5)
    ax.annotate('boundary = perpendicular\nbisector of two centroids', m, xytext=(.03, .97),
                textcoords='axes fraction', ha='left', va='top', fontsize=12.5, color=INK,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkB=4, connectionstyle='arc3,rad=-.2'),
                bbox=dict(boxstyle='round,pad=.3', fc='white', ec='none', alpha=.92), zorder=7)
    ax.set_aspect('equal')
    clean(ax, 'Feature 1 (standardized)', 'Feature 2 (standardized)', grid=False)
    save('ml3_voronoi', f, 'Synthetic blobs (make_blobs, random_state=6), standardized; KMeans(K=5). '
         'Shading = nearest-centroid (Voronoi) cells.', kind='computed')


def kpp_blobs():
    X, y = make_blobs(360, centers=[[-4, 2.5], [-3.2, -2.5], [2.5, 3.0], [3.8, -2.2]], cluster_std=.85,
                      random_state=2)
    return X, y


def kpp_seeds(X, rng, K=4):
    """Textbook k-means++: first centre uniform, then sample proportional to D^2."""
    seeds, first_D2 = [int(rng.integers(len(X)))], None
    for k in range(K - 1):
        D2 = ((X[:, None] - X[seeds][None]) ** 2).sum(2).min(1)
        if k == 0:
            first_D2 = D2.copy()
        seeds.append(int(rng.choice(len(X), p=D2 / D2.sum())))
    return seeds, first_D2


@fig
def kmeanspp():
    X, _ = kpp_blobs()
    seeds, D2 = kpp_seeds(X, np.random.default_rng(0))
    rnd = np.random.default_rng(0).choice(len(X), 4, replace=False)
    f, axs = sub(3, height=3.9)
    lim = dict(xlim=(-7.2, 6.8), ylim=(-5.4, 5.6))

    def star(ax, idx, col=ORANGE):
        ax.scatter(X[idx, 0], X[idx, 1], marker='*', s=520, c=col, edgecolors=INK, linewidths=1.0, zorder=6)
        for n, i in enumerate(np.atleast_1d(idx)):
            ax.text(X[i, 0] + .5, X[i, 1] + .45, str(n + 1), fontsize=15, fontweight='bold', color=INK, zorder=7,
                    path_effects=HALO)
    ax = axs[0]
    order = np.argsort(D2)
    ax.scatter(X[order, 0], X[order, 1], s=15, c=D2[order], cmap=CMAP_SEQ, vmin=-D2.max() * .15,
               vmax=D2.max(), zorder=2)
    star(ax, seeds[:2])
    ax.text(.5, -.03, 'darker = farther from centre 1\n= more likely to be centre 2', transform=ax.transAxes,
            fontsize=12, color=INK, ha='center', va='top', linespacing=1.2)
    panel(ax, r'After centre 1: weights $\propto$ D²')
    ax = axs[1]
    ax.scatter(X[:, 0], X[:, 1], s=13, c='#AFBBC7', zorder=2)
    star(ax, seeds)
    panel(ax, 'k-means++ seeds, K = 4')
    ax = axs[2]
    ax.scatter(X[:, 0], X[:, 1], s=13, c='#AFBBC7', zorder=2)
    star(ax, rnd, col=GOLD)
    panel(ax, 'Uniform random seeds')
    for ax in axs:
        ax.set(**lim)
        ax.set_aspect('equal')
        bare(ax)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='none', ec=RULE, lw=1.2))
    # share of draws that put one seed in every blob (quoted on the slide)
    _, y = kpp_blobs()
    rng = np.random.default_rng(0)
    p_text = np.mean([len(set(y[kpp_seeds(X, rng)[0]])) == 4 for _ in range(2000)])
    p_skl = np.mean([len(set(y[kmeans_plusplus(X, 4, random_state=i)[1]])) == 4 for i in range(2000)])
    sizes = np.bincount(y)
    p_unif = np.prod(sizes) * 24 / np.prod(len(X) - np.arange(4))      # four distinct points, exact
    assert (round(100 * p_text), round(100 * p_skl), round(1000 * p_unif)) == (66, 95, 95)
    save('ml3_kmeanspp', f, 'Synthetic four-blob data (random_state=2). One textbook k-means++ draw (D² sampling, '
         f'numpy seed 0) and one uniform draw (seed 0). One seed in every blob: {p_text:.0%} of 2,000 textbook '
         f'k-means++ draws (numpy seed 0), {p_skl:.0%} of 2,000 scikit-learn kmeans_plusplus draws '
         f'(random_state 0-1999); uniform sampling of four distinct points {p_unif:.1%} (exact).', kind='computed')


def restart_data():
    grid = [[x, yv] for x in (0, 3.2, 6.4) for yv in (0, 3.2, 6.4)]
    X, y = make_blobs(55 * len(grid), centers=grid, cluster_std=.55, random_state=7)
    return X, y


def lloyd(X, C0, iters=25):
    C = C0.astype(float).copy()
    hist = []
    lab = None
    for _ in range(iters):
        D = ((X[:, None] - C[None]) ** 2).sum(2)
        lab = D.argmin(1)
        hist.append(D[np.arange(len(X)), lab].sum())
        for k in range(len(C)):
            if np.any(lab == k):
                C[k] = X[lab == k].mean(0)
    return C, lab, np.array(hist)


def badge_spot(candidates, X, C, lim, r=.5):
    """Candidate position for a letter badge of radius r (data units) that crosses no
    nearest-centroid boundary; among those, the one farthest from points and centroids."""
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rim = np.c_[np.cos(t), np.sin(t)] * r
    best, score = None, -np.inf
    for p in candidates:
        if not (lim[0][0] + r < p[0] < lim[0][1] - r and lim[1][0] + r < p[1] < lim[1][1] - r):
            continue
        S = np.vstack([p, p + rim])
        if len(set(((S[:, None] - C[None]) ** 2).sum(2).argmin(1))) > 1:
            continue
        clear = min(np.linalg.norm(X - p, axis=1).min(), np.linalg.norm(C - p, axis=1).min())
        if clear > score:
            best, score = p, clear
    return best


RESTART_K = 9
GRID9 = [BLUE, ORANGE, TEAL, PURPLE, RED, GOLD, TEAL, BLUE, ORANGE]   # no equal neighbours on the 3x3 grid
GRID9_M = ['o', '^', 's', 'D', 'v', 'P', 's', 'o', '^']


def restart_runs(n_runs=20, K=RESTART_K):
    X, _ = restart_data()
    runs = []
    for seed in range(n_runs):
        idx = np.random.default_rng(100 + seed).choice(len(X), K, replace=False)
        runs.append(lloyd(X, X[idx]))
    return X, runs


@fig
def restarts():
    X, runs = restart_runs()
    _, yref = restart_data()
    K = RESTART_K
    finals = np.array([r[2][-1] for r in runs])
    best = finals.min()
    good = np.isclose(finals, best, rtol=1e-6)
    f, axs = plt.subplots(1, 3, figsize=(11.8, 4.3), layout='constrained',
                          gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axs[0]
    for (C, lab, hist), g in sorted(zip(runs, good), key=lambda t: t[1]):
        ax.plot(np.arange(1, len(hist) + 1), hist, color=BLUE if g else ORANGE, lw=1.7, alpha=.85 if g else .6)
    clean(ax, 'Iteration', 'Inertia (within-cluster SS)')
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 1500)
    ax.text(.97, .95, f'{good.sum()} of {len(runs)} starts reach\nthe lowest inertia, {best:.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=12.5, color=BLUE)
    ax.text(.97, .72, f'{(~good).sum()} stop in worse\nlocal minima', transform=ax.transAxes, ha='right',
            va='top', fontsize=12.5, color=ORANGE)
    panel(ax, f'20 random starts, K = {K}')
    i_best = int(np.argmin(finals))
    i_bad = int(np.argmax(finals))
    grid = np.array([[x, yv] for x in (0, 3.2, 6.4) for yv in (0, 3.2, 6.4)])
    for ax, i, title, col in [(axs[1], i_best, f'Lowest inertia: {finals[i_best]:.0f}', BLUE),
                              (axs[2], i_bad, f'Worst local minimum: {finals[i_bad]:.0f}', ORANGE)]:
        C, lab, _ = runs[i]
        xl, yl = (-1.9, 8.3), (-2.0, 8.4)
        xx, yy = np.meshgrid(np.linspace(*xl, 500), np.linspace(*yl, 500))
        zz = ((np.c_[xx.ravel(), yy.ravel()][:, None] - C[None]) ** 2).sum(2).argmin(1).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=np.arange(K) + .5, colors=['#8FA1B3'], linewidths=1.0, zorder=1)
        ax.scatter(X[:, 0], X[:, 1], s=10, c='#A9B6C3', zorder=2)
        ax.scatter(C[:, 0], C[:, 1], marker='X', s=150, c=col, edgecolors=INK, linewidths=1.1, zorder=6)
        ax.set_title(title, fontsize=14, loc='left', color=col)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect('equal')
        clean(ax, 'Feature 1', 'Feature 2' if ax is axs[1] else None, grid=False)
        if i == i_bad:
            dist = np.linalg.norm(grid[:, None] - C[None], axis=2)
            shared = [b for b in range(K) if (dist[b] < 1.3).sum() >= 2]
            span = [k for k in range(K) if np.sort(np.bincount(yref[lab == k], minlength=K))[-2]
                    > .3 * (lab == k).sum()]
            badge = dict(fontsize=13, fontweight='bold', color=INK, ha='center', va='center', zorder=9,
                         bbox=dict(boxstyle='circle,pad=.25', fc='white', ec=INK, lw=1.2))
            s = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            if span:
                k = span[0]
                members = np.bincount(yref[lab == k], minlength=K)
                two = grid[np.argsort(members)[-2:]]
                c = two.mean(0)
                d = np.linalg.norm(two[0] - two[1])
                ang = np.arctan2(*(two[1] - two[0])[::-1])
                ax.add_patch(Ellipse(c, d + 2.6, 2.7, angle=np.degrees(ang), fc='none', ec=INK, lw=1.3,
                                     ls=(0, (3, 2)), zorder=5))
                rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                ring = c + (np.c_[((d + 2.6) / 2 + .6) * np.cos(s), (1.35 + .6) * np.sin(s)] @ rot.T)
                ax.text(*badge_spot(ring, X, C, (xl, yl)), 'A', **badge)
            if shared:
                b = grid[shared[0]]
                ax.add_patch(Circle(b, 1.45, fc='none', ec=INK, lw=1.3, ls=(0, (3, 2)), zorder=5))
                ring = b + 2.05 * np.c_[np.cos(s), np.sin(s)]
                ax.text(*badge_spot(ring, X, C, (xl, yl)), 'B', **badge)
    save('ml3_restarts', f, 'Synthetic 3 x 3 blob grid (make_blobs random_state=7, 495 points). Lloyd iterations '
         'from 20 uniform random starts (numpy seeds 100-119), K=9, 25 iterations each.', kind='computed')


@fig
def silhouette_geometry():
    rng = np.random.default_rng(21)
    A = rng.normal([0, .2], .55, (9, 2))
    B = rng.normal([3.3, -.5], .55, (9, 2))
    Cc = rng.normal([5.6, 2.3], .55, (9, 2))
    X = np.vstack([A, B, Cc])
    lab = np.repeat([0, 1, 2], 9)
    i = int(np.argmax(A[:, 0] - .3 * A[:, 1]))          # point of A facing B
    xi = A[i]
    a = np.linalg.norm(A - xi, axis=1).sum() / (len(A) - 1)
    bB = np.linalg.norm(B - xi, axis=1).mean()
    bC = np.linalg.norm(Cc - xi, axis=1).mean()
    b = min(bB, bC)
    s = (b - a) / max(a, b)
    assert np.isclose(s, silhouette_samples(X, lab)[i]) and bB < bC
    f, axs = half()
    ax = axs[0]
    for P in A:
        ax.plot([xi[0], P[0]], [xi[1], P[1]], color=BLUE, lw=1.2, alpha=.75, zorder=1)
    for P in B:
        ax.plot([xi[0], P[0]], [xi[1], P[1]], color=ORANGE, lw=1.2, alpha=.65, zorder=1)
    points(ax, X, lab, s=48)
    ax.scatter(*xi, s=300, facecolors='white', edgecolors=INK, linewidths=2.2, zorder=6)
    ax.scatter(*xi, s=48, c=BLUE, zorder=7)
    ax.text(xi[0] - .28, xi[1] + .38, 'i', fontsize=17, fontweight='bold', color=INK, ha='center', zorder=8)
    for P, name, dy in [(A, 'own cluster', -1.15), (B, 'nearest other cluster', -1.15), (Cc, 'third cluster', 1.05)]:
        c = P.mean(0)
        ax.text(c[0], c[1] + dy, name, ha='center', fontsize=13, color=MUTED)
    ax.text(.02, .98, f'a = {a:.2f}   mean distance to own cluster\nb = {b:.2f}   mean distance to nearest other\n'
            f's = (b − a) / max(a, b) = {s:.2f}', transform=ax.transAxes, va='top', ha='left', fontsize=13,
            color=INK, linespacing=1.5, bbox=dict(boxstyle='round,pad=.4', fc=PANEL, ec='none'))
    ax.set_xlim(-1.7, 7.0)
    ax.set_ylim(-2.05, 4.55)
    ax.set_aspect('equal')
    bare(ax)
    save('ml3_silhouette_geometry', f, 'Synthetic three-cluster example (seed 21); a, b and s computed with numpy and '
         'checked against sklearn silhouette_samples.', kind='computed')


def knife(ax, X, lab, k):
    s = silhouette_samples(X, lab)
    y0 = 0
    for c in range(k):
        v = np.sort(s[lab == c])
        ys = np.arange(y0, y0 + len(v))
        ax.fill_betweenx(ys, 0, v, color=SERIES[c], alpha=.85, lw=0, zorder=2)
        ax.text(-.05, y0 + len(v) / 2, str(c + 1), ha='right', va='center', fontsize=13, color=SERIES[c],
                fontweight='semibold')
        y0 += len(v) + 12
    m = s.mean()
    ax.plot([m, m], [-8, y0 + 2], color=INK, ls='--', lw=1.4, zorder=3)
    ax.text(m, y0 + 14, f'mean {m:.2f}', fontsize=12.5, color=INK, va='bottom', ha='center')
    ax.axvline(0, color=MUTED, lw=.9)
    ax.set_xlim(-.25, 1)
    ax.set_ylim(-8, y0 + 55)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Silhouette value')
    ax.grid(axis='x', alpha=1)
    ax.set_axisbelow(True)
    return s


@fig
def silhouette_knife():
    X, y = make_blobs(450, centers=[[-3, 0], [2.5, 2.5], [2.8, -2.6]], cluster_std=[1.0, .7, 1.1],
                      random_state=5)
    f, axs = plt.subplots(1, 4, figsize=(11.8, 4.3), layout='constrained',
                          gridspec_kw=dict(width_ratios=[1, .8, 1, .8]))
    for (sax, kax), K in [((axs[0], axs[1]), 3), ((axs[2], axs[3]), 5)]:
        km = KMeans(K, n_init=10, random_state=0).fit(X)
        lab = match_labels(km.labels_, y)
        points(sax, X, lab, s=11)
        sax.set_xlim(-6.5, 6)
        sax.set_ylim(-6, 5.5)
        sax.set_aspect('equal', adjustable='datalim')
        clean(sax, 'Feature 1', 'Feature 2' if K == 3 else None, grid=False)
        panel(sax, f'K-means, K = {K}')
        knife(kax, X, lab, K)
        panel(kax, 'Silhouette plot')
    save('ml3_silhouette_knife', f, 'Synthetic three-blob data (random_state=5); KMeans(n_init=10, random_state=0) '
         'with K=3 and K=5; per-sample values from sklearn silhouette_samples.', kind='computed')


def failure_sets():
    X, y = make_blobs(600, random_state=170)
    Xa = X @ np.array([[.6, -.6], [-.4, .8]])
    Xv, yv = make_blobs(600, cluster_std=[1.0, 2.5, .5], random_state=170)
    rng = np.random.default_rng(3)
    big = rng.normal([0, 0], [2.2, 2.2], (520, 2))
    mid = rng.normal([6.3, 0], .7, (60, 2))
    small = rng.normal([3.1, 3.4], .35, (18, 2))
    Xs = np.vstack([big, mid, small])
    ys = np.repeat([0, 1, 2], [520, 60, 18])
    return [(Xa, y, 'Anisotropic groups'), (Xv, yv, 'Unequal spread'), (Xs, ys, 'Unequal sizes')]


@fig
def kmeans_failures():
    r90 = np.sqrt(-2 * np.log(.1))              # Mahalanobis radius of a 90% ellipse in 2-D
    f, axs = sub(3, height=4.3)
    for ax, (X, y, title) in zip(axs, failure_sets()):
        km = KMeans(3, n_init=10, random_state=0).fit(X)
        lab = match_labels(km.labels_, y)
        points(ax, X, lab, s=11, alpha=.85)
        for k in range(3):
            P = X[y == k]
            vals, vecs = np.linalg.eigh(np.cov(P.T))
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax.add_patch(Ellipse(P.mean(0), 2 * r90 * np.sqrt(vals[1]), 2 * r90 * np.sqrt(vals[0]), angle=ang,
                                 fc='none', ec=INK, lw=1.3, ls=(0, (4, 3)), zorder=5))
        Cm = np.array([X[lab == k].mean(0) for k in range(3)])
        centers(ax, Cm, s=150)
        share = np.mean(lab == y)
        ax.set_aspect('equal', adjustable='datalim')
        titled(ax, title, f'{share:.0%} in their generating group')
        clean(ax, 'Feature 1', 'Feature 2' if ax is axs[0] else None, grid=False)
    save('ml3_kmeans_failures', f, 'Synthetic data after the scikit-learn k-means assumptions example '
         '(make_blobs random_state=170; unequal sizes numpy seed 3). KMeans(K=3, n_init=10). Dashed ellipses: '
         'generating groups, 90% coverage. Share computed after best label matching.', kind='computed')


# ------------------------------------------------------ Lab 1 debrief (course data)
W5_FEATURES = ["Pressure", "AI", "VpVs", "SW", "phid", "vcl"]


def w5_kmeans():
    d = pd.read_csv(DATA / '01_kmeans' / 'w5.csv')
    Z = StandardScaler().fit_transform(d[W5_FEATURES])
    sil = {}
    for K in range(2, 11):
        sil[K] = silhouette_score(Z, KMeans(K, random_state=42, n_init=10).fit_predict(Z))
    lab = KMeans(3, random_state=42, n_init=10).fit_predict(Z)
    # deterministic naming: 0 = lowest mean pressure (shallow), 2 = lowest mean SW, 1 = the rest
    g = d.groupby(lab)[['Pressure', 'SW']].mean()
    shallow = int(g.Pressure.idxmin())
    lowsw = int(g.drop(index=shallow).SW.idxmin())
    rest = ({0, 1, 2} - {shallow, lowsw}).pop()
    lab = np.select([lab == shallow, lab == rest, lab == lowsw], [0, 1, 2])
    return d, sil, lab


@fig
def w5_debrief():
    d, sil, lab = w5_kmeans()
    f, axs = plt.subplots(1, 3, figsize=(11.8, 4.3), layout='constrained',
                          gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axs[0]
    ks = list(sil)
    ax.plot(ks, [sil[k] for k in ks], '-o', color=TEAL, lw=2.4, ms=7)
    kb = max(range(2, 7), key=lambda k: sil[k])
    ax.scatter([kb], [sil[kb]], s=160, facecolors='none', edgecolors=ORANGE, linewidths=2.2, zorder=5)
    ax.annotate(f'highest in 2–6: K = {kb}, {sil[kb]:.2f}', (kb, sil[kb]), xytext=(4.6, .392), fontsize=12.5,
                color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1, shrinkB=7))
    ax.set_ylim(.25, .45)
    ax.set_xticks(ks)
    clean(ax, 'Number of clusters K', 'Mean silhouette')
    panel(ax, 'Silhouette, six standardized logs')
    names = ['C1  shallow interval', 'C2  deeper, SW ≈ 1', 'C3  low SW']
    for ax, col, xl, unit in [(axs[1], 'Pressure', 'Pressure (as supplied)', None),
                              (axs[2], 'SW', 'SW (v/v)', None)]:
        for k in range(3):
            m = lab == k
            ax.scatter(d[col][m], d.DEPTH[m], s=2.5, c=SERIES[k], alpha=.8, lw=0, rasterized=True, zorder=3)
        ax.set_ylim(d.DEPTH.max() + 20, d.DEPTH.min() - 20)
        clean(ax, xl, 'Depth (m)' if ax is axs[1] else None)
        panel(ax, f'Clusters on the {col} log')
    axs[2].set_yticklabels([])
    handles = [Line2D([], [], marker='s', ls='', color=SERIES[k], ms=9, label=names[k]) for k in range(3)]
    axs[1].legend(handles=handles, loc='upper right', fontsize=12, handletextpad=.3, borderaxespad=.2,
                  frameon=True, facecolor='white', edgecolor='none', framealpha=.95)
    save('ml3_w5_debrief', f, 'Computed from w5.csv (course data, 7,071 samples): KMeans(n_init=10, random_state=42) on '
         'six standardized features as in the Lab 1 notebook; K=3 clusters plotted against depth.', kind='data')


# ==================================================================== PART 3
CORE_M, BORDER_M = 'o', '^'


def dbscan_types(X, eps, ms):
    m = DBSCAN(eps=eps, min_samples=ms).fit(X)
    core = np.zeros(len(X), bool)
    core[m.core_sample_indices_] = True
    border = (m.labels_ >= 0) & ~core
    noise = m.labels_ < 0
    return m, core, border, noise


@fig
def dbscan_concept():
    P = np.array([[0.0, 0.0], [.45, .30], [-.40, .35], [.25, -.45], [-.35, -.40], [.80, -.15], [-.85, -.05],
                  [.05, .75], [1.25, .35], [1.70, .20], [-1.3, .55], [.45, -1.05], [2.95, -.9], [-2.05, -1.35]])
    eps, ms = 1.0, 4
    m, core, border, noise = dbscan_types(P, eps, ms)
    ic, ib, inn = 0, 9, 12                      # highlighted core, border and noise points
    assert core[ic] and border[ib] and noise[inn]
    f, axs = half()
    ax = axs[0]
    for i, col, ls, fc in [(ic, BLUE, '-', tint(BLUE, .88)), (ib, ORANGE, (0, (4, 3)), 'none'),
                           (inn, GREY, (0, (2, 2.5)), 'none')]:
        ax.add_patch(Circle(P[i], eps, fc=fc, ec=col, lw=1.8, ls=ls, zorder=1))
    u = np.array([np.cos(np.radians(-96)), np.sin(np.radians(-96))])   # widest gap between neighbours
    ax.plot(*zip(P[ic], P[ic] + eps * u), color=BLUE, lw=1.4)
    ax.text(*(P[ic] + .6 * eps * u + .13 * np.array([-u[1], u[0]])), 'ε', fontsize=17, color=BLUE,
            ha='center', va='center', fontstyle='italic')
    ax.scatter(*P[core].T, s=95, c=BLUE, marker=CORE_M, edgecolors='white', linewidths=.8, zorder=4)
    ax.scatter(*P[border].T, s=110, c=ORANGE, marker=BORDER_M, edgecolors='white', linewidths=.8, zorder=4)
    ax.scatter(*P[noise].T, s=110, c=GREY, marker='x', linewidths=2.2, zorder=4)
    n_c = int((np.linalg.norm(P - P[ic], axis=1) <= eps).sum())
    n_b = int((np.linalg.norm(P - P[ib], axis=1) <= eps).sum())
    n_n = int((np.linalg.norm(P - P[inn], axis=1) <= eps).sum())
    ax.text(-.55, 1.32, f'core: {n_c} points within ε', fontsize=13, color=BLUE, ha='center', fontweight='medium')
    ax.text(2.45, 1.55, f'border: {n_b} within ε,\nbut within ε of a core point', fontsize=13, color=ORANGE,
            ha='center', fontweight='medium', linespacing=1.2)
    ax.text(2.95, -2.3, f'noise: {n_n} within ε,\nno core point within ε', fontsize=13, color=GREY, ha='center',
            va='top', fontweight='medium', linespacing=1.2)
    ax.text(-2.55, -2.3, f'min_samples = {ms}\n(the point itself counts)', fontsize=12.5, color=INK, ha='left',
            va='top', linespacing=1.2)
    ax.set_xlim(-2.6, 4.1)
    ax.set_ylim(-3.1, 2.05)
    ax.set_aspect('equal')
    bare(ax)
    save('ml3_dbscan_concept', f, f'Hand-placed points; types computed with DBSCAN(eps={eps}, min_samples={ms}).',
         kind='computed')


CHAIN_EPS, CHAIN_MS = .2, 8


@fig
def dbscan_chain():
    """Two moons plus three outliers: point types, the two clusters, and a zoom on a
    chain of core points whose eps-neighbourhoods overlap."""
    Xm, _ = make_moons(n_samples=230, noise=.065, random_state=42)
    X = np.r_[Xm, [[-1.4, -.6], [2.2, 1.2], [.6, 1.5]]]
    eps = CHAIN_EPS
    m, core, border, noise = dbscan_types(X, eps, CHAIN_MS)
    lab = m.labels_
    assert len(set(lab) - {-1}) == 2 and noise.sum() == 3
    up = int(np.argmax([X[lab == k, 1].mean() for k in range(2)]))
    Q = np.where(core & (lab == up))[0]
    ang = np.arctan2(X[Q, 1], X[Q, 0])
    cur = Q[np.argmin(np.abs(ang - np.radians(140)))]
    chain = [cur]
    for _ in range(5):                         # next core point: farthest one along the moon within 0.95 eps
        d = np.linalg.norm(X[Q] - X[cur], axis=1)
        ok = (d < .95 * eps) & (ang < np.arctan2(X[cur, 1], X[cur, 0]))
        cur = Q[ok][np.argmax(d[ok])]
        chain.append(cur)
    Cc = X[chain]
    zx = (Cc[:, 0].min() - eps - .12, Cc[:, 0].max() + eps + .12)
    zy = (Cc[:, 1].min() - eps - .12, Cc[:, 1].max() + eps + .12)
    f, axs = plt.subplots(1, 2, figsize=(11.8, 4.3), layout='constrained', gridspec_kw=dict(width_ratios=[1.55, 1]))
    for ax in axs:
        ax.scatter(*X[core].T, s=40, c=BLUE, marker=CORE_M, edgecolors='white', linewidths=.6, zorder=4,
                   label='core')
        ax.scatter(*X[border].T, s=55, c=ORANGE, marker=BORDER_M, edgecolors='white', linewidths=.6, zorder=4,
                   label='border')
        ax.scatter(*X[noise].T, s=60, c=GREY, marker='x', linewidths=2, zorder=4, label='noise')
        ax.set_aspect('equal')
    ax = axs[0]
    ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fc='none', ec=INK, lw=1.2,
                           ls=(0, (3, 2)), zorder=5))
    for text, xy in [('cluster 1', (.62, 1.22)), ('cluster 2', (1.75, -.78))]:
        ax.text(*xy, text, fontsize=13, color=INK, ha='center', va='center', fontweight='semibold')
    ax.set_xlim(-1.6, 2.4)
    ax.set_ylim(-.9, 1.65)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), ncol=3, fontsize=12.5, handletextpad=.2,
              columnspacing=1.0, borderaxespad=.2)
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    titled(ax, f'DBSCAN(ε = {eps:g}, min_samples = {CHAIN_MS})', f'2 clusters, {int(noise.sum())} noise points')
    ax = axs[1]
    for p in Cc:
        ax.add_patch(Circle(p, eps, fc=tint(BLUE, .9), ec=BLUE, lw=1.3, alpha=.9, zorder=1))
    ax.scatter(*Cc.T, s=190, facecolors='none', edgecolors=INK, linewidths=1.6, zorder=5)
    ax.set_xlim(*zx)
    ax.set_ylim(*zy)
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    titled(ax, 'Zoom: a chain of ε-neighbourhoods', 'each ringed core point lies within ε of the next')
    save('ml3_dbscan_chain', f, f'Synthetic two moons (make_moons, 230 points, noise 0.065, random_state=42) plus '
         f'three hand-placed outliers; DBSCAN(eps={eps}, min_samples={CHAIN_MS}). Discs: eps-neighbourhoods of '
         'a chain of core points.', kind='computed')


def moons_noise():
    X, _ = make_moons(320, noise=.07, random_state=3)
    rng = np.random.default_rng(3)
    N = rng.uniform([-1.3, -.8], [2.3, 1.3], (28, 2))
    return np.vstack([X, N])


KD_MS = 5
KD_EPS = [(.06, 'too small', ORANGE), (.14, 'near the knee', TEAL), (.40, 'too large', PURPLE)]


@fig
def kdistance():
    X = moons_noise()
    dist, _ = NearestNeighbors(n_neighbors=KD_MS).fit(X).kneighbors(X)
    kd = np.sort(dist[:, -1])
    f, axs = half()
    ax = axs[0]
    ax.plot(np.arange(len(kd)), kd, color=BLUE, lw=2.6)
    for eps, name, col in KD_EPS:
        frac = np.mean(kd <= eps)
        ax.axhline(eps, color=col, lw=1.5, ls=(0, (5, 3)))
        below = eps < .1
        ax.text(70 if below else 8, eps - .012 if below else eps + .012,
                f'ε = {eps:g}  {name}: {frac:.0%} of points are core', color=col, fontsize=12.5,
                va='top' if below else 'bottom')
    clean(ax, 'Points sorted by k-distance', f'Distance to {KD_MS}th neighbour (self included)')
    ax.set_xlim(0, len(kd))
    ax.set_ylim(0, .62)
    save('ml3_kdistance', f, f'Synthetic moons (random_state=3) plus 28 uniform noise points; NearestNeighbors('
         f'n_neighbors={KD_MS}) on the training points, so each point counts itself, as DBSCAN min_samples does.',
         kind='computed')


@fig
def eps_panels():
    X = moons_noise()
    f, axs = sub(3, height=4.1)
    for ax, (eps, name, col) in zip(axs, KD_EPS):
        m = DBSCAN(eps=eps, min_samples=KD_MS).fit(X)
        lab = m.labels_
        k = len(set(lab)) - (1 if -1 in lab else 0)
        points(ax, X, lab, s=13)
        ax.set_title(f'ε = {eps:g}: {k} cluster{"s" if k != 1 else ""}, {np.mean(lab < 0):.0%} noise', fontsize=14,
                     loc='left', color=col)
        ax.set_xlim(-1.4, 2.4)
        ax.set_ylim(-.9, 1.4)
        ax.set_aspect('equal')
        clean(ax, 'Feature 1', 'Feature 2' if ax is axs[0] else None, grid=False)
    save('ml3_eps_panels', f, f'Same synthetic data as the k-distance plot; DBSCAN(min_samples={KD_MS}) at three eps '
         'values. Grey crosses are noise.', kind='computed')


def density_data():
    rng = np.random.default_rng(14)
    dense = rng.normal([0, 0], .22, (220, 2))
    medium = rng.normal([2.1, .3], .45, (160, 2))
    sparse = rng.normal([.9, 3.6], 1.0, (110, 2))
    bg = rng.uniform([-2.5, -2.0], [5.0, 6.8], (30, 2))
    X = np.vstack([dense, medium, sparse, bg])
    y = np.repeat([0, 1, 2, -1], [220, 160, 110, 30])
    return X, y


@fig
def hdbscan_vs_dbscan():
    X, y = density_data()
    models = [(DBSCAN(eps=.25, min_samples=10), 'DBSCAN, ε = 0.25'),
              (DBSCAN(eps=.6, min_samples=10), 'DBSCAN, ε = 0.6'),
              (HDBSCAN(min_cluster_size=20), 'HDBSCAN, min_cluster_size = 20')]
    f, axs = sub(3, height=4.3)
    for ax, (m, title) in zip(axs, models):
        lab = m.fit_predict(X)
        lab = match_labels(lab, y)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        points(ax, X, lab, s=12)
        titled(ax, title, f'{k} clusters, {np.mean(lab < 0):.0%} noise')
        ax.set_xlim(-2.6, 5.1)
        ax.set_ylim(-2.1, 6.9)
        ax.set_aspect('equal', adjustable='datalim')
        clean(ax, 'Feature 1', 'Feature 2' if ax is axs[0] else None, grid=False)
    save('ml3_hdbscan_vs_dbscan', f, 'Synthetic dense, medium and sparse groups plus uniform background (seed 14); '
         'scikit-learn DBSCAN(min_samples=10) and HDBSCAN(min_cluster_size=20). Grey crosses are noise.',
         kind='computed')


OPTICS_CUTS = [(.35, ORANGE), (.9, PURPLE)]


@fig
def optics():
    X, y = density_data()
    m = OPTICS(min_samples=30, xi=.03).fit(X)
    lab = match_labels(m.labels_, y)
    r = m.reachability_[m.ordering_]
    lo = lab[m.ordering_]
    f, axs = plt.subplots(1, 2, figsize=(11.8, 4.3), layout='constrained', gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax = axs[0]
    xs = np.arange(len(r))
    rr = np.where(np.isinf(r), np.nan, r)
    for k in sorted(set(lo)):
        mk = lo == k
        ax.bar(xs[mk], rr[mk], width=1.0, color=SERIES[k % 6] if k >= 0 else '#B7C2CD', lw=0)
    ax.set_xlim(0, len(r))
    ax.set_ylim(0, 1.6)
    labels = []
    for eps, col in OPTICS_CUTS:
        ax.axhline(eps, color=col, lw=1.5, ls=(0, (5, 3)))
        labels.append(ax.text(6, eps + .025, f'cut at {eps:g} ≈ DBSCAN, ε = {eps:g}', color=col, fontsize=12.5,
                              ha='left', va='bottom', zorder=6))
    rax = ax
    clean(ax, 'Points in OPTICS order', 'Reachability distance')
    panel(ax, 'Reachability plot: valleys are clusters')
    ax = axs[1]
    points(ax, X, lab, s=11)
    ax.set_xlim(-2.6, 5.1)
    ax.set_ylim(-2.1, 6.9)
    ax.set_aspect('equal', adjustable='datalim')
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    panel(ax, 'OPTICS clusters (ξ = 0.03)')
    f.canvas.draw()
    for t in labels:                          # no reachability bar may reach into a label
        bb = t.get_window_extent().transformed(rax.transData.inverted())
        span = (xs >= bb.x0 - 1) & (xs <= bb.x1 + 1)
        assert np.nanmax(rr[span]) < bb.y0, t.get_text()
    save('ml3_optics', f, 'Same synthetic data as the HDBSCAN comparison; OPTICS(min_samples=30, xi=0.03). Bars are '
         'coloured by the extracted cluster; grey = noise. A horizontal cut reproduces DBSCAN at that eps.',
         kind='computed')


# ==================================================================== PART 4
AGG_PTS = np.array([[0.0, 0.0], [.55, .25], [.15, 1.15], [3.0, .2], [3.65, .0], [2.35, 2.45], [3.05, 2.85]])
AGG_NAMES = list('ABCDEFG')


def leaf_colors_link_func(Z, leaf_cluster, colors, other=GREY):
    """link_color_func colouring a link by its cluster when all its leaves share one cluster."""
    n = len(leaf_cluster)
    members = {i: {i} for i in range(n)}
    for i, (a, b, _, _) in enumerate(Z):
        members[n + i] = members[int(a)] | members[int(b)]

    def func(k):
        cl = {leaf_cluster[j] for j in members[k]}
        return colors[cl.pop()] if len(cl) == 1 else other
    return func


@fig
def agglomerative_steps():
    X = AGG_PTS
    Z = linkage(X, method='ward')
    n = len(X)
    f, axs = plt.subplots(1, 2, figsize=(11.8, 4.3), layout='constrained', gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axs[0]
    members = {i: [i] for i in range(n)}
    badge = dict(fontsize=12, fontweight='bold', color='white', ha='center', va='center', zorder=9,
                 bbox=dict(boxstyle='circle,pad=.22', fc=TEAL, ec='none'))
    hulls = []
    for step, (a, b, h, cnt) in enumerate(Z, start=1):
        members[n + step - 1] = members[int(a)] + members[int(b)]
        P = X[members[n + step - 1]]
        r = .40 + .19 * (step - 1) ** .9
        if len(P) < 3:
            P = np.vstack([P, P + 1e-3 * np.array([[1, -1]])])
        H = rounded_hull(P, r)
        hulls.append(H)
        ax.add_patch(Polygon(H, closed=True, fc='none', ec=MUTED if step < n - 1 else INK, lw=1.4,
                             ls='-' if step < n - 1 else (0, (4, 3)), zorder=2))

    def outline(H, ds=.03):                   # closed hull outline sampled every ds data units
        C = np.vstack([H, H[:1]])
        return np.vstack([a + np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / ds)), endpoint=False)[:, None]
                          * (b - a) for a, b in zip(C[:-1], C[1:])])
    lines = [outline(H) for H in hulls]
    placed = []
    for k, H in enumerate(hulls):             # each number on its own outline, clear of everything else
        best, score = None, -np.inf
        for p in lines[k]:
            c = min(np.linalg.norm(X - p, axis=1).min() - .38,
                    min(np.linalg.norm(L - p, axis=1).min() for j, L in enumerate(lines) if j != k) - .19,
                    min([np.linalg.norm(q - p) for q in placed] + [9.]) - .36)
            if c > score:
                best, score = p, c
        assert score > 0, f'no clear place for merge {k + 1}'
        placed.append(best)
        ax.text(*best, str(k + 1), **badge)
    ax.scatter(X[:, 0], X[:, 1], s=330, c=BLUE, edgecolors='white', linewidths=1.2, zorder=5)
    for (x, y), name in zip(X, AGG_NAMES):
        ax.text(x, y, name, fontsize=12.5, fontweight='bold', color='white', ha='center', va='center', zorder=6)
    ax.set_xlim(-1.45, 5.05)
    ax.set_ylim(-1.45, 4.3)
    ax.set_aspect('equal')
    bare(ax)
    panel(ax, 'Nested merges (numbers = merge order)')
    ax = axs[1]
    dn = dendrogram(Z, labels=AGG_NAMES, ax=ax, color_threshold=0, above_threshold_color=INK, leaf_font_size=14)
    xs, ys = np.array(dn['icoord']), np.array(dn['dcoord'])
    order = np.argsort(ys[:, 1])
    for step, k in enumerate(order, start=1):
        ax.text(xs[k, 1:3].mean(), ys[k, 1], str(step), **badge)
    clean(ax, None, 'Ward merge height', grid=False)
    ax.grid(axis='y', alpha=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, ys.max() * 1.12)
    panel(ax, 'Dendrogram of the same merges')
    save('ml3_agglomerative_steps', f, 'Seven hand-placed points; scipy linkage(method="ward"). Merge numbers are the '
         'order of the rows of the linkage matrix.', kind='computed')


def two_groups():
    L = np.array([[0.0, 0.0], [.5, .45], [-.45, .6], [.35, -.5], [.9, -.05]])
    R = np.array([[2.35, 2.05], [2.9, 2.55], [2.75, 1.65], [3.35, 2.15], [2.2, 2.75]])
    return L, R


@fig
def linkage_defs():
    L, R = two_groups()
    D = np.linalg.norm(L[:, None] - R[None], axis=2)
    cL, cR = L.mean(0), R.mean(0)
    nL, nR = len(L), len(R)
    dss = nL * nR / (nL + nR) * np.sum((cL - cR) ** 2)
    f, axs = sub(4, height=3.6)
    titles = [f'Single: closest pair\nd = {D.min():.2f}', f'Complete: farthest pair\nd = {D.max():.2f}',
              f'Average: mean of all {D.size} pairs\nd = {D.mean():.2f}',
              f'Ward: growth in within-SS\nΔSS = {dss:.2f}']
    for k, ax in enumerate(axs):
        if k == 0 or k == 1:
            i, j = np.unravel_index(D.argmin() if k == 0 else D.argmax(), D.shape)
            ax.plot(*zip(L[i], R[j]), color=INK, lw=2.4, zorder=3)
        elif k == 2:
            for p in L:
                for q in R:
                    ax.plot(*zip(p, q), color=MUTED, lw=.8, alpha=.55, zorder=1)
        else:
            c = np.vstack([L, R]).mean(0)
            for p in np.vstack([L, R]):
                ax.plot(*zip(p, c), color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
            for P, cc, col in [(L, cL, BLUE), (R, cR, ORANGE)]:
                for p in P:
                    ax.plot(*zip(p, cc), color=col, lw=1.3, zorder=2)
                ax.scatter(*cc, marker='X', s=130, c=col, edgecolors=INK, linewidths=1, zorder=5)
            ax.scatter(*c, marker='X', s=150, c='white', edgecolors=INK, linewidths=1.4, zorder=5)
        ax.scatter(L[:, 0], L[:, 1], s=110, c=BLUE, edgecolors='white', linewidths=.9, zorder=4)
        ax.scatter(R[:, 0], R[:, 1], s=120, c=ORANGE, marker='^', edgecolors='white', linewidths=.9, zorder=4)
        ax.set_xlim(-.8, 3.7)
        ax.set_ylim(-.85, 3.1)
        ax.set_aspect('equal')
        bare(ax)
        ax.set_title(titles[k], fontsize=14, loc='left', linespacing=1.3)
    save('ml3_linkage_defs', f, 'Two hand-placed five-point groups; distances computed exactly. Ward: increase in the '
         'within-cluster sum of squares if the groups merge (dashed = distances to the merged centroid).',
         kind='computed')


def bridge_data(seed=5):
    rng = np.random.default_rng(seed)
    b1 = rng.normal([0, 0], .38, (110, 2))
    b2 = rng.normal([3.2, 0], .38, (110, 2))
    t = np.linspace(.55, 2.65, 14)
    br = np.c_[t, rng.normal(0, .05, 14)]
    b3 = rng.normal([1.6, 2.6], [.75, .35], (90, 2))
    X = np.vstack([b1, b2, br, b3])
    y = np.repeat([0, 1, 3, 2], [110, 110, 14, 90])
    return X, y


@fig
def linkage_compare():
    X, y = bridge_data()
    f, axs = sub(4, height=3.5)
    for ax, link in zip(axs, ['single', 'complete', 'average', 'ward']):
        lab = match_labels(AgglomerativeClustering(3, linkage=link).fit_predict(X), y)
        points(ax, X, lab, s=11)
        counts = np.bincount(lab[lab >= 0])
        sizes = sorted([c for c in counts.tolist() if c > 0], reverse=True)
        for k in np.where(counts == 1)[0]:     # make a one-point cluster visible
            ax.add_patch(Circle(X[lab == k][0], .2, fc='none', ec=INK, lw=1.4, zorder=5))
        ax.set_xlim(-1.3, 4.5)
        ax.set_ylim(-1.3, 3.6)
        ax.set_aspect('equal')
        bare(ax)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='none', ec=RULE, lw=1.2))
        titled(ax, link.capitalize(), 'sizes ' + ' · '.join(map(str, sizes))
               + (' (ringed)' if (counts == 1).any() else ''))
    save('ml3_linkage_compare', f, 'Synthetic: two dense blobs joined by a 14-point bridge plus an elongated third '
         'group (seed 5); AgglomerativeClustering(n_clusters=3) with each linkage.', kind='computed')


CUT2_LS, CUT4_LS = (0, (6, 3)), (0, (1.2, 2.2))     # coarse and fine cut lines


@fig
def dendrogram_basic():
    """Ward dendrogram of twelve points with one illustrative cut (module variant of
    the shared dendrogram figure: same data)."""
    small, _ = make_blobs(n_samples=12, centers=[[-3, 0], [0, 2], [3, 0]], cluster_std=.45, random_state=42)
    Z = linkage(small, method='ward')
    lab = fcluster(Z, 3, 'distance')
    order = []
    for leaf in dendrogram(Z, no_plot=True)['leaves']:
        if lab[leaf] not in order:
            order.append(lab[leaf])
    leaf_cluster = np.array([order.index(c) for c in lab])
    f, axs = sub(1, height=4.3)
    ax = axs[0]
    dn = dendrogram(Z, labels=[f'S{i + 1}' for i in range(12)], ax=ax, leaf_font_size=13,
                    link_color_func=leaf_colors_link_func(Z, leaf_cluster, SERIES, GREY))
    ax.axhline(3, color=INK, lw=1.8, ls=CUT2_LS)
    x0, x1 = free_gap(dn, 3, 3 + .09 * Z[:, 2].max(), 120)
    ax.text((x0 + x1) / 2, 3.15, 'illustrative cut at height 3', color=INK, fontsize=13, ha='center',
            va='bottom', fontweight='medium')
    clean(ax, 'Observation (leaf order is not a physical coordinate)', 'Ward merge height', grid=False)
    ax.grid(axis='y', alpha=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    save('ml3_dendrogram_basic', f, 'Twelve synthetic points from three blobs (make_blobs random_state=42); scipy '
         'Ward linkage; colours mark the three clusters below the cut at height 3.', kind='computed')


def cut_data():
    X, _ = make_blobs(34, centers=[[0, 0], [1.6, .4], [5.0, 3.2], [6.6, 2.6]], cluster_std=.42, random_state=1)
    return X


@fig
def dendrogram_cut():
    X = cut_data()
    Z = linkage(X, method='ward')
    h = Z[:, 2]
    cut2, cut4 = (h[-2] + h[-1]) / 2, (h[-4] + h[-3]) / 2
    l4 = fcluster(Z, 4, 'maxclust')
    dn0 = dendrogram(Z, no_plot=True)
    order = []
    for leaf in dn0['leaves']:
        if l4[leaf] not in order:
            order.append(l4[leaf])
    cmap = {c: k for k, c in enumerate(order)}
    leaf_cluster = np.array([cmap[c] for c in l4])
    f, axs = plt.subplots(1, 2, figsize=(11.8, 4.3), layout='constrained', gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axs[0]
    dn = dendrogram(Z, ax=ax, no_labels=True, link_color_func=leaf_colors_link_func(Z, leaf_cluster, SERIES, GREY))
    top = h.max()
    for c, k, ls in [(cut2, 2, CUT2_LS), (cut4, 4, CUT4_LS)]:
        ax.axhline(c, color=INK, lw=1.8, ls=ls)
        x0, x1 = free_gap(dn, c, c + .08 * top, 10 * len(X))
        ax.text((x0 + x1) / 2, c + .012 * top, f'cut → {k} clusters', color=INK, fontsize=13, ha='center',
                va='bottom', fontweight='medium')
    clean(ax, 'Observations (leaf order is arbitrary)', 'Ward merge height', grid=False)
    ax.grid(axis='y', alpha=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    panel(ax, 'One tree, many partitions')
    ax = axs[1]
    points(ax, X, leaf_cluster, s=46)
    l2 = fcluster(Z, 2, 'maxclust')
    for c in np.unique(l2):
        P = X[l2 == c]
        ax.add_patch(Polygon(rounded_hull(P, .45), closed=True, fc='none', ec=INK, lw=1.6, ls=CUT2_LS))
    ax.set_aspect('equal', adjustable='datalim')
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    panel(ax, 'Cut at 4 (colours) nests in cut at 2 (dashed)')
    save('ml3_dendrogram_cut', f, 'Synthetic 34 points from four blobs (random_state=1); scipy Ward linkage; '
         'cuts halfway between merge heights; fcluster(maxclust).', kind='computed')


ZONE_COARSE, ZONE_FINE, FACIES_K = 6, 16, 4


def connectivity_chain(n):
    """Sparse adjacency linking each depth sample to the samples above and below."""
    return diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1])


def zonation_fit():
    d = synth_well()
    Z = StandardScaler().fit_transform(well_features(d))
    A = connectivity_chain(len(d))
    unc = AgglomerativeClustering(FACIES_K, linkage='ward').fit_predict(Z)
    coarse = AgglomerativeClustering(ZONE_COARSE, linkage='ward', connectivity=A).fit_predict(Z)
    fine = AgglomerativeClustering(ZONE_FINE, linkage='ward', connectivity=A).fit_predict(Z)
    return d, unc, coarse, fine


def runs(labels):
    """(start, end, label) index runs of a label sequence."""
    cut = np.r_[0, np.where(np.diff(labels) != 0)[0] + 1, len(labels)]
    return [(cut[i], cut[i + 1], labels[cut[i]]) for i in range(len(cut) - 1)]


def depth_band(ax, depth, labels, colors, step=.5):
    for a, b, l in runs(labels):
        ax.axhspan(depth[a] - step / 2, depth[b - 1] + step / 2, color=colors[l], lw=0, zorder=0)


def zone_track(ax, dep, coarse, fine, label=False):
    """Coarse zones as alternating bands with solid boundaries, extra fine boundaries dashed.
    label=True writes the zone names in a column just right of the track."""
    zr = runs(coarse)
    for z, (a, b, l) in enumerate(zr):
        ax.axhspan(dep[a] - .25, dep[b - 1] + .25, color=('#E3EBF3' if z % 2 == 0 else 'white'), lw=0, zorder=0)
        if z:
            ax.axhline(dep[a] - .25, color=TEAL, lw=2.2, zorder=2)
        if label:
            ax.text(1.03, (dep[a] + dep[b - 1]) / 2, f'Z{z + 1}', transform=ax.get_yaxis_transform(), fontsize=13,
                    fontweight='bold', color=TEAL, ha='left', va='center')
    coarse_tops = {a for a, _, _ in zr}
    for a, b, l in runs(fine)[1:]:
        if a not in coarse_tops:
            ax.axhline(dep[a] - .25, color=TEAL, lw=.9, ls=(0, (3, 2)), zorder=2)
    return zr


@fig
def zonation():
    d, unc, coarse, fine = zonation_fit()
    dep = d.DEPTH.values
    unc = match_labels(unc, d.LITH.values)
    f, axs = plt.subplots(1, 4, figsize=(11.8, 4.2), layout='constrained', sharey=True,
                          gridspec_kw=dict(width_ratios=[.42, 1, 1, 1]))
    lith_cols = [tint(SERIES[k], .35) for k in range(4)]
    ax = axs[0]
    depth_band(ax, dep, d.LITH.values, lith_cols)
    ax.set_xticks([])
    ax.set_ylabel('Depth (m)')
    panel(ax, 'Lithology', fs=13)
    ax = axs[1]
    depth_band(ax, dep, unc, [tint(SERIES[k % 6], .55) for k in range(FACIES_K)])
    ax.plot(d.GR, dep, color=INK, lw=.9)
    panel(ax, f'Unconstrained, K = {FACIES_K}: {len(runs(unc))} intervals', fs=13)
    ax = axs[2]
    zr = zone_track(ax, dep, coarse, fine)
    ax.plot(d.GR, dep, color=INK, lw=.9, zorder=3)
    panel(ax, f'Constrained: {len(zr)} and {ZONE_FINE} zones', fs=13)
    ax = axs[3]
    zone_track(ax, dep, coarse, fine, label=True)
    ax.plot(d.RHOB, dep, color=INK, lw=.9, zorder=3)
    ax.set_xlim(1.95, 2.95)
    panel(ax, 'RHOB, same zones', fs=13)
    for ax in axs[1:3]:
        ax.set_xlim(0, 150)
        ax.set_xlabel('GR (API)')
    axs[3].set_xlabel('RHOB (g/cm³)')
    axs[0].set_ylim(dep.max() + .25, dep.min() - .25)
    for ax in axs:
        ax.grid(False)
    handles = [Rectangle((0, 0), 1, 1, fc=lith_cols[k], ec='none', label=LITH[k]) for k in range(4)]
    handles += [Line2D([], [], color=TEAL, lw=2.2, label=f'{ZONE_COARSE}-zone boundary'),
                Line2D([], [], color=TEAL, lw=.9, ls=(0, (3, 2)), label=f'extra {ZONE_FINE}-zone boundary')]
    f.legend(handles=handles, loc='outside lower center', ncol=6, fontsize=12, frameon=False, handlelength=1.3,
             columnspacing=1.4)
    save('ml3_zonation', f, f'Synthetic well S-1 (seed 7): Ward clustering of standardized GR, RHOB, NPHI and log10 RT; '
         f'unconstrained K={FACIES_K}, and with a connectivity matrix linking adjacent samples (cuts at '
         f'{ZONE_COARSE} and {ZONE_FINE} zones).', kind='computed')


# ==================================================================== PART 5
from scipy.stats import norm as _norm, multivariate_normal as _mvn

MIX = dict(pi=(.4, .6), mu=(45., 100.), sd=(12., 14.))     # sand, shale gamma-ray mixture


@fig
def mixture_1d():
    rng = np.random.default_rng(9)
    n = 700
    comp = rng.random(n) < MIX['pi'][0]
    x = np.where(comp, rng.normal(MIX['mu'][0], MIX['sd'][0], n), rng.normal(MIX['mu'][1], MIX['sd'][1], n))
    g = np.linspace(0, 160, 400)
    parts = [MIX['pi'][k] * _norm.pdf(g, MIX['mu'][k], MIX['sd'][k]) for k in range(2)]
    f, axs = half()
    ax = axs[0]
    ax.hist(x, bins=np.arange(0, 162, 5), density=True, color='#D5DEE7', edgecolor='white', lw=.6, zorder=1)
    cols = [ORANGE, BLUE]
    names = ['sand component, π = 0.4', 'shale component, π = 0.6']
    for part, col in zip(parts, cols):
        ax.fill_between(g, part, color=col, alpha=.12, lw=0, zorder=2)
        ax.plot(g, part, color=col, lw=2.2, ls=(0, (5, 2.5)), zorder=3)
    ax.plot(g, parts[0] + parts[1], color=INK, lw=2.6, zorder=4)
    ax.text(3, .0150, names[0], color=ORANGE, fontsize=12.5, ha='left')
    ax.text(118, .0184, names[1], color=BLUE, fontsize=12.5, ha='center')
    ax.annotate('mixture p(x)', (123, .0048), xytext=(141, .0118), color=INK, fontsize=13, ha='center',
                fontweight='medium', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkB=2))
    x0 = 70.0
    d0 = [MIX['pi'][k] * _norm.pdf(x0, MIX['mu'][k], MIX['sd'][k]) for k in range(2)]
    r = d0[0] / sum(d0)
    ax.plot([x0, x0], [0, .0226], color=MUTED, lw=1.3, ls=(0, (2, 2)), zorder=2)
    ax.text(x0, .0229, f'GR = {x0:.0f}: responsibility\nsand {r:.2f}, shale {1 - r:.2f}', fontsize=12.5,
            color=INK, va='bottom', ha='center', linespacing=1.2)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, .0268)
    ax.set_yticks([0, .005, .01, .015, .02, .025])
    clean(ax, 'GR (API)', 'Density')
    save('ml3_mixture_1d', f, 'Exact two-component normal mixture (weights 0.4/0.6, means 45/100 API, SD 12/14 API) '
         'with a 700-sample synthetic histogram (seed 9).', kind='computed')


RESP_CMAP = LinearSegmentedColormap.from_list('ml3_resp', [ORANGE, '#A9B4BF', BLUE])   # grey = ambiguous


@fig
def gmm_soft():
    """Soft memberships of a two-component mixture (module variant of the shared gmm
    figure: same data and model, course colours, visible 50/50 points)."""
    rng = np.random.default_rng(42)
    X = np.r_[rng.multivariate_normal([-1, 0], [[1, .55], [.55, .65]], 150),
              rng.multivariate_normal([1.3, .6], [[.8, -.45], [-.45, .6]], 150)]
    m = GaussianMixture(2, n_init=5, random_state=42).fit(X)
    p = m.predict_proba(X)[:, 0]
    f, axs = sub(1, height=4.3)
    ax = axs[0]
    sc = ax.scatter(X[:, 0], X[:, 1], c=p, cmap=RESP_CMAP, vmin=0, vmax=1, s=34, edgecolors='white',
                    linewidths=.4, zorder=3)
    cb = f.colorbar(sc, ax=ax, ticks=[0, .25, .5, .75, 1])
    cb.set_label('Responsibility for component 1')
    clean(ax, 'Feature 1', 'Feature 2')
    save('ml3_gmm_soft', f, 'Two synthetic overlapping bivariate normal groups of 150 points (numpy seed 42); '
         'GaussianMixture(2, n_init=5, random_state=42); colour = predict_proba for component 1.',
         kind='computed')


def em_data():
    rng = np.random.default_rng(31)
    comps = [([0, 0], [[1.0, .6], [.6, .7]], 160), ([3.6, 1.2], [[.5, -.25], [-.25, .6]], 120),
             ([1.2, 3.4], [[1.1, 0], [0, .35]], 120)]
    X = np.vstack([rng.multivariate_normal(m, c, n) for m, c, n in comps])
    y = np.repeat([0, 1, 2], [c[2] for c in comps])
    return X, y


def em_run(X, iters, K=3):
    pi = np.full(K, 1 / K)
    mu = np.array([[1.4, 1.0], [1.8, 1.5], [1.0, 1.6]])
    S = np.array([np.eye(2) * 1.2 for _ in range(K)])
    states = []
    for t in range(iters + 1):
        dens = np.column_stack([pi[k] * _mvn.pdf(X, mu[k], S[k]) for k in range(K)])
        ll = np.log(dens.sum(1)).mean()
        R = dens / dens.sum(1, keepdims=True)
        states.append((pi.copy(), mu.copy(), S.copy(), R, ll))
        Nk = R.sum(0)
        pi = Nk / len(X)
        mu = (R.T @ X) / Nk[:, None]
        S = np.array([((R[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + 1e-6 * np.eye(2) for k in range(K)])
    return states


def blend(R, cols):
    C = np.array([to_rgb(c) for c in cols])
    return np.clip(R @ C, 0, 1)


@fig
def em_iterations():
    X, y = em_data()
    states = em_run(X, 40)
    cols = [BLUE, ORANGE, TEAL]
    f, axs = sub(4, height=4.0)
    for ax, t in zip(axs, [0, 1, 5, 40]):
        pi, mu, S, R, ll = states[t]
        ax.scatter(X[:, 0], X[:, 1], s=14, c=blend(R, cols), edgecolors='white', linewidths=.25, zorder=3)
        for k in range(3):
            ellipse_from_cov(ax, mu[k], S[k], cols[k], n_std=(1, 2), alpha=.10, lw=1.6, z=4)
            ax.scatter(*mu[k], marker='X', s=110, c=cols[k], edgecolors=INK, linewidths=1, zorder=6)
        titled(ax, f'Iteration {t}' if t else 'Start', 'mean log-likelihood ' + f'{ll:.3f}'.replace('-', '−'))
        ax.set_xlim(-3.0, 6.0)                 # contains every 2-SD ellipse drawn
        ax.set_ylim(-2.5, 5.75)
        ax.set_aspect('equal')
        bare(ax)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='none', ec=RULE, lw=1.2))
    save('ml3_em_iterations', f, 'Synthetic three-component data (seed 31). EM for a full-covariance Gaussian mixture '
         'written in numpy from a deliberately poor start; point colour mixes component colours by responsibility.',
         kind='computed')


def cov_data():
    rng = np.random.default_rng(17)
    comps = [([0, 0], [[1.6, 1.0], [1.0, .9]], 150), ([4.2, .2], [[.35, 0], [0, 1.4]], 120),
             ([1.8, 3.6], [[1.2, -.8], [-.8, .9]], 130)]
    X = np.vstack([rng.multivariate_normal(m, c, n) for m, c, n in comps])
    y = np.repeat([0, 1, 2], [c[2] for c in comps])
    return X, y


@fig
def cov_types():
    X, y = cov_data()
    f, axs = sub(4, height=3.7)
    for ax, ct in zip(axs, ['spherical', 'diag', 'tied', 'full']):
        m = GaussianMixture(3, covariance_type=ct, n_init=5, random_state=0).fit(X)
        pred = m.predict(X)
        lab = match_labels(pred, y)
        order = [int(np.bincount(lab[pred == j], minlength=3).argmax()) for j in range(3)]
        points(ax, X, lab, s=10, alpha=.8)
        for j, cov in enumerate(gmm_covs(m)):
            ellipse_from_cov(ax, m.means_[j], cov, SERIES[order[j]], n_std=(1, 2), alpha=.12, lw=1.7, z=4)
        ax.set_title(f'{ct}: {m._n_parameters()} parameters\nBIC {m.bic(X):.0f}', fontsize=14, loc='left',
                     linespacing=1.3)
        ax.set_xlim(-3.6, 6.6)
        ax.set_ylim(-3.4, 6.9)
        ax.set_aspect('equal')
        bare(ax)
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='none', ec=RULE, lw=1.2))
    save('ml3_cov_types', f, 'Synthetic three-component data (seed 17); GaussianMixture(K=3, n_init=5, random_state=0) '
         'with each covariance_type; ellipses at 1 and 2 standard deviations.', kind='computed')


@fig
def bic_curve():
    X, _ = cov_data()
    ks = list(range(1, 9))
    fits = [GaussianMixture(k, covariance_type='full', n_init=5, random_state=0).fit(X) for k in ks]
    bic = np.array([m.bic(X) for m in fits])
    aic = np.array([m.aic(X) for m in fits])
    f, axs = half()
    ax = axs[0]
    ax.plot(ks, bic, '-o', color=BLUE, lw=2.4, ms=7)
    ax.plot(ks, aic, '-s', color=ORANGE, lw=2.4, ms=6)
    kb, ka = ks[int(np.argmin(bic))], ks[int(np.argmin(aic))]
    ax.scatter([kb], [bic.min()], s=190, facecolors='none', edgecolors=BLUE, linewidths=2, zorder=5)
    ax.scatter([ka], [aic.min()], s=190, facecolors='none', edgecolors=ORANGE, linewidths=2, zorder=5)
    endlabel(ax, ks[-1], bic[-1], 'BIC', BLUE, fs=14, dy=8)
    endlabel(ax, ks[-1], aic[-1], 'AIC', ORANGE, fs=14, dy=-8)
    ax.text(kb + .75, (bic[kb] + aic[kb]) / 2 + 4, f'lowest BIC: K = {kb}', color=BLUE, fontsize=12.5, ha='left',
            va='center')
    ax.set_xticks(ks)
    ax.set_xlim(.6, 9.0)
    clean(ax, 'Number of components K', 'Criterion (lower is preferred)')
    save('ml3_bic_curve', f, 'Same synthetic data as the covariance-type figure (three generating components, seed 17); '
         'GaussianMixture(covariance_type="full", n_init=5, random_state=0) for K = 1-8.', kind='computed')


def skewed_groups():
    rng = np.random.default_rng(23)
    t = rng.uniform(0, np.pi, 240)
    arc = np.c_[np.cos(t) * 2.2, np.sin(t) * 1.6] + rng.normal(0, .16, (240, 2))
    blob = np.exp(rng.normal([0, 0], [.42, .30], (220, 2))) + np.array([-.9, -2.6])
    X = np.vstack([arc, blob])
    y = np.repeat([0, 1], [240, 220])
    return X, y


@fig
def components_vs_groups():
    X, y = skewed_groups()
    ks = list(range(1, 11))
    fits = [GaussianMixture(k, covariance_type='full', n_init=5, random_state=0).fit(X) for k in ks]
    bic = np.array([m.bic(X) for m in fits])
    kb = ks[int(np.argmin(bic))]
    m = fits[kb - 1]
    f, axs = plt.subplots(1, 3, figsize=(11.8, 4.3), layout='constrained', gridspec_kw=dict(width_ratios=[1, 1, .9]))
    ax = axs[0]
    points(ax, X, y, s=12)
    panel(ax, 'Two generating groups')
    ax = axs[1]
    pred = m.predict(X)
    ax.scatter(X[:, 0], X[:, 1], s=10, c='#AFBBC7', zorder=2)
    for j in range(kb):
        col = SERIES[(j + 2) % 6]
        ellipse_from_cov(ax, m.means_[j], m.covariances_[j], col, n_std=(1.5,), alpha=.16, lw=1.8, z=3)
        ax.scatter(*m.means_[j], marker='X', s=100, c=col, edgecolors=INK, linewidths=1, zorder=5)
    panel(ax, f'BIC-preferred mixture: K = {kb}')
    for a in axs[:2]:
        a.set_xlim(-2.9, 3.4)
        a.set_ylim(-2.3, 2.3)
        a.set_aspect('equal', adjustable='datalim')
        clean(a, 'Feature 1', 'Feature 2' if a is axs[0] else None, grid=False)
    ax = axs[2]
    ax.plot(ks, bic, '-o', color=BLUE, lw=2.4, ms=6)
    ax.scatter([kb], [bic.min()], s=180, facecolors='none', edgecolors=BLUE, linewidths=2, zorder=5)
    ax.set_xticks(ks[::1])
    clean(ax, 'Components K', 'BIC')
    panel(ax, 'BIC by K')
    save('ml3_components_vs_groups', f, 'Synthetic: a curved group and a log-normal (skewed) group, seed 23. '
         'GaussianMixture(full, n_init=5, random_state=0); ellipses at 1.5 standard deviations.', kind='computed')


def overlap_data():
    rng = np.random.default_rng(41)
    comps = [([0, 0], [[.9, .3], [.3, .6]], 170), ([2.7, .5], [[.6, -.2], [-.2, .7]], 150),
             ([1.1, 2.8], [[.8, .1], [.1, .5]], 130)]
    X = np.vstack([rng.multivariate_normal(m, c, n) for m, c, n in comps])
    y = np.repeat([0, 1, 2], [c[2] for c in comps])
    return X, y


@fig
def gmm_uncertainty():
    X, y = overlap_data()
    m = GaussianMixture(3, covariance_type='full', n_init=5, random_state=0).fit(X)
    pred = m.predict(X)
    lab = match_labels(pred, y)
    order = [int(np.bincount(lab[pred == j], minlength=3).argmax()) for j in range(3)]
    rmax = m.predict_proba(X).max(1)
    unsure = rmax < .8
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    points(ax, X, lab, s=16)
    ax.scatter(X[unsure, 0], X[unsure, 1], s=95, facecolors='none', edgecolors=INK, linewidths=1.3, zorder=5)
    titled(ax, 'Hard labels (argmax)', f'ringed: max responsibility < 0.8 ({unsure.sum()} of {len(X)}, '
           f'{unsure.mean():.0%})')
    ax = axs[1]
    xx, yy = np.meshgrid(np.linspace(-7, 9, 640), np.linspace(-5, 8, 520))
    U = 1 - m.predict_proba(np.c_[xx.ravel(), yy.ravel()]).max(1).reshape(xx.shape)
    cf = ax.contourf(xx, yy, U, levels=np.linspace(0, .67, 12), cmap=CMAP_ORANGE, zorder=0)
    ax.contour(xx, yy, U, levels=[.2], colors=[INK], linewidths=1.4, zorder=2)
    for j, cov in enumerate(gmm_covs(m)):
        ellipse_from_cov(ax, m.means_[j], cov, SERIES[order[j]], n_std=(2,), alpha=0, lw=1.8, z=3)
    ax.scatter(X[:, 0], X[:, 1], s=5, c=INK, alpha=.35, zorder=4)
    cb = f.colorbar(cf, ax=ax, shrink=.9, ticks=[0, .2, .4, .6])
    cb.set_label('1 − max responsibility')
    titled(ax, 'Uncertainty map', 'black line: 1 − max responsibility = 0.2')
    for a in axs:
        a.set_xlim(-3, 5)
        a.set_ylim(-2.6, 5)
        a.set_aspect('equal', adjustable='datalim')
        clean(a, 'Feature 1', 'Feature 2' if a is axs[0] else None, grid=False)
    save('ml3_gmm_uncertainty', f, 'Synthetic overlapping groups (seed 41); GaussianMixture(K=3, full, n_init=5, '
         'random_state=0). 2-SD ellipses of the fitted components.', kind='computed')


# ==================================================================== PART 6
def index_data():
    X, _ = make_blobs(300, centers=[[0, 0], [1.9, 0], [6, .3], [7.9, 0], [3.2, 4.4]], cluster_std=.55,
                      random_state=0)
    return X


@fig
def internal_indices():
    X = index_data()
    ks = list(range(2, 9))
    labs = {k: KMeans(k, n_init=10, random_state=0).fit_predict(X) for k in ks}
    sil = [silhouette_score(X, labs[k]) for k in ks]
    db = [davies_bouldin_score(X, labs[k]) for k in ks]
    ch = [calinski_harabasz_score(X, labs[k]) for k in ks]
    f, axs = plt.subplots(1, 4, figsize=(11.8, 4.0), layout='constrained', gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    ax = axs[0]
    points(ax, X, labs[5], s=12)
    ax.set_aspect('equal', adjustable='datalim')
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    panel(ax, 'Data (K-means, K = 5)')
    for ax, vals, name, better, col in [(axs[1], sil, 'Silhouette', 'higher', TEAL),
                                        (axs[2], db, 'Davies–Bouldin', 'lower', ORANGE),
                                        (axs[3], ch, 'Calinski–Harabasz', 'higher', BLUE)]:
        vals = np.array(vals)
        kb = ks[int(np.argmax(vals) if better == 'higher' else np.argmin(vals))]
        ax.plot(ks, vals, '-o', color=col, lw=2.2, ms=6)
        ax.scatter([kb], [vals[ks.index(kb)]], s=170, facecolors='none', edgecolors=INK, linewidths=2, zorder=5)
        ax.set_xticks(ks)
        clean(ax, 'K', None)
        panel(ax, f'{name}\n{better} is better: K = {kb}')
        ax.title.set_linespacing(1.3)
    save('ml3_internal_indices', f, 'Synthetic five blobs with two close pairs (make_blobs random_state=0); '
         'KMeans(n_init=10, random_state=0) for K = 2-8; scikit-learn metric implementations.', kind='computed')


@fig
def contingency():
    d = synth_well()
    Z = StandardScaler().fit_transform(well_features(d))
    lab = KMeans(5, n_init=10, random_state=0).fit_predict(Z)
    ct = pd.crosstab(d.LITH, lab).values
    ari = adjusted_rand_score(d.LITH, lab)
    nmi = normalized_mutual_info_score(d.LITH, lab)
    f, axs = half()
    ax = axs[0]
    ax.imshow(np.log1p(ct), cmap=CMAP_SEQ, vmin=0, vmax=np.log1p(ct.max()) * 1.15, aspect='auto')
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct[i, j]
            ax.text(j, i, str(v), ha='center', va='center', fontsize=14, fontweight='medium',
                    color='white' if np.log1p(v) > .55 * np.log1p(ct.max()) else INK)
    ax.set_xticks(range(ct.shape[1]), [f'C{j + 1}' for j in range(ct.shape[1])])
    ax.set_yticks(range(4), LITH)
    ax.tick_params(length=0, labelsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('Cluster (numbering is arbitrary)')
    ax.set_title(f'ARI = {ari:.2f}     NMI = {nmi:.2f}', fontsize=15, loc='left')
    save('ml3_contingency', f, 'Synthetic well S-1 (seed 7): reference lithology versus KMeans(K=5, n_init=10, '
         'random_state=0) on standardized logs; counts of depth samples.', kind='computed')


def stability_runs(n_boot=30):
    d = synth_well()
    Z = StandardScaler().fit_transform(well_features(d))
    rng = np.random.default_rng(0)
    out = {}
    for K in range(2, 8):
        ref = KMeans(K, n_init=10, random_state=0).fit(Z)
        vals = []
        for b in range(n_boot):
            idx = rng.integers(0, len(Z), len(Z))
            m = KMeans(K, n_init=3, random_state=b).fit(Z[idx])
            vals.append(adjusted_rand_score(ref.labels_, m.predict(Z)))
        out[K] = np.array(vals)
    return out


@fig
def stability():
    out = stability_runs()
    ks = list(out)
    f, axs = half()
    ax = axs[0]
    rng = np.random.default_rng(1)
    for k in ks:
        v = out[k]
        ax.scatter(k + rng.uniform(-.17, .17, len(v)), v, s=22, c=BLUE, alpha=.55, edgecolors='none', zorder=3)
        ax.plot([k - .28, k + .28], [np.median(v)] * 2, color=INK, lw=2.6, zorder=4)
    ax.set_xticks(ks)
    ax.set_ylim(.3, 1.03)
    clean(ax, 'Number of clusters K', 'ARI with the full-data clustering')
    ax.text(2.0, .345, 'dots: 30 bootstrap refits per K; bar: median', fontsize=12.5, color=MUTED)
    save('ml3_stability', f, 'Synthetic well S-1 (seed 7), standardized logs. For each K, KMeans refitted on 30 '
         'bootstrap resamples (numpy seed 0), labels predicted for all samples and compared with the full-data fit.',
         kind='computed')


def s1_kmeans4():
    d = synth_well()
    sc = StandardScaler().fit(well_features(d))
    Z = sc.transform(well_features(d))
    km = KMeans(4, n_init=10, random_state=0).fit(Z)
    lab = match_labels(km.labels_, d.LITH.values)
    return d, Z, lab


@fig
def profile_heatmap():
    d, Z, lab = s1_kmeans4()
    cols = ['GR', 'RHOB', 'NPHI', 'LRT']
    heads = ['GR\n(API)', 'RHOB\n(g/cm³)', 'NPHI\n(v/v)', 'RT\n(ohm·m)']
    zc = np.array([Z[lab == k].mean(0) for k in range(4)])
    raw = d.groupby(lab)[cols].mean()
    fmt = [lambda v: f'{v:.0f}', lambda v: f'{v:.2f}', lambda v: f'{v:.2f}', lambda v: f'{10 ** v:.1f}']
    reading = ['high GR, low RT:\nshale-like', 'low GR, low RT:\nwet-sand-like', 'low GR, high RT,\nlow NPHI: gas-sand-like',
               'high RHOB, low NPHI:\ntight-carbonate-like']
    f, axs = plt.subplots(1, 2, figsize=(6.3, 4.6), layout='constrained', gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    ax.imshow(zc, cmap=CMAP_DIV, vmin=-2.2, vmax=2.2, aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, fmt[j](raw.iloc[i, j]), ha='center', va='center', fontsize=13.5, fontweight='medium',
                    color='white' if abs(zc[i, j]) > 1.5 else INK)
    ax.set_xticks(range(4), heads)
    ax.xaxis.tick_top()
    ax.set_yticks(range(4), [f'C{k + 1}  n={int((lab == k).sum())}' for k in range(4)])
    ax.tick_params(length=0, labelsize=12.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax = axs[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(3.5, -.5)
    for k in range(4):
        ax.text(.02, k, reading[k], fontsize=12, color=SERIES[k], va='center', ha='left', linespacing=1.2,
                fontstyle='italic')
    ax.set_axis_off()
    ax.set_title('hypothesis to test', fontsize=12.5, color=MUTED, loc='left', fontweight='normal')
    save('ml3_profile_heatmap', f, 'Synthetic well S-1 (seed 7): KMeans(K=4) on standardized logs. Cells show cluster '
         'means in original units (RT as geometric mean); colour = standardized mean, blue above and orange below '
         'the well average.',
         kind='computed')


@fig
def depth_tracks():
    d, Z, lab = s1_kmeans4()
    s = silhouette_samples(Z, lab)
    dep = d.DEPTH.values
    f, axs = plt.subplots(1, 6, figsize=(11.8, 4.2), layout='constrained', sharey=True,
                          gridspec_kw=dict(width_ratios=[1, 1, 1, .45, 1, .45]))
    low = s < .25
    for ax in axs:
        for a, b, l in runs(low.astype(int)):
            if l == 1:
                ax.axhspan(dep[a] - .25, dep[b - 1] + .25, color=tint(GOLD, .62), lw=0, zorder=0)
    axs[0].plot(d.GR, dep, color=INK, lw=.9)
    axs[0].set_xlim(0, 150)
    axs[0].set_xlabel('GR (API)')
    axs[1].plot(d.RHOB, dep, color=INK, lw=.9)
    axs[1].set_xlim(1.95, 2.95)
    axs[1].set_xlabel('RHOB (g/cm³)')
    axs[2].semilogx(d.RT, dep, color=INK, lw=.9)
    axs[2].set_xlim(.5, 500)
    axs[2].set_xticks([1, 10, 100], ['1', '10', '100'])
    axs[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[2].set_xlabel('RT (ohm·m)')
    depth_band(axs[3], dep, lab, [SERIES[k] for k in range(4)])
    axs[3].set_xticks([])
    axs[3].set_xlabel('K-means\nK = 4', linespacing=1.1)
    axs[4].fill_betweenx(dep, 0, s, color=TEAL, alpha=.22, lw=0, zorder=1)
    axs[4].plot(s, dep, color=TEAL, lw=.9)
    axs[4].axvline(.25, color=GOLD, lw=1.4, ls=(0, (4, 2)))
    axs[4].set_xlim(-.2, 1)
    axs[4].set_xlabel('Silhouette s')
    depth_band(axs[5], dep, d.LITH.values, [SERIES[k] for k in range(4)])
    axs[5].set_xticks([])
    axs[5].set_xlabel('Core\nlithology', linespacing=1.1)
    axs[0].set_ylim(dep.max() + .25, dep.min() - .25)
    axs[0].set_ylabel('Depth (m)')
    for ax in axs:
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=11.5)
    handles = [Rectangle((0, 0), 1, 1, fc=SERIES[k], ec='none', label=f'C{k + 1} / {LITH[k]}') for k in range(4)]
    handles.append(Rectangle((0, 0), 1, 1, fc=tint(GOLD, .62), ec='none', label='silhouette < 0.25'))
    f.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=12, frameon=False, handlelength=1.2)
    save('ml3_depth_tracks', f, 'Synthetic well S-1 (seed 7): KMeans(K=4) on standardized GR, RHOB, NPHI, log10 RT; '
         'per-sample silhouette from sklearn; "core lithology" is the simulation truth.', kind='computed')


@fig
def method_flow():
    f, ax = canvas(11.8, 4.3)
    rows = [('Compact, similar-size groups;\nK can be tested', 'K-means\nMiniBatchKMeans', 'scaling, restarts,\noutliers'),
            ('Irregular shapes, noise,\nunknown number of groups', 'DBSCAN\nHDBSCAN', 'density scale,\nnoise fraction'),
            ('Nested structure, or ordered\nsamples (depth, time)', 'Agglomerative\n(+ connectivity)',
             'linkage, memory\ngrows with n²'),
            ('Overlapping groups; need\nmembership probabilities', 'Gaussian\nmixture',
             'covariance type,\nBIC, stability')]
    xs = [(.2, 4.3), (5.1, 2.9), (8.6, 3.0)]
    heads = ['What you expect', 'Candidate method', 'First check']
    for (x, w), h, col in zip(xs, heads, [MUTED, BLUE, TEAL]):
        ax.text(x + .05, 4.08, h.upper(), fontsize=12, color=col, fontweight='semibold', ha='left', va='center')
    rh, gap, y0 = .76, .16, 3.74
    for i, (a, b, c) in enumerate(rows):
        y = y0 - (i + 1) * rh - i * gap
        b1 = box(ax, xs[0][0], y, xs[0][1], rh, a, GREY, fs=13, weight='normal')
        b2 = box(ax, xs[1][0], y, xs[1][1], rh, b, BLUE, fs=13.5, weight='semibold')
        b3 = box(ax, xs[2][0], y, xs[2][1], rh, c, TEAL, fs=13, weight='normal')
        arrow(ax, b1.r, b2.l, color=INK, shrink=2)
        arrow(ax, b2.r, b3.l, color=INK, shrink=2)
    save('ml3_method_flow', f, 'Schematic method shortlist; a starting point, not a rule.', kind='schematic')
