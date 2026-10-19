"""Figures for module key 'ml5'. Figure names must start with 'ml5_'.

Module 5: rare events, anomalies, and model selection. Colour meaning used
throughout this module: ordinary rows = BLUE, events / anomalies = ORANGE,
thresholds and robust rules = TEAL (dashed), classical rules = grey dashed,
training = BLUE, validation = TEAL, test = ORANGE.
"""
from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

KEY = 'ml5'
fig = register(KEY)
GREY = '#56687A'
LIGHT = '#C9D3DD'
MODULE = ROOT / 'machine_learning_course' / 'Module_5_Advanced_Supervised'
WELL_CSV = MODULE / 'exercises' / 'anomaly_detection' / 'well_logs_data.csv'


def _hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ====================================================================== opening
@fig
def ml5_fleet_timeline():
    """Motivating case: rare pump trips in a fleet, plus a zoom on one trip."""
    rng = np.random.default_rng(5)
    n_pumps, days, n_trips = 10, 730, 9
    pumps = rng.integers(0, n_pumps, n_trips)
    tdays = np.sort(rng.uniform(25, days - 10, n_trips))
    f = plt.figure(figsize=FULL, layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[2.25, 1])
    ax, az = f.add_subplot(gs[0]), f.add_subplot(gs[1])
    for p in range(n_pumps):
        ax.plot([0, days], [p, p], color=LIGHT, lw=7, solid_capstyle='butt', zorder=1)
    ax.scatter(tdays, pumps, marker='v', s=150, c=ORANGE, edgecolors='white', linewidths=.8, zorder=3)
    ax.set_yticks(range(n_pumps), [f'ESP-{i + 1:02d}' for i in range(n_pumps)])
    ax.set_ylim(n_pumps - .4, -.8)
    ax.set_xlim(0, days)
    ax.set_xticks([0, 180, 365, 545, 730])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Day of operation')
    ax.spines['left'].set_visible(False)
    ax.set_title('10 pumps × 730 days of hourly rows; triangles mark the 9 trips', loc='left', fontsize=14)
    # zoom on one trip: a slow drift that starts long before the labelled rows
    h = np.arange(-72, 0)
    base = 42 + .45 * np.sin(2 * np.pi * h / 24) + rng.normal(0, .28, h.size)
    cur = base + np.where(h > -30, .09 * (h + 30), 0)
    az.axvspan(-6, 0, color=ORANGE, alpha=.18, lw=0)
    az.axvspan(-30, -6, color=GOLD, alpha=.10, lw=0)
    az.plot(h, cur, color=BLUE, lw=2)
    az.axvline(0, color=ORANGE, lw=2)
    az.text(-29.3, 40.35, 'drift\nstarts', ha='left', va='bottom', fontsize=12, color=GOLD)
    az.annotate('6 labelled\nrows', (-3, 46.2), (-50, 46.4), fontsize=12, color=ORANGE, va='center',
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    az.text(-1, 40.35, 'trip', ha='right', va='bottom', fontsize=12, color=ORANGE)
    az.set_xlim(-72, 3)
    az.set_ylim(40, 47.5)
    az.set_xticks([-72, -48, -24, 0])
    clean(az, 'Hours relative to trip', 'Motor current (A)')
    az.set_title('Zoom on one trip', loc='left', fontsize=14)
    save('ml5_fleet_timeline', f, 'Synthetic fleet (seed 5): 10 pumps, 730 days, 9 trips; zoomed current trace is '
         'synthetic and illustrative, not field data.', kind='computed')


# ====================================================================== part 1
@fig
def ml5_settings():
    """Outlier detection vs novelty detection vs supervised rare-event learning."""
    rng = np.random.default_rng(7)
    cov = [[1, .55], [.55, 1]]
    normal = rng.multivariate_normal([0, 0], cov, 190)
    ang = rng.uniform(0, 2 * np.pi, 8)
    rad = rng.uniform(3.1, 3.9, 8)
    odd = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    f, axs = sub(3)
    lim = (-4.6, 4.6)
    # (a) outlier detection: one unlabelled, contaminated set; rank within it
    X = np.r_[normal, odd]
    iso = IsolationForest(random_state=0).fit(X)
    top = np.argsort(iso.score_samples(X))[:8]
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=20, c='#8C99A6', edgecolors='white', linewidths=.4, zorder=2)
    ax.scatter(X[top, 0], X[top, 1], s=170, facecolors='none', edgecolors=ORANGE, linewidths=2, zorder=3)
    ax.set_title('Outlier detection')
    ax.set_xlabel('Training labels: none\n(the data are contaminated)')
    ax.text(-4.4, 4.2, 'rings: 8 most isolated rows', fontsize=12, color=ORANGE, va='top')
    # (b) novelty detection: clean reference, score new rows
    ax = axs[1]
    ref = normal
    new = np.array([[.3, .2], [-.8, -.6], [1.1, .9], [-.2, .9], [3.2, -2.4], [-2.9, 2.8]])
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(ref)
    flag = lof.predict(new) == -1
    ax.scatter(ref[:, 0], ref[:, 1], s=20, c=BLUE, edgecolors='white', linewidths=.4, zorder=2)
    ax.scatter(new[:, 0], new[:, 1], s=90, marker='X', c=INK, zorder=4)
    ax.scatter(new[flag, 0], new[flag, 1], s=260, facecolors='none', edgecolors=ORANGE, linewidths=2, zorder=3)
    ax.set_title('Novelty detection')
    ax.set_xlabel('Training labels: normal only\n(new rows scored later)')
    ax.text(-4.4, 4.2, '× new rows; rings: flagged', fontsize=12, color=INK, va='top')
    # (c) supervised: both classes labelled, events rare
    ax = axs[2]
    ev = rng.multivariate_normal([2.1, 2.0], [[.35, 0], [0, .35]], 14)
    Xs = np.r_[normal, ev]
    ys = np.r_[np.zeros(len(normal)), np.ones(len(ev))]
    clf = LogisticRegression().fit(Xs, ys)
    xx, yy = np.meshgrid(np.linspace(*lim, 300), np.linspace(*lim, 300))
    zz = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, .5, 1], colors=[BLUE_T, ORANGE_T], zorder=0)
    ax.contour(xx, yy, zz, levels=[.5], colors=[INK], linewidths=1.3, zorder=1)
    ax.scatter(normal[:, 0], normal[:, 1], s=20, c=BLUE, edgecolors='white', linewidths=.4, zorder=2)
    ax.scatter(ev[:, 0], ev[:, 1], s=60, marker='^', c=ORANGE, edgecolors='white', linewidths=.5, zorder=3)
    ax.set_title('Supervised rare events')
    ax.set_xlabel('Training labels: normal and events\n(14 events among 204 rows)')
    for ax in axs:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal')
        _hide_ticks(ax)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
    save('ml5_settings', f, 'Synthetic 2-D data (seed 7). Isolation Forest ranks the contaminated set; novelty LOF '
         '(n_neighbors=20) scores new rows; logistic regression separates labelled events.', kind='computed')


def _sensor_series():
    """Synthetic hourly discharge pressure with three injected anomaly types."""
    rng = np.random.default_rng(11)
    t = np.arange(240)
    x = 60 + 6 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, .6, t.size)
    x[52] = 81.0                                         # point anomaly
    ctx = np.arange(5 * 24 + 1, 5 * 24 + 5)               # early hours: normally near the trough
    x[ctx] = 65.2 + rng.normal(0, .5, ctx.size)          # peak-level values at trough time
    col = np.arange(178, 200)                            # stuck sensor: flat, plausible level
    x[col] = 60.4
    return t, x, ctx, col


@fig
def ml5_anomaly_types():
    t, x, ctx, col = _sensor_series()
    mu, sd = x.mean(), x.std()
    f, axs = sub(1)
    ax = axs[0]
    ax.axhspan(mu - 3 * sd, mu + 3 * sd, color=PANEL, zorder=0)
    ax.axhline(mu + 3 * sd, color=GREY, ls='--', lw=1.3)
    ax.axhline(mu - 3 * sd, color=GREY, ls='--', lw=1.3)
    ax.text(10.08, mu + 3 * sd, 'mean + 3 SD', fontsize=12, color=GREY, va='center')
    ax.text(10.08, mu - 3 * sd, 'mean − 3 SD', fontsize=12, color=GREY, va='center')
    ax.plot(t / 24, x, color=BLUE, lw=1.8, zorder=2)
    ax.scatter(t[52] / 24, x[52], s=90, c=ORANGE, zorder=4, edgecolors='white')
    ax.axvspan((ctx[0] - .5) / 24, (ctx[-1] + .5) / 24, color=PURPLE, alpha=.16, lw=0)
    ax.plot(t[ctx] / 24, x[ctx], color=PURPLE, lw=3, zorder=3)
    ax.axvspan((col[0] - .5) / 24, (col[-1] + .5) / 24, color=GOLD, alpha=.16, lw=0)
    ax.plot(t[col] / 24, x[col], color=GOLD, lw=3.2, zorder=3)
    ax.annotate('Point: outside the\nnormal range', (t[52] / 24, x[52]), (.1, 84.5), fontsize=13,
                color=ORANGE, va='center', arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    ax.annotate('Contextual: a normal value\nat the wrong time of day', (ctx[1] / 24, x[ctx[1]] + .5),
                (3.0, 84.5), fontsize=13, color=PURPLE, va='center',
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.2))
    ax.annotate('Collective: each value plausible,\nthe flat sequence is not', (col[10] / 24, 60.6),
                (6.15, 84.5), fontsize=13, color=GOLD, va='center',
                arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(44, 88)
    ax.set_xticks(range(11))
    clean(ax, 'Time (days)', 'Discharge pressure (bar)')
    save('ml5_anomaly_types', f, 'Synthetic hourly series (seed 11): daily cycle plus noise with an injected spike, '
         'four peak-level hours at trough time, and a 22-hour flat segment.', kind='computed')


@fig
def ml5_masking():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(50, 2, 120), rng.normal(65, .8, 25), rng.normal(50, 2, 80)]
    bad = np.zeros(x.size, bool)
    bad[120:145] = True
    mu, sd = x.mean(), x.std(ddof=1)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    rsd = 1.4826 * mad
    z = (x - mu) / sd
    rz = (x - med) / rsd
    f, axs = half()
    ax = axs[0]
    idx = np.arange(x.size)
    ax.axhspan(med - 3.5 * rsd, med + 3.5 * rsd, color=TEAL_T, zorder=0)
    for k in (-1, 1):
        ax.axhline(mu + 3 * k * sd, color=GREY, ls='--', lw=1.4)
        ax.axhline(med + 3.5 * k * rsd, color=TEAL, ls='--', lw=1.4)
    ax.scatter(idx[~bad], x[~bad], s=16, c=BLUE, edgecolors='white', linewidths=.3, zorder=3)
    ax.scatter(idx[bad], x[bad], s=26, c=ORANGE, edgecolors='white', linewidths=.3, zorder=3)
    n_z = int((np.abs(z[bad]) > 3).sum())
    n_r = int((np.abs(rz[bad]) > 3.5).sum())
    n_rf = int((np.abs(rz[~bad]) > 3.5).sum())
    ax.text(229, mu + 3 * sd + .5, f'mean ± 3 SD\nflags {n_z} of 25', fontsize=12, color=GREY, va='bottom')
    ax.text(229, med + 3.5 * rsd - .6, f'median ± 3.5\nrobust SD\nflags {n_r} of 25', fontsize=12,
            color=TEAL, va='top')
    ax.text(132, 67.8, 'offset\nreadings', fontsize=12, color=ORANGE, ha='center', va='bottom')
    ax.set_xlim(-5, 300)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(33, 72)
    clean(ax, 'Reading index', 'Pressure reading (bar)')
    save('ml5_masking', f, f'Synthetic readings (seed 3): 200 ordinary N(50, 2) plus 25 offset N(65, 0.8). '
         f'Mean {mu:.2f}, SD {sd:.2f}; median {med:.2f}, MAD {mad:.2f}; ordinary rows flagged by the robust '
         f'rule: {n_rf}.', kind='computed')


def _gr_sample():
    rng = np.random.default_rng(21)
    sand = rng.normal(38, 9, 900)
    shale = rng.normal(102, 9, 100)
    spikes = np.array([168., 174., 181.])
    return np.r_[sand, shale, spikes], np.r_[np.zeros(900), np.ones(100), 2 * np.ones(3)]


@fig
def ml5_fences():
    x, kind = _gr_sample()
    mu, sd = x.mean(), x.std(ddof=1)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    med = np.median(x)
    rsd = 1.4826 * np.median(np.abs(x - med))
    rules = [('mean + 3 SD', mu + 3 * sd, GREY), ('Tukey upper fence', q3 + 1.5 * iqr, PURPLE),
             ('median + 3.5 robust SD', med + 3.5 * rsd, TEAL)]
    f, axs = half()
    ax = axs[0]
    bins = np.arange(0, 190, 5)
    ax.hist(x[kind == 0], bins=bins, color=BLUE, alpha=.75, label='clean sand')
    ax.hist(x[kind == 1], bins=bins, color=GOLD, alpha=.6)
    ax.hist(x[kind == 2], bins=bins, color=ORANGE)
    counts, _ = np.histogram(x, bins)
    ytop = counts.max() * 1.12
    for v in x[kind == 2]:                               # one row each: mark the spikes so they stay visible
        ax.plot([v, v], [0, ytop * .035], color=ORANGE, lw=2.4, solid_capstyle='butt', zorder=4)
    for k, (name, v, c) in enumerate(rules):
        n_flag = int((x > v).sum())
        n_shale = int(((x > v) & (kind == 1)).sum())
        ax.axvline(v, color=c, ls='--', lw=1.8)
        ax.text(118, ytop * (.97 - .2 * k), f'{name}\n{n_flag} flagged ({n_shale} shale)', fontsize=12,
                color=c, va='top')
    ax.text(33, 12, 'sand\n(900)', fontsize=12, color='white', ha='center', va='bottom', fontweight='medium')
    ax.text(102, 22, 'shale\n(100)', fontsize=12, color=GOLD, ha='center', va='bottom')
    ax.text(174, 10, 'tool\nspikes (3)', fontsize=12, color=ORANGE, ha='center', va='bottom')
    ax.set_xlim(0, 190)
    ax.set_ylim(0, ytop)
    clean(ax, 'Gamma ray (API)', 'Rows')
    save('ml5_fences', f, 'Synthetic GR-like mixture (seed 21): 900 sand rows N(38, 9), 100 shale rows N(102, 9), '
         'three tool spikes. Fence counts computed on the full sample.', kind='computed')


@fig
def ml5_mahalanobis():
    rng = np.random.default_rng(4)
    cov = np.array([[1, .9], [.9, 1]])
    X = rng.multivariate_normal([0, 0], cov, 300)
    p = np.array([1.5, -1.3])
    ec = EmpiricalCovariance().fit(X)
    d_p = float(np.sqrt(ec.mahalanobis(p[None])[0]))
    e_p = float(np.linalg.norm(p - ec.location_))
    thr = float(np.sqrt(stats.chi2.ppf(.99, 2)))
    f, axs = half()
    ax = axs[0]
    m, s = X.mean(0), X.std(0, ddof=1)
    ax.add_patch(Rectangle((m[0] - 3 * s[0], m[1] - 3 * s[1]), 6 * s[0], 6 * s[1], fc='none', ec=GREY,
                           ls='--', lw=1.5, zorder=1))
    xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
    dd = np.sqrt(ec.mahalanobis(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, dd, levels=[0, thr], colors=[TEAL_T], zorder=0)
    ax.contour(xx, yy, dd, levels=[thr], colors=[TEAL], linewidths=1.8, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=14, c=BLUE, edgecolors='white', linewidths=.3, zorder=2)
    ax.scatter(*p, s=120, c=ORANGE, edgecolors='white', zorder=4)
    ax.annotate(f'Euclidean {e_p:.1f}\nMahalanobis {d_p:.1f}', p, (.45, -2.45), fontsize=12,
                color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    ax.text(-3.85, 3.75, '±3 SD per feature', fontsize=12, color=GREY, va='top')
    ax.text(-1.2, 2.25, f'Mahalanobis\nd = {thr:.2f}', fontsize=12, color=TEAL, ha='center')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    clean(ax, 'Flow rate (standardized)', 'Motor current (standardized)')
    save('ml5_mahalanobis', f, f'Synthetic correlated features (seed 4, correlation 0.9, 300 rows). Contour at the '
         f'chi-square(2) 99% radius {thr:.2f}; test point distance {d_p:.2f}.', kind='computed')


# ====================================================================== part 2
def _iforest_data():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(0, 1, (60, 2)), [[3.0, 2.7]]]
    out = len(X) - 1
    inl = int(np.argmin(np.linalg.norm(X[:-1] - np.array([.15, -.1]), axis=1)))
    return X, out, inl


_BOUNDS = (np.array([-3.4, -3.4]), np.array([3.6, 3.4]))


def _isolate(X, t, rng):
    """Grow one random isolation path for row t; return the cuts and the final cell."""
    idx = np.arange(len(X))
    lo, hi = _BOUNDS[0].copy(), _BOUNDS[1].copy()
    cuts = []
    while len(idx) > 1:
        f = int(rng.integers(2))
        v = X[idx, f]
        a, b = v.min(), v.max()
        if a == b:
            break
        s = rng.uniform(a, b)
        cuts.append((f, s, lo.copy(), hi.copy()))
        if X[t, f] < s:
            idx, hi[f] = idx[X[idx, f] < s], s
        else:
            idx, lo[f] = idx[X[idx, f] >= s], s
    return cuts, lo, hi


def _c(n):
    """Average path length of an unsuccessful binary-search-tree search (Liu et al. 2008)."""
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


@fig
def ml5_iforest_splits():
    X, out, inl = _iforest_data()
    f, axs = sub(3)

    def draw(ax, t, cuts, lo, hi, col, first, last, win, fs=11.5):
        (x0, x1), (y0, y1) = win
        ax.add_patch(Rectangle(lo, *(hi - lo), fc=tint(col, .72), ec='none', zorder=0))
        ax.scatter(X[:, 0], X[:, 1], s=22, c='#8C99A6', edgecolors='white', linewidths=.4, zorder=2)
        ax.scatter(*X[t], s=130, c=col, edgecolors='white', linewidths=1, zorder=5)
        placed = [X[t]]
        size = max(x1 - x0, y1 - y0)
        for k, (fe, sv, clo, chi) in enumerate(cuts, 1):
            if fe == 0:
                seg = np.array([[sv, max(clo[1], y0)], [sv, min(chi[1], y1)]])
            else:
                seg = np.array([[max(clo[0], x0), sv], [min(chi[0], x1), sv]])
            ax.plot(*np.array([[sv, clo[1]], [sv, chi[1]]]).T if fe == 0 else np.array([[clo[0], sv], [chi[0], sv]]).T,
                    color=INK, lw=1.5, zorder=3)
            if not first <= k <= last:
                continue
            cands = [seg[0] + a * (seg[1] - seg[0]) for a in (.5, .3, .7, .15, .85)]
            best = max(cands, key=lambda c: min(np.linalg.norm(c - q) for q in placed))
            if min(np.linalg.norm(best - q) for q in placed) < .07 * size:
                continue
            placed.append(best)
            ax.text(*best, str(k), fontsize=fs, ha='center', va='center', color=INK, zorder=4,
                    fontweight='semibold', bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec=INK, lw=.9))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    full = ((_BOUNDS[0][0], _BOUNDS[1][0]), (_BOUNDS[0][1], _BOUNDS[1][1]))
    zoom = ((-.15, 1.40), (-.52, .98))
    cuts, lo, hi = _isolate(X, out, np.random.default_rng(2))
    draw(axs[0], out, cuts, lo, hi, ORANGE, 1, 99, full)
    axs[0].set_title(f'Unusual row: {len(cuts)} cuts')
    cuts, lo, hi = _isolate(X, inl, np.random.default_rng(1))
    draw(axs[1], inl, cuts, lo, hi, BLUE, 1, 4, full)
    axs[1].set_title(f'Ordinary row: {len(cuts)} cuts')
    (zx0, zx1), (zy0, zy1) = zoom
    axs[1].add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc='none', ec=TEAL, ls='--', lw=1.8, zorder=6))
    draw(axs[2], inl, cuts, lo, hi, BLUE, 5, 99, zoom, fs=12)
    axs[2].set_title('Zoom on the dashed box: cuts 5–10')
    for sp in axs[2].spines.values():
        sp.set_visible(True)
        sp.set_color(TEAL)
        sp.set_linestyle('--')
        sp.set_linewidth(1.6)
    for ax in axs:
        ax.set_aspect('equal')
        _hide_ticks(ax)
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    save('ml5_iforest_splits', f, 'Synthetic 2-D sample (seed 2): 60 ordinary rows plus one unusual row. One random '
         'isolation path per row (path seeds 2 and 1); cuts numbered in order, the right panel zooms on the dashed '
         'box.', kind='computed')


@fig
def ml5_iforest_paths():
    X, out, inl = _iforest_data()
    n = len(X)
    L = {t: np.array([len(_isolate(X, t, np.random.default_rng(k))[0]) for k in range(1000)]) for t in (out, inl)}
    cn = _c(n)
    f, axs = half()
    ax = axs[0]
    ks = np.arange(1, 21)
    cnt = {t: np.bincount(L[t], minlength=21)[1:21] for t in (out, inl)}
    ax.bar(ks - .21, cnt[out], .42, color=ORANGE)       # side by side: no blended overlap colour
    ax.bar(ks + .21, cnt[inl], .42, color=BLUE)
    top = max(cnt[out].max(), cnt[inl].max())
    ax.axvline(cn, color=TEAL, ls='--', lw=1.6)
    ax.text(cn + .3, top * 1.32, f'c(n) = {cn:.2f}\n(n = {n})', color=TEAL, fontsize=12, va='top')
    for t, col, xt, yt in [(out, ORANGE, .8, 1.32), (inl, BLUE, 12.2, .78)]:
        m = L[t].mean()
        sc = 2 ** (-m / cn)
        ax.text(xt, top * yt, f'mean path {m:.2f}\nscore s = {sc:.2f}', color=col, fontsize=12.5, va='top',
                fontweight='medium')
    ax.set_xlim(.5, 20.5)
    ax.set_ylim(0, top * 1.36)
    ax.set_xticks([1, 5, 10, 15, 20])
    clean(ax, 'Cuts needed to isolate the row (path length)', 'Random trees (of 1,000)')
    save('ml5_iforest_paths', f, 'Path lengths of the two highlighted rows over 1,000 fully grown random isolation '
         'paths (seeds 0-999) on the 61-row synthetic sample; s = 2^(-mean path / c(n)).', kind='computed')


@fig
def ml5_lof_mechanism():
    rng = np.random.default_rng(8)
    dense = rng.normal([0, 0], .2, (25, 2))
    sparse = rng.normal([3.9, .2], .8, (25, 2))
    A = np.array([.95, .8])
    B = np.array([5.6, 1.7])
    X = np.r_[dense, sparse, [A], [B]]
    k = 6
    lof = LocalOutlierFactor(n_neighbors=k).fit(X)
    scores = -lof.negative_outlier_factor_
    ia, ib = len(X) - 2, len(X) - 1
    dist, nn = lof.kneighbors()          # training rows; each row's own index excluded
    kd = dist[:, -1]
    f, axs = half()
    ax = axs[0]
    ax.scatter(dense[:, 0], dense[:, 1], s=20, c=BLUE, edgecolors='white', linewidths=.3, zorder=2)
    ax.scatter(sparse[:, 0], sparse[:, 1], s=20, c=BLUE, edgecolors='white', linewidths=.3, zorder=2)
    for i, col in [(ia, ORANGE), (ib, PURPLE)]:
        ax.add_patch(Circle(X[i], kd[i], fc='none', ec=col, ls='--', lw=1.6, zorder=3))
        j = nn[i][0]
        ax.add_patch(Circle(X[j], kd[j], fc='none', ec=GREY, lw=1.1, zorder=3))
    ax.scatter(*A, s=110, c=ORANGE, edgecolors='white', zorder=5)
    ax.scatter(*B, s=110, c=PURPLE, edgecolors='white', zorder=5)
    ax.annotate(f'A: LOF = {scores[ia]:.1f}\nk-distance {kd[ia]:.2f}', A, (-.9, 2.35), fontsize=12.5, color=ORANGE,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ax.annotate(f'B: LOF = {scores[ib]:.1f}\nk-distance {kd[ib]:.2f}', B, (3.6, 3.35), fontsize=12.5, color=PURPLE,
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.1))
    ax.text(0, -.75, 'dense\ncluster', ha='center', va='top', fontsize=12, color=BLUE)
    ax.text(3.9, -1.75, 'sparse cluster', ha='center', va='top', fontsize=12, color=BLUE)
    ax.set_xlim(-1.3, 7.5)
    ax.set_ylim(-2.4, 4.1)
    ax.set_aspect('equal')
    clean(ax, 'Feature 1', 'Feature 2')
    save('ml5_lof_mechanism', f, f'Synthetic clusters (seed 8); scikit-learn LocalOutlierFactor with n_neighbors={k}. '
         'Dashed circles: k-distance of A and B; grey circles: k-distance of one neighbour of each.', kind='computed')


def _micro_data():
    rng = np.random.default_rng(12)
    main = rng.normal(0, 1, (300, 2))
    micro = rng.normal([3.6, 3.6], .1, (8, 2))
    lone = np.array([[-3.8, 2.9], [3.9, -3.2], [-3.3, -3.6]])
    X = np.r_[main, micro, lone]
    grp = np.r_[np.zeros(300), np.ones(8), 2 * np.ones(3)].astype(int)
    return X, grp


@fig
def ml5_lof_k():
    X, grp = _micro_data()
    f, axs = sub(3)
    for ax, k in zip(axs, (5, 20, 50)):
        s = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
        top = np.argsort(-s)[:11]
        ax.scatter(X[:, 0], X[:, 1], s=14, c='#8C99A6', edgecolors='white', linewidths=.3, zorder=2)
        ax.scatter(X[grp == 1, 0], X[grp == 1, 1], s=22, c=GOLD, edgecolors='white', linewidths=.3, zorder=3)
        ax.scatter(X[top, 0], X[top, 1], s=120, facecolors='none', edgecolors=ORANGE, linewidths=1.8, zorder=4)
        n_micro = int(np.isin(np.where(grp == 1)[0], top).sum())
        n_lone = int(np.isin(np.where(grp == 2)[0], top).sum())
        ax.set_title(f'n_neighbors = {k}')
        ax.set_xlabel(f'micro-cluster flagged: {n_micro} of 8\nisolated rows flagged: {n_lone} of 3', fontsize=13)
        ax.set_xlim(-4.6, 4.8)
        ax.set_ylim(-4.4, 4.6)
        ax.set_aspect('equal')
        _hide_ticks(ax)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
    axs[0].annotate('8-row micro-cluster', (3.5, 3.72), (-.4, 4.05), fontsize=12, color=GOLD, va='center',
                    arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=1.1))
    save('ml5_lof_k', f, 'Synthetic data (seed 12): 300 ordinary rows, an 8-row micro-cluster, 3 isolated rows. '
         'Rings mark the 11 highest LOF scores at each n_neighbors.', kind='computed')


@fig
def ml5_ocsvm_gamma():
    def arc(rng, n):
        t = rng.uniform(.15 * np.pi, 1.05 * np.pi, n)
        return np.c_[np.cos(t) * 2.2, np.sin(t) * 1.6] + rng.normal(0, .28, (n, 2))
    sc = StandardScaler().fit(arc(np.random.default_rng(6), 260))
    X = sc.transform(arc(np.random.default_rng(6), 260))
    Xnew = sc.transform(arc(np.random.default_rng(60), 2000))
    probe = sc.transform([[0.35, 0.15]])
    xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
    G = np.c_[xx.ravel(), yy.ravel()]
    f, axs = sub(3)
    for ax, g in zip(axs, (.05, .5, 5)):
        m = OneClassSVM(nu=.05, gamma=g).fit(X)
        d = m.decision_function(G).reshape(xx.shape)
        dx = m.decision_function(X)
        out = dx < -1e-3                      # strictly outside; boundary support vectors have d close to 0
        ax.contourf(xx, yy, d, levels=[0, d.max() + 1], colors=[BLUE_T], zorder=0)
        ax.contour(xx, yy, d, levels=[0], colors=[BLUE], linewidths=1.8, zorder=1)
        ax.scatter(X[~out, 0], X[~out, 1], s=13, c=BLUE, edgecolors='white', linewidths=.3, zorder=2)
        ax.scatter(X[out, 0], X[out, 1], s=30, c=ORANGE, edgecolors='white', linewidths=.4, zorder=3)
        flagged = m.decision_function(probe)[0] < 0
        ax.scatter(*probe[0], s=120, marker='X', c=ORANGE if flagged else INK, zorder=5)
        new_rate = 100 * (m.decision_function(Xnew) < 0).mean()
        ax.set_title(f'gamma = {g:g}')
        ax.set_xlabel(f'new ordinary rows flagged: {new_rate:.1f}%\ninterior probe (X) flagged: '
                      f'{"yes" if flagged else "no"}', fontsize=13)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect('equal')
        _hide_ticks(ax)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
    save('ml5_ocsvm_gamma', f, 'Synthetic arc-shaped reference data (seed 6, 260 rows, standardized); scikit-learn '
         'OneClassSVM, RBF kernel, nu = 0.05. Shaded: decision function >= 0. Flag rates on 2,000 new rows from the '
         'same generator (seed 60).', kind='computed')


@fig
def ml5_anomaly_methods():
    """Module copy of the shared anomaly_methods figure: ordinary reference rows in BLUE, scores on a grey scale."""
    rng = np.random.default_rng(42)
    X = np.r_[rng.normal([-1, 0], [.4, .65], (160, 2)), rng.normal([1.3, .5], [.22, .3], (80, 2))]
    xx, yy = np.meshgrid(np.linspace(-3, 3, 170), np.linspace(-2.6, 2.6, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    models = [IsolationForest(random_state=42), LocalOutlierFactor(n_neighbors=20, novelty=True),
              OneClassSVM(nu=.08, gamma=.8)]
    cmap = LinearSegmentedColormap.from_list('ml5_score', ['#FFFFFF', '#E3E8EE', '#AAB6C2', '#56687A'])
    f, axs = sub(3, height=4.2)
    for ax, m, title in zip(axs, models, ['Isolation Forest', 'Novelty LOF', 'One-Class SVM']):
        m.fit(X)
        score = -m.score_samples(grid)                   # higher = more unusual
        score = np.clip(score, *np.quantile(score, [.02, .98]))
        ax.contourf(xx, yy, score.reshape(xx.shape), levels=12, cmap=cmap)
        ax.scatter(X[:, 0], X[:, 1], s=14, c=BLUE, edgecolors='white', linewidths=.5, zorder=3)
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2' if ax is axs[0] else '')
    f.suptitle('Darker background = more unusual under each model; blue points: the reference rows', fontsize=13,
               color=MUTED)
    save('ml5_anomaly_methods', f, 'Synthetic two-cluster reference set (seed 42); IsolationForest(random_state=42), '
         'LocalOutlierFactor(n_neighbors=20, novelty=True), OneClassSVM(nu=0.08, gamma=0.8); shading = '
         '-score_samples clipped to its 2nd-98th percentiles.', kind='computed')


def _ar1(rng, n, phi, s):
    e = rng.normal(0, s, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _six_sensors():
    """Six standardized channels driven by two latent factors; channel 4 drifts from hour 520."""
    rng = np.random.default_rng(32)
    T = 720
    t = np.arange(T)
    speed = .9 * np.sin(2 * np.pi * t / 24) + _ar1(rng, T, .9, .15)
    load = .7 * np.sin(2 * np.pi * t / 168 + 1) + _ar1(rng, T, .95, .1)
    A = np.array([[1, .2], [.9, .5], [.8, -.3], [.7, .6], [.3, 1], [.5, .8]])
    X = np.c_[speed, load] @ A.T + rng.normal(0, .08, (T, 6))
    mu, sd = X[:300].mean(0), X[:300].std(0)
    Xs = (X - mu) / sd
    Xs[:, 3] += np.where(t >= 520, .008 * (t - 520), 0)
    return t, Xs


@fig
def ml5_pca_recon():
    f, axs = sub(2)
    # left: geometry with two correlated sensors
    rng = np.random.default_rng(13)
    Z = rng.multivariate_normal([0, 0], [[1, .95], [.95, 1]], 200)
    pca1 = PCA(1).fit(Z)
    v = pca1.components_[0]
    ax = axs[0]
    ax.plot([-3.2 * v[0], 3.2 * v[0]], [-3.2 * v[1], 3.2 * v[1]], color=TEAL, lw=2, zorder=1)
    ax.scatter(Z[:, 0], Z[:, 1], s=14, c=BLUE, edgecolors='white', linewidths=.3, zorder=2)
    for pnt, col, lab, xy in [(np.array([-1.2, -1.05]), BLUE, 'ordinary', (-3.0, .35)),
                              (np.array([1.25, -.95]), ORANGE, 'unusual', (1.1, -2.9))]:
        proj = pca1.inverse_transform(pca1.transform(pnt[None]))[0]
        err = float(((pnt - proj) ** 2).sum())
        ax.plot([pnt[0], proj[0]], [pnt[1], proj[1]], color=col, ls='--', lw=1.8, zorder=3)
        ax.scatter(*pnt, s=110, c=col, edgecolors='white', zorder=5)
        ax.scatter(*proj, s=40, c='white', edgecolors=col, linewidths=1.5, zorder=4)
        ax.annotate(f'{lab}: error {err:.2f}', pnt, xy, fontsize=12.5, color=col,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1))
    on_line = (1.2, 1.2 * v[1] / v[0])
    ax.annotate('first principal\ndirection', on_line, (-3.0, 2.3), fontsize=12, color=TEAL, va='center',
                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1.1))
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.set_title('Error = squared distance to the subspace', fontsize=14)
    clean(ax, 'Pump speed (standardized)', 'Discharge pressure (standardized)')
    # right: six channels, PCA(2) fitted on hours 0-299, threshold from hours 300-419
    t, X = _six_sensors()
    pca = PCA(2).fit(X[:300])
    spe = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(1)
    thr = np.percentile(spe[300:420], 99)
    over = spe > thr
    sustained = next(i for i in range(520, 718) if over[i] and over[i + 1] and over[i + 2])
    uni = int(np.argmax((np.abs(X[:, 3]) > 3) & (t >= 520)))
    ax = axs[1]
    m = t >= 300
    ax.axvspan(520, 720, color=ORANGE, alpha=.07, lw=0)
    ax.semilogy(t[m], spe[m], color=BLUE, lw=1.3)
    ax.axhline(thr, color=TEAL, ls='--', lw=1.6)
    ax.text(305, 9, 'dashed: threshold = 99th\npercentile of hours 300–419', color=TEAL,
            fontsize=12, va='top')
    ax.axvline(520, color=INK, ls=':', lw=1.4)
    ax.text(524, 2.4, 'drift\nstarts', fontsize=12, va='top', ha='left')
    ax.axvline(uni, color=GREY, ls=':', lw=1.4)
    ax.text(uni - 4, 9, f'sensor 4 alone:\n|z| > 3 at +{uni - 520} h', fontsize=12, va='top', ha='right',
            color=GREY)
    ax.annotate(f'sustained alarm\n+{sustained - 520} h', (sustained, spe[sustained]), (585, .012), fontsize=12,
                color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ax.set_xlim(300, 720)
    ax.set_ylim(.005, 15)
    ax.set_title('Six sensors, two components', fontsize=14)
    clean(ax, 'Hour', 'Reconstruction error (log)')
    save('ml5_pca_recon', f, f'Left: synthetic correlated pair (seed 13), one principal component. Right: six synthetic '
         f'channels from two latent factors (seed 32); PCA fitted on hours 0–299; threshold {thr:.3f} = 99th '
         f'percentile of hours 300–419; sensor 4 drifts +0.008 SD per hour from hour 520.', kind='computed')


@fig
def ml5_autoencoder():
    f, ax = canvas(6.3, 4.9)
    pos = network(ax, [6, 4, 2, 4, 6], .75, 5.55, 1.45, 4.05, r=.16,
                  colors=[GREY, BLUE, PURPLE, BLUE, ORANGE], fills=['tint', 'tint', 'solid', 'tint', 'tint'])
    for x, lines in zip(np.linspace(.75, 5.55, 5), [('Input $x$', '6 sensors'), ('Encoder',), ('Code $z$', '2 values'),
                                                     ('Decoder',), ('Output $\\hat{x}$', '6 values')]):
        for y, txt in zip((.96, .75), lines):            # fixed baselines: the hat does not shift the line
            label(ax, x, y, txt, fs=12.5, color=INK, va='baseline')
    bracket(ax, .55, 3.15, 4.35, 'compress', color=BLUE, below=False, fs=12.5)
    bracket(ax, 3.15, 5.75, 4.35, 'reconstruct', color=ORANGE, below=False, fs=12.5)
    b = box(ax, 1.55, .06, 3.2, .52, r'Anomaly score = $\Vert x-\hat{x}\Vert^2$', ORANGE, fill='white', fs=13.5)
    elbow(ax, (.75, .58), b.l, color=MUTED, lw=1.3, first='v', head=12)
    elbow(ax, (5.55, .58), b.r, color=MUTED, lw=1.3, first='v', head=12)
    save('ml5_autoencoder', f, 'Schematic undercomplete autoencoder, 6-4-2-4-6 units.', kind='schematic')


def _load_series():
    """Synthetic hourly feeder load (MW); hours 395-399 (about day 16.5 on the axis) set to night level."""
    rng = np.random.default_rng(17)
    h = np.arange(21 * 24)
    x = 50 + 8 * np.sin(2 * np.pi * (h - 8) / 24) + .004 * h + _ar1(rng, h.size, .6, .8)
    anom = np.arange(16 * 24 + 11, 16 * 24 + 16)
    x[anom] = 42.5 + rng.normal(0, .6, anom.size)
    return h, x, anom


@fig
def ml5_ts_rolling():
    h, x, anom = _load_series()
    ref = h < 14 * 24
    mu, sd = x[ref].mean(), x[ref].std()
    s = pd.Series(x)
    rz = ((s - s.rolling(24).mean().shift(1)) / s.rolling(24).std().shift(1)).to_numpy()
    d = h / 24
    show = d >= 7
    f, axs = grid_axes(2, 1, height=FULL[1])
    ax, az = axs[0, 0], axs[1, 0]
    for a in (ax, az):
        a.axvspan(anom[0] / 24, (anom[-1] + 1) / 24, color=ORANGE, alpha=.18, lw=0)
    ax.axhspan(mu - 3 * sd, mu + 3 * sd, color=PANEL, zorder=0)
    ax.axhline(mu + 3 * sd, color=GREY, ls='--', lw=1.3)
    ax.axhline(mu - 3 * sd, color=GREY, ls='--', lw=1.3)
    ax.plot(d[show], x[show], color=BLUE, lw=1.6)
    ax.text(7.1, mu + 3 * sd + .8, 'global mean ± 3 SD (first 14 days)', color=GREY, fontsize=12, va='bottom')
    ax.annotate('afternoon load at\nnight level', (anom[2] / 24, x[anom[2]]), (17.6, 38), fontsize=12,
                color=ORANGE, va='center', arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ax.set_ylim(28, 76)
    ax.set_ylabel('Load (MW)')
    az.axhline(3, color=TEAL, ls='--', lw=1.3)
    az.axhline(-3, color=TEAL, ls='--', lw=1.3)
    az.plot(d[show], rz[show], color=BLUE, lw=1.4)
    az.text(7.1, 3.3, '±3', color=TEAL, fontsize=12, va='bottom')
    az.set_ylim(-4.5, 4.5)
    az.set_ylabel('Trailing z')
    for a in (ax, az):
        a.set_xlim(7, 21)
        a.grid(alpha=1)
        a.set_axisbelow(True)
    ax.tick_params(labelbottom=False)
    az.set_xlabel('Day')
    save('ml5_ts_rolling', f, f'Synthetic hourly load (seed 17): daily cycle, slow trend, AR(1) noise; five afternoon '
         f'hours near day 16.5 set to night level. Minimum trailing 24-h z-score in the anomaly: '
         f'{np.nanmin(rz[anom]):.2f}.', kind='computed')


@fig
def ml5_ts_residual():
    h, x, anom = _load_series()
    X = x.reshape(21, 24)
    fc = np.full_like(X, np.nan)
    for day in range(7, 21):
        fc[day] = np.median(X[day - 7:day], axis=0)
    fc = fc.ravel()
    r = x - fc
    ref = (h >= 7 * 24) & (h < 14 * 24)
    med = np.nanmedian(r[ref])
    rsd = 1.4826 * np.nanmedian(np.abs(r[ref] - med))
    rz = (r - med) / rsd
    flag = np.abs(rz) > 3.5
    d = h / 24
    show = d >= 7
    f, axs = grid_axes(2, 1, height=FULL[1])
    ax, az = axs[0, 0], axs[1, 0]
    for a in (ax, az):
        a.axvspan(anom[0] / 24, (anom[-1] + 1) / 24, color=ORANGE, alpha=.18, lw=0)
    ax.plot(d[show], x[show], color=BLUE, lw=1.6)
    ax.plot(d[show], fc[show], color=PURPLE, lw=1.4, ls='--')
    ax.text(7.1, 70.5, 'actual', color=BLUE, fontsize=12, va='center')
    ax.text(8.1, 70.5, 'forecast: median of the same hour over the previous 7 days', color=PURPLE, fontsize=12,
            va='center')
    ax.set_ylim(34, 74)
    ax.set_ylabel('Load (MW)')
    az.axhline(3.5, color=TEAL, ls='--', lw=1.3)
    az.axhline(-3.5, color=TEAL, ls='--', lw=1.3)
    az.plot(d[show], rz[show], color=BLUE, lw=1.4)
    az.scatter(d[flag & show], rz[flag & show], s=36, c=ORANGE, zorder=4, edgecolors='white')
    az.text(7.1, 4.0, '±3.5 robust SD', color=TEAL, fontsize=12, va='bottom')
    n_in = int(np.isin(anom, np.where(flag)[0]).sum())
    false = np.where(flag & show & ~np.isin(h, anom))[0]
    az.annotate(f'{n_in} of {anom.size} anomaly hours flagged', (anom[-1] / 24, rz[anom[-1]]), (17.4, -13),
                fontsize=12, color=ORANGE, va='center', arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    if false.size:
        az.annotate(f'{false.size} ordinary hours also cross\n(noise excursion: false alarms)', (false[-1] / 24, rz[false[-1]]),
                    (11.2, -9.5), fontsize=12, color=GREY, va='center',
                    arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.1))
    az.set_ylim(-21, 8)
    az.set_ylabel('Residual z')
    for a in (ax, az):
        a.set_xlim(7, 21)
        a.grid(alpha=1)
        a.set_axisbelow(True)
    ax.tick_params(labelbottom=False)
    az.set_xlabel('Day')
    save('ml5_ts_residual', f, 'Same synthetic load (seed 17). Seasonal forecast = median of the same hour over the '
         'previous 7 days; residual scaled by median and MAD of days 7–14; threshold 3.5.', kind='computed')


# ====================================================================== part 3
@fig
def ml5_prevalence_tree():
    f, ax = canvas(6.3, 4.9)
    w, h, g = 1.38, .82, .13
    xs = [.16 + i * (w + g) for i in range(4)]
    root = box(ax, 2.2, 4.1, 1.9, .62, '10,000 rows', GREY, fs=14, weight='semibold')
    ev = box(ax, (xs[0] + xs[1] + w) / 2 - .95, 2.78, 1.9, .66, '100 events', ORANGE, fs=14, weight='semibold')
    od = box(ax, (xs[2] + xs[3] + w) / 2 - .95, 2.78, 1.9, .66, '9,900 ordinary', BLUE, fs=14, weight='semibold')
    arrow(ax, root.b, ev.t, color=MUTED, lw=1.4, shrink=1)
    arrow(ax, root.b, od.t, color=MUTED, lw=1.4, shrink=1)
    leaves = [('20', 'missed (FN)', ORANGE, 'white', 1.3), ('80', 'true alerts (TP)', ORANGE, 'tint', 2.6),
              ('495', 'false alerts (FP)', BLUE, 'tint', 2.6), ('9,405', 'quiet (TN)', BLUE, 'white', 1.3)]
    anchors = []
    for x, (n, lab, col, fill, lw) in zip(xs, leaves):
        a = box(ax, x, 1.32, w, h, n, col, fill=fill, fs=17, weight='semibold', sub=lab, sub_fs=11.5, lw=lw)
        anchors.append(a)
    for parent, kids, labs in [(ev, anchors[:2], ['20%', 'recall 80%']), (od, anchors[2:], ['FPR 5%', '95%'])]:
        for kid, lab in zip(kids, labs):
            arrow(ax, parent.b, kid.t, color=MUTED, lw=1.4, shrink=1)
            mx, my = (parent.b[0] + kid.t[0]) / 2, (parent.b[1] + kid.t[1]) / 2
            label(ax, mx + (.28 if kid.t[0] > parent.b[0] else -.28), my + .05, lab, fs=12,
                  color=MUTED, ha='left' if kid.t[0] > parent.b[0] else 'right')
    bracket(ax, anchors[1].x, anchors[2].x + w, 1.18, '', color=INK, below=True, lw=1.6)
    label(ax, 3.15, .72, 'Alerts: 80 + 495 = 575', fs=14, weight='semibold')
    label(ax, 3.15, .3, 'Precision = 80 / 575 = 13.9%', fs=14, weight='semibold', color=ORANGE)
    save('ml5_prevalence_tree', f, 'Exact arithmetic for 10,000 rows, 1% prevalence, recall 80%, false-positive '
         'rate 5%.', kind='computed')


@fig
def ml5_roc_pr_prevalence():
    t = np.linspace(-6, 8, 4000)
    d = 1.5
    tpr = stats.norm.sf(t - d)
    fpr = stats.norm.sf(t)
    auc = stats.norm.cdf(d / np.sqrt(2))
    t5 = stats.norm.isf(.05)
    tpr5 = stats.norm.sf(t5 - d)
    f, axs = sub(2)
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1.3, ls='--')
    ax.plot(fpr, tpr, color=INK, lw=2.4)
    ax.scatter([.05], [tpr5], s=60, c=INK, zorder=4)
    ax.annotate(f'FPR 5%, recall {tpr5:.0%}', (.05, tpr5), (.5, .38), fontsize=12.5, va='center',
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    ax.text(.97, .05, f'one curve for every prevalence\nAUC = {auc:.2f}', fontsize=13, color=INK, ha='right',
            va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect('equal')
    ax.set_title('ROC: prevalence cancels out')
    clean(ax, 'False-positive rate', 'Recall (true-positive rate)')
    ax = axs[1]
    for pi, col, ly in [(.5, BLUE, None), (.1, TEAL, None), (.01, ORANGE, None)]:
        prec = pi * tpr / (pi * tpr + (1 - pi) * fpr)
        ap = float(np.sum((prec[1:] + prec[:-1]) / 2 * -np.diff(tpr)))
        ax.plot(tpr, prec, color=col, lw=2.4)
        ax.axhline(pi, color=col, lw=1, ls=':')
        p5 = pi * tpr5 / (pi * tpr5 + (1 - pi) * .05)
        ax.scatter([tpr5], [p5], s=55, c=col, zorder=4, edgecolors='white')
        ax.text(tpr5 + .03, p5 + .025, f'{p5:.2f}', color=col, fontsize=12.5, va='bottom', fontweight='semibold')
        yl = {.5: .37, .1: .30, .01: .23}[pi]
        ax.text(.2, yl, f'{pi:.0%}: AP {ap:.2f}', color=col, fontsize=12.5, va='center', fontweight='medium')
    ax.text(.2, .44, 'prevalence: AP', color=INK, fontsize=12.5, va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title('Precision-recall: precision collapses')
    clean(ax, 'Recall', 'Precision')
    save('ml5_roc_pr_prevalence', f, 'Exact binormal scores: ordinary N(0, 1), events N(1.5, 1). Dots mark the '
         'threshold with FPR 5% (recall 0.44) and its precision; dotted lines show each prevalence.', kind='computed')


def _event_timeline():
    rng = np.random.default_rng(22)
    T = 720
    score = _ar1(rng, T, .8, .25)
    events, dur, W = [190, 420, 610], 6, 48
    for e, L, a in [(190, 40, 3.2), (420, 12, 2.6)]:
        score[e - L:e] += a * np.linspace(0, 1, L)
    for st, L, a in [(78, 5, 2.3), (300, 4, 2.2), (505, 6, 2.4), (655, 3, 2.1)]:
        score[st:st + L] += a * np.sin(np.linspace(0, np.pi, L + 2)[1:-1])
    for e in events:
        score[e:e + dur] = np.nan
    thr = 1.5
    rows = np.where(np.nan_to_num(score) > thr)[0]
    eps = []
    for r in rows:
        if eps and r - eps[-1][1] <= 6:
            eps[-1][1] = r
        else:
            eps.append([r, r])
    win = np.zeros(T, bool)
    for e in events:
        win[e - W:e] = True
    return dict(score=score, events=events, dur=dur, W=W, thr=thr, rows=rows, eps=eps, win=win)


@fig
def ml5_event_timeline():
    tl = _event_timeline()
    d = np.arange(720) / 24
    f = plt.figure(figsize=FULL, layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[2.7, 1.25])
    ax = f.add_subplot(gs[0])
    ae = f.add_subplot(gs[1], sharex=ax)
    for k, e in enumerate(tl['events'], 1):
        for a in (ax, ae):
            a.axvspan((e - tl['W']) / 24, e / 24, color=ORANGE, alpha=.1, lw=0)
        ax.axvspan(e / 24, (e + tl['dur']) / 24, color=ORANGE, alpha=.85, lw=0)
        ae.axvspan(e / 24, (e + tl['dur']) / 24, ymax=.38, color=ORANGE, alpha=.85, lw=0)   # below the A labels
        ax.text((e + tl['dur']) / 24 + .12, 3.6, f'E{k}\nh {e}', ha='left', va='top', fontsize=12,
                color=ORANGE, fontweight='semibold')
    ax.plot(d, tl['score'], color=BLUE, lw=1.4)
    ax.axhline(tl['thr'], color=TEAL, ls='--', lw=1.5)
    ax.text(8.7, tl['thr'] + .12, 'alert threshold', color=TEAL, fontsize=12, ha='left', va='bottom')
    ax.set_title('Shaded: 48-h warning window before each event; dark bars: events E1–E3 (start hour h)',
                 loc='left', fontsize=12.5, color=ORANGE, fontweight='normal')
    ax.set_ylim(-1.4, 4.1)
    ax.set_ylabel('Detector score')
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=1)
    for k, (a0, a1) in enumerate(tl['eps'], 1):
        ae.add_patch(Rectangle((a0 / 24, .25), max((a1 - a0 + 1) / 24, .12), .5, fc=INK, ec='none', zorder=3))
        ae.text((a0 + a1) / 48, .9, f'A{k}\nh {a0}', ha='center', va='bottom', fontsize=12, color=INK,
                linespacing=1.05)
    ae.set_ylim(0, 2.2)
    ae.set_yticks([])
    ae.set_ylabel('Alerts', rotation=0, ha='right', va='center')
    ae.set_xlim(0, 30)
    ae.set_xticks(range(0, 31, 2))
    ae.set_xlabel('Day')
    ae.spines['left'].set_visible(False)
    save('ml5_event_timeline', f, 'Synthetic 30-day hourly detector score (seed 22) with three events of 6 h; alert '
         'rows above threshold 1.5 merged into episodes when gaps are 6 h or less.', kind='computed')


@fig
def ml5_alert_budget():
    rng = np.random.default_rng(41)
    P, D, n_ev = 50, 365, 30
    S = rng.normal(0, 1, (P, D)) + rng.normal(0, .3, (P, 1))
    events = []                                          # 30 distinct events, at least 13 days apart on a pump
    while len(events) < n_ev:
        p_, d_ = int(rng.integers(0, P)), int(rng.integers(20, D - 10))
        if all(p_ != q or abs(d_ - e) > 12 for q, e in events):
            events.append((p_, d_))
    ev_p, ev_d = np.array(events).T
    valid, win = np.ones((P, D), bool), np.zeros((P, D), bool)
    for p_, d_, a in zip(ev_p, ev_d, rng.uniform(0, 3.5, n_ev)):
        S[p_, d_ - 5:d_] += a * np.array([.4, .55, .7, .85, 1.0])
        win[p_, d_ - 5:d_] = True
        valid[p_, d_:d_ + 7] = False
    weeks = D / 7
    rec, fal = [], []
    for thr in np.linspace(.8, 5.2, 221):
        A = (S > thr) & valid
        rec.append(np.mean([A[p_, d_ - 5:d_].any() for p_, d_ in zip(ev_p, ev_d)]))
        fp = (A & ~win).astype(int)
        fal.append((np.diff(np.c_[np.zeros(P, int), fp], axis=1) == 1).sum() / weeks)
    rec, fal = np.array(rec), np.array(fal)
    f, axs = half()
    ax = axs[0]
    ok = fal > 0
    ax.step(fal[ok], rec[ok], where='post', color=BLUE, lw=2.2)
    for cap, col in [(5, TEAL), (20, GREY)]:
        r = rec[fal <= cap].max()
        ax.axvline(cap, color=col, ls='--', lw=1.4)
        ax.scatter([cap], [r], s=60, c=col, zorder=4)
        if cap == 5:                                     # above-left: clear of the curve and of the 20 line
            ax.text(cap / 1.08, r + .06, f'{cap} per week:\nrecall {r:.0%}', color=col, fontsize=12.5,
                    ha='right', va='bottom')
        else:
            ax.text(cap * 1.08, r - .05, f'{cap} per week:\nrecall {r:.0%}', color=col, fontsize=12.5, va='top')
    ax.set_xscale('log')
    ax.set_xlim(.05, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    clean(ax, 'False-alert episodes per week (50 pumps)', 'Event recall (30 events)')
    save('ml5_alert_budget', f, 'Synthetic fleet (seed 41): 50 pumps x 365 days of daily scores, 30 distinct events '
         '(at least 13 days apart on a pump) with precursors of random strength in the 5 days before; alert '
         'episodes = runs of alert days outside warning windows.', kind='computed')


# ====================================================================== part 4
def _imbalanced_2d(n=3000, seed=9):
    rng = np.random.default_rng(seed)
    n1 = int(n * .05)
    X = np.r_[rng.normal([0, 0], 1, (n - n1, 2)), rng.normal([1.9, 1.9], .85, (n1, 2))]
    y = np.r_[np.zeros(n - n1, int), np.ones(n1, int)]
    return X, y


@fig
def ml5_class_weight():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score
    X, y = _imbalanced_2d()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, stratify=y, random_state=0)
    plain = LogisticRegression().fit(Xtr, ytr)
    bal = LogisticRegression(class_weight='balanced').fit(Xtr, ytr)
    # threshold for the unweighted model that matches the balanced model's training recall
    rb = recall_score(ytr, bal.predict(Xtr))
    ps = plain.predict_proba(Xtr)[:, 1]
    cand = np.sort(np.unique(ps))[::-1]
    t_match = next(t for t in cand if recall_score(ytr, ps >= t) >= rb)
    xx, yy = np.meshgrid(np.linspace(-3.6, 5.2, 300), np.linspace(-3.6, 5.2, 300))
    G = np.c_[xx.ravel(), yy.ravel()]
    f, axs = sub(3)
    idx = np.random.default_rng(1).permutation(len(Xte))[:900]
    for ax, (m, t, title) in zip(axs, [(plain, .5, 'No weights, threshold 0.5'),
                                       (bal, .5, "class_weight='balanced'"),
                                       (plain, t_match, f'No weights, threshold {t_match:.2f}')]):
        pg = m.predict_proba(G)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, pg, levels=[0, t, 1], colors=[BLUE_T, ORANGE_T], zorder=0)
        ax.contour(xx, yy, pg, levels=[t], colors=[INK], linewidths=1.6, zorder=1)
        Xs, ys = Xte[idx], yte[idx]
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=9, c=BLUE, alpha=.6, zorder=2, lw=0)
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=34, c=ORANGE, marker='^', zorder=3, edgecolors='white',
                   linewidths=.4)
        pred = m.predict_proba(Xte)[:, 1] >= t
        ax.set_title(title)
        ax.set_xlabel(f'test recall {recall_score(yte, pred):.2f} · precision {precision_score(yte, pred):.2f}',
                      fontsize=13)
        ax.set_xlim(-3.6, 5.2)
        ax.set_ylim(-3.6, 5.2)
        ax.set_aspect('equal')
        _hide_ticks(ax)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
    save('ml5_class_weight', f, f'Synthetic 2-D data (seed 9), 5% events, 50/50 stratified split; logistic '
         f'regression. Right: unweighted model with threshold {t_match:.3f}, chosen to match the balanced model\'s '
         f'training recall. Plots show 900 random test rows.', kind='computed')


def _calibration_models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=40000, n_features=8, n_informative=4, n_redundant=2, weights=[.95],
                               class_sep=.9, flip_y=.01, random_state=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    neg, pos = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    keep = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
    beta = len(pos) / len(neg)                               # fraction of negatives kept
    plain = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    bal = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr)
    under = LogisticRegression(max_iter=1000).fit(Xtr[keep], ytr[keep])
    iso = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight='balanced'), method='isotonic',
                                 cv=5).fit(Xtr, ytr)
    p = {'plain': plain.predict_proba(Xte)[:, 1], 'bal': bal.predict_proba(Xte)[:, 1],
         'under': under.predict_proba(Xte)[:, 1], 'iso': iso.predict_proba(Xte)[:, 1]}
    ps = p['under']
    p['prior'] = beta * ps / (beta * ps - ps + 1)
    return yte, p, beta


@fig
def ml5_reliability():
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    yte, p, beta = _calibration_models()
    f, axs = sub(2)
    specs = [[('plain', 'no rebalancing', BLUE), ('bal', "class_weight='balanced'", ORANGE),
              ('under', '1:1 undersampling', PURPLE)],
             [('plain', 'no rebalancing', BLUE), ('iso', 'balanced + isotonic (5-fold)', TEAL),
              ('under' if False else 'prior', 'undersampled + prior correction', GOLD)]]
    for ax, spec, title in zip(axs, specs, ['Rebalancing inflates probabilities', 'Recalibration restores them']):
        ax.plot([0, 1], [0, 1], color=LIGHT, ls='--', lw=1.3)
        for k, (key, name, col) in enumerate(spec):
            fr, mp = calibration_curve(yte, p[key], n_bins=10, strategy='quantile')
            style = dict(lw=2, ms=5, ls='-', zorder=3)
            if key == 'bal':                             # coincides with the undersampled curve: draw it wide beneath
                style = dict(lw=6, ms=10, ls='-', zorder=2)
            elif key == 'under':
                style = dict(lw=2, ms=5, ls='--', zorder=3)
            ax.plot(mp, fr, marker='o', color=col, **style)
            ax.text(.03, .95 - .085 * k, f'{name} · Brier {brier_score_loss(yte, p[key]):.4f}', color=col,
                    fontsize=12.5, va='top', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        clean(ax, 'Mean predicted probability (decile bins)', 'Observed event fraction' if ax is axs[0] else None)
    save('ml5_reliability', f, f'Synthetic data (make_classification, seed 5, 40,000 rows, 5.5% events, 50/50 split); '
         f'logistic regression. Prior correction p = b*ps / (b*ps - ps + 1) with b = {beta:.4f} (negatives kept).',
         kind='computed')


@fig
def ml5_recall_ci():
    ns = [5, 10, 20, 50, 100, 500]
    f, axs = half()
    ax = axs[0]
    for i, n in enumerate(ns):
        k = int(round(.8 * n))
        lo = stats.beta.ppf(.025, k, n - k + 1)
        hi = stats.beta.ppf(.975, k + 1, n - k)
        y = len(ns) - 1 - i
        col = ORANGE if n == 10 else BLUE
        ax.plot([lo, hi], [y, y], color=col, lw=3.2, solid_capstyle='round')
        ax.scatter([k / n], [y], s=60, c=col, zorder=3, edgecolors='white')
        ax.text(1.02, y, f'{lo:.2f}–{hi:.2f}', va='center', fontsize=12.5, color=col,
                fontweight='semibold' if n == 10 else 'normal')
    ax.set_yticks(range(len(ns)), [f'{int(round(.8 * n))} of {n}' for n in ns][::-1])
    ax.axvline(.8, color=LIGHT, lw=1, ls='--')
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_ylim(-.6, len(ns) - .4)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    clean(ax, 'Event recall (95% Clopper–Pearson interval)', 'Events detected')
    ax.grid(axis='y', alpha=0)
    save('ml5_recall_ci', f, 'Exact Clopper–Pearson 95% intervals for an observed recall of 80% at six event '
         'counts (scipy.stats.beta).', kind='computed')


# ====================================================================== part 5
def _g_important(x):
    return .2 + .8 * np.exp(-((x - .72) / .07) ** 2)


def _h_unimportant(y):
    return .03 * np.sin(2 * np.pi * y)


@fig
def ml5_grid_random():
    f = plt.figure(figsize=FULL, layout='constrained')
    outer = f.add_gridspec(1, 2, wspace=.08)
    g1 = np.linspace(1 / 6, 5 / 6, 3)
    grid = np.array([(a, b) for a in g1 for b in g1])
    rnd = np.random.default_rng(40).uniform(0, 1, (9, 2))
    xs = np.linspace(0, 1, 400)
    for k, (pts, title) in enumerate([(grid, 'Grid: 9 trials, 3 distinct values'),
                                      (rnd, 'Random: 9 trials, 9 distinct values')]):
        gs = outer[k].subgridspec(2, 2, height_ratios=[1, 2.3], width_ratios=[3.2, 1], hspace=.05, wspace=.05)
        am = f.add_subplot(gs[1, 0])
        at = f.add_subplot(gs[0, 0], sharex=am)
        ar = f.add_subplot(gs[1, 1], sharey=am)
        val = _g_important(pts[:, 0]) + _h_unimportant(pts[:, 1])
        best = int(np.argmax(val))
        am.scatter(pts[:, 0], pts[:, 1], s=70, c=BLUE, edgecolors='white', zorder=3)
        am.scatter(*pts[best], s=150, facecolors='none', edgecolors=ORANGE, linewidths=2, zorder=4)
        am.set_xlim(0, 1)
        am.set_ylim(0, 1)
        am.set_xticks([0, .5, 1])
        am.set_yticks([0, .5, 1])
        clean(am, 'Important parameter', 'Unimportant parameter' if k == 0 else None)
        at.plot(xs, _g_important(xs), color=TEAL, lw=2)
        for x in pts[:, 0]:
            at.plot([x, x], [0, _g_important(x)], color=BLUE, lw=1.4)
        at.scatter(pts[:, 0], _g_important(pts[:, 0]), s=22, c=BLUE, zorder=3)
        at.set_ylim(0, 1.1)
        at.set_yticks([])
        at.tick_params(labelbottom=False)
        at.set_title(title, fontsize=14)
        at.spines['left'].set_visible(False)
        ys = np.linspace(0, 1, 200)
        ar.plot(.2 + _h_unimportant(ys), ys, color=TEAL, lw=2)
        ar.set_xlim(0, 1.1)
        ar.set_xticks([])
        ar.tick_params(labelleft=False)
        ar.spines['bottom'].set_visible(False)
        am.text(.03, .975, f'best score {val[best]:.2f}', fontsize=12.5, color=ORANGE, zorder=5, va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=.85))
    save('ml5_grid_random', f, 'Synthetic objective: score = g(important) + h(unimportant), with a narrow optimum '
         'at 0.72 on the important axis (teal curves). Random trials: one draw (seed 40); 9 random trials land '
         'within 0.05 of the optimum with probability 1 - 0.9^9 = 0.61.', kind='computed')


@fig
def ml5_halving():
    rng = np.random.default_rng(8)
    n = 27
    q = rng.uniform(.60, .82, n)
    b = rng.uniform(.05, .28, n)
    rungs = [1, 3, 9, 27]
    alive = np.arange(n)
    hist = {i: [] for i in range(n)}
    for r in rungs:
        obs = q[alive] - b[alive] / np.sqrt(r) + rng.normal(0, .02 / np.sqrt(r), alive.size)
        for i, o in zip(alive, obs):
            hist[i].append((r, o))
        if r != rungs[-1]:
            alive = alive[np.argsort(-obs)[:alive.size // 3]]
    winner = int(alive[0])
    true_best = int(np.argmax(q))
    f, axs = sub(1)
    ax = axs[0]
    for i in range(n):
        rs, os_ = zip(*hist[i])
        if i == winner:
            continue
        col = GREY if len(rs) < 4 else BLUE
        ax.plot(rs, os_, color=col, lw=1.1, alpha=.55 if len(rs) == 1 else .8, marker='o', ms=4)
        if len(rs) < 4:
            ax.scatter(rs[-1], os_[-1], marker='x', s=38, c=GREY, zorder=3)
    rs, os_ = zip(*hist[winner])
    ax.plot(rs, os_, color=BLUE, lw=3, marker='o', ms=7, zorder=5)
    ax.annotate('selected configuration', (27, os_[-1]), (15.5, os_[-1] + .065), fontsize=12.5, color=BLUE,
                arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.1))
    if true_best != winner:
        rs_b, os_b = zip(*hist[true_best])
        ax.scatter(rs_b[-1], os_b[-1], s=140, marker='*', c=ORANGE, zorder=6)
        ax.annotate(f'best at full budget (true {q[true_best]:.2f})\ndropped after {rs_b[-1]} unit'
                    f'{"s" if rs_b[-1] > 1 else ""}', (rs_b[-1], os_b[-1]), (3.4, .5), fontsize=12, color=ORANGE,
                    arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    for r, m in zip(rungs, [27, 9, 3, 1]):
        ax.text(r, .905, f'{m} × {r}', ha='center', fontsize=13, fontweight='semibold', color=INK)
    ax.text(1.02, .95, 'candidates × budget units per rung', fontsize=12, color=MUTED, ha='left')
    ax.set_xscale('log')
    ax.set_xticks(rungs, [str(r) for r in rungs])
    ax.set_xlim(.8, 34)
    ax.set_ylim(.3, .98)
    clean(ax, 'Budget per candidate (units, e.g. 100s of rows or boosting rounds)', 'Validation score')
    total = sum(m * r for r, m in zip(rungs, [27, 9, 3, 1]))
    ax.text(30, .33, f'total {total} units vs {27 * 27} for 27 full runs', ha='right', fontsize=12.5, color=INK)
    save('ml5_halving', f, 'Simulated learning curves for 27 candidates (seed 8): score = q - b / sqrt(budget) + '
         'noise; top third kept at each rung (factor 3).', kind='computed')


@fig
def ml5_bayesopt():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    def obj(x):
        return .62 + .14 * np.exp(-(x - .6) ** 2 / .8) + .04 * np.exp(-(x + 1.9) ** 2 / .15)
    xs = np.linspace(-3, 2, 500)
    X = np.array([-2.5, -1.2, .1, 1.9])
    y = obj(X)
    gp = GaussianProcessRegressor(ConstantKernel(2.0, 'fixed') * Matern(.8, 'fixed', nu=2.5), alpha=1e-6,
                                  normalize_y=True, optimizer=None).fit(X[:, None], y)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    best = y.max()
    xi = .01
    z = (mu - best - xi) / np.maximum(sd, 1e-12)
    ei = (mu - best - xi) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)
    nxt = xs[int(np.argmax(ei))]
    f = plt.figure(figsize=HALF, layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[2.1, 1])
    ax = f.add_subplot(gs[0])
    ae = f.add_subplot(gs[1], sharex=ax)
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=BLUE_T, lw=0)
    ax.plot(xs, obj(xs), color=GREY, ls='--', lw=1.4)
    ax.plot(xs, mu, color=BLUE, lw=2.2)
    ax.scatter(X, y, s=60, c=INK, zorder=4)
    ax.axvline(nxt, color=ORANGE, lw=1.4, ls=':')
    ax.text(-2.95, .852, 'dashed: true score (unknown)\nblue: surrogate mean ± 2 SD', fontsize=12, va='top')
    ax.set_ylim(.52, .86)
    ax.set_ylabel('Validation score')
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=1)
    ae.fill_between(xs, 0, ei, color=ORANGE, alpha=.25, lw=0)
    ae.plot(xs, ei, color=ORANGE, lw=2)
    ae.axvline(nxt, color=ORANGE, lw=1.4, ls=':')
    ae.annotate(f'next trial: log10 C = {nxt:.2f}', (nxt, ei.max() * .97), (-2.9, ei.max() * .8), fontsize=12,
                color=ORANGE, va='center', arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ae.set_yticks([])
    ae.set_ylabel('Expected\nimprovement')
    ae.set_xlim(-3, 2)
    ae.grid(alpha=1)
    ae.set_xlabel('Hyperparameter (log10 C)')
    save('ml5_bayesopt', f, 'Synthetic 1-D objective; scikit-learn GaussianProcessRegressor, Matern 2.5 kernel with fixed '
         'length scale 0.8 and variance 2 on normalized scores, fitted to 4 trials; expected improvement, xi = 0.01.', kind='computed')


@fig
def ml5_nested_cv():
    f, ax = canvas(11.8, 4.2)
    bw, bh, gap = .56, .4, .06
    x0, ytop = 1.3, 3.45
    label(ax, x0 + 2.5 * (bw + gap), 4.0, 'Outer loop: estimate the procedure', fs=14, weight='semibold')
    for r in range(5):
        y = ytop - r * .62
        label(ax, x0 - .15, y + bh / 2, f'Outer {r + 1}', fs=12.5, ha='right', color=MUTED)
        for c in range(5):
            test = c == r
            box(ax, x0 + c * (bw + gap), y, bw, bh, 'test' if test else '', ORANGE if test else BLUE,
                fill='solid' if test else 'tint', fs=11.5, radius=.05, lw=1.2)
    # inner loop for outer fold 1
    ix0, iy = 5.45, 3.45
    label(ax, ix0 + 2 * (bw + gap) - gap / 2, 4.0, 'Inner loop: tune on them', fs=14, weight='semibold')
    for r in range(4):
        y = iy - r * .62
        for c in range(4):
            val = c == r
            box(ax, ix0 + c * (bw + gap), y, bw, bh, 'val' if val else '', TEAL if val else BLUE,
                fill='solid' if val else 'tint', fs=11.5, radius=.05, lw=1.2)
    arrow(ax, (x0 + 5 * (bw + gap) + .05, ytop + bh / 2), (ix0 - .12, iy + bh / 2), color=MUTED, lw=1.4)
    label(ax, (x0 + 5 * (bw + gap) + ix0) / 2, ytop - .12, "its training\nrows", fs=11.5, color=MUTED,
          va='top')
    steps = [('Pick the best settings', 'mean inner validation score', TEAL),
             ('Refit on all outer training rows', 'same pipeline, chosen settings', BLUE),
             ('Score once on the outer test fold', 'never used for any choice', ORANGE)]
    sx, sy = 8.45, iy + bh / 2 - .35
    prev = None
    for k, (t, sub_, col) in enumerate(steps):
        b = box(ax, sx, sy - k * .98, 3.05, .7, t, col, fs=12.5, sub=sub_, sub_fs=11, weight='semibold')
        if prev is not None:
            arrow(ax, prev.b, b.t, color=MUTED, lw=1.3, shrink=1)
        prev = b
    arrow(ax, (ix0 + 4 * (bw + gap) + .02, iy + bh / 2), (sx - .04, iy + bh / 2), color=MUTED, lw=1.3)
    label(ax, sx + 1.52, .42, 'Repeat for each outer fold: 5 test scores\nestimate tuning + fitting together',
          fs=12.5, color=INK)
    label(ax, x0 + 2.5 * (bw + gap), .42, 'Tuning happens inside every outer\ntraining split, never on its test fold',
          fs=12.5, color=INK)
    save('ml5_nested_cv', f, 'Schematic nested cross-validation: 5 outer folds, 4 inner folds.', kind='schematic')


@fig
def ml5_paired_folds():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5, n_redundant=0, weights=[.85],
                               class_sep=.8, flip_y=.02, random_state=4)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    a = cross_val_score(RandomForestClassifier(n_estimators=300, max_features=1.0, random_state=0, n_jobs=-1), X, y,
                        cv=cv, scoring='average_precision')
    b = cross_val_score(RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=0, n_jobs=-1), X, y,
                        cv=cv, scoring='average_precision')
    d = b - a
    k = len(d)
    tq = stats.t.ppf(.975, k - 1)
    se_naive = d.std(ddof=1) / np.sqrt(k)
    se_corr = d.std(ddof=1) * np.sqrt(1 / k + .25)          # Nadeau-Bengio: test/train = 1/4 for 5-fold
    f, axs = sub(2)
    ax = axs[0]
    for i in range(k):
        ax.plot([0, 1], [a[i], b[i]], color=TEAL if d[i] > 0 else GREY, lw=1.4, marker='o', ms=5, alpha=.9)
    for xx, v, col in [(-.08, a, GREY), (1.08, b, BLUE)]:
        m, e = v.mean(), tq * v.std(ddof=1) / np.sqrt(k)
        ax.errorbar([xx], [m], yerr=[[e], [e]], fmt='s', color=col, ms=8, capsize=5, lw=2)
    ax.set_xticks([0, 1], ['A: max_features=1.0', "B: max_features='sqrt'"])
    ax.set_xlim(-.35, 1.35)
    ax.set_title('Same 10 folds, both models')
    ax.text(.5, a.min() - .005, f'{int((d > 0).sum())} of {k} folds favour B', ha='center', va='top', fontsize=12.5,
            color=TEAL)
    clean(ax, None, 'Average precision (fold)')
    ax.set_ylim(a.min() - .03, max(a.max(), b.max()) + .02)
    ax = axs[1]
    ax.axhline(0, color=LIGHT, lw=1.2)
    ax.scatter(np.arange(1, k + 1), d, s=46, c=np.where(d > 0, TEAL, GREY), zorder=3)
    ax.axhline(d.mean(), color=BLUE, lw=2)
    for xx, se, col, name in [(11.4, se_naive, GREY, 'naive'), (12.5, se_corr, ORANGE, 'corrected')]:
        ax.errorbar([xx], [d.mean()], yerr=[[tq * se], [tq * se]], fmt='o', color=col, capsize=6, lw=2.2, ms=6)
        ax.text(xx, d.mean() - tq * se - .0015, name, ha='center', va='top', fontsize=12, color=col)
    ax.text(5.45, d.mean() + .0015, f'mean B − A = {d.mean():.3f}', color=BLUE, fontsize=12.5, va='bottom')
    ax.set_xticks(range(1, k + 1))
    ax.set_xlim(.4, 13.7)
    ax.set_ylim(min(d.min(), d.mean() - tq * se_corr) - .009, max(d.max(), d.mean() + tq * se_corr) + .005)
    ax.set_title('Paired differences with 95% intervals')
    clean(ax, 'Fold (5-fold CV repeated twice)', 'B − A')
    save('ml5_paired_folds', f, f'Synthetic data (make_classification, seed 4, 1,200 rows, about 16% events); '
         f'RepeatedStratifiedKFold 5x2 (seed 0). Mean difference {d.mean():.4f}, SD {d.std(ddof=1):.4f}; naive SE '
         f'{se_naive:.4f}; corrected SE {se_corr:.4f} (Nadeau-Bengio, test/train = 0.25).', kind='computed')


@fig
def ml5_one_se():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=800, n_features=12, n_informative=4, n_redundant=2, weights=[.8],
                               class_sep=.9, flip_y=.05, random_state=20)
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    depths = np.arange(1, 13)
    m, se = [], []
    for dpt in depths:
        s = cross_val_score(DecisionTreeClassifier(max_depth=dpt, random_state=0), X, y, cv=cv,
                            scoring='balanced_accuracy')
        m.append(s.mean())
        se.append(s.std(ddof=1) / np.sqrt(len(s)))
    m, se = np.array(m), np.array(se)
    ib = int(np.argmax(m))
    line = m[ib] - se[ib]
    io = int(np.argmax(m >= line))
    f, axs = half()
    ax = axs[0]
    ax.axhline(line, color=TEAL, ls='--', lw=1.5)
    ax.errorbar(depths, m, yerr=se, fmt='o-', color=BLUE, capsize=4, lw=1.8, ms=6)
    ax.scatter([depths[ib]], [m[ib]], s=170, facecolors='none', edgecolors=GREY, linewidths=2, zorder=4)
    ax.scatter([depths[io]], [m[io]], s=170, facecolors='none', edgecolors=ORANGE, linewidths=2.4, zorder=4)
    ax.annotate(f'best mean: depth {depths[ib]}\n{m[ib]:.3f} ± {se[ib]:.3f}', (depths[ib], m[ib]), (7.3, .63),
                fontsize=12.5, color=GREY, arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.1))
    ax.annotate(f'simplest within 1 SE:\ndepth {depths[io]} ({m[io]:.3f})', (depths[io], m[io]), (3.6, .6),
                fontsize=12.5, color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ax.text(.7, .792, f'dashed: best − 1 SE = {line:.3f}', color=TEAL, fontsize=12.5, ha='left', va='top')
    ax.set_xticks(depths)
    ax.set_ylim(.55, .8)
    clean(ax, 'Tree depth', 'Balanced accuracy (10-fold CV)')
    save('ml5_one_se', f, 'Synthetic data (make_classification, seed 20, 800 rows); DecisionTreeClassifier, '
         'StratifiedKFold(10, shuffle, seed 0); error bars = standard error across folds.', kind='computed')


# ====================================================================== part 6
def _truth(X, c=1.0):
    return (X[:, 1] > .25 * X[:, 0] ** 2 + c).astype(int)


@fig
def ml5_drift_types():
    rng = np.random.default_rng(14)
    ref = rng.normal(0, 1, (500, 2))
    model = LogisticRegression().fit(ref, _truth(ref))
    shifted = rng.normal([2.4, 1.4], .75, (500, 2))
    same = rng.normal(0, 1, (500, 2))
    cases = [(ref, 1.0, 'Reference period'), (shifted, 1.0, 'Data drift: inputs move'),
             (same, .2, 'Concept drift: labels change')]
    xx = np.linspace(-3.5, 4.8, 300)
    f, axs = sub(3)
    for ax, (X, c, title) in zip(axs, cases):
        y = _truth(X, c)
        err = (model.predict(X) != y).mean()
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c=BLUE, alpha=.7, lw=0)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, c=ORANGE, marker='^', lw=0)
        ax.plot(xx, .25 * xx ** 2 + c, color=INK, ls='--', lw=1.8)
        w, b0 = model.coef_[0], model.intercept_[0]
        ax.plot(xx, -(w[0] * xx + b0) / w[1], color=PURPLE, lw=2.2)
        ax.set_title(title)
        ax.set_xlabel(f'model error rate {err:.0%}', fontsize=13)
        ax.set_xlim(-3.5, 4.8)
        ax.set_ylim(-3.5, 4.8)
        ax.set_aspect('equal')
        _hide_ticks(ax)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
    axs[1].text(-3.3, -2.2, 'dashed: true boundary\npurple: model (fitted on reference)', fontsize=12, va='top')
    save('ml5_drift_types', f, 'Synthetic data (seed 14): events when x2 > 0.25 x1^2 + c; logistic regression fitted '
         'on the reference period only. Data drift moves the inputs; concept drift lowers c from 1.0 to 0.2.',
         kind='computed')


@fig
def ml5_drift_psi():
    rng = np.random.default_rng(15)
    ref = rng.normal(120, 10, 5000)
    cur = rng.normal(125, 12, 1500)
    edges = np.r_[-np.inf, np.quantile(ref, np.linspace(.1, .9, 9)), np.inf]
    e = np.histogram(ref, edges)[0] / ref.size
    a = np.histogram(cur, edges)[0] / cur.size
    psi = float(np.sum((a - e) * np.log(a / e)))
    ks = stats.ks_2samp(ref, cur)
    f = plt.figure(figsize=HALF, layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[1, 1.05])
    ax, ak = f.add_subplot(gs[0]), f.add_subplot(gs[1])
    k = np.arange(10)
    ax.bar(k - .2, e * 100, .4, color=BLUE, label='reference')
    ax.bar(k + .2, a * 100, .4, color=ORANGE, label='current')
    ax.set_xticks(k, [str(i + 1) for i in k])
    ax.set_ylabel('Rows (%)')
    ax.set_xlabel('Reference decile bin')
    top = max(a.max(), e.max()) * 100 * 1.22
    ax.text(-.4, top * .98, 'reference', color=BLUE, fontsize=12.5, va='top')
    ax.text(1.9, top * .98, 'current', color=ORANGE, fontsize=12.5, va='top')
    ax.text(4.5, top * .98, f'PSI = {psi:.2f}', fontsize=13, fontweight='semibold', va='top')
    ax.set_ylim(0, top)
    ax.grid(axis='y', alpha=1)
    ax.set_axisbelow(True)
    xs = np.linspace(80, 170, 600)
    fr = np.searchsorted(np.sort(ref), xs, side='right') / ref.size
    fc = np.searchsorted(np.sort(cur), xs, side='right') / cur.size
    i = int(np.argmax(np.abs(fr - fc)))
    ak.plot(xs, fr, color=BLUE, lw=2)
    ak.plot(xs, fc, color=ORANGE, lw=2)
    ak.plot([xs[i], xs[i]], [fc[i], fr[i]], color=INK, lw=2.2)
    ak.text(xs[i] + 3.5, fc[i] + .02, f'KS D = {ks.statistic:.2f}', fontsize=13, fontweight='semibold',
            va='top')                                     # below both curves, next to the foot of the bar
    ak.set_xlim(80, 170)
    clean(ak, 'Intake pressure (bar)', 'Cumulative share')
    save('ml5_drift_psi', f, f'Synthetic intake pressure (seed 15): reference N(120, 10), 5,000 rows; current '
         f'N(125, 12), 1,500 rows. PSI over reference deciles = {psi:.3f}; two-sample KS D = {ks.statistic:.3f}, '
         f'p = {ks.pvalue:.1e}.', kind='computed')


@fig
def ml5_lifecycle():
    f, ax = canvas(11.8, 4.1)
    w, h = 2.5, .95
    top = [('Frame and data contract', 'target, horizon, units', GREY),
           ('Train and tune', 'pipelines, inner CV', BLUE),
           ('Validate and calibrate', 'frozen test, threshold', TEAL),
           ('Deploy', 'versioned model + policy', ORANGE)]
    bottom = [('Monitor', 'inputs, scores, alerts, outcomes', PURPLE),
              ('Review trigger', 'recalibrate, retrain, or retire', GOLD)]
    xs = [.15, 3.0, 5.85, 8.7]
    yt, yb = 2.75, .55
    ta = [box(ax, x, yt, w, h, t, c, fs=13.5, sub=sb, sub_fs=11.5, weight='semibold',
              fill='solid' if t == 'Deploy' else 'tint') for x, (t, sb, c) in zip(xs, top)]
    ba = [box(ax, x, yb, 3.15, h, t, c, fs=13.5, sub=sb, sub_fs=11.5, weight='semibold')
          for x, (t, sb, c) in zip([8.05, 1.75], bottom)]
    for a, b in zip(ta[:-1], ta[1:]):
        arrow(ax, a.r, b.l, color=MUTED, lw=1.6, shrink=2)
    arrow(ax, ta[3].b, (ta[3].b[0], ba[0].t[1]), color=MUTED, lw=1.6, shrink=2)
    label(ax, 5.35, 2.5, 'Monitor, with a pre-agreed response for each signal:', fs=12.5, color=PURPLE)
    for (cx, cy, t) in [(3.3, 2.13, 'input drift (PSI, KS)'), (5.35, 2.13, 'stuck or missing sensors'),
                        (7.45, 2.13, 'score distribution'), (3.3, 1.8, 'alert rate vs capacity'),
                        (5.35, 1.8, 'reviewed precision'), (7.45, 1.8, 'event recall (late labels)')]:
        chip(ax, cx, cy, t, color=PURPLE, fs=11.5, weight='medium')
    arrow(ax, ba[0].l, ba[1].r, color=MUTED, lw=1.6, shrink=2, text='evidence of drift or decay', fs=12,
          offset=(0, .2), text_color=INK)
    ax.plot([ba[1].l[0], .9], [ba[1].l[1], ba[1].l[1]], color=MUTED, lw=1.6)
    arrow(ax, (.9, ba[1].l[1]), (.9, ta[0].b[1]), color=MUTED, lw=1.6)
    label(ax, 1.0, 2.05, 'reframe if the\ndecision changed', fs=12, ha='left', color=INK)
    save('ml5_lifecycle', f, 'Schematic model lifecycle with a monitoring feedback loop.', kind='schematic')


@fig
def ml5_course_map():
    f, ax = canvas(11.8, 4.2)
    stages = [('Frame', 'contract,\nprediction time', ['ML1']),
              ('Audit and split', 'grouped / time\nsplits, leakage', ['ML1']),
              ('Features and\npipelines', 'scaling, selection,\nPCA', ['ML1', 'ML2', 'ML4']),
              ('Models', 'linear to ensembles,\nclusters, detectors', ['ML1', 'ML2', 'ML3', 'ML5']),
              ('Evaluate and\nselect', 'metrics, events,\nnested CV', ['ML1', 'ML5']),
              ('Explain and\ndocument', 'importance,\nmodel cards', ['ML2']),
              ('Deploy and\nmonitor', 'thresholds,\ndrift', ['ML5'])]
    w, g = 1.47, .2
    x0, y = .2, 2.05
    cols = {'ML1': BLUE, 'ML2': TEAL, 'ML3': PURPLE, 'ML4': GOLD, 'ML5': ORANGE}
    prev = None
    for i, (t, sb, mods) in enumerate(stages):
        x = x0 + i * (w + g)
        b = box(ax, x, y, w, 1.05, t, INK, fs=13, weight='semibold', fill='tint')
        label(ax, x + w / 2, y - .12, sb, fs=11.5, color=MUTED, va='top')
        rows = [mods[:2], mods[2:]] if len(mods) > 2 else [mods]
        for r, row in enumerate(rows):
            for j, m in enumerate(row):
                chip(ax, x + w / 2 + (j - (len(row) - 1) / 2) * .56, y + 1.3 + .36 * (len(rows) - 1 - r), m,
                     color=cols[m], fs=11.5)
        if prev is not None:
            arrow(ax, prev.r, b.l, color=MUTED, lw=1.5, shrink=1)
        prev = b
    ax.plot([x0 + 6 * (w + g) + w / 2, x0 + 6 * (w + g) + w / 2], [y - .95, y - 1.35], color=MUTED, lw=1.5)
    ax.plot([x0 + 6 * (w + g) + w / 2, x0 + w / 2], [y - 1.35, y - 1.35], color=MUTED, lw=1.5)
    arrow(ax, (x0 + w / 2, y - 1.35), (x0 + w / 2, y - .95), color=MUTED, lw=1.5)
    label(ax, 5.9, y - 1.55, 'monitoring evidence sends you back to framing', fs=12.5, color=INK, va='top')
    save('ml5_course_map', f, 'Schematic of the course workflow; chips show where each module teaches the stage.',
         kind='schematic')


@fig
def ml5_alert_policy():
    f, ax = canvas(11.8, 3.2)
    labels = ['Sensor\nrows', 'Trailing\nfeatures', 'Detector\nscore', 'Threshold', 'Persistence', 'Cooldown',
              'Alert with\ncontext']
    subs = [None, None, None, 'from costs or\ncapacity', 'e.g. 3 of 4 h', 'e.g. 12 h per\nasset', None]
    cols = [GREY, GREY, BLUE, TEAL, TEAL, TEAL, ORANGE]
    w, g, y, h = 1.4, .25, 1.3, 1.2
    xs = [.2 + i * (w + g) for i in range(len(labels))]
    anchors = []
    for i, (x, t, sb, c) in enumerate(zip(xs, labels, subs, cols)):
        if sb:
            b = box(ax, x, y, w, h, t, c, fs=13, weight='semibold', sub=sb, sub_fs=11)
            ax.texts[-2].set_y(y + h * .68)
            ax.texts[-1].set_y(y + h * .28)
        else:
            b = box(ax, x, y, w, h, t, c, fs=13, weight='semibold', fill='solid' if i == 6 else 'tint')
        anchors.append(b)
        if i:
            arrow(ax, anchors[i - 1].r, b.l, color=MUTED, lw=1.5, shrink=1)
    bracket(ax, xs[3], xs[5] + w, y + h + .12, 'alert policy: written down, versioned, reviewed', color=TEAL,
            below=False, fs=12.5)
    ax.plot([anchors[6].b[0], anchors[6].b[0]], [y, .45], color=MUTED, lw=1.4)
    ax.plot([anchors[6].b[0], anchors[3].b[0]], [.45, .45], color=MUTED, lw=1.4)
    arrow(ax, (anchors[3].b[0], .45), anchors[3].b, color=MUTED, lw=1.4)
    label(ax, (anchors[6].b[0] + anchors[3].b[0]) / 2, .22, 'reviewed outcomes become labels for the next '
          'threshold review', fs=12, color=INK)
    save('ml5_alert_policy', f, 'Schematic alert pipeline; persistence and cooldown values are illustrative policy '
         'choices, not recommendations.', kind='schematic')


# ====================================================================== lab debriefs (course data)
def _lab_frame():
    df = pd.read_csv(WELL_CSV)
    feats = ['GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'BS']
    dc = df.dropna(subset=feats).copy()
    Xs = StandardScaler().fit_transform(dc[feats].values)
    z = (np.abs(Xs) > 3).any(1)
    q1, q3 = dc[feats].quantile(.25), dc[feats].quantile(.75)
    iqr = q3 - q1
    out_iqr = (dc[feats] < q1 - 1.5 * iqr) | (dc[feats] > q3 + 1.5 * iqr)
    return df, dc, feats, Xs, z, out_iqr


@fig
def ml5_lab1_results():
    df, dc, feats, Xs, z, out_iqr = _lab_frame()
    iso = {c: int((IsolationForest(contamination=c, n_estimators=100, random_state=42).fit_predict(Xs) == -1).sum())
           for c in (.01, .05, .1)}
    f = plt.figure(figsize=HALF, layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[1.25, 1])
    ax, ab = f.add_subplot(gs[0]), f.add_subplot(gs[1])
    names = ['|z| > 3', 'IQR fences', 'IF, contamination 0.01', 'IF, contamination 0.05', 'IF, contamination 0.10']
    vals = [int(z.sum()), int(out_iqr.any(axis=1).sum()), iso[.01], iso[.05], iso[.1]]
    cols = [GREY, PURPLE, BLUE, BLUE, BLUE]
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, vals, color=cols, height=.62)
    for yv, v in zip(yy, vals):
        ax.text(v + 8, yv, f'{v}', va='center', fontsize=12.5)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, 640)
    ax.set_title(f'Flagged rows of {len(dc):,} complete rows', fontsize=13.5, loc='left')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=1)
    ax.set_axisbelow(True)
    per = out_iqr.sum().sort_values()
    yb = np.arange(len(per))
    ab.barh(yb, per.values, color=PURPLE, height=.62)
    for yv, v in zip(yb, per.values):
        ab.text(v + 6, yv, f'{int(v)}', va='center', fontsize=12.5)
    ab.set_yticks(yb, per.index)
    ab.set_xlim(0, 640)
    ab.set_title('IQR flags by curve', fontsize=13.5, loc='left')
    ab.tick_params(axis='y', length=0)
    ab.grid(axis='x', alpha=1)
    ab.set_axisbelow(True)
    ab.set_xlabel('Rows flagged')
    save('ml5_lab1_results', f, f'Computed from well_logs_data.csv (course data): {len(df):,} rows, {len(dc):,} complete '
         f'in GR, RHOB, NPHI, DT, CALI, BS; StandardScaler; IsolationForest(n_estimators=100, random_state=42); '
         f'scikit-learn 1.7.', kind='data')


@fig
def ml5_lab2_overlap():
    df, dc, feats, Xs, z, out_iqr = _lab_frame()
    flags = {
        '|z| > 3': z,
        'IQR': out_iqr.any(axis=1).values,
        'IF': IsolationForest(contamination=.05, n_estimators=100, random_state=42).fit_predict(Xs) == -1,
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=.05).fit_predict(Xs) == -1,
        'OCSVM': OneClassSVM(nu=.05, kernel='rbf', gamma='scale').fit(Xs).predict(Xs) == -1,
    }
    M = np.array([[int((a & b).sum()) for b in flags.values()] for a in flags.values()])
    f, ax = canvas(6.3, 4.9)
    cell, x0, yt = .74, 1.75, 4.3
    matrix(ax, x0, yt, M, cell=cell, cmap='course_seq', vmin=0, vmax=M.max(), fs=14, fmt='{:.0f}')
    for i, n in enumerate(flags):
        label(ax, x0 - .1, yt - (i + .5) * cell, n, fs=13, ha='right')
        label(ax, x0 + (i + .5) * cell, yt + .1, n, fs=13, va='bottom')
    label(ax, x0 + 2.5 * cell, .3, 'diagonal: rows flagged by each method;\noff-diagonal: rows flagged by both',
          fs=12, color=MUTED)
    save('ml5_lab2_overlap', f, f'Computed from well_logs_data.csv (course data), {len(dc):,} complete rows; '
         f'lab settings (contamination / nu = 0.05, LOF n_neighbors = 20, OCSVM gamma = "scale"); '
         f'scikit-learn 1.7.', kind='data')
