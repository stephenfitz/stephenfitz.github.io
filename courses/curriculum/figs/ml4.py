"""Figures for module key 'ml4' (dimensionality reduction and feature selection).

Figure names start with 'ml4_'. Synthetic energy datasets shared by several
figures are generated deterministically below; relationships between channels
are illustrative, not field-calibrated physics:

* drill_channels(): ten surface drilling channels (WOB, torque, RPM, ROP, SPP,
  flow, ECD, differential pressure, hook load, gas) driven by three latent
  drivers (formation hardness, pump rate, rotary speed) plus noise; the gas
  channel is independent of all three drivers.
* log_suite(): a synthetic well-log suite (GR, SP, PEF, RHOB, NPHI, DT, RT,
  CALI) generated from shale volume, porosity and water saturation, with a
  log-permeability target.

Colour meaning used throughout the deck: class 0/1 = BLUE/ORANGE; data = grey;
PCA = BLUE; selected / kept features = TEAL; dropped or irrelevant = light grey
or ORANGE; wrong (leaky) protocol = ORANGE, correct protocol = TEAL; anomaly
threshold = ORANGE. Latent drivers: hardness = BLUE, pump rate = TEAL,
rotary speed = PURPLE, independent channel = GOLD.
"""
from functools import lru_cache

import pandas as pd
from scipy.special import gamma as gamma_fn

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

KEY = 'ml4'
fig = register(KEY)

GREY = '#56687A'
LIGHT = '#C9D3DD'
DRIVER_COL = {'hardness': BLUE, 'pump': TEAL, 'rotary': PURPLE, 'none': GOLD}


# ------------------------------------------------------------------ data
def _ar1(rng, n, phi=.9):
    z = rng.normal(size=n)
    e = np.empty(n)
    e[0] = z[0]
    for i in range(1, n):
        e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * z[i]
    return e


def _steps(rng, n, lo, hi, step_lo=40, step_hi=140):
    """Piecewise-constant setpoints (driller or formation changes)."""
    out = np.empty(n)
    j = 0
    while j < n:
        t = int(rng.integers(step_lo, step_hi))
        out[j:j + t] = rng.uniform(lo, hi)
        j += t
    return out


# channel: (unit, mean, sd, loadings on hardness, pump, rotary; noise share)
DRILL_SPEC = {
    'WOB': ('klbf', 25, 5, (.95, 0, 0), .25),
    'Torque': ('kft·lbf', 15, 3, (.70, 0, .55), .30),
    'RPM': ('rpm', 120, 20, (0, 0, .95), .25),
    'ROP': ('ft/h', 60, 15, (-.70, .30, .45), .35),
    'SPP': ('psi', 3200, 250, (.20, .92, 0), .25),
    'Flow': ('gpm', 700, 60, (0, .95, 0), .22),
    'ECD': ('ppg', 10.8, .10, (0, .85, 0), .40),
    'DiffP': ('psi', 450, 80, (.80, .30, 0), .35),
    'Hookload': ('klbf', 260, 6, (-.85, 0, 0), .35),
    'Gas': ('units', 20, 8, (0, 0, 0), 1.0),
}
DRILL_DRIVER = {'WOB': 'hardness', 'Torque': 'hardness', 'RPM': 'rotary', 'ROP': 'hardness',
                'SPP': 'pump', 'Flow': 'pump', 'ECD': 'pump', 'DiffP': 'hardness',
                'Hookload': 'hardness', 'Gas': 'none'}
DRIVER_LABEL = {'hardness': 'formation hardness', 'pump': 'pump rate', 'rotary': 'rotary speed',
                'none': 'none (independent)'}


def driver_legend(target, **kw):
    """Legend explaining the channel colours: the hidden driver behind each drilling channel."""
    handles = [Line2D([], [], ls='', marker='', label='Colour = hidden driver:')]
    handles += [Line2D([], [], marker='s', ls='', ms=10, color=DRIVER_COL[k], label=v) for k, v in DRIVER_LABEL.items()]
    opts = dict(ncol=5, fontsize=12, frameon=False, handletextpad=.3, columnspacing=1.2, handlelength=.8)
    opts.update(kw)
    return target.legend(handles=handles, **opts)


@lru_cache(maxsize=None)
def drill_channels(n: int = 720, seed: int = 4):
    """Ten synthetic drilling channels (one row per minute) and the latent drivers."""
    rng = np.random.default_rng(seed)
    form = np.repeat([0, 1, 2, 1], [170, 200, 190, n - 560])[:n]        # formation sequence
    hard = np.array([-1.1, .1, 1.2])[form] + .35 * _ar1(rng, n, .97)
    pump = _steps(rng, n, -1.3, 1.3, 60, 160) + .25 * _ar1(rng, n, .9)
    rot = _steps(rng, n, -1.4, 1.4, 50, 150) + .25 * _ar1(rng, n, .9)
    L = np.c_[hard, pump, rot]
    L = L - L.mean(0)
    for j in range(1, 3):                       # drivers exactly uncorrelated (Gram-Schmidt)
        for i in range(j):
            L[:, j] -= (L[:, j] @ L[:, i]) / (L[:, i] @ L[:, i]) * L[:, i]
    L = L / L.std(0)
    cols = {}
    for name, (unit, mu, sd, w, noise) in DRILL_SPEC.items():
        w = np.asarray(w, float)
        if name == 'Gas':
            z = .8 * _ar1(rng, n, .95) + .6 * rng.normal(size=n)
            spikes = rng.random(n) < .015
            z = z + spikes * rng.uniform(1.5, 3, n)
            z = z - z.mean()
            z = z - L @ np.linalg.lstsq(L, z, rcond=None)[0]   # independent of all drivers
        else:
            z = L @ w + noise * rng.normal(size=n)
        z = (z - z.mean()) / z.std()
        cols[name] = mu + sd * z
    df = pd.DataFrame(cols)
    return df, pd.DataFrame(L, columns=['hardness', 'pump', 'rotary']), form


LOG_NAMES = ['GR', 'SP', 'PEF', 'CALI', 'RHOB', 'NPHI', 'DT', 'RT']


@lru_cache(maxsize=None)
def log_suite(n: int = 900, seed: int = 11):
    """Synthetic log suite from shale volume, porosity and saturation (depth order)."""
    rng = np.random.default_rng(seed)
    vsh = np.clip(.45 + .32 * _ar1(rng, n, .985), 0, 1)
    phi = np.clip(.19 - .07 * (vsh - .45) + .045 * _ar1(rng, n, .96), .02, .34)
    sw = np.clip(.55 + .22 * _ar1(rng, n, .97) + .15 * (vsh - .45), .12, 1)
    gr = 20 + 110 * vsh + rng.normal(0, 6, n)
    sp = -80 + 65 * vsh + rng.normal(0, 6, n)
    pef = 1.9 + 1.4 * vsh + rng.normal(0, .2, n)
    rhob = 2.65 - 1.65 * phi + .05 * vsh + rng.normal(0, .02, n)
    nphi = phi + .08 * vsh + rng.normal(0, .015, n)
    dt = 53 + 160 * phi + 8 * vsh + rng.normal(0, 3, n)
    rt = np.log10(.05 / (phi ** 2 * sw ** 2)) + rng.normal(0, .05, n)
    cali = 8.5 + .5 * vsh + .6 * (rng.random(n) < .03 * (1 + 3 * vsh)) + rng.normal(0, .12, n)
    logk = -1.2 + 14 * phi - 1.4 * vsh + rng.normal(0, .22, n)
    df = pd.DataFrame(dict(GR=gr, SP=sp, PEF=pef, CALI=cali, RHOB=rhob, NPHI=nphi, DT=dt, RT=rt))
    return df, logk, pd.DataFrame(dict(vsh=vsh, phi=phi, sw=sw))


# ------------------------------------------------------------------ helpers
def panel_label(ax, text, x=0.0, y=1.02, fs=13.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, fontweight='semibold', color=INK,
            ha='left', va='bottom')


def corr_heatmap(ax, C, names, fs=11.5, cmap='course_div', vmin=-1, vmax=1, fmt='{:.2f}', text=True):
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    n = len(names)
    ax.set_xticks(range(n), names, rotation=90 if n > 6 else 0)
    ax.set_yticks(range(n), names)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if text:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                v = C[i, j]
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=fs,
                        color='white' if abs(v) > .62 * max(abs(vmin), abs(vmax)) else INK)
    return im


# ======================================================== motivating case
@fig
def case_channels():
    df, L, form = drill_channels()
    names = list(df.columns)
    Z = (df - df.mean()) / df.std()
    t = np.arange(len(df))
    f = plt.figure(figsize=(11.8, 4.35))
    gs = f.add_gridspec(1, 2, width_ratios=[2.05, 1], wspace=.16, left=.085, right=.99, top=.9, bottom=.2)
    ax = f.add_subplot(gs[0])
    gap = 2.3
    for k, name in enumerate(names):
        col = DRIVER_COL[DRILL_DRIVER[name]]
        off = -k * gap
        ax.plot(t, np.clip(Z[name], -2.6, 2.6) * .42 + off, color=col, lw=1.05)
        ax.text(-12, off, name, ha='right', va='center', fontsize=12, color=col, fontweight='medium')
    for b in [170, 370, 560]:
        ax.axvline(b, color=LIGHT, lw=1, ls='--', zorder=0)
    ax.set_xlim(0, len(df))
    ax.set_ylim(-(len(names) - 1) * gap - 1.3, 1.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time (minutes)')
    panel_label(ax, 'Ten standardized channels', x=-.08)
    ax2 = f.add_subplot(gs[1])
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.values.T)))[::-1]
    # components 1-3 follow hardness, pump rate and rotary speed (see the loadings figure); 4 is gas
    cols = [DRIVER_COL['hardness'], DRIVER_COL['pump'], DRIVER_COL['rotary'], DRIVER_COL['none']] + [LIGHT] * 6
    ax2.bar(np.arange(1, 11), ev, color=cols, width=.7)
    for k in range(4):
        ax2.text(k + 1, ev[k] + .08, f'{ev[k]:.1f}', ha='center', va='bottom', fontsize=11.5, color=INK)
    ax2.set_xticks(range(1, 11))
    ax2.set_ylim(0, ev[0] * 1.2)
    clean(ax2, 'Component', 'Eigenvalue')
    ax2.grid(axis='x', visible=False)
    ax2.text(4.3, ev[0] * .92, 'three shared drivers', fontsize=12.5, color=INK, ha='left', va='center')
    ax2.text(4.6, ev[3] + .5, 'gas moves\non its own', fontsize=12.5, color=GOLD, ha='left', va='bottom')
    panel_label(ax2, 'Correlation-matrix eigenvalues', x=-.02)
    driver_legend(f, loc='lower center', bbox_to_anchor=(.5, -.02))
    save('ml4_case_channels', f, 'Synthetic drilling channels from three latent drivers plus noise; '
         'illustrative relationships; seed 4.', kind='computed')


# ============================================== part 1: curse of dimensionality
@fig
def curse_geometry():
    d = np.arange(1, 51)
    f, axs = sub(3, height=3.75)
    ax = axs[0]
    ax.plot(d, 100 * (1 - .9 ** d), color=BLUE)
    for dd, dx, dy, ha in [(3, 7, -6, 'left'), (10, 8, -8, 'left'), (50, -2, -20, 'right')]:
        v = 100 * (1 - .9 ** dd)
        ax.scatter([dd], [v], color=BLUE, s=36, zorder=4)
        ax.annotate(f'd = {dd}: {v:.1f}%' if dd == 50 else f'd = {dd}: {v:.0f}%', (dd, v), xytext=(dx, dy),
                    textcoords='offset points', fontsize=12, color=INK, ha=ha)
    ax.set_ylim(0, 105)
    clean(ax, 'Dimension d', 'Volume within 5% of a face (%)')
    ax.set_title('Points crowd near the faces', fontsize=14)
    ax = axs[1]
    ball = np.pi ** (d / 2) / gamma_fn(d / 2 + 1) * .5 ** d
    ax.semilogy(d, 100 * ball, color=ORANGE)
    for dd, txt in [(3, 'd = 3: 52%'), (10, 'd = 10: 0.25%'), (20, r'd = 20: $2.5\times10^{-6}$%')]:
        v = 100 * np.pi ** (dd / 2) / gamma_fn(dd / 2 + 1) * .5 ** dd
        ax.scatter([dd], [v], color=ORANGE, s=36, zorder=4)
        ax.annotate(txt, (dd, v), xytext=(8, 3), textcoords='offset points', fontsize=12, color=INK)
    ax.set_ylim(1e-18, 400)
    ax.set_yticks([1e2, 1e-4, 1e-10, 1e-16], ['100%', r'$10^{-4}$%', r'$10^{-10}$%', r'$10^{-16}$%'])
    ax.minorticks_off()
    clean(ax, 'Dimension d', 'Share of cube inside the ball')
    ax.set_title('The inscribed ball vanishes', fontsize=14)
    ax = axs[2]
    e10, e01 = .1 ** (1 / d), .01 ** (1 / d)
    ax.plot(d, e10, color=TEAL)
    ax.plot(d, e01, color=PURPLE)
    ax.text(30, .1 ** (1 / 30) + .045, '10% of the data', color=TEAL, fontsize=12, ha='center', va='bottom')
    ax.text(37, .01 ** (1 / 26) - .07, '1% of the data', color=PURPLE, fontsize=12, ha='center', va='top')
    v = .1 ** (1 / 10)
    ax.scatter([10], [v], color=TEAL, s=36, zorder=4)
    ax.annotate('d = 10: 0.79 of\nevery axis', (10, v), xytext=(22, .36), textcoords='data', fontsize=12,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkB=4))
    ax.set_ylim(0, 1.05)
    clean(ax, 'Dimension d', 'Sub-cube edge / axis range')
    ax.set_title('A local box spans most of each axis', fontsize=14)
    save('ml4_curse_geometry', f, 'Exact formulas for uniform data in the unit cube: 1 - 0.9^d, '
         'ball volume pi^(d/2)/Gamma(d/2+1)(1/2)^d, and edge r^(1/d).', kind='computed')


@fig
def knn_noise():
    from sklearn.datasets import make_classification
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    extra = [0, 1, 2, 5, 10, 20, 50, 100]
    res = {'all': [], 'sel': []}
    for m in extra:
        a, s = [], []
        for rep in range(5):
            X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                       n_clusters_per_class=1, class_sep=1.0, random_state=rep)
            rng = np.random.default_rng(100 + rep)
            Xn = np.c_[X, rng.normal(size=(len(X), m))]
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            knn = make_pipeline(StandardScaler(), KNeighborsClassifier(15))
            a.append(cross_val_score(knn, Xn, y, cv=cv).mean())
            sel = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=2), KNeighborsClassifier(15))
            s.append(cross_val_score(sel, Xn, y, cv=cv).mean())
        res['all'].append(np.mean(a))
        res['sel'].append(np.mean(s))
    print('  knn_noise all:', np.round(res['all'], 3), 'sel:', np.round(res['sel'], 3))
    f, axs = half()
    ax = axs[0]
    x = np.arange(len(extra))
    ax.plot(x, res['all'], '-o', color=BLUE, ms=6)
    ax.plot(x, res['sel'], '-o', color=TEAL, ms=6)
    endlabel(ax, x[-1], res['all'][-1], 'all features', BLUE, dx=-4, dy=-16, ha='right')
    endlabel(ax, x[-1], res['sel'][-1], 'best 2 selected\ninside each fold', TEAL, dx=-4, dy=22, ha='right')
    ax.axhline(.5, color=GREY, lw=1, ls=':')
    ax.text(0, .515, 'chance', fontsize=12, color=GREY)
    ax.set_xticks(x, [str(m) for m in extra])
    ax.set_ylim(.45, 1.0)
    clean(ax, 'Irrelevant features added', 'Cross-validated accuracy (15-NN)')
    save('ml4_knn_noise', f, 'Synthetic two-class data with two informative features plus Gaussian noise features; '
         '5 datasets (seeds 0-4) x 5-fold CV.', kind='computed')


@fig
def select_vs_extract():
    f, ax = canvas(11.8, 4.2)
    names = ['GR', 'RHOB', 'NPHI', 'DT', 'RT']
    keep = [0, 1, 4]
    cw, ch, nrow = .54, .34, 5

    def table(x, ytop, cols, colors):
        for j, (nm, col) in enumerate(zip(cols, colors)):
            faded = col == LIGHT
            box(ax, x + j * cw + .02, ytop - .42, cw - .04, .38, nm, col, fill='tint' if faded else 'solid',
                fs=11.5, weight='semibold', radius=.04, text_color=MUTED if faded else 'white')
            for i in range(nrow):
                ax.add_patch(Rectangle((x + j * cw + .02, ytop - .48 - (i + 1) * ch), cw - .04, ch - .05,
                                       fc='#F1F4F7' if faded else tint(col, .8), ec='none', zorder=2))
        return x + len(cols) * cw

    label(ax, 2.6, 3.95, 'Feature selection', fs=15, weight='semibold', color=TEAL)
    x_end = table(.2, 3.55, names, [TEAL if j in keep else LIGHT for j in range(5)])
    arrow(ax, (x_end + .1, 2.45), (x_end + .6, 2.45), color=INK, lw=1.6)
    table(x_end + .68, 3.55, [names[j] for j in keep], [TEAL] * 3)
    chip(ax, 1.35, .8, 'keeps names and units', TEAL, fs=12)
    chip(ax, 3.75, .8, 'unused sensors can go', TEAL, fs=12)
    label(ax, 2.6, .3, 'A subset of the original columns', fs=12.5, color=MUTED)
    ax.plot([5.3, 5.3], [.25, 4.0], color=RULE, lw=1.4)
    label(ax, 8.4, 3.95, 'Feature extraction', fs=15, weight='semibold', color=BLUE)
    x_end = table(5.6, 3.55, names, [GREY] * 5)
    arrow(ax, (x_end + .08, 2.45), (x_end + .46, 2.45), color=INK, lw=1.6)
    b = box(ax, x_end + .52, 2.0, .92, .9, 'Z = XW', BLUE, fill='tint', fs=13.5, weight='semibold')
    arrow(ax, b.r, (b.r[0] + .38, 2.45), color=INK, lw=1.6)
    table(b.r[0] + .44, 3.55, ['Z1', 'Z2'], [BLUE, BLUE])
    chip(ax, 6.95, .8, 'every column contributes', BLUE, fs=12)
    chip(ax, 9.75, .8, 'new axes need interpretation', BLUE, fs=12)
    label(ax, 8.4, .3, 'New columns built from all of them', fs=12.5, color=MUTED)
    save('ml4_select_vs_extract', f, 'Schematic.', kind='schematic')


# ================================================== part 2: feature selection
def log_features_with_noise(n_noise: int = 6, seed: int = 5):
    df, logk, lat = log_suite()
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(rng.normal(size=(len(df), n_noise)), columns=[f'Noise {i}' for i in range(1, n_noise + 1)])
    return pd.concat([df, noise], axis=1), logk


@fig
def selection_families():
    f, ax = canvas(11.8, 4.3)
    lanes = [('Filter', TEAL, 3.35), ('Wrapper', PURPLE, 1.95), ('Embedded', BLUE, .55)]
    w, h = 1.75, .82
    for name, col, y in lanes:
        label(ax, .12, y + h / 2, name, fs=15, weight='semibold', color=col, ha='left')
    # filter
    y = 3.35
    b0 = box(ax, 1.45, y, w, h, 'All features', GREY, fs=13)
    b1 = box(ax, 3.6, y, 2.1, h, 'Score each feature', TEAL, fs=13, sub='variance · |r| · MI', sub_fs=11.5)
    b2 = box(ax, 6.1, y, w, h, 'Keep top k', TEAL, fs=13)
    b3 = box(ax, 8.25, y, w, h, 'Fit model', GREY, fill='white', fs=13)
    for a, b in [(b0, b1), (b1, b2), (b2, b3)]:
        arrow(ax, a.r, b.l, color=INK, lw=1.5, shrink=1)
    # wrapper
    y = 1.95
    c0 = box(ax, 1.45, y, w, h, 'All features', GREY, fs=13)
    c1 = box(ax, 3.6, y, 2.1, h, 'Candidate subset', PURPLE, fs=13)
    c2 = box(ax, 6.1, y, w, h, 'Fit + CV score', PURPLE, fs=13, fill='solid')
    c3 = box(ax, 8.25, y, w, h, 'Best subset', PURPLE, fs=13)
    arrow(ax, c0.r, c1.l, color=INK, lw=1.5, shrink=1)
    arrow(ax, c1.r, c2.l, color=INK, lw=1.5, shrink=1)
    arrow(ax, c2.r, c3.l, color=INK, lw=1.5, shrink=1)
    arrow(ax, (c2.x + .55, c2.y), (c1.x + 1.55, c1.y), color=PURPLE, lw=1.5, rad=-.55)
    label(ax, 6.95, 1.66, 'drop or add a feature, repeat', fs=11.5, color=PURPLE, ha='left')
    # embedded
    y = .55
    e0 = box(ax, 1.45, y, w, h, 'All features', GREY, fs=13)
    e1 = box(ax, 3.6, y, 2.1, h, 'Fit the model', BLUE, fs=13, sub='L1 penalty · tree splits', sub_fs=11.5)
    e2 = box(ax, 6.1, y, w, h, 'Weak → zero', BLUE, fs=13, sub='or low importance', sub_fs=11.5)
    e3 = box(ax, 8.25, y, w, h, 'Selected set', BLUE, fs=13)
    for a, b in [(e0, e1), (e1, e2), (e2, e3)]:
        arrow(ax, a.r, b.l, color=INK, lw=1.5, shrink=1)
    for (txt1, txt2), col, yy in [(('cheap', 'one feature at a time'), TEAL, 3.35),
                                  (('expensive', 'can overfit the search'), PURPLE, 1.95),
                                  (('moderate', 'tied to one model'), BLUE, .55)]:
        chip(ax, 10.85, yy + .56, txt1, col, fs=12)
        label(ax, 10.85, yy + .18, txt2, fs=11.5, color=MUTED)
    save('ml4_selection_families', f, 'Schematic.', kind='schematic')


@fig
def corr_redundancy():
    df, logk, lat = log_suite()
    C = df.corr().values
    names = list(df.columns)
    f, axs = half()
    ax = axs[0]
    corr_heatmap(ax, C, names, fs=11.5)
    for (a, b), col in [((0, 2), GOLD), ((4, 6), TEAL)]:
        ax.add_patch(Rectangle((a - .5, a - .5), b - a + 1, b - a + 1, fc='none', ec=col, lw=3, zorder=5))
    ax.set_xticks(range(len(names)), names, rotation=0, fontsize=11.5)
    ax.text(1, -.62, 'shale block', ha='center', va='bottom', fontsize=12.5, color=GOLD, fontweight='semibold')
    ax.text(5, -.62, 'porosity block', ha='center', va='bottom', fontsize=12.5, color=TEAL, fontweight='semibold')
    ax.set_ylim(7.5, -1.1)
    save('ml4_corr_redundancy', f, 'Pearson correlations of a synthetic log suite (seed 11); blocks outlined where |r| > 0.7.',
         kind='computed')


@fig
def mutual_info():
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(7)
    n = 800
    rpm = rng.uniform(60, 180, n)
    wob = rng.uniform(5, 25, n)
    torque = .55 * wob + .01 * rpm + rng.normal(0, 1.0, n)
    flow = rng.uniform(400, 900, n)
    mw = rng.uniform(9, 14, n)
    vib = 2.0 * np.exp(-((rpm - 120) / 14) ** 2) + .045 * wob + rng.normal(0, .25, n)
    F = pd.DataFrame({'RPM': rpm, 'WOB': wob, 'Torque': torque, 'Flow': flow, 'Mud weight': mw})
    r = F.corrwith(pd.Series(vib)).abs().values
    mi = mutual_info_regression(F.values, vib, random_state=0, n_neighbors=3)
    print('  mutual_info |r|', np.round(r, 3), 'MI', np.round(mi, 3))
    f = plt.figure(figsize=HALF, layout='constrained')
    gs = f.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax = f.add_subplot(gs[0, :])
    ax.scatter(rpm, vib, s=9, color=BLUE, alpha=.55, edgecolors='none')
    clean(ax, 'RPM', 'Vibration')
    ax.set_title('Vibration peaks in an RPM band', fontsize=13.5, loc='left')
    y = np.arange(len(F.columns))
    for k, (vals, col, title, xl) in enumerate([(r, GREY, '|Pearson r|', (0, .5)),
                                                 (mi, TEAL, 'Mutual information', (0, .75))]):
        a = f.add_subplot(gs[1, k])
        a.barh(y, vals, color=[col if i else (ORANGE if k == 0 else TEAL) for i in range(len(y))], height=.62)
        for i, v in enumerate(vals):
            a.text(v + xl[1] * .02, i, f'{v:.2f}', va='center', fontsize=11.5, color=INK)
        a.set_yticks(y, list(F.columns) if k == 0 else [])
        a.invert_yaxis()
        a.set_xlim(*xl)
        a.set_title(title, fontsize=13.5, loc='left')
        a.grid(axis='x', alpha=1)
        a.set_axisbelow(True)
        a.tick_params(axis='y', length=0)
    save('ml4_mutual_info', f, 'Synthetic drilling example (seed 7): vibration = resonance peak in RPM + weak WOB effect + noise; '
         'MI by the k-NN estimator.', kind='computed')


@fig
def rfe_curve():
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    X, logk = log_features_with_noise()
    Xs = StandardScaler().fit_transform(X)
    rfe = RFECV(LinearRegression(), step=1, cv=KFold(5), scoring='r2', min_features_to_select=1).fit(Xs, logk)
    m = rfe.cv_results_['mean_test_score']
    s = rfe.cv_results_['std_test_score']
    k = np.arange(1, len(m) + 1)
    best = int(np.argmax(m))
    se = s[best] / np.sqrt(5)
    one_se = int(np.argmax(m >= m[best] - se))
    print('  rfe', np.round(m, 3), 'best', k[best], 'one-SE', k[one_se], 'support', list(X.columns[rfe.support_]))
    f, axs = half()
    ax = axs[0]
    ax.fill_between(k, m - s, m + s, color=BLUE, alpha=.14, lw=0)
    ax.plot(k, m, '-o', color=BLUE, ms=5)
    ax.scatter([k[best]], [m[best]], s=110, facecolors='none', edgecolors=ORANGE, lw=2, zorder=5)
    ax.annotate(f'best mean: {k[best]} features', (k[best], m[best]), xytext=(k[best] + 1.2, .922),
                fontsize=12, color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1, shrinkB=7))
    ax.scatter([k[one_se]], [m[one_se]], s=110, facecolors='none', edgecolors=TEAL, lw=2, zorder=5)
    ax.annotate(f'within one SE of best:\n{k[one_se]} features', (k[one_se], m[one_se]), xytext=(k[one_se] + .6, .75),
                fontsize=12, color=TEAL, arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1.1, shrinkB=7))
    ax.set_xticks(k)
    ax.set_ylim(.6, .95)
    clean(ax, 'Number of features kept', 'Cross-validated R² (5 depth blocks)')
    ax.text(14, .64, '8 logs + 6 noise channels', ha='right', fontsize=12, color=MUTED)
    save('ml4_rfe_curve', f, 'RFECV with linear regression on a synthetic log suite plus 6 noise channels (seeds 11, 5); '
         'band = ±1 SD across folds.', kind='computed')


@fig
def l1_path():
    from sklearn.linear_model import lasso_path, LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    X, logk = log_features_with_noise()
    Xs = StandardScaler().fit_transform(X)
    y = logk - logk.mean()
    alphas, coefs, _ = lasso_path(Xs, y, n_alphas=120, eps=1e-3)
    a_cv = LassoCV(cv=KFold(5)).fit(Xs, logk).alpha_
    names = list(X.columns)
    colors = {'RHOB': BLUE, 'GR': GOLD, 'SP': ORANGE, 'DT': TEAL, 'NPHI': PURPLE}
    f, axs = half()
    ax = axs[0]
    for j, nm in enumerate(names):
        if nm in colors:
            continue
        ax.semilogx(alphas, coefs[j], color=LIGHT if nm.startswith('Noise') else '#9AA9B8', lw=1.4, zorder=2)
    for j, nm in enumerate(names):
        if nm in colors:
            ax.semilogx(alphas, coefs[j], color=colors[nm], lw=2.4, zorder=3)
            ax.text(alphas[-1] * .8, coefs[j, -1], nm, color=colors[nm], fontsize=12, fontweight='semibold',
                    va='center', ha='left')
    ax.axvline(a_cv, color=INK, lw=1.2, ls='--')
    ax.text(a_cv * .85, -.52, 'α chosen by CV', fontsize=12, color=INK, va='center', ha='left')
    ax.axhline(0, color=GREY, lw=.8)
    ax.invert_xaxis()
    ax.set_xlim(alphas[0] * 1.3, alphas[-1] * .15)
    clean(ax, 'L1 penalty α (log scale, weaker to the right)', 'Standardized coefficient')
    ax.text(alphas[0] * .95, .06, 'grey: CALI, PEF, RT\nlight: 6 noise channels', fontsize=11.5, color=MUTED, va='bottom')
    save('ml4_l1_path', f, 'Lasso path (scikit-learn lasso_path) on the synthetic log suite plus 6 noise channels; '
         'target log permeability.', kind='computed')


@fig
def tree_importance():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    X, logk = log_features_with_noise()
    names = list(X.columns)
    r = np.abs([np.corrcoef(X[c], logk)[0, 1] for c in names])
    rf = RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X, logk)
    imp = rf.feature_importances_
    keep = SelectFromModel(rf, prefit=True, threshold='mean').get_support()
    print('  tree |r|', dict(zip(names, r.round(2))), 'imp', dict(zip(names, imp.round(3))),
          'thr', round(imp.mean(), 3), 'kept', [n for n, k in zip(names, keep) if k])
    f, axs = half(2)
    y = np.arange(len(names))
    base = [GREY if not n.startswith('Noise') else LIGHT for n in names]
    for ax, vals, title, xl in [(axs[0], r, 'Filter view: |r| with target', (0, 1.05)),
                                (axs[1], imp, 'Forest impurity importance', (0, 1.05))]:
        cols = list(base)
        if ax is axs[1]:
            cols = [TEAL if k else c for k, c in zip(keep, cols)]
        else:
            top3 = set(np.argsort(vals)[-3:])
            cols = [TEAL if i in top3 else c for i, c in enumerate(cols)]
        ax.barh(y, vals, color=cols, height=.66)
        for i, v in enumerate(vals):
            if names[i] in ('GR', 'RHOB', 'NPHI', 'DT'):
                x_txt = max(v, imp.mean()) + .03 if ax is axs[1] else v + .03
                ax.text(x_txt, i, f'{v:.2f}', va='center', fontsize=11.5, color=INK)
        ax.set_yticks(y, names if ax is axs[0] else [])
        ax.set_ylim(len(names) - .5, -.5)
        ax.set_xlim(*xl)
        ax.set_title(title, fontsize=13, loc='left')
        clean(ax)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
    for a, txt in [(axs[0], 'teal: top 3 by |r|,\nkept by a filter'), (axs[1], 'teal: kept by\nSelectFromModel')]:
        a.text(1.03, 10.4, txt, color=TEAL, fontsize=11.5, ha='right', va='center',
               bbox=dict(boxstyle='square,pad=.15', fc='white', ec='none'))
    axs[1].axvline(imp.mean(), color=ORANGE, lw=1.4, ls='--')
    axs[1].text(imp.mean() + .04, 12.9, 'SelectFromModel\nthreshold (mean)', color=ORANGE, fontsize=11.5,
                va='center')
    save('ml4_tree_importance', f, 'Random forest (300 trees, seed 0) on the synthetic log suite plus six noise channels '
         '(seeds 11, 5); target log permeability.', kind='computed')


@fig
def selection_leakage():
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    ps = [10, 30, 100, 300, 1000, 3000]
    W, R = [], []
    for p in ps:
        w, r = [], []
        for rep in range(30):
            rng = np.random.default_rng(2000 + rep)
            X = rng.normal(size=(60, p))
            y = rng.permutation(np.repeat([0, 1], 30))
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            Xs = SelectKBest(f_classif, k=10).fit_transform(X, y)
            w.append(cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=cv).mean())
            pipe = make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression(max_iter=1000))
            r.append(cross_val_score(pipe, X, y, cv=cv).mean())
        W.append(w)
        R.append(r)
    W, R = np.array(W), np.array(R)
    print('  leakage wrong', W.mean(1).round(3), 'right', R.mean(1).round(3))
    f, axs = half()
    ax = axs[0]
    for M, col in [(W, ORANGE), (R, TEAL)]:
        m, sd = M.mean(1), M.std(1)
        ax.fill_between(ps, m - sd, m + sd, color=col, alpha=.14, lw=0)
        ax.plot(ps, m, '-o', color=col, ms=6)
    ax.text(3000, W.mean(1)[-1] + .06, 'select on all rows,\nthen cross-validate', color=ORANGE, fontsize=12.5,
            ha='right', va='bottom', fontweight='semibold')
    ax.text(3000, R.mean(1)[-1] - .08, 'select inside each\ntraining fold', color=TEAL, fontsize=12.5,
            ha='right', va='top', fontweight='semibold')
    ax.axhline(.5, color=GREY, lw=1, ls=':')
    ax.text(10.5, .485, 'chance', fontsize=12, color=GREY, va='top')
    ax.set_xscale('log')
    ax.set_xticks(ps, [str(p) for p in ps])
    ax.minorticks_off()
    ax.set_ylim(.25, 1.02)
    clean(ax, 'Candidate features screened (pure noise)', 'Cross-validated accuracy')
    save('ml4_selection_leakage', f, 'Pure-noise data: 60 rows, random labels, 30 datasets per point (seeds 2000-2029); '
         'SelectKBest(k = 10) + logistic regression; 5-fold CV; bands = ±1 SD.', kind='computed')


@fig
def selection_stability():
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    X, logk = log_features_with_noise()
    names = list(X.columns)
    S = []
    for tr, te in KFold(10).split(X):
        Xt = StandardScaler().fit(X.iloc[tr]).transform(X.iloc[tr])
        S.append(LassoCV(cv=KFold(5)).fit(Xt, logk[tr]).coef_ != 0)
    S = np.array(S).T.astype(float)            # features x folds
    jac = [((S[:, i] > 0) & (S[:, j] > 0)).sum() / ((S[:, i] > 0) | (S[:, j] > 0)).sum()
           for i in range(S.shape[1]) for j in range(i + 1, S.shape[1])]
    freq = S.mean(1)
    print('  stability freq', dict(zip(names, freq.round(2))), 'jaccard', round(float(np.mean(jac)), 3))
    f = plt.figure(figsize=HALF, layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[2.3, 1])
    ax = f.add_subplot(gs[0])
    for i in range(S.shape[0]):                  # one vector cell per feature and fold, white gaps
        for j in range(S.shape[1]):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=TEAL if S[i, j] else '#EEF2F6', ec='white', lw=1.5))
    ax.set_xlim(-.5, S.shape[1] - .5)
    ax.set_ylim(len(names) - .5, -.5)
    ax.set_xticks(range(10), [str(i + 1) for i in range(10)])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    ax.tick_params(which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('Fold (depth block left out)')
    ax2 = f.add_subplot(gs[1])
    y = np.arange(len(names))
    ax2.barh(y, freq, color=[TEAL if v == 1 else '#8FC6CB' if v > 0 else LIGHT for v in freq], height=.7)
    for i, v in enumerate(freq):
        ax2.text(v + .04, i, f'{v:.0%}' if v < 1 else '100%', va='center', fontsize=11, color=INK)
    ax2.set_ylim(len(names) - .5, -.5)
    ax2.set_yticks([])
    ax2.set_xlim(0, 1.45)
    ax2.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    ax2.set_xlabel('Selection rate')
    for sp in ['left']:
        ax2.spines[sp].set_visible(False)
    f.suptitle(f'Mean pairwise Jaccard overlap of selected sets: {np.mean(jac):.2f}', fontsize=12.5, color=MUTED)
    save('ml4_selection_stability', f, 'LassoCV refitted in each of 10 contiguous depth folds; synthetic log suite plus '
         '6 noise channels (seeds 11, 5).', kind='computed')


# ======================================================= part 3: PCA in depth
FORM_COL = [BLUE, ORANGE, TEAL]
FORM_NAMES = ['Formation A', 'Formation B', 'Formation C']


def drill_std():
    from sklearn.preprocessing import StandardScaler
    df, L, form = drill_channels()
    return df, StandardScaler().fit_transform(df), form


@fig
def cov_eigen():
    df, Z, form = drill_std()
    X = Z[:, [0, 1]]                                   # WOB, Torque (standardized)
    C = np.cov(X.T, ddof=0)
    vals, vecs = np.linalg.eigh(C)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    print('  cov', np.round(C, 3), 'eig', np.round(vals, 3))
    f, axs = half()
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, color=GREY, alpha=.35, edgecolors='none')
    gaussian_blob(ax, (0, 0), C, color=BLUE, n_std=(1, 2), alpha=.10, lw=1.3)
    for k, (col, nm) in enumerate([(BLUE, 'PC1'), (ORANGE, 'PC2')]):
        v = vecs[:, k] * (1 if vecs[0, k] > 0 else -1)
        L_ = 2 * np.sqrt(vals[k])
        ax.annotate('', xy=v * L_, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6,
                                                                   mutation_scale=18), zorder=5)
        off = np.array([.2, .12]) if k == 0 else np.array([.2, -.5])
        ax.text(*(v * L_ + off), f'{nm}: λ = {vals[k]:.2f}', color=col, fontsize=13, fontweight='semibold',
                ha='left', bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85), zorder=6)
    x0, y0 = -3.3, 2.55
    ax.text(x0, y0, 'C =', fontsize=14, va='center', ha='left', color=INK)
    for i in range(2):
        for j in range(2):
            ax.text(x0 + 1.05 + j * .78, y0 + .28 - i * .56, f'{C[i, j]:.2f}', fontsize=13, ha='center',
                    va='center', color=INK)
    for side in (-1, 1):
        xb = x0 + 1.05 + .39 + side * .83
        ax.plot([xb - side * .08, xb, xb, xb - side * .08], [y0 + .58, y0 + .58, y0 - .58, y0 - .58],
                color=INK, lw=1.3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    clean(ax, 'WOB (standardized)', 'Torque (standardized)')
    save('ml4_cov_eigen', f, 'Synthetic drilling channels (seed 4); eigen-decomposition of the 2 x 2 covariance of '
         'standardized WOB and torque; arrows = 2 sqrt(lambda).', kind='computed')


@fig
def svd_blocks():
    f, ax = canvas(6.3, 4.9)
    y0 = 2.35
    H = 2.1
    # X = U S V^T
    bx = box(ax, .2, y0, .95, H, '', GREY, fill='tint', radius=.03)
    label(ax, .675, y0 + H / 2, r'$X_c$', fs=16, weight='semibold')
    label(ax, .675, y0 - .22, 'n × d', fs=12, color=MUTED)
    label(ax, 1.42, y0 + H / 2, '=', fs=20)
    ux = 1.7
    box(ax, ux, y0, .95, H, '', BLUE, fill='tint', radius=.03)
    box(ax, ux, y0, .38, H, '', BLUE, fill='solid', radius=.03, alpha=.9)
    label(ax, ux + .66, y0 + H / 2, r'$U$', fs=16, weight='semibold')
    label(ax, ux + .475, y0 - .22, 'n × d', fs=12, color=MUTED)
    sx = 2.95
    box(ax, sx, y0 + H - .95, .95, .95, '', GOLD, fill='tint', radius=.03)
    for i in range(4):
        ax.add_patch(Circle((sx + .12 + i * .235, y0 + H - .12 - i * .235), .075 - i * .012,
                            color=GOLD, zorder=4))
    label(ax, sx + .475, y0 + H - 1.17, 'Σ  (d × d)', fs=12, color=MUTED)
    vx = 4.2
    box(ax, vx, y0 + H - .95, .95, .95, '', TEAL, fill='tint', radius=.03)
    box(ax, vx, y0 + H - .38, .95, .38, '', TEAL, fill='solid', radius=.03, alpha=.9)
    label(ax, vx + .475, y0 + H - .64, r'$V^{T}$', fs=16, weight='semibold')
    label(ax, vx + .475, y0 + H - 1.17, 'd × d', fs=12, color=MUTED)
    label(ax, 5.3, y0 + H - .45, 'rows:\nprincipal\ndirections', fs=12, color=TEAL, ha='left', va='center')
    label(ax, sx + .475, y0 + .35, 'singular values\n' + r'$\sigma_1 \geq \sigma_2 \geq \ldots$', fs=12, color=GOLD)
    # scores
    y1 = .35
    label(ax, .2, 1.78, 'Keep the first k components (shaded):', fs=13, weight='semibold', ha='left')
    box(ax, .2, y1, .5, 1.1, r'$Z$', BLUE, fill='solid', radius=.03, fs=15, weight='semibold')
    label(ax, .45, y1 - .2, 'n × k', fs=12, color=MUTED)
    label(ax, 1.05, y1 + .72, r'$Z = X_c V_k = U_k \Sigma_k$', fs=16, ha='left')
    label(ax, 1.05, y1 + .22, 'scores; variance of score j: ' + r'$\lambda_j = \sigma_j^2/(n-1)$', fs=12.5,
          color=MUTED, ha='left')
    save('ml4_svd_blocks', f, 'Schematic of the thin singular value decomposition of a centred data matrix.',
         kind='schematic')


@fig
def scaling_pc1():
    from sklearn.decomposition import PCA
    df, Z, form = drill_std()
    names = list(df.columns)
    p_raw = PCA().fit(df - df.mean())
    p_std = PCA().fit(Z)
    f, axs = plt.subplots(1, 2, figsize=(11.8, 4.25), sharey=True, layout='constrained')
    y = np.arange(len(names))
    cols = [DRIVER_COL[DRILL_DRIVER[n]] for n in names]
    for ax, p, title in [(axs[0], p_raw, 'Raw units (covariance matrix)'),
                         (axs[1], p_std, 'Standardized (correlation matrix)')]:
        w = p.components_[0] * np.sign(p.components_[0][np.argmax(np.abs(p.components_[0]))])
        ax.barh(y, w, color=cols, height=.68)
        ax.axvline(0, color=GREY, lw=.8)
        ax.set_xlim(-.62, 1.08)
        clean(ax, 'PC1 loading')
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
        ax.set_title(f'{title}\nPC1 = {p.explained_variance_ratio_[0]:.0%} of variance', fontsize=14, loc='left')
    axs[0].set_yticks(y, names)
    axs[0].set_ylim(len(names) - .5, -.5)
    axs[0].annotate('SPP has the largest\nnumbers (psi), so it\nbecomes "PC1"', (.96, 4.3), xytext=(.3, 8.3),
                    fontsize=12, color=INK, va='center', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    driver_legend(f, loc='outside lower center')
    save('ml4_scaling_pc1', f, 'Synthetic drilling channels in their units (seed 4): SPP psi, flow gpm, WOB klbf, etc.; '
         'first principal component before and after standardization.', kind='computed')


@fig
def choose_k():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    df, Z, form = drill_std()
    p = PCA().fit(Z)
    ev = p.explained_variance_ratio_
    k = np.arange(1, 11)
    tr, va = df.iloc[:430], df.iloc[430:]
    sc = StandardScaler().fit(tr)
    Ztr, Zva = sc.transform(tr), sc.transform(va)
    etr, eva = [], []
    for kk in k:
        m = PCA(kk).fit(Ztr)
        etr.append(((Ztr - m.inverse_transform(m.transform(Ztr))) ** 2).mean())
        eva.append(((Zva - m.inverse_transform(m.transform(Zva))) ** 2).mean())
    print('  choose_k evr', np.round(ev, 3), 'cum', np.round(np.cumsum(ev), 3), 'val', np.round(eva, 3))
    f, axs = sub(3, height=3.9)
    ax = axs[0]
    ax.bar(k, ev, color=[BLUE] * 4 + [LIGHT] * 6, width=.7)
    ax.plot(k, ev, '-o', color=INK, lw=1.2, ms=4)
    ax.annotate('elbow after 4', (4.3, ev[3]), xytext=(6, ev[1]), fontsize=12.5, color=INK,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    ax.set_xticks(k)
    clean(ax, 'Component', 'Explained variance ratio')
    ax.set_title('Scree plot', fontsize=14)
    ax = axs[1]
    cum = np.cumsum(ev)
    ax.plot(k, cum, '-o', color=BLUE, ms=5)
    for thr, col in [(.9, GOLD), (.95, PURPLE)]:
        kk = int(np.argmax(cum >= thr)) + 1
        ax.axhline(thr, color=col, lw=1.2, ls='--')
        ax.scatter([kk], [cum[kk - 1]], s=90, facecolors='none', edgecolors=col, lw=2, zorder=5)
        if thr > .92:
            ax.text(1.1, thr + .012, f'{thr:.0%} → k = {kk}', color=col, fontsize=12.5, ha='left', va='bottom')
        else:
            ax.text(10, thr - .015, f'{thr:.0%} → k = {kk}', color=col, fontsize=12.5, ha='right', va='top')
    ax.set_ylim(.3, 1.03)
    ax.set_xticks(k)
    clean(ax, 'Components kept', 'Cumulative variance')
    ax.set_title('Variance threshold', fontsize=14)
    ax = axs[2]
    ax.plot(k, etr, '-o', color=BLUE, ms=5)
    ax.plot(k, eva, '-o', color=TEAL, ms=5)
    ax.text(5.0, .6, 'training period', color=BLUE, fontsize=12.5, ha='left', va='center')
    ax.text(5.0, .52, 'later validation period', color=TEAL, fontsize=12.5, ha='left', va='center')
    ax.set_xticks(k)
    clean(ax, 'Components kept', 'Reconstruction MSE (std. units)')
    ax.set_title('Held-out reconstruction error', fontsize=14)
    save('ml4_choose_k', f, 'Synthetic drilling channels (seed 4); scree and cumulative variance from all 720 minutes; '
         'held-out error from PCA fitted on minutes 0-429 and validated on 430-719.', kind='computed')


@fig
def loadings_heatmap():
    from sklearn.decomposition import PCA
    df, Z, form = drill_std()
    p = PCA(4).fit(Z)
    W = p.components_.copy()
    for i in range(4):
        W[i] *= np.sign(W[i][np.argmax(np.abs(W[i]))])
    names = list(df.columns)
    f, axs = half()
    ax = axs[0]
    im = ax.imshow(W.T, cmap='course_div', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    for i in range(W.shape[1]):
        for j in range(4):
            v = W[j, i]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=11.5,
                    color='white' if abs(v) > .55 else INK)
    ax.set_xticks(range(4), [f'PC{j + 1}\n{p.explained_variance_ratio_[j]:.0%}' for j in range(4)])
    ax.xaxis.tick_top()
    ax.set_yticks(range(10), names)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = f.colorbar(im, ax=ax, shrink=.75, pad=.03)
    cb.set_label('Loading (unit-length direction)', fontsize=12)
    cb.ax.tick_params(labelsize=11)
    save('ml4_loadings_heatmap', f, 'Synthetic drilling channels (seed 4), standardized; rows of PCA components_ '
         '(signs fixed so the largest entry is positive).', kind='computed')


@fig
def biplot():
    from sklearn.decomposition import PCA
    df, Z, form = drill_std()
    p = PCA(2).fit(Z)
    S = p.transform(Z)
    W = p.components_ * 5.0
    names = list(df.columns)
    f, axs = half()
    ax = axs[0]
    for k in range(3):
        m = form == k
        ax.scatter(S[m, 0], S[m, 1], s=11, color=FORM_COL[k], alpha=.4, edgecolors='none')
    for txt, xy, k in [('Formation A', (-4.6, -1.3), 0), ('Formation B', (-1.0, -4.6), 1), ('Formation C', (3.0, 3.6), 2)]:
        ax.text(*xy, txt, color=FORM_COL[k], fontsize=12.5, fontweight='semibold', ha='left')
    groups = [(['SPP', 'Flow', 'ECD'], 'SPP · Flow · ECD', (0, .35)),
              (['WOB', 'DiffP', 'Torque'], 'WOB · DiffP\n· Torque', (1.3, -.5)),
              (['Hookload'], 'Hookload', (-.55, .35)), (['ROP'], 'ROP', (-.25, .3))]
    for members, txt, (dx, dy) in groups:
        for n in members:
            v = W[:, names.index(n)]
            ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.7), zorder=5)
        v = np.mean([W[:, names.index(n)] for n in members], axis=0)
        ax.text(v[0] + dx, v[1] + dy, txt, color=INK, fontsize=12, fontweight='semibold', zorder=6, ha='center',
                va='center', bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.85))
    ax.axhline(0, color=LIGHT, lw=.8, zorder=0)
    ax.axvline(0, color=LIGHT, lw=.8, zorder=0)
    clean(ax, f'PC1 score ({p.explained_variance_ratio_[0]:.0%})', f'PC2 score ({p.explained_variance_ratio_[1]:.0%})',
          grid=False)
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 4.4)
    save('ml4_biplot', f, 'Synthetic drilling channels (seed 4); scores coloured by formation after fitting; arrows = '
         'loadings x 5 (display scale); RPM and gas load on PC3-PC4.', kind='computed')


@fig
def recon_anomaly():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    df, L, form = drill_channels()
    df = df.copy()
    t = np.arange(len(df))
    ramp = np.clip((t - 520) / 70, 0, 1)                # washout: SPP falls at constant flow
    df['SPP'] = df['SPP'] - 450 * ramp
    spike = (t >= 640) & (t < 646)                      # hook-load sensor glitch
    df.loc[spike, 'Hookload'] += 22
    tr = t < 400
    sc = StandardScaler().fit(df[tr])
    Z = sc.transform(df)
    m = PCA(4).fit(Z[tr])
    spe = ((Z - m.inverse_transform(m.transform(Z))) ** 2).sum(1)
    thr = np.quantile(spe[tr], .99)
    print('  recon_anomaly thr', round(thr, 3), 'max test', round(spe[~tr].max(), 2),
          'first alarm after 520:', t[(t > 520) & (spe > thr)][:3])
    f = plt.figure(figsize=(11.8, 4.2), layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[1, 1.35])
    a1 = f.add_subplot(gs[0])
    a1.plot(t, Z[:, 4], color=TEAL, lw=1.1)
    a1.plot(t, Z[:, 5], color=GREY, lw=1.1, alpha=.8)
    a1.text(712, Z[-60:, 4].min() - .35, 'SPP', color=TEAL, fontsize=12, fontweight='semibold', ha='right', va='top')
    a1.text(712, Z[-60:, 5].max() + .35, 'Flow', color=GREY, fontsize=12, fontweight='semibold', ha='right',
            va='bottom')
    a1.axvspan(0, 400, color=BLUE, alpha=.06, lw=0)
    a1.text(8, 3.5, 'training period (normal operation)', fontsize=12, color=BLUE, va='top')
    a1.set_xlim(0, 720)
    a1.set_ylim(-5.1, 3.8)
    a1.set_yticks([-2.5, 0, 2.5])
    clean(a1, None, 'Std. value')
    a1.tick_params(labelbottom=False)
    a2 = f.add_subplot(gs[1], sharex=a1)
    a2.fill_between(t, 0, spe, color=BLUE, alpha=.25, lw=0)
    a2.plot(t, spe, color=BLUE, lw=1)
    a2.axhline(thr, color=ORANGE, lw=1.4, ls='--')
    a2.text(8, thr * 1.15, 'threshold: 99th percentile of training SPE', color=ORANGE, fontsize=12, va='bottom')
    a2.axvspan(0, 400, color=BLUE, alpha=.06, lw=0)
    hit = spe > thr
    a2.scatter(t[hit & ~tr], spe[hit & ~tr], s=10, color=ORANGE, zorder=4)
    a2.set_yscale('log')
    a2.set_ylim(.05, spe.max() * 6)
    a2.annotate('SPP drifts down at\nconstant flow (washout?)', (590, spe[590]), xytext=(418, spe.max() * 4),
                fontsize=12, color=INK, va='top', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    a2.annotate('hook-load glitch', (644, spe[641:646].max()), xytext=(590, spe.max() * 4.2), fontsize=12,
                color=INK, va='top', ha='left', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    clean(a2, 'Time (minutes)', 'SPE (log scale)')
    save('ml4_recon_anomaly', f, 'Synthetic drilling channels (seed 4) with an injected SPP drift and a hook-load glitch; '
         'PCA (k = 4) fitted on minutes 0-399.', kind='computed')


@fig
def pca_k_cv():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    n, d = 600, 20
    F = rng.normal(size=(n, 3)) * np.array([3.0, 2.5, 2.0])
    A = rng.normal(size=(3, d))
    y = rng.integers(0, 2, n)
    u = rng.normal(size=d)
    u -= A.T @ np.linalg.lstsq(A.T, u, rcond=None)[0]
    u /= np.linalg.norm(u)
    X = F @ A + rng.normal(0, 1.0, (n, d)) + np.outer((y - .5) * 2.0, u)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    base = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv).mean()
    ks = np.arange(1, 21)
    sc = [cross_val_score(make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=2000)), X, y,
                          cv=cv).mean() for k in ks]
    cum = np.cumsum(PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_)
    print('  pca_k_cv base', round(base, 3), 'scores', np.round(sc, 3), 'cum3', round(cum[2], 3))
    f, axs = half()
    ax = axs[0]
    ax.plot(ks, sc, '-o', color=BLUE, ms=5)
    ax.axhline(base, color=TEAL, lw=1.6, ls='--')
    ax.text(20, base + .012, 'no PCA, same folds', color=TEAL, fontsize=12.5, ha='right', va='bottom')
    ax.axhline(.5, color=GREY, lw=1, ls=':')
    ax.text(20, .505, 'chance', color=GREY, fontsize=12, ha='right', va='bottom')
    ax.axvspan(.5, 3.5, color=ORANGE, alpha=.08, lw=0)
    ax.text(4.1, .585, f'k ≤ 3 (shaded): {cum[2]:.0%} of the\nvariance, accuracy at chance', color=ORANGE, fontsize=12,
            ha='left', va='center')
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.set_ylim(.4, .95)
    ax.set_xlim(.5, 20.5)
    clean(ax, 'n_components inside the pipeline', 'Cross-validated accuracy')
    save('ml4_pca_k_cv', f, 'Synthetic 20-feature data (seed 3): three high-variance nuisance factors; the class signal '
         'lies in a low-variance direction.', kind='computed')


# ====================================================== part 4: beyond PCA
@fig
def kernel_pca():
    from sklearn.datasets import make_circles
    from sklearn.decomposition import PCA, KernelPCA
    X, y = make_circles(n_samples=400, factor=.35, noise=.06, random_state=0)
    Zl = PCA(2).fit_transform(X)
    Zk = KernelPCA(2, kernel='rbf', gamma=4).fit_transform(X)
    f, axs = sub(3, height=4.0)
    for ax, Z, title, xl, yl in [(axs[0], X, 'Original: two rings', 'Feature 1', 'Feature 2'),
                                 (axs[1], Zl, 'Linear PCA scores', 'PC1', 'PC2'),
                                 (axs[2], Zk, 'RBF kernel PCA scores (γ = 4)', 'Kernel PC1', 'Kernel PC2')]:
        points(ax, Z, y, s=16)
        clean(ax, xl, yl, grid=False)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axs[2]
    lo, hi = ax.get_ylim()
    x0 = (Zk[y == 0, 0].max() + Zk[y == 1, 0].min()) / 2
    ax.axvline(x0, color=GREY, lw=1.2, ls='--')
    ax.text(x0, lo + .03 * (hi - lo), 'linear cut', fontsize=12, color=INK, va='bottom', ha='center',
            bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none'))
    save('ml4_kernel_pca', f, 'scikit-learn make_circles (400 points, noise 0.06, seed 0); colours = ring membership, '
         'not used in fitting.', kind='computed')


@fig
def lda_vs_pca():
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(2)
    cov = np.array([[3.0, 1.1], [1.1, .6]])
    X0 = rng.multivariate_normal([0, 0], cov, 250)
    X1 = rng.multivariate_normal([-.35, 1.0], cov, 250)
    X = np.r_[X0, X1]
    y = np.r_[np.zeros(250, int), np.ones(250, int)]
    mu = X.mean(0)
    pca = PCA(1).fit(X)
    lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    w_p = pca.components_[0]
    w_l = lda.scalings_[:, 0] / np.linalg.norm(lda.scalings_[:, 0])
    zp = (X - mu) @ w_p
    zl = (X - mu) @ w_l

    def sep(z):
        a, b = z[y == 0], z[y == 1]
        return abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    print('  lda_vs_pca separation (d-prime): PCA', round(sep(zp), 2), 'LDA', round(sep(zl), 2))
    f, axs = sub(3, height=4.0)
    ax = axs[0]
    points(ax, X, y, s=14)
    for w, col, nm in [(w_p, BLUE, 'PCA direction'), (w_l, TEAL, 'LDA direction')]:
        t = np.array([-4.2, 4.2])
        ax.plot(mu[0] + t * w[0], mu[1] + t * w[1], color=col if nm.startswith('LDA') else INK, lw=2.2)
    ax.text(mu[0] + 4.3 * w_p[0], mu[1] + 4.3 * w_p[1] + .15, 'PCA', color=INK, fontsize=12.5, fontweight='semibold',
            ha='left')
    ax.text(mu[0] + 2.4 * w_l[0] + .15, mu[1] + 2.4 * w_l[1], 'LDA', color=TEAL, fontsize=12.5,
            fontweight='semibold', ha='left')
    ax.set_aspect('equal', adjustable='datalim')
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    ax.set_title('Two facies, shared elongated spread', fontsize=14)
    bins = np.linspace(-6, 6, 40)
    for ax, z, title in [(axs[1], zp, 'Projected on PCA direction'), (axs[2], zl, 'Projected on LDA direction')]:
        z = (z - z.mean()) / z.std()
        bins = np.linspace(-3.5, 3.5, 36)
        ax.hist(z[y == 0], bins=bins, color=BLUE, alpha=.55)
        ax.hist(z[y == 1], bins=bins, color=ORANGE, alpha=.55)
        clean(ax, 'Projected value (standardized)', 'Count' if ax is axs[1] else None)
        ax.set_title(title, fontsize=14)
        ax.text(.03, .95, f'separation d′ = {sep(z):.1f}', transform=ax.transAxes, fontsize=12.5, va='top',
                color=INK)
    save('ml4_lda_vs_pca', f, 'Synthetic two-class Gaussian data with a shared covariance (seed 2); d′ = mean '
         'difference / pooled SD.', kind='computed')


@fig
def ica_sources():
    from sklearn.decomposition import FastICA, PCA
    t = np.linspace(0, 2, 1000)
    s1 = np.sin(2 * np.pi * 3 * t)                                 # rotating equipment
    s2 = np.sign(np.sin(2 * np.pi * .9 * t))                      # valve cycling
    rng = np.random.default_rng(1)
    s3 = ((t * 5.3) % 1) * 2 - 1                                 # sawtooth (pressure build-and-bleed)
    S = np.c_[s1, s2, s3] + .04 * rng.normal(size=(len(t), 3))
    A = np.array([[1.0, .6, .5], [.5, 1.0, .7], [.4, .5, 1.0]])
    Xm = S @ A.T
    P = PCA(3).fit_transform(Xm)
    ica = FastICA(3, whiten='unit-variance', random_state=0, max_iter=1000)
    R = ica.fit_transform(Xm)
    C = np.corrcoef(np.c_[S, R].T)[:3, 3:]
    order = [int(np.argmax(np.abs(C[i]))) for i in range(3)]
    R = R[:, order] * np.sign([C[i, order[i]] for i in range(3)])
    Cp = np.corrcoef(np.c_[S, P].T)[:3, 3:]
    print('  ica |corr| recovered', np.round([abs(C[i, order[i]]) for i in range(3)], 3),
          'pca best |corr|', np.round(np.abs(Cp).max(1), 3))
    cols = [BLUE, ORANGE, TEAL]
    f, axs = plt.subplots(3, 4, figsize=(11.8, 4.2), sharex=True, layout='constrained')
    titles = ['True sources', 'Three sensor mixtures', 'PCA components', 'ICA components']
    for j, (M, title) in enumerate(zip([S, Xm, P, R], titles)):
        for i in range(3):
            ax = axs[i, j]
            col = cols[i] if j in (0, 3) else GREY
            ax.plot(t, M[:, i], color=col, lw=1.2)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            if i < 2:
                ax.tick_params(labelbottom=False, length=0)
            ax.grid(False)
        axs[0, j].set_title(title, fontsize=14)
    axs[2, 0].set_xticks([.5, 1, 1.5])                  # shared x: interior ticks keep panels apart
    for j in range(4):
        axs[2, j].set_xlabel('Time (s)', fontsize=12.5)
    save('ml4_ica_sources', f, 'Synthetic sources (seed 1) mixed by a fixed 3 x 3 matrix; FastICA (seed 0); ICA rows '
         'reordered and sign-matched for display.', kind='computed')


@fig
def nmf_parts():
    from sklearn.decomposition import NMF, PCA
    rng = np.random.default_rng(9)
    m = np.arange(36)
    P1 = 1 / (1 + .9 * .35 * m) ** (1 / .9)                    # steep early decline
    P2 = np.clip(1 - .012 * np.maximum(m - 8, 0), 0, None) * (1 - np.exp(-(m + 1) / 2.5))  # plateau then slow decline
    P3 = 1 / (1 + np.exp(-(m - 16) / 2.5))                       # delayed support response
    parts = np.c_[P1, P2, P3]
    W = rng.gamma(1.2, 1.0, (80, 3)) * rng.binomial(1, .75, (80, 3))
    W[W.sum(1) == 0, 0] = 1
    X = np.clip(W @ parts.T + rng.normal(0, .03, (80, 36)), 0, None)
    nmf = NMF(3, init='nndsvda', max_iter=2000, random_state=0).fit(X)
    H = nmf.components_
    order = [int(np.argmax([np.corrcoef(H[k], parts[:, i])[0, 1] for k in range(3)])) for i in range(3)]
    H = H[order]
    pca = PCA(3).fit(X)
    f, axs = sub(3, height=3.9)
    ax = axs[0]
    for i in range(8):
        ax.plot(m, X[i], color=GREY, lw=1.2, alpha=.8)
    clean(ax, 'Months on production', 'Rate (normalized units)')
    ax.set_title('Eight of 80 synthetic wells', fontsize=14)
    ax = axs[1]
    for k, col in enumerate([PURPLE, GOLD, RED]):
        ax.plot(m, pca.components_[k], color=col, lw=2)
        ax.text(35.5, pca.components_[k][-1], f'PC{k + 1}', color=col, fontsize=12, ha='left', va='center',
                fontweight='semibold')
    ax.axhline(0, color=GREY, lw=.8)
    ax.set_xlim(0, 40)
    clean(ax, 'Months on production', 'Component weight')
    ax.set_title('PCA: signed contrasts', fontsize=14)
    ax = axs[2]
    names = ['early decline', 'plateau', 'delayed response']
    for k, col in enumerate([BLUE, ORANGE, TEAL]):
        ax.plot(m, parts[:, k] / parts[:, k].max(), color=col, lw=1.2, ls='--', alpha=.8)
        ax.plot(m, H[k] / H[k].max(), color=col, lw=2.4)
    ax.text(23, .2, 'early decline', color=BLUE, fontsize=12, va='bottom')
    ax.text(10, .62, 'plateau', color=ORANGE, fontsize=12)
    ax.text(21, .42, 'delayed\nresponse', color=TEAL, fontsize=12)
    ax.set_ylim(-.05, 1.12)
    clean(ax, 'Months on production', 'Part (scaled to max 1)')
    ax.set_title('NMF: additive parts (dashed = truth)', fontsize=14)
    save('ml4_nmf_parts', f, 'Synthetic production profiles (seed 9): non-negative mixtures of three shapes plus noise; '
         'NMF (k = 3, seed 0) and PCA (k = 3).', kind='computed')


@fig
def autoencoder():
    f, ax = canvas(6.3, 4.9)
    network(ax, [8, 5, 2, 5, 8], .6, 5.7, 1.35, 4.15, r=.15, max_nodes=8,
            colors=[GREY, BLUE, ORANGE, TEAL, GREY], fills=['tint', 'tint', 'solid', 'tint', 'tint'], edge_lw=.6)
    label(ax, .6, 4.6, r'$x$', fs=17)
    label(ax, 3.15, 4.6, 'code z (2 numbers)', fs=13.5, weight='semibold', color=ORANGE)
    label(ax, 5.7, 4.6, r'$\hat{x}$', fs=17)
    bracket(ax, .5, 3.05, 1.0, 'encoder', color=BLUE, fs=12.5, below=True)
    bracket(ax, 3.25, 5.8, 1.0, 'decoder', color=TEAL, fs=12.5, below=True)
    box(ax, 1.15, .08, 4.0, .46, '', INK, fill='white', lw=1.2)
    label(ax, 3.15, .31, 'train to minimize ' + r'$\Vert x-\hat{x}\Vert^2$', fs=13)
    save('ml4_autoencoder', f, 'Schematic.', kind='schematic')


# ====================================================== part 5: manifold learning
CMAP_ROLL = LinearSegmentedColormap.from_list('ml4_roll', [ORANGE, GOLD, TEAL, BLUE, PURPLE])


@lru_cache(maxsize=None)
def swiss_embeddings():
    from sklearn.datasets import make_swiss_roll
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
    X, t = make_swiss_roll(n_samples=1500, noise=.05, random_state=0)
    embs = {'PCA': PCA(2).fit_transform(X),
            'Isomap': Isomap(n_neighbors=10, n_components=2).fit_transform(X),
            'LLE': LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=0).fit_transform(X),
            't-SNE': TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(X)}
    return X, t, embs


@fig
def geodesic():
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph
    rng = np.random.default_rng(0)
    lo, hi = 1.5 * np.pi, 4.5 * np.pi
    tt = np.sqrt(np.linspace(lo ** 2, hi ** 2, 400)) + rng.normal(0, .01, 400)   # even spacing along the arc
    X = np.c_[tt * np.cos(tt), tt * np.sin(tt)] + rng.normal(0, .2, (400, 2))
    G = kneighbors_graph(X, 7, mode='distance')
    D, pred = shortest_path(G, directed=False, return_predecessors=True)
    ia = int(np.argmin(np.abs(tt - 2.05 * np.pi)))
    cands = np.where(np.abs(tt - (tt[ia] + 2 * np.pi)) < .3)[0]
    ib = int(cands[np.argmin(np.linalg.norm(X[cands] - X[ia], axis=1))])
    path = [ib]
    while path[-1] != ia:
        path.append(pred[ia, path[-1]])
    eu = np.linalg.norm(X[ia] - X[ib])
    print('  geodesic euclid', round(eu, 2), 'graph', round(D[ia, ib], 2))
    f, axs = half()
    ax = axs[0]
    rows, cols = G.nonzero()
    for i, j in zip(rows, cols):
        ax.plot(X[[i, j], 0], X[[i, j], 1], color='#8A99A8', lw=.8, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=9, c=tt, cmap=CMAP_ROLL, zorder=2, edgecolors='none')
    P = X[path]
    ax.plot(P[:, 0], P[:, 1], color=TEAL, lw=3.2, zorder=3, solid_capstyle='round')
    ax.plot(X[[ia, ib], 0], X[[ia, ib], 1], color=INK, lw=2, ls='--', zorder=4)
    for i, nm in [(ia, 'A'), (ib, 'B')]:
        ax.scatter(*X[i], s=110, color='white', edgecolors=INK, lw=2, zorder=5)
        ax.text(X[i, 0] + .7, X[i, 1] + .6, nm, fontsize=14, fontweight='semibold', color=INK, zorder=6)
    ax.text(15.2, -5.6, 'grey lines: graph of\n7 nearest neighbours', fontsize=12, color=GREY, ha='left', va='top')
    ax.text(15.2, -9.5, f'straight line A–B:\n{eu:.1f}', fontsize=13, color=INK, ha='left', va='top')
    ax.text(15.2, -13.2, f'path along the\ngraph: {D[ia, ib]:.1f}', fontsize=13, color=TEAL, ha='left', va='top',
            fontweight='semibold')
    ax.set_aspect('equal')
    ax.set_xlim(-15, 23.5)
    ax.set_ylim(-16.5, 15.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    save('ml4_geodesic', f, 'Synthetic noisy spiral (seed 0, 400 points); 7-nearest-neighbour graph and scipy '
         'shortest path; colour = position along the spiral.', kind='computed')


@fig
def swiss_roll():
    X, t, embs = swiss_embeddings()
    f = plt.figure(figsize=(11.8, 3.7), layout='constrained')
    gs = f.add_gridspec(1, 5)
    ax = f.add_subplot(gs[0], projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=CMAP_ROLL, s=4, depthshade=False)
    ax.view_init(elev=12, azim=-72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect(None, zoom=1.2)
    ax.text2D(.5, .93, 'Swiss roll (3-D)', transform=ax.transAxes, fontsize=14, fontweight='semibold',
              ha='center', va='bottom', color=INK)
    for k, name in enumerate(['PCA', 'Isomap', 'LLE', 't-SNE']):
        a = f.add_subplot(gs[k + 1])
        E = embs[name]
        a.scatter(E[:, 0], E[:, 1], c=t, cmap=CMAP_ROLL, s=4)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(name, fontsize=14)
        for sp in a.spines.values():
            sp.set_color(LIGHT)
    save('ml4_swiss_roll', f, 'scikit-learn make_swiss_roll (1500 points, noise 0.05, seed 0); Isomap k = 10, LLE k = 12, '
         't-SNE perplexity 30; colour = position along the roll.', kind='computed')


@fig
def tsne_sizes():
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(0)
    A = rng.normal([0, 0], .3, (200, 2))
    B = rng.normal([5, 0], 1.5, (200, 2))
    C = rng.normal([25, 0], .3, (200, 2))
    X = np.r_[A, B, C]
    lab = np.repeat([0, 1, 2], 200)
    E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(X)

    def stats(Z):
        cent = np.array([Z[lab == k].mean(0) for k in range(3)])
        rad = [np.sqrt(((Z[lab == k] - cent[k]) ** 2).sum(1).mean()) for k in range(3)]
        return cent, rad
    cO, rO = stats(X)
    cE, rE = stats(E)
    print('  tsne_sizes orig radii', np.round(rO, 2), 'map radii', np.round(rE, 2),
          'map dAB', round(np.linalg.norm(cE[0] - cE[1]), 1), 'dBC', round(np.linalg.norm(cE[1] - cE[2]), 1))
    cols = [BLUE, ORANGE, TEAL]
    f, axs = sub(2, height=3.9)
    for ax, Z, (cent, rad), title in [(axs[0], X, (cO, rO), 'Original data (2-D)'),
                                      (axs[1], E, (cE, rE), 't-SNE map, perplexity 30')]:
        for k in range(3):
            ax.scatter(Z[lab == k, 0], Z[lab == k, 1], s=10, color=cols[k], alpha=.7, edgecolors='none')
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LIGHT)
    ax = axs[0]
    ax.set_ylim(-7, 7.5)
    ax.set_xlim(-4, 29)
    for k, nm in enumerate('ABC'):
        ax.text(cO[k, 0] + (-.8 if k == 0 else 0), 4.9 if k == 1 else 1.3, nm, ha='center', va='bottom', fontsize=13,
                color=cols[k], fontweight='semibold')
    ax.text(14.5, 5.6, f'B is {rO[1] / rO[0]:.0f}× wider than A\ngap B–C = {np.linalg.norm(cO[1] - cO[2]) / np.linalg.norm(cO[0] - cO[1]):.0f}× gap A–B',
            fontsize=12.5, color=INK, ha='center', va='top')
    bracket(ax, cO[0, 0], cO[1, 0], -5.3, f'gap {np.linalg.norm(cO[0] - cO[1]):.0f}', color=INK, fs=12)
    bracket(ax, cO[1, 0], cO[2, 0], -5.3, f'gap {np.linalg.norm(cO[1] - cO[2]):.0f}', color=INK, fs=12)
    ax = axs[1]
    for k, nm in enumerate('ABC'):
        ax.text(cE[k, 0], cE[k, 1] + rE[k] * 1.55, nm, ha='center', va='bottom',
                fontsize=13, color=cols[k], fontweight='semibold')
    for (i, j) in [(0, 1), (1, 2)]:
        ax.plot(*zip(cE[i], cE[j]), color=INK, lw=1, ls=':')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 4, hi + 22)
    x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    ax.text(x_mid, ax.get_ylim()[1] - 2, f'B is {rE[1] / rE[0]:.1f}× wider than A\ngap B–C = '
            f'{np.linalg.norm(cE[1] - cE[2]) / np.linalg.norm(cE[0] - cE[1]):.1f}× gap A–B', fontsize=12.5, color=INK,
            ha='center', va='top')
    save('ml4_tsne_sizes', f, 'Synthetic 2-D clusters (seed 0): spreads 0.3 / 1.5 / 0.3, centres at 0, 5, 25; '
         't-SNE seed 0. Spread = RMS distance to the cluster centre.', kind='computed')


@fig
def tsne_seeds():
    from sklearn.datasets import make_blobs
    from sklearn.manifold import TSNE, trustworthiness
    X, y = make_blobs(600, n_features=10, centers=6, cluster_std=[1.0, 1.5, .7, 2.0, 1.2, .9], random_state=11)
    f, axs = sub(3, height=3.8)
    for ax, s in zip(axs, [0, 1, 2]):
        E = TSNE(2, perplexity=30, init='random', random_state=s).fit_transform(X)
        tw = trustworthiness(X, E, n_neighbors=10)
        for k in range(6):
            ax.scatter(E[y == k, 0], E[y == k, 1], s=9, color=SERIES[k], alpha=.8, edgecolors='none')
        ax.set_title(f'random_state = {s}   (T₁₀ = {tw:.2f})', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LIGHT)
    save('ml4_tsne_seeds', f, 'Six synthetic 10-D Gaussian clusters (make_blobs seed 11); t-SNE perplexity 30 with '
         'init="random"; colours = generating cluster, not used in fitting.', kind='computed')


@fig
def umap_schematic():
    f, ax = canvas(6.3, 4.9)
    rng = np.random.default_rng(5)
    centres = np.array([(0, 0), (1.1, .45), (2.2, 0)])
    pts = np.vstack([c + rng.normal(0, .18, (6, 2)) for c in centres])

    def graph_panel(x0, y0, k, title, sub_):
        P = pts * 1.0 + np.array([x0 + .35, y0 + .75])
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        drawn = set()
        for i in range(len(P)):
            for j in np.argsort(D[i])[1:k + 1]:
                e = (min(i, j), max(i, j))
                if e in drawn:
                    continue
                drawn.add(e)
                ax.plot(*zip(P[i], P[j]), color=TEAL, lw=1.2, alpha=.55, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], s=58, color=BLUE, zorder=3, edgecolors='white', lw=.8)
        label(ax, x0 + 1.45, y0 + 2.0, title, fs=14, weight='semibold')
        label(ax, x0 + 1.45, y0 + .25, sub_, fs=12, color=MUTED)

    def pack_panel(x0, y0, spread, title, sub_):
        for c in centres:
            Q = c + np.array([x0 + .35, y0 + .75]) + rng.normal(0, 1, (6, 2)) * spread
            ax.scatter(Q[:, 0], Q[:, 1], s=58, color=BLUE, zorder=3, edgecolors='white', lw=.8)
        label(ax, x0 + 1.45, y0 + 2.0, title, fs=14, weight='semibold')
        label(ax, x0 + 1.45, y0 + .25, sub_, fs=12, color=MUTED)
    graph_panel(.05, 2.55, 3, 'n_neighbors small', 'separate local pieces')
    graph_panel(3.2, 2.55, 7, 'n_neighbors large', 'pieces linked together')
    pack_panel(.05, .1, .025, 'min_dist small', 'tight clumps')
    pack_panel(3.2, .1, .13, 'min_dist large', 'points kept apart')
    ax.plot([.15, 6.15], [2.5, 2.5], color=RULE, lw=1.2)
    ax.plot([3.15, 3.15], [.15, 4.8], color=RULE, lw=1.2)
    save('ml4_umap_schematic', f, 'Schematic of UMAP parameters drawn by hand; not UMAP output (umap-learn is an '
         'optional package).', kind='schematic')


@fig
def trust_compare():
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, trustworthiness
    from sklearn.preprocessing import StandardScaler
    X, t, embs = swiss_embeddings()
    d = load_digits()
    ix = np.random.default_rng(42).choice(len(d.data), 600, replace=False)
    Xd = StandardScaler().fit_transform(d.data[ix])
    embd = {'PCA': PCA(2).fit_transform(Xd),
            'Isomap': Isomap(n_neighbors=10, n_components=2).fit_transform(Xd),
            'LLE': LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=0).fit_transform(Xd),
            't-SNE': TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Xd)}
    names = ['PCA', 'Isomap', 'LLE', 't-SNE']
    cols = {'PCA': BLUE, 'Isomap': TEAL, 'LLE': PURPLE, 't-SNE': ORANGE}
    f, axs = half(2)
    for ax, (XX, EE, title) in zip(axs, [(X, embs, 'Swiss roll'), (Xd, embd, 'Digits (64-D)')]):
        for i, nm in enumerate(names):
            T = trustworthiness(XX, EE[nm], n_neighbors=10)
            C = trustworthiness(EE[nm], XX, n_neighbors=10)
            print(f'  trust {title} {nm}: T={T:.3f} C={C:.3f}')
            ax.plot([min(T, C), max(T, C)], [i, i], color=LIGHT, lw=3, zorder=1)
            ax.scatter([C], [i], s=190, facecolors='white', edgecolors=cols[nm], lw=2, zorder=3)
            ax.scatter([T], [i], s=55, color=cols[nm], zorder=4)      # dot on top: visible inside the ring
            ax.text(T, i - .3, f'{T:.2f}', ha='center', va='bottom', fontsize=11.5, color=cols[nm])
        ax.set_yticks(range(4), names if ax is axs[0] else [])
        ax.set_ylim(3.6, -.8)
        ax.set_xlim(.74, 1.01)
        ax.set_xticks([.8, .9, 1.0])
        ax.set_title(title, fontsize=14)
        clean(ax, 'Score at k = 10')
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
    h1 = Line2D([], [], marker='o', ls='', color=INK, ms=7, label='trustworthiness')
    h2 = Line2D([], [], marker='o', ls='', mfc='white', mec=INK, mew=2, ms=12, label='continuity')
    f.legend(handles=[h1, h2], loc='outside lower center', ncol=2, fontsize=12, handletextpad=.2)
    save('ml4_trust_compare', f, 'sklearn.manifold.trustworthiness at k = 10 (continuity = same function with the '
         'arguments swapped); Swiss roll seed 0; 600 standardized digit images (seed 42).', kind='computed')


@fig
def batch_map():
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(21)
    means = {'GR': [45, 100, 30], 'RHOB': [2.34, 2.52, 2.62], 'NPHI': [.21, .31, .10], 'PEF': [1.9, 3.2, 4.9]}
    sds = {'GR': [9, 11, 7], 'RHOB': [.04, .04, .035], 'NPHI': [.03, .035, .025], 'PEF': [.3, .35, .4]}
    rows = []
    for w in range(4):
        fac = rng.choice(3, 150, p=[.4, .35, .25])
        rec = {'well': np.full(150, w), 'facies': fac}
        for k in means:
            rec[k] = np.asarray(means[k])[fac] + np.asarray(sds[k])[fac] * rng.normal(size=150)
        if w == 2:                                   # density tool offset in one well
            rec['RHOB'] = rec['RHOB'] + .13
            rec['NPHI'] = rec['NPHI'] - .05
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    Z = StandardScaler().fit_transform(df[list(means)])
    E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Z)
    fac_col = [GOLD, '#7D8B99', BLUE]
    well_col = [TEAL, PURPLE, ORANGE, INK]
    f, axs = half(2, height=4.6)
    for ax, key, cols, names, title in [(axs[0], 'facies', fac_col, ['Sand', 'Shale', 'Limestone'], 'Coloured by facies'),
                                        (axs[1], 'well', well_col, ['W1', 'W2', 'W3', 'W4'], 'Coloured by well')]:
        for k, (c, nm) in enumerate(zip(cols, names)):
            m = df[key].values == k
            ax.scatter(E[m, 0], E[m, 1], s=8, color=c, alpha=.8, edgecolors='none', label=nm)
        ax.set_title(title, fontsize=13.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        for sp in ax.spines.values():
            sp.set_color(LIGHT)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.02), ncol=len(names), fontsize=11.5, handletextpad=.1,
                  columnspacing=.7, markerscale=2.2)
    save('ml4_batch_map', f, 'Synthetic logs from four wells (seed 21); well W3 has a density-neutron offset; t-SNE '
         'perplexity 30 on standardized GR, RHOB, NPHI, PEF.', kind='computed')


# ====================================================== part 6: reduction in practice
@fig
def drilling_compression():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    df, L, form = drill_channels()
    names = list(df.columns)
    units = {n: DRILL_SPEC[n][0] for n in names}
    tr, va = df.iloc[:430], df.iloc[430:]
    sc = StandardScaler().fit(tr)
    ks = [1, 2, 3, 4, 5]
    R = np.zeros((len(names), len(ks)))
    for j, k in enumerate(ks):
        m = PCA(k).fit(sc.transform(tr))
        rec = sc.inverse_transform(m.inverse_transform(m.transform(sc.transform(va))))
        R[:, j] = np.sqrt(((rec - va.values) ** 2).mean(0))
    rel = R / tr.std().values[:, None]
    print('  compression rmse k=3', dict(zip(names, np.round(R[:, 2], 2))), 'rel', np.round(rel[:, 2], 2))
    f, axs = half()
    ax = axs[0]
    ax.imshow(rel, cmap=CMAP_ORANGE, vmin=0, vmax=1.1, aspect='auto', interpolation='nearest')
    for i in range(len(names)):
        for j in range(len(ks)):
            v = R[i, j]
            txt = f'{v:.0f}' if v >= 10 else (f'{v:.1f}' if v >= .1 else f'{v:.2f}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=11.5, color='white' if rel[i, j] > .6 else INK)
    ax.set_xticks(range(len(ks)), [f'k = {k}' for k in ks])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)), [f'{n} ({units[n]})' for n in names])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((1.5, -.5), 1, len(names), fc='none', ec=INK, lw=2))
    ax.set_xlabel('Validation RMSE in engineering units (shade = RMSE / channel SD)', fontsize=12)
    save('ml4_drilling_compression', f, 'Synthetic drilling channels (seed 4); PCA on standardized minutes 0-429, '
         'reconstruction error on minutes 430-719 in original units.', kind='computed')


@fig
def log_qc():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ref, _, _ = log_suite()
    new, _, _ = log_suite(n=600, seed=23)
    new = new.copy()
    depth = 2000 + .5 * np.arange(len(new))
    wash = (depth >= 2120) & (depth < 2138)
    new.loc[wash, 'CALI'] += 2.2
    new.loc[wash, 'RHOB'] -= .22
    new.loc[wash, 'NPHI'] += .07
    new.loc[wash, 'DT'] += 12
    cols = list(ref.columns)
    sc = StandardScaler().fit(ref)
    m = PCA(4).fit(sc.transform(ref))
    def spe(D):
        Z = sc.transform(D)
        return ((Z - m.inverse_transform(m.transform(Z))) ** 2).sum(1)
    thr = np.quantile(spe(ref), .99)
    e = spe(new)
    flag = e > thr
    print('  log_qc thr', round(thr, 2), 'flagged in washout', flag[wash].mean().round(2), 'outside', flag[~wash].mean().round(3))
    tracks = [('GR', 'GR (API)', GOLD), ('CALI', 'CALI (in)', GREY), ('RHOB', 'RHOB (g/cm³)', ORANGE),
              ('NPHI', 'NPHI (v/v)', BLUE)]
    f, axs = plt.subplots(1, 5, figsize=(11.8, 4.0), sharey=True, layout='constrained',
                          gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25]))
    for ax, (c, lab, col) in zip(axs, tracks):
        ax.plot(new[c], depth, color=col, lw=1)
        ax.set_xlabel(lab, fontsize=12.5)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', labelsize=11)
        ax.grid(alpha=1)
    ax = axs[4]
    ax.fill_betweenx(depth, 0, e, color=BLUE, alpha=.3, lw=0)
    ax.plot(e, depth, color=BLUE, lw=1)
    ax.axvline(thr, color=ORANGE, ls='--', lw=1.4)
    ax.set_xscale('log')
    ax.set_xlim(.05, e.max() * 2)
    ax.set_xlabel('PCA reconstruction error', fontsize=12.5)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=11)
    ax.text(thr * 1.15, 2190, '99th pct of\nreference well', color=ORANGE, fontsize=11.5, va='center')
    for a in axs:
        for lo, hi in _runs(depth, flag):
            a.axhspan(lo, hi, color=ORANGE, alpha=.14, lw=0)
    axs[0].set_ylim(depth.max(), depth.min())
    axs[0].set_ylabel('Depth (m)')
    axs[1].annotate('washout:\nlogs disagree', (10.9, 2118), xytext=(9.95, 2075), fontsize=12, color=INK,
                    va='center', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    save('ml4_log_qc', f, 'Synthetic log suites (reference seed 11, new well seed 23) with an injected washout at '
         '2120-2138 m; PCA (k = 4) fitted on the standardized reference well.', kind='computed')


def _runs(depth, mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if not v else i + 1
            out.append((depth[start] - .25, depth[min(end, len(depth) - 1)] + .25))
            start = None
    return out


@fig
def production_shapes():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(13)
    m = np.arange(36)
    P1 = 1 / (1 + .9 * .35 * m) ** (1 / .9)
    P2 = np.clip(1 - .012 * np.maximum(m - 8, 0), 0, None) * (1 - np.exp(-(m + 1) / 2.5))
    P3 = 1 / (1 + np.exp(-(m - 16) / 2.5))
    shapes = np.c_[P1, P2, P3].T
    fam = np.repeat([0, 1, 2], 30)
    amp = np.exp(rng.normal(0, .8, 90)) * 500                      # well size varies about 10-fold
    mix = np.eye(3)[fam] * .8 + rng.dirichlet([1, 1, 1], 90) * .2
    X = amp[:, None] * (mix @ shapes) * np.exp(rng.normal(0, .06, (90, 36)))
    Xn = X / X.sum(1, keepdims=True)
    Zr = PCA(2).fit(X).transform(X)
    Zn = PCA(2).fit(Xn).transform(Xn)
    r_amp = abs(np.corrcoef(Zr[:, 0], np.log(amp))[0, 1])
    print('  production |corr(PC1 raw, log amplitude)|', round(r_amp, 2))
    cols = [BLUE, ORANGE, TEAL]
    names = ['early decline', 'plateau', 'delayed response']
    f, axs = sub(2, height=3.9)
    for ax, Z, title in [(axs[0], Zr, 'Raw monthly rates: PC1 tracks well size'),
                         (axs[1], Zn, 'Rates / cumulative: shapes separate')]:
        for k in range(3):
            sel = fam == k
            ax.scatter(Z[sel, 0], Z[sel, 1], s=np.clip(amp[sel] / 25, 10, 120), color=cols[k], alpha=.75,
                       edgecolors='white', lw=.5, label=names[k])
        ax.set_title(title, fontsize=14, loc='left')
        clean(ax, 'PC1', 'PC2', grid=False)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[1].legend(loc='lower left', fontsize=12, markerscale=.8, handletextpad=.2)
    axs[0].text(.02, .04, 'marker size = well size', transform=axs[0].transAxes, fontsize=12, color=MUTED)
    save('ml4_production_shapes', f, 'Synthetic production profiles for 90 wells (seed 13): three shape families, '
         'well sizes varying more than tenfold; colours added after fitting.', kind='computed')
