"""Figures for module key 'ml1' (Supervised learning foundations).

Figure names start with 'ml1_'. Every computed figure uses fixed seeds and
synthetic data (labelled as synthetic in captions); diagrams are schematic.

Shared synthetic data
---------------------
_wells() builds an 8-well, depth-indexed log dataset (GR, RHOB, NPHI, log10 RT,
PE, DT) with five facies drawn as beds from a Markov chain, tool-like smoothing
(adjacent samples are strongly correlated) and well-specific calibration
offsets. It is used by the motivating case, the EDA figures and the
classification-evaluation figures so that numbers are consistent across slides.
"""
from functools import lru_cache

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import make_moons
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.linear_model import lasso_path as sk_lasso_path
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, recall_score,
                             balanced_accuracy_score)
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, RepeatedKFold,
                                     LeaveOneGroupOut, cross_val_score, cross_val_predict,
                                     GridSearchCV, ShuffleSplit, learning_curve,
                                     validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

KEY = 'ml1'
fig = register(KEY)

SLATE = '#56687A'
GREY_T = '#EEF1F4'
FACIES = ['Sandstone', 'Siltstone', 'Shale', 'Limestone', 'Dolomite']
# Multi-class facies colours avoid BLUE/ORANGE, which mean class 0/1 in binary figures.
FAC_COL = [GOLD, TEAL, SLATE, PURPLE, RED]
LOGS = ['GR', 'RHOB', 'NPHI', 'LRT', 'PE', 'DT']
FEATS = ['GR', 'RHOB', 'NPHI', 'LRT', 'PE']

# ---------------------------------------------------------------- data
_MEAN = np.array([[42, 2.33, 0.17, 1.00, 2.0, 86],     # sandstone
                  [72, 2.44, 0.23, 0.72, 2.7, 90],     # siltstone
                  [102, 2.51, 0.31, 0.45, 3.2, 99],    # shale
                  [30, 2.58, 0.09, 1.65, 4.6, 62],     # limestone
                  [34, 2.65, 0.11, 1.72, 4.0, 59]])    # dolomite (rare, overlaps limestone)
_SD = np.array([[14, .05, .035, .30, .35, 6],
                [14, .05, .035, .25, .40, 6],
                [14, .05, .04, .20, .35, 7],
                [11, .05, .035, .30, .45, 5],
                [11, .05, .035, .30, .45, 5]])
_P = np.array([[0, .47, .43, .08, .02],                # facies transition probabilities between beds
               [.42, 0, .46, .10, .02],
               [.36, .32, 0, .28, .04],
               [.12, .06, .52, 0, .30],
               [.05, .05, .30, .60, 0]])
_THICK = np.array([2.5, 2.0, 3.0, 2.5, 1.3])        # mean bed thickness, m
DZ = 0.15                                           # sample spacing, m


def _ar1(n, rho, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    return x


@lru_cache(maxsize=4)
def _wells(n_wells=8, n=400, seed=7):
    """Synthetic multi-well log table (one row per 0.15 m depth sample)."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n_wells):
        fac = np.empty(n, int)
        i, f = 0, rng.integers(5)
        while i < n:
            k = int(round(max(.6, rng.exponential(_THICK[f])) / DZ))
            fac[i:i + k] = f
            i += k
            f = rng.choice(5, p=_P[f])
        base = _MEAN[fac] + _SD[fac] * np.column_stack([_ar1(n, .9, rng) for _ in range(6)])
        dphi = .02 * _ar1(n, .92, rng)                      # shared porosity variation
        base[:, 1] -= 1.65 * dphi
        base[:, 2] += dphi
        base[:, 5] += 134 * dphi
        logs = np.column_stack([gaussian_filter1d(base[:, j], 2.0) for j in range(6)])
        logs += np.array([rng.normal(0, s) for s in (9, .03, .02, .2, .25, 4)])  # well offsets
        logs[:, 0] = 5 + 4 * np.logaddexp(0, (logs[:, 0] - 5) / 4)            # GR stays positive
        top = rng.uniform(1450, 1600)
        d = pd.DataFrame(logs, columns=LOGS)
        d.insert(0, 'depth', top + DZ * np.arange(n))
        d.insert(0, 'well', f'W{w + 1}')
        d['facies'] = fac
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _xy(df=None):
    df = _wells() if df is None else df
    return df[FEATS].to_numpy(), df['facies'].to_numpy(), df['well'].to_numpy()


def _moons():
    """Same synthetic moons as the shared boundary figures (seed 42)."""
    return make_moons(n_samples=220, noise=.23, random_state=42)


# ------------------------------------------------------------- helpers
ROLE = {'train': ('Train', BLUE), 'val': ('Val', TEAL), 'test': ('Test', ORANGE),
        'none': ('', '#C9D3DD')}


def _cell(ax, x, y, w, h, role, text=None, fs=12, weight='semibold'):
    name, col = ROLE[role]
    if role == 'none':
        box(ax, x, y, w, h, '', '#9AA9B8', fill='white', lw=1.1, ls=(0, (2, 2)), radius=.05)
    else:
        box(ax, x, y, w, h, name if text is None else text, col, fill='tint', fs=fs,
            weight=weight, text_color=col, lw=1.3, radius=.05)


def _legend_roles(ax, x, y, roles=('train', 'val'), fs=12.5, gap=None):
    """Legend of cell roles; entries are spaced by their text length."""
    names = {'train': 'training rows', 'val': 'validation rows', 'test': 'test rows', 'none': 'not used'}
    for r in roles:
        name, col = ROLE[r]
        if r == 'none':
            box(ax, x, y - .13, .34, .26, '', '#9AA9B8', fill='white', lw=1.1, ls=(0, (2, 2)), radius=.04)
        else:
            box(ax, x, y - .13, .34, .26, '', col, fill='tint', lw=1.3, radius=.04)
        label(ax, x + .44, y, names[r], fs=fs, ha='left', color=INK)
        x += .44 + len(names[r]) * fs * .0078 + .35
    return x


def _m(text):
    """Typographic minus signs in figure text."""
    return text.replace('-', '−')


def _luminance(rgb):
    c = np.asarray(matplotlib.colors.to_rgb(rgb))
    c = np.where(c <= .03928, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]


def _text_on(fill):
    """White or INK text, whichever contrasts more with the cell colour."""
    lum = _luminance(fill)
    return 'white' if 1.05 / (lum + .05) > (lum + .05) / (_luminance(INK) + .05) else INK


def _panel_title(ax, x, y, text, fs=15, ha='left'):
    label(ax, x, y, text, fs=fs, weight='semibold', ha=ha)


def _swarm(vals, step, tol):
    """Horizontal offsets so that nearly equal values do not overlap (simple beeswarm)."""
    order = np.argsort(vals)
    offs = np.zeros(len(vals))
    placed = []
    for i in order:
        for cand in [0] + [s * k * step for k in range(1, 8) for s in (1, -1)]:
            if all(abs(cand - o) > step * .9 or abs(vals[i] - v) > tol for o, v in placed):
                offs[i] = cand
                placed.append((cand, vals[i]))
                break
    return offs


# ================================================================ opening
@fig
def case_split_gap():
    X, y, g = _xy()
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    rnd = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    lowo = cross_val_score(model, X, y, cv=LeaveOneGroupOut(), groups=g)
    f, axs = half()
    ax = axs[0]
    for k, (vals, col) in enumerate([(rnd, ORANGE), (lowo, BLUE)]):
        m = vals.mean()
        ax.plot([k - .3, k + .3], [m, m], color=col, lw=2.6, zorder=2, solid_capstyle='butt')
        ax.scatter(k + _swarm(vals, .06, .007), vals, s=80, color=col, edgecolors='white',
                   linewidths=1.2, zorder=5)
        if k == 0:
            ax.annotate(f'mean {m:.2f}', (k + .3, m), xytext=(6, 0), textcoords='offset points',
                        fontsize=14, color=col, fontweight='semibold', va='center')
        else:
            ax.annotate(f'mean {m:.2f}', (k - .3, m), xytext=(-6, 0), textcoords='offset points',
                        fontsize=14, color=col, fontweight='semibold', va='center', ha='right')
    lo, hi = lowo.min(), lowo.max()
    ax.annotate(f'worst well {lo:.3f}', (1.02, lo), xytext=(1.25, lo + .01), fontsize=12.5,
                color=MUTED, va='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    # gap bracket between the two means, right of the held-out group
    xb = 1.5
    ax.plot([.78, xb], [rnd.mean()] * 2, color=ORANGE, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.annotate('', (xb, lowo.mean()), (xb, rnd.mean()),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text(xb + .05, (lowo.mean() + rnd.mean()) / 2, f'optimism\n{rnd.mean() - lowo.mean():.2f}',
            fontsize=13.5, color=INK, va='center', ha='left')
    ax.set_xticks([0, 1], ['Random row folds\n(5 folds)', 'Held-out wells\n(8 folds)'])
    ax.set_xlim(-.45, 1.95)
    ax.set_ylim(.6, 1.0)
    clean(ax, None, 'Fold accuracy')
    ax.tick_params(axis='x', length=0, labelsize=13.5, labelcolor=INK)
    ax.grid(axis='x', visible=False)
    save('ml1_case_split_gap', f,
         f'Synthetic 8-well facies dataset (seed 7), 5 log features; standardized 1-nearest-neighbour '
         f'classifier. Random 5-fold mean {rnd.mean():.3f}; leave-one-well-out mean {lowo.mean():.3f} '
         f'(range {lo:.3f}-{hi:.3f}).', kind='computed')


# ================================================================ part 1: framing
@fig
def contract_timeline():
    f, ax = canvas(11.8, 3.6)
    # contract terms
    terms = [('Unit of prediction', 'one well-day'), ('Prediction time', '00:00 every day'),
             ('Decision supported', 'choke and intervention plans'), ('Metric', 'MAE in m³/d')]
    xs = [.15, 2.75, 5.35, 9.05]
    ws = [2.45, 2.45, 3.55, 2.6]
    for (k, v), x, w in zip(terms, xs, ws):
        box(ax, x, 2.72, w, .8, '', INK, fill='tint', lw=1.1, radius=.08)
        label(ax, x + .16, 3.3, k, fs=12, weight='semibold', ha='left', color=MUTED)
        label(ax, x + .16, 2.95, v, fs=14, ha='left', color=INK, weight='medium')
    # time axis
    y0 = .72
    arrow(ax, (.25, y0), (11.55, y0), color=MUTED, lw=1.5, head=13)
    t0, t1, t2 = 6.25, 8.25, 10.55
    for x, t in [(.55, 't₀ − 30 days'), (t0, 't₀'), (t1, 't₀ + 24 h'), (t2, 'days later')]:
        ax.plot([x, x], [y0 - .07, y0 + .07], color=MUTED, lw=1.5)
        label(ax, x, y0 - .3, t, fs=12.5, color=MUTED)
    ax.plot([t0, t0], [y0, 2.6], color=INK, lw=2, zorder=5)
    label(ax, t0 - .08, .98, 'prediction made', fs=12, weight='semibold', ha='right', color=INK)
    # windows
    box(ax, .55, 1.2, t0 - .65, .95, 'Inputs known before t₀', BLUE, fs=14,
        sub='rates, pressures, choke, last well test, history', sub_fs=12.5)
    box(ax, t0 + .1, 1.2, t1 - t0 - .15, .95, 'Target', ORANGE, fill='solid', fs=14,
        sub='next 24 h oil rate', sub_fs=12.5)
    box(ax, t1 + .1, 1.2, t2 - t1 + .8, .95, 'Label confirmed', SLATE, fill='white', fs=14,
        sub='allocated rate, after the fact', sub_fs=12.5, ls=(0, (3, 2)))
    # allowed / not allowed inputs
    check(ax, .8, 2.42, s=.13, color=TEAL, lw=2.6)
    label(ax, 1.02, 2.42, "yesterday's pressure is a valid input", fs=13, ha='left', color=TEAL)
    cross(ax, t0 + .32, 2.42, s=.1, color=ORANGE, lw=2.6)
    label(ax, t0 + .52, 2.42, "tomorrow's pressure is not", fs=13, ha='left', color=ORANGE)
    save('ml1_contract_timeline', f, 'Schematic prediction contract for a next-day oil-rate forecast.',
         kind='schematic')


@fig
def workflow():
    f, ax = canvas(11.8, 3.55)
    steps = [('Frame', 'target, unit,\nprediction time', INK),
             ('Audit data', 'units, gaps,\nnull codes, EDA', SLATE),
             ('Split', 'lock the test\nset early', ORANGE),
             ('Prepare', 'impute, scale,\nencode', SLATE),
             ('Train & tune', 'fit and compare\nwith CV', BLUE),
             ('Test once', 'locked pipeline,\none estimate', ORANGE),
             ('Deploy', 'monitor drift\nand errors', PURPLE)]
    x0, w, gap, y, h = .2, 1.48, .2, 1.3, 1.25
    xs = [x0 + i * (w + gap) for i in range(len(steps))]
    for i, ((name, sub_, col), x) in enumerate(zip(steps, xs)):
        solid = name == 'Test once'
        box(ax, x, y, w, h, '', col, fill='solid' if solid else 'tint', lw=1.6, radius=.1)
        label(ax, x + w / 2, y + h - .3, name, fs=14, weight='semibold',
              color='white' if solid else INK)
        label(ax, x + w / 2, y + .42, sub_, fs=12, color='white' if solid else MUTED, linespacing=1.2)
        label(ax, x + .17, y + h - .03, str(i + 1), fs=12, color=col if not solid else 'white',
              weight='semibold', va='top')
        if i:
            arrow(ax, (xs[i - 1] + w + .02, y + h / 2), (x - .02, y + h / 2), color=INK, lw=1.5, head=12)
    # CV bracket over prepare + train
    bracket(ax, xs[3], xs[4] + w, y + h + .22, 'repeated inside every CV fold', color=TEAL,
            fs=12.5, below=False)
    chip(ax, xs[2] + w / 2, y + h + .33, 'test rows set aside', color=ORANGE, fs=11.5)
    # iterate loop under prepare/train
    arrow(ax, (xs[4] + w / 2, y - .05), (xs[3] + w / 2, y - .05), color=TEAL, lw=1.5, rad=-.45, head=12)
    label(ax, (xs[3] + xs[4] + w) / 2, y - .72, 'iterate on validation scores, not test', fs=12.5,
          color=TEAL)
    # big feedback loop
    xa, xb = xs[6] + w / 2, xs[0] + w / 2
    ax.plot([xa, xa, xb], [y - .05, .12, .12], color=PURPLE, lw=1.5, zorder=1)
    arrow(ax, (xb, .12), (xb, y - .05), color=PURPLE, lw=1.5, head=12)
    label(ax, 1.2, .3, 'errors and drift in operation send you back to framing', fs=12.5,
          color=PURPLE, ha='left')
    save('ml1_workflow', f, 'Schematic supervised-learning workflow.', kind='schematic')


@fig
def energy_landscape():
    f, ax = canvas(11.8, 4.4)
    cols = [('Subsurface', [('Facies from logs', 'C', 'per depth sample'),
                            ('Porosity from logs', 'R', 'per depth sample'),
                            ('Missing sonic log', 'R', 'per depth sample')]),
            ('Drilling', [('Rate of penetration', 'R', 'per metre drilled'),
                          ('Stuck-pipe warning', 'C', 'per rig hour'),
                          ('Lithology at the bit', 'C', 'per depth sample')]),
            ('Production', [('Next-day oil rate', 'R', 'per well-day'),
                            ('ESP trip in 7 days', 'C', 'per pump-day'),
                            ('Gas-lock detection', 'C', 'per time window')]),
            ('Facilities', [('Corrosion rate', 'R', 'per pipe segment'),
                            ('Leak-alarm triage', 'C', 'per alarm'),
                            ('Compressor fault', 'C', 'per trip event')]),
            ('Power', [('Day-ahead load', 'R', 'per hour'),
                       ('Wind-farm output', 'R', 'per 10 minutes'),
                       ('Transformer fault', 'C', 'per gas-analysis sample')])]
    x0, w, gap = .05, 2.2, .15
    for i, (head, tasks) in enumerate(cols):
        x = x0 + i * (w + gap)
        box(ax, x, 3.55, w, .6, head, INK, fill='solid', fs=14.5, weight='semibold')
        for j, (task, kind, unit) in enumerate(tasks):
            yy = 2.55 - j * 1.0
            col = TEAL if kind == 'R' else PURPLE
            box(ax, x, yy, w, .85, '', col, fill='white', lw=1.3, radius=.08)
            label(ax, x + .14, yy + .56, task, fs=13, ha='left', weight='medium')
            label(ax, x + .14, yy + .24, unit, fs=12, ha='left', color=MUTED)
            chip(ax, x + w - .2, yy + .56, kind, color=col, fs=12, solid=True)
    chip(ax, 3.35, .18, 'R', color=TEAL, fs=12, solid=True)
    label(ax, 3.55, .18, 'regression: numeric target', fs=12.5, ha='left')
    chip(ax, 6.35, .18, 'C', color=PURPLE, fs=12, solid=True)
    label(ax, 6.55, .18, 'classification: class label', fs=12.5, ha='left')
    save('ml1_energy_landscape', f, 'Schematic overview of typical supervised problem formulations; '
         'no performance implied.', kind='schematic')


# ================================================================ part 2: data and EDA
def _log_axes(f, gs, i, sharey=None):
    ax = f.add_subplot(gs[0, i], sharey=sharey)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid(True, axis='both', alpha=1)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color('#9DAFC0')
    return ax


@fig
def log_tracks():
    d = _wells()
    w = d[d.well == 'W2'].iloc[40:260]
    z = w.depth.to_numpy()
    f = plt.figure(figsize=(6.3, 4.9), layout='constrained')
    gs = f.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, .62])
    a0 = _log_axes(f, gs, 0)
    a0.plot(w.GR, z, color=GREEN, lw=1.6)
    a0.set_xlim(0, 150)
    a0.set_xticks([0, 75, 150])
    a0.set_xlabel('GR (API)', color=GREEN, fontsize=13)
    a0.set_ylim(z.max(), z.min())
    a0.set_ylabel('Depth (m)')
    a1 = _log_axes(f, gs, 1, sharey=a0)
    a1.semilogx(10 ** w.LRT, z, color=INK, lw=1.6)
    a1.set_xlim(.5, 500)
    a1.set_xticks([1, 10, 100], ['1', '10', '100'])
    a1.set_xlabel('RT (Ω·m)', fontsize=13)
    a2 = _log_axes(f, gs, 2, sharey=a0)
    a2.plot(w.RHOB, z, color=RED, lw=1.6)
    a2.set_xlim(1.95, 2.95)
    a2.set_xticks([2.0, 2.45, 2.9])
    a2.set_xlabel('RHOB (g/cm³)', color=RED, fontsize=13)
    a2b = a2.twiny()
    a2.xaxis.tick_top()
    a2.xaxis.set_label_position('top')
    a2b.plot(w.NPHI, z, color=BLUE, lw=1.6, ls=(0, (4, 2)))
    a2b.set_xlim(.45, -.15)
    a2b.set_xticks([.45, .15, -.15], ['0.45', '0.15', '−0.15'])
    a2b.xaxis.tick_bottom()
    a2b.xaxis.set_label_position('bottom')
    a2b.set_xlabel('NPHI (v/v)', color=BLUE, fontsize=13)
    a2b.tick_params(axis='x', colors=BLUE, labelsize=12)
    a2b.spines['bottom'].set_visible(True)
    a3 = _log_axes(f, gs, 3, sharey=a0)
    fac = w.facies.to_numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(fac)) + 1, len(fac)]
    for s0, s1 in zip(edges[:-1], edges[1:]):
        top, bot = z[s0] - DZ / 2, z[s1 - 1] + DZ / 2
        a3.add_patch(Rectangle((0, top), 1, bot - top, fc=FAC_COL[fac[s0]], ec='white', lw=.6))
    a3.set_xlim(0, 1)
    a3.set_xticks([])
    a3.set_xlabel('Facies', fontsize=13)
    a3.grid(False)
    for a in (a1, a2, a3):
        a.tick_params(axis='y', labelleft=False)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec='none') for c in FAC_COL]
    f.legend(handles, FACIES, loc='outside lower center', ncol=5, fontsize=12, handlelength=1.1,
             handletextpad=.4, columnspacing=.9)
    save('ml1_log_tracks', f, 'Synthetic well W2 (seed 7), 33 m at 0.15 m sampling; curves generated '
         'from facies-dependent distributions with tool-like smoothing.', kind='computed')


@fig
def eda_distributions():
    d = _wells()
    f, axs = sub(3, height=4.0)
    rt = 10 ** d.LRT
    panels = [(d.GR, np.linspace(0, 160, 41), 'GR (API)', 'Gamma ray: a mixture of rocks'),
              (rt, np.linspace(0, 400, 41), 'RT (Ω·m)', 'Resistivity, linear axis'),
              (d.LRT, np.linspace(-.5, 3, 36), 'log₁₀ RT', 'Resistivity, log axis')]
    for ax, (v, bins, xl, title) in zip(axs, panels):
        ax.hist(v, bins=bins, color=BLUE, alpha=.85, edgecolor='white', linewidth=.6)
        ax.set_title(title, fontsize=14.5)
        clean(ax, xl, 'Samples' if ax is axs[0] else None)
        ax.text(.97, .95, f'skewness {skew(v):.2f}', transform=ax.transAxes, ha='right', va='top',
                fontsize=13, color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))
    axs[0].set_ylim(0, 235)
    axs[0].annotate('clean rock:\nsand, carbonate', (38, 158), xytext=(-2, 196), fontsize=12.5,
                    color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    axs[0].annotate('shaly rock', (98, 146), xytext=(92, 184), fontsize=12.5, color=MUTED,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    save('ml1_eda_distributions', f, 'Synthetic 8-well log dataset (seed 7), 3,200 depth samples; '
         'skewness computed with scipy.stats.skew.', kind='computed')


@fig
def eda_corr():
    d = _wells()
    names = ['GR', 'RHOB', 'NPHI', 'log RT', 'PE', 'DT']
    c = d[['GR', 'RHOB', 'NPHI', 'LRT', 'PE', 'DT']].corr().to_numpy()
    f, axs = half()
    ax = axs[0]
    im = ax.imshow(c, cmap=CMAP_DIV, vmin=-1, vmax=1)
    for i in range(6):
        for j in range(6):
            v = c[i, j]
            ax.text(j, i, '1' if i == j else _m(f'{v:.2f}'), ha='center', va='center', fontsize=13,
                    color=_text_on(im.cmap(im.norm(v))))
    ax.set_xticks(range(6), names, rotation=0)
    ax.set_yticks(range(6), names)
    ax.tick_params(length=0, labelsize=13, labelcolor=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = f.colorbar(im, ax=ax, shrink=.82, ticks=[-1, -.5, 0, .5, 1])
    cb.set_label('Pearson correlation r', fontsize=13)
    cb.outline.set_visible(False)
    save('ml1_eda_corr', f, 'Pearson correlations over all 3,200 samples of the synthetic 8-well '
         'dataset (seed 7).', kind='computed')


def _missing_mask():
    d = _wells()
    cols = ['GR', 'RHOB', 'NPHI', 'RT', 'PE', 'DT']
    m = pd.DataFrame(False, index=d.index, columns=cols)
    idx = {w: np.flatnonzero(d.well.to_numpy() == w) for w in d.well.unique()}
    for w in ('W3', 'W6'):
        m.loc[idx[w], 'PE'] = True                       # PE not run
    for w in ('W1', 'W2', 'W5'):
        r = idx[w]
        m.loc[r[:int(.45 * len(r))], 'DT'] = True        # sonic only over the lower interval
    r = idx['W7'][230:275]
    m.loc[r, ['RHOB', 'PE']] = True                      # washout: pad tools lose contact
    r = idx['W4'][150:172]
    m.loc[r, cols] = True                                # gap between logging runs
    return d, m


@fig
def eda_missing():
    d, m = _missing_mask()
    f = plt.figure(figsize=(11.8, 3.9), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[5.2, 1])
    ax = f.add_subplot(gs[0, 0])
    arr = m.to_numpy().T.astype(float)
    ax.imshow(arr, aspect='auto', cmap=ListedColormap([BLUE_T, INK]), interpolation='nearest')
    ax.set_yticks(range(6), m.columns)
    wells = d.well.to_numpy()
    bounds = np.flatnonzero(wells[1:] != wells[:-1]) + .5
    for b in bounds:
        ax.axvline(b, color='white', lw=3)
    centers = [np.flatnonzero(wells == w).mean() for w in pd.unique(wells)]
    ax.set_xticks(centers, pd.unique(wells))
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=13, labelcolor=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('Depth samples, grouped by well (depth increases to the right within each well)',
                  fontsize=13, color=MUTED)
    w3 = np.flatnonzero(wells == 'W3')
    w5 = np.flatnonzero(wells == 'W5')
    w4 = np.flatnonzero(wells == 'W4')
    w7 = np.flatnonzero(wells == 'W7')
    ax.text(w3.mean(), 4, 'PE not run', ha='center', va='center', fontsize=12.5, color='white',
            fontweight='semibold')
    ax.text(np.flatnonzero(wells == 'W6').mean(), 4, 'PE not run', ha='center', va='center',
            fontsize=12.5, color='white', fontweight='semibold')
    box_kw = dict(boxstyle='round,pad=0.25', fc='white', ec=ORANGE, lw=1)
    ax.annotate('upper interval not logged', (w5[178], 5), xytext=(w5[265], 5), ha='left', va='center',
                fontsize=12.5, color=INK, bbox=box_kw,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    ax.annotate('gap between logging runs', (w4[161], 2), xytext=(w4[250], 2), ha='left', va='center',
                fontsize=12.5, color=INK, bbox=box_kw,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    ax.annotate('washout', (w7[252], 1), xytext=(w7[330], 1), ha='left', va='center',
                fontsize=12.5, color=INK, bbox=box_kw,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    bx = f.add_subplot(gs[0, 1], sharey=ax)
    pct = m.mean().to_numpy() * 100
    bx.barh(range(6), pct, color=INK, height=.6)
    for i, p in enumerate(pct):
        bx.text(p + 1, i, f'{p:.1f}%', va='center', fontsize=12.5, color=INK)
    bx.set_xlim(0, 45)
    bx.set_xlabel('Share missing (%)', fontsize=13)
    bx.tick_params(axis='y', labelleft=False, length=0)
    bx.grid(axis='x', alpha=1)
    bx.set_axisbelow(True)
    save('ml1_eda_missing', f, 'Synthetic 8-well dataset (seed 7) with injected, patterned missingness: '
         'PE absent in W3 and W6, DT only in the lower 55% of W1, W2, W5, a washout in W7 and a '
         'run gap in W4.', kind='computed')


@lru_cache(maxsize=2)
def _trap_data():
    """Well table with two injected quality problems: GR sentinels (W3) and NPHI in percent (W5)."""
    d = _wells().copy()
    rng = np.random.default_rng(21)
    w3 = np.flatnonzero(d.well.to_numpy() == 'W3')
    starts = rng.choice(np.arange(20, 380), 3, replace=False)
    sent = np.unique(np.concatenate([w3[s:s + rng.integers(3, 6)] for s in starts]))
    d.loc[sent, 'GR'] = -999.25
    w5 = d.well == 'W5'
    d.loc[w5, 'NPHI'] = d.loc[w5, 'NPHI'] * 100
    return d, sent


@fig
def eda_traps():
    d, sent = _trap_data()
    w3 = d[d.well == 'W3']
    f, axs = sub(2, height=4.1)
    ax = axs[0]
    ax.plot(w3.depth, w3.GR, color=GREEN, lw=1.4)
    bad = w3.GR == -999.25
    ax.scatter(w3.depth[bad], w3.GR[bad], s=26, color=ORANGE, zorder=4)
    ax.set_ylim(-1100, 800)                                  # headroom above the log for the summary box
    ax.set_yticks(np.arange(-1000, 201, 200))
    clean(ax, 'Depth in well W3 (m)', 'GR as read (API)')
    ax.set_title('Null code −999.25 read as a measurement', fontsize=14.5)
    raw_mean, clean_mean = w3.GR.mean(), w3.GR[~bad].mean()
    ax.text(.03, .97, f'{bad.sum()} of {len(w3)} samples are −999.25\n'
                      f'mean GR as read: {raw_mean:.0f} API\n'
                      f'mean after masking: {clean_mean:.0f} API'.replace('-', '−'),
            transform=ax.transAxes, fontsize=13, color=INK, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=ORANGE, lw=1))
    ax = axs[1]
    wells = sorted(d.well.unique(), key=lambda s: int(s[1:]))
    data = [d.loc[d.well == w, 'NPHI'].to_numpy() for w in wells]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6))
    for patch, w in zip(bp['boxes'], wells):
        patch.set_facecolor(ORANGE_T if w == 'W5' else BLUE_T)
        patch.set_edgecolor(ORANGE if w == 'W5' else BLUE)
    for key in ('whiskers', 'caps'):
        for line in bp[key]:
            line.set_color('#8A9AAB')
    ax.set_xticks(range(1, 9), wells)
    clean(ax, 'Well', 'NPHI as stored')
    ax.set_title('One well stored in percent, the rest in v/v', fontsize=14.5)
    med5 = np.median(data[wells.index('W5')])
    med_other = np.median(np.concatenate([x for w, x in zip(wells, data) if w != 'W5']))
    ax.annotate(f'W5 median {med5:.1f}', (5.3, med5), xytext=(6.1, 32), fontsize=13, color=ORANGE,
                arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.9))
    ax.annotate(f'other wells: median {med_other:.2f}', (2, med_other), xytext=(.7, 12), fontsize=13,
                color=BLUE, arrowprops=dict(arrowstyle='-', color=BLUE, lw=.9))
    save('ml1_eda_traps', f, 'Synthetic 8-well dataset (seed 7) with injected problems: -999.25 null '
         'codes in W3 GR (seed 21) and W5 NPHI multiplied by 100.', kind='computed')


@fig
def eda_depth_shift():
    d = _wells()
    w = d[d.well == 'W4'].reset_index(drop=True)
    rng = np.random.default_rng(4)
    true_shift = 8                                            # samples = 1.2 m
    gr1 = np.clip(w.GR.to_numpy() + rng.normal(0, 5.0, len(w)), 0, None)      # a GR tool never reads below 0
    gr2 = np.clip(np.r_[np.full(true_shift, np.nan), w.GR.to_numpy()[:-true_shift]] + rng.normal(0, 5.0, len(w)),
                  0, None)
    z = w.depth.to_numpy()
    lags = np.arange(-20, 21)
    corr = []
    for L in lags:                                            # move run 2 up by L samples
        a = gr1[max(0, -L) + 20: len(gr1) - max(0, L) - 20]
        b = gr2[max(0, L) + 20: len(gr2) - max(0, -L) - 20]
        m = np.isfinite(a) & np.isfinite(b)
        corr.append(np.corrcoef(a[m], b[m])[0, 1])
    corr = np.array(corr)
    best = lags[np.argmax(corr)]
    f = plt.figure(figsize=(6.3, 4.9), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[1, 1.35])
    ax = f.add_subplot(gs[0, 0])
    sl = slice(150, 280)
    ax.plot(gr1[sl], z[sl], color=GREEN, lw=1.7)
    ax.plot(gr2[sl], z[sl], color=PURPLE, lw=1.7, ls=(0, (4, 2)))
    ax.set_ylim(z[sl].max(), z[sl].min())
    ax.set_xlim(-4, 160)
    clean(ax, 'GR (API)', 'Depth (m)')
    ax.text(116, z[196], 'run 1', color=GREEN, fontsize=13, va='center', fontweight='semibold')
    ax.text(116, z[208], 'run 2', color=PURPLE, fontsize=13, va='center', fontweight='semibold')
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.set_title('Same beds, two depths', fontsize=14)
    bx = f.add_subplot(gs[0, 1])
    bx.plot(lags * DZ, corr, color=INK, lw=2)
    bx.scatter([best * DZ], [corr.max()], s=70, color=ORANGE, zorder=4)
    bx.annotate(f'best shift\n{best * DZ:.2f} m\nr = {corr.max():.2f}', (best * DZ, corr.max()),
                xytext=(-2.9, corr.max() - .16), fontsize=13, color=ORANGE,
                arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.9))
    bx.axvline(0, color=MUTED, lw=1, ls=':')
    bx.text(-.3, corr[lags == 0][0], f'no shift\nr = {corr[lags == 0][0]:.2f}', fontsize=12.5,
            color=MUTED, ha='right', va='center')
    clean(bx, 'Shift applied to run 2 (m)', 'Correlation with run 1')
    bx.set_title('Cross-correlation', fontsize=14)
    save('ml1_eda_depth_shift', f, f'Synthetic well W4 (seed 7); run 2 GR recorded {true_shift * DZ:.1f} m '
         'deep with independent noise (seed 4); Pearson correlation at each trial shift.', kind='computed')


# ---------------------------------------------------------------- preprocessing
@fig
def scalers():
    rng = np.random.default_rng(5)
    bulk = rng.normal(50, 10, 40)
    out = 400.0
    with_out = np.r_[bulk, out]

    def standard(train, x):
        return (x - train.mean()) / train.std()

    def minmax(train, x):
        return (x - train.min()) / (train.max() - train.min())

    def robust(train, x):
        q1, q2, q3 = np.percentile(train, [25, 50, 75])
        return (x - q2) / (q3 - q1)

    f, axs = half(rows=3, n=1)
    for ax, (name, fn, desc) in zip(axs, [('StandardScaler', standard, 'mean and standard deviation'),
                                          ('MinMaxScaler', minmax, 'minimum and maximum'),
                                          ('RobustScaler', robust, 'median and interquartile range')]):
        for row, (train, lab, col) in enumerate([(bulk, 'fit without outlier', BLUE),
                                                 (with_out, 'fit with one outlier', ORANGE)]):
            yv = 1 - row
            t = fn(train, bulk)
            ax.scatter(t, np.full_like(t, yv), s=26, color=col, alpha=.75, edgecolors='white',
                       linewidths=.5, zorder=3)
            ax.plot([t.min(), t.max()], [yv - .28, yv - .28], color=col, lw=2.2, solid_capstyle='butt')
            if row == 1:
                o = fn(train, np.array([out]))[0]
                if o > 2.6:
                    ax.annotate('', (2.68, yv), xytext=(2.0, yv),
                                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.3))
                    ax.text(2.66, yv + .4, f'outlier at {o:.1f}', fontsize=12.5, color=ORANGE,
                            ha='right', va='center')
                else:
                    ax.scatter([o], [yv], s=46, marker='D', color=ORANGE, zorder=4)
                    ax.text(o + .1, yv + .4, 'outlier', fontsize=12.5, color=ORANGE, va='center')
        ax.set_yticks([1, 0], ['without', 'with'])
        ax.set_ylim(-.55, 1.45)
        ax.set_xlim(-2.7, 2.7)
        ax.set_title(f'{name}: {desc}', fontsize=13.5, loc='left')
        clean(ax)
        ax.tick_params(axis='y', length=0, labelsize=12.5, labelcolor=INK)
    axs[-1].set_xlabel('Scaled value of the 40 ordinary samples')
    save('ml1_scalers', f, 'Synthetic feature: 40 samples from N(50, 10), seed 5, plus one outlier at 400. '
         'Each scaler fitted with and without the outlier; bars show the range of the ordinary samples.',
         kind='computed')


def _table(ax, x, y_top, header, rows, widths, h=.4, fs=13, head_col=INK, fills=None, mono=False):
    """Small drawn table: header row and text cells. fills[(r, c)] = colour tint."""
    fills = fills or {}
    xs = np.r_[x, x + np.cumsum(widths)]
    for j, t in enumerate(header):
        ax.add_patch(Rectangle((xs[j], y_top - h), widths[j], h, fc=head_col, ec='white', lw=1.2))
        ax.text(xs[j] + widths[j] / 2, y_top - h / 2, t, ha='center', va='center', fontsize=fs - .5,
                color='white', fontweight='semibold')
    for i, row in enumerate(rows):
        yy = y_top - (i + 2) * h
        for j, t in enumerate(row):
            fc = fills.get((i, j), 'white' if i % 2 else PANEL)
            ax.add_patch(Rectangle((xs[j], yy), widths[j], h, fc=fc, ec='#D5DEE7', lw=1))
            ax.text(xs[j] + widths[j] / 2, yy + h / 2, t, ha='center', va='center', fontsize=fs,
                    color=INK, family='IBM Plex Mono' if mono else None)
    return xs[-1], y_top - (len(rows) + 1) * h


@fig
def encode_impute():
    f, ax = canvas(11.8, 3.9)
    # --- one-hot
    _panel_title(ax, .15, 3.66, 'One-hot encoding: one column per category')
    forms = ['A1 SH', 'A1 LM', 'B1 SH', 'A1 LM', 'B1 SH']
    _table(ax, .25, 3.3, ['Formation'], [[v] for v in forms], [1.25], h=.4)
    cats = ['A1 LM', 'A1 SH', 'B1 SH']
    rows = [['1' if v == c else '0' for c in cats] for v in forms]
    fills = {(i, j): BLUE_T for i, r in enumerate(rows) for j, v in enumerate(r) if v == '1'}
    arrow(ax, (1.62, 1.9), (2.12, 1.9), color=INK, lw=1.6)
    _table(ax, 2.25, 3.3, cats, rows, [1.05, 1.05, 1.05], h=.4, fills=fills)
    label(ax, .25, .38, 'Integer codes 0, 1, 2 would invent an order\nthat linear models and KNN take literally',
          fs=12.5, color=ORANGE, ha='left', linespacing=1.25)
    # --- imputation
    x0 = 6.35
    _panel_title(ax, x0 - .1, 3.66, 'Median imputation plus a missing indicator')
    raw = ['3.1', 'NaN', '2.8', 'NaN', '4.0']
    _table(ax, x0, 3.3, ['PE'], [[v] for v in raw], [1.0], h=.4,
           fills={(1, 0): ORANGE_T, (3, 0): ORANGE_T})
    arrow(ax, (x0 + 1.12, 1.9), (x0 + 1.62, 1.9), color=INK, lw=1.6)
    imp = [['3.1', '0'], ['3.1', '1'], ['2.8', '0'], ['3.1', '1'], ['4.0', '0']]
    _table(ax, x0 + 1.75, 3.3, ['PE', 'PE_missing'], imp, [1.0, 1.55], h=.4,
           fills={(1, 0): TEAL_T, (3, 0): TEAL_T, (1, 1): TEAL_T, (3, 1): TEAL_T})
    label(ax, x0 + 4.45, 2.3, 'median of the\ntraining values\n(2.8, 3.1, 4.0)\n= 3.1', fs=12.5,
          color=TEAL, ha='left', linespacing=1.25)
    label(ax, x0, .38, 'Both steps are fitted: categories and medians\ncome from training rows only',
          fs=12.5, color=TEAL, ha='left', linespacing=1.25)
    ax.plot([6.0, 6.0], [.6, 3.7], color=RULE, lw=1.2)
    save('ml1_encode_impute', f, 'Schematic worked example; the median is computed from the three '
         'observed training values.', kind='schematic')


# ================================================================ part 3: splits and CV
def _fold_rows(ax, x0, y_rows, n, val_sets, cw=.22, gap=.03, rh=.42, marks=None, texts=None):
    """Rows of train/validation cells. marks: indices that get an event dot."""
    for yy, vs in zip(y_rows, val_sets):
        for j in range(n):
            role = 'val' if j in vs else 'train'
            _cell(ax, x0 + j * (cw + gap), yy, cw, rh, role,
                  text=(texts[j] if texts else ''), fs=12)
            if marks is not None and j in marks:
                ax.add_patch(Circle((x0 + j * (cw + gap) + cw / 2, yy + rh / 2), .055, fc=INK, ec='none',
                                    zorder=6))


@fig
def cv_basic():
    y = np.zeros(16, int)
    y[[2, 7, 11, 14]] = 1
    ev = set(np.flatnonzero(y))
    f, ax = canvas(11.8, 4.2)
    cw, gap = .22, .03
    for p, (title, cv) in enumerate([('K-fold (shuffled)', KFold(4, shuffle=True, random_state=1)),
                                      ('Stratified K-fold', StratifiedKFold(4, shuffle=True, random_state=1))]):
        px = .1 + p * 6.0
        _panel_title(ax, px, 3.95, title)
        x0 = px + .95
        folds = [set(v) for _, v in cv.split(np.zeros(16), y)]
        label(ax, px, 3.33, 'Class', fs=13, ha='left', color=MUTED)
        for j in range(16):
            cx = x0 + j * (cw + gap) + cw / 2
            ax.add_patch(Circle((cx, 3.33), .07, fc=INK if j in ev else 'white', ec=INK, lw=1.1, zorder=4))
        ys = [2.6, 2.02, 1.44, .86]
        _fold_rows(ax, x0, ys, 16, folds, cw, gap, marks=ev)
        for k, (yy, vs) in enumerate(zip(ys, folds)):
            label(ax, px, yy + .21, f'Fold {k + 1}', fs=13, ha='left')
            n_ev = len(vs & ev)
            label(ax, x0 + 16 * (cw + gap) + .12, yy + .21, f'{n_ev} event{"s" if n_ev != 1 else ""}',
                  fs=13, ha='left', color=ORANGE if n_ev != 1 else TEAL, weight='semibold')
    xe = _legend_roles(ax, 1.05, .3)
    ax.add_patch(Circle((xe + .07, .3), .07, fc=INK, ec=INK, zorder=4))
    label(ax, xe + .22, .3, 'event (minority class): 4 of 16 rows', fs=12.5, ha='left')
    ax.plot([5.95, 5.95], [.7, 4.1], color=RULE, lw=1.2)
    save('ml1_cv_basic', f, 'Fold assignments computed with scikit-learn KFold and StratifiedKFold '
         '(4 folds, shuffle, random_state=1) for 16 rows with 4 events.', kind='computed')


@fig
def cv_structured():
    f, ax = canvas(11.8, 4.2)
    # grouped folds: one cell per well
    g = np.repeat(np.arange(8), 3)
    folds = [set(g[v]) for _, v in GroupKFold(4).split(g, groups=g)]
    _panel_title(ax, .1, 3.95, 'Grouped K-fold: whole wells held out')
    x0, cw, gap = 1.05, .52, .06
    for j in range(8):
        label(ax, x0 + j * (cw + gap) + cw / 2, 3.33, f'W{j + 1}', fs=13, color=MUTED)
    ys = [2.6, 2.02, 1.44, .86]
    for k, (yy, vs) in enumerate(zip(ys, folds)):
        label(ax, .1, yy + .21, f'Fold {k + 1}', fs=13, ha='left')
        for j in range(8):
            _cell(ax, x0 + j * (cw + gap), yy, cw, .42, 'val' if j in vs else 'train', text='')
    # time-series split
    px = 6.1
    _panel_title(ax, px, 3.95, 'Time-series split: train on the past only')
    x1, cw2, gap2 = px + .95, .4, .05
    splits = list(TimeSeriesSplit(4).split(np.arange(10)))
    for j in range(10):
        label(ax, x1 + j * (cw2 + gap2) + cw2 / 2, 3.33, f'{j + 1}', fs=13, color=MUTED)
    for k, (yy, (tr, va)) in enumerate(zip(ys, splits)):
        label(ax, px, yy + .21, f'Split {k + 1}', fs=13, ha='left')
        for j in range(10):
            role = 'train' if j in tr else ('val' if j in va else 'none')
            _cell(ax, x1 + j * (cw2 + gap2), yy, cw2, .42, role, text='')
    arrow(ax, (x1, .55), (x1 + 10 * (cw2 + gap2) - gap2, .55), color=MUTED, lw=1.3, head=11)
    label(ax, x1 + 5 * (cw2 + gap2), .33, 'time blocks (e.g. months)', fs=12.5, color=MUTED)
    _legend_roles(ax, .3, .33, roles=('train', 'val', 'none'))
    ax.plot([5.95, 5.95], [.7, 4.1], color=RULE, lw=1.2)
    save('ml1_cv_structured', f, 'Fold assignments computed with scikit-learn GroupKFold(4) for 8 equal-size '
         'wells and TimeSeriesSplit(4) for 10 time blocks.', kind='computed')


@fig
def fold_pipeline():
    f, ax = canvas(11.8, 3.45)
    # data stack on the left
    label(ax, .75, 3.2, 'Training data', fs=13.5, weight='semibold')
    for i in range(5):
        role = 'val' if i == 2 else 'train'
        _cell(ax, .3, 2.63 - i * .5, .9, .4, role, text=('fold k' if i == 2 else ''), fs=12)
    ax.plot([1.3, 1.55], [1.68, 2.58], color=BLUE, lw=1.2)
    ax.plot([1.3, 1.55], [1.58, .63], color=TEAL, lw=1.2)
    ytop, ybot, w, h = 2.13, .28, 1.75, .95
    steps = [('Imputer', 'medians'), ('Scaler', 'means, SDs'), ('Model', 'weights')]
    xs = [3.65, 5.95, 8.25]
    box(ax, 1.6, ytop, 1.6, h, 'Other folds', BLUE, fs=13.5, sub='training rows', sub_fs=12)
    box(ax, 1.6, ybot, 1.6, h, 'Fold k', TEAL, fs=13.5, sub='validation rows', sub_fs=12)
    prev_t, prev_b = (3.2, ytop + h / 2), (3.2, ybot + h / 2)
    for (name, learned), x in zip(steps, xs):
        box(ax, x, ytop, w, h, name, BLUE, fs=13.5, sub='fit + transform' if name != 'Model' else 'fit',
            sub_fs=12)
        box(ax, x, ybot, w, h, name, TEAL, fill='white', fs=13.5,
            sub='transform only' if name != 'Model' else 'predict', sub_fs=12)
        arrow(ax, prev_t, (x - .02, ytop + h / 2), color=INK, lw=1.5, head=12)
        arrow(ax, prev_b, (x - .02, ybot + h / 2), color=INK, lw=1.5, head=12)
        arrow(ax, (x + w / 2, ytop - .02), (x + w / 2, ybot + h + .02), color=BLUE, lw=1.4, ls=(0, (3, 2)),
              head=12)
        label(ax, x + w / 2 + .1, (ytop + ybot + h) / 2, learned, fs=12.5, color=BLUE, ha='left',
              style='italic')
        prev_t, prev_b = (x + w + .02, ytop + h / 2), (x + w + .02, ybot + h / 2)
    box(ax, 10.45, ybot, 1.2, h, 'Score k', TEAL, fill='solid', fs=13.5)
    arrow(ax, prev_b, (10.43, ybot + h / 2), color=INK, lw=1.5, head=12)
    label(ax, 11.05, ytop + h / 2 + .15, 'repeat for\nk = 1 … K,\nreport mean\nand spread', fs=12.5,
          color=MUTED, linespacing=1.2)
    label(ax, 7.0, 3.3, 'Everything the pipeline learns comes from the training rows of the fold',
          fs=13, color=INK, weight='medium')
    save('ml1_fold_pipeline', f, 'Schematic of fitting preprocessing and model inside one '
         'cross-validation fold.', kind='schematic')


# ================================================================ part 4: regression
@fig
def linear_density():
    rng = np.random.default_rng(3)
    rhob = rng.uniform(2.15, 2.60, 60)
    phi = (2.65 - rhob) / 1.65 + rng.normal(0, .012, 60)
    m = LinearRegression().fit(rhob[:, None], phi)
    w, b = m.coef_[0], m.intercept_
    f, axs = half()
    ax = axs[0]
    ax.scatter(rhob, phi, s=36, color=BLUE, edgecolors='white', linewidths=.6, zorder=3)
    xx = np.linspace(2.12, 2.63, 50)
    ax.plot(xx, b + w * xx, color=ORANGE, lw=2.6)
    x0 = 2.40
    y0 = b + w * x0
    ax.plot([x0, x0], [-.02, y0], color=MUTED, lw=1, ls=':')
    ax.plot([2.1, x0], [y0, y0], color=MUTED, lw=1, ls=':')
    ax.scatter([x0], [y0], s=70, color=ORANGE, zorder=4, edgecolors='white')
    ax.annotate(f'RHOB = {x0:.2f} g/cm³\n→ predicted porosity {y0:.3f}', (x0, y0), xytext=(2.125, .03),
                fontsize=13, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8, relpos=(.6, 1),
                                                        shrinkB=6))
    ax.text(.98, .97, _m(f'fitted: porosity = {b:.2f} {w:+.3f} × RHOB\n'
                         f'physics: slope = -1/(2.65 - 1.00) = {-1 / 1.65:.3f}').replace('−0.', '− 0.', 1),
            transform=ax.transAxes, fontsize=13, color=INK, va='top', ha='right', multialignment='left',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=ORANGE, lw=1))
    ax.set_xlim(2.1, 2.65)
    ax.set_ylim(-.02, .32)
    clean(ax, 'Bulk density RHOB (g/cm³)', 'Porosity (fraction)')
    save('ml1_linear_density', f, f'Synthetic data (seed 3): porosity = (2.65 - RHOB)/1.65 plus noise '
         f'(SD 0.012); least-squares fit intercept {b:.3f}, slope {w:.4f}.', kind='computed')


@fig
def basis_expansion():
    rng = np.random.default_rng(8)
    Q = np.sort(rng.uniform(100, 1150, 45))
    H = 1500 * (1 - (Q / 1300) ** 2.6) + rng.normal(0, 40, 45)
    mu, sd = Q.mean(), Q.std()
    z = (Q - mu) / sd
    m = LinearRegression().fit(np.column_stack([z, z ** 2, z ** 3]), H)
    lin = LinearRegression().fit(z[:, None], H)
    qq = np.linspace(80, 1170, 300)
    zz = (qq - mu) / sd
    f, axs = half(rows=2, n=1, height=4.9)
    ax = axs[0]
    ax.scatter(Q, H, s=26, color=BLUE, edgecolors='white', linewidths=.5, zorder=3)
    ax.plot(qq, lin.predict(zz[:, None]), color=SLATE, lw=1.6, ls=(0, (4, 2)))
    ax.plot(qq, m.predict(np.column_stack([zz, zz ** 2, zz ** 3])), color=ORANGE, lw=2.6)
    endlabel(ax, qq[-1], lin.predict(zz[-1:, None])[0], 'straight line', SLATE, fs=12.5, dy=-4)
    z7 = (700 - mu) / sd
    endlabel(ax, 700, m.predict(np.array([[z7, z7 ** 2, z7 ** 3]]))[0], 'cubic basis', ORANGE, fs=12.5, dy=14)
    clean(ax, None, 'Head (m)')
    ax.set_xlim(60, 1330)
    ax.tick_params(labelbottom=False)
    bx = axs[1]
    terms = [(m.coef_[0] * zz, TEAL, _m(f'w₁·z,  w₁ = {m.coef_[0]:.0f}')),
             (m.coef_[1] * zz ** 2, PURPLE, _m(f'w₂·z²,  w₂ = {m.coef_[1]:.0f}')),
             (m.coef_[2] * zz ** 3, GOLD, _m(f'w₃·z³,  w₃ = {m.coef_[2]:.0f}'))]
    for v, col, lab in terms:
        bx.plot(qq, v, color=col, lw=2.2, label=lab)
    bx.legend(loc='lower left', fontsize=12, handlelength=1.4)
    bx.axhline(0, color=MUTED, lw=.8)
    clean(bx, 'Flow rate Q (m³/d)', 'Term (m)')
    bx.set_xlim(60, 1330)
    save('ml1_basis_expansion', f, f'Synthetic head-flow data (seed 8), 45 points; cubic fit on the standardized '
         f'flow z; intercept {m.intercept_:.0f} m.', kind='computed')


def _quad_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return x[:, None], 1 + x + 1.6 * x ** 2 + rng.normal(0, .24, n)


def _poly(deg):
    return make_pipeline(PolynomialFeatures(deg), StandardScaler(), LinearRegression())


@fig
def diagnostic_curves():
    f, axs = sub(3, height=4.2)
    # validation curve on a small sample; seed 94 gives a typical draw (realized noise SD 0.238)
    X, y = _quad_data(40, 94)
    noise_sd = (y - (1 + X[:, 0] + 1.6 * X[:, 0] ** 2)).std()
    degs = np.arange(1, 13)
    cv = RepeatedKFold(n_splits=5, n_repeats=12, random_state=0)
    tr, va = validation_curve(make_pipeline(PolynomialFeatures(), StandardScaler(), LinearRegression()),
                              X, y, param_name='polynomialfeatures__degree', param_range=degs, cv=cv,
                              scoring='neg_root_mean_squared_error')
    tr, va = -tr.mean(1), -va.mean(1)
    ax = axs[0]
    ax.plot(degs, tr, '-o', color=BLUE, ms=5)
    ax.plot(degs, va, '-o', color=TEAL, ms=5)
    best = degs[np.argmin(va)]
    ax.scatter([best], [va.min()], s=110, facecolors='none', edgecolors=ORANGE, lw=2, zorder=5)
    ax.annotate(f'lowest validation\nerror: degree {best}', (best, va.min()), xytext=(1.9, .66), fontsize=12.5,
                color=ORANGE, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.9, shrinkB=7, relpos=(.06, 0)))
    ax.axhline(.24, color=MUTED, lw=1, ls=':')
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    endlabel(ax, degs[-1], tr[-1], 'training', BLUE, dy=-10, fs=12.5, ha='right', dx=-6)
    ax.text(6.1, .41, 'validation', fontsize=12.5, color=TEAL, ha='right', fontweight='medium')
    ax.text(12.3, .255, 'noise SD 0.24', fontsize=12, color=MUTED, ha='right', va='bottom')
    ax.set_ylim(0, .8)
    ax.set_title('Validation curve (40 rows)', fontsize=14.5)
    clean(ax, 'Polynomial degree', 'RMSE')
    # learning curves
    Xp, yp = _quad_data(400, 12)
    sizes = np.array([12, 20, 30, 45, 70, 100, 150, 220, 300])
    for ax, deg, title in [(axs[1], 1, 'Learning curve: degree 1'), (axs[2], 9, 'Learning curve: degree 9')]:
        ns, trs, vas = learning_curve(_poly(deg), Xp, yp, train_sizes=sizes,
                                      cv=ShuffleSplit(n_splits=40, test_size=.25, random_state=1),
                                      scoring='neg_root_mean_squared_error')
        trs, vas = -trs.mean(1), -vas.mean(1)
        ax.plot(ns, trs, '-o', color=BLUE, ms=5)
        ax.plot(ns, vas, '-o', color=TEAL, ms=5)
        ax.axhline(.24, color=MUTED, lw=1, ls=':')
        ax.set_xscale('log')
        ax.set_xticks([12, 30, 100, 300], ['12', '30', '100', '300'])
        ax.set_ylim(0, .8)
        ax.set_title(title, fontsize=14.5)
        clean(ax, 'Training rows (log scale)')
        endlabel(ax, ns[-1], trs[-1], 'training', BLUE, fs=12.5, ha='right', dx=-6, dy=-10)
        endlabel(ax, ns[-1], vas[-1], 'validation', TEAL, fs=12.5, ha='right', dx=-6, dy=10)
    axs[1].annotate('gap closes, but both\nstay above the noise:\nhigh bias', (45, .45), xytext=(15, .66),
                    fontsize=12, color=MUTED)
    axs[2].annotate('large gap with few rows:\nhigh variance', (20, .6), xytext=(33, .62), fontsize=12,
                    color=MUTED)
    axs[1].text(300, .225, 'noise SD 0.24', fontsize=12, color=MUTED, ha='right', va='top')
    save('ml1_diagnostic_curves', f, 'Synthetic data y = 1 + x + 1.6x² + noise (SD 0.24): validation curve on '
         f'40 rows (seed 94, realized noise SD {noise_sd:.3f}; 5-fold CV repeated 12 times), learning curves on '
         'a 400-row pool (seed 12, 40 shuffle splits); mean RMSE.', kind='computed')


@fig
def l1_l2_geometry():
    A = np.array([[1, .35], [.35, .6]])
    c = np.array([2.0, .75])
    loss = lambda w: (w - c) @ A @ (w - c)
    con = {'L2': lambda w: 1 - w @ w,
           'EN': lambda w: 1 - (.5 * np.abs(w).sum() + .5 * (w @ w))}
    sol = {'L1': np.array([1.0, 0.0])}    # corner; -grad (2.53, 1.60) lies inside the L1 corner cone
    for k, fn in con.items():
        sol[k] = minimize(loss, [.5, .5], constraints=[{'type': 'ineq', 'fun': fn}], method='SLSQP',
                          options={'ftol': 1e-12}).x
    g1, g2 = np.meshgrid(np.linspace(-1.5, 3.0, 300), np.linspace(-1.45, 2.1, 260))
    Z = A[0, 0] * (g1 - c[0]) ** 2 + 2 * A[0, 1] * (g1 - c[0]) * (g2 - c[1]) + A[1, 1] * (g2 - c[1]) ** 2
    f, axs = sub(3, height=4.2)
    t = np.linspace(0, 2 * np.pi, 721)
    u = np.c_[np.cos(t), np.sin(t)]
    for ax, kind, title in [(axs[0], 'L1', 'Lasso: |w₁| + |w₂| ≤ 1'),
                            (axs[1], 'EN', 'Elastic net: ½ L1 + ½ L2 ≤ 1'),
                            (axs[2], 'L2', 'Ridge: w₁² + w₂² ≤ 1')]:
        if kind == 'L1':
            ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], closed=True, fc=BLUE_T, ec=BLUE, lw=2))
        elif kind == 'L2':
            ax.add_patch(Circle((0, 0), 1, fc=BLUE_T, ec=BLUE, lw=2))
        else:
            # radius r along direction u solves .5 r |u|_1 + .5 r^2 = 1
            l1 = np.abs(u).sum(1)
            r = (-.5 * l1 + np.sqrt(.25 * l1 ** 2 + 2)) / 1.0
            ax.add_patch(Polygon(u * r[:, None], closed=True, fc=BLUE_T, ec=BLUE, lw=2))
        w = sol[kind]
        lev = loss(w)
        ax.contour(g1, g2, Z, levels=[lev * .2, lev * .5, lev * 1.7, lev * 2.8], colors=['#B8C4D0'],
                   linewidths=1)
        ax.contour(g1, g2, Z, levels=[lev], colors=[ORANGE], linewidths=2.2)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.axvline(0, color=MUTED, lw=.8)
        ax.scatter(*c, s=44, color=INK, zorder=5)
        halo = dict(boxstyle='round,pad=0.12', fc='white', ec='none')     # labels sit over contour lines
        ax.annotate('least-squares ŵ', c, xytext=(c[0], c[1] + .5), fontsize=12.5, color=INK, ha='center',
                    bbox=halo)
        ax.scatter(*w, s=80, color=ORANGE, zorder=6, edgecolors='white')
        txt = 'w₂ = 0 exactly' if kind == 'L1' else f'w = ({w[0]:.2f}, {w[1]:.2f})'
        ax.annotate(txt, w, xytext=(-.2, -1.25), fontsize=13, color=ORANGE, bbox=halo,
                    arrowprops=dict(arrowstyle='-', color=ORANGE, lw=1))
        ax.set_xlim(-1.5, 3.0)
        ax.set_ylim(-1.45, 2.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        clean(ax, 'w₁', 'w₂' if kind == 'L1' else None, grid=False)
        ax.set_xticks([-1, 0, 1, 2, 3])
        ax.set_yticks([-1, 0, 1, 2])
    save('ml1_l1_l2_geometry', f, 'Exact quadratic loss centred at (2.0, 0.75); constrained minima: L1 corner (1, 0), '
         f'elastic net ({sol["EN"][0]:.3f}, {sol["EN"][1]:.3f}), L2 ({sol["L2"][0]:.3f}, {sol["L2"][1]:.3f}) by SLSQP.',
         kind='computed')


def _ridge_core_data():
    """Reproduces the dataset of the shared ridge_path figure (same seed and draw order)."""
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(-1, 1, 32))
    rng.normal(0, .24, len(x))
    X = rng.normal(size=(120, 5))
    X[:, 1] = X[:, 0] + rng.normal(0, .2, 120)
    target = X @ np.array([2, 1, -1, .5, 0]) + rng.normal(0, 1, 120)
    return StandardScaler().fit_transform(X), target


@fig
def lasso_coef_path():
    X, y = _ridge_core_data()
    alphas, coefs, _ = sk_lasso_path(X, y - y.mean(), alphas=np.logspace(-3, .6, 160))
    f, axs = half()
    ax = axs[0]
    cols = [BLUE, ORANGE, TEAL, PURPLE, RED]
    for i in range(5):
        ax.semilogx(alphas, coefs[i], color=cols[i], lw=2.4, label=f'Feature {i + 1}')
    ax.axhline(0, color=MUTED, lw=.8)
    a_max = np.abs(X.T @ (y - y.mean())).max() / len(y)
    ax.axvline(a_max, color=MUTED, lw=1, ls=':')
    ax.text(a_max * .85, -1.12, f'all coefficients\nzero for α ≥ {a_max:.2f}', fontsize=12.5, color=MUTED,
            ha='right', va='bottom')
    ax.legend(ncol=2, loc='upper left', fontsize=12, bbox_to_anchor=(0, .88))
    clean(ax, 'Lasso penalty α (log scale)', 'Coefficient')
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ['0.001', '0.01', '0.1', '1'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-1.2, 2.8)
    save('ml1_lasso_path', f, 'Same standardized synthetic data as the shared ridge_path figure (seed 42; '
         'true coefficients 2, 1, -1, 0.5, 0; feature 2 = feature 1 + noise); scikit-learn lasso_path.',
         kind='computed')


@fig
def huber():
    rng = np.random.default_rng(6)
    fq = rng.uniform(40, 60, 40)
    q = 10 * fq + rng.normal(0, 12, 40)
    hi = np.argsort(fq)[-8:]
    out = rng.choice(hi, 5, replace=False)
    q[out] += rng.uniform(150, 260, 5)
    ols = LinearRegression().fit(fq[:, None], q)
    hub = HuberRegressor(epsilon=1.35).fit(fq[:, None], q)
    f, axs = sub(2, height=4.2)
    ax = axs[0]
    r = np.linspace(-4, 4, 400)
    ax.plot(r, .5 * r ** 2, color=ORANGE, lw=2.4)
    ax.plot(r, np.where(np.abs(r) <= 1, .5 * r ** 2, np.abs(r) - .5), color=TEAL, lw=2.4)
    ax.plot(r, np.abs(r), color=PURPLE, lw=2, ls=(0, (4, 2)))
    ax.lines[0].set_label('squared: ½r²')
    ax.lines[1].set_label('Huber (δ = 1)')
    ax.lines[2].set_label('absolute: |r|')
    ax.legend(loc='upper center', fontsize=12.5, handlelength=1.8)
    ax.set_ylim(0, 6)
    ax.set_title('Loss for one residual r', fontsize=14.5)
    clean(ax, 'Residual r = observed − predicted', 'Loss')
    ax = axs[1]
    ok = np.ones(40, bool)
    ok[out] = False
    ax.scatter(fq[ok], q[ok], s=34, color=BLUE, edgecolors='white', linewidths=.5, zorder=3)
    ax.scatter(fq[~ok], q[~ok], s=60, color=ORANGE, marker='^', edgecolors='white', linewidths=.5, zorder=3)
    xx = np.array([39, 61])
    ax.plot(xx, 10 * xx, color=MUTED, lw=1.4, ls=':')
    ax.plot(xx, ols.predict(xx[:, None]), color=ORANGE, lw=2.4)
    ax.plot(xx, hub.predict(xx[:, None]), color=TEAL, lw=2.4)
    endlabel(ax, 61, ols.predict([[61]])[0], f'least squares: {ols.coef_[0]:.1f}', ORANGE, fs=12.5)
    endlabel(ax, 61, hub.predict([[61]])[0], f'Huber: {hub.coef_[0]:.1f}', TEAL, fs=12.5, dy=-4)
    ax.text(61.3, 420, 'slopes in m³/d per Hz;\ndotted: generating\nslope 10.0', fontsize=12.5, color=MUTED,
            va='top')
    ax.annotate('5 spiky meter readings', (fq[out].mean() - .3, q[out].mean()), xytext=(40.2, 790),
                fontsize=12.5, color=ORANGE, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.9))
    ax.set_xlim(39, 70)                                      # room on the right for the line labels
    ax.set_xticks([40, 45, 50, 55, 60])
    ax.set_title('ESP liquid rate versus pump frequency', fontsize=14.5)
    clean(ax, 'Pump frequency (Hz)', 'Liquid rate (m³/d)')
    save('ml1_huber', f, f'Synthetic data (seed 6): rate = 10 x frequency + noise (SD 12) with 5 upward '
         f'spikes; OLS slope {ols.coef_[0]:.2f}, HuberRegressor(epsilon=1.35) slope {hub.coef_[0]:.2f}.',
         kind='computed')


# ================================================================ part 5: classification
def _sand_shale_1d(seed=2, n=60):
    rng = np.random.default_rng(seed)
    gr = np.r_[rng.normal(52, 15, n), rng.normal(88, 15, n)]
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return gr, y


@fig
def sigmoid_logodds():
    gr, y = _sand_shale_1d()
    m = LogisticRegression(C=1e4, max_iter=2000).fit(gr[:, None], y)
    w, b = m.coef_[0, 0], m.intercept_[0]
    g0 = -b / w
    xx = np.linspace(0, 150, 300)
    z = b + w * xx
    f, axs = half(rows=2, n=1)
    ax = axs[0]
    ax.plot(xx, z, color=INK, lw=2.4)
    ax.axhline(0, color=MUTED, lw=1, ls=':')
    ax.plot([g0, g0], [-7, 0], color=MUTED, lw=1, ls=':')        # drop line to where z = 0
    clean(ax, None, 'Log-odds z')
    ax.set_ylim(-7, 11)
    ax.tick_params(labelbottom=False)
    ax.text(3, 8.2, _m(f'z = {b:.2f} + {w:.3f} × GR'), fontsize=13.5, color=INK, ha='left')
    ax.text(3, 5.2, f'+10 API: odds × exp(10w) = {np.exp(10 * w):.2f}', fontsize=12.5,
            color=MUTED, ha='left')
    bx = axs[1]
    rng = np.random.default_rng(0)
    bx.scatter(gr[y == 0], rng.uniform(-.04, .04, (y == 0).sum()), s=22, color=BLUE, edgecolors='white',
               linewidths=.4, zorder=3)
    bx.scatter(gr[y == 1], 1 + rng.uniform(-.04, .04, (y == 1).sum()), s=26, color=ORANGE, marker='^',
               edgecolors='white', linewidths=.4, zorder=3)
    bx.plot(xx, expit(z), color=INK, lw=2.4)
    bx.axhline(.5, color=MUTED, lw=1, ls=':')
    bx.axvline(g0, color=MUTED, lw=1, ls=':')
    bx.text(g0 + 3, .45, f'GR = {g0:.0f} API\nz = 0, p = 0.5', fontsize=12.5, color=INK, va='top')
    bx.text(2, .9, 'shale (class 1)', fontsize=12.5, color=ORANGE, va='center')
    bx.text(148, .1, 'sand (class 0)', fontsize=12.5, color=BLUE, va='center', ha='right')
    clean(bx, 'Gamma ray GR (API)', 'P(shale)')
    bx.set_ylim(-.1, 1.1)
    save('ml1_sigmoid_logodds', f, f'Synthetic sand (N(52, 15)) and shale (N(88, 15)) GR, 60 each, seed 2; '
         f'logistic regression (C=1e4): intercept {b:.3f}, weight {w:.4f} per API.', kind='computed')


@fig
def logreg_boundary():
    rng = np.random.default_rng(4)
    n = 80
    X = np.r_[np.column_stack([rng.normal(45, 13, n), rng.normal(.17, .035, n)]),
              np.column_stack([rng.normal(95, 15, n), rng.normal(.29, .04, n)])]
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    m = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    f, axs = half()
    ax = axs[0]
    xl, yl = (5, 145), (.05, .42)
    xx, yy = np.meshgrid(np.linspace(*xl, 300), np.linspace(*yl, 240))
    pp = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, pp, levels=np.linspace(0, 1, 11), cmap=CMAP_DIV.reversed(), alpha=.35)
    cs = ax.contour(xx, yy, pp, levels=[.1, .5, .9], colors=[MUTED, INK, MUTED],
                    linewidths=[1.2, 2.2, 1.2], linestyles=['--', '-', '--'])
    ax.clabel(cs, fmt={.1: 'p = 0.1', .5: 'p = 0.5', .9: 'p = 0.9'}, fontsize=12, inline_spacing=4)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=28, color=BLUE, edgecolors='white', linewidths=.5, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=34, color=ORANGE, marker='^', edgecolors='white',
               linewidths=.5, zorder=3)
    ax.text(40, .395, 'shale (class 1)', fontsize=12.5, color=ORANGE, fontweight='medium')
    ax.text(8, .062, 'sand (class 0)', fontsize=12.5, color=BLUE, fontweight='medium')
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    clean(ax, 'Gamma ray GR (API)', 'Neutron porosity NPHI (v/v)', grid=False)
    save('ml1_logreg_boundary', f, 'Synthetic sand and shale samples (80 each, seed 4); standardized '
         'logistic regression (C=1); contours of predicted probability.', kind='computed')


@fig
def logloss():
    p = np.linspace(.005, .995, 400)
    f, axs = half()
    ax = axs[0]
    ax.plot(p, -np.log(p), color=ORANGE, lw=2.6)
    ax.plot(p, -np.log(1 - p), color=BLUE, lw=2.6, ls=(0, (5, 2)))
    ax.text(.06, 4.6, 'true class 1:  −ln p', fontsize=13, color=ORANGE, fontweight='medium')
    ax.text(.94, 4.6, 'true class 0:\n−ln(1 − p)', fontsize=13, color=BLUE, fontweight='medium', ha='right')
    for pv, dx, dy in [(.9, -.02, .55), (.5, .03, .55), (.1, .06, .25)]:
        v = -np.log(pv)
        ax.scatter([pv], [v], s=60, color=ORANGE, zorder=4, edgecolors='white')
        ax.annotate(f'p = {pv}: {v:.3f}', (pv, v), xytext=(pv + dx, v + dy), fontsize=12.5, color=INK,
                    ha='center' if pv != .1 else 'left')
    ax.set_ylim(0, 5.2)
    ax.set_xlim(0, 1)
    clean(ax, 'Predicted probability of class 1, p', 'Log loss for one row')
    save('ml1_logloss', f, 'Exact log-loss (binary cross-entropy) functions; marked values -ln(0.9), -ln(0.5), '
         '-ln(0.1).', kind='computed')


@fig
def knn_scaling():
    rng = np.random.default_rng(9)
    n = 25
    X = np.r_[np.column_stack([rng.normal(42, 14, n), rng.normal(2.33, .05, n)]),
              np.column_stack([rng.normal(30, 11, n), rng.normal(2.60, .05, n)])]
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    q = np.array([28.0, 2.35])
    sc = StandardScaler().fit(X)
    f, axs = half(n=2, height=4.6)
    names = ['sand', 'limestone']
    for ax, scaled in [(axs[0], False), (axs[1], True)]:
        Z = sc.transform(X) if scaled else X
        qz = sc.transform(q[None])[0] if scaled else q
        d = np.sqrt(((Z - qz) ** 2).sum(1))
        nn = np.argsort(d)[:5]
        votes = np.bincount(y[nn], minlength=2)
        if scaled:
            ax.add_patch(Circle(qz, d[nn].max(), fc=GREY_T, ec=MUTED, lw=1, ls='--', zorder=1))
        else:
            ax.axvspan(qz[0] - d[nn].max(), qz[0] + d[nn].max(), color=GREY_T, zorder=0)
        for i in nn:
            ax.plot([qz[0], Z[i, 0]], [qz[1], Z[i, 1]], color=MUTED, lw=1, zorder=2)
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=30, color=BLUE, edgecolors='white', linewidths=.5, zorder=3)
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=36, color=ORANGE, marker='^', edgecolors='white',
                   linewidths=.5, zorder=3)
        ax.scatter(Z[nn, 0], Z[nn, 1], s=120, facecolors='none', edgecolors=INK, lw=1.4, zorder=4)
        ax.scatter([qz[0]], [qz[1]], s=170, marker='*', color=INK, zorder=5)
        spot = (-1.25, -2.25) if scaled else (46, 2.47)
        ax.annotate('query', qz, xytext=spot, fontsize=12.5, color=INK, fontweight='semibold', ha='center',
                    va='center', zorder=6, arrowprops=dict(arrowstyle='-', color=INK, lw=1, shrinkB=7))
        ax.set_title(f'{"Standardized" if scaled else "Raw units"}\nvote: {votes[0]} sand, {votes[1]} limestone',
                     fontsize=13)
        if scaled:
            ax.set_aspect('equal', adjustable='datalim')
            clean(ax, 'GR (z-score)', 'RHOB (z-score)')
        else:
            clean(ax, 'GR (API)', 'RHOB (g/cm³)')
            ax.text(78, 2.69, 'limestone', fontsize=12.5, color=ORANGE, fontweight='medium', ha='right')
            ax.text(78, 2.40, 'sand', fontsize=12.5, color=BLUE, fontweight='medium', ha='right')
    save('ml1_knn_scaling', f, 'Synthetic sand and limestone samples (25 each, seed 9); query GR 28 API, '
         'RHOB 2.35 g/cm³ (star); K = 5 nearest neighbours by Euclidean distance before and after '
         'standardization.', kind='computed')


@fig
def hinge():
    m = np.linspace(-2, 3, 500)
    f, axs = half()
    ax = axs[0]
    ax.axvspan(1, 3, color=TEAL_T, zorder=0)
    ax.plot(m, (m < 0).astype(float), color=SLATE, lw=2, ls=(0, (1.5, 1.5)))
    ax.plot(m, np.maximum(0, 1 - m), color=BLUE, lw=2.8)
    ax.plot(m, np.log2(1 + np.exp(-m)), color=PURPLE, lw=2.4)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(1, color=TEAL, lw=1.2, ls='--')
    ax.lines[0].set_label('0–1 error')
    ax.lines[1].set_label('hinge (SVM)')
    ax.lines[2].set_label('logistic (base 2)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (1, 2, 0)], [labels[i] for i in (1, 2, 0)], loc='upper right',
              fontsize=12.5, facecolor='white', framealpha=1, frameon=True, edgecolor='none')
    ax.text(2.0, .95, 'correct side,\noutside the margin:\nhinge loss 0', fontsize=12.5, color=TEAL,
            ha='center', va='center')
    ax.text(.5, 2.95, 'inside\nmargin', fontsize=12, color=MUTED, ha='center', va='top')
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-2, 3)
    clean(ax, 'Margin m = y · f(x), with y = ±1', 'Loss for one row')
    save('ml1_hinge', f, 'Exact loss functions of the margin: 0-1 error, hinge max(0, 1 - m) and logistic '
         'log2(1 + exp(-m)).', kind='computed')


@fig
def kernel_lift():
    x0 = np.array([-.9, -.65, -.4, -.15, .15, .4, .65, .9])
    x1 = np.array([-2.1, -1.8, -1.55, -1.3, 1.3, 1.55, 1.8, 2.1])
    x = np.r_[x0, x1]
    y = np.r_[np.zeros(len(x0), int), np.ones(len(x1), int)]
    t = ((x0 ** 2).max() + (x1 ** 2).min()) / 2        # maximum-margin threshold on x²
    thr = np.sqrt(t)
    f = plt.figure(figsize=(6.3, 4.9), layout='constrained')
    gs = f.add_gridspec(2, 1, height_ratios=[1, 2.3])
    ax = f.add_subplot(gs[0])
    ax.scatter(x0, np.zeros_like(x0), s=60, color=BLUE, edgecolors='white', zorder=3)
    ax.scatter(x1, np.zeros_like(x1), s=70, color=ORANGE, marker='^', edgecolors='white', zorder=3)
    xx = np.linspace(-2.4, 2.4, 400)
    for r in (-thr, thr):
        ax.axvline(r, color=INK, lw=1.6, ls='--')
    ax.set_yticks([])
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-.5, .6)
    ax.spines['left'].set_visible(False)
    ax.set_title('One feature x: no single threshold works', fontsize=13.5, loc='left')
    ax.text(0, .32, f'thresholds x = ±{thr:.2f}', fontsize=12.5, ha='center', color=INK)
    clean(ax, 'x', grid=False)
    bx = f.add_subplot(gs[1])
    bx.scatter(x0, x0 ** 2, s=60, color=BLUE, edgecolors='white', zorder=3)
    bx.scatter(x1, x1 ** 2, s=70, color=ORANGE, marker='^', edgecolors='white', zorder=3)
    bx.plot(xx, xx ** 2, color=RULE, lw=1.2, zorder=1)
    bx.axhline(t, color=INK, lw=2)
    bx.axhspan((x0 ** 2).max(), (x1 ** 2).min(), color=GREY_T, zorder=0)
    bx.text(0, t + .55, f'maximum-margin line x² = {t:.2f}', fontsize=12.5, color=INK, ha='center')
    bx.set_xlim(-2.4, 2.4)
    bx.set_ylim(-.3, 5)
    bx.set_title('Lifted: φ(x) = (x, x²), now linearly separable', fontsize=13.5, loc='left')
    clean(bx, 'x', 'x²')
    save('ml1_kernel_lift', f, f'Symmetric toy data (8 points per class). In (x, x²) the maximum-margin line is '
         f'x² = {t:.2f}, midway between {(x0 ** 2).max():.2f} and {(x1 ** 2).min():.2f}; thresholds ±{thr:.3f}.',
         kind='computed')


@fig
def svm_gamma():
    X, y = _moons()
    f, axs = sub(3, height=4.2)
    for ax, g in zip(axs, [.1, 1, 30]):
        m = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma=g))
        cv = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        m.fit(X, y)
        tr = m.score(X, y)
        regions(ax, m, (-1.75, 2.7), (-1.25, 1.8), res=500)       # leftmost moon point is at x = -1.59
        points(ax, X, y, s=20)
        ax.set_title(f'γ = {g}: training {tr:.2f}, CV {cv:.2f}', fontsize=13.5)
        clean(ax, 'Feature 1', 'Feature 2' if ax is axs[0] else None, grid=False)
    save('ml1_svm_gamma', f, 'Synthetic moons (220 rows, noise 0.23, seed 42) as in the shared boundary '
         'figures; standardized RBF SVM with C = 1; accuracy on training rows and 5-fold stratified CV '
         '(seed 0).', kind='computed')


TREE_CLS = [('Sandstone', GOLD, 'o'), ('Shale', SLATE, '^'), ('Limestone', PURPLE, 's')]


def _tree_data():
    rng = np.random.default_rng(2)
    n = 60
    X = np.r_[np.column_stack([rng.normal(48, 16, n), rng.normal(2.36, .06, n)]),
              np.column_stack([rng.normal(96, 16, n), rng.normal(2.50, .06, n)]),
              np.column_stack([rng.normal(34, 12, n), rng.normal(2.60, .06, n)])]
    return X, np.repeat([0, 1, 2], n)


@fig
def tree_fitted():
    X, y = _tree_data()
    clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0).fit(X, y)
    t = clf.tree_
    names = ['GR', 'RHOB']
    units = {'GR': ' API', 'RHOB': ' g/cm³'}
    f = plt.figure(figsize=(11.8, 4.0))
    ax = f.add_axes([0, 0, 6.5 / 11.8, 1])
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 4.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # internal nodes are H_INT tall, leaves H_LEAF; the root's badge must stay below the top edge (y = 4.0)
    H_INT, H_LEAF = .84, 1.08
    pos = {0: (3.25, 3.36), 1: (1.65, 2.08), 4: (4.85, 2.08), 2: (.85, .62), 3: (2.45, .62), 5: (4.05, .62),
           6: (5.65, .62)}
    order = {0: 1, 1: 2, 4: 3}
    for node, (cx, cy) in pos.items():
        counts = t.value[node][0] * t.n_node_samples[node] if t.value[node][0].sum() <= 1.0001 else t.value[node][0]
        counts = np.round(counts).astype(int)
        n = t.n_node_samples[node]
        gini = t.impurity[node]
        if t.children_left[node] != -1:
            feat, thr = names[t.feature[node]], t.threshold[node]
            w, h = 2.25, H_INT
            box(ax, cx - w / 2, cy - h / 2, w, h, '', INK, fill='white', lw=1.5, radius=.08)
            label(ax, cx, cy + .15, f'{feat} ≤ {thr:.2f}{units[feat]}' if feat == 'RHOB' else
                  f'{feat} ≤ {thr:.1f}{units[feat]}', fs=14, weight='semibold')
            label(ax, cx, cy - .19, f'n = {n}, Gini {gini:.3f}', fs=12.5, color=MUTED)
            node_no = order[node]
            ax.add_patch(Circle((cx - w / 2, cy + h / 2), .17, fc=INK, ec='white', lw=1.2, zorder=6))
            label(ax, cx - w / 2, cy + h / 2, str(node_no), fs=12, color='white', weight='semibold', z=7)
            for child, word in [(t.children_left[node], 'yes'), (t.children_right[node], 'no')]:
                px, py = pos[child]
                ch = H_INT if t.children_left[child] != -1 else H_LEAF
                arrow(ax, (cx, cy - h / 2), (px, py + ch / 2), color=MUTED, lw=1.4, head=11)
                label(ax, (cx + px) / 2 + (-.2 if word == 'yes' else .2), (cy - h / 2 + py + ch / 2) / 2 + .05,
                      word, fs=12.5, color=MUTED, style='italic')
        else:
            k = int(np.argmax(counts))
            name, col, _ = TREE_CLS[k]
            w, h = 1.48, H_LEAF
            box(ax, cx - w / 2, cy - h / 2, w, h, '', col, fill='tint', lw=1.6, radius=.08)
            label(ax, cx, cy + .3, name, fs=13.5, weight='semibold')
            label(ax, cx, cy - .02, f'n = {n}, Gini {gini:.2f}', fs=12, color=MUTED)
            x0 = cx - w / 2 + .12
            bw = w - .24
            acc = 0
            for j, c in enumerate(counts):
                if c:
                    ax.add_patch(Rectangle((x0 + bw * acc / n, cy - .42), bw * c / n, .16, fc=TREE_CLS[j][1],
                                           ec='white', lw=.6, zorder=4))
                acc += c
    # partition panel
    bx = f.add_axes([.605, .15, .38, .8])
    gx, gy = np.meshgrid(np.linspace(0, 150, 400), np.linspace(2.15, 2.8, 300))
    pred = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    bx.contourf(gx, gy, pred, levels=[-.5, .5, 1.5, 2.5],
                colors=[tint(GOLD, .82), tint(SLATE, .85), tint(PURPLE, .85)])
    for k, (name, col, mk) in enumerate(TREE_CLS):
        bx.scatter(X[y == k, 0], X[y == k, 1], s=26, color=col, marker=mk, edgecolors='white', linewidths=.4,
                   zorder=3, label=name)
    r = t.threshold[0]
    bx.axvline(r, color=INK, lw=2)
    bx.plot([0, r], [t.threshold[1]] * 2, color=INK, lw=2)
    bx.plot([r, 150], [t.threshold[4]] * 2, color=INK, lw=2)
    # split badges sit inside the plot area, clear of the spines
    for (xx, yy), k in [((r + 1.5, 2.76), 1), ((10, t.threshold[1] + .012), 2), ((142, t.threshold[4] + .012), 3)]:
        bx.text(xx, yy, str(k), fontsize=12, color='white', fontweight='semibold', ha='center' if k != 1 else 'left',
                va='bottom' if k != 1 else 'top', bbox=dict(boxstyle='circle,pad=0.18', fc=INK, ec='none'))
    bx.set_xlim(0, 150)
    bx.set_ylim(2.15, 2.8)
    clean(bx, 'GR (API)', 'RHOB (g/cm³)', grid=False)
    bx.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.0, .93), framealpha=.9, frameon=True,
              edgecolor='none', handletextpad=.3, borderpad=.3)
    save('ml1_tree_fitted', f, f'Synthetic facies samples (60 per class, seed 2); DecisionTreeClassifier(max_depth=2, '
         f'min_samples_leaf=5) drawn from its tree_ arrays; training accuracy {clf.score(X, y):.3f}.',
         kind='computed')


@fig
def impurity():
    p = np.linspace(0, 1, 501)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = np.nan_to_num(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    gini = 2 * p * (1 - p)
    err = 1 - np.maximum(p, 1 - p)
    f, axs = half()
    ax = axs[0]
    ax.plot(p, ent, color=PURPLE, lw=2.6)
    ax.plot(p, gini, color=BLUE, lw=2.6)
    ax.plot(p, err, color=SLATE, lw=2, ls=(0, (4, 2)))
    ax.text(.5, 1.03, 'entropy (bits)', color=PURPLE, fontsize=13, ha='center', fontweight='medium')
    ax.text(.5, .535, 'Gini', color=BLUE, fontsize=13, ha='center', fontweight='medium')
    ax.text(.29, .04, 'misclassification', color=SLATE, fontsize=12.5, ha='left')
    for v, col in [(-(.8 * np.log2(.8) + .2 * np.log2(.2)), PURPLE), (2 * .8 * .2, BLUE)]:
        ax.scatter([.8], [v], s=60, color=col, zorder=4, edgecolors='white')
        ax.annotate(f'{v:.3f}', (.8, v), xytext=(8, 4), textcoords='offset points', fontsize=12.5, color=col)
    ax.axvline(.8, color=MUTED, lw=1, ls=':')
    ax.text(.815, 1.0, 'node with\n80% sand', fontsize=12, color=MUTED, va='top')
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0, 1)
    clean(ax, 'Share of one class in the node, p', 'Impurity')
    save('ml1_impurity', f, 'Exact two-class impurity functions: entropy -p log2 p - (1-p) log2(1-p), Gini 2p(1-p), '
         'misclassification 1 - max(p, 1-p).', kind='computed')


@fig
def tree_regression():
    rng = np.random.default_rng(10)
    z = np.sort(rng.uniform(1000, 3000, 120))
    phi = .35 * np.exp(-(z - 1000) / 1800) + rng.normal(0, .015, 120)
    zz = np.linspace(1000, 3900, 1500)                     # predictions continue past the deepest sample
    f, axs = half()
    ax = axs[0]
    ax.axvspan(3000, 3950, color=GREY_T, zorder=0)
    ax.text(3475, .365, 'no training data\nbelow 3,000 m', fontsize=12, color=MUTED, ha='center', va='top')
    ax.plot(zz, .35 * np.exp(-(zz - 1000) / 1800), color=MUTED, lw=1.4, ls=':', zorder=2)
    ax.text(3890, .056, 'generating trend', fontsize=12, color=MUTED, ha='right', va='center')
    ax.scatter(z, phi, s=22, color=BLUE, alpha=.8, edgecolors='white', linewidths=.4, zorder=3)
    for depth, col, dy in [(2, ORANGE, .017), (5, TEAL, -.0125)]:
        m = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(z[:, None], phi)
        pred = m.predict(zz[:, None])
        ax.plot(zz, pred, color=col, lw=2.4)
        ax.text(3890, pred[-1] + dy, f'depth {depth}: {m.get_n_leaves()} leaves', fontsize=12.5, color=col,
                fontweight='medium', ha='right', va='center')
    ax.set_xlim(950, 3950)
    ax.set_ylim(.04, .38)
    ax.set_xticks([1000, 1500, 2000, 2500, 3000, 3500])
    clean(ax, 'Depth (m)', 'Porosity (fraction)')
    save('ml1_tree_regression', f, 'Synthetic compaction trend porosity = 0.35 exp(-(z - 1000)/1800) + noise '
         '(SD 0.015), 120 samples, seed 10; DecisionTreeRegressor with max_depth 2 and 5.', kind='computed')


# ================================================================ part 6: evaluation and selection
def _esp_events():
    """Synthetic binary event data: ESP trip within 7 days (15% events), two sensor features."""
    rng = np.random.default_rng(13)
    n = 1600
    y = (rng.uniform(size=n) < .15).astype(int)
    X = np.column_stack([rng.normal(0, 1, n) + 1.2 * y, rng.normal(0, 1, n) + .9 * y])
    return X[:1000], y[:1000], X[1000:], y[1000:]


@fig
def roc():
    Xtr, ytr, Xva, yva = _esp_events()
    full = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
    weak = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr[:, [1]], ytr)
    pf = full.predict_proba(Xva)[:, 1]
    pw = weak.predict_proba(Xva[:, [1]])[:, 1]
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(pf[yva == 0], bins=bins, color=BLUE, alpha=.5, label='no event', density=True)
    ax.hist(pf[yva == 1], bins=bins, histtype='step', color=ORANGE, lw=2.2, label='event', density=True)
    ops = {}
    for t, col in [(.5, INK), (.2, TEAL)]:
        pred = pf >= t
        tpr = (pred & (yva == 1)).sum() / (yva == 1).sum()
        fpr = (pred & (yva == 0)).sum() / (yva == 0).sum()
        ops[t] = (fpr, tpr, col)
        ax.axvline(t, color=col, lw=2, ls='--')
        ax.text(t + .015, ax.get_ylim()[1] * (.93 if t == .5 else .72), f'threshold {t}\nTPR {tpr:.2f}\nFPR {fpr:.2f}',
                fontsize=12.5, color=col, va='top')
    ax.legend(loc='upper right', fontsize=12.5, handlelength=1.2)
    clean(ax, 'Predicted event probability', 'Density (validation rows)')
    ax.set_title('Scores for the two classes', fontsize=14.5)
    bx = axs[1]
    for k, (p, col, ls, name) in enumerate([(pf, INK, '-', 'two sensors'), (pw, SLATE, (0, (4, 2)), 'one sensor')]):
        fpr, tpr, _ = roc_curve(yva, p)
        auc = roc_auc_score(yva, p)
        bx.plot(fpr, tpr, color=col, lw=2.4, ls=ls)
        bx.plot([.5, .6], [.2 - k * .09] * 2, color=col, lw=2.4, ls=ls)
        bx.text(.63, .2 - k * .09, f'{name}: AUC {auc:.2f}', fontsize=12.5, color=col, va='center')
    bx.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=':')
    bx.text(.63, .47, 'chance: AUC 0.5', fontsize=12, color=MUTED)
    for t, (fpr, tpr, col) in ops.items():
        bx.scatter([fpr], [tpr], s=80, color=col, zorder=5, edgecolors='white')
        xt = (.16, .035) if t == .5 else (.02, .92)          # 0.5 label sits below the chance diagonal
        bx.annotate(f'threshold {t}', (fpr, tpr), xytext=xt, fontsize=12.5, color=col,
                    arrowprops=dict(arrowstyle='-', color=col, lw=.9, relpos=(0, .5) if t == .5 else (.5, .5)))
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1.02)
    bx.set_aspect('equal', adjustable='box')
    clean(bx, 'False positive rate (FPR)', 'True positive rate (TPR)')
    bx.set_title('ROC curve', fontsize=14.5)
    save('ml1_roc', f, f'Synthetic ESP-trip events (15% prevalence, seed 13): logistic regression trained on 1,000 rows, '
         f'evaluated on 600 validation rows; AUC two sensors {roc_auc_score(yva, pf):.3f}, one sensor '
         f'{roc_auc_score(yva, pw):.3f}.', kind='computed')


@lru_cache(maxsize=1)
def _facies_oof():
    X, y, g = _xy()
    pred = cross_val_predict(make_pipeline(StandardScaler(), SVC()), X, y, cv=LeaveOneGroupOut(), groups=g)
    return y, pred


@fig
def facies_confusion():
    y, pred = _facies_oof()
    cm = confusion_matrix(y, pred)
    rec = recall_score(y, pred, average=None)
    f = plt.figure(figsize=(11.8, 4.5), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = f.add_subplot(gs[0])
    frac = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(frac, cmap=CMAP_SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=13,
                    color=_text_on(im.cmap(im.norm(frac[i, j]))), fontweight='semibold' if i == j else 'normal')
    short = ['Sand', 'Silt', 'Shale', 'Lime', 'Dolo']
    ax.set_xticks(range(5), short)
    ax.set_yticks(range(5), FACIES)
    ax.set_xlabel('Predicted facies')
    ax.set_ylabel('Actual facies')
    ax.tick_params(length=0, labelsize=13, labelcolor=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((2.5, 3.5), 1, 1, fc='none', ec=ORANGE, lw=2.5))
    bx = f.add_subplot(gs[1])
    order = np.arange(5)[::-1]
    bx.barh(order, rec, color=[FAC_COL[i] for i in range(5)], height=.62)
    for i in range(5):
        bx.text(rec[i] - .02, order[i], f'{rec[i]:.2f}', va='center', ha='right', fontsize=13, color='white',
                fontweight='semibold')
        bx.text(1.04, order[i], f'n = {cm[i].sum()}', va='center', fontsize=12.5, color=MUTED)
    macro = rec.mean()
    weighted = (rec * cm.sum(1)).sum() / cm.sum()
    bx.axvline(macro, color=INK, lw=1.6, ls='--', zorder=.9)          # beneath the bars and their labels
    bx.axvline(weighted, color=MUTED, lw=1.6, ls=':', zorder=.9)
    bx.text(macro - .01, 5.05, f'macro\n{macro:.2f}', ha='right', va='bottom', fontsize=12.5, color=INK)
    bx.text(weighted + .01, 5.05, f'weighted\n{weighted:.2f}', ha='left', va='bottom', fontsize=12.5, color=MUTED)
    bx.set_yticks(order, FACIES)
    bx.set_xlim(0, 1.25)
    bx.set_ylim(-.6, 5.9)
    bx.set_xticks([0, .25, .5, .75, 1])
    clean(bx, 'Recall (share of each facies found)')
    bx.tick_params(axis='y', length=0, labelsize=13, labelcolor=INK)
    bx.grid(axis='y', visible=False)
    save('ml1_facies_confusion', f, f'Synthetic 8-well facies data (seed 7); standardized RBF SVM, leave-one-well-out '
         f'out-of-fold predictions; macro recall {macro:.3f}, weighted recall (= accuracy) {weighted:.3f}.',
         kind='computed')


@lru_cache(maxsize=1)
def _grid():
    d = _wells()
    train = d[~d.well.isin(['W7', 'W8'])]
    test = d[d.well.isin(['W7', 'W8'])]
    Cs, gammas = [.1, 1, 10, 100], [.01, .1, 1, 10]
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {'svc__C': Cs, 'svc__gamma': gammas},
                          cv=GroupKFold(3), scoring='f1_macro')
    search.fit(train[FEATS].to_numpy(), train.facies.to_numpy(), groups=train.well.to_numpy())
    res = pd.DataFrame(search.cv_results_)
    table = res.pivot_table(index='param_svc__C', columns='param_svc__gamma', values='mean_test_score')
    from sklearn.metrics import f1_score
    test_f1 = f1_score(test.facies, search.predict(test[FEATS].to_numpy()), average='macro')
    return table, search.best_params_, search.best_score_, test_f1


@fig
def grid_search():
    table, best, best_score, test_f1 = _grid()
    f = plt.figure(figsize=(11.8, 4.0))
    ax = f.add_axes([0, 0, 6.0 / 11.8, 1])
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 4.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    steps = [('Grid: 4 values of C × 4 of γ = 16 settings', TEAL, 'tint'),
             ('Each setting: grouped 3-fold CV on 6 training wells', TEAL, 'tint'),
             (f'Keep the best: C = {best["svc__C"]:g}, γ = {best["svc__gamma"]:g}, CV macro F1 {best_score:.2f}', TEAL,
              'tint'),
             ('Refit that pipeline on all 6 training wells', BLUE, 'tint'),
             (f'Score once on test wells W7, W8: macro F1 {test_f1:.2f}', ORANGE, 'solid')]
    y0, h, gap = 3.36, .56, .19
    for i, (t, col, fill) in enumerate(steps):
        yy = y0 - i * (h + gap)
        box(ax, .15, yy, 5.7, h, t, col, fill=fill, fs=13, weight='medium', ha='left', pad_x=.2)
        label(ax, .15 - .02, yy + h / 2, '', fs=12)
        if i:
            arrow(ax, (3.0, yy + h + gap - .02), (3.0, yy + h + .02), color=INK, lw=1.4, head=11)
    bx = f.add_axes([.585, .13, .34, .76])
    vals = table.to_numpy()
    im = bx.imshow(vals, cmap=CMAP_TEAL, vmin=0, vmax=1)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            bx.text(j, i, f'{vals[i, j]:.2f}', ha='center', va='center', fontsize=13,
                    color='white' if vals[i, j] > .6 else INK)
    bi = list(table.index).index(best['svc__C'])
    bj = list(table.columns).index(best['svc__gamma'])
    bx.add_patch(Rectangle((bj - .5, bi - .5), 1, 1, fc='none', ec=INK, lw=3))
    bx.set_xticks(range(4), [f'{v:g}' for v in table.columns])
    bx.set_yticks(range(4), [f'{v:g}' for v in table.index])
    bx.set_xlabel('γ (larger = more local)')
    bx.set_ylabel('C (penalty)')
    bx.tick_params(length=0, labelsize=13, labelcolor=INK)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_title('Mean CV macro F1', fontsize=14)
    save('ml1_grid_search', f, f'Synthetic 8-well facies data (seed 7); wells W1-W6 for GridSearchCV(GroupKFold(3), '
         f'f1_macro) over a standardized RBF SVM; best {best}, CV {best_score:.3f}; test wells W7-W8 macro F1 '
         f'{test_f1:.3f}.', kind='computed')


@fig
def model_compare():
    X, y, g = _xy()
    models = [('Majority\nclass', DummyClassifier()),
              ('Logistic\nregression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
              ('KNN\n(K = 15)', make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
              ('SVM\n(RBF)', make_pipeline(StandardScaler(), SVC())),
              ('Tree\n(depth 5)', DecisionTreeClassifier(max_depth=5, random_state=0))]
    cv = LeaveOneGroupOut()
    scores = np.array([cross_val_score(m, X, y, cv=cv, groups=g, scoring='f1_macro') for _, m in models])
    f, axs = sub(1, height=3.9)
    ax = axs[0]
    xs = np.arange(len(models))
    for w in range(scores.shape[1]):
        ax.plot(xs, scores[:, w], color='#C3CDD8', lw=1, zorder=1)
    for k in range(len(models)):
        col = SLATE if k == 0 else BLUE
        ax.scatter(np.full(scores.shape[1], xs[k]), scores[k], s=46, color=col, edgecolors='white', lw=.8,
                   zorder=3)
        m = scores[k].mean()
        ax.plot([xs[k] - .22, xs[k] + .22], [m, m], color=INK, lw=2.6, zorder=4)
        ax.text(xs[k] + .27, m, f'{m:.2f}', va='center', fontsize=13, color=INK, fontweight='semibold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))    # grey pairing lines pass behind
    wins = int((scores[1] > scores[3]).sum())
    ax.text(4.55, .98, f'Logistic beats SVM on {wins} of 8 wells;\nmean difference {scores[1].mean() - scores[3].mean():+.3f}',
            fontsize=12.5, color=MUTED, ha='right', va='top')
    ax.set_xticks(xs, [n for n, _ in models])
    ax.set_xlim(-.5, 4.7)
    ax.set_ylim(0, 1)
    clean(ax, None, 'Macro F1 on the held-out well')
    ax.tick_params(axis='x', length=0, labelsize=13, labelcolor=INK)
    ax.grid(axis='x', visible=False)
    save('ml1_model_compare', f, 'Synthetic 8-well facies data (seed 7); leave-one-well-out macro F1 for five '
         'pipelines on identical folds; grey lines join scores of the same held-out well; bars are means.',
         kind='computed')
