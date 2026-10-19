"""Figures for module key 'ml2' (ensembles, interpretation and pipelines).

Figure names start with 'ml2_'. Two synthetic energy datasets are shared by
several figures and generated deterministically below:

* facies_wells(): eight synthetic wells with GR, RHOB, NPHI and PE logs and
  three facies (sandstone, shale, limestone), with well-level calibration
  offsets and depth-correlated noise so that rows of one well are related.
* rop_wells(): six synthetic drilling wells with weight on bit, rotary speed,
  flow rate, bit hours and a formation label; rate of penetration (ROP) comes
  from an illustrative formula, not a field-calibrated drilling model.

Colour meaning used throughout the deck: class 0/1 = BLUE/ORANGE; training =
BLUE, validation = TEAL, test = ORANGE; single tree = grey, forest/bagging =
BLUE, boosting = ORANGE, stacking = PURPLE; informative feature = BLUE,
correlated copy = TEAL, noise feature = ORANGE.
"""
from functools import lru_cache
from itertools import combinations
from math import comb, factorial

import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_moons, make_classification, make_friedman1
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, BaggingClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

KEY = 'ml2'
fig = register(KEY)

GREY = '#56687A'
LIGHT = '#C9D3DD'
FACIES = ['Sandstone', 'Shale', 'Limestone']
FAC_COL = [GOLD, '#7D8B99', BLUE]
LOGS = ['GR', 'RHOB', 'NPHI', 'PE']


# ------------------------------------------------------------------ data
def _ar1(rng, n, phi=.85):
    z = rng.normal(size=n)
    e = np.empty(n)
    e[0] = z[0]
    for i in range(1, n):
        e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * z[i]
    return e


@lru_cache(maxsize=None)
def facies_wells(n_wells: int = 8, n: int = 160, seed: int = 7) -> pd.DataFrame:
    """Synthetic well logs with facies labels (depth step 0.5 m).

    Each well has its own calibration offset per log and depth-correlated noise
    (AR(1), phi = 0.95), so neighbouring rows of one well are strongly related.
    """
    rng = np.random.default_rng(seed)
    means = {'GR': [50, 95, 38], 'RHOB': [2.36, 2.49, 2.60], 'NPHI': [.20, .28, .12], 'PE': [2.2, 3.1, 4.3]}
    sds = {'GR': [11, 12, 9], 'RHOB': [.05, .05, .05], 'NPHI': [.04, .04, .035], 'PE': [.45, .45, .5]}
    offs_sd = {'GR': 12, 'RHOB': .04, 'NPHI': .03, 'PE': .45}
    P = np.array([[.93, .05, .02], [.04, .93, .03], [.03, .05, .92]])
    frames = []
    for w in range(n_wells):
        f = np.empty(n, int)
        f[0] = rng.integers(3)
        for i in range(1, n):
            f[i] = rng.choice(3, p=P[f[i - 1]])
        d = {'WELL': f'W{w + 1}', 'DEPTH': 1500 + .5 * np.arange(n), 'FACIES': f}
        for k in LOGS:
            m = np.asarray(means[k], float)
            m = m.mean() + .8 * (m - m.mean())
            base = np.convolve(np.pad(m[f], 2, mode='edge'), np.ones(5) / 5, mode='valid')
            noise = np.asarray(sds[k])[f] * (.9 * _ar1(rng, n, .95) + .35 * rng.normal(size=n))
            d[k] = base + 1.3 * rng.normal(0, offs_sd[k]) + noise
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


FORMATIONS = ['Claystone', 'Sandstone', 'Limestone', 'Dolomite']
UCS = np.array([30., 45., 70., 105.])     # illustrative strengths (MPa) behind the synthetic ROP


@lru_cache(maxsize=None)
def rop_wells(n_wells: int = 6, n: int = 300, seed: int = 11) -> pd.DataFrame:
    """Synthetic drilling records, one row per metre; ROP from an illustrative formula."""
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_wells):
        depth = 2000 + np.arange(n) * 1.0
        form = np.empty(n, int)
        i = 0
        while i < n:                                    # layered formations 15-60 m thick
            t = rng.integers(15, 60)
            form[i:i + t] = rng.integers(4)
            i += t
        def setpoints(lo, hi, step_lo=20, step_hi=70):   # driller setpoints, piecewise constant
            out = np.empty(n)
            j = 0
            while j < n:
                t = rng.integers(step_lo, step_hi)
                out[j:j + t] = rng.uniform(lo, hi)
                j += t
            return out
        wob = setpoints(60, 220) + rng.normal(0, 6, n)
        rpm = setpoints(70, 190) + rng.normal(0, 4, n)
        flow = setpoints(1800, 3400) + rng.normal(0, 40, n)
        hours = np.zeros(n)
        trip = rng.integers(120, 200)
        for j in range(1, n):
            hours[j] = 0 if j == trip else hours[j - 1] + rng.uniform(.08, .16)
        ucs = UCS[form] * np.exp(rng.normal(0, .08, n))
        hard = ucs > 60
        rpm_exp = np.where(hard, .15, .75)            # illustrative interaction: rpm helps less in hard rock
        rop = (32 * (wob / 120) ** .9 * (rpm / 120) ** rpm_exp * (40 / ucs) ** .8
               * (1 - .012 * hours) * (1 - np.exp(-flow / 1400)) * np.exp(rng.normal(0, .12, n)))
        frames.append(pd.DataFrame({'WELL': f'D{w + 1}', 'DEPTH': depth, 'WOB': wob, 'RPM': rpm,
                                    'FLOW': flow, 'BIT_HOURS': hours, 'FORMATION': form, 'ROP': rop}))
    return pd.concat(frames, ignore_index=True)


ROP_X = ['WOB', 'RPM', 'FLOW', 'BIT_HOURS', 'FORMATION']


def p_majority(n, p):
    """Probability that more than half of n independent voters are correct."""
    k = np.arange(n // 2 + 1, n + 1)
    return float(sum(comb(n, int(j)) * p ** j * (1 - p) ** (n - j) for j in k))


# ------------------------------------------------------------ helpers
def tree_nodes(ax, nodes, x0, x1, top, dy, color=BLUE, r=.13, fs=12, leaf_w=.30, leaf_h=.22,
               edge=GREY, labels=None, node_fill='tint', leaf_color=None, lw=1.4, node_w=0.0,
               node_h=.3, node_color=None):
    """Draw a binary tree given the set of internal-node paths ('' = root, 'L', 'LR', ...).

    Children of internal nodes that are not themselves internal are leaves. Internal
    nodes are circles of radius r, or rounded boxes node_w wide when node_w > 0.
    Returns a dict path -> (x, y).
    """
    nodes = set(nodes)
    pos = {}

    def place(path, lo, hi, depth):
        x = (lo + hi) / 2
        y = top - depth * dy
        pos[path] = (x, y)
        if path in nodes:
            place(path + 'L', lo, x, depth + 1)
            place(path + 'R', x, hi, depth + 1)

    place('', x0, x1, 0)
    for path, (x, y) in pos.items():
        if path:
            px, py = pos[path[:-1]]
            ax.plot([px, x], [py, y], color=edge, lw=lw, zorder=1)
    for path, (x, y) in pos.items():
        if path in nodes:
            txt = labels.get(path, '') if labels else ''
            col = node_color(path) if callable(node_color) else color
            if node_w:
                box(ax, x - node_w / 2, y - node_h / 2, node_w, node_h, txt, col, fill=node_fill, fs=fs,
                    z=3, radius=.06, lw=1.3)
            else:
                node(ax, x, y, r, col, node_fill, txt, fs=fs, z=3)
        else:
            lc = leaf_color(path) if callable(leaf_color) else (leaf_color or LIGHT)
            ax.add_patch(FancyBboxPatch((x - leaf_w / 2, y - leaf_h / 2), leaf_w, leaf_h,
                                        boxstyle='round,pad=0,rounding_size=.04', fc=tint(lc, .55),
                                        ec=lc, lw=1.1, zorder=3))
    return pos


def vdots(ax, x, y, gap=.13, r=.028, color=MUTED):
    """Vertical ellipsis drawn as three dots (the font has no U+22EE glyph)."""
    for k in (-1, 0, 1):
        ax.add_patch(Circle((x, y + k * gap), r, fc=color, ec='none', zorder=4))


def hdots(ax, x, y, gap=.13, r=.028, color=MUTED):
    for k in (-1, 0, 1):
        ax.add_patch(Circle((x + k * gap, y), r, fc=color, ec='none', zorder=4))


def hbar_label(ax, y, x, text, color=INK, fs=12, dx=.01, ha='left'):
    ax.text(x + dx, y, text, va='center', ha=ha, fontsize=fs, color=color)


# =================================================================== PART 0: case
@fig
def rop_motivation():
    d = rop_wells()
    train = d[d.WELL != 'D6']
    test = d[d.WELL == 'D6'].sort_values('DEPTH')
    Xtr, ytr = train[ROP_X].values, train.ROP.values
    Xte = test[ROP_X].values
    f, axs = sub(2, height=4.3)
    rng = np.random.default_rng(3)
    preds = {'tree': [], 'forest': []}
    for r in range(5):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        preds['tree'].append(DecisionTreeRegressor(random_state=r).fit(Xtr[idx], ytr[idx]).predict(Xte))
        preds['forest'].append(RandomForestRegressor(200, min_samples_leaf=3, random_state=r, n_jobs=-1)
                               .fit(Xtr[idx], ytr[idx]).predict(Xte))
    spread = {k: np.mean(np.std(np.array(v), axis=0)) for k, v in preds.items()}
    for ax, k, col, title in [(axs[0], 'tree', GREY, 'Five deep trees'), (axs[1], 'forest', BLUE, 'Five random forests')]:
        ax.scatter(test.DEPTH, test.ROP, s=7, c='#AEBBC8', zorder=1, label='Observed ROP')
        for p in preds[k]:
            ax.plot(test.DEPTH, p, color=col, lw=1.4 if k == 'tree' else 1.8, alpha=.75, zorder=2)
        ax.set_title(f'{title}: mean spread {spread[k]:.2f} m/h')
        ax.set_xlim(test.DEPTH.min(), test.DEPTH.max())
        ax.set_ylim(0, 58)
        clean(ax, 'Measured depth on held-out well D6 (m)', 'ROP (m/h)' if ax is axs[0] else None)
    axs[0].text(2003, 53, 'grey dots: observed ROP', fontsize=12, color=MUTED)
    save('ml2_rop_motivation', f, 'Synthetic drilling wells (seed 11); five bootstrap resamples of wells D1-D5 '
         '(seed 3); each model predicts held-out well D6. Spread = standard deviation across the five models, '
         f'averaged over depth (trees {spread["tree"]:.2f}, forests {spread["forest"]:.2f} m/h).', kind='computed')


# =================================================================== PART 1: voting
@fig
def vote_pmf():
    n, p = 5, .7
    k = np.arange(n + 1)
    pmf = np.array([comb(n, int(j)) * p ** j * (1 - p) ** (n - j) for j in k])
    f, axs = half()
    ax = axs[0]
    cols = [ORANGE if j <= n // 2 else BLUE for j in k]
    ax.bar(k, pmf, .68, color=cols, zorder=3)
    for j, v in zip(k, pmf):
        ax.text(j, v + .012, f'{v:.3f}', ha='center', va='bottom', fontsize=12, color=INK)
    ax.set_xticks(k)
    ax.set_ylim(0, .47)
    clean(ax, 'Number of the 5 voters that are correct', 'Probability')
    ax.grid(axis='x', visible=False)
    maj = pmf[3:].sum()
    ax.text(4, .43, f'Majority correct\n{maj:.3f}', ha='center', va='center', fontsize=13, color=BLUE,
            fontweight='semibold')
    ax.text(1, .43, f'Majority wrong\n{1 - maj:.3f}', ha='center', va='center', fontsize=13, color=ORANGE,
            fontweight='semibold')
    ax.axvline(2.5, color=MUTED, lw=1.2, ls='--', zorder=2)
    save('ml2_vote_pmf', f, 'Exact binomial probabilities, n = 5 independent voters, each correct with p = 0.7.',
         kind='computed')


@fig
def vote_curves():
    ns = np.arange(1, 52, 2)
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    for p, col in [(.7, BLUE), (.6, TEAL), (.55, PURPLE), (.45, ORANGE)]:
        acc = [p_majority(n, p) for n in ns]
        ax.plot(ns, acc, color=col, ls='--' if p < .5 else '-', lw=2.4)
        endlabel(ax, ns[-1], acc[-1], f'p = {p}', col, fs=12.5)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.03)
    ax.set_title('Independent voters')
    clean(ax, 'Number of voters (odd)', 'Majority-vote accuracy')
    ax = axs[1]
    p = .6
    for rho, col in [(0, TEAL), (.1, BLUE), (.3, PURPLE), (.6, ORANGE)]:
        acc = [rho * p + (1 - rho) * p_majority(n, p) for n in ns]
        ax.plot(ns, acc, color=col, lw=2.4)
        endlabel(ax, ns[-1], acc[-1], f'ρ = {rho}', col, fs=12.5)
    ax.axhline(p, color=MUTED, lw=1, ls=':')
    ax.text(2, p - .035, 'one voter: 0.60', fontsize=12, color=MUTED)
    ax.set_xlim(0, 60)
    ax.set_ylim(.5, 1.01)
    ax.set_title('Voters with p = 0.6 and shared errors')
    clean(ax, 'Number of voters (odd)', None)
    save('ml2_vote_curves', f, 'Exact binomial majority probabilities. Right: with probability rho all voters '
         'copy one shared vote (pairwise correlation of correctness = rho), otherwise they vote independently; '
         'accuracy = rho*p + (1-rho)*P(majority).', kind='computed')


@fig
def hard_soft():
    probs = np.array([.95, .40, .45])
    share = (probs >= .5).mean()
    f, axs = half()
    ax = axs[0]
    y = np.array([4.4, 3.5, 2.6, 1.25, .35])
    vals = list(probs) + [share, probs.mean()]
    names = ['Model A', 'Model B', 'Model C', 'Hard vote', 'Soft vote']
    cols = [ORANGE if v >= .5 else BLUE for v in probs] + [GREY, PURPLE]
    fills = [tint(c, .45) for c in cols[:3]] + [tint(GREY, .45), PURPLE]
    ax.barh(y, vals, .6, color=fills, edgecolor=cols, lw=1.5, zorder=3)
    ax.axvline(.5, color=INK, lw=1.3, ls='--', zorder=4)
    ax.text(.515, 5.02, 'threshold 0.5', ha='left', fontsize=12.5, color=INK)
    notes = ['probability', 'probability', 'probability', 'share voting event', 'mean probability']
    for yy, v, i in zip(y, vals, range(5)):
        verdict = 'event' if v >= .5 else 'no event'
        vc = ORANGE if v >= .5 else BLUE
        ax.text(1.03, yy + .1, f'{v:.2f}', va='center', ha='left', fontsize=13.5, color=INK, fontweight='semibold')
        ax.text(1.03, yy - .2, '→ ' + verdict, va='center', ha='left', fontsize=12.5, color=vc,
                fontweight='semibold' if i >= 3 else 'normal')
    ax.axhline(1.93, color=RULE, lw=1.4)
    ax.set_yticks(y, names)
    for t, i in zip(ax.get_yticklabels(), range(5)):
        if i >= 3:
            t.set_fontweight('semibold')
            t.set_color(INK)
    ax.set_xlim(0, 1.36)
    ax.set_ylim(-.15, 5.3)
    ax.set_xticks([0, .25, .5, .75, 1])
    clean(ax, 'Probability of event, or share of votes', None)
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save('ml2_hard_soft', f, 'Exact arithmetic for three illustrative probabilities (0.95, 0.40, 0.45): '
         'hard vote share 1/3, soft vote mean 0.60.', kind='computed')


# =================================================================== PART 2: bagging and forests
@fig
def tree_instability():
    X, y = make_moons(n_samples=220, noise=.23, random_state=42)
    xlim, ylim = (-1.6, 2.7), (-1.25, 1.8)
    f, axs = sub(4, height=4.0)
    rng = np.random.default_rng(18)
    for k, ax in enumerate(axs[:3]):
        idx = rng.integers(0, len(X), len(X))
        m = DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])
        regions(ax, m, xlim, ylim)
        inbag = np.zeros(len(X), bool)
        inbag[idx] = True
        ax.scatter(X[~inbag, 0], X[~inbag, 1], s=16, facecolors='none', edgecolors='#9AA8B6', lw=.8, zorder=2)
        points(ax, X[inbag], y[inbag], s=15)
        t = m.tree_
        fn = ['x₁', 'x₂'][t.feature[0]]
        ax.set_title(f'Sample {k + 1}: root {fn} ≤ {t.threshold[0]:.2f}', fontsize=13.5)
        ax.text(-1.5, 1.72, f'{m.get_n_leaves()} leaves', fontsize=12, color=INK, va='top',
                bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.85))
        clean(ax, 'Feature x₁', 'Feature x₂' if k == 0 else None, grid=False)
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0).fit(X, y)
    ax = axs[3]
    regions(ax, bag, xlim, ylim)
    points(ax, X, y, s=15)
    ax.set_title('Vote of 200 bagged trees', fontsize=13.5)
    clean(ax, 'Feature x₁', None, grid=False)
    save('ml2_tree_instability', f, 'Synthetic two-moons data (n = 220, noise 0.23, seed 42); bootstrap seed 18; '
         'unpruned scikit-learn trees. Hollow markers are rows left out of that bootstrap sample.', kind='computed')


@fig
def forest_diagram():
    f, ax = canvas(11.8, 4.3)
    rows, cols = 8, 4
    cell = .22
    x0, ytop = .25, 3.05
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x0 + j * cell, ytop - (i + 1) * cell), cell, cell, fc=tint(GREY, .8),
                                   ec='white', lw=1.2, zorder=2))
    label(ax, x0 + cols * cell / 2, ytop + .25, 'Training data', fs=14, weight='semibold')
    label(ax, x0 + cols * cell / 2, ytop - rows * cell - .25, 'n rows × p features', fs=12.5, color=MUTED)
    ys = [3.3, 2.05, .8]
    strips = [[1, 0, 1, 1, 1, 2, 2, 0], [2, 1, 1, 2, 1, 0, 0, 1], [2, 3, 0, 0, 0, 1, 1, 1]]
    feats = [['GR', 'PE', 'NPHI'], ['RHOB', 'GR', 'PE'], ['NPHI', 'RHOB', 'GR']]
    probs = [.82, .64, .71]
    for k, yc in enumerate(ys):
        bx = 2.05
        for i, cnt in enumerate(strips[k]):
            fc = BLUE if cnt >= 2 else (tint(BLUE, .55) if cnt == 1 else 'white')
            ax.add_patch(Rectangle((bx + i * .2, yc - .12), .2, .24, fc=fc, ec=BLUE if cnt else LIGHT, lw=1,
                                   zorder=2))
        arrow(ax, (x0 + cols * cell + .1, 2.17), (bx - .08, yc), color=MUTED, lw=1.3, head=11)
        label(ax, bx + .8, yc + .34, f'bootstrap sample {k + 1 if k < 2 else "B"}', fs=12, color=MUTED)
        tx0 = 4.15
        tree_nodes(ax, ['', 'L', 'R'], tx0, tx0 + 2.0, yc + .36, .42, color=BLUE, fs=11.5,
                   labels={'': feats[k][0], 'L': feats[k][1], 'R': feats[k][2]}, leaf_w=.24, leaf_h=.17,
                   leaf_color=BLUE, node_w=.62, node_h=.3)
        arrow(ax, (bx + 1.68, yc), (tx0 - .02, yc), color=MUTED, lw=1.3, head=11)
        box(ax, 6.75, yc - .26, 1.45, .52, f'P(sand) = {probs[k]:.2f}', BLUE, fill='white', fs=12.5)
        arrow(ax, (tx0 + 2.05, yc), (6.7, yc), color=MUTED, lw=1.3, head=11)
        # each arrow meets the averaging box at its own height, so the three heads stay separate
        arrow(ax, (8.25, yc), (8.9, 2.05 + .3 * (yc - 2.05)), color=MUTED, lw=1.3, head=11)
    vdots(ax, 2.85, 1.43)
    vdots(ax, 7.475, 1.43)
    out = box(ax, 8.95, 1.57, 1.6, .96, 'Average', BLUE, fill='solid', fs=14, sub='P(sand) = 0.72', sub_fs=12.5)
    arrow(ax, out.r, (11.0, 2.05), color=INK, lw=1.6)
    label(ax, 11.38, 2.05, 'sand', fs=14, color=INK, weight='semibold')
    chip(ax, 2.85, .12, 'rows: drawn with replacement', color=BLUE, fs=12)
    chip(ax, 5.15, 4.08, 'each split: best of a random subset of features', color=TEAL, fs=12)
    chip(ax, 9.75, .12, 'combine: average the probabilities', color=BLUE, fs=12)
    save('ml2_forest_diagram', f, 'Schematic of a random forest classifier; feature names and probabilities '
         'are illustrative (the average of 0.82, 0.64 and 0.71 is 0.72).', kind='schematic')


@fig
def forest_regression():
    rng = np.random.default_rng(8)
    x = np.sort(rng.uniform(0, 8, 70))            # training inputs cover 0-8 only
    truth = lambda z: np.sin(z) + .25 * z  # noqa: E731
    y = truth(x) + rng.normal(0, .35, len(x))
    xx = np.linspace(0, 10, 600)
    rf = RandomForestRegressor(300, min_samples_leaf=5, random_state=0).fit(x[:, None], y)
    f, axs = half()
    ax = axs[0]
    ax.axvspan(8, 10, color=PANEL, zorder=0)
    ax.text(9, -.5, 'no training\ndata', ha='center', va='center', fontsize=12, color=MUTED)
    ax.scatter(x, y, s=18, c='#AEBBC8', zorder=2)
    for t in rf.estimators_[:3]:
        ax.plot(xx, t.predict(xx[:, None]), color=GREY, lw=1.2, alpha=.85, zorder=3)
    ax.plot(xx, rf.predict(xx[:, None]), color=BLUE, lw=3.2, zorder=4)
    ax.plot(xx, truth(xx), color=INK, lw=1.5, ls='--', zorder=5)
    ax.text(.15, 3.4, 'three of the trees', fontsize=12.5, color=GREY)
    ax.text(.15, 3.05, 'forest average (300 trees)', fontsize=12.5, color=BLUE, fontweight='semibold')
    ax.text(.15, 2.7, 'generating function (dashed)', fontsize=12.5, color=INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3, 3.75)
    clean(ax, 'Input x', 'Target y')
    save('ml2_forest_regression', f, 'Synthetic data y = sin(x) + 0.25x + noise (sd 0.35), n = 70 with x drawn '
         'uniformly on 0-8, seed 8; RandomForestRegressor, 300 trees, min_samples_leaf = 5, seed 0. Shaded: '
         'no training data (x > 8).', kind='computed')


@fig
def max_features():
    rng = np.random.default_rng(0)
    mfs = [1, 2, 3, 4, 6, 8, 10, 12]
    Xte, yte = make_friedman1(n_samples=400, n_features=12, noise=1.0, random_state=100)
    Z = 16
    sets = [make_friedman1(n_samples=300, n_features=12, noise=1.0, random_state=1 + z) for z in range(Z)]
    rho, tree_err, forest_err = [], [], []
    for mf in mfs:
        P = []
        for z, (X, y) in enumerate(sets):
            rf = RandomForestRegressor(60, max_features=mf, random_state=z, n_jobs=-1).fit(X, y)
            P.append(np.array([t.predict(Xte) for t in rf.estimators_]))
        P = np.array(P)                                   # (Z, B, n_test)
        B = P.shape[1]
        total = P.reshape(-1, P.shape[2]).var(0, ddof=1)
        within = P.var(1, ddof=1).mean(0)
        between = P.mean(1).var(0, ddof=1) - within / B
        rho.append(float(np.mean(between) / np.mean(total)))
        tree_err.append(float(((P - yte) ** 2).mean()))
        forest_err.append(float(((P.mean(1) - yte) ** 2).mean()))
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    ax.plot(mfs, rho, '-o', color=TEAL)
    ax.set_ylim(0, .12)
    ax.set_xticks(mfs)
    clean(ax, 'max_features (of 12 features)', 'Correlation between two trees')
    ax.set_title('Lower max_features: more diverse trees')
    ax = axs[1]
    ax.plot(mfs, tree_err, '-o', color=GREY)
    ax.plot(mfs, forest_err, '-o', color=BLUE)
    endlabel(ax, mfs[-1], tree_err[-1], 'one tree', GREY, dx=4)
    endlabel(ax, mfs[-1], forest_err[-1], 'forest', BLUE, dx=4)
    best = int(np.argmin(forest_err))
    ax.scatter([mfs[best]], [forest_err[best]], s=120, facecolors='none', edgecolors=ORANGE, lw=2, zorder=5)
    ax.set_xticks(mfs)
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, max(tree_err) * 1.08)
    clean(ax, 'max_features (of 12 features)', 'Test mean squared error')
    ax.set_title('Higher max_features: stronger single trees')
    save('ml2_max_features', f, 'Friedman #1 synthetic regression (12 features, 5 informative, noise sd 1): '
         f'{Z} independent training sets of 300 rows, one 400-row test set; forests of 60 trees. Correlation '
         '= between-training-set variance of the forest mean / total variance of one tree (ESL eq. 15.1).',
         kind='computed')
    print('  rho', dict(zip(mfs, np.round(rho, 3))), 'tree', np.round(tree_err, 2), 'forest', np.round(forest_err, 2))


@fig
def oob_matrix():
    rng = np.random.default_rng(2)
    n, B = 10, 6
    counts = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).T   # rows x trees
    f, ax = canvas(6.3, 4.9)
    cw, ch = .56, .34
    x0, ytop = 1.55, 4.25
    for j in range(B):
        label(ax, x0 + (j + .5) * cw, ytop + .2, f'T{j + 1}', fs=12.5, weight='semibold', color=BLUE)
    label(ax, x0 + B * cw / 2, ytop + .52, 'Trees (each fitted to one bootstrap sample)', fs=12.5, color=MUTED)
    label(ax, x0 + B * cw + .75, ytop + .2, 'OOB vote', fs=12.5, weight='semibold', color=ORANGE)
    for i in range(n):
        yb = ytop - (i + 1) * ch
        label(ax, x0 - .1, yb + ch / 2, f'W1 {1500 + .5 * i:.1f} m', fs=12, ha='right', color=INK)
        for j in range(B):
            c = counts[i, j]
            if c == 0:
                ax.add_patch(Rectangle((x0 + j * cw, yb), cw, ch, fc=ORANGE_T, ec='white', lw=1.5, zorder=2))
                label(ax, x0 + (j + .5) * cw, yb + ch / 2, 'out', fs=11.5, color=ORANGE, weight='semibold')
            else:
                ax.add_patch(Rectangle((x0 + j * cw, yb), cw, ch, fc=tint(BLUE, .8 if c == 1 else .45),
                                       ec='white', lw=1.5, zorder=2))
                label(ax, x0 + (j + .5) * cw, yb + ch / 2, f'×{c}', fs=12, color=INK)
        oob = [f'T{j + 1}' for j in range(B) if counts[i, j] == 0]
        label(ax, x0 + B * cw + .12, yb + ch / 2, ', '.join(oob) if oob else 'none', fs=12, ha='left',
              color=ORANGE if oob else MUTED)
    frac = (counts == 0).mean()
    label(ax, 3.15, .3, f'{(counts == 0).sum()} of {n * B} cells out of bag ({frac:.0%}); '
          f'expected $(1 - 1/n)^n$ = {(1 - 1 / n) ** n:.0%}', fs=12, color=MUTED)
    save('ml2_oob_matrix', f, 'Bootstrap draws for 10 adjacent depth rows of one synthetic well and 6 trees '
         '(seed 2). ×k = row drawn k times; "out" = out of bag for that tree.', kind='computed')


@fig
def oob_curve():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5, n_redundant=2,
                               flip_y=.05, random_state=3)
    ns = np.arange(15, 301, 5)
    f, axs = half()
    ax = axs[0]
    for mf, col, lab in [(1, PURPLE, 'max_features = 1'), ('sqrt', BLUE, "max_features = 'sqrt' (3)"),
                         (None, GREY, 'max_features = None (bagging)')]:
        rf = RandomForestClassifier(warm_start=True, oob_score=True, max_features=mf, random_state=0, n_jobs=-1)
        err = []
        for n in ns:
            rf.set_params(n_estimators=int(n))
            rf.fit(X, y)
            err.append(1 - rf.oob_score_)
        ax.plot(ns, err, color=col, lw=2.2)
        endlabel(ax, ns[-1], err[-1], lab.split(' = ')[1].split(' ')[0], col, fs=12.5)
    ax.set_xlim(0, 360)
    clean(ax, 'Number of trees', 'Out-of-bag error rate')
    ax.text(345, ax.get_ylim()[1] * .99, 'line labels: max_features', fontsize=12, color=MUTED, va='top',
            ha='right')
    save('ml2_oob_curve', f, 'Synthetic classification data (600 rows, 10 features, 5 informative, 5% label '
         'noise, seed 3); RandomForestClassifier grown with warm_start, seed 0.', kind='computed')


@fig
def oob_grouped():
    d = facies_wells()
    X, y, g = d[LOGS].values, d.FACIES.values, d.WELL.values
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1, oob_score=True).fit(X, y)
    kf = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y,
                         cv=KFold(5, shuffle=True, random_state=0))
    gk = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y, cv=GroupKFold(4), groups=g)
    f, axs = half()
    ax = axs[0]
    vals = [rf.oob_score_, kf.mean(), gk.mean()]
    names = ['Out-of-bag\n(rows)', 'Shuffled 5-fold\n(rows)', 'Grouped 4-fold\n(new wells)']
    cols = [GREY, GREY, TEAL]
    ax.bar(range(3), vals, .6, color=[tint(c, .35) for c in cols], edgecolor=cols, lw=1.5, zorder=3)
    for k, v in enumerate(vals):
        ax.text(k, .06, f'{v:.2f}', ha='center', va='bottom', fontsize=15, fontweight='semibold',
                color=INK if k < 2 else shade(TEAL, .35))
    ax.scatter(np.full(len(kf), 1) + np.linspace(-.12, .12, len(kf)), kf, s=16, c=INK, zorder=4)
    ax.scatter(np.full(len(gk), 2) + np.linspace(-.12, .12, len(gk)), gk, s=16, c=INK, zorder=4)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0, 1.0)
    clean(ax, None, 'Facies accuracy')
    ax.grid(axis='x', visible=False)
    ax.text(2.4, 1.0, 'dots: individual folds', fontsize=12, color=MUTED, ha='right', va='bottom')
    save('ml2_oob_grouped', f, 'Synthetic facies logs, 8 wells x 160 rows with well offsets and depth-correlated '
         'noise (seed 7); RandomForestClassifier, 300 trees, seed 0.', kind='computed')
    print('  oob', round(rf.oob_score_, 3), 'kfold', kf.round(3), round(kf.mean(), 3), 'group', gk.round(3), round(gk.mean(), 3))


@fig
def facies_case():
    d = facies_wells()
    tr, te = d[d.WELL != 'W8'], d[d.WELL == 'W8']
    rf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(tr[LOGS], tr.FACIES)
    pred = rf.predict(te[LOGS])
    acc = accuracy_score(te.FACIES, pred)
    z = te.DEPTH.values
    fig_, axs = plt.subplots(1, 6, figsize=(6.3, 4.9), sharey=True, layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, .5, .5]})
    specs = [('GR', (0, 150), 'API', GREEN), ('RHOB', (2.1, 2.8), 'g/cm³', RED), ('NPHI', (.4, 0), 'v/v', BLUE),
             ('PE', (0, 7), 'b/e', PURPLE)]
    for ax, (k, lim, unit, col) in zip(axs, specs):
        ax.plot(te[k], z, color=col, lw=1.2)
        ax.set_xlim(*lim)
        ax.set_title(k, fontsize=13)
        ax.set_xlabel(unit, fontsize=12)
        ax.set_xticks(lim)
        ax.tick_params(axis='x', labelsize=11)
        ticks = ax.xaxis.get_major_ticks()          # keep each track's scale labels inside the track
        ticks[0].label1.set_horizontalalignment('left')
        ticks[-1].label1.set_horizontalalignment('right')
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', alpha=1)
    from matplotlib.colors import ListedColormap as _LC
    cm = _LC(FAC_COL)
    for ax, vals, title in [(axs[4], te.FACIES.values, 'True'), (axs[5], pred, 'Forest')]:
        ax.imshow(vals[:, None], aspect='auto', cmap=cm, vmin=0, vmax=2, extent=(0, 1, z.max() + .25, z.min() - .25),
                  interpolation='nearest')
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_xlabel(' ', fontsize=12)
    miss = pred != te.FACIES.values
    axs[5].scatter(np.full(miss.sum(), 1.18), z[miss], marker='<', s=10, color=INK, clip_on=False, zorder=5)
    axs[0].set_ylim(z.max(), z.min())
    axs[0].set_ylabel('Depth (m)')
    for k, (name, col) in enumerate(zip(['sand', 'shale', 'lime'], FAC_COL)):
        axs[4].text(1.12, -.035 - .052 * k, name, transform=axs[4].transAxes, ha='center', va='top',
                    fontsize=12, color=col, fontweight='semibold')
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    save('ml2_facies_case', fig_, f'Synthetic well W8 held out; forest (300 trees, min_samples_leaf 3, seed 0) '
         f'trained on wells W1-W7; accuracy on W8 = {acc:.3f}. Facies: gold sandstone, grey shale, blue limestone; '
         'arrows mark misclassified rows.', kind='computed')
    print('  facies W8 acc', round(acc, 3))


# =================================================================== PART 3: boosting
@fig
def bag_vs_boost():
    f, ax = canvas(11.8, 4.3)
    # ---- bagging (left)
    chip(ax, 2.75, 4.05, 'Bagging: independent models, trained in parallel', color=BLUE, fs=13)
    d = box(ax, .2, 1.7, 1.05, .8, 'Data', GREY, fs=14)
    ys = [3.0, 2.1, 1.2]
    for k, yc in enumerate(ys):
        r = box(ax, 1.85, yc - .27, 1.35, .54, f'resample {k + 1}', GREY, fill='white', fs=12.5, weight='normal')
        t = box(ax, 3.55, yc - .27, 1.05, .54, f'tree {k + 1}', BLUE, fs=13)
        arrow(ax, d.r, r.l, color=MUTED, lw=1.3, head=11)
        arrow(ax, r.r, t.l, color=MUTED, lw=1.3, head=11)
        arrow(ax, t.r, (4.98, 2.1 + .28 * (yc - 2.1)), color=MUTED, lw=1.3, head=11)
    a = box(ax, 5.0, 1.68, 1.02, .84, 'average\nor vote', BLUE, fill='solid', fs=13)
    label(ax, 2.9, .5, 'Trees never see each other’s errors.\nDeep trees: low bias, averaging cuts variance.',
          fs=12.5, color=MUTED)
    ax.plot([6.35, 6.35], [.3, 4.2], color=RULE, lw=1.5)
    # ---- boosting (right)
    chip(ax, 9.1, 4.05, 'Boosting: each model corrects the ensemble so far', color=ORANGE, fs=13)
    xs = [6.6, 7.9, 9.2, 10.5]
    labs = ['F₀\nmean', '+ η·tree 1', '+ η·tree 2', '+ η·tree 3']
    for k, x in enumerate(xs):
        b = box(ax, x, 2.35, 1.05, .8, labs[k], ORANGE if k else GREY, fill='tint', fs=13)
        if k:
            arrow(ax, (xs[k - 1] + 1.05, 2.75), (x, 2.75), color=INK, lw=1.5, head=12)
            label(ax, x + .52, 1.95, 'fits the errors', fs=12, color=ORANGE)
            label(ax, x + .52, 1.68, f'of F{"₀₁₂"[k - 1]}', fs=12, color=ORANGE)
            ax.add_patch(FancyArrowPatch((xs[k - 1] + .52, 2.33), (x + .2, 2.33), connectionstyle='arc3,rad=.55',
                                         arrowstyle='-|>', mutation_scale=11, color=ORANGE, lw=1.2, ls='--', zorder=1))
    label(ax, 9.1, .95, 'F₃(x) = F₀ + η·h₁(x) + η·h₂(x) + η·h₃(x)', fs=14, color=INK, weight='semibold')
    label(ax, 9.1, .5, 'Shallow trees: each step is small; the sum becomes flexible.', fs=12.5, color=MUTED)
    save('ml2_bag_vs_boost', f, 'Schematic comparison of bagging and boosting.', kind='schematic')


def _adaboost(X, y, rounds):
    """Discrete AdaBoost with stumps; returns per-round (weights, stump, miss, err, alpha)."""
    n = len(y)
    w = np.full(n, 1 / n)
    out = []
    for m in range(rounds):
        st = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        miss = st.predict(X) != y
        err = w[miss].sum()
        alpha = .5 * np.log((1 - err) / err)
        out.append((w.copy(), st, miss, err, alpha))
        w = w * np.exp(np.where(miss, alpha, -alpha))
        w = w / w.sum()
    return out


def _ada_data():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1, 1, (36, 2))
    y = (X[:, 0] + X[:, 1] > .15).astype(int)
    return X, y


@fig
def adaboost_rounds():
    X, y = _ada_data()
    hist = _adaboost(X, y, 3)
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    f, axs = sub(4, height=4.1)
    for k, ax in enumerate(axs[:3]):
        w, st, miss, err, alpha = hist[k]
        zz = st.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-.5, .5, 1.5], cmap=BG)
        ax.contour(xx, yy, zz, levels=[.5], colors=[INK], linewidths=1.6)
        size = 900 * w
        for c, mk, col in [(0, 'o', BLUE), (1, '^', ORANGE)]:
            mask = y == c
            ax.scatter(X[mask, 0], X[mask, 1], s=size[mask], c=col, marker=mk, edgecolors='white', lw=.6, zorder=3)
        ax.scatter(X[miss, 0], X[miss, 1], s=size[miss] * 2.1 + 40, facecolors='none', edgecolors=INK, lw=1.4,
                   zorder=4)
        ax.set_title(f'Round {k + 1}: ε = {err:.2f}, α = {alpha:.2f}', fontsize=13.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        clean(ax, 'Feature x₁', 'Feature x₂' if k == 0 else None, grid=False)
    ax = axs[3]
    F = sum(a * (2 * s_.predict(grid) - 1) for _, s_, _, _, a in hist).reshape(xx.shape)
    ax.contourf(xx, yy, (F > 0).astype(int), levels=[-.5, .5, 1.5], cmap=BG)
    ax.contour(xx, yy, F, levels=[0], colors=[INK], linewidths=1.6)
    Ft = sum(a * (2 * s_.predict(X) - 1) for _, s_, _, _, a in hist)
    wrong = (Ft > 0).astype(int) != y
    points(ax, X, y, s=34)
    ax.scatter(X[wrong, 0], X[wrong, 1], s=150, facecolors='none', edgecolors=INK, lw=1.4, zorder=4)
    ax.plot([-1.1, 1.1], [1.25, -.95], color=MUTED, ls=':', lw=1.4)
    ax.set_title(f'Weighted vote of 3 stumps: {wrong.sum()} error' + ('s' if wrong.sum() != 1 else ''),
                 fontsize=13.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    clean(ax, 'Feature x₁', None, grid=False)
    save('ml2_adaboost_rounds', f, 'Discrete AdaBoost with decision stumps on 36 synthetic points (seed 12; true '
         'boundary x1 + x2 = 0.15, dotted). Marker area is proportional to the row weight at the start of each '
         'round; rings mark rows the current stump misclassifies.', kind='computed')
    print('  adaboost', [(round(e, 3), round(a, 3), int(m.sum())) for _, _, m, e, a in hist], 'final errors', wrong.sum())


@fig
def adaboost_alpha():
    X, y = _ada_data()
    hist = _adaboost(X, y, 3)
    e = np.linspace(.02, .98, 400)
    a = .5 * np.log((1 - e) / e)
    f, axs = half()
    ax = axs[0]
    ax.plot(e, a, color=INK, lw=2.4)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(.5, color=MUTED, lw=1, ls=':')
    errs = [h[3] for h in hist]
    alphas = [h[4] for h in hist]
    ax.scatter(errs, alphas, s=60, color=ORANGE, zorder=4, edgecolors='white', lw=.8)
    ax.annotate('rounds 1–3 of the example\n(ε = ' + ', '.join(f'{e:.2f}' for e in errs) + ')',
                (np.mean(errs), np.mean(alphas)), xytext=(.55, 1.38), textcoords='data', fontsize=12.5,
                color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    ax.text(.53, .25, 'ε = 0.5: no better than\na coin, α = 0', fontsize=12.5, color=MUTED, va='bottom')
    ax.text(.53, -1.62, 'ε > 0.5: the vote is\nflipped (α < 0)', fontsize=12.5, color=MUTED, va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    clean(ax, 'Weighted error ε of the stump', 'Say in the final vote α')
    save('ml2_adaboost_alpha', f, 'Exact formula alpha = 0.5 ln((1 - eps) / eps); points are the three rounds of '
         'the synthetic AdaBoost example (seed 12).', kind='computed')


@fig
def gb_stages():
    rng = np.random.default_rng(21)
    x = np.sort(rng.uniform(0, 10, 60))
    y = 2 * np.sin(.9 * x) + .15 * x + rng.normal(0, .35, len(x))
    xx = np.linspace(0, 10, 500)
    eta = .5
    F = np.full(len(x), y.mean())
    Fg = np.full(len(xx), y.mean())
    f, axs = grid_axes(2, 3, height=4.7)
    for m in range(3):
        r = y - F
        h = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x[:, None], r)
        ax = axs[0, m]
        ax.axhline(0, color=MUTED, lw=1)
        ax.scatter(x, r, s=14, c=INK, alpha=.55, zorder=2)
        ax.plot(xx, h.predict(xx[:, None]), color=ORANGE, lw=2.6, zorder=3)
        ax.set_ylim(-3.6, 3.6)
        ax.set_title(f'Stage {m + 1}: tree h{"₁₂₃"[m]} fits the residuals', fontsize=13)
        clean(ax, None, 'Residual' if m == 0 else None)
        ax.set_xticklabels([])
        ax = axs[1, m]
        Fg_new = Fg + eta * h.predict(xx[:, None])
        ax.scatter(x, y, s=14, c='#AEBBC8', zorder=2)
        ax.plot(xx, Fg, color=GREY, lw=1.6, ls='--', zorder=3)
        ax.plot(xx, Fg_new, color=BLUE, lw=2.6, zorder=4)
        ax.set_ylim(-2.4, 3.9)
        ax.set_title(f'F{"₁₂₃"[m]} = F{"₀₁₂"[m]} + 0.5·h{"₁₂₃"[m]}', fontsize=13)
        clean(ax, 'Input x', 'Target y' if m == 0 else None)
        F = F + eta * h.predict(x[:, None])
        Fg = Fg_new
        mse = np.mean((y - F) ** 2)
        ax.text(9.8, -2.1, f'training MSE {mse:.2f}', ha='right', fontsize=12, color=BLUE)
    axs[1, 0].text(.2, 3.35, 'dashed: previous model', fontsize=12, color=GREY)
    save('ml2_gb_stages', f, f'Synthetic data y = 2 sin(0.9x) + 0.15x + noise (sd 0.35), n = 60, seed 21; '
         f'F0 = mean of y = {y.mean():.2f}; depth-2 regression trees; learning rate 0.5.', kind='computed')


@fig
def loss_gradients():
    u = np.linspace(-3, 3, 601)
    d = 1.0
    hub = np.where(np.abs(u) <= d, .5 * u ** 2, d * (np.abs(u) - .5 * d))
    f, axs = sub(3, height=4.3)
    ax = axs[0]
    for v, col, lab, yl in [(.5 * u ** 2, INK, 'squared', 4.6), (np.abs(u), PURPLE, 'absolute', 2.95),
                            (hub, TEAL, 'Huber (δ = 1)', 2.2)]:
        ax.plot(u, v, color=col, lw=2.4)
        endlabel(ax, 3, v[-1] if lab != 'squared' else 4.5, lab, col, fs=12.5)
    ax.set_ylim(0, 4.8)
    ax.set_xlim(-3, 4.9)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    clean(ax, 'Residual y − F', 'Loss')
    ax.set_title('Regression losses')
    ax = axs[1]
    ax.plot(u, u, color=INK, lw=2.4)
    ax.plot(u, np.clip(u, -d, d), color=TEAL, lw=2.4)
    # dashed on top: sign(y - F) equals the clipped line for |y - F| > 1, and both stay visible there
    ax.plot(u, np.sign(u), color=PURPLE, lw=2.4, ls=(0, (3, 2.2)), zorder=3)
    endlabel(ax, 3, 3, 'y − F', INK, fs=12.5)
    endlabel(ax, 3, 1.25, 'sign(y − F)', PURPLE, fs=12.5)
    endlabel(ax, 3, .75, 'clip(y − F, ±δ)', TEAL, fs=12.5)
    ax.set_xlim(-3, 6.6)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_ylim(-3.2, 3.4)
    clean(ax, 'Residual y − F', 'Pseudo-residual (what the tree fits)')
    ax.set_title('Negative gradients')
    ax = axs[2]
    F = np.linspace(-5, 5, 400)
    pr = expit(F)
    ax.plot(F, 1 - pr, color=ORANGE, lw=2.4)
    ax.plot(F, -pr, color=BLUE, lw=2.4)
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(-4.9, .76, 'y = 1:  1 − p', fontsize=13, color=ORANGE, va='top', fontweight='semibold')
    ax.text(4.9, -.76, 'y = 0:  −p', fontsize=13, color=BLUE, ha='right', va='bottom', fontweight='semibold')
    ax.set_ylim(-1.1, 1.1)
    clean(ax, 'Current log-odds F', 'Pseudo-residual y − p')
    ax.set_title('Log-loss (classification)')
    save('ml2_loss_gradients', f, 'Exact functions. Huber threshold delta = 1; p = 1 / (1 + exp(-F)).', kind='computed')


@fig
def early_stopping():
    X, y = make_friedman1(400, n_features=10, noise=3.0, random_state=1)
    Xv, yv = make_friedman1(2000, n_features=10, noise=3.0, random_state=2)
    m = GradientBoostingRegressor(n_estimators=500, learning_rate=.1, max_depth=3, random_state=0).fit(X, y)
    val = np.array([mean_squared_error(yv, p) for p in m.staged_predict(Xv)])
    trn = np.array([mean_squared_error(y, p) for p in m.staged_predict(X)])
    patience = 20
    best, best_i, stop = np.inf, 0, len(val)
    for i, v in enumerate(val):
        if v < best - 1e-4:
            best, best_i = v, i
        elif i - best_i >= patience:
            stop = i
            break
    st = np.arange(1, len(val) + 1)
    f, axs = half()
    ax = axs[0]
    ax.plot(st, trn, color=BLUE, lw=2.3)
    ax.plot(st, val, color=TEAL, lw=2.3)
    ax.axvspan(best_i + 1, stop + 1, color=GOLD_T, zorder=0)
    ax.axvline(best_i + 1, color=GOLD, lw=1.5, ls='--')
    ax.scatter([best_i + 1], [val[best_i]], s=60, color=GOLD, zorder=5)
    endlabel(ax, 500, trn[-1] + 1.2, 'training', BLUE, fs=12.5, ha='right', dx=-6)
    endlabel(ax, 500, val[-1] + 1.3, 'validation', TEAL, fs=12.5, ha='right', dx=-6)
    ax.annotate(f'best: stage {best_i + 1}\nstop: stage {stop + 1}\n(patience {patience})',
                (best_i + 1, val[best_i]), xytext=(150, 26), textcoords='data', fontsize=12.5, color=shade(GOLD, .2),
                arrowprops=dict(arrowstyle='-|>', color=GOLD, lw=1.1))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 32)
    clean(ax, 'Boosting stage (trees added)', 'Mean squared error')
    save('ml2_early_stopping', f, 'Friedman #1 synthetic regression (10 features, noise sd 3): 400 training rows '
         '(seed 1), 2,000 validation rows (seed 2); GradientBoostingRegressor, learning rate 0.1, depth 3. Stop '
         f'rule: {patience} stages without improvement.', kind='computed')
    print('  early stop best', best_i + 1, round(val[best_i], 2), 'stop', stop + 1, 'val end', round(val[-1], 2))


@fig
def subsample_curves():
    res = {1.0: [], .5: []}
    for z in range(5):
        X, y = make_friedman1(600, n_features=10, noise=2.0, random_state=z)
        Xv, yv = make_friedman1(2000, n_features=10, noise=2.0, random_state=100 + z)
        for ss in res:
            m = GradientBoostingRegressor(n_estimators=400, learning_rate=.1, max_depth=3, subsample=ss,
                                          random_state=z).fit(X, y)
            res[ss].append([mean_squared_error(yv, p) for p in m.staged_predict(Xv)])
    st = np.arange(1, 401)
    f, axs = half()
    ax = axs[0]
    for ss, col, lab in [(1.0, GREY, 'subsample = 1.0'), (.5, ORANGE, 'subsample = 0.5')]:
        c = np.mean(res[ss], 0)
        ax.plot(st, c, color=col, lw=2.4)
        i = int(np.argmin(c))
        ax.scatter([i + 1], [c[i]], s=55, color=col, zorder=4)
        ax.annotate(f'min {c[i]:.2f}', (i + 1, c[i]), xytext=(0, -20 if ss == .5 else 12), textcoords='offset points',
                    ha='center', fontsize=12.5, color=col)
        ax.text(395, 9.6 if ss == 1.0 else 9.1, lab, fontsize=12.5, color=col, ha='right',
                fontweight='semibold')
    ax.set_xlim(0, 400)
    ax.set_ylim(4, 12)
    clean(ax, 'Boosting stage', 'Validation MSE (mean of 5 datasets)')
    save('ml2_subsample', f, 'Friedman #1 synthetic regression (noise sd 2), five independent datasets of 600 '
         'training and 2,000 validation rows (seeds 0-4); GradientBoostingRegressor, learning rate 0.1, depth 3.',
         kind='computed')
    print('  subsample mins', {k: round(float(np.min(np.mean(v, 0))), 3) for k, v in res.items()})


@fig
def rop_boosting():
    d = rop_wells()
    tr, te = d[d.WELL != 'D6'], d[d.WELL == 'D6'].sort_values('DEPTH')
    models = {'Training mean': None,
              'Deep tree': DecisionTreeRegressor(random_state=0),
              'Random forest': RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
              'Hist. gradient boosting': HistGradientBoostingRegressor(random_state=0)}
    preds, mae = {}, {}
    for k, m in models.items():
        p = np.full(len(te), tr.ROP.mean()) if m is None else m.fit(tr[ROP_X], tr.ROP).predict(te[ROP_X])
        preds[k], mae[k] = p, mean_absolute_error(te.ROP, p)
    fig_ = plt.figure(figsize=(11.8, 4.3), layout='constrained')
    gs = fig_.add_gridspec(1, 3, width_ratios=[2.25, .08, 1])
    ax = fig_.add_subplot(gs[0, 0])
    ax.scatter(te.DEPTH, te.ROP, s=8, c='#AEBBC8', zorder=1)
    ax.plot(te.DEPTH, preds['Random forest'], color=BLUE, lw=1.6, zorder=3)
    ax.plot(te.DEPTH, preds['Hist. gradient boosting'], color=ORANGE, lw=1.6, zorder=2)
    ax.axhline(tr.ROP.mean(), color=GREY, lw=1.3, ls='--')
    ax.text(2003, tr.ROP.mean() + 1, 'training mean', fontsize=12, color=GREY, va='bottom')
    ax.text(2003, 43, 'forest', fontsize=12.5, color=BLUE, fontweight='semibold')
    ax.text(2045, 43, 'boosting', fontsize=12.5, color=ORANGE, fontweight='semibold')
    ax.text(2110, 43, 'grey dots: observed', fontsize=12, color=MUTED)
    ax.set_xlim(2000, 2300)
    ax.set_ylim(0, 47)
    clean(ax, 'Measured depth on held-out well D6 (m)', 'ROP (m/h)')
    ax2 = fig_.add_subplot(gs[0, 2])
    names = list(mae)
    cols = [GREY, GREY, BLUE, ORANGE]
    yy = np.arange(len(names))[::-1]
    ax2.barh(yy, [mae[k] for k in names], .6, color=[tint(c, .35) for c in cols], edgecolor=cols, lw=1.4)
    for yv, k in zip(yy, names):
        ax2.text(mae[k] + .12, yv, f'{mae[k]:.2f}', va='center', fontsize=13, color=INK)
    ax2.set_yticks(yy, ['Training\nmean', 'Deep\ntree', 'Random\nforest', 'Gradient\nboosting'])
    ax2.set_xlim(0, max(mae.values()) * 1.3)
    clean(ax2, 'MAE on well D6 (m/h)', None)
    ax2.grid(axis='y', visible=False)
    save('ml2_rop_boosting', fig_, 'Synthetic drilling wells (seed 11): trained on D1-D5, evaluated on held-out '
         'well D6. RandomForestRegressor (300 trees, min_samples_leaf 3) and HistGradientBoostingRegressor '
         '(defaults), seed 0. MAE: ' + ', '.join(f'{k} {v:.2f}' for k, v in mae.items()) + ' m/h.', kind='computed')
    print('  rop mae', {k: round(v, 2) for k, v in mae.items()})


# =================================================================== PART 4: modern libraries, stacking
@fig
def histogram_binning():
    d = rop_wells()
    rng = np.random.default_rng(4)
    sm = d.iloc[rng.choice(len(d), 60, replace=False)].sort_values('WOB')
    x = sm.WOB.values
    r = sm.ROP.values - sm.ROP.values.mean()          # first-stage residuals (squared loss, F0 = mean)

    def best_split(thresholds):
        gains = [r[x <= t].sum() ** 2 / (x <= t).sum() + r[x > t].sum() ** 2 / (x > t).sum() for t in thresholds]
        return thresholds[int(np.argmax(gains))]
    mids = (x[1:] + x[:-1]) / 2
    nb = 16
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))[1:-1]
    t_exact, t_hist = best_split(mids), best_split(edges)
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    for c in mids:
        ax.axvline(c, color='#B8C4D0', lw=.9, zorder=1)
    ax.axvline(t_exact, color=ORANGE, lw=2.6, zorder=2)
    ax.scatter(x, r, s=28, c=INK, edgecolors='white', lw=.5, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    halo = dict(boxstyle='round,pad=.15', fc='white', ec='none')   # keeps the candidate lines off the label
    ax.text(t_exact + 3, 36, f'best: {t_exact:.1f} kN', color=ORANGE, fontsize=13, fontweight='semibold',
            bbox=halo, zorder=4)
    ax.set_title(f'Exact: every gap is a candidate ({len(mids)})')
    ax.set_ylim(-30, 42)
    clean(ax, 'Weight on bit (kN)', 'Residual ROP (m/h)', grid=False)
    ax = axs[1]
    allb = np.r_[x.min(), edges, x.max()]
    idx = np.clip(np.searchsorted(edges, x, side='left'), 0, nb - 1)
    sums = np.array([r[idx == k].sum() for k in range(nb)])
    for k in range(nb):
        lo, hi = allb[k], allb[k + 1]
        col = BLUE if sums[k] >= 0 else ORANGE
        ax.add_patch(Rectangle((lo, 0), hi - lo, sums[k], fc=tint(col, .55), ec=col, lw=1.2, zorder=2))
    for c in edges:
        ax.axvline(c, color='#9FB0C1', lw=1.2, zorder=1)
    ax.axvline(t_hist, color=ORANGE, lw=2.6, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(t_hist + 3, sums.max() * .92, f'best: {t_hist:.1f} kN', color=ORANGE, fontsize=13, fontweight='semibold',
            bbox=halo, zorder=4)
    ax.set_xlim(axs[0].get_xlim())
    lim = np.abs(sums).max() * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_title(f'Histogram: {nb} bins, {len(edges)} candidates')
    clean(ax, 'Weight on bit (kN)', 'Sum of residuals in bin (m/h)', grid=False)
    save('ml2_histogram_binning', f, 'Sixty rows of the synthetic drilling data (seed 4); residuals after the '
         'first-stage mean prediction. Candidate splits scored with the squared-error gain G_L²/n_L + G_R²/n_R; '
         f'best exact threshold {t_exact:.1f} kN, best histogram threshold {t_hist:.1f} kN (16 quantile bins; '
         'both give the same partition of the 60 rows).',
         kind='computed')
    print('  hist split', round(t_exact, 1), round(t_hist, 1))


@fig
def tree_growth():
    f, ax = canvas(11.8, 4.3)
    specs = [
        ('Level-wise (depth-wise)', 'fills each level before going deeper', 'XGBoost default · limit: max_depth',
         ['', 'L', 'R'], {'': '1', 'L': '2', 'R': '3'}, BLUE),
        ('Leaf-wise (best-first)', 'splits the leaf with the largest gain', 'LightGBM, sklearn HistGB · limit: leaves',
         ['', 'R', 'RL'], {'': '1', 'R': '2', 'RL': '3'}, ORANGE),
        ('Symmetric (oblivious)', 'one condition per level (A, then B)', 'CatBoost default · limit: depth',
         ['', 'L', 'R'], {'': 'A', 'L': 'B', 'R': 'B'}, TEAL),
    ]
    for k, (title, how, lib, nodes, labs, col) in enumerate(specs):
        x0 = .25 + k * 3.95
        tree_nodes(ax, nodes, x0 - .35, x0 + 3.65, 3.02, .74, color=col, r=.23, fs=14, labels=labs, leaf_w=.36,
                   leaf_h=.24, leaf_color=col, lw=1.6)
        label(ax, x0 + 1.65, 4.05, title, fs=14.5, weight='semibold', color=col)
        label(ax, x0 + 1.65, 3.62, how, fs=12.5, color=INK)
        label(ax, x0 + 1.65, .32, lib, fs=12.5, color=MUTED)
        if k < 2:
            ax.plot([x0 + 3.62, x0 + 3.62], [.15, 4.25], color=RULE, lw=1.3)
    save('ml2_tree_growth', f, 'Schematic: three trees with three split nodes each, grown by different policies; '
         'numbers give the order in which splits are made.', kind='schematic')


@fig
def missing_route():
    f, ax = canvas(6.3, 4.9)
    n = box(ax, 2.05, 3.55, 2.2, .72, 'GR ≤ 75 API ?', BLUE, fs=15, weight='semibold')
    lft = box(ax, .35, 1.75, 2.1, .95, 'yes', BLUE, fill='white', fs=14, sub='40, 55, 70 API', sub_fs=12.5)
    rgt = box(ax, 3.85, 1.75, 2.1, .95, 'no', BLUE, fill='white', fs=14, sub='90, 110 API + missing', sub_fs=12.5)
    arrow(ax, (2.6, 3.55), lft.t, color=INK, lw=1.5)
    arrow(ax, (3.7, 3.55), rgt.t, color=INK, lw=1.5)
    chip(ax, 5.2, 4.55, 'GR missing', color=ORANGE, fs=13, solid=True)
    ax.add_patch(FancyArrowPatch((5.2, 4.35), (5.3, 2.75), connectionstyle='arc3,rad=-.25', arrowstyle='-|>',
                                 mutation_scale=13, color=ORANGE, lw=1.8, ls='--', zorder=3))
    label(ax, 3.15, 1.15, 'During training both directions are scored;', fs=12.5, color=INK)
    label(ax, 3.15, .8, 'the one with the larger gain is stored.', fs=12.5, color=INK)
    label(ax, 3.15, .3, 'No missing GR in training? sklearn sends them to the larger child.', fs=11.5, color=MUTED)
    save('ml2_missing_route', f, 'Schematic of native missing-value handling in histogram gradient boosting '
         '(scikit-learn HistGradientBoosting documentation behaviour).', kind='schematic')


@fig
def stacking_diagram():
    f, ax = canvas(11.8, 4.3)
    # folds
    fy = [3.45, 2.95, 2.45, 1.95, 1.45]
    label(ax, .95, 3.98, 'Training rows', fs=13.5, weight='semibold')
    for k, y in enumerate(fy):
        held = k == 2
        box(ax, .3, y - .2, 1.3, .4, f'fold {k + 1}', TEAL if held else BLUE, fill='solid' if held else 'tint',
            fs=12.5, weight='semibold' if held else 'normal')
    label(ax, .95, 1.0, 'repeat for\neach fold', fs=12, color=MUTED)
    # base models trained on other folds
    arrow(ax, (1.7, 2.45), (2.35, 2.45), color=MUTED, lw=1.4)
    names = [('forest', BLUE), ('boosting', ORANGE), ('logistic', GREY)]
    for k, (nm, col) in enumerate(names):
        box(ax, 2.45, 3.05 - k * .62, 1.45, .48, nm, col, fs=13)
    label(ax, 3.18, 3.8, 'Base models fitted on\nfolds 1, 2, 4, 5', fs=12.5, color=INK)
    arrow(ax, (4.0, 2.45), (4.75, 2.45), color=MUTED, lw=1.4)
    label(ax, 4.38, 1.72, 'predict\nfold 3', fs=12.5, color=TEAL, weight='semibold')
    # OOF matrix
    x0, top = 4.9, 3.5
    cw, ch = .52, .3
    for i in range(7):
        for j, (nm, col) in enumerate(names):
            filled = i in (2, 3) or i < 2 or i > 3
            ax.add_patch(Rectangle((x0 + j * cw, top - (i + 1) * ch), cw, ch,
                                   fc=tint(col, .45 if i in (3, 4) else .82), ec='white', lw=1.3, zorder=2))
    ax.add_patch(Rectangle((x0, top - 5 * ch), 3 * cw, 2 * ch, fc='none', ec=TEAL, lw=2.4, zorder=3))
    label(ax, x0 + 1.5 * cw, 3.8, 'Out-of-fold predictions', fs=12.5, weight='semibold')
    label(ax, x0 + 1.5 * cw, top - 7 * ch - .25, 'one column per base model', fs=12, color=MUTED)
    # meta-model
    arrow(ax, (x0 + 3 * cw + .15, 2.45), (7.25, 2.45), color=MUTED, lw=1.4)
    m = box(ax, 7.35, 2.0, 1.75, .9, 'meta-model', PURPLE, fill='solid', fs=14, sub='e.g. logistic', sub_fs=12)
    arrow(ax, m.r, (9.75, 2.45), color=INK, lw=1.6)
    label(ax, 10.55, 2.45, 'stacked\nprediction', fs=13.5, weight='semibold', color=PURPLE)
    # inference note
    ax.plot([2.3, 11.6], [.95, .95], color=RULE, lw=1.2)
    label(ax, 6.95, .55, 'For new rows: refit each base model on all training rows, then feed its predictions to the '
          'meta-model.', fs=12.5, color=MUTED)
    save('ml2_stacking_diagram', f, 'Schematic of stacking with K = 5 folds and three base models.', kind='schematic')


# =================================================================== PART 5: interpretation
@fig
def mdi_bias():
    d = facies_wells().copy()
    rng = np.random.default_rng(0)
    d['RANDOM_NUM'] = rng.normal(size=len(d))
    d['RANDOM_CAT'] = rng.integers(0, 3, len(d))
    cols = LOGS + ['RANDOM_NUM', 'RANDOM_CAT']
    tr, te = d[d.WELL.isin([f'W{i}' for i in range(1, 7)])], d[d.WELL.isin(['W7', 'W8'])]
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(tr[cols], tr.FACIES)
    pi = permutation_importance(rf, te[cols], te.FACIES, n_repeats=20, random_state=0, n_jobs=-1)
    names = ['GR', 'RHOB', 'NPHI', 'PE', 'random number', 'random category']
    cols_c = [BLUE] * 4 + [ORANGE, ORANGE]
    f, axs = sub(2, height=4.3)
    y = np.arange(6)[::-1]
    for ax, vals, err, title, xl in [(axs[0], rf.feature_importances_, None, 'Impurity decrease (MDI), training data',
                                      'Share of impurity decrease'),
                                     (axs[1], pi.importances_mean, pi.importances_std,
                                      'Permutation, held-out wells W7–W8', 'Accuracy drop after shuffling')]:
        ax.barh(y, vals, .62, xerr=err, color=[tint(c, .35) for c in cols_c], edgecolor=cols_c, lw=1.3,
                error_kw=dict(ecolor=INK, lw=1.2, capsize=3), zorder=3)
        for yy, v, e in zip(y, vals, err if err is not None else [0] * 6):
            ax.text(max(v, 0) + e + .006, yy, f'{v:.3f}'.replace('-', '−'), va='center', fontsize=12.5, color=INK)
        ax.set_yticks(y, names)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(min(0, vals.min() - .01), vals.max() * 1.3 + .01)
        clean(ax, xl, None)
        ax.grid(axis='y', visible=False)
    save('ml2_mdi_bias', f, 'Synthetic facies logs plus two pure-noise columns (seed 0); RandomForestClassifier, '
         '300 unpruned trees, trained on wells W1-W6; permutation importance on W7-W8, 20 repeats (bars: mean, '
         'whiskers: standard deviation).', kind='computed')
    print('  mdi', dict(zip(names, rf.feature_importances_.round(3))), 'perm', dict(zip(names, pi.importances_mean.round(3))))


def _rop_split():
    d = rop_wells()
    return d[~d.WELL.isin(['D5', 'D6'])], d[d.WELL.isin(['D5', 'D6'])]


@fig
def correlated_importance():
    tr, te = _rop_split()
    tr, te = tr.copy(), te.copy()
    r2 = np.random.default_rng(1)
    for dd in (tr, te):
        dd['WOB_DH'] = .92 * dd.WOB + r2.normal(0, 4, len(dd))
    res = {}
    for key, cols in [('alone', ROP_X), ('copy', ROP_X + ['WOB_DH'])]:
        m = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(tr[cols], tr.ROP)
        base = mean_absolute_error(te.ROP, m.predict(te[cols]))
        pi = permutation_importance(m, te[cols], te.ROP, n_repeats=30, random_state=0, n_jobs=-1,
                                    scoring='neg_mean_absolute_error')
        res[key] = dict(zip(cols, pi.importances_mean))
        if key == 'copy':
            rs = np.random.default_rng(0)
            drops = []
            for _ in range(30):
                Xp = te[cols].copy()
                idx = rs.permutation(len(Xp))
                Xp['WOB'] = Xp['WOB'].values[idx]
                Xp['WOB_DH'] = Xp['WOB_DH'].values[idx]
                drops.append(mean_absolute_error(te.ROP, m.predict(Xp)) - base)
            res['group'] = float(np.mean(drops))
    corr = np.corrcoef(te.WOB, te.WOB_DH)[0, 1]
    f, axs = half()
    ax = axs[0]
    bars = [('Surface WOB\n(model without copy)', res['alone']['WOB'], BLUE, 'tint'),
            ('Surface WOB\n(copy present)', res['copy']['WOB'], BLUE, 'tint'),
            ('Downhole WOB\n(copy present)', res['copy']['WOB_DH'], TEAL, 'tint'),
            ('Both shuffled\ntogether', res['group'], PURPLE, 'solid')]
    y = np.arange(4)[::-1]
    for yy, (lab, v, col, fill) in zip(y, bars):
        ax.barh(yy, v, .6, color=col if fill == 'solid' else tint(col, .4), edgecolor=col, lw=1.4, zorder=3)
        ax.text(v + .12, yy, f'{v:.2f}', va='center', fontsize=13, color=INK)
    ax.set_yticks(y, [b_[0] for b_ in bars])
    ax.set_xlim(0, max(b_[1] for b_ in bars) * 1.25)
    ax.axhline(2.5, color=RULE, lw=1.2)
    clean(ax, 'MAE increase after shuffling (m/h)', None)
    ax.grid(axis='y', visible=False)
    save('ml2_correlated_importance', f, 'Synthetic drilling data (seed 11) with a downhole WOB column = 0.92 x '
         f'surface WOB + noise (sd 4 kN, seed 1; correlation {corr:.3f}); RandomForestRegressor (300 trees, '
         'min_samples_leaf 3) trained on D1-D4, permutation importance on D5-D6, 30 repeats.', kind='computed')
    print('  corr imp', {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in res.items()}, 'corr', round(corr, 4))


def _rop_hgb():
    d = rop_wells()
    tr = d[d.WELL != 'D6']
    m = HistGradientBoostingRegressor(categorical_features=[4], random_state=0).fit(tr[ROP_X], tr.ROP)
    return m, tr


@fig
def pdp_ice():
    m, tr = _rop_hgb()
    X = tr[ROP_X].sample(300, random_state=0)
    f, axs = sub(2, height=4.3)
    for ax, feat, xl in [(axs[0], 'WOB', 'Weight on bit (kN)'), (axs[1], 'RPM', 'Rotary speed (rpm)')]:
        j = ROP_X.index(feat)
        pdr = partial_dependence(m, X, [j], kind='both', grid_resolution=40)
        grid = pdr['grid_values'][0]
        ind = pdr['individual'][0]
        for row in ind[:70]:
            ax.plot(grid, row, color='#9DB6CF', lw=.8, alpha=.7, zorder=1)
        ax.plot(grid, pdr['average'][0], color=INK, lw=3.2, zorder=3)
        deciles = np.quantile(tr[feat], np.linspace(.1, .9, 9))
        for q in deciles:
            ax.plot([q, q], [0, 2.8], color=MUTED, lw=1.3, zorder=2)
        ax.set_ylim(0, 90)                      # the highest ICE lines reach about 84 m/h
        clean(ax, xl, 'Predicted ROP (m/h)' if feat == 'WOB' else None)
    axs[0].text(62, 83, 'thick: partial dependence (average)', fontsize=12.5, color=INK, fontweight='semibold')
    axs[0].text(62, 77, 'thin: ICE, one line per row (70 shown)', fontsize=12.5, color='#5E83A8')
    axs[1].text(.03, .95, 'short ticks on the x-axis: data deciles', fontsize=12.5, color=MUTED,
                transform=axs[1].transAxes, va='top')
    save('ml2_pdp_ice', f, 'HistGradientBoostingRegressor (FORMATION categorical, seed 0) trained on synthetic wells '
         'D1-D5; sklearn.inspection.partial_dependence over 300 sampled training rows, 40 grid points between the '
         '5th and 95th percentiles.', kind='computed')


@fig
def ice_interaction():
    m, tr = _rop_hgb()
    X = tr[ROP_X].sample(300, random_state=0)
    j = ROP_X.index('RPM')
    pdr = partial_dependence(m, X, [j], kind='both', grid_resolution=40)
    grid = pdr['grid_values'][0]
    ind = pdr['individual'][0]
    cen = ind - ind[:, :1]
    hard = X.FORMATION.values >= 2
    f, axs = half()
    ax = axs[0]
    for k in range(len(X))[:120]:
        ax.plot(grid, cen[k], color=PURPLE if hard[k] else TEAL, lw=.9, alpha=.55, zorder=1)
    # white halos keep the group means visible on top of the ICE lines of the same colour
    from matplotlib import patheffects as pe
    halo = [pe.Stroke(linewidth=5.2, foreground='white'), pe.Normal()]
    ax.plot(grid, cen[~hard].mean(0), color=shade(TEAL, .3), lw=2.6, ls='--', zorder=3, path_effects=halo)
    ax.plot(grid, cen[hard].mean(0), color=shade(PURPLE, .3), lw=2.6, ls='--', zorder=3, path_effects=halo)
    ax.plot(grid, cen.mean(0), color=INK, lw=3.2, zorder=4, path_effects=halo)
    for i, (txt, col, wt) in enumerate([('teal: soft rock', shade(TEAL, .2), 'semibold'),
                                        ('purple: hard rock', PURPLE, 'semibold'),
                                        ('dashed: group means', INK, 'normal'),
                                        ('solid: all rows (PD)', INK, 'normal')]):
        ax.text(grid[0] + 2, 25 - 2.2 * i, txt, color=col, fontsize=12.5, fontweight=wt, va='center')
    ax.set_ylim(-4, 27)
    clean(ax, 'Rotary speed (rpm)', f'Change in predicted ROP from {grid[0]:.0f} rpm (m/h)')
    save('ml2_ice_interaction', f, 'Centred ICE curves from the synthetic-drilling HistGradientBoostingRegressor '
         '(wells D1-D5); 120 of 300 sampled rows shown, coloured by formation (soft: claystone, sandstone; '
         'hard: limestone, dolomite). The generator gives rpm less effect in hard rock.', kind='computed')


@fig
def pdp_extrapolation():
    d = rop_wells().copy()
    r2 = np.random.default_rng(1)
    d['WOB_DH'] = .92 * d.WOB + r2.normal(0, 4, len(d))
    Z = d[['WOB', 'WOB_DH']].values
    mu, sd = Z.mean(0), Z.std(0)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit((Z - mu) / sd)
    dd, _ = nn.kneighbors((Z - mu) / sd)
    thr = np.quantile(dd[:, 1], .99)
    grid = np.linspace(np.quantile(Z[:, 0], .05), np.quantile(Z[:, 0], .95), 8)
    rng = np.random.default_rng(2)
    nsub = rng.choice(len(Z), 120, replace=False)
    G = np.array([[g, v_] for g in grid for v_ in Z[nsub, 1]])
    gd, _ = nn.kneighbors((G - mu) / sd, n_neighbors=1)
    off = gd[:, 0] > thr
    f, axs = half()
    ax = axs[0]
    ax.scatter(Z[:, 0], Z[:, 1], s=8, c='#B9C6D3', zorder=1)
    ax.scatter(G[~off, 0], G[~off, 1], s=30, marker='o', facecolors='none', edgecolors=BLUE, lw=1.3, zorder=3)
    ax.scatter(G[off, 0], G[off, 1], s=22, marker='x', c=ORANGE, lw=1.3, zorder=2)
    ax.text(.03, .97, f'{off.mean():.0%} of the evaluation points\nnever occur in the data (×)',
            transform=ax.transAxes, fontsize=13, color=ORANGE, va='top', fontweight='semibold', zorder=6,
            bbox=dict(boxstyle='round,pad=.3', fc='white', ec=ORANGE_T, alpha=.95))
    ax.text(.97, .04, 'grey dots: observed rows\nblue circles: evaluation points on the data',
            transform=ax.transAxes, fontsize=12, color=MUTED, ha='right', zorder=6,
            bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.9))
    ax.set_ylim(0, 270)                          # room above and below the data for the two notes
    clean(ax, 'Surface WOB, varied by the plot (kN)', 'Downhole WOB, kept as observed (kN)')
    save('ml2_pdp_extrapolation', f, 'Synthetic drilling data (seed 11) with downhole WOB = 0.92 x surface WOB + noise '
         '(sd 4 kN, seed 1). A partial-dependence curve for surface WOB pairs each of 8 grid values with 120 sampled '
         'observed downhole values. x = farther (standardized) from every observed row than 99% of rows are from '
         'their nearest neighbour.', kind='computed')
    print('  pdp off-support', round(off.mean(), 3))


def _shap_setup():
    d = rop_wells()
    tr, te = d[d.WELL != 'D6'], d[d.WELL == 'D6']
    F3 = ['WOB', 'RPM', 'FORMATION']
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, random_state=0).fit(
        tr[F3].values, tr.ROP.values)
    bg = tr[F3].sample(200, random_state=0).values
    x = te[F3].values[20]

    def v(S):
        Zb = bg.copy()
        for j in S:
            Zb[:, j] = x[j]
        return float(g.predict(Zb).mean())
    vals = {S: v(S) for r in range(4) for S in combinations(range(3), r)}
    phi = []
    for j in range(3):
        tot = 0.0
        for S, val in vals.items():
            if j in S:
                continue
            Sj = tuple(sorted(S + (j,)))
            tot += factorial(len(S)) * factorial(2 - len(S)) / factorial(3) * (vals[Sj] - val)
        phi.append(tot)
    return x, vals, phi, float(g.predict([x])[0]), float(te.ROP.values[20])


@fig
def shapley_lattice():
    x, vals, phi, pred, obs = _shap_setup()
    names = ['WOB', 'RPM', 'FORM']
    f, ax = canvas(11.8, 3.6)
    xs = [.95, 3.75, 6.55, 9.35]
    pos = {}
    for S in vals:
        lvl = len(S)
        members = [T for T in vals if len(T) == lvl]
        k = members.index(S)
        yc = 1.9 + (len(members) - 1) / 2 * 1.02 - k * 1.02
        pos[S] = (xs[lvl], yc)
    for S in vals:
        for j in range(3):
            if j in S:
                continue
            T = tuple(sorted(S + (j,)))
            (x0, y0), (x1, y1) = pos[S], pos[T]
            is_w = j == 0
            ax.plot([x0 + .8, x1 - .8], [y0, y1], color=BLUE if is_w else '#C9D3DD', lw=2.4 if is_w else 1.2,
                    zorder=1 if not is_w else 2)
            if is_w:
                wgt = factorial(len(S)) * factorial(2 - len(S)) / factorial(3)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my + .13, f'{vals[T] - vals[S]:+.1f} × {["1/3", "1/6", "1/6", "1/3"][[(), (1,), (2,), (1, 2)].index(S)]}',
                        ha='center', va='bottom', fontsize=12, color=BLUE, fontweight='semibold',
                        bbox=dict(boxstyle='round,pad=.12', fc='white', ec='none'), zorder=4)
    for S, (xc, yc) in pos.items():
        lab = '{ }' if not S else '{' + ', '.join(names[j] for j in S) + '}'
        full = len(S) == 3
        bw = 1.9 if full else 1.6
        box(ax, xc - bw / 2, yc - .32, bw, .64, lab, PURPLE if full else GREY, fill='solid' if full else 'white',
            fs=12.5, sub=f'v = {vals[S]:.1f}', sub_fs=12.5, z=3)
    label(ax, xs[0], 3.3, 'no feature known', fs=12, color=MUTED)
    label(ax, xs[3], 3.3, 'all features known', fs=12, color=MUTED)
    label(ax, 5.15, .18, f'φ(WOB) = weighted sum of the four blue steps = {phi[0]:+.2f} m/h   ·   v(S): mean prediction with '
          'the features in S fixed to this row', fs=12.5, color=INK)
    save('ml2_shapley_lattice', f, 'Exact interventional Shapley computation for one row of held-out well D6 '
         f'(WOB {x[0]:.0f} kN, {x[1]:.0f} rpm, dolomite): GradientBoostingRegressor on WOB, RPM and FORMATION '
         '(wells D1-D5, seed 0); v(S) = mean prediction over 200 background rows with the features in S set to '
         'this row. Values in m/h.', kind='computed')
    print('  shap', np.round(phi, 3), 'v', {k: round(v_, 3) for k, v_ in vals.items()}, 'pred', round(pred, 3), 'obs', obs)


@fig
def shap_waterfall():
    x, vals, phi, pred, obs = _shap_setup()
    names = [f'WOB = {x[0]:.0f} kN', f'RPM = {x[1]:.0f}', 'Formation = dolomite']
    order = np.argsort(-np.abs(phi))
    f, axs = half()
    ax = axs[0]
    base = vals[()]
    cur = base
    ys = np.arange(len(phi) + 2)[::-1]
    ax.barh(ys[0], base, .6, color=tint(GREY, .45), edgecolor=GREY, lw=1.3, zorder=3)
    ax.text(base + .5, ys[0], f'{base:.1f}', va='center', fontsize=13, color=INK)
    labels = ['Average prediction E[f]']
    for k, j in enumerate(order):
        yv = ys[k + 1]
        col = BLUE if phi[j] >= 0 else ORANGE
        left = cur if phi[j] >= 0 else cur + phi[j]
        ax.barh(yv, abs(phi[j]), .6, left=left, color=tint(col, .35), edgecolor=col, lw=1.3, zorder=3)
        ax.plot([cur, cur], [yv + .3 + .4, yv - .3], color=MUTED, lw=1, ls=':', zorder=2)
        ax.text(max(cur, cur + phi[j]) + .5, yv, f'{phi[j]:+.1f}'.replace('-', '−'), va='center', fontsize=13, color=col,
                fontweight='semibold')
        cur += phi[j]
        labels.append(names[j])
    ax.barh(ys[-1], cur, .6, color=PURPLE, edgecolor=PURPLE, lw=1.3, zorder=3)
    ax.text(cur + .5, ys[-1], f'{cur:.1f}', va='center', fontsize=13, color=PURPLE, fontweight='semibold')
    ax.text(cur + 3.6, ys[-1], f'(observed {obs:.1f})', va='center', fontsize=12.5, color=MUTED)
    labels.append('This prediction f(x)')
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, 30)
    clean(ax, 'Predicted ROP (m/h)', None)
    ax.grid(axis='y', visible=False)
    save('ml2_shap_waterfall', f, 'Exact interventional Shapley values for one row of synthetic well D6 (see '
         f'ml2_shapley_lattice); base {base:.2f} + contributions {", ".join(f"{p_:+.2f}" for p_ in phi)} = '
         f'{pred:.2f} m/h.', kind='computed')


# =================================================================== PART 6: pipelines and deployment
@fig
def column_transformer():
    f, ax = canvas(11.8, 4.3)
    cols = [('WOB', BLUE), ('RPM', BLUE), ('FLOW', BLUE), ('FORMATION', TEAL), ('BIT_TYPE', TEAL),
            ('WELL_ID', GREY), ('TIME', GREY)]
    x0, ytop, cw = .2, 3.95, .5
    label(ax, 1.1, 4.2, 'Input table', fs=13.5, weight='semibold')
    for k, (c, col) in enumerate(cols):
        yb = ytop - (k + 1) * .47
        box(ax, x0, yb, 1.75, .38, c, col, fill='tint', fs=12, radius=.05)
    # numeric branch
    yn, yc, ym = 3.1, 1.85, .62
    bracket_x = 2.1
    for (y0, y1, yy, col) in [(ytop - .47 * 3, ytop - .02, yn, BLUE), (ytop - .47 * 5, ytop - .47 * 3 - .05, yc, TEAL),
                              (ytop - .47 * 7, ytop - .47 * 5 - .05, ym, GREY)]:
        ax.plot([bracket_x, bracket_x + .12, bracket_x + .12, bracket_x], [y0, y0, y1, y1], color=col, lw=1.4)
        arrow(ax, (bracket_x + .12, (y0 + y1) / 2), (2.75, yy), color=col, lw=1.4, head=11)
    b1 = box(ax, 2.8, yn - .3, 1.8, .6, 'SimpleImputer', BLUE, fs=13, sub='median', sub_fs=11.5)
    b2 = box(ax, 4.85, yn - .3, 2.15, .6, 'StandardScaler', BLUE, fs=13, sub='optional for trees', sub_fs=11.5)
    arrow(ax, b1.r, b2.l, color=BLUE, lw=1.4, head=11)
    c1 = box(ax, 2.8, yc - .3, 1.8, .6, 'SimpleImputer', TEAL, fs=13, sub='most frequent', sub_fs=11.5)
    c2 = box(ax, 4.85, yc - .3, 2.15, .6, 'OneHotEncoder', TEAL, fs=13, sub='handle_unknown="ignore"', sub_fs=11.5)
    arrow(ax, c1.r, c2.l, color=TEAL, lw=1.4, head=11)
    m1 = box(ax, 2.8, ym - .3, 4.2, .6, 'not predictors: used for GroupKFold and time ordering', GREY, fill='white',
             fs=12, ls='--', weight='normal')
    # merge
    mx = 7.55
    for yy in (yn, yc):
        arrow(ax, (7.05, yy), (mx, 2.45), color=MUTED, lw=1.3, head=11)
    fm = box(ax, mx, 2.1, 1.35, .7, 'feature\nmatrix', GREY, fs=12.5)
    mdl = box(ax, 9.25, 2.1, 1.2, .7, 'model', BLUE, fill='solid', fs=13.5)
    arrow(ax, fm.r, mdl.l, color=INK, lw=1.5, head=12)
    arrow(ax, mdl.r, (11.0, 2.45), color=INK, lw=1.5, head=12)
    label(ax, 11.35, 2.45, 'ROP', fs=13.5, weight='semibold', color=ORANGE)
    bracket(ax, 2.8, 10.45, 3.72, 'Pipeline: every fitted step learns from training folds only', color=BLUE,
            below=False, fs=12.5)
    save('ml2_column_transformer', f, 'Schematic of a ColumnTransformer inside a Pipeline for the synthetic drilling '
         'table; column names are illustrative.', kind='schematic')


@fig
def leakage_selection():
    rng = np.random.default_rng(0)
    n, p, k, reps = 50, 2000, 20, 20
    outside, inside = [], []
    for rep in range(reps):
        X = rng.normal(size=(n, p))
        y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
        rng.shuffle(y)
        cv = StratifiedKFold(5, shuffle=True, random_state=rep)
        Xs = SelectKBest(f_classif, k=k).fit_transform(X, y)
        outside.append(cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=cv).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000))
        inside.append(cross_val_score(pipe, X, y, cv=cv).mean())
    f, axs = half()
    ax = axs[0]
    for k_, (vals, col, lab) in enumerate([(outside, ORANGE, 'Select on all rows,\nthen cross-validate'),
                                            (inside, TEAL, 'Select inside each\ntraining fold')]):
        xs = k_ + rng.uniform(-.13, .13, len(vals))
        ax.scatter(xs, vals, s=34, color=col, alpha=.85, edgecolors='white', lw=.6, zorder=3)
        ax.plot([k_ - .25, k_ + .25], [np.mean(vals)] * 2, color=INK, lw=2.6, zorder=4)
        ax.text(k_ + .29, np.mean(vals), f'{np.mean(vals):.2f}', va='center', fontsize=14, color=INK,
                fontweight='semibold', zorder=5, bbox=dict(boxstyle='round,pad=.12', fc='white', ec='none'))
    ax.axhline(.5, color=MUTED, lw=1.2, ls='--')
    ax.text(-.45, .485, 'chance = 0.50', fontsize=12, color=MUTED, ha='left', va='top')
    ax.set_xticks([0, 1], ['Select on all rows,\nthen cross-validate', 'Select inside each\ntraining fold'])
    ax.set_xlim(-.5, 1.55)
    ax.set_ylim(.2, 1.02)
    clean(ax, None, 'Cross-validated accuracy')
    ax.grid(axis='x', visible=False)
    save('ml2_leakage_selection', f, f'{reps} synthetic datasets (seed 0), each {n} rows x {p} pure-noise features '
         f'with random balanced labels; SelectKBest(f_classif, k = {k}) + LogisticRegression; 5-fold stratified CV. '
         f'Means: outside {np.mean(outside):.3f}, inside {np.mean(inside):.3f}.', kind='computed')
    print('  leakage', round(float(np.mean(outside)), 3), round(float(np.mean(inside)), 3), round(float(np.min(outside)), 3), round(float(np.max(inside)), 3))


@fig
def artifact_bundle():
    f, ax = canvas(6.3, 4.9)
    ax.add_patch(FancyBboxPatch((.25, .95), 5.8, 3.55, boxstyle='round,pad=0,rounding_size=.12', fc=PANEL,
                                ec='#AEBBC8', lw=1.4, zorder=0))
    ax.add_patch(FancyBboxPatch((.25, 4.35), 2.4, .4, boxstyle='round,pad=0,rounding_size=.08', fc=PANEL,
                                ec='#AEBBC8', lw=1.4, zorder=0))
    label(ax, .42, 4.55, 'rop_model_v3/', fs=13.5, weight='semibold', ha='left', color=INK)
    items = [('pipeline.joblib', 'imputers, encoder and forest, fitted together', BLUE),
             ('metadata.json', 'features, units, versions, data version, threshold', TEAL),
             ('reference_batch.csv', '20 rows with their expected predictions', GOLD),
             ('model_card.md', 'intended use, evaluation, limitations, owner', PURPLE)]
    for k, (name, desc, col) in enumerate(items):
        yb = 3.65 - k * .8
        box(ax, .5, yb, 5.3, .66, '', col, fill='white', radius=.06, lw=1.3)
        ax.add_patch(Rectangle((.5, yb), .09, .66, fc=col, ec='none', zorder=3))
        label(ax, .78, yb + .45, name, fs=13, weight='semibold', ha='left', color=INK, family='IBM Plex Mono')
        label(ax, .78, yb + .19, desc, fs=11.5, ha='left', color=MUTED)
    check(ax, .55, .45, s=.15)
    label(ax, .85, .45, 'Reload in a fresh process and match the reference predictions', fs=12, ha='left',
          color=INK)
    save('ml2_artifact_bundle', f, 'Schematic of a versioned model artifact; file names are illustrative.',
         kind='schematic')


@fig
def monitoring_loop():
    f, ax = canvas(6.3, 4.9)
    cx, cy, R = 3.15, 2.45, 1.72
    steps = [('Inputs arrive', GREY), ('Check schema,\nranges, gaps', TEAL), ('Predict\nand log', BLUE),
             ('Labels arrive\n(days later)', GREY), ('Errors by well,\nperiod, regime', ORANGE),
             ('Investigate,\nretrain or fall back', PURPLE)]
    n = len(steps)
    angs = np.pi / 2 - np.arange(n) * 2 * np.pi / n
    pts = [(cx + R * np.cos(a) * 1.25, cy + R * np.sin(a)) for a in angs]
    boxes = []
    for (lab, col), (x, y) in zip(steps, pts):
        box(ax, x - .9, y - .34, 1.8, .68, lab, col, fs=12, radius=.08, z=3)
        boxes.append(ax.patches[-1])
    for k in range(n):
        a0, a1 = angs[k] - .33, angs[k] - 2 * np.pi / n + .33
        p0 = (cx + R * np.cos(a0) * 1.25, cy + R * np.sin(a0))
        p1 = (cx + R * np.cos(a1) * 1.25, cy + R * np.sin(a1))
        # patchA/patchB clip the arc at the box outlines, so every arrowhead lands visibly on its box
        ax.add_patch(FancyArrowPatch(p0, p1, patchA=boxes[k], patchB=boxes[(k + 1) % n], shrinkA=2, shrinkB=3,
                                     connectionstyle='arc3,rad=-.28', arrowstyle='-|>', mutation_scale=13,
                                     color=MUTED, lw=1.5, zorder=1))
    chip(ax, cx, cy + .12, 'drift tests: Module 5', color=MUTED, fs=12)
    label(ax, cx, cy - .3, 'owner + review interval', fs=12, color=INK)
    save('ml2_monitoring_loop', f, 'Schematic of a monitoring loop for a deployed model.', kind='schematic')
