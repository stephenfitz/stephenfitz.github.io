"""Figures for module key 'nn3' (energy modelling capstone). Names start with 'nn3_'.

Data figures are computed from the supplied, read-only capstone files in
neural_networks_course/Day_3_Capstone/exercises (training files only, plus the
schema, keys, ENTITY/TimeSlot and Depth columns of the test inputs where a
figure says so; final test labels are never read). Models are simple scikit-
learn baselines or the reference workflow in curriculum/capstone.py, always
evaluated with an explicit split.

Colour meaning used throughout the deck:
  class 0 / GAS_LOCK = 0 / sensor label -1 = BLUE; class 1 / GAS_LOCK = 1 / label 1 = ORANGE
  training portion = BLUE, validation portion = TEAL, supplied test file = ORANGE (locked)
  no-skill reference = grey, linear baseline = BLUE, tree ensemble = TEAL,
  neural network = ORANGE, k-nearest neighbours = PURPLE
  random row split (optimistic) = ORANGE, grouped / blocked split = TEAL
  day plan: lecture = BLUE, sprint = ORANGE, checkpoint = TEAL, presentation = PURPLE
"""
from functools import lru_cache

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut,
                                     StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

KEY = 'nn3'
fig = register(KEY)

DATA = ROOT / 'neural_networks_course' / 'Day_3_Capstone' / 'exercises'
GREY = '#56687A'
LIGHT = '#C9D3DD'
TRACE = '#2B3F55'
ESP_F = ['PIP', 'PDP', 'Q', 'THP', 'Mamps', 'PIT']
LOG_F = ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'RELPOS']
SEN_F = [f'col{i}' for i in range(1, 11)]


# ------------------------------------------------------------------ data
@lru_cache(maxsize=None)
def _esp():
    d = pd.read_csv(DATA / '01_esp_gas_lock' / 'esp_train.csv')
    d['t'] = pd.to_datetime(d.TimeSlot)
    return d


def esp():
    return _esp().copy()


@lru_cache(maxsize=None)
def _esp_test_inputs():
    """Test inputs only (ENTITY, TimeSlot); the label file is never read here."""
    t = pd.read_csv(DATA / '01_esp_gas_lock' / 'esp_test.csv', usecols=['uuid', 'ENTITY', 'TimeSlot'])
    t['t'] = pd.to_datetime(t.TimeSlot)
    return t


@lru_cache(maxsize=None)
def _logs():
    return pd.read_csv(DATA / '02_log_curves' / 'Log curve - Train.csv')


def logs():
    return _logs().copy()


@lru_cache(maxsize=None)
def _sensor():
    return pd.read_csv(DATA / '03_sensor_signals' / 'Sensor_train.csv')


def sensor():
    return _sensor().copy()


def log_segments(d):
    """Segment number per row: a new segment starts wherever Depth decreases."""
    resets = np.where(np.diff(d.Depth.to_numpy()) < 0)[0] + 1
    seg = np.zeros(len(d), int)
    for r in resets:
        seg[r:] += 1
    return seg, resets


def runs(mask):
    """(start, stop) index pairs of consecutive True values."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def lr_pipe(**kw):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=1500, **kw))


def esp_xyg():
    d = _esp()
    return d[ESP_F], d.GAS_LOCK.to_numpy(), d.ENTITY.to_numpy()


def date_axis(ax, fmt='%b %d'):
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# ------------------------------------------------------ motivating case
@lru_cache(maxsize=None)
def esp_split_gap():
    X, y, g = esp_xyg()
    models = {
        'Logistic regression': lambda: lr_pipe(),
        'k-nearest neighbours': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                                      KNeighborsClassifier(5)),
        'Random forest': lambda: make_pipeline(SimpleImputer(strategy='median'),
                                               RandomForestClassifier(200, random_state=0, n_jobs=1)),
        'Gradient boosting': lambda: HistGradientBoostingClassifier(random_state=0),
    }
    out = {}
    for name, mk in models.items():
        pr = cross_val_predict(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0),
                               method='predict_proba')[:, 1]
        pg = cross_val_predict(mk(), X, y, cv=LeaveOneGroupOut(), groups=g, method='predict_proba')[:, 1]
        out[name] = (accuracy_score(y, pr >= .5), accuracy_score(y, pg >= .5),
                     average_precision_score(y, pr), average_precision_score(y, pg))
    return out, float(1 - y.mean())


@fig
def split_gap():
    res, majority = esp_split_gap()
    f, axs = sub(1, height=3.9)
    ax = axs[0]
    names = list(res)[::-1]
    for i, name in enumerate(names):
        r, g = res[name][:2]
        ax.plot([g, r], [i, i], color=LIGHT, lw=5, solid_capstyle='round', zorder=1)
        ax.scatter([r], [i], s=150, color=ORANGE, zorder=3, edgecolors='white', linewidths=1.2)
        ax.scatter([g], [i], s=150, color=TEAL, zorder=3, edgecolors='white', linewidths=1.2)
        ax.annotate(f'{r:.2f}', (r, i), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=13, color=ORANGE, fontweight='semibold')
        ax.annotate(f'{g:.2f}', (g, i), xytext=(-10, 0), textcoords='offset points', va='center',
                    ha='right', fontsize=13, color=TEAL, fontweight='semibold')
    ax.axvline(majority, color=GREY, ls='--', lw=1.5, zorder=0)
    ax.text(majority + .004, -.48, f'Always predict "no gas lock": {majority:.2f}',
            fontsize=12, color=GREY, va='center')
    for xx, txt, col in ((.765, 'Random 5-fold row split', ORANGE), (.885, 'Leave one entity out', TEAL)):
        ax.scatter([xx], [-.5], s=110, color=col, edgecolors='white')
        ax.text(xx + .007, -.5, txt, fontsize=13, color=col, fontweight='semibold', va='center')
    ax.set_yticks(range(len(names)), names, fontsize=14)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(.55, 1.0)
    ax.set_ylim(-.75, len(names) - .35)
    ax.spines['left'].set_visible(False)
    clean(ax, 'Pooled out-of-fold accuracy (threshold 0.5)')
    ax.grid(axis='y', visible=False)
    save('nn3_split_gap', f, 'Computed from esp_train.csv (3,243 rows, 7 entities): scikit-learn models with '
         'median imputation and scaling fitted inside each fold; KFold(5, shuffle, seed 0) versus '
         'LeaveOneGroupOut on ENTITY; pooled out-of-fold accuracy at threshold 0.5.', kind='data')


# ------------------------------------------------------------ ESP figures
@fig
def esp_traces():
    d = esp()
    f, axs = plt.subplots(3, 2, figsize=(11.8, 4.6), sharex='col', layout='constrained')
    cols = ['PIP', 'PDP', 'Mamps']
    for j, ent in enumerate(['esp_30', 'esp_39']):
        g = d[d.ENTITY == ent].sort_values('t', kind='stable').reset_index(drop=True)
        t = g.t.to_numpy()
        step = pd.Timedelta(hours=1).to_numpy()
        for i, col in enumerate(cols):
            ax = axs[i, j]
            for a, b in runs(g.GAS_LOCK.to_numpy() == 1):
                ax.axvspan(t[a] - step / 2, t[b - 1] + step / 2, color=ORANGE, alpha=.16, lw=0)
            miss = g[col].isna().to_numpy()
            for a, b in runs(miss):
                ax.axvspan(t[a] - step / 2, t[b - 1] + step / 2, facecolor='none', edgecolor=FAINT,
                           hatch='///', lw=0)
            ax.plot(t, g[col], color=TRACE, lw=1.3)
            ax.set_ylabel(col, fontsize=13)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.grid(axis='y', alpha=1)
            ax.set_axisbelow(True)
            date_axis(ax)
        axs[0, j].set_title(ent if ent == 'esp_30' else 'esp_39 (sensors missing in the last 50 rows)',
                            fontsize=14)
    f.legend(handles=[Patch(fc=tint(ORANGE, .75), ec='none', label='GAS_LOCK = 1'),
                      Patch(fc='white', ec=FAINT, hatch='///', label='All six sensors missing')],
             loc='outside lower center', ncol=2, fontsize=12.5)
    save('nn3_esp_traces', f, 'Computed from esp_train.csv: entities esp_30 and esp_39, rows sorted by '
         'TimeSlot; values as supplied (units undocumented); shading marks GAS_LOCK = 1 and rows whose six '
         'sensors are all missing.', kind='data')


@fig
def esp_class_dist():
    d = esp().dropna(subset=ESP_F)
    f, axs = plt.subplots(2, 3, figsize=(11.8, 4.5), layout='constrained')
    for ax, col in zip(axs.ravel(), ESP_F):
        lo, hi = d[col].min(), d[col].max()
        bins = np.linspace(lo, hi, 36)
        for lab, c in ((0, BLUE), (1, ORANGE)):
            v = d.loc[d.GAS_LOCK == lab, col]
            ax.hist(v, bins=bins, density=True, histtype='stepfilled', color=c, alpha=.22)
            ax.hist(v, bins=bins, density=True, histtype='step', color=c, lw=1.6)
        ax.set_title(col, fontsize=14, pad=4)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    f.legend(handles=[Patch(fc=tint(BLUE, .7), ec=BLUE, label='GAS_LOCK = 0 (1,969 rows)'),
                      Patch(fc=tint(ORANGE, .7), ec=ORANGE, label='GAS_LOCK = 1 (1,224 rows)')],
             loc='outside upper center', ncol=2, fontsize=12.5)
    f.supxlabel('Value as supplied (units undocumented); histograms normalized within each class',
                fontsize=13, color=INK)
    save('nn3_esp_class_dist', f, 'Computed from esp_train.csv: 3,193 rows with all six sensors present, '
         'pooled over seven entities; step histograms normalized within each class.', kind='data')


@fig
def esp_timeline():
    d = esp()
    test = _esp_test_inputs()
    train_ents = sorted(d.ENTITY.unique())
    test_ents = sorted(test.ENTITY.unique())
    f, ax = plt.subplots(figsize=(11.8, 4.5), layout='constrained')
    step = pd.Timedelta(hours=1)
    ys = {e: i for i, e in enumerate(train_ents)}
    ys.update({e: len(train_ents) + .6 + i for i, e in enumerate(test_ents)})
    h = .56
    XM = mdates.date2num(pd.Timestamp('2022-01-22T20:00Z'))
    XT = mdates.date2num(pd.Timestamp('2022-01-24T12:00Z'))
    for e in train_ents:
        g = d[d.ENTITY == e].reset_index(drop=True)
        y = ys[e]
        t0, t1 = g.t.min(), g.t.max() + step
        ax.add_patch(Rectangle((mdates.date2num(t0), y - h / 2), mdates.date2num(t1) - mdates.date2num(t0), h,
                               fc=PANEL, ec=LIGHT, lw=1, zorder=1))
        gs = g.sort_values('t', kind='stable').reset_index(drop=True)
        for a, b in runs(gs.GAS_LOCK.to_numpy() == 1):
            x0 = mdates.date2num(gs.t[a])
            x1 = mdates.date2num(gs.t[b - 1] + step)
            ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, fc=ORANGE, ec='none', alpha=.8, zorder=2))
        for a, b in runs(gs.PIP.isna().to_numpy()):
            x0 = mdates.date2num(gs.t[a])
            x1 = mdates.date2num(gs.t[b - 1] + step)
            ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, fc='white', ec=GREY, hatch='////', lw=1,
                                   zorder=3))
        # repeated or out-of-order timestamps, in file order
        tt = g.t.to_numpy()
        bad = np.zeros(len(g), bool)
        bad[1:] = tt[1:] <= tt[:-1]
        for tb in g.t[bad]:
            xb = mdates.date2num(tb)
            ax.plot([xb, xb], [y - h / 2 - .12, y + h / 2 + .12], color=RED, lw=2.2, zorder=4)
        n_missing = int(g[ESP_F].isna().sum().sum())
        n_bad = int(bad.sum())
        ax.text(XM, y, f'{n_missing:,}', fontsize=12.5, va='center', ha='center',
                color=INK if n_missing else FAINT, fontweight='semibold' if n_missing else 'normal')
        ax.text(XT, y, f'{n_bad}', fontsize=12.5, va='center', ha='center',
                color=INK if n_bad else FAINT, fontweight='semibold' if n_bad else 'normal')
    ht = .78                     # taller than the training bars so the label clears the dashed border
    for e in test_ents:
        g = test[test.ENTITY == e]
        y = ys[e]
        t0, t1 = g.t.min(), g.t.max() + step
        ax.add_patch(Rectangle((mdates.date2num(t0), y - ht / 2), mdates.date2num(t1) - mdates.date2num(t0), ht,
                               fc='white', ec=GREY, ls='--', lw=1.3, zorder=1))
        ax.text(mdates.date2num(t0 + (t1 - t0) / 2), y, 'test inputs only; labels locked', fontsize=11.5,
                color=GREY, ha='center', va='center')
    ax.text(XM, -.85, 'Missing\ncells', fontsize=12, color=MUTED, va='center', ha='center', linespacing=1.05)
    ax.text(XT, -.85, 'TimeSlot\nissues', fontsize=12, color=MUTED, va='center', ha='center', linespacing=1.05)
    ax.set_yticks([ys[e] for e in train_ents + test_ents], train_ents + test_ents, fontsize=12.5)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(len(train_ents) + len(test_ents) + .2, -1.3)
    ax.set_xlim(mdates.date2num(pd.Timestamp('2022-01-01T00:00Z')) - .3,
                mdates.date2num(pd.Timestamp('2022-01-25T12:00Z')))
    date_axis(ax)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_bounds(mdates.date2num(pd.Timestamp('2022-01-01T00:00Z')) - .3,
                                   mdates.date2num(pd.Timestamp('2022-01-22T00:00Z')))
    ax.set_xlabel('TimeSlot (hourly rows)')
    ax.xaxis.set_label_coords(.42, -.1)
    ax.legend(handles=[Patch(fc=ORANGE, alpha=.8, label='GAS_LOCK = 1'),
                       Patch(fc='white', ec=GREY, hatch='////', label='All six sensors missing'),
                       Line2D([], [], color=RED, lw=2.2, label='Repeated or out-of-order TimeSlot')],
              loc='upper left', bbox_to_anchor=(0, 1.02), ncol=3, fontsize=12, frameon=False)
    save('nn3_esp_timeline', f, 'Computed from esp_train.csv (labels, missing cells, timestamps) and from the '
         'ENTITY and TimeSlot columns of esp_test.csv; the test label file is not read.', kind='data')


@lru_cache(maxsize=None)
def esp_reference_split():
    X, y, g = esp_xyg()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=2, random_state=42).split(X, y, groups=g))
    return sorted(set(g[va]))


@fig
def esp_split():
    d = _esp()
    train_ents = sorted(d.ENTITY.unique())
    test_ents = sorted(_esp_test_inputs().ENTITY.unique())
    val_ref = esp_reference_split()
    f, ax = canvas(6.3, 4.9)
    x0, cw, ch, gap, dy = 1.62, .43, .33, .22, .065
    xt = x0 + len(train_ents) * cw + gap

    def cell(x, y, col, hatch=None):
        ax.add_patch(FancyBboxPatch((x + .025, y), cw - .05, ch, boxstyle='round,pad=0,rounding_size=0.05',
                                    fc=col, ec=ORANGE if hatch else 'none', lw=.9, hatch=hatch))

    def row(y, name, vals, bold=False):
        label(ax, x0 - .1, y + ch / 2, name, fs=12.5, ha='right', weight='semibold' if bold else 'normal',
              color=INK if bold else MUTED)
        for c, e in enumerate(train_ents):
            cell(x0 + c * cw, y, TEAL if e in vals else BLUE)
        for c in range(len(test_ents)):
            cell(xt + c * cw, y, tint(ORANGE, .8), '///')

    ytop = 3.5
    for c, e in enumerate(train_ents):
        label(ax, x0 + (c + .5) * cw, ytop + ch + .08, e, fs=12, rotation=90, va='bottom')
    for c, e in enumerate(test_ents):
        label(ax, xt + (c + .5) * cw, ytop + ch + .08, e, fs=12, rotation=90, va='bottom', color=ORANGE)
    row(ytop, 'Reference split', val_ref, bold=True)
    label(ax, x0 - .1, ytop - .3, 'Leave one entity out:', fs=12, ha='right', color=MUTED, style='italic')
    y = ytop - .3
    for k, e in enumerate(train_ents):
        y -= ch + dy if k else ch / 2 + dy + .12
        row(y, f'fold {k + 1}', [e])
    yb = y - .1
    bracket(ax, x0 + .03, x0 + len(train_ents) * cw - .03, yb, 'esp_train.csv, 7 entities', color=MUTED,
            below=True, fs=12)
    bracket(ax, xt + .03, xt + len(test_ents) * cw - .03, yb, 'esp_test.csv', color=ORANGE, below=True, fs=12)
    for k, (txt, col, hatch) in enumerate([('Training', BLUE, None), ('Validation', TEAL, None),
                                           ('Locked test', tint(ORANGE, .8), '///')]):
        yy = 4.6 - k * .3
        ax.add_patch(FancyBboxPatch((.2, yy - .1), .3, .22, boxstyle='round,pad=0,rounding_size=0.04', fc=col,
                                    ec=ORANGE if hatch else 'none', lw=.9, hatch=hatch))
        label(ax, .6, yy + .01, txt, fs=12.5, ha='left')
    save('nn3_esp_split', f, 'Computed split membership: GroupShuffleSplit(n_splits=1, test_size=2, '
         'random_state=42) on ENTITY as in curriculum/capstone.py, and LeaveOneGroupOut folds; test entity '
         'names from esp_test.csv.', kind='computed')


# ------------------------------------------------------------ PE log figures
LOG_CUT = int(.8 * 2922)       # reference workflow: last 20% of source rows is validation (row 2337 on)


@fig
def log_tracks():
    d = logs()
    seg, resets = log_segments(d)
    g = d[seg == 0]
    tracks = [('GR', [('GR', TEAL, '-')]), ('ILD_log10', [('ILD_log10', PURPLE, '-')]),
              ('PHIND, DeltaPHI', [('PHIND', BLUE, '-'), ('DeltaPHI', GREY, '--')]),
              ('RELPOS', [('RELPOS', GOLD, '-')]), ('PE (target)', [('PE', ORANGE, '-')])]
    f, axs = plt.subplots(1, len(tracks), figsize=(11.8, 4.7), sharey=True, layout='constrained')
    for ax, (title, curves) in zip(axs, tracks):
        for col, c, ls in curves:
            ax.plot(g[col], g.Depth, color=c, lw=1.4, ls=ls)
        ax.set_title(title, fontsize=14, color=curves[0][1] if len(curves) == 1 else INK, pad=6)
        ax.grid(alpha=1)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.tick_top()
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
    axs[2].text(.97, .02, 'PHIND solid\nDeltaPHI dashed', transform=axs[2].transAxes, ha='right', va='bottom',
                fontsize=11.5, color=MUTED, zorder=1,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    axs[0].invert_yaxis()
    axs[0].set_ylabel('Depth as supplied (increasing downward)')
    save('nn3_log_tracks', f, f'Computed from Log curve - Train.csv: source rows 0-{len(g) - 1}, the first '
         'block of increasing Depth; curves as supplied, units undocumented; well identity is not given.',
         kind='data')


@fig
def log_corr():
    d = logs()
    cols = LOG_F + ['PE']
    c = d[cols].corr().to_numpy()
    f, axs = half(width=5.8, height=4.4)
    ax = axs[0]
    shown = c.copy()
    np.fill_diagonal(shown, 0)
    ax.imshow(shown, cmap=CMAP_DIV, vmin=-1, vmax=1)
    for i in range(len(cols)):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc=PANEL, ec='white', lw=1))
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c[i, j]
            ax.text(j, i, '' if i == j else f'{v:.2f}'.replace('-0.', '−.').replace('0.', '.'),
                    ha='center', va='center', fontsize=12,
                    color='white' if (abs(v) > .9 and i != j) else INK, fontweight='semibold' if 'PE' in (cols[i], cols[j]) and i != j else 'normal')
    k = cols.index('PE')
    ax.add_patch(Rectangle((-.5, k - .5), len(cols), 1, fc='none', ec=ORANGE, lw=2.2))
    ax.set_xticks(range(len(cols)), [x.replace('ILD_log10', 'ILD') for x in cols], rotation=45, ha='right',
                  fontsize=12)
    ax.set_yticks(range(len(cols)), [x.replace('ILD_log10', 'ILD') for x in cols], fontsize=12)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    save('nn3_log_corr', f, 'Computed from Log curve - Train.csv: Pearson correlations over all 2,922 training '
         'rows (descriptive; rows are not independent). ILD = ILD_log10.', kind='data')


@fig
def log_pe_dist():
    d = logs()
    tr, va = d.PE.iloc[:LOG_CUT], d.PE.iloc[LOG_CUT:]
    f, axs = half(width=5.8, height=4.4)
    ax = axs[0]
    bins = np.arange(0, 8.6, .25)
    for v, c, lab in ((tr, BLUE, f'Training portion\n(rows 0–{LOG_CUT - 1:,})'),
                      (va, TEAL, f'Validation block\n(rows {LOG_CUT:,}–{len(d) - 1:,})')):
        ax.hist(v, bins=bins, density=True, histtype='stepfilled', color=c, alpha=.2)
        ax.hist(v, bins=bins, density=True, histtype='step', color=c, lw=1.8)
        ax.axvline(v.mean(), color=c, lw=1.6, ls='--')
    ax.text(tr.mean() - .12, 1.02, f'mean {tr.mean():.2f}', color=BLUE, fontsize=12, ha='right',
            transform=ax.get_xaxis_transform())
    ax.text(va.mean() + .12, 1.02, f'mean {va.mean():.2f}', color=TEAL, fontsize=12, ha='left',
            transform=ax.get_xaxis_transform())
    ax.text(.98, .62, f'Training portion\nrows 0–{LOG_CUT - 1:,}', color=BLUE, fontsize=12.5, ha='right',
            transform=ax.transAxes, fontweight='semibold')
    ax.text(.98, .40, f'Validation block\nrows {LOG_CUT:,}–{len(d) - 1:,}', color=TEAL, fontsize=12.5,
            ha='right', transform=ax.transAxes, fontweight='semibold')
    clean(ax, 'PE as supplied', 'Density')
    ax.set_xlim(0, 8.5)
    save('nn3_log_pe_dist', f, 'Computed from Log curve - Train.csv: PE in the training portion and in the '
         'validation block of the reference split (last 20% of source rows).', kind='data')


@fig
def log_split():
    d = logs()
    seg, resets = log_segments(d)
    rows = np.arange(len(d))
    f, axs = half()
    ax = axs[0]
    ax.plot(rows[:LOG_CUT], d.Depth.iloc[:LOG_CUT], color=BLUE, lw=1.6)
    ax.plot(rows[LOG_CUT:], d.Depth.iloc[LOG_CUT:], color=TEAL, lw=1.9)
    ax.axvline(LOG_CUT - .5, color=INK, lw=1.2, ls='--')
    ax.text(LOG_CUT - 40, 2560, f'cut at row {LOG_CUT:,}', ha='right', va='bottom', fontsize=12, color=INK)
    ax.text(1100, 3155, 'Training portion', color=BLUE, fontsize=13.5, fontweight='semibold', ha='center')
    ax.text(2640, 3155, 'Validation', color=TEAL, fontsize=13.5, fontweight='semibold', ha='center')
    # duplicated block: rows 2297-2308 repeat rows 450-461
    dup = d.duplicated(keep=False).to_numpy() & (seg == seg[2297])
    xs = rows[dup]
    ax.annotate('12 rows repeat\nrows 450–461', (xs.mean(), d.Depth.iloc[xs].mean()), xytext=(1380, 3110),
                fontsize=12, color=RED, ha='center', va='center',
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.2, connectionstyle='arc3,rad=-.2'))
    ax.scatter(xs, d.Depth.iloc[xs], s=14, color=RED, zorder=4)
    ax.annotate(f'{len(resets)} depth resets:\n{len(resets) + 1} increasing blocks', (resets[2], d.Depth.iloc[resets[2]]),
                xytext=(330, 2600), fontsize=12, color=MUTED, ha='center',
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.1))
    clean(ax, 'Source row position', 'Depth as supplied')
    ax.set_ylim(2530, 3200)
    save('nn3_log_split', f, f'Computed from Log curve - Train.csv: Depth in source-row order; the reference split '
         f'keeps rows 0-{LOG_CUT - 1} for training and holds out rows {LOG_CUT}-{len(d) - 1}.', kind='data')


# ------------------------------------------------------------ sensor figures
@fig
def sensor_class_dist():
    d = sensor()
    y = d.target.to_numpy()
    f = plt.figure(figsize=(11.8, 4.4), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax = f.add_subplot(gs[0])
    bins = np.logspace(np.log10(1.0), np.log10(90), 45)
    for lab, c, name in ((-1, BLUE, 'target = −1'), (1, ORANGE, 'target = 1')):
        v = d.loc[y == lab, 'col1']
        ax.hist(v, bins=bins, histtype='stepfilled', color=c, alpha=.22)
        ax.hist(v, bins=bins, histtype='step', color=c, lw=1.7)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 80], ['1', '2', '5', '10', '20', '50', '80'])
    n_neg, n_pos = int((y == -1).sum()), int((y == 1).sum())
    ax.text(2.3, .93, f'target = −1\n{n_neg:,} rows', color=BLUE, fontsize=13, fontweight='semibold',
            transform=ax.get_xaxis_transform(), va='top')
    ax.text(40, .93, f'target = 1\n{n_pos:,} rows', color=ORANGE, fontsize=13, fontweight='semibold',
            transform=ax.get_xaxis_transform(), va='top', ha='right')
    clean(ax, 'col1 value as supplied (log scale)', 'Rows')
    ax.set_title('One column, both classes', fontsize=14)
    ax2 = f.add_subplot(gs[1])
    xs = np.arange(1, 11)
    for lab, c, off in ((-1, BLUE, -.12), (1, ORANGE, .12)):
        sub_ = d.loc[y == lab, SEN_F]
        q1, med, q3 = sub_.quantile(.25), sub_.median(), sub_.quantile(.75)
        ax2.vlines(xs + off, q1, q3, color=c, lw=3, alpha=.55)
        ax2.scatter(xs + off, med, color=c, s=40, zorder=3, edgecolors='white', linewidths=.8)
    ax2.set_yscale('log')
    ax2.set_yticks([1, 2, 5, 10, 20, 50, 80], ['1', '2', '5', '10', '20', '50', '80'])
    ax2.set_xticks(xs, [f'col{i}' for i in xs], rotation=45, ha='right', fontsize=12)
    clean(ax2, None, 'Median and IQR (log scale)')
    ax2.set_title('All ten columns, both classes', fontsize=14)
    ax2.grid(axis='x', visible=False)
    save('nn3_sensor_class_dist', f, 'Computed from Sensor_train.csv (3,580 rows): histogram of col1 by class, '
         'and per-column medians with interquartile ranges; column index is a name, not a verified time axis.',
         kind='data')


@fig
def sensor_corr():
    d = sensor()
    c = d[SEN_F].corr().to_numpy()
    f, axs = half(width=6.3, height=4.9)
    ax = axs[0]
    im = ax.imshow(c, cmap=CMAP_SEQ, vmin=.9995, vmax=1.0)
    ax.set_xticks(range(10), [f'col{i}' for i in range(1, 11)], rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(10), [f'col{i}' for i in range(1, 11)], fontsize=12)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = f.colorbar(im, ax=ax, shrink=.85, ticks=[.9995, .99975, 1.0])
    cb.ax.set_yticklabels(['0.99950', '0.99975', '1.00000'], fontsize=12)
    cb.set_label('Pearson correlation', fontsize=13)
    cb.outline.set_visible(False)
    save('nn3_sensor_corr', f, 'Computed from Sensor_train.csv: Pearson correlations between col1-col10 over '
         f'3,580 rows; the minimum off-diagonal value is {c[np.triu_indices(10, 1)].min():.4f}.', kind='data')


@lru_cache(maxsize=None)
def sensor_split_scores():
    d = _sensor()
    X = d[SEN_F]
    y = (d.target == 1).astype(int).to_numpy()
    order = np.argsort(d.uuid.to_numpy())
    block = np.empty(len(d), int)
    block[order] = np.arange(len(d)) * 5 // len(d)
    models = {'Logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)),
              'k-nearest neighbours': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
              'Gradient boosting': lambda: HistGradientBoostingClassifier(random_state=0)}
    out = {}
    for name, mk in models.items():
        pr = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               method='predict_proba')[:, 1]
        pb = cross_val_predict(mk(), X, y, cv=GroupKFold(5), groups=block, method='predict_proba')[:, 1]
        out[name] = (f1_score(y, pr >= .5, average='macro'), f1_score(y, pb >= .5, average='macro'))
    return out


@fig
def sensor_uuid():
    d = sensor().sort_values('uuid').reset_index(drop=True)
    m = d[SEN_F].mean(axis=1)
    y = d.target.to_numpy()
    f = plt.figure(figsize=(11.8, 4.4), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = f.add_subplot(gs[0])
    for lab, c in ((-1, BLUE), (1, ORANGE)):
        k = y == lab
        ax.scatter(np.flatnonzero(k), m[k], s=5, color=c, alpha=.8, linewidths=0)
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 5, 10, 20, 50, 80], ['1', '2', '5', '10', '20', '50', '80'])
    clean(ax, 'Rank of uuid (sorted)', 'Row mean of col1–col10 (log)')
    changes = int((y[1:] != y[:-1]).sum())
    ax.set_title(f'Sorted by uuid: the label changes {changes} times in {len(y) - 1:,} steps', fontsize=13.5)
    ax.text(.01, .03, 'target = −1', color=BLUE, fontsize=12.5, fontweight='semibold', transform=ax.transAxes)
    ax.text(.01, .93, 'target = 1', color=ORANGE, fontsize=12.5, fontweight='semibold', transform=ax.transAxes)
    ax2 = f.add_subplot(gs[1])
    res = sensor_split_scores()
    names = list(res)[::-1]
    for i, name in enumerate(names):
        r, b = res[name]
        ax2.plot([b, r], [i, i], color=LIGHT, lw=5, solid_capstyle='round', zorder=1)
        ax2.scatter([r], [i], s=120, color=ORANGE, zorder=3, edgecolors='white')
        ax2.scatter([b], [i], s=120, color=TEAL, zorder=3, edgecolors='white')
        ax2.annotate(f'{r:.3f}', (r, i), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=12,
                     color=ORANGE, fontweight='semibold')
        ax2.annotate(f'{b:.3f}', (b, i), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=12,
                     color=TEAL, fontweight='semibold')
    ax2.set_yticks(range(len(names)), [n.replace('k-nearest neighbours', 'k-nearest\nneighbours')
                                       .replace('Logistic regression', 'Logistic\nregression')
                                       .replace('Gradient boosting', 'Gradient\nboosting') for n in names],
                   fontsize=12.5)
    ax2.tick_params(axis='y', length=0)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlim(.88, 1.0)
    ax2.set_ylim(-.9, len(names) - .4)
    clean(ax2, 'Macro-F1, pooled out-of-fold')
    ax2.grid(axis='y', visible=False)
    for xx, txt, col in ((.884, 'stratified random rows', ORANGE), (.95, 'uuid blocks', TEAL)):
        ax2.scatter([xx], [-.62], s=80, color=col, edgecolors='white')
        ax2.text(xx + .0028, -.62, txt, color=col, fontsize=12, fontweight='semibold', va='center')
    ax2.set_title('Five-fold scores under two split designs', fontsize=13.5)
    save('nn3_sensor_uuid', f, 'Computed from Sensor_train.csv: row means in uuid order; macro-F1 from '
         'StratifiedKFold(5, shuffle, seed 0) versus GroupKFold(5) on five contiguous uuid blocks, scaling '
         'fitted inside folds, threshold 0.5.', kind='data')


# ------------------------------------------------------------ modelling playbook
REF_TRAIN_ROWS = {'ESP': 2295, 'PE': 2337, 'Sensor': 2864}   # training portions of the reference splits


@fig
def param_count():
    widths = np.array([4, 8, 16, 32, 64, 128, 256])
    one = 6 * widths + widths + widths + 1              # 6 -> w -> 1
    two = 6 * widths + widths + widths ** 2 + widths + widths + 1   # 6 -> w -> w -> 1
    f, axs = half()
    ax = axs[0]
    lo, hi = min(REF_TRAIN_ROWS.values()), max(REF_TRAIN_ROWS.values())
    ax.axhspan(lo, hi, color=GOLD, alpha=.22, lw=0)
    ax.text(4.1, (lo * hi) ** .5, f'training rows {lo:,}–{hi:,}', fontsize=12, color=shade(GOLD, .2),
            va='center', fontweight='semibold')
    ax.plot(widths, one, color=BLUE, marker='o', ms=6)
    ax.plot(widths, two, color=PURPLE, marker='s', ms=6)
    endlabel(ax, widths[-1], one[-1], 'one hidden layer', BLUE, dy=-2, fs=12.5)
    endlabel(ax, widths[-1], two[-1], 'two hidden\nlayers', PURPLE, fs=12.5)
    ax.scatter([32], [257], s=190, facecolors='none', edgecolors=ORANGE, linewidths=2.2, zorder=4)
    ax.annotate('reference network\n6 → 32 → 1: 257 parameters', (32, 257), xytext=(45, 55), fontsize=12,
                color=ORANGE, arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlim(3.3, 900)
    ax.set_ylim(20, 2e5)
    clean(ax, 'Hidden-layer width (units)', 'Trainable parameters (log scale)')
    save('nn3_param_count', f, 'Exact parameter counts for dense networks with 6 inputs and one output '
         '(weights plus biases); training-row counts from the reference splits in curriculum/capstone.py.',
         kind='computed')


@fig
def preprocess():
    d = _esp()
    X, y, g = esp_xyg()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=2, random_state=42).split(X, y, groups=g))
    Xt = X.iloc[tr]
    f, ax = canvas(11.8, 4.3)
    # raw columns
    box(ax, .15, .55, 3.05, 3.35, '', GREY, fill='white', lw=1.4)
    label(ax, 1.675, 3.62, 'Training portion, as supplied', fs=13.5, weight='semibold')
    for k, col in enumerate(ESP_F):
        lo, hi = Xt[col].min(), Xt[col].max()
        yy = 3.18 - k * .4
        label(ax, .35, yy, col, fs=12.5, ha='left', weight='medium')
        label(ax, 3.0, yy, f'{lo:,.0f} – {hi:,.0f}', fs=12.5, ha='right', color=MUTED)
    n_miss = int(Xt[ESP_F].isna().all(axis=1).sum())
    label(ax, 1.675, .82, f'{n_miss} rows: all six missing', fs=12, color=RED)
    # decision, imputer, scaler: title and a muted second line inside each box
    specs = [(3.75, 2.35, 'Decide what a\nmissing row means', 'drop and document,\nor impute and say so', GOLD),
             (6.55, 2.1, 'Median\nimputation', 'optional missing-\nindicator column', TEAL),
             (9.05, 2.55, 'Standard scaling', 'mean 0, SD 1\nper column', TEAL)]
    bxs = []
    for x, w, title, subtxt, col in specs:
        b = box(ax, x, 1.95, w, 1.7, '', col, fs=13)
        label(ax, x + w / 2, 3.13, title, fs=13.5, weight='semibold', linespacing=1.15)
        label(ax, x + w / 2, 2.4, subtxt, fs=12, color=MUTED, linespacing=1.15)
        bxs.append(b)
    b1, b2, b3 = bxs
    arrow(ax, (3.2, 2.8), b1.l, color=INK)
    arrow(ax, b1.r, b2.l, color=INK)
    arrow(ax, b2.r, b3.l, color=INK)
    bracket(ax, 6.55, 11.6, 3.95, 'fit on the training portion only; applied unchanged to validation and test',
            color=TEAL, below=False, fs=12.5)
    # outcome row
    box(ax, 3.75, .45, 7.85, .9, '', BLUE, fill='tint', lw=1.2)
    prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(Xt)
    zv = prep.transform(X.iloc[va])
    j = int(np.argmax(np.abs(zv).max(axis=0)))
    zext = zv[:, j][np.argmax(np.abs(zv[:, j]))]
    label(ax, 7.675, 1.03, 'Network input: training columns centred at 0 with SD 1', fs=13.5, weight='semibold')
    label(ax, 7.675, .7, f'held-out entities can still fall outside the training range: {ESP_F[j]} reaches '
          f'z = {zext:.1f}'.replace('= -', '= −'), fs=12, color=MUTED)
    arrow(ax, (10.325, 1.95), (10.325, 1.37), color=INK)
    save('nn3_preprocess', f, 'Schematic of the preprocessing order; value ranges computed from the training '
         'portion (five entities) of the reference ESP split in esp_train.csv.', kind='schematic')


@lru_cache(maxsize=None)
def esp_logo(model='lr'):
    X, y, g = esp_xyg()
    if model == 'lr':
        m = lr_pipe()
    elif model == 'lr_bal':
        m = lr_pipe(class_weight='balanced')
    elif model == 'mlp':
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=1000, random_state=0))
    else:
        raise ValueError(model)
    return cross_val_predict(m, X, y, cv=LeaveOneGroupOut(), groups=g, method='predict_proba')[:, 1]


@fig
def class_weights():
    X, y, g = esp_xyg()
    f, axs = half()
    ax = axs[0]
    for key, c, name in (('lr', BLUE, 'unweighted'), ('lr_bal', PURPLE, 'balanced class weights')):
        p = esp_logo(key)
        pr, rc, th = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=c, lw=2.2)
        k = p >= .5
        r5, p5 = recall_score(y, k), precision_score(y, k)
        ax.scatter([r5], [p5], s=110, color=c, zorder=4, edgecolors='white', linewidths=1.2)
        ap = average_precision_score(y, p)
        ty = .66 if key == 'lr' else .5
        ax.annotate(f'{name}: AP {ap:.3f}\nthreshold 0.5: recall {r5:.2f}, precision {p5:.2f}', (r5, p5),
                    xytext=(.03, ty), ha='left', va='center', fontsize=12, color=c,
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.1, shrinkB=6,
                                    connectionstyle='arc3,rad=-.15' if key == 'lr' else 'arc3,rad=.2'))
    ax.axhline(y.mean(), color=GREY, ls='--', lw=1.3)
    ax.text(.02, y.mean() - .025, f'no-skill precision = prevalence {y.mean():.2f}', fontsize=12, color=GREY,
            va='top')
    clean(ax, 'Recall (gas-lock rows found)', 'Precision (alerts that are gas lock)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(.3, 1.02)
    save('nn3_class_weights', f, 'Computed from esp_train.csv: logistic regression with and without '
         "class_weight='balanced', leave-one-entity-out predictions pooled over seven entities.", kind='data')


def _capstone():
    from curriculum import capstone
    return capstone


def train_reference(project, width=32, lr=1e-3, seed=42, epochs=40, patience=8, data=None):
    """Re-run the training loop of curriculum/capstone.py (same model, loss, optimizer, batching,
    early stopping) without writing artifacts. Returns (model, history, best_epoch, data)."""
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    cap = _capstone()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    data = data or cap.prepare(project, 42)
    task = data['spec']['task']
    model = cap.build_model(data['Xt'].shape[1], width)
    crit = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(data['Xt']), torch.from_numpy(data['yt'])),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, stale, best_epoch, best_state, hist = float('inf'), 0, 0, None, []
    for ep in range(epochs):
        model.train()
        tot = 0.
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.inference_mode():
            vl = crit(model(torch.from_numpy(data['Xv'])).squeeze(-1), torch.from_numpy(data['yv'])).item()
        hist.append((ep + 1, tot / len(data['Xt']), vl))
        if vl < best - 1e-7:
            best, best_state, best_epoch, stale = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, hist, best_epoch, data


@fig
def early_stop():
    _, hist, best_epoch, _ = train_reference('esp')
    ep = np.array([h[0] for h in hist])
    trl = np.array([h[1] for h in hist])
    val = np.array([h[2] for h in hist])
    f, axs = half()
    ax = axs[0]
    ax.axvspan(best_epoch + .5, ep[-1] + .5, color=PANEL, lw=0)
    ax.text((best_epoch + ep[-1]) / 2 + .5, .7, f'patience: {ep[-1] - best_epoch} epochs\nwithout improvement',
            ha='center', fontsize=12, color=MUTED)
    ax.plot(ep, trl, color=BLUE, marker='o', ms=4.5)
    ax.plot(ep, val, color=TEAL, marker='o', ms=4.5)
    endlabel(ax, ep[-1], trl[-1], 'training', BLUE, fs=12.5)
    endlabel(ax, ep[-1], val[-1], 'validation', TEAL, fs=12.5)
    b = best_epoch - 1
    ax.scatter([best_epoch], [val[b]], s=170, facecolors='none', edgecolors=ORANGE, linewidths=2.2, zorder=4)
    ax.annotate(f'checkpoint kept: epoch {best_epoch}\nvalidation loss {val[b]:.3f}', (best_epoch, val[b]),
                xytext=(1.4, .2), fontsize=12, color=ORANGE,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    clean(ax, 'Epoch', 'Binary cross-entropy')
    ax.set_xlim(.5, ep[-1] + 3.3)
    ax.set_ylim(.15, .8)
    ax.set_xticks(range(1, ep[-1] + 1, 2))
    save('nn3_early_stop', f, 'Computed: the reference workflow in curriculum/capstone.py on the ESP project '
         '(width 32, AdamW lr 0.001, seed 42, patience 8), validation = held-out entities esp_30 and esp_31.',
         kind='computed')


@fig
def tree_bias():
    rng = np.random.default_rng(0)

    def truth(x):
        return np.where(x < 3, 0., np.where(x < 5, 2., np.where(x < 7.5, .5, 2.5)))
    x = rng.uniform(0, 10, 300)
    y = truth(x) + rng.normal(0, .3, 300)
    xt = np.linspace(0, 10, 1000)
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                                                       random_state=0)).fit(x[:, None], y)
    gbt = HistGradientBoostingRegressor(random_state=0).fit(x[:, None], y)
    pm, pg = mlp.predict(xt[:, None]), gbt.predict(xt[:, None])
    rm = np.sqrt(np.mean((pm - truth(xt)) ** 2))
    rg = np.sqrt(np.mean((pg - truth(xt)) ** 2))
    f, axs = half()
    ax = axs[0]
    ax.scatter(x, y, s=12, color=LIGHT, zorder=1, linewidths=0)
    ax.plot(xt, truth(xt), color=INK, lw=1.6, ls='--', zorder=2)
    ax.plot(xt, pg, color=TEAL, lw=2.2, zorder=3)
    ax.plot(xt, pm, color=ORANGE, lw=2.4, zorder=4)
    ax.text(.3, 3.35, 'true step function (dashed)', fontsize=12, color=INK)
    ax.text(.3, 3.0, f'gradient-boosted trees: RMSE {rg:.2f}', fontsize=12.5, color=TEAL, fontweight='semibold')
    ax.text(.3, 2.65, f'MLP 64-64: RMSE {rm:.2f}', fontsize=12.5, color=ORANGE, fontweight='semibold')
    ax.annotate('smooth ramp where\nthe target jumps', (5.0, 1.25), xytext=(6.2, -.85), fontsize=12, color=ORANGE,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))
    clean(ax, 'Input x (synthetic)', 'Target y (synthetic)')
    ax.set_ylim(-1.2, 3.7)
    save('nn3_tree_bias', f, 'Synthetic example, seed 0: 300 noisy samples of a step function; scikit-learn '
         'HistGradientBoostingRegressor and MLPRegressor (64, 64) with default settings; RMSE against the true '
         'function on a fine grid.', kind='computed')


@lru_cache(maxsize=None)
def pe_seed_runs(widths=(8, 32, 128), seeds=(0, 1, 2, 3, 4)):
    cap = _capstone()
    data = cap.prepare('log', 42)
    yv = data['y'][data['val_idx']]
    Xv = data['X'].iloc[data['val_idx']]
    out = {}
    for w in widths + (32,):
        for sd in (seeds if w in widths else ()):
            m, hist, be, _ = train_reference('log', width=w, seed=sd, data=data)
            p = cap.predict_neural({'prep': data['prep'], 'target_scaler': data['target_scaler'], 'model': m,
                                    'task': 'regression'}, Xv)
            out[(w, sd)] = (mean_absolute_error(yv, p), be)
    m, hist, be, _ = train_reference('log', width=32, seed=42, data=data)
    p = cap.predict_neural({'prep': data['prep'], 'target_scaler': data['target_scaler'], 'model': m,
                            'task': 'regression'}, Xv)
    out[(32, 42)] = (mean_absolute_error(yv, p), be)
    Xtr, ytr = data['X'].iloc[data['train_idx']], data['y'][data['train_idx']]
    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1)).fit(Xtr, ytr)
    base = {'Ridge baseline': mean_absolute_error(yv, ridge.predict(Xv)),
            'training-mean baseline': mean_absolute_error(yv, np.full(len(yv), ytr.mean()))}
    return out, base


@fig
def seed_spread():
    runs_, base = pe_seed_runs()
    widths = [8, 32, 128]
    f, axs = half()
    ax = axs[0]
    for name, c, ls in (('training-mean baseline', GREY, '--'), ('Ridge baseline', BLUE, '-')):
        ax.axhline(base[name], color=c, ls=ls, lw=1.6)
        ax.text(2.45, base[name] + .004, f'{name} {base[name]:.3f}', color=c, fontsize=12, ha='right',
                va='bottom')
    for i, w in enumerate(widths):
        vals = [runs_[(w, s)][0] for s in range(5)]
        xs = i + np.linspace(-.16, .16, 5)
        ax.scatter(xs, vals, s=70, color=ORANGE, zorder=3, edgecolors='white', linewidths=.8)
        ax.plot([i - .26, i + .26], [np.median(vals)] * 2, color=ORANGE, lw=2.4)
    r42 = runs_[(32, 42)][0]
    ax.scatter([1.3], [r42], s=90, facecolors='none', edgecolors=INK, linewidths=1.8, zorder=4)
    ax.annotate(f'reference run\n(seed 42): {r42:.3f}', (1.3, r42), xytext=(1.5, .455), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    ax.set_xticks(range(3), [f'width {w}' for w in widths])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(.36, .57)
    clean(ax, None, 'Validation MAE (PE units as supplied)')
    ax.grid(axis='x', visible=False)
    save('nn3_seed_spread', f, 'Computed: the reference training loop from curriculum/capstone.py on the PE '
         'project, seeds 0-4 per width (bars = medians), same split and preprocessing; baselines fitted on the '
         'same training portion.', kind='computed')


# ------------------------------------------------------------ error analysis
@lru_cache(maxsize=None)
def pe_ridge_validation():
    d = _logs()
    seg, _ = log_segments(d)
    tr, va = np.arange(LOG_CUT), np.arange(LOG_CUT, len(d))
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1))
    m.fit(d[LOG_F].iloc[tr], d.PE.iloc[tr])
    p = m.predict(d[LOG_F].iloc[va])
    out = pd.DataFrame({'row': va, 'Depth': d.Depth.iloc[va].to_numpy(), 'PE': d.PE.iloc[va].to_numpy(),
                        'pred': p, 'seg': seg[va]})
    out['res'] = out.PE - out.pred
    out['mean_pred'] = d.PE.iloc[tr].mean()
    return out


@fig
def residuals():
    v = pe_ridge_validation()
    frag = v.seg == v.seg.min()        # rows 2337-2364 continue the short block that starts at row 2309
    f, axs = half(width=5.8, height=4.4)
    ax = axs[0]
    ax.axhline(0, color=INK, lw=1.2)
    ax.scatter(v.pred[~frag], v.res[~frag], s=16, color=GREY, alpha=.55, linewidths=0)
    ax.scatter(v.pred[frag], v.res[frag], s=30, color=RED, alpha=.9, linewidths=0)
    r0, r1 = int(v.row[frag].min()), int(v.row[frag].max())
    ax.annotate(f'{int(frag.sum())} rows, source rows\n{r0:,}–{r1:,}:\nmean residual '
                f'{v.res[frag].mean():+.2f}'.replace('-', '−'),
                (v.pred[frag].mean(), v.res[frag].mean()), xytext=(2.3, 2.9), fontsize=12, color=RED,
                ha='left', va='center', arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.1))
    ax.text(2.35, -1.95, f'other {int((~frag).sum())} rows: mean residual '
            f'{v.res[~frag].mean():+.2f}'.replace('-', '−'), fontsize=12, color=GREY)
    clean(ax, 'Predicted PE (Ridge)', 'Residual = observed − predicted')
    ax.set_xlim(2.2, 5.6)
    ax.set_ylim(-2.3, 3.7)
    save('nn3_residuals', f, 'Computed from Log curve - Train.csv: Ridge baseline fitted on source rows 0-2336, '
         'residuals on the validation block (rows 2337-2921).', kind='data')


@fig
def depth_errors():
    v = pe_ridge_validation().copy()
    edges = np.arange(2825, 3126, 25)
    v['blk'] = pd.cut(v.Depth, edges)
    v['abs_r'] = v.res.abs()
    v['abs_m'] = (v.PE - v.mean_pred).abs()
    g = v.groupby('blk', observed=True).agg(n=('abs_r', 'size'), ridge=('abs_r', 'mean'), mean=('abs_m', 'mean'))
    g = g[g.n >= 5]
    f, axs = half(width=5.8, height=4.4)
    ax = axs[0]
    y = np.arange(len(g))
    ax.barh(y, g.ridge, color=BLUE, height=.62, alpha=.85)
    ax.scatter(g['mean'], y, color=GREY, s=36, zorder=3, marker='D')
    ax.set_yticks(y, [f'{int(iv.left)}–{int(iv.right)}  (n={n})' for iv, n in zip(g.index, g.n)], fontsize=12)
    ax.invert_yaxis()
    clean(ax, 'Mean absolute error (PE units)', 'Depth block as supplied (n = rows)')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 1.6)
    ax.text(1.58, -.35, 'bars: Ridge baseline', ha='right', va='center', fontsize=12, color=BLUE)
    ax.scatter([1.06], [.45], color=GREY, s=36, marker='D')
    ax.text(1.58, .45, 'training mean', ha='right', va='center', fontsize=12, color=GREY)
    save('nn3_depth_errors', f, 'Computed from Log curve - Train.csv: absolute errors of the Ridge baseline and '
         'of the training-mean prediction on the validation block, grouped in 25-unit Depth blocks with at least '
         'five rows.', kind='data')


@fig
def entity_errors():
    X, y, g = esp_xyg()
    p = esp_logo('lr') >= .5
    ents = sorted(set(g))
    fp = np.array([int(((g == e) & (y == 0) & p).sum()) for e in ents])
    fn = np.array([int(((g == e) & (y == 1) & ~p).sum()) for e in ents])
    neg = np.array([int(((g == e) & (y == 0)).sum()) for e in ents])
    pos = np.array([int(((g == e) & (y == 1)).sum()) for e in ents])
    f, axs = half()
    ax = axs[0]
    yy = np.arange(len(ents))
    ax.barh(yy, -fp, color=BLUE, height=.62, alpha=.85)
    ax.barh(yy, fn, color=ORANGE, height=.62, alpha=.9)
    for i in range(len(ents)):
        if fp[i] > 150:
            ax.text(-fp[i] + 6, i, f'{fp[i]}/{neg[i]}', ha='left', va='center', fontsize=11.5, color='white',
                    fontweight='semibold')
        elif fp[i]:
            ax.text(-fp[i] - 4, i, f'{fp[i]}/{neg[i]}', ha='right', va='center', fontsize=11.5, color=BLUE)
        if fn[i]:
            ax.text(fn[i] + 4, i, f'{fn[i]}/{pos[i]}', ha='left', va='center', fontsize=11.5, color=ORANGE)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(yy, ents, fontsize=12.5)
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    lim = 300
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-300, -200, -100, 0, 100, 200, 300], ['300', '200', '100', '0', '100', '200', '300'])
    ax.text(-lim + 8, -.95, 'false alarms\n(true GAS_LOCK = 0)', color=BLUE, fontsize=12.5, fontweight='semibold',
            va='bottom')
    ax.text(lim - 8, -.95, 'missed gas lock\n(true GAS_LOCK = 1)', color=ORANGE, fontsize=12.5,
            fontweight='semibold', va='bottom', ha='right')
    ax.set_ylim(len(ents) - .5, -1.1)
    clean(ax, 'Rows per held-out entity (threshold 0.5)')
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    save('nn3_entity_errors', f, 'Computed from esp_train.csv: logistic-regression baseline, leave-one-entity-out; '
         'each entity scored by a model fitted on the other six. Labels give errors/rows of that class.',
         kind='data')


@fig
def confusion():
    X, y, g = esp_xyg()
    p = esp_logo('lr') >= .5
    cm = confusion_matrix(y, p, labels=[0, 1])
    f, ax = canvas(6.3, 4.9)
    x0, y0, cw, ch = 1.9, .75, 2.0, 1.55
    names = [['True negative', 'False alarm'], ['Missed gas lock', 'True positive']]
    for i in range(2):
        for j in range(2):
            ok = i == j
            col = TEAL if ok else (BLUE if i == 0 else ORANGE)
            yy = y0 + (1 - i) * ch
            ax.add_patch(FancyBboxPatch((x0 + j * cw + .04, yy + .04), cw - .08, ch - .08,
                                        boxstyle='round,pad=0,rounding_size=0.06', fc=tint(col, .8 if ok else .72),
                                        ec=col, lw=1.5))
            rate = cm[i, j] / cm[i].sum()
            label(ax, x0 + (j + .5) * cw, yy + ch * .64, f'{cm[i, j]:,}', fs=24, weight='semibold', color=INK)
            label(ax, x0 + (j + .5) * cw, yy + ch * .34, f'{names[i][j]}\n{rate:.0%} of row', fs=12, color=MUTED,
                  linespacing=1.1)
    label(ax, x0 + cw, y0 + 2 * ch + .5, 'Predicted', fs=13.5, weight='semibold')
    for j, t in enumerate(['GAS_LOCK = 0', 'GAS_LOCK = 1']):
        label(ax, x0 + (j + .5) * cw, y0 + 2 * ch + .18, t, fs=12.5)
    label(ax, .35, y0 + ch, 'Actual', fs=13.5, weight='semibold', rotation=90)
    for i, t in enumerate(['GAS_LOCK = 0', 'GAS_LOCK = 1']):
        label(ax, x0 - .12, y0 + (1.5 - i) * ch, f'{t}\n{cm[i].sum():,} rows', fs=12.5, ha='right', linespacing=1.15)
    label(ax, x0 + cw, .3, f'pooled over 7 held-out entities, {cm.sum():,} rows', fs=12, color=MUTED)
    save('nn3_confusion', f, 'Computed from esp_train.csv: logistic-regression baseline, leave-one-entity-out '
         'predictions pooled over the seven entities, threshold 0.5.', kind='data')


@lru_cache(maxsize=None)
def sensor_reference_probs():
    d = _sensor()
    y = (d.target == 1).astype(int).to_numpy()
    tr, va = train_test_split(np.arange(len(d)), test_size=.2, stratify=y, random_state=42)
    m = lr_pipe().fit(d[SEN_F].iloc[tr], y[tr])
    return y[va], m.predict_proba(d[SEN_F].iloc[va])[:, 1]


@fig
def calibration():
    X, y, g = esp_xyg()
    panels = [('ESP, held-out entities (leave one out)', y, esp_logo('lr')),
              ('Sensor, stratified random validation rows', *sensor_reference_probs())]
    f = plt.figure(figsize=(11.8, 4.5), layout='constrained')
    gs = f.add_gridspec(2, 2, height_ratios=[3.2, 1])
    for k, (title, yy, pp) in enumerate(panels):
        ax = f.add_subplot(gs[0, k])
        axh = f.add_subplot(gs[1, k], sharex=ax)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pp, bins) - 1, 0, 9)
        pts = {}
        for b in range(10):
            m = idx == b
            if m.sum() >= 10:
                pts[b] = (pp[m].mean(), yy[m].mean(), int(m.sum()))
        ax.plot([0, 1], [0, 1], color=GREY, ls='--', lw=1.3)
        for b in pts:
            if b + 1 in pts:
                ax.plot([pts[b][0], pts[b + 1][0]], [pts[b][1], pts[b + 1][1]], color=PURPLE, lw=2)
        ax.scatter([v[0] for v in pts.values()], [v[1] for v in pts.values()],
                   s=[18 + 1.1 * np.sqrt(v[2]) * 6 for v in pts.values()], color=PURPLE, zorder=3,
                   edgecolors='white', linewidths=1, clip_on=False)
        ax.text(.97, .05, 'dashed: perfect calibration\nmarker area grows with rows per bin', ha='right',
                fontsize=12, color=GREY, transform=ax.transAxes)
        ax.set_title(title, fontsize=13.5)
        clean(ax, None, 'Observed rate of class 1' if k == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(-.02, 1.02)
        plt.setp(ax.get_xticklabels(), visible=False)
        axh.hist(pp, bins=bins, color=PURPLE, alpha=.45, rwidth=.9)
        clean(axh, 'Predicted probability of class 1', 'Rows' if k == 0 else None)
        axh.yaxis.set_major_locator(plt.MaxNLocator(2))
    save('nn3_calibration', f, 'Computed: logistic-regression baselines. Left, esp_train.csv with leave-one-'
         'entity-out predictions (3,243 rows); right, Sensor_train.csv with the reference stratified 20% '
         'validation split (716 rows). Bins with fewer than 10 rows are omitted.', kind='data')


# ------------------------------------------------------------ communicating evidence
@lru_cache(maxsize=None)
def mlp_random_cv_accuracy():
    X, y, g = esp_xyg()
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=1000, random_state=0))
    p = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    return accuracy_score(y, p >= .5)


def verdict(ax, text, color):
    ax.text(0, 1.05, text, transform=ax.transAxes, fontsize=12.5, fontweight='semibold', color='white',
            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.5', fc=color, ec='none'), va='bottom')


@fig
def evidence():
    X, y, g = esp_xyg()
    acc = mlp_random_cv_accuracy()
    ents = sorted(set(g))
    pl, pm = esp_logo('lr'), esp_logo('mlp')
    ap_l = [average_precision_score(y[g == e], pl[g == e]) for e in ents]
    ap_m = [average_precision_score(y[g == e], pm[g == e]) for e in ents]
    prev = [y[g == e].mean() for e in ents]
    order = np.argsort(ap_l)[::-1]
    f = plt.figure(figsize=(11.8, 4.3), layout='constrained')
    gs = f.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=.08)
    ax = f.add_subplot(gs[0])
    ax.bar([0], [acc], width=.5, color=ORANGE, alpha=.9)
    ax.text(0, acc + .003, f'{acc:.0%}', ha='center', va='bottom', fontsize=22, fontweight='semibold', color=INK)
    ax.set_xticks([0], ['Our neural network'], fontsize=13)
    ax.set_ylim(.9, 1.0)
    ax.set_xlim(-.8, .8)
    clean(ax, None, 'Accuracy')
    ax.grid(axis='x', visible=False)
    verdict(ax, 'WEAK: one number, no reference', ORANGE)
    ax = f.add_subplot(gs[1])
    for r, i in enumerate(order):
        ax.plot([prev[i], prev[i]], [r - .34, r + .34], color=GREY, lw=2.5)
        ax.scatter([ap_l[i]], [r - .15], s=80, color=BLUE, zorder=3, edgecolors='white')
        ax.scatter([ap_m[i]], [r + .15], s=80, color=ORANGE, marker='D', zorder=3, edgecolors='white')
    ax.set_yticks(range(len(ents)), [ents[i] for i in order], fontsize=12.5)
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(len(ents) - .45, -1.35)
    clean(ax, 'Average precision on the held-out entity')
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    for xx, txt, col, mk in ((.03, 'logistic baseline', BLUE, 'o'), (.33, 'MLP', ORANGE, 'D')):
        ax.scatter([xx], [-.95], s=60, color=col, marker=mk, edgecolors='white')
        ax.text(xx + .02, -.95, txt, fontsize=12, color=col, va='center')
    ax.plot([.48, .48], [-1.12, -.78], color=GREY, lw=2.5)
    ax.text(.495, -.95, 'no-skill level (positive rate)', fontsize=12, color=GREY, va='center')
    verdict(ax, 'STRONG: named split, baseline, reference, spread', TEAL)
    save('nn3_evidence', f, 'Computed from esp_train.csv. Left: scikit-learn MLP (32 units), random 5-fold row '
         'split, pooled accuracy (a deliberately weak evidence figure). Right: leave-one-entity-out average '
         'precision per entity for logistic regression and the same MLP with identical preprocessing.',
         kind='data')


# ------------------------------------------------------------ day structure diagrams
@fig
def lifecycle():
    f, ax = canvas(11.8, 4.3)
    stages = [('Frame', 'target, row unit,\nquestion', GREY), ('Audit', 'schema, missing,\nduplicates', GREY),
              ('Split', 'groups, order,\nroles', GREY), ('Baseline', 'reference and\nlinear model', BLUE),
              ('Experiment', 'one change,\nseveral seeds', BLUE), ('Lock', 'configuration\nin writing', INK),
              ('Test once', '--final, join\nchecks', ORANGE), ('Analyse and\nreport', 'errors, limits,\nfive slides', PURPLE)]
    w, h, gap, x0, y0 = 1.21, .95, .22, .2, 1.95
    anchors = []
    for i, (name, subtxt, col) in enumerate(stages):
        x = x0 + i * (w + gap)
        fill = 'solid' if name == 'Lock' else 'tint'
        a = box(ax, x, y0, w, h, name, col, fill=fill, fs=13.5, weight='semibold')
        label(ax, x + w / 2, y0 - .42, subtxt, fs=12, color=MUTED, linespacing=1.15)
        anchors.append(a)
        if i:
            arrow(ax, anchors[i - 1].r, a.l, color=INK, lw=1.5, head=11)
    # milestone badges above the stage where each checkpoint happens
    for k, i in enumerate([2, 3, 5, 6]):
        a = anchors[i]
        cx, cy = a.x + a.w / 2, y0 + h + .38
        ax.add_patch(Circle((cx, cy), .21, fc=TEAL, ec='none', zorder=4))
        label(ax, cx, cy, f'M{k + 1}', fs=12, color='white', weight='semibold', z=5)
        ax.plot([cx, cx], [cy - .21, y0 + h], color=TEAL, lw=1.4, zorder=3)
    m1 = anchors[2]
    label(ax, m1.x + m1.w / 2 - .32, y0 + h + .38, 'Milestone checkpoints', fs=12.5, color=TEAL, ha='right',
          weight='semibold')
    # loops
    a3, a4 = anchors[3], anchors[4]
    arrow(ax, (a4.x + a4.w / 2, y0 - .95), (a3.x + a3.w / 2, y0 - .95), color=BLUE, ls='--', lw=1.4, rad=-.35,
          head=11)
    label(ax, (a3.x + a4.x + a4.w) / 2, y0 - 1.52, 'iterate on validation only', fs=12, color=BLUE)
    a7, a0 = anchors[7], anchors[0]
    xa, xb, yt = a7.x + a7.w / 2, a0.x + a0.w / 2, 3.82
    ax.plot([xa, xa, xb], [y0 + h, yt, yt], color=PURPLE, lw=1.4, ls='--', zorder=1)
    arrow(ax, (xb, yt), (xb, y0 + h + .02), color=PURPLE, lw=1.4, ls='--', head=11)
    label(ax, (xa + xb) / 2, yt + .2, 'a new question or a changed model needs new, untouched test data', fs=12,
          color=PURPLE)
    lk = anchors[5]
    ax.plot([lk.x + lk.w + gap / 2] * 2, [y0 - .75, y0 + h + .1], color=ORANGE, lw=2, ls=':')
    label(ax, lk.x + lk.w + gap / 2, y0 - .95, 'no changes after this line', fs=12, color=ORANGE)
    save('nn3_lifecycle', f, 'Schematic of the capstone project lifecycle and the four milestone checkpoints.',
         kind='schematic')


DAY = [  # (start, end, row, label) -- rows: 0 lecture, 1 sprint, 2 checkpoint, 3 presentation, 4 break
    ('09:00', '09:35', 0, 'Brief'), ('09:35', '09:55', 0, 'ESP'), ('09:55', '10:15', 0, 'PE'),
    ('10:15', '10:30', 4, ''), ('10:30', '10:45', 0, 'Sensor'), ('10:45', '11:20', 1, 'Sprint 1'),
    ('11:20', '11:30', 2, 'M1'), ('11:30', '11:50', 0, 'Playbook'), ('11:50', '12:30', 1, 'Sprint 2'),
    ('12:30', '12:40', 2, 'M2'), ('12:40', '13:25', 4, 'Lunch'), ('13:25', '13:40', 0, 'Experiments'),
    ('13:40', '14:20', 1, 'Sprint 3'), ('14:20', '14:30', 2, 'M3'), ('14:30', '14:45', 0, 'Errors'),
    ('14:45', '15:25', 1, 'Sprint 4'), ('15:25', '15:35', 2, 'M4'), ('15:35', '15:45', 4, ''),
    ('15:45', '16:00', 0, 'Evidence'), ('16:00', '16:15', 1, 'Sprint 5'), ('16:15', '16:50', 3, 'Talks'),
    ('16:50', '17:00', 0, 'Close')]


def _mins(t):
    h, m = t.split(':')
    return int(h) * 60 + int(m) - 9 * 60


@fig
def day_gantt():
    f, ax = canvas(11.8, 4.3)
    rows = [('Mini-lectures', BLUE), ('Project sprints', ORANGE), ('Checkpoints', TEAL),
            ('Presentations', PURPLE), ('Breaks', FAINT)]
    xl, xr, ytop, rh = 1.95, 11.55, 3.75, .6
    sc = (xr - xl) / 480

    def X(t):
        return xl + _mins(t) * sc
    for k in range(9):
        x = xl + k * 60 * sc
        ax.plot([x, x], [ytop - len(rows) * rh - .02, ytop - .1], color=RULE, lw=1, zorder=0)
        label(ax, x, ytop - len(rows) * rh - .2, f'{9 + k:02d}:00', fs=12, color=MUTED)
    totals = {}
    for s, e, r, txt in DAY:
        totals[r] = totals.get(r, 0) + _mins(e) - _mins(s)
    for r, (name, col) in enumerate(rows):
        yc = ytop - (r + .5) * rh
        label(ax, xl - .15, yc + .07, name, fs=13, ha='right', weight='medium')
        label(ax, xl - .15, yc - .17, f'{totals[r]} min', fs=11.5, ha='right', color=MUTED)
    for s, e, r, txt in DAY:
        x0, x1 = X(s), X(e)
        yc = ytop - (r + .5) * rh
        col = rows[r][1]
        ax.add_patch(FancyBboxPatch((x0 + .012, yc - rh * .36), x1 - x0 - .024, rh * .72,
                                    boxstyle='round,pad=0,rounding_size=0.05', fc=col,
                                    ec='none', alpha=.95 if r != 4 else .5))
        if txt:
            inside = x1 - x0 >= .55
            tc = INK if r == 4 else ('white' if inside else col)
            if inside:
                label(ax, (x0 + x1) / 2, yc, txt, fs=12, color=tc, weight='semibold')
            elif r in (1, 2):
                label(ax, (x0 + x1) / 2, yc - rh * .56, txt, fs=11.5, color=tc, weight='semibold')
            else:
                label(ax, (x0 + x1) / 2, yc + rh * .56, txt, fs=11.5, color=tc, weight='semibold')
    assert sum(totals.values()) == 480, totals
    save('nn3_day_gantt', f, f'Schematic day plan matching the agenda: {totals[0]} min of lectures including the '
         f'close, {totals[1]} min of project sprints, four 10-min checkpoints, {totals[3]} min of presentations, '
         f'{totals[4]} min of breaks.', kind='schematic')


@fig
def data_roles():
    f, ax = canvas(11.8, 4.3)
    # file bars
    label(ax, .2, 3.95, 'Supplied training file', fs=13, ha='left', weight='semibold')
    tr = box(ax, .2, 3.0, 4.95, .65, 'Training portion', BLUE, fill='solid', fs=13.5, weight='semibold')
    va = box(ax, 5.23, 3.0, 2.5, .65, 'Validation portion', TEAL, fill='solid', fs=13.5, weight='semibold')
    label(ax, 8.35, 3.95, 'Supplied test file + labels', fs=13, ha='left', weight='semibold')
    te = box(ax, 8.35, 3.0, 3.25, .65, 'Test (locked)', ORANGE, fill='tint', fs=13.5, weight='semibold', lw=1.8)
    ax.add_patch(FancyBboxPatch((8.35, 3.0), 3.25, .65, boxstyle='round,pad=0,rounding_size=0.08', fc='none',
                                ec=ORANGE, hatch='///', lw=.8, alpha=.35, zorder=2.5))
    ax.plot([8.05, 8.05], [.45, 3.9], color=ORANGE, lw=2, ls=':')
    label(ax, 8.05, .25, 'gate: configuration locked at milestone 3', fs=12, color=ORANGE)
    # role cards
    cards = [(.2, 4.95, BLUE, 'Fit', ['imputer medians, scaler mean and SD', 'network weights and the baselines',
                                     'never sees validation or test rows']),
             (5.23, 2.5, TEAL, 'Select', ['width, epoch, threshold', 'one controlled change', 'scores are selected']),
             (8.35, 3.25, ORANGE, 'Assess once', ['run the locked configuration', 'report; do not tune',
                                                  'a rerun after changes is exploratory'])]
    for x, w, col, head, lines in cards:
        box(ax, x, .6, w, 2.1, '', col, fill='white', lw=1.4)
        label(ax, x + .18, 2.36, head, fs=14, ha='left', weight='semibold', color=col)
        for k, t in enumerate(lines):
            label(ax, x + .18, 1.9 - k * .44, t, fs=13, ha='left', color=INK)
        arrow(ax, (x + w / 2, 3.0), (x + w / 2, 2.72), color=col, lw=1.6, head=11)
    save('nn3_data_roles', f, 'Schematic of the three roles for data in the capstone: fit on the training '
         'portion, select on the validation portion, assess once on the supplied test file.', kind='schematic')


# ------------------------------------------------------------ audit, exercise and communication diagrams
def status(ax, x, y, kind):
    if kind == 'ok':
        check(ax, x, y, s=.1, color=TEAL, lw=2.4)
    elif kind == 'warn':
        ax.add_patch(Circle((x, y), .11, fc=GOLD, ec='none', zorder=5))
        label(ax, x, y - .005, '!', fs=11, color='white', weight='bold', z=6)
    else:
        cross(ax, x, y, s=.085, color=ORANGE, lw=2.4)


@lru_cache(maxsize=None)
def audit_facts():
    e, lg, sn = _esp(), _logs(), _sensor()
    et = pd.read_csv(DATA / '01_esp_gas_lock' / 'esp_test.csv', usecols=['uuid', 'ENTITY'])
    lt = pd.read_csv(DATA / '02_log_curves' / 'Log curve - Test.csv')
    st = pd.read_csv(DATA / '03_sensor_signals' / 'Sensor_test.csv')
    seg, resets = log_segments(lg)
    miss_rows = int(e[ESP_F].isna().all(axis=1).sum())
    dup_et = int(e.duplicated(['ENTITY', 'TimeSlot']).sum())
    train_vecs = set(map(tuple, sn[SEN_F].to_numpy()))
    test_dup = sum(tuple(r) in train_vecs for r in st[SEN_F].to_numpy())
    n_cols_esp = len(pd.read_csv(DATA / '01_esp_gas_lock' / 'esp_train.csv', nrows=0).columns)
    assert et.uuid.is_unique and sn.uuid.is_unique and st.uuid.is_unique
    return {
        'shape': [(f'{len(e):,} rows × {n_cols_esp} columns', 'ok'), (f'{len(lg):,} rows × {lg.shape[1]} columns', 'ok'),
                  (f'{len(sn):,} rows × {sn.shape[1]} columns', 'ok')],
        'target': [(f'GAS_LOCK = 1 in {e.GAS_LOCK.mean():.1%} of rows', 'ok'),
                   (f'PE {lg.PE.min():.2f}–{lg.PE.max():.2f}, median {lg.PE.median():.2f}', 'ok'),
                   (f'labels −1/1 ({(sn.target == 1).mean():.1%} = 1): map to 0/1', 'warn')],
        'missing': [(f'{int(e[ESP_F].isna().sum().sum())} cells, all in {miss_rows} esp_39 rows', 'warn'),
                    ('none', 'ok'), ('none', 'ok')],
        'dups': [(f'{int(e.duplicated(subset=list(e.columns[:-1])).sum())} identical rows; {dup_et} repeat stamps', 'warn'),
                 (f'{int(lg.duplicated().sum())} rows; a 12-row block repeats', 'warn'),
                 (f'{int(sn.duplicated(SEN_F).sum()):,} rows repeat a feature vector', 'bad')],
        'keys': [('uuid unique in test; join on uuid', 'ok'), ('no key: labels align by position', 'warn'),
                 ('uuid unique; labels join on uuid', 'ok')],
        'groups': [(f'{e.ENTITY.nunique()} entities, hourly TimeSlot', 'ok'),
                   (f'no well ID; {len(resets)} depth resets', 'bad'),
                   ('no groups; uuid order has long runs', 'warn')],
        'units': [('units undocumented', 'warn'), ('units undocumented', 'warn'),
                  ('meaning and units undocumented', 'warn')],
        'test': [(f'{et.ENTITY.nunique()} unseen entities, same dates', 'ok'),
                 (f'{len(lt)} rows, Depth {lt.Depth.min():g}–{lt.Depth.max():g}', 'ok'),
                 (f'{test_dup} of {len(st)} repeat a training row', 'bad')],
    }


@fig
def audit_board():
    facts = audit_facts()
    checks = [('shape', 'Shape and schema'), ('target', 'Target and balance'), ('missing', 'Missing values'),
              ('dups', 'Duplicates'), ('keys', 'Keys and joins'), ('groups', 'Groups and order'),
              ('units', 'Units and meaning'), ('test', 'Test file inputs')]
    f, ax = canvas(11.8, 4.25)
    x0, cw, rh, ytop = 2.25, 3.18, .455, 3.85
    for j, name in enumerate(['ESP gas lock', 'PE log curve', 'Sensor signals']):
        box(ax, x0 + j * cw + .04, ytop, cw - .08, .36, name, INK, fill='solid', fs=13, weight='semibold')
    for i, (key, name) in enumerate(checks):
        y = ytop - (i + .5) * rh - .06
        if i % 2 == 0:
            ax.add_patch(Rectangle((.1, y - rh / 2), x0 + 3 * cw - .1, rh, fc=PANEL, ec='none', zorder=0))
        label(ax, .22, y, name, fs=12.5, ha='left', weight='semibold')
        for j, (txt, kind) in enumerate(facts[key]):
            xx = x0 + j * cw
            status(ax, xx + .22, y, kind)
            label(ax, xx + .42, y, txt, fs=12, ha='left')
    save('nn3_audit_board', f, 'Computed from the supplied training files and the test input files (shape, '
         'Depth, keys and feature vectors only); no test labels are read.', kind='data')


@fig
def positional():
    f, ax = canvas(6.3, 4.9)
    gr = [54, 55, 61, 68, 47, 52]
    order = list(np.argsort(gr, kind='stable'))
    cols = [(.25, 'Test inputs\n(file order)'), (2.35, 'Predictions after\nsort by GR'), (4.45, 'Label file\n(file order)')]
    for x, t in cols:
        label(ax, x + .8, 4.55, t, fs=12.5, weight='semibold', linespacing=1.1)
    ytop, rh = 3.85, .5
    for k in range(6):
        y = ytop - k * rh
        box(ax, .25, y - .19, 1.6, .38, f'row {k}   GR {gr[k]}', GREY, fill='tint', fs=12)
        box(ax, 2.35, y - .19, 1.6, .38, f'ŷ for row {order[k]}', ORANGE, fill='tint', fs=12)
        box(ax, 4.45, y - .19, 1.6, .38, f'y for row {k}', BLUE, fill='tint', fs=12)
        ok = order[k] == k
        ax.plot([3.95, 4.45], [y, y], color=TEAL if ok else RED, lw=1.6)
        (check if ok else cross)(ax, 4.2, y + .17, s=.07, color=TEAL if ok else RED, lw=2)
    label(ax, 3.15, .72, 'Row counts match (310 = 310), so the length check passes,', fs=12, color=INK)
    label(ax, 3.15, .42, 'yet every prediction is scored against another row’s label.', fs=12, color=RED)
    save('nn3_positional', f, 'Schematic with illustrative GR values: positional labels paired with predictions '
         'made after re-ordering the inputs.', kind='schematic')


@fig
def presentation():
    f, ax = canvas(11.8, 3.75)
    slides_ = [('Problem and contract', 'target, row unit, split;\nwhat the test can show'),
               ('Baseline vs candidate', 'same rows, same metric;\nreference listed first'),
               ('Evidence plot', 'one readable figure;\nevaluation set named'),
               ('One failure examined', 'a slice or an example;\nhypothesis and next test'),
               ('Conclusion and limits', 'recommendation, limits,\nwhat would change it')]
    w, h, gap, x0, y0 = 2.14, 1.95, .2, .15, 1.45
    Z = 4
    for i, (title, guide) in enumerate(slides_):
        x = x0 + i * (w + gap)
        box(ax, x, y0, w, h, '', GREY, fill='white', lw=1.3)
        ax.add_patch(FancyBboxPatch((x, y0 + h - .42), w, .42, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc=INK, ec='none', zorder=Z))
        ax.add_patch(Rectangle((x, y0 + h - .42), w, .2, fc=INK, ec='none', zorder=Z))
        label(ax, x + w / 2, y0 + h - .21, f'Minute {i + 1}', fs=12, color='white', weight='semibold', z=Z + 1)
        label(ax, x + w / 2, y0 + h - .7, title, fs=12.5, weight='semibold', z=Z + 1)
        cx, cy = x + w / 2, y0 + .6
        if i == 0:
            for r in range(3):
                ax.add_patch(Rectangle((cx - .7, cy + .16 - r * .22), 1.4, .16,
                                       fc=PANEL if r else tint(BLUE, .7), ec=RULE, lw=.8, zorder=Z))
        elif i == 1:
            for k, (hh, col) in enumerate([(.3, FAINT), (.55, BLUE), (.62, ORANGE)]):
                ax.add_patch(Rectangle((cx - .55 + k * .4, cy - .3), .28, hh, fc=col, ec='none', zorder=Z))
            ax.plot([cx - .7, cx + .7], [cy - .3, cy - .3], color=MUTED, lw=1, zorder=Z)
        elif i == 2:
            for k, (a, b) in enumerate([(.2, .75), (.3, .9), (.1, .45)]):
                yy = cy + .2 - k * .22
                ax.plot([cx - .7 + a * 1.4, cx - .7 + b * 1.4], [yy, yy], color=LIGHT, lw=3, zorder=Z)
                ax.scatter([cx - .7 + a * 1.4], [yy], s=22, color=TEAL, zorder=Z + 1)
                ax.scatter([cx - .7 + b * 1.4], [yy], s=22, color=ORANGE, zorder=Z + 1)
        elif i == 3:
            rng = np.random.default_rng(3)
            px, py = rng.uniform(-.65, .65, 14), rng.uniform(-.3, .2, 14)
            ax.scatter(cx + px, cy + py, s=10, color=FAINT, zorder=Z)
            ax.scatter([cx + .45], [cy + .28], s=16, color=RED, zorder=Z + 1)
            ax.add_patch(Circle((cx + .45, cy + .28), .12, fc='none', ec=RED, lw=1.5, zorder=Z + 1))
        else:
            for k in range(3):
                check(ax, cx - .6, cy + .2 - k * .24, s=.06, color=TEAL, lw=1.8)
                ax.plot([cx - .45, cx + .6 - k * .15], [cy + .2 - k * .24] * 2, color=LIGHT, lw=3, zorder=Z)
        label(ax, x + w / 2, y0 - .42, guide, fs=12, color=MUTED, linespacing=1.2)
        if i:
            arrow(ax, (x - gap + .02, y0 + h / 2), (x - .02, y0 + h / 2), color=MUTED, lw=1.3, head=9)
    bracket(ax, x0, x0 + 5 * w + 4 * gap, .42, 'then two minutes of questions: what can the evaluation establish, '
            'and what would change the conclusion?', color=PURPLE, below=True, fs=12.5)
    save('nn3_presentation', f, 'Schematic storyboard for the five-minute team presentation.', kind='schematic')


@fig
def model_card():
    X, y, g = esp_xyg()
    p = esp_logo('lr')
    aps = {e: average_precision_score(y[g == e], p[g == e]) for e in sorted(set(g))}
    lo_e, hi_e = min(aps, key=aps.get), max(aps, key=aps.get)
    f, ax = canvas(11.8, 4.4)
    ax.add_patch(FancyBboxPatch((.1, 3.72), 11.6, .55, boxstyle='round,pad=0,rounding_size=0.08', fc=INK, ec='none'))
    label(ax, .35, 3.995, 'Model card', fs=15, color='white', weight='semibold', ha='left')
    label(ax, 2.0, 3.995, 'ESP gas-lock logistic baseline · teaching example, not a field model', fs=13,
          color=MIST, ha='left')
    secs = [('Intended use', BLUE, ['label hourly rows as gas lock or not', 'for entities unseen in training']),
            ('Not intended for', ORANGE, ['early warning: no lead time was tested', 'inputs outside the training ranges']),
            ('Data and split', GREY, [f'esp_train.csv: {len(y):,} rows, {len(set(g))} entities',
                                      'leave one entity out; test entities locked']),
            ('Evaluation', TEAL, [f'average precision by entity {aps[lo_e]:.2f}–{aps[hi_e]:.2f}',
                                  f'weakest: {lo_e} and {sorted(aps, key=aps.get)[1]}']),
            ('Limitations', RED, ['units undocumented; 300 missing cells', 'labelled periods look highly regular']),
            ('Reproduce', PURPLE, ['LeaveOneGroupOut on ENTITY, 7 folds', 'baseline pipeline of curriculum/capstone.py'])]
    w, h = 3.75, 1.55
    for k, (head, col, lines) in enumerate(secs):
        r, c = divmod(k, 3)
        x, yb = .1 + c * (w + .175), 1.95 - r * (h + .2)
        box(ax, x, yb, w, h, '', col, fill='tint' if col != GREY else 'white', lw=1.3)
        label(ax, x + .18, yb + h - .32, head, fs=13.5, weight='semibold', ha='left', color=col if col != GREY else INK)
        for i, t in enumerate(lines):
            label(ax, x + .18, yb + h - .78 - i * .38, t, fs=12, ha='left',
                  family='IBM Plex Mono' if t.startswith('python3') else None)
    save('nn3_model_card', f, 'Template with values computed from esp_train.csv (logistic baseline, leave-one-'
         'entity-out average precision per entity).', kind='computed')


@fig
def repro():
    f, ax = canvas(11.8, 4.0)
    box(ax, .15, 1.9, 4.05, 1.75, '', INK, fill='solid')
    label(ax, .35, 3.36, 'fresh process', fs=12, color=SKY, ha='left', weight='semibold')
    for k, t in enumerate(['$ python3 curriculum/capstone.py log', '    --epochs 40 --width 32', '    --lr 0.001 --seed 42',
                           '    --out runs/pe_w32_s42']):
        label(ax, .35, 2.98 - k * .3, t, fs=12, color='white', ha='left', family='IBM Plex Mono')
    label(ax, 2.175, 1.5, 'one command, pinned\nenvironment, unchanged data', fs=12, color=MUTED, linespacing=1.2)
    arts = ['split.csv', 'learning_curve.csv', 'validation_predictions.csv', 'results.json',
            'preprocessing_and_baseline.joblib', 'neural_weights.pt']
    label(ax, 6.45, 3.75, 'Saved artifacts', fs=13.5, weight='semibold')
    for k, name in enumerate(arts):
        yy = 3.3 - k * .52
        box(ax, 4.6, yy - .2, 3.7, .4, '', BLUE, fill='tint', lw=1)
        label(ax, 4.75, yy, name, fs=11.5, ha='left', family='IBM Plex Mono')
    arrow(ax, (4.2, 2.8), (4.55, 2.8), color=INK)
    checks = [('reload gives identical\npredictions', 'ok'), ('seed, width, lr and epochs\nrecorded in results.json', 'ok'),
              ('split ids let others\nrescore the run', 'ok'), ('final: same flags plus\n--final, run once', 'warn')]
    label(ax, 10.1, 3.75, 'Checks', fs=13.5, weight='semibold')
    for k, (t, kind) in enumerate(checks):
        yy = 3.2 - k * .78
        status(ax, 8.85, yy, kind)
        label(ax, 9.1, yy, t, fs=12, ha='left', linespacing=1.15)
    save('nn3_repro', f, 'Schematic of a reproducible capstone run; command-line options and artifact names from '
         'curriculum/capstone.py.', kind='schematic')


@lru_cache(maxsize=None)
def pe_ladder():
    cap = _capstone()
    data = cap.prepare('log', 42)
    Xtr, ytr = data['X'].iloc[data['train_idx']], data['y'][data['train_idx']]
    Xv, yv = data['X'].iloc[data['val_idx']], data['y'][data['val_idx']]
    runs_, base = pe_seed_runs()
    gbt = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    return [('Reference: training mean', base['training-mean baseline'], GREY),
            ('Linear: Ridge', base['Ridge baseline'], BLUE),
            ('Tree ensemble: gradient boosting', mean_absolute_error(yv, gbt.predict(Xv)), TEAL),
            ('Small network: width 32, seed 42', runs_[(32, 42)][0], ORANGE),
            ('One change: width 128, median of 5 seeds', float(np.median([runs_[(128, s)][0] for s in range(5)])),
             shade(ORANGE, .25))]


@fig
def ladder():
    rungs = pe_ladder()
    f, axs = half()
    ax = axs[0]
    yy = np.arange(len(rungs))
    for i, (name, v, col) in enumerate(rungs):
        ax.barh(i, v, color=col, height=.58)
        ax.text(v - .01, i, f'{v:.3f}', va='center', ha='right', fontsize=12.5, color='white',
                fontweight='semibold')
        ax.text(.01, i - .43, f'{i + 1}  {name}', fontsize=12.5, color=INK, va='bottom')
    ref = rungs[0][1]
    ax.axvline(ref, color=GREY, ls='--', lw=1.4, zorder=.9)     # behind the bars and their value labels
    ax.text(ref + .006, -.72, 'reference', ha='left', fontsize=12, color=GREY)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, .66)
    ax.set_ylim(len(rungs) - .3, -.9)
    clean(ax, 'Validation MAE, PE units as supplied (lower is better)')
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    save('nn3_ladder', f, 'Computed on the PE reference split (source rows 0-2336 train, 2337-2921 validation): '
         'scikit-learn baselines and the reference training loop from curriculum/capstone.py.', kind='computed')


@lru_cache(maxsize=None)
def project_comparison():
    """Validation scores of four model families on each project's reference split."""
    cap = _capstone()
    out = {}
    for project in ('esp', 'log', 'sensor'):
        data = cap.prepare(project, 42)
        tr, va = data['train_idx'], data['val_idx']
        Xtr, ytr, Xv, yv = data['X'].iloc[tr], data['y'][tr], data['X'].iloc[va], data['y'][va]
        model, _, _, _ = train_reference(project, data=data)
        pn = cap.predict_neural({'prep': data['prep'], 'target_scaler': data['target_scaler'], 'model': model,
                                 'task': data['spec']['task']}, Xv)
        if project == 'log':
            lin = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1)).fit(Xtr, ytr)
            tree = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
            out[project] = {'ref': mean_absolute_error(yv, np.full(len(yv), ytr.mean())),
                            'lin': mean_absolute_error(yv, lin.predict(Xv)),
                            'tree': mean_absolute_error(yv, tree.predict(Xv)),
                            'net': mean_absolute_error(yv, pn)}
            continue
        lin = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                            LogisticRegression(max_iter=1500, random_state=42)).fit(Xtr, ytr)
        tree = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        pl, pt = lin.predict_proba(Xv)[:, 1], tree.predict_proba(Xv)[:, 1]
        if project == 'esp':
            out[project] = {'ref': float(yv.mean()), 'lin': average_precision_score(yv, pl),
                            'tree': average_precision_score(yv, pt), 'net': average_precision_score(yv, pn)}
        else:
            def f1_at_selected(p):
                return f1_score(yv, p >= cap.choose_threshold(yv, p, 'sensor'), average='macro')
            majority = int(ytr.mean() >= .5)
            out[project] = {'ref': f1_score(yv, np.full(len(yv), majority), average='macro', zero_division=0),
                            'lin': f1_at_selected(pl), 'tree': f1_at_selected(pt), 'net': f1_at_selected(pn)}
    return out


@fig
def three_projects():
    res = project_comparison()
    panels = [('esp', 'ESP: average precision, 2 held-out entities', 'higher is better', (0, 1.08)),
              ('log', 'PE: MAE on the last 20% of rows', 'lower is better', (0, .66)),
              ('sensor', 'Sensor: macro-F1, stratified 20%', 'higher is better', (0, 1.08))]
    names = [('ref', 'Reference', GREY), ('lin', 'Linear', BLUE), ('tree', 'Trees', TEAL), ('net', 'Network', ORANGE)]
    f, axs = plt.subplots(3, 1, figsize=(6.3, 4.9), layout='constrained')
    for ax, (key, title, better, lim) in zip(axs, panels):
        vals = [res[key][k] for k, _, _ in names]
        ax.barh(range(4), vals, color=[c for _, _, c in names], height=.68, alpha=.9)
        for i, v in enumerate(vals):
            ax.text(v - .012 * lim[1], i, f'{v:.3f}', ha='right', va='center', fontsize=11.5, color='white',
                    fontweight='semibold')
        ax.set_yticks(range(4), [n for _, n, _ in names], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlim(*lim)
        ax.tick_params(axis='y', length=0)
        ax.set_title(f'{title} ({better})', fontsize=12.5, loc='left', pad=4)
        ax.grid(axis='x', alpha=1)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    save('nn3_three_projects', f, 'Computed on the reference splits of curriculum/capstone.py (training files only): '
         'reference = prevalence, training mean or majority class; linear = logistic or Ridge; trees = '
         'HistGradientBoosting (defaults); network = reference training loop, width 32, seed 42. Sensor thresholds '
         'chosen on validation.', kind='computed')


@fig
def shared_pipeline():
    f, ax = canvas(6.3, 4.9)
    w, x0 = 5.5, .4
    data = box(ax, x0, 4.2, w, .5, 'Supplied training file', GREY, fs=13, weight='semibold')
    sp_t = box(ax, x0, 3.12, 3.55, .52, 'Training portion', BLUE, fill='solid', fs=12.5, weight='semibold')
    sp_v = box(ax, x0 + 3.65, 3.12, w - 3.65, .52, 'Validation', TEAL, fill='solid', fs=12.5, weight='semibold')
    arrow(ax, data.b, (data.b[0], 3.66), color=INK, lw=1.4, head=10)
    label(ax, data.b[0] + .12, 3.92, 'split once, by group or order', fs=11.5, color=MUTED, ha='left')
    prep = box(ax, x0, 2.12, w, .56, 'Imputer + scaler', TEAL, fs=13, weight='semibold')
    arrow(ax, (sp_t.c[0], 3.12), (sp_t.c[0], 2.7), color=BLUE, lw=1.6, head=10)
    label(ax, sp_t.c[0] + .1, 2.9, 'fit', fs=12, color=BLUE, ha='left', weight='semibold')
    arrow(ax, (sp_v.c[0], 3.12), (sp_v.c[0], 2.7), color=TEAL, lw=1.6, head=10, ls='--')
    label(ax, sp_v.c[0] + .1, 2.9, 'transform only', fs=12, color=TEAL, ha='left', weight='semibold')
    lin = box(ax, x0, 1.12, 2.62, .72, 'Baseline\nlogistic or Ridge', BLUE, fs=12.5)
    net = box(ax, x0 + 2.88, 1.12, 2.62, .72, 'Candidate\ndense net, one logit', ORANGE, fs=12.5)
    arrow(ax, (lin.c[0], 2.12), (lin.c[0], 1.86), color=INK, lw=1.4, head=10)
    arrow(ax, (net.c[0], 2.12), (net.c[0], 1.86), color=INK, lw=1.4, head=10)
    ev = box(ax, x0, .14, w, .72, 'Same validation rows, same metric\nand the same threshold rule', PURPLE, fs=12.5)
    arrow(ax, (lin.c[0], 1.12), (lin.c[0], .88), color=INK, lw=1.4, head=10)
    arrow(ax, (net.c[0], 1.12), (net.c[0], .88), color=INK, lw=1.4, head=10)
    save('nn3_shared_pipeline', f, 'Schematic of a matched comparison: one split, one fitted preprocessing, two '
         'models, one evaluation.', kind='schematic')
