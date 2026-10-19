"""Figures for module key 'nn1' (Neural networks, Day 1: foundations).

Figure names start with 'nn1_'. Colour meaning used throughout the deck:
data/inputs = grey, parameters (weights, biases) = BLUE, hidden units = TEAL,
outputs/predictions = ORANGE, gradients = PURPLE, class 0/1 = BLUE/ORANGE,
training/validation/test = BLUE/TEAL/ORANGE.
"""
from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Ellipse
from scipy.special import expit, erf

KEY = 'nn1'
fig = register(KEY)

GREY = '#56687A'          # data / inputs
DATA_ROOT = ROOT / 'neural_networks_course' / 'Day_1_Intro_to_NN' / 'exercises'


# ------------------------------------------------------------------ helpers
def _rect(ax, x, y, w, h, fc, ec='none', lw=1.2, r=.08, z=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def _sonar():
    import pandas as pd
    df = pd.read_csv(DATA_ROOT / '01_sklearn_pipeline' / 'sonar.all-data.csv', header=None)
    X = df.iloc[:, :60].to_numpy(float)
    y = (df[60].to_numpy() == 'M').astype(int)      # M (metal) = 1, R (rock) = 0
    return X, y


# ============================================================ Part 1: context
@fig
def course_route():
    f, ax = canvas(11.8, 4.3)
    # Card titles are the day titles; topics are drawn from each day's parts.
    days = [
        ('Neural network\nfoundations', ['Tensors and shapes', 'Neuron, perceptron', 'MLPs, activations',
                                         'Losses, gradients', 'Backpropagation'], 'Check a forward and\nbackward pass'),
        ('Training neural\nnetworks reliably', ['Learning curves', 'Initialization', 'Normalization',
                                                'Optimizers', 'Regularization'], 'Reproduce a training\nexperiment'),
        ('Energy modelling\ncapstone', ['Data audits', 'ESP gas lock', 'PE log curve',
                                       'Sensor signals', 'Error analysis'], 'Defend a model\ncomparison'),
        ('Convolutional\nneural networks', ['Convolution, pooling', 'Architectures', 'Transfer learning',
                                           'Segmentation', 'Seismic, borehole'], 'Check an image or\nsignal model'),
        ('Sequence models\nand attention', ['Windows, baselines', 'RNN, LSTM, TCN', 'Attention',
                                           'Transformers', 'Forecast intervals'], 'Evaluate a future\nforecast'),
    ]
    w, gap, x0, y0, h = 2.06, .3, .08, .12, 4.06
    for i, (title, topics, outcome) in enumerate(days):
        x = x0 + i * (w + gap)
        today = i == 0
        col = BLUE if today else '#5B7089'
        _rect(ax, x, y0, w, h, fc=tint(BLUE, .93) if today else PANEL, ec=BLUE if today else RULE,
              lw=2 if today else 1, r=.1)
        # header
        _rect(ax, x, y0 + h - 1.12, w, 1.12, fc=col if today else tint(INK, .9), r=.1, z=2.1)
        ax.add_patch(Rectangle((x, y0 + h - 1.12), w, .2, fc=col if today else tint(INK, .9), ec='none', zorder=2.1))
        tc = 'white' if today else INK
        label(ax, x + .16, y0 + h - .25, f'DAY {i + 1}' + ('  ·  TODAY' if today else ''), fs=12,
              color=SKY if today else MUTED, weight='semibold', ha='left')
        label(ax, x + .16, y0 + h - .72, title, fs=14.5, color=tc, weight='semibold', ha='left',
              linespacing=1.1)
        # topics
        ty = y0 + h - 1.45
        for t in topics:
            ax.add_patch(Rectangle((x + .15, ty - .045), .085, .085, fc=col, ec='none', zorder=3))
            label(ax, x + .32, ty, t, fs=12.5, ha='left', color=INK)
            ty -= .37
        # outcome
        ax.plot([x + .16, x + w - .16], [y0 + .88, y0 + .88], color=RULE, lw=1.2, zorder=3)
        label(ax, x + .16, y0 + .48, outcome, fs=12, ha='left', color=BLUE if today else MUTED,
              weight='medium', linespacing=1.15)
        if i:
            xa = x - gap
            arrow(ax, (xa + .06, y0 + h / 2 + .3), (x - .05, y0 + h / 2 + .3), color=FAINT, lw=1.6, head=12)
    save('nn1_course_route', f, "Schematic course map; card titles are the day titles and topics are drawn from each "
         "day's parts.", kind='schematic')


@fig
def ai_ml_dl():
    f, ax = canvas(6.3, 4.9)
    rings = [(.1, .1, 6.1, 4.7, INK, 'Artificial intelligence',
              'search · planning · rules · learning'),
             (.45, .35, 5.4, 3.55, BLUE, 'Machine learning',
              'linear models · trees · SVMs · clustering'),
             (.8, .6, 4.7, 2.2, TEAL, 'Deep learning',
              'many learned layers:\nMLPs · CNNs\nRNNs · transformers')]
    for x, y, w, h, col, name, ex in rings:
        _rect(ax, x, y, w, h, fc=tint(col, .9 if col != INK else .95), ec=col, lw=1.8, r=.25)
        label(ax, x + .22, y + h - .3, name, fs=15.5, weight='semibold', color=col if col != INK else INK,
              ha='left')
        label(ax, x + .22, y + h - .62, ex, fs=12.5, color=MUTED, ha='left', va='top', linespacing=1.3)
    # A layered network filling the right half of the deep-learning ring. Edges are dark and thick enough
    # to read from the back of a room; they sit above the ring's fill and below the nodes.
    network(ax, [3, 4, 4, 2], 3.2, 5.12, .88, 2.52, r=.115, colors=[GREY, TEAL, TEAL, ORANGE],
            edge_color='#5F7389', edge_lw=1.3)
    chip(ax, 1.9, .98, 'today starts here', TEAL, fs=12, solid=True)
    save('nn1_ai_ml_dl', f, 'Schematic nesting of fields; examples are illustrative, not exhaustive.',
         kind='schematic')


@fig
def history():
    f, ax = canvas(11.8, 4.3)
    y_axis = 2.2
    x_of = lambda yr: .35 + (yr - 1938) * (11.1 / 87)  # noqa: E731
    ax.plot([x_of(1938), x_of(2025)], [y_axis, y_axis], color=FAINT, lw=2, zorder=1)
    down = [x_of(yr) for yr in (1958, 1986, 2012)]      # connectors that run below the axis
    for yr in range(1940, 2030, 10):
        ax.plot([x_of(yr)] * 2, [y_axis - .06, y_axis + .06], color=FAINT, lw=1.2, zorder=1)
        xl = x_of(yr)
        for px in down:                                   # keep decade labels clear of those connectors
            if abs(px - xl) < .32:
                xl = px - .32 if px > xl else px + .32
        label(ax, xl, y_axis - .2, str(yr), fs=12, color=FAINT, z=1)
    # (year, year text, name, authors, side, box centre, covered today)
    events = [
        (1943, '1943', 'Threshold neuron', 'McCulloch & Pitts', 'up', 1.2, True),
        (1958, '1958', 'Perceptron', 'Rosenblatt', 'down', 2.95, True),
        (1969, '1969', 'Perceptron limits', 'Minsky & Papert', 'up', 4.45, True),
        (1986, '1986', 'Backpropagation', 'Rumelhart, Hinton, Williams', 'down', 6.3, True),
        (1998, '1998', 'LeNet-5 (CNN)', 'LeCun et al.', 'up', 7.8, False),
        (2012, '2012', 'AlexNet', 'Krizhevsky et al.', 'down', 9.6, False),
        (2017, '2017 and 2018+', 'Transformer, then\nlarge pretrained models', 'Vaswani et al.; BERT, GPT',
         'up', 10.55, False),
    ]
    bw = 2.42
    for yr, ytxt, name, who, side, cx, today in events:
        px = x_of(yr)
        col = BLUE if today else GREY
        two = '\n' in name
        bh = 1.42 if two else 1.12
        by = y_axis + .5 if side == 'up' else y_axis - .5 - bh
        cx = min(max(cx, bw / 2 + .06), 11.8 - bw / 2 - .06)
        _rect(ax, cx - bw / 2, by, bw, bh, fc=tint(BLUE, .9) if today else PANEL, ec=col, lw=1.4, r=.08, z=3)
        x_t = cx - bw / 2 + .15
        label(ax, x_t, by + bh - .24, ytxt, fs=13.5, weight='semibold', color=col, ha='left', z=4)
        label(ax, x_t, by + bh - .5, name, fs=13.5, weight='semibold', color=INK, ha='left', va='top', z=4,
              linespacing=1.1)
        label(ax, x_t, by + .22, who, fs=12, color=MUTED, ha='left', z=4)
        edge_y = by if side == 'up' else by + bh
        ax.plot([px, px], [y_axis, edge_y], color=col, lw=1.2, zorder=2)
        for p in ([px] if yr != 2017 else [px, x_of(2018)]):
            ax.add_patch(Circle((p, y_axis), .075, fc=col, ec='white', lw=1.2, zorder=4))
    chip(ax, .95, 1.35, 'covered today', BLUE, fs=12)
    chip(ax, .95, .9, 'Days 4 and 5', GREY, fs=12)
    save('nn1_history', f, 'Schematic timeline; publication years of the cited works.', kind='schematic')


def _glyph_seismic(ax, x, y, w, h):
    t = np.linspace(0, 1, 200)
    for k in range(7):
        xc = x + .15 + k * (w - .3) / 6
        tr = .09 * (np.sin(2 * np.pi * (4 * t + .35 * k)) * np.exp(-((t - .5 - .03 * k) / .28) ** 2))
        yy = y + .05 + t * (h - .1)
        ax.plot(xc + tr, yy, color=INK, lw=.9, zorder=4)
        ax.fill_betweenx(yy, xc, xc + np.clip(tr, 0, None), color=BLUE, lw=0, zorder=3)


def _glyph_log(ax, x, y, w, h):
    d = np.linspace(0, 1, 300)
    val = .5 + .22 * np.sin(9 * d) + .12 * np.sin(23 * d + 1) + .05 * np.sin(61 * d)
    xx = x + .2 + val * (w - .4)
    yy = y + h - .08 - d * (h - .16)
    gap = (d > .45) & (d < .7)
    ax.plot(np.where(gap, np.nan, xx), yy, color=GREY, lw=1.4, zorder=4)
    ax.plot(np.where(gap, xx, np.nan), yy, color=ORANGE, lw=1.6, ls=(0, (2, 1.5)), zorder=4)


def _glyph_decline(ax, x, y, w, h):
    t = np.linspace(0, 1, 200)
    q = 1 / (1 + 4 * t) ** .9 + .04 * np.sin(40 * t)
    xx, yy = x + .12 + t * (w - .24), y + .1 + q * (h - .25)
    hist = t <= .6
    ax.plot(xx[hist], yy[hist], color=GREY, lw=1.6, zorder=4)
    ax.plot(xx[~hist], yy[~hist], color=ORANGE, lw=1.6, ls=(0, (2, 1.5)), zorder=4)
    ax.fill_between(xx[~hist], yy[~hist] - .12 * (t[~hist] - .6) * 3, yy[~hist] + .12 * (t[~hist] - .6) * 3,
                    color=ORANGE, alpha=.18, lw=0, zorder=3)


def _glyph_sensor(ax, x, y, w, h):
    t = np.linspace(0, 1, 300)
    peak = None
    for k, off in enumerate([.8, .47, .16]):
        s = .065 * np.sin(2 * np.pi * (6 + 2 * k) * t + k)
        if k == 1:                                          # one spike on the middle channel
            s = s + .16 * np.exp(-((t - .72) / .025) ** 2)
            i = int(np.argmax(s))
            peak = (x + .1 + t[i] * (w - .2), y + off * h + s[i])
        ax.plot(x + .1 + t * (w - .2), y + off * h + s, color=GREY, lw=1.1, zorder=4)
    ax.add_patch(Circle((peak[0], peak[1] - .05), .1, fc='none', ec=ORANGE, lw=1.6, zorder=5))


def _glyph_power(ax, x, y, w, h):
    hr = np.linspace(0, 2, 400)
    frac = hr % 1
    p = .3 + .28 * np.exp(-((frac - .33) / .09) ** 2) + .42 * np.exp(-((frac - .78) / .11) ** 2)
    p = p * (1 + .06 * (hr > 1))
    xx = x + .1 + hr / 2 * (w - .2)
    yy = y + .08 + p * (h - .2)
    hist = hr <= 1
    ax.plot(xx[hist], yy[hist], color=GREY, lw=1.5, zorder=4)
    ax.plot(xx[~hist], yy[~hist], color=ORANGE, lw=1.5, ls=(0, (2, 1.5)), zorder=4)


@fig
def energy_map():
    f, ax = canvas(11.8, 4.3)
    cols = [
        ('Seismic\ninterpretation', _glyph_seismic, 'image patches\n(B, 1, H, W)', 'CNN',
         'fault or facies map', 'Day 4'),
        ('Log\nprediction', _glyph_log, 'log curves\n(B, D)', 'MLP, 1-D CNN',
         'missing curve (PE)', 'Day 3'),
        ('Production\nforecasting', _glyph_decline, 'rate history\n(B, T, F)', 'LSTM, TCN',
         'future rates', 'Day 5'),
        ('Equipment\nhealth', _glyph_sensor, 'sensor channels\n(B, T, F)', 'MLP, autoencoder',
         'anomaly score', 'Days 2 to 5'),
        ('Power and load\nforecasting', _glyph_power, 'weather, history\n(B, T, F)', 'MLP, LSTM',
         'next-day profile', 'Day 5'),
    ]
    w, gap, x0 = 2.16, .21, .05
    for i, (title, glyph, inp, model, out, day) in enumerate(cols):
        x = x0 + i * (w + gap)
        _rect(ax, x, .08, w, 4.14, fc='white', ec=RULE, lw=1.2, r=.1)
        label(ax, x + w / 2, 3.87, title, fs=14.5, weight='semibold', linespacing=1.1)
        _rect(ax, x + .15, 2.62, w - .3, .88, fc=PANEL, r=.06, z=2.5)
        glyph(ax, x + .15, 2.62, w - .3, .88)
        bx, bw = x + .15, w - .3
        a = box(ax, bx, 1.84, bw, .6, '', GREY, fs=12.5)
        name, shape = inp.split('\n')
        label(ax, bx + bw / 2, 2.27, name, fs=12.5)
        ax.text(bx + bw / 2, 2.0, shape, fontsize=12.5, color=INK, ha='center', va='center',
                family='IBM Plex Mono', zorder=4)
        b = box(ax, bx, 1.26, bw, .4, model, BLUE, fs=12.5)
        c = box(ax, bx, .68, bw, .4, out, ORANGE, fs=12.5)
        arrow(ax, a.b, b.t, color=FAINT, lw=1.3, head=10)
        arrow(ax, b.b, c.t, color=FAINT, lw=1.3, head=10)
        chip(ax, x + w / 2, .36, day, TEAL, fs=12)
    save('nn1_energy_map', f, 'Schematic. Typical data-architecture pairings, not recommendations; '
         'glyphs are drawings, not data.', kind='schematic')


# ============================================================ Part 2: tensors
def _grid(ax, x, y_top, rows, cols, cell, color, hl_rows=(), hl_color=ORANGE, lw=1.0, fill=.8, ghost=False):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x + j * cell, y_top - (i + 1) * cell), cell, cell,
                                   fc='none' if ghost else tint(color, fill), ec=color if ghost else 'white',
                                   lw=lw, ls='--' if ghost else '-', zorder=2))
    for r in hl_rows:
        ax.add_patch(Rectangle((x, y_top - (r + 1) * cell), cols * cell, cell, fc='none', ec=hl_color,
                               lw=2.4, zorder=4))
    return x + cols * cell, y_top - rows * cell


@fig
def tensor_ranks():
    f, ax = canvas(11.8, 4.3)
    col = '#5B7089'
    heads = [(1.0, 'Rank 0', 'scalar'), (2.8, 'Rank 1', 'vector'), (5.45, 'Rank 2', 'matrix'),
             (7.85, 'Rank 3', '3-D tensor'), (10.45, 'Rank 4', '4-D tensor')]
    for cx, r, name in heads:
        label(ax, cx, 4.05, r, fs=14.5, weight='semibold', color=BLUE)
        label(ax, cx, 3.72, name, fs=13, color=MUTED)
    # rank 0
    ax.add_patch(Rectangle((.72, 2.05), .56, .56, fc=tint(col, .78), ec=col, lw=1.2, zorder=2))
    label(ax, 1.0, 2.33, '0.34', fs=13)
    # rank 1
    xs = 1.72
    for j in range(6):
        if j == 4:
            label(ax, xs + j * .36 + .18, 2.33, '…', fs=14, color=MUTED)
            continue
        ax.add_patch(Rectangle((xs + j * .36, 2.15), .36, .36, fc=tint(col, .78), ec='white', lw=1.2, zorder=2))
    label(ax, 2.8, 2.75, '60 band energies', fs=12, color=MUTED)
    # rank 2
    _grid(ax, 4.7, 3.2, 6, 5, .3, col, fill=.78)
    arrow(ax, (4.53, 3.2), (4.53, 1.4), color=BLUE, lw=1.4, head=10)
    label(ax, 4.38, 2.3, 'B', fs=13.5, color=BLUE, weight='semibold')
    arrow(ax, (4.7, 3.36), (6.2, 3.36), color=MUTED, lw=1.4, head=10)
    label(ax, 6.33, 3.36, 'D', fs=13.5, color=MUTED, weight='semibold', ha='left')
    # rank 3: batch of sequences
    block3d(ax, 7.05, 1.45, 1.15, 1.45, .75, col)
    label(ax, 7.62, 1.3, 'F channels', fs=12, color=MUTED)
    label(ax, 6.9, 2.18, 'T', fs=13.5, color=MUTED, weight='semibold')
    label(ax, 8.55, 3.33, 'B', fs=13.5, color=BLUE, weight='semibold')
    # rank 4: batch of images
    for k in range(3):
        block3d(ax, 9.55 + k * .42, 1.45 + k * .3, .95, .95, .3, col, alpha=1)
    label(ax, 10.05, 1.3, 'H × W', fs=12, color=MUTED)
    label(ax, 11.45, 3.28, 'B', fs=13.5, color=BLUE, weight='semibold')    # stacking direction, clear of edges
    label(ax, 9.45, 2.6, 'C', fs=13.5, color=MUTED, weight='semibold')     # depth of each image block
    shapes = ['()', '(60,)', '(B, D)', '(B, T, F)', '(B, C, H, W)']
    examples = ['a loss value', 'one sonar echo', 'a batch of echoes\nor log samples',
                'sensor windows', 'seismic patches']
    for (cx, _, _), shp, ex in zip(heads, shapes, examples):
        ax.text(cx, .82, shp, fontsize=14, family='IBM Plex Mono', ha='center', va='center', color=INK,
                zorder=4)
        label(ax, cx, .38, ex, fs=12.5, color=MUTED, va='center', linespacing=1.15)
    save('nn1_tensor_ranks', f, 'Schematic. B = batch, D = features, T = time steps, F = channels, '
         'C = image channels, H x W = image size.', kind='schematic')


@fig
def shape_contract():
    f, ax = canvas(6.3, 4.3)
    cell = .3
    y_top = 3.85
    # X (B x D)
    xX, xW, xb, xZ = .4, 2.3, 3.8, 5.25
    _grid(ax, xX, y_top, 5, 4, cell, GREY, hl_rows=[1])
    label(ax, xX + 2 * cell, y_top + .22, 'D = 60', fs=12.5, color=ORANGE, weight='semibold')
    label(ax, xX - .2, y_top - 2.5 * cell, 'B', fs=13.5, color=MUTED, weight='semibold')
    label(ax, 1.8, y_top - 2 * cell, '@', fs=18, color=INK)
    # W (D x H)
    _grid(ax, xW, y_top, 4, 3, cell, BLUE)
    label(ax, xW + 1.5 * cell, y_top + .22, 'H = 16', fs=12.5, color=MUTED)
    label(ax, xW - .18, y_top - 2 * cell, 'D', fs=13.5, color=ORANGE, weight='semibold')
    label(ax, 3.46, y_top - 2 * cell, '+', fs=18, color=INK)
    # b (H,)
    _grid(ax, xb, y_top - .45, 1, 3, cell, BLUE)
    label(ax, xb + 1.5 * cell, y_top + .22, 'H', fs=12.5, color=MUTED)
    label(ax, 4.96, y_top - 2 * cell, '=', fs=18, color=INK)
    # Z (B x H)
    _grid(ax, xZ, y_top, 5, 3, cell, TEAL, hl_rows=[1])
    label(ax, xZ + 1.5 * cell, y_top + .22, 'H', fs=12.5, color=MUTED)
    names = [(xX + 2 * cell, 'X', '(32, 60)'), (xW + 1.5 * cell, 'W', '(60, 16)'),
             (xb + 1.5 * cell, 'b', '(16,)'), (xZ + 1.5 * cell, 'Z', '(32, 16)')]
    for cx, nm, shp in names:
        label(ax, cx, 2.02, nm, fs=15, weight='semibold', style='italic')
        ax.text(cx, 1.68, shp, fontsize=13, family='IBM Plex Mono', ha='center', va='center', color=INK)
    _rect(ax, .3, .25, 5.75, 1.0, fc=PANEL, r=.08)
    ax.text(.55, .92, '(32, 60) @ (60, 16)', fontsize=13, family='IBM Plex Mono', ha='left', va='center',
            color=INK, zorder=4)
    label(ax, 3.55, .92, 'inner sizes must match', fs=12.5, color=ORANGE, ha='left', weight='medium')
    label(ax, .55, .52, 'row i of Z uses only row i of X (outlined)', fs=12.5, color=MUTED, ha='left')
    save('nn1_shape_contract', f, 'Schematic of Z = XW + b with B = 32 examples, D = 60 features and '
         'H = 16 hidden units.', kind='schematic')


@fig
def broadcasting():
    f, ax = canvas(6.3, 4.9)
    c = .44
    # row 1: bias broadcast
    label(ax, .2, 4.62, 'Adding a bias: (3, 2) + (2,)', fs=13.5, weight='semibold', ha='left')
    check(ax, 5.9, 4.62, s=.13)
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[10, 20]])
    matrix(ax, .3, 4.3, Z, cell=c, color=TEAL, fs=13)
    label(ax, 1.4, 3.64, '+', fs=17)
    matrix(ax, 1.7, 4.3, b, cell=c, color=BLUE, fs=13)
    for k in (1, 2):
        _grid(ax, 1.7, 4.3 - k * c, 1, 2, c, BLUE, ghost=True, lw=1.1)
    label(ax, 2.3 + .52, 3.64, '=', fs=17)
    matrix(ax, 3.25, 4.3, Z + b, cell=c, color=TEAL, fs=13)
    label(ax, 4.25, 3.9, 'b is copied\ndown the batch', fs=12, color=MUTED, ha='left', va='center',
          linespacing=1.15)
    # row 2: target shape bug
    label(ax, .2, 2.62, 'A loss with mismatched shapes: (3, 1) − (3,)', fs=13.5, weight='semibold', ha='left')
    cross(ax, 5.9, 2.62, s=.11)
    yp = np.array([[.9], [.2], [.7]])
    yt = np.array([[1, 0, 1]])
    matrix(ax, .3, 2.3, yp, cell=c, color=ORANGE, fmt='{:.1f}', fs=13)
    label(ax, .98, 1.64, '−', fs=17)
    matrix(ax, 1.25, 2.3, yt, cell=c, color=GREY, fs=13)
    label(ax, 2.72, 1.64, '=', fs=17)
    D = yp - yt
    x0 = 3.0
    for i in range(3):
        for j in range(3):
            diag = i == j
            ax.add_patch(Rectangle((x0 + j * c, 2.3 - (i + 1) * c), c, c,
                                   fc=tint(TEAL, .8) if diag else tint(RED, .8), ec='white', lw=1.2, zorder=2))
            label(ax, x0 + (j + .5) * c, 2.3 - (i + .5) * c, f'{D[i, j]:.1f}'.replace('-', '−'), fs=12.5)
    mse_ok = float(np.mean(np.diag(D) ** 2))
    mse_bug = float(np.mean(D ** 2))
    label(ax, 4.45, 2.2, 'result: (3, 3)', fs=12, color=RED, ha='left', weight='semibold')
    label(ax, 4.45, 1.87, 'diagonal = intended', fs=12, color=TEAL, ha='left')
    label(ax, 4.45, 1.56, 'others = nonsense', fs=12, color=RED, ha='left')
    _rect(ax, .25, .22, 5.8, .72, fc=PANEL, r=.08)
    label(ax, .45, .58, f'MSE intended {mse_ok:.3f}   ·   MSE computed {mse_bug:.3f}   ·   no error raised',
          fs=12.5, ha='left')
    save('nn1_broadcasting', f, f'Exact NumPy broadcasting arithmetic; MSE values {mse_ok:.4f} and '
         f'{mse_bug:.4f}.', kind='computed')


@fig
def cpu_gpu():
    f, ax = canvas(6.3, 4.9)
    # CPU
    _rect(ax, .15, 2.15, 2.75, 2.55, fc=PANEL, ec='#8193A6', lw=1.4, r=.1)
    label(ax, 1.525, 4.42, 'CPU', fs=15, weight='semibold')
    for i in range(2):
        for j in range(2):
            box(ax, .38 + j * 1.2, 3.235 - i * .9, 1.08, .78, 'core', BLUE, fs=13)   # centred below the label
    label(ax, 1.525, 1.88, 'a few powerful cores', fs=12.5, color=MUTED)
    # GPU
    _rect(ax, 3.3, 2.15, 2.85, 2.55, fc=PANEL, ec='#8193A6', lw=1.4, r=.1)
    label(ax, 4.725, 4.42, 'GPU', fs=15, weight='semibold')
    s, g = .165, .045
    for i in range(9):
        for j in range(12):
            ax.add_patch(Rectangle((3.49 + j * (s + g), 2.3 + i * (s + g)), s, s, fc=tint(TEAL, .55),
                                   ec='none', zorder=3))
    label(ax, 4.725, 1.88, 'many simple cores', fs=12.5, color=MUTED)
    # matrix product = many independent dot products
    ca = .27
    _grid(ax, .5, 1.45, 3, 3, ca, GREY, hl_rows=[1])
    label(ax, 1.5, 1.05, '×', fs=16)
    _grid(ax, 1.8, 1.45, 3, 3, ca, BLUE)
    ax.add_patch(Rectangle((1.8 + 2 * ca, 1.45 - 3 * ca), ca, 3 * ca, fc='none', ec=ORANGE, lw=2.4, zorder=4))
    label(ax, 2.8, 1.05, '=', fs=16)
    _grid(ax, 3.1, 1.45, 3, 3, ca, ORANGE, fill=.85)
    ax.add_patch(Rectangle((3.1 + 2 * ca, 1.45 - 2 * ca), ca, ca, fc=ORANGE, ec='white', lw=1, zorder=4))
    label(ax, 4.1, 1.2, 'each output cell is an\nindependent dot product,\nso all can run at once',
          fs=12, color=INK, ha='left', va='center', linespacing=1.2)
    save('nn1_cpu_gpu', f, 'Schematic; core counts are illustrative, not a specific device.', kind='schematic')


@fig
def sonar_profiles():
    X, y = _sonar()
    f, axs = half()
    ax = axs[0]
    band = np.arange(1, 61)
    for cls, col, name in [(0, BLUE, 'Rock (R)'), (1, ORANGE, 'Metal cylinder (M)')]:
        Xi = X[y == cls]
        q1, med, q3 = np.percentile(Xi, [25, 50, 75], axis=0)
        ax.fill_between(band, q1, q3, color=col, alpha=.16, lw=0)
        ax.plot(band, med, color=col, lw=2.4)
    ax.annotate(f'Rock (R), n = {(y == 0).sum()}', (41, .55), color=BLUE, fontsize=13, fontweight='medium')
    ax.annotate(f'Metal cylinder (M), n = {(y == 1).sum()}', (32.5, .86), color=ORANGE, fontsize=13,
                fontweight='medium')
    ax.set_xlim(1, 60)
    ax.set_ylim(0, 1)
    clean(ax, 'Band index (column order in the file)', 'Band energy (0 to 1)')
    save('nn1_sonar_profiles', f, 'sonar.all-data.csv (208 echoes): median and interquartile range of each '
         'band by class.', kind='data')


def _sonar_protocol_scores():
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    X, y = _sonar()    # Mine = 1, Rock = 0, as in the Lab 1 notebook
    # The notebook stratifies on the M/R labels; the same strings give the same split.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42,
                                          stratify=np.where(y == 1, 'M', 'R'))
    models = {
        'Logistic regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'RBF SVM': SVC(kernel='rbf', random_state=42),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        test = pipe.fit(Xtr, ytr).score(Xte, yte)
        cv_file = cross_val_score(pipe, X, y, cv=5)
        cv_shuf = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        out[name] = (test, cv_file, cv_shuf)
    majority = max(np.mean(yte), 1 - np.mean(yte))
    return out, majority


@fig
def sonar_protocols():
    res, majority = _sonar_protocol_scores()
    f, axs = sub(1, height=4.1)
    ax = axs[0]
    offs = {'test': -.27, 'file': 0, 'shuf': .27}
    for g, (name, (test, cv_file, cv_shuf)) in enumerate(res.items()):
        ax.scatter(g + offs['test'], test, marker='D', s=110, color=ORANGE, zorder=4,
                   label='Single test split (42 rows)' if g == 0 else None)
        for key, vals, col, lab in [('file', cv_file, PURPLE, '5-fold CV in file order (all 208 rows)'),
                                    ('shuf', cv_shuf, TEAL, '5-fold CV, shuffled (166 training rows)')]:
            xx = g + offs[key] + np.linspace(-.06, .06, len(vals))
            ax.scatter(xx, vals, s=46, color=col, zorder=4, edgecolors='white', linewidths=.6,
                       label=lab if g == 0 else None)
            ax.plot([g + offs[key] - .1, g + offs[key] + .1], [vals.mean()] * 2, color=col, lw=3, zorder=3)
    ax.axhline(majority, color=GREY, ls='--', lw=1.4)
    ax.text(2.55, majority - .012, f'majority class on the test split {majority:.2f}', color=GREY, fontsize=12,
            ha='right', va='top')
    ax.set_xticks(range(3), list(res))
    ax.tick_params(axis='x', labelsize=13.5, length=0)
    ax.set_xlim(-.55, 2.6)
    ax.set_ylim(.35, 1.02)
    clean(ax, None, 'Accuracy')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', ncol=3, fontsize=12.5, bbox_to_anchor=(.5, 1.0), handletextpad=.3,
              columnspacing=1.6, markerscale=1)
    save('nn1_sonar_protocols', f, 'sonar.all-data.csv; Lab 1 pipelines (scaler + model, default settings); '
         'test split random_state=42; shuffled folds random_state=0.', kind='data')


# ============================================================ Part 3: neuron, perceptron, XOR
@fig
def bio_vs_artificial():
    f, ax = canvas(11.8, 4.3)
    bio = '#7D8FA3'
    # ---- biological sketch (left)
    label(ax, 2.75, 4.08, 'Biological neuron (sketch)', fs=14.5, weight='semibold', color=INK)
    soma = np.array([2.6, 2.4])
    ax.add_patch(Ellipse(soma, 1.25, 1.0, fc=tint(bio, .75), ec=bio, lw=1.8, zorder=3))
    ax.add_patch(Circle(soma, .16, fc=bio, ec='none', zorder=4))
    rng = np.random.default_rng(3)
    for ang in np.linspace(132, 228, 5):
        a = np.radians(ang)
        p0 = soma + np.array([.58 * np.cos(a), .46 * np.sin(a)])
        p1 = p0 + .62 * np.array([np.cos(a), np.sin(a)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=bio, lw=2.8, solid_capstyle='round', zorder=2)
        for da in (-.45, .4):
            p2 = p1 + .38 * np.array([np.cos(a + da), np.sin(a + da)])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=bio, lw=1.8, solid_capstyle='round', zorder=2)
            for dd in (-.35, .35):
                p3 = p2 + .2 * np.array([np.cos(a + da + dd + rng.uniform(-.2, .2)),
                                          np.sin(a + da + dd + rng.uniform(-.2, .2))])
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=bio, lw=1.1, solid_capstyle='round', zorder=2)
    ax.plot([3.22, 4.95], [2.4, 2.4], color=bio, lw=2.8, zorder=2)
    for xm in (3.35, 3.87, 4.39):
        _rect(ax, xm, 2.26, .42, .28, fc=tint(bio, .6), ec=bio, lw=1.2, r=.12, z=3)
    for ang in (-38, 0, 38):
        a = np.radians(ang)
        p1 = np.array([4.95, 2.4]) + .45 * np.array([np.cos(a), np.sin(a)])
        ax.plot([4.95, p1[0]], [2.4, p1[1]], color=bio, lw=1.8, zorder=2)
        ax.add_patch(Circle(p1, .1, fc=bio, ec='none', zorder=3))
    for x, y, t in [(.75, 1.05, 'dendrites'), (2.6, 3.12, 'cell body'), (4.1, 1.9, 'axon'),
                    (5.25, 3.02, 'synapses')]:
        label(ax, x, y, t, fs=13, color=MUTED)
    # ---- artificial unit (right)
    label(ax, 9.05, 4.08, 'Artificial unit (a formula)', fs=14.5, weight='semibold', color=INK)
    ins = [(6.55, 3.3), (6.55, 2.45), (6.55, 1.6)]
    s_c = (8.55, 2.45)
    for k, p in enumerate(ins):
        node(ax, *p, r=.24, color=GREY, text=f'$x_{k + 1}$', fs=14)
        arrow(ax, (p[0] + .26, p[1]), (s_c[0] - .34, s_c[1] + (p[1] - s_c[1]) * .25), color=BLUE, lw=1.6,
              head=11)
        label(ax, *[(7.66, 3.2), (7.5, 2.66), (7.66, 1.7)][k], f'$w_{k + 1}$', fs=14, color=BLUE)
    node(ax, *s_c, r=.34, color=TEAL, text='Σ', fs=18)
    box(ax, 8.2, 3.35, .7, .42, '+ $b$', BLUE, fs=13.5)
    arrow(ax, (8.55, 3.35), (8.55, 2.81), color=BLUE, lw=1.4, head=10)
    b2 = box(ax, 9.3, 2.14, .78, .62, 'φ', TEAL, fs=18, fill='tint')
    arrow(ax, (s_c[0] + .35, s_c[1]), b2.l, color=INK, lw=1.6, head=11)
    node(ax, 10.85, 2.45, r=.28, color=ORANGE, text='$a$', fs=15)
    arrow(ax, b2.r, (10.55, 2.45), color=INK, lw=1.6, head=11)
    for x, y, t in [(6.55, 1.08, 'inputs'), (7.55, 1.08, 'weights'), (8.55, 1.6, 'sum'),
                    (9.69, 1.6, 'activation'), (10.85, 1.8, 'output')]:
        label(ax, x, y, t, fs=13, color=MUTED)
    ax.plot([5.75, 5.75], [1.0, 4.2], color=RULE, lw=1.2)
    # ---- caveat strip
    _rect(ax, .1, .08, 11.6, .62, fc=ORANGE_T, r=.08)
    label(ax, .3, .39, 'What the analogy leaves out:', fs=13, color=ORANGE, weight='semibold', ha='left')
    label(ax, 3.2, .39, 'spike timing  ·  computation inside dendrites  ·  chemical signalling  ·  '
          'adaptation over time', fs=13, color=INK, ha='left')
    save('nn1_bio_vs_artificial', f, 'Schematic drawing; not anatomically accurate.', kind='schematic')


@fig
def neuron():
    f, ax = canvas(6.3, 4.9)
    x1, x2 = (.75, 3.55), (.75, 1.75)
    sc = (3.0, 2.65)
    node(ax, *x1, r=.36, color=GREY, text='1', fs=16)
    node(ax, *x2, r=.36, color=GREY, text='2', fs=16)
    label(ax, .75, 4.12, '$x_1$', fs=15, color=MUTED)
    label(ax, .75, 1.18, '$x_2$', fs=15, color=MUTED)
    arrow(ax, (1.12, 3.47), (2.58, 2.82), color=BLUE, lw=2, head=12)
    arrow(ax, (1.12, 1.83), (2.58, 2.48), color=BLUE, lw=2, head=12)
    label(ax, 1.85, 3.52, '$w_1 = 0.5$', fs=14, color=BLUE)
    label(ax, 1.85, 1.78, '$w_2 = -1$', fs=14, color=BLUE)
    node(ax, *sc, r=.42, color=TEAL, text='Σ', fs=19)
    box(ax, 2.55, 3.72, .9, .45, '$b = 1$', BLUE, fs=14)
    arrow(ax, (3.0, 3.72), (3.0, 3.09), color=BLUE, lw=1.5, head=10)
    label(ax, 3.0, 1.98, '$z = -0.5$', fs=15, weight='semibold', color=TEAL)
    # two activation choices
    for (y, name, val) in [(3.55, 'ReLU', '0'), (1.75, 'sigmoid', '0.378')]:
        p0 = np.array([3.44, 2.65 + (y - 2.65) * .15])
        u = (np.array([5.3, y]) - p0) / np.linalg.norm(np.array([5.3, y]) - p0)
        arrow(ax, tuple(p0), tuple(np.array([5.3, y]) - u * .5), color=INK, lw=1.6, head=11)   # stops at the node
        node(ax, 5.3, y, r=.46, color=ORANGE, text=val, fs=15)
        label(ax, 5.3, y + .66 if y > 3 else y - .66, name, fs=13.5, color=MUTED)
    _rect(ax, .25, .15, 5.8, .62, fc=PANEL, r=.08)
    label(ax, 3.15, .46, '$z = 0.5·1 + (-1)·2 + 1 = -0.5$', fs=14)
    save('nn1_neuron', f, 'Exact arithmetic for inputs (1, 2), weights (0.5, -1), bias 1; '
         'sigmoid(-0.5) = 0.3775.', kind='computed')


def _perceptron_axes(ax, lim=(-.6, 3.6)):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    clean(ax, '$x_1$', '$x_2$')


@fig
def perceptron():
    f, axs = half()
    ax = axs[0]
    w, b = np.array([1., 2.]), -3.
    xx, yy = np.meshgrid(np.linspace(-.6, 3.6, 300), np.linspace(-.6, 3.6, 300))
    zz = w[0] * xx + w[1] * yy + b
    ax.contourf(xx, yy, (zz > 0).astype(int), levels=[-.5, .5, 1.5], cmap=BG, zorder=0)
    xl = np.array([-.6, 3.6])
    ax.plot(xl, (-b - w[0] * xl) / w[1], color=INK, lw=2.2)
    P0 = np.array([[.2, .4], [.8, .2], [1.4, .5], [.3, 1.0], [.1, 1.2]])
    P1 = np.array([[2.4, 1.5], [1.6, 2.4], [3.0, .9], [.9, 2.9], [2.6, 2.8]])
    ax.scatter(P0[:, 0], P0[:, 1], s=70, c=BLUE, marker='o', edgecolors='white', lw=.8, zorder=3)
    ax.scatter(P1[:, 0], P1[:, 1], s=80, c=ORANGE, marker='^', edgecolors='white', lw=.8, zorder=3)
    p = np.array([1.0, 1.0])
    ax.annotate('', p + .75 * w / np.linalg.norm(w), p, arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.2,
                                                                        mutation_scale=16), zorder=4)
    ax.text(1.42, 1.72, '$w$', color=BLUE, fontsize=16, fontweight='semibold')
    ax.text(2.45, .62, '$w·x + b = 0$', fontsize=13.5, ha='center', color=INK, rotation=-26.57,
            rotation_mode='anchor')
    ax.text(3.35, 3.3, '$\\hat{y} = 1$', fontsize=15, ha='right', color=ORANGE, fontweight='semibold')
    ax.text(-.4, -.35, '$\\hat{y} = 0$', fontsize=15, ha='left', color=BLUE, fontweight='semibold')
    _perceptron_axes(ax)
    save('nn1_perceptron', f, 'Exact boundary x1 + 2 x2 - 3 = 0 with w = (1, 2), b = -3; '
         'illustrative points.', kind='computed')


@fig
def perceptron_update():
    f, axs = half()
    ax = axs[0]
    P1 = np.array([[2.0, .5], [1.5, -.5], [2.5, 1.6]])
    P0 = np.array([[-1.0, .5], [-1.5, -1.0], [-.5, 1.5], [-1.2, 2.2]])
    x = np.array([1.0, 2.0])
    w_old, b_old = np.array([1.0, -1.0]), -.5
    eta = .5
    w_new, b_new = w_old + eta * x, b_old + eta
    lim = (-2, 3)
    xl = np.array(lim)
    ax.plot(xl, (-b_old - w_old[0] * xl) / w_old[1], color=GREY, lw=2, ls='--')
    ax.axvline(0, color=INK, lw=2.4) if abs(w_new[1]) < 1e-12 else ax.plot(xl, (-b_new - w_new[0] * xl) / w_new[1])
    ax.fill_betweenx(np.array(lim), 0, lim[1], color=ORANGE_T, zorder=0)
    ax.fill_betweenx(np.array(lim), lim[0], 0, color=BLUE_T, zorder=0)
    ax.scatter(P0[:, 0], P0[:, 1], s=70, c=BLUE, marker='o', edgecolors='white', lw=.8, zorder=3)
    ax.scatter(P1[:, 0], P1[:, 1], s=80, c=ORANGE, marker='^', edgecolors='white', lw=.8, zorder=3)
    ax.scatter([x[0]], [x[1]], s=110, c=ORANGE, marker='^', edgecolors=INK, lw=1.2, zorder=4)
    ax.add_patch(Circle(x, .3, fc='none', ec=RED, lw=2, zorder=4))
    ax.text(1.45, 2.72, 'misclassified before:\n$y = 1$, $\\hat{y} = 0$', fontsize=13, color=RED, va='center',
            ha='center', linespacing=1.15)
    ax.text(1.5, .45, 'before', color=GREY, fontsize=13, ha='center', va='center', rotation=45,
            rotation_mode='anchor')                     # beside the dashed boundary (slope 1, equal aspect)
    ax.text(.1, -1.75, 'after one update', color=INK, fontsize=13, ha='left')
    ax.set_xlim(*lim)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    clean(ax, '$x_1$', '$x_2$')
    save('nn1_perceptron_update', f, f'Exact perceptron update: w {w_old.tolist()} -> {w_new.tolist()}, '
         f'b {b_old} -> {b_new}, learning rate {eta}, x = (1, 2), y = 1.', kind='computed')


def _perceptron_train(X, y, eta=.5, epochs=20):
    w, b = np.zeros(2), 0.
    hist = []
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            yhat = int(w @ xi + b > 0)
            if yhat != yi:
                w, b = w + eta * (yi - yhat) * xi, b + eta * (yi - yhat)
                errors += 1
        hist.append(errors)
    return w, b, hist


@fig
def logic_gates():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    # (name, labels, w, b, title)
    panels = [('AND', np.array([0, 0, 0, 1]), np.array([1., 1.]), -1.5, 'AND:  w = (1, 1),  b = −1.5'),
              ('OR', np.array([0, 1, 1, 1]), np.array([1., 1.]), -.5, 'OR:  w = (1, 1),  b = −0.5'),
              ('XOR', np.array([0, 1, 1, 0]), np.array([1., 1.]), -.5, 'XOR:  every line misses a point')]
    f, axs = sub(3, height=4.2)
    for ax, (name, y, w, b, title) in zip(axs, panels):
        pred = (X @ w + b > 0).astype(int)
        wrong = pred != y
        xx, yy = np.meshgrid(np.linspace(-.5, 1.5, 300), np.linspace(-.5, 1.5, 300))
        zz = (w[0] * xx + w[1] * yy + b > 0).astype(int)
        ax.contourf(xx, yy, zz, levels=[-.5, .5, 1.5], cmap=BG, zorder=0)
        xl = np.array([-.5, 1.5])
        ax.plot(xl, (-b - w[0] * xl) / w[1], color=INK if not wrong.any() else RED, lw=2.2,
                ls='-' if not wrong.any() else '--')
        for (px, py), yi, bad in zip(X, y, wrong):
            ax.scatter(px, py, s=200, c=ORANGE if yi else BLUE, marker='^' if yi else 'o', edgecolors='white',
                       lw=1, zorder=3)
            dx = .2 if bad else .1                     # clear of the red ring on a misclassified point
            ax.text(px + dx, py + .1, str(yi), fontsize=14, fontweight='semibold', color=INK, zorder=4)
            if bad:
                ax.add_patch(Circle((px, py), .15, fc='none', ec=RED, lw=2.2, zorder=4))
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(-.5, 1.5)
        ax.set_aspect('equal')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        clean(ax, '$x_1$', '$x_2$' if name == 'AND' else None, grid=False)
        ax.set_title(title, fontsize=14.5)
    # the perceptron rule itself never settles on XOR (checked, not drawn)
    _, _, hist = _perceptron_train(X, np.array([0, 1, 1, 0]))
    assert hist[-1] > 0
    save('nn1_logic_gates', f, 'Exact threshold units w = (1, 1) with b = -1.5 (AND) and b = -0.5 (OR); '
         'on XOR the OR line misclassifies (1, 1). The perceptron rule on XOR still makes errors every '
         'epoch after 20 epochs.', kind='computed')


@fig
def xor_mlp():
    f, ax = canvas(11.8, 4.3)
    # ---- network with hand-set weights and step activations
    x1, x2, h1, h2, yo = (.55, 3.25), (.55, 1.25), (2.75, 3.25), (2.75, 1.25), (4.75, 2.25)
    for p, t in [(x1, '$x_1$'), (x2, '$x_2$')]:
        node(ax, *p, r=.3, color=GREY, text=t, fs=15)
    for p, t in [(h1, '$h_1$'), (h2, '$h_2$')]:
        node(ax, *p, r=.33, color=TEAL, text=t, fs=15)
    node(ax, *yo, r=.33, color=ORANGE, text='$y$', fs=15)
    for a, b_, lab, pos in [(x1, h1, '1', (1.65, 3.43)), (x2, h1, '1', (1.2, 2.55)),
                            (x1, h2, '1', (1.2, 1.95)), (x2, h2, '1', (1.65, 1.07)),
                            (h1, yo, '+1', (3.85, 2.98)), (h2, yo, '−1', (3.85, 1.52))]:
        arrow(ax, (a[0] + .3, a[1]), (b_[0] - .34, b_[1] + (a[1] - b_[1]) * .12), color=BLUE, lw=1.6, head=11)
        label(ax, *pos, lab, fs=14, color=BLUE, weight='semibold')
    label(ax, 2.75, 3.85, '$b = -0.5$  (OR)', fs=13, color=BLUE)
    label(ax, 2.75, .65, '$b = -1.5$  (AND)', fs=13, color=BLUE)
    label(ax, 4.75, 2.85, '$b = -0.5$', fs=13, color=BLUE)
    label(ax, 2.75, .25, 'all units: step(z) = 1 if z > 0, else 0', fs=12.5, color=MUTED)
    # ---- truth table
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    tx, ty, cw, rh = 5.45, 3.75, .5, .52
    heads = ['$x_1$', '$x_2$', '$h_1$', '$h_2$', '$y$']
    cols_c = [GREY, GREY, TEAL, TEAL, ORANGE]
    for j, (hd, cc) in enumerate(zip(heads, cols_c)):
        ax.add_patch(Rectangle((tx + j * cw, ty), cw, rh, fc=tint(cc, .75), ec='white', lw=1.5, zorder=2))
        label(ax, tx + (j + .5) * cw, ty + rh / 2, hd, fs=14)
    for i, (a, b_) in enumerate(rows):
        hh1 = int(a + b_ - .5 > 0)
        hh2 = int(a + b_ - 1.5 > 0)
        yy = int(hh1 - hh2 - .5 > 0)
        for j, v in enumerate([a, b_, hh1, hh2, yy]):
            ax.add_patch(Rectangle((tx + j * cw, ty - (i + 1) * rh), cw, rh, fc=PANEL if i % 2 == 0 else 'white',
                                   ec='white', lw=1.5, zorder=2))
            label(ax, tx + (j + .5) * cw, ty - (i + .5) * rh, str(v), fs=14,
                  weight='semibold' if j == 4 else 'normal', color=ORANGE if (j == 4 and v) else INK)
    label(ax, tx + 2.5 * cw, ty - 4 * rh - .35, '$y = h_1$ AND NOT $h_2$ = XOR', fs=13, color=MUTED)
    # ---- input-space view
    ax2 = f.add_axes([.765, .15, .225, .78])
    xx, yy = np.meshgrid(np.linspace(-.5, 1.5, 300), np.linspace(-.5, 1.5, 300))
    band = ((xx + yy - .5 > 0) & ~(xx + yy - 1.5 > 0)).astype(int)
    ax2.contourf(xx, yy, band, levels=[-.5, .5, 1.5], cmap=BG, zorder=0)
    xl = np.array([-.5, 1.5])
    ax2.plot(xl, .5 - xl, color=TEAL, lw=2)
    ax2.plot(xl, 1.5 - xl, color=TEAL, lw=2)
    ax2.text(-.42, .6, '$h_1$', color=TEAL, fontsize=14, fontweight='semibold')
    ax2.text(1.28, .42, '$h_2$', color=TEAL, fontsize=14, fontweight='semibold')
    for (a, b_), lab in zip(rows, [0, 1, 1, 0]):
        ax2.scatter(a, b_, s=190, c=ORANGE if lab else BLUE, marker='^' if lab else 'o', edgecolors='white',
                    lw=1, zorder=3)
    ax2.set_xlim(-.5, 1.5)
    ax2.set_ylim(-.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    clean(ax2, '$x_1$', '$x_2$', grid=False)
    save('nn1_xor_mlp', f, 'Exact hand-set 2-2-1 threshold network: h1 = step(x1 + x2 - 0.5), '
         'h2 = step(x1 + x2 - 1.5), y = step(h1 - h2 - 0.5).', kind='computed')


def _xor_data(seed=0, n=40, sd=.13):
    rng = np.random.default_rng(seed)
    C = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    X = np.concatenate([c + rng.normal(0, sd, (n, 2)) for c in C])
    y = np.repeat([0, 1, 1, 0], n).astype(float)
    return X, y


def _xor_train(X, y, H, seed, lr=1.0, epochs=4000):
    """Full-batch gradient descent, tanh hidden layer, sigmoid output, BCE; inputs centred at 0.5."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, H))
    b1 = np.zeros(H)
    w2 = rng.normal(0, 1, H) / np.sqrt(H)
    b2 = 0.
    Xc = X - .5
    for _ in range(epochs):
        A = np.tanh(Xc @ W1 + b1)
        p = expit(A @ w2 + b2)
        g = (p - y) / len(y)
        gZ = np.outer(g, w2) * (1 - A ** 2)
        W1 -= lr * Xc.T @ gZ
        b1 -= lr * gZ.sum(0)
        w2 -= lr * A.T @ g
        b2 -= lr * g.sum()
    A = np.tanh(Xc @ W1 + b1)
    p = expit(A @ w2 + b2)
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return dict(W1=W1, b1=b1, w2=w2, b2=b2, loss=loss, acc=np.mean((p > .5) == y))


def _xor_hidden_notes(ax, Hs, y):
    """Label where the class clusters land in hidden space (positions from the data)."""
    one = Hs[y == 1].mean(0)
    ax.annotate('both class-1\nclusters', one, xytext=(one[0] - .95, one[1] - .45), fontsize=12.5, color=ORANGE,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2), ha='center', va='center')


@fig
def xor_hidden_space():
    X, y = _xor_data()
    m = _xor_train(X, y, 2, seed=2, epochs=300)
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    xx, yy = np.meshgrid(np.linspace(-.5, 1.5, 300), np.linspace(-.5, 1.5, 300))
    G = np.c_[xx.ravel(), yy.ravel()] - .5
    P = expit(np.tanh(G @ m['W1'] + m['b1']) @ m['w2'] + m['b2']).reshape(xx.shape)
    ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), cmap=CMAP_DIV.reversed(), alpha=.28)
    ax.contour(xx, yy, P, levels=[.5], colors=[INK], linewidths=1.6)
    for k in range(2):
        w, b = m['W1'][:, k], m['b1'][k]
        xl = np.array([-.5, 1.5])
        ax.plot(xl, .5 + (-b - w[0] * (xl - .5)) / w[1], color=TEAL, lw=1.8, ls='--')
    points(ax, X, y.astype(int), s=26)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(-.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Input space: two hidden lines form a band')
    clean(ax, '$x_1$', '$x_2$', grid=False)
    ax = axs[1]
    Hs = np.tanh((X - .5) @ m['W1'] + m['b1'])
    hx, hy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
    Ph = expit(np.c_[hx.ravel(), hy.ravel()] @ m['w2'] + m['b2']).reshape(hx.shape)
    ax.contourf(hx, hy, Ph, levels=np.linspace(0, 1, 11), cmap=CMAP_DIV.reversed(), alpha=.28)
    ax.contour(hx, hy, Ph, levels=[.5], colors=[INK], linewidths=1.6)
    points(ax, Hs, y.astype(int), s=26)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    _xor_hidden_notes(ax, Hs, y)
    ax.set_title('Hidden space: a straight line suffices')
    clean(ax, '$h_1$ (tanh)', '$h_2$ (tanh)', grid=False)
    save('nn1_xor_hidden_space', f, f'Noisy XOR (4 x 40 points, sd 0.13, seed 0); 2-2-1 network, tanh hidden '
         f'units, 300 full-batch steps (learning rate 1.0), init seed 2; training accuracy {m["acc"]:.3f}. '
         'Dashed lines: where each hidden unit is zero.', kind='computed')


@fig
def xor_seeds():
    X, y = _xor_data()
    f, axs = half()
    ax = axs[0]
    rng = np.random.default_rng(1)
    summary = {}
    for k, H in enumerate([2, 4, 8]):
        res = [_xor_train(X, y, H, seed=s) for s in range(20)]
        loss = np.array([r['loss'] for r in res])
        ok = np.array([r['acc'] >= .95 for r in res])
        jit = rng.uniform(-.17, .17, len(loss))
        ax.scatter(k + jit[ok], loss[ok], s=60, c=TEAL, edgecolors='white', lw=.6, zorder=3)
        ax.scatter(k + jit[~ok], loss[~ok], s=70, c=ORANGE, marker='X', edgecolors='white', lw=.6, zorder=3)
        ax.text(k, 1.25, f'{ok.sum()} / 20 solved', ha='center', fontsize=13.5, fontweight='semibold',
                color=TEAL if ok.all() else INK)
        summary[H] = int(ok.sum())
    ax.set_yscale('log')
    ax.set_ylim(3e-4, 2.5)
    ax.set_xlim(-.5, 2.5)
    ax.set_xticks(range(3), ['2 hidden units', '4 hidden units', '8 hidden units'])
    ax.tick_params(axis='x', labelsize=13, length=0)
    ax.text(.3, .349, 'stuck: two clusters\nleft near $p$ = 0.5', fontsize=12.5, color=ORANGE, va='center')
    clean(ax, None, 'Final training loss (log scale)')
    ax.grid(axis='x', visible=False)
    save('nn1_xor_seeds', f, f'Noisy XOR, 20 initialization seeds per width, 4000 full-batch steps, learning '
         f'rate 1.0; solved = training accuracy >= 0.95: {summary}.', kind='computed')


# ============================================================ Part 4: multilayer networks
def _dense(ax, sizes, xs, y0, y1, colors, r=.17, max_nodes=7, edge='#9FB0C2'):
    """Layered network; layers larger than max_nodes show three dots in place of hidden nodes."""
    layers = []
    for size in sizes:
        shown = min(size, max_nodes)
        ys = np.linspace(y1, y0, shown) if shown > 1 else np.array([(y0 + y1) / 2])
        keep = [i for i in range(shown) if not (size > max_nodes and i == shown - 2)]
        layers.append((ys, keep, size > max_nodes))
    for (ya, ka, _), (yb, kb, _), xa, xb in zip(layers[:-1], layers[1:], xs[:-1], xs[1:]):
        for i in ka:
            for j in kb:
                ax.plot([xa, xb], [ya[i], yb[j]], color=edge, lw=.75, zorder=1)
    for (ys, keep, trunc), x, col in zip(layers, xs, colors):
        for i in keep:
            node(ax, x, ys[i], r, col)
        if trunc:
            yc = ys[len(ys) - 2]
            for dy in (-.1, 0, .1):
                ax.add_patch(Circle((x, yc + dy), .028, fc=MUTED, ec='none', zorder=4))


@fig
def mlp_shapes():
    f, ax = canvas(11.8, 4.3)
    sizes = [60, 32, 16, 1]
    xs = np.linspace(1.0, 10.6, 4)
    _dense(ax, sizes, xs, .95, 3.35, [GREY, TEAL, TEAL, ORANGE], r=.17)
    names = ['Input', 'Hidden 1 (ReLU)', 'Hidden 2 (ReLU)', 'Output (logit)']
    shapes = ['(B, 60)', '(B, 32)', '(B, 16)', '(B, 1)']
    for x, nm, shp in zip(xs, names, shapes):
        label(ax, x, .52, nm, fs=13.5, weight='semibold')
        ax.text(x, .2, shp, fontsize=13, family='IBM Plex Mono', ha='center', va='center', color=MUTED)
    params = [(60, 32), (32, 16), (16, 1)]
    total = 0
    for k, (a, b) in enumerate(params):
        n = a * b + b
        total += n
        xm = (xs[k] + xs[k + 1]) / 2
        _rect(ax, xm - 1.45, 3.62, 2.9, .56, fc=tint(BLUE, .9), ec=BLUE, lw=1.2, r=.08, z=5)
        label(ax, xm, 3.9, f'$W_{k + 1}, b_{k + 1}$:  {a}×{b} + {b} = {n:,}', fs=13, color=BLUE, z=6)
    chip(ax, 10.6, 1.3, f'total {total:,} parameters', INK, fs=13, solid=True)
    save('nn1_mlp_shapes', f, f'Schematic of a 60-32-16-1 network; parameter counts exact ({total}).',
         kind='schematic')


@fig
def layer_matrix():
    f, ax = canvas(6.3, 4.9)
    W = np.array([[.5, 1.], [-1., .5], [.2, -.3]])
    x = np.array([[1.], [2.]])
    b = np.array([[0.], [1.], [.1]])
    z = W @ x + b
    a = np.maximum(z, 0)
    c = .56
    top = 3.6
    matrix(ax, .25, top, W, cell=c, color=BLUE, fmt='{:g}', fs=13.5, highlight=(0, 0, 1, 2), hl_color=ORANGE)
    matrix(ax, 1.62, top - .28, x, cell=c, color=GREY, fmt='{:g}', fs=13.5, highlight=(0, 0, 2, 1),
           hl_color=ORANGE)
    label(ax, 2.47, top - 1.5 * c, '+', fs=17)
    matrix(ax, 2.72, top, b, cell=c, color=BLUE, fmt='{:g}', fs=13.5)
    label(ax, 3.58, top - 1.5 * c, '=', fs=17)
    matrix(ax, 3.83, top, z, cell=c, color=TEAL, fmt='{:g}', fs=13.5, highlight=(0, 0, 1, 1), hl_color=ORANGE)
    arrow(ax, (4.47, top - 1.5 * c), (5.02, top - 1.5 * c), color=INK, lw=1.5, head=11)
    label(ax, 4.75, top - 1.5 * c + .25, 'ReLU', fs=12.5, color=MUTED)
    matrix(ax, 5.1, top, a, cell=c, color=TEAL, fmt='{:g}', fs=13.5)
    for t in ax.texts:                                  # typographic minus signs in the matrix cells
        if t.get_text().startswith('-'):
            t.set_text('−' + t.get_text()[1:])
    for xc, nm, shp in [(.25 + c, '$W$', '(3, 2)'), (1.62 + c / 2, '$x$', '(2, 1)'), (2.72 + c / 2, '$b$', '(3, 1)'),
                        (3.83 + c / 2, '$z$', '(3, 1)'), (5.1 + c / 2, '$a$', '(3, 1)')]:
        label(ax, xc, top + .38, nm, fs=15.5)
        ax.text(xc, top - 3 * c - .3, shp, fontsize=12.5, family='IBM Plex Mono', ha='center', va='center',
                color=MUTED)
    _rect(ax, .2, .25, 5.9, 1.05, fc=PANEL, r=.08)
    label(ax, .4, .98, '$z_1 = 0.5·1 + 1.0·2 + 0 = 2.5$', fs=14, ha='left')
    label(ax, .4, .55, 'row $j$ of $W$ holds the weights of unit $j$', fs=13, ha='left', color=MUTED)
    save('nn1_layer_matrix', f, 'Exact arithmetic: z = Wx + b = (2.5, 1.0, -0.3), a = ReLU(z) = (2.5, 1.0, 0).',
         kind='computed')


@fig
def linear_collapse():
    from sklearn.datasets import make_circles
    from sklearn.neural_network import MLPClassifier
    X, y = make_circles(300, noise=.08, factor=.45, random_state=0)
    f, axs = half(2, height=3.55)
    for ax, act, title in zip(axs, ['identity', 'relu'], ['3 layers, no activation', '3 layers, ReLU']):
        m = MLPClassifier((16, 16, 16), activation=act, max_iter=3000, random_state=0).fit(X, y)
        regions(ax, m, (-1.5, 1.5), (-1.5, 1.5))
        points(ax, X, y, s=16)
        ax.set_aspect('equal')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f'{title}\ntraining accuracy {m.score(X, y):.2f}', fontsize=13.5)
        clean(ax, 'Feature 1', 'Feature 2' if act == 'identity' else None, grid=False)
    save('nn1_linear_collapse', f, 'Synthetic circles (300 points, noise 0.08, seed 0); scikit-learn '
         'MLPClassifier (16, 16, 16), random_state 0.', kind='computed')


@fig
def gelu_silu():
    x = np.linspace(-4, 3, 500)
    relu = np.maximum(x, 0)
    gelu = x * .5 * (1 + erf(x / np.sqrt(2)))
    silu = x * expit(x)
    f, axs = half()
    ax = axs[0]
    ax.plot(x, relu, color=GREY, lw=2, ls='--')
    ax.plot(x, gelu, color=BLUE, lw=2.6)
    ax.plot(x, silu, color=TEAL, lw=2.6)
    ax.axhline(0, color=FAINT, lw=.8)
    ax.axvline(0, color=FAINT, lw=.8)
    endlabel(ax, 3, relu[-1], 'ReLU', GREY, dy=10)
    endlabel(ax, 3, gelu[-1], 'GELU', BLUE, dy=-4)
    endlabel(ax, 3, silu[-1], 'SiLU', TEAL, dy=-14)
    i_g, i_s = np.argmin(gelu), np.argmin(silu)
    ax.annotate(f'GELU min {gelu[i_g]:.2f}'.replace('-', '−'), (x[i_g], gelu[i_g]), xytext=(.35, -.62),
                fontsize=12.5, color=BLUE, arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1))
    ax.annotate(f'SiLU min {silu[i_s]:.2f}'.replace('-', '−'), (x[i_s], silu[i_s]), xytext=(-3.85, -.72),
                fontsize=12.5, color=TEAL, arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1))
    ax.set_xlim(-4, 3.9)
    ax.set_ylim(-.9, 3.2)
    clean(ax, 'Input z', 'Output')
    save('nn1_gelu_silu', f, f'Exact functions: GELU(z) = z Phi(z) (erf form), SiLU(z) = z sigmoid(z); minima '
         f'{gelu[i_g]:.3f} and {silu[i_s]:.3f}.', kind='computed')


def _load_shape(t):
    return (.55 + .25 * np.exp(-(t - 8) ** 2 / (2 * 1.6 ** 2)) + .4 * np.exp(-(t - 19) ** 2 / (2 * 2.2 ** 2))
            - .12 * np.exp(-(t - 3.5) ** 2 / (2 * 2.5 ** 2)))


@fig
def universal_approx():
    t = np.linspace(0, 24, 961)
    target = _load_shape(t)
    f, axs = sub(3, height=4.0)
    rmse = {}
    for ax, n in zip(axs, [2, 5, 20]):
        kinks = np.arange(n) * 24 / n
        Phi = np.c_[np.ones_like(t), np.maximum(t[:, None] - kinks[None, :], 0)]
        coef, *_ = np.linalg.lstsq(Phi, target, rcond=None)
        approx = Phi @ coef
        rmse[n] = float(np.sqrt(np.mean((approx - target) ** 2)))
        ax.plot(t, target, color=GREY, lw=4, alpha=.45)
        ax.plot(t, approx, color=BLUE, lw=2.3)
        ax.scatter(kinks, np.full(n, .08), marker='|', s=160, color=TEAL, lw=2, zorder=3)
        ax.set_title(f'{n} ReLU units   ·   RMSE {rmse[n]:.3f}', fontsize=14)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.25)
        ax.set_xticks([0, 6, 12, 18, 24])
        clean(ax, 'Hour of day', 'Load (per unit, synthetic)' if n == 2 else None)
    axs[0].text(23.5, .98, 'target', color=GREY, fontsize=12.5, ha='right')
    axs[0].text(14, .52, 'network', color=BLUE, fontsize=12.5)
    axs[0].text(.6, .17, 'kinks', color=TEAL, fontsize=12.5)
    save('nn1_universal_approx', f, f'Synthetic daily-load-shaped curve; one hidden layer of n ReLU units with '
         f'evenly spaced kinks (ticks), output weights by least squares; RMSE {rmse}.', kind='computed')


def _spirals(n=400, noise=.08, seed=0, turns=1.5):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, n)) * turns * 2 * np.pi
    X0 = np.c_[t * np.cos(t), t * np.sin(t)]
    X = np.r_[X0, -X0] / (turns * 2 * np.pi) + rng.normal(0, noise, (2 * n, 2))
    return X, np.r_[np.zeros(n), np.ones(n)].astype(int)


@fig
def depth_width():
    import warnings
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    X, y = _spirals()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    configs = [(2,), (16,), (256,), (16, 16, 16)]
    f, axs = sub(4, height=3.75)
    info = {}
    for ax, h in zip(axs, configs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MLPClassifier(h, max_iter=5000, random_state=0).fit(Xtr, ytr)
        sizes = [2, *h, 1]
        n_par = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
        acc = m.score(Xva, yva)
        info[str(h)] = (n_par, round(acc, 3), m.n_iter_)
        regions(ax, m, (-1.35, 1.35), (-1.35, 1.35))
        points(ax, Xtr, ytr, s=9)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{len(h)} × {h[0]} units · {n_par:,} params\nvalidation accuracy {acc:.2f}', fontsize=13)
    save('nn1_depth_width', f, f'Synthetic two spirals (800 points, noise 0.08, seed 0), 70/30 split; '
         f'scikit-learn MLPClassifier defaults (ReLU, Adam), random_state 0; (params, val acc, iterations) {info}.',
         kind='computed')


@fig
def forward_pass():
    f, ax = canvas(6.3, 4.9)
    x1, x2, h1, h2, zo = (.6, 3.55), (.6, 1.65), (2.85, 3.55), (2.85, 1.65), (5.05, 2.6)
    node(ax, *x1, r=.36, color=GREY, text='1', fs=16)
    node(ax, *x2, r=.36, color=GREY, text='2', fs=16)
    label(ax, .6, 4.15, '$x_1$', fs=15, color=MUTED)
    label(ax, .6, 1.05, '$x_2$', fs=15, color=MUTED)
    node(ax, *h1, r=.4, color=TEAL, text='2.5', fs=15)
    node(ax, *h2, r=.4, color=TEAL, text='1.0', fs=15)
    label(ax, 2.85, 4.2, '$h_1$  (b = 0)', fs=13.5, color=MUTED)
    label(ax, 2.85, 1.0, '$h_2$  (b = 1)', fs=13.5, color=MUTED)
    for a, b_, w, pos in [(x1, h1, '0.5', (1.72, 3.75)), (x2, h1, '1.0', (1.92, 3.12)),
                          (x1, h2, '−1.0', (1.92, 2.08)), (x2, h2, '0.5', (1.72, 1.45))]:
        arrow(ax, (a[0] + .37, a[1] + (b_[1] - a[1]) * .15), (b_[0] - .42, b_[1] - (b_[1] - a[1]) * .12),
              color=BLUE, lw=1.6, head=11)
        label(ax, *pos, w, fs=13.5, color=BLUE, weight='semibold')
    node(ax, *zo, r=.46, color=ORANGE, text='0.9', fs=15)
    label(ax, 5.05, 3.3, 'logit $z$  (c = 0.1)', fs=13.5, color=MUTED)
    for a, w, pos in [(h1, '0.4', (4.0, 3.3)), (h2, '−0.2', (4.0, 1.95))]:
        arrow(ax, (a[0] + .41, a[1] + (zo[1] - a[1]) * .12), (zo[0] - .47, zo[1] + (a[1] - zo[1]) * .12),
              color=BLUE, lw=1.6, head=11)
        label(ax, *pos, w, fs=13.5, color=BLUE, weight='semibold')
    _rect(ax, .2, .12, 5.95, .62, fc=PANEL, r=.08)
    label(ax, 3.17, .43, '$p = \\sigma(0.9) = 0.711$   ·   ReLU keeps both hidden values', fs=13.5)
    save('nn1_forward_pass', f, 'Exact arithmetic: h = ReLU((2.5, 1.0)), z = 0.4*2.5 - 0.2*1.0 + 0.1 = 0.9, '
         'sigmoid(0.9) = 0.7109.', kind='computed')


# ============================================================ Part 5: losses and gradient descent
@fig
def regression_losses():
    import warnings
    from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
    f, axs = sub(2, height=4.2)
    ax = axs[0]
    r = np.linspace(-3, 3, 400)
    hub = np.where(np.abs(r) <= 1, .5 * r ** 2, np.abs(r) - .5)
    ax.plot(r, r ** 2, color=BLUE, lw=2.6)
    ax.plot(r, np.abs(r), color=ORANGE, lw=2.6)
    ax.plot(r, hub, color=TEAL, lw=2.6)
    ax.text(2.2, 4.35, 'MSE  $r^2$', color=BLUE, fontsize=13.5, ha='left', va='center')
    ax.text(3.08, 3.0, 'MAE  $|r|$', color=ORANGE, fontsize=13.5, ha='left', va='center')
    ax.text(3.08, 2.42, 'Huber, $\\delta = 1$', color=TEAL, fontsize=13.5, ha='left', va='center')
    ax.set_ylim(0, 4.6)
    ax.set_xlim(-3, 4.6)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_title('Loss for one residual', fontsize=14.5)
    clean(ax, 'Residual $r = y - \\hat{y}$ (observed − predicted)', 'Loss')
    ax = axs[1]
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 10, 40))
    y = 2 + .8 * x + rng.normal(0, .6, 40)
    out = np.array([30, 33, 36, 38])
    y[out] += np.array([7, 8.5, 9, 7.5])
    X = x[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fits = {'MSE': LinearRegression().fit(X, y),
                'MAE': QuantileRegressor(quantile=.5, alpha=0, solver='highs').fit(X, y),
                'Huber': HuberRegressor().fit(X, y)}
    ok = np.ones(len(x), bool)
    ok[out] = False
    ax.scatter(x[ok], y[ok], s=34, color=GREY, zorder=3, edgecolors='white', lw=.5)
    ax.scatter(x[~ok], y[~ok], s=60, color=RED, marker='D', zorder=3, edgecolors='white', lw=.5)
    xx = np.linspace(0, 10, 50)
    slopes = {}
    for (name, m), col in zip(fits.items(), [BLUE, ORANGE, TEAL]):
        # MAE and Huber nearly coincide: draw MAE dashed on top so both lines stay visible
        ax.plot(xx, m.predict(xx[:, None]), color=col, lw=2.4, ls=(0, (4, 2.5)) if name == 'MAE' else '-',
                zorder=2.6 if name == 'MAE' else 2.4)
        slopes[name] = float(m.coef_[0])
    ax.text(7.35, 17.6, '4 gauge spikes', color=RED, fontsize=12.5, ha='right')
    ax.text(10, fits['MSE'].predict([[10]])[0] + .55, 'MSE', color=BLUE, fontsize=13, ha='right')
    # Huber ends just above MAE: label Huber above its line end and MAE below its own
    ax.text(10.2, fits['Huber'].predict([[10]])[0] + .45, 'Huber', color=TEAL, fontsize=13, ha='left', va='bottom')
    ax.text(10.2, fits['MAE'].predict([[10]])[0] - .45, 'MAE', color=ORANGE, fontsize=13, ha='left', va='top')
    ax.set_xlim(0, 11.6)
    ax.set_title('Straight-line fits with outliers', fontsize=14.5)
    clean(ax, 'Pressure drop (scaled)', 'Flow rate (scaled)')
    save('nn1_regression_losses', f, f'Left: exact loss functions. Right: synthetic line (slope 0.8, noise sd 0.6, '
         f'seed 5) with 4 added spikes; fitted slopes {({k: round(v, 3) for k, v in slopes.items()})}.',
         kind='computed')


@fig
def bce_curves():
    """Binary cross-entropy against the predicted probability, in the deck's class colours (0 BLUE, 1 ORANGE)."""
    p = np.linspace(.003, .997, 600)
    f, axs = sub(1)
    ax = axs[0]
    ax.plot(p, -np.log(1 - p), color=BLUE, lw=2.6)
    ax.plot(p, -np.log(p), color=ORANGE, lw=2.6)
    ax.text(.035, 4.6, 'true label $y$ = 1:  $-\\ln p$', color=ORANGE, fontsize=14, ha='left', va='center',
            fontweight='medium')
    ax.text(.965, 4.6, 'true label $y$ = 0:  $-\\ln(1 - p)$', color=BLUE, fontsize=14, ha='right', va='center',
            fontweight='medium')
    # the worked example from the previous slide: y = 1, p = 0.711
    pw = expit(.9)
    ax.scatter([pw], [-np.log(pw)], s=70, color=ORANGE, edgecolors=INK, lw=1.2, zorder=4)
    ax.annotate(f'worked example\n$p$ = {pw:.3f}, $L$ = {-np.log(pw):.3f}', (pw, -np.log(pw)),
                xytext=(.865, .76), fontsize=13, color=INK, ha='center', va='center', linespacing=1.25,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkA=4, shrinkB=5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.2)
    clean(ax, 'Predicted event probability $p$', 'Binary cross-entropy')
    save('nn1_bce_curves', f, f'Exact binary cross-entropy: -ln p for true label 1 and -ln(1 - p) for true label 0; '
         f'marked point p = {pw:.4f}, L = {-np.log(pw):.4f}.', kind='computed')


@fig
def softmax_temperature():
    from scipy.special import softmax as _softmax
    z = np.array([2.0, 1.0, .1])
    names = ['Sandstone', 'Shale', 'Limestone']
    cols = [BLUE, ORANGE, TEAL]
    f, axs = sub(4, height=3.9)
    ax = axs[0]
    ax.bar(range(3), z, color=cols, width=.62)
    for k, v in enumerate(z):
        ax.text(k, v + .06, f'{v:.1f}', ha='center', fontsize=13.5, color=INK)
    ax.set_ylim(0, 2.45)
    ax.set_title('Logits $z$', fontsize=14.5)
    ax.set_xticks(range(3), names, fontsize=12.5)
    clean(ax, None, 'Logit', grid=False)
    probs = {}
    for ax, T in zip(axs[1:], [.5, 1, 2]):
        p = _softmax(z / T)
        probs[T] = p.round(3).tolist()
        ax.bar(range(3), p, color=cols, width=.62)
        for k, v in enumerate(p):
            ax.text(k, v + .02, f'{v:.2f}', ha='center', fontsize=13.5, color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Softmax, $T$ = {T:g}' + (' (standard)' if T == 1 else ''), fontsize=14.5)
        ax.set_xticks(range(3), names, fontsize=12.5)
        clean(ax, None, 'Probability' if T == .5 else None, grid=False)
    save('nn1_softmax', f, f'Exact softmax of logits (2.0, 1.0, 0.1) divided by temperature T: {probs}.',
         kind='computed')


def _gas_lock_data(n=200, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = (rng.uniform(size=n) < expit(2.0 * x - .5)).astype(float)
    return x, y


def _lr_loss(w, b, x, y):
    z = np.multiply.outer(w, x) + np.asarray(b)[..., None]
    return np.mean(np.logaddexp(0, z) - y * z, axis=-1)


def _lr_grad(w, b, x, y):
    p = expit(w * x + b)
    return np.mean((p - y) * x), np.mean(p - y)


def _loss_contours(ax, x, y, wlim=(-2, 3.5), blim=(-2.5, 2.8)):
    W, Bm = np.meshgrid(np.linspace(*wlim, 220), np.linspace(*blim, 220))
    L = _lr_loss(W, Bm, x, y)
    levels = [.5, .55, .62, .7, .8, .95, 1.15, 1.4, 1.7, 2.1]
    ax.contour(W, Bm, L, levels=levels, cmap='course_seq', linewidths=1.3, vmin=0, vmax=2.6)
    from scipy.optimize import minimize
    opt = minimize(lambda v: _lr_loss(np.array(v[0]), np.array(v[1]), x, y), [0, 0]).x
    ax.scatter(*opt, marker='*', s=260, color=INK, zorder=5, edgecolors='white', lw=.8)   # the minimum
    ax.set_xlim(*wlim)
    ax.set_ylim(*blim)
    return opt


@fig
def gd_surface():
    x, y = _gas_lock_data()
    f, axs = half()
    ax = axs[0]
    opt = _loss_contours(ax, x, y)
    w, b, lr = -1.5, 2.0, 1.0
    path = [(w, b)]
    for _ in range(40):
        gw, gb = _lr_grad(w, b, x, y)
        w, b = w - lr * gw, b - lr * gb
        path.append((w, b))
    P = np.array(path)
    ax.plot(P[:, 0], P[:, 1], '-o', color=ORANGE, ms=4.5, lw=1.8, zorder=4)
    gw, gb = _lr_grad(-1.5, 2.0, x, y)
    ax.annotate('', (-1.5 - 1.2 * gw, 2.0 - 1.2 * gb), (-1.5, 2.0),
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=2.2, mutation_scale=16), zorder=6)
    bb = dict(boxstyle='round,pad=0.15', fc='white', ec='none')
    ax.text(-1.45, 2.35, 'start', fontsize=13, color=ORANGE, bbox=bb)
    ax.text(-1.9, 1.05, '− gradient', fontsize=13, color=PURPLE, bbox=bb)
    ax.text(opt[0] + .12, opt[1] - .45, 'minimum', fontsize=13, color=INK, bbox=bb)
    clean(ax, 'Weight $w$', 'Bias $b$', grid=False)
    L_end = float(_lr_loss(np.array(P[-1, 0]), np.array(P[-1, 1]), x, y))
    save('nn1_gd_surface', f, f'Mean binary cross-entropy of a one-feature logistic model on synthetic data '
         f'(200 points, seed 7); full-batch gradient descent, learning rate 1.0, 40 steps from (-1.5, 2.0); '
         f'optimum ({opt[0]:.2f}, {opt[1]:.2f}); final loss {L_end:.4f}.', kind='computed')


@fig
def batch_modes():
    x, y = _gas_lock_data()
    N = len(x)
    f = plt.figure(figsize=(11.8, 4.1), layout='constrained')
    gs = f.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    axs = [f.add_subplot(gs[0, i]) for i in range(4)]
    lr, epochs = .5, 3
    curves = {}
    for ax, (bs, name, col) in zip(axs[:3], [(N, 'Full batch', BLUE), (16, 'Mini-batch of 16', TEAL),
                                             (1, 'One example', ORANGE)]):
        _loss_contours(ax, x, y)
        rng = np.random.default_rng(0)
        w, b = -1.5, 2.0
        path, losses, ep_axis = [(w, b)], [float(_lr_loss(np.array(w), np.array(b), x, y))], [0.]
        n_up = 0
        for ep in range(epochs):
            perm = rng.permutation(N)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                gw, gb = _lr_grad(w, b, x[idx], y[idx])
                w, b = w - lr * gw, b - lr * gb
                n_up += 1
                path.append((w, b))
                losses.append(float(_lr_loss(np.array(w), np.array(b), x, y)))
                ep_axis.append(ep + min(i + bs, N) / N)
        P = np.array(path)
        ax.plot(P[:, 0], P[:, 1], '-' if bs == 1 else '-o', color=col, ms=4, lw=1.1 if bs == 1 else 1.8, zorder=4)
        ax.set_title(f'{name}\n{n_up} updates', fontsize=13.5)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        clean(ax, 'Weight $w$', 'Bias $b$' if bs == N else None, grid=False)
        curves[name] = (np.array(ep_axis), np.array(losses), col, n_up)
    ax = axs[3]
    for name, (e, l, col, n_up) in curves.items():
        ax.plot(e, l, color=col, lw=1.2 if n_up > 100 else 2.2, marker='o' if n_up < 10 else None, ms=5)
    ax.axhline(float(_lr_loss(np.array(1.7904), np.array(-.5316), x, y)), color=GREY, ls=':', lw=1.6)
    ax.text(1.5, 1.86, 'full batch', color=BLUE, fontsize=12.5)
    ax.text(.55, 1.2, 'mini-batch', color=TEAL, fontsize=12.5)
    ax.text(1.55, .7, 'one example', color=ORANGE, fontsize=12.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(.35, 2.4)
    ax.set_title('Loss on all 200 rows', fontsize=13.5)
    clean(ax, 'Epochs (passes over the data)', 'Mean BCE')
    save('nn1_batch_modes', f, 'Same synthetic data and start as the gradient-descent figure; 3 epochs, '
         'learning rate 0.5, shuffling seed 0. Dotted line: minimum loss.', kind='computed')


@fig
def lr_preview():
    f, axs = sub(4, height=3.7)
    ww = np.linspace(-4, 4, 200)
    settings = [(.05, 'slow'), (.45, 'fast'), (.9, 'oscillates'), (1.05, 'diverges')]
    for ax, (eta, tag) in zip(axs, settings):
        ax.plot(ww, ww ** 2, color=FAINT, lw=2)
        w = 2.0
        pts = [w]
        for _ in range(6):
            w = w - eta * 2 * w
            pts.append(w)
        pts = np.array(pts)
        ax.plot(pts, pts ** 2, '-o', color=ORANGE if eta > 1 else (BLUE if eta < .5 else TEAL), ms=6, lw=1.6,
                zorder=3)
        ax.scatter(pts[:1], pts[:1] ** 2, s=90, color=INK, zorder=4)
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 13.5)
        ax.set_title(f'$\\eta$ = {eta:g}: {tag}\n' + f'after 6 steps $w$ = {pts[-1]:.2f}'.replace('-', '−'),
                     fontsize=14)
        clean(ax, 'Weight $w$', 'Loss $w^2$' if eta == .05 else None)
    save('nn1_lr_preview', f, 'Exact gradient descent on L(w) = w squared from w = 2, update '
         'w <- w(1 - 2 eta), six steps per panel.', kind='computed')


# ============================================================ Part 6: backpropagation
def _worked_net():
    """The worked 2-2-1 example: x = (1, 2), y = 1; returns forward values and exact gradients."""
    x = np.array([1., 2.])
    W1 = np.array([[.5, 1.], [-1., .5]])
    b1 = np.array([0., 1.])
    v = np.array([.4, -.2])
    c = .1
    u = W1 @ x + b1
    h = np.maximum(u, 0)
    z = v @ h + c
    p = expit(z)
    L = -np.log(p)
    dz = p - 1
    dh = dz * v
    du = dh * (u > 0)
    return dict(x=x, u=u, h=h, z=z, p=p, L=L, dp=-1 / p, dz=dz, dv=dz * h, dc=dz, dh=dh, du=du,
                dW1=np.outer(du, x), db1=du)


def _fmt(v):
    return f'{v:.3f}'.replace('-', '−') if abs(v) < 10 else f'{v:.1f}'


def _graph(mode):
    n = _worked_net()
    f, ax = canvas(11.8, 4.3)
    back = mode == 'backward'
    X_IN, X_U, X_H, X_Z, X_P, X_L = .55, 2.75, 4.65, 6.75, 8.5, 10.3
    Y1, Y2, YM = 2.9, 1.3, 2.1          # rows leave room for value, gradient and chip labels
    nodes = {'x1': (X_IN, Y1), 'x2': (X_IN, Y2), 'u1': (X_U, Y1), 'u2': (X_U, Y2), 'h1': (X_H, Y1),
             'h2': (X_H, Y2), 'z': (X_Z, YM), 'p': (X_P, YM), 'L': (X_L, YM)}
    edges = [('x1', 'u1'), ('x2', 'u1'), ('x1', 'u2'), ('x2', 'u2'), ('u1', 'h1'), ('u2', 'h2'),
             ('h1', 'z'), ('h2', 'z'), ('z', 'p'), ('p', 'L')]
    r = .34
    for a, b in edges:
        pa, pb = np.array(nodes[a]), np.array(nodes[b])
        d = (pb - pa) / np.linalg.norm(pb - pa)
        arrow(ax, tuple(pa + d * r), tuple(pb - d * (r + .02)), color='#9AABBC' if back else INK, lw=1.4, head=11)
        if back and a not in ('x1', 'x2'):
            off = np.array([-d[1], d[0]]) * .16
            arrow(ax, tuple(pb - d * (r + .05) + off), tuple(pa + d * (r + .07) + off), color=PURPLE, lw=1.6,
                  head=11)
    style = {'x1': (GREY, '$x_1$'), 'x2': (GREY, '$x_2$'), 'u1': (TEAL, 'Σ'), 'u2': (TEAL, 'Σ'),
             'h1': (TEAL, 'ReLU'), 'h2': (TEAL, 'ReLU'), 'z': (ORANGE, 'Σ'), 'p': (ORANGE, 'σ'), 'L': (INK, 'L')}
    for k, (px, py) in nodes.items():
        col, txt = style[k]
        node(ax, px, py, r=r, color=col, fill='solid' if k == 'L' else 'tint', text=txt,
             fs=12 if txt == 'ReLU' else 16)
    fwd = {'x1': ('', 1.0), 'x2': ('', 2.0), 'u1': ('u_1', n['u'][0]), 'u2': ('u_2', n['u'][1]),
           'h1': ('h_1', n['h'][0]), 'h2': ('h_2', n['h'][1]), 'z': ('z', n['z']), 'p': ('p', n['p']),
           'L': ('L', n['L'])}
    grads = {'u1': ('u_1', n['du'][0]), 'u2': ('u_2', n['du'][1]), 'h1': ('h_1', n['dh'][0]),
             'h2': ('h_2', n['dh'][1]), 'z': ('z', n['dz']), 'p': ('p', n['dp']), 'L': ('L', 1.0)}
    for k, (px, py) in nodes.items():
        below = py < 1.5 or k in ('z', 'p', 'L')
        nm, val = fwd[k]
        num = f'{val:.3f}' if k in ('p', 'L') else (f'{val:g}' if k in ('x1', 'x2') else f'{val:.1f}')
        txt = (f'${nm} = $' if nm else '') + num.replace('-', '−')
        yv = py - r - .22 if below else py + r + .22
        label(ax, px, yv, txt, fs=13, color=MUTED if back else INK, weight='normal' if back else 'semibold')
        if back and k in grads:
            gn, gv = grads[k]
            yg = yv - .3 if below else yv + .3
            label(ax, px, yg, f'$∂L/∂{gn} = $' + _fmt(gv), fs=13, color=PURPLE, weight='semibold')
    # parameters
    if not back:
        chips = [(X_U, 3.95, 'w = (0.5, 1.0),  b = 0'), (X_U, .25, 'w = (−1.0, 0.5),  b = 1'),
                 (X_Z, 3.25, 'v = (0.4, −0.2),  c = 0.1')]
        for cx, cy, t in chips:
            chip(ax, cx, cy, t, BLUE, fs=12.5)
        chip(ax, X_L, 3.25, 'target y = 1', GREY, fs=12.5)
    else:
        dW, db, dv, dc = n['dW1'], n['db1'], n['dv'], n['dc']
        chips = [(X_U + .2, 4.14, f'∂L/∂w = ({_fmt(dW[0, 0])}, {_fmt(dW[0, 1])}),  ∂L/∂b = {_fmt(db[0])}'),
                 (X_U + .2, .1, f'∂L/∂w = ({_fmt(dW[1, 0])}, {_fmt(dW[1, 1])}),  ∂L/∂b = {_fmt(db[1])}'),
                 (X_Z + .3, 3.75, f'∂L/∂v = ({_fmt(dv[0])}, {_fmt(dv[1])})'),
                 (X_Z + .3, 3.35, f'∂L/∂c = {_fmt(dc)}')]
        for cx, cy, t in chips:
            chip(ax, cx, cy, t, PURPLE, fs=12.5)
    return f, n


@fig
def graph_forward():
    f, n = _graph('forward')
    save('nn1_graph_forward', f, f'Exact forward pass of the worked network: u = (2.5, 1.0), h = (2.5, 1.0), '
         f'z = 0.9, p = {n["p"]:.4f}, L = {n["L"]:.4f} for y = 1.', kind='computed')


@fig
def graph_backward():
    f, n = _graph('backward')
    save('nn1_graph_backward', f, f'Exact reverse-mode gradients of the worked network: dL/dz = p - y = '
         f'{n["dz"]:.4f}; dL/dv = {np.round(n["dv"], 4).tolist()}; dL/dW1 = {np.round(n["dW1"], 4).tolist()}.',
         kind='computed')


@fig
def gradcheck():
    f, axs = half()
    ax = axs[0]
    eps = np.logspace(-12, -.5, 60)
    out = {}
    for dt, col, name in [(np.float64, BLUE, 'float64'), (np.float32, ORANGE, 'float32')]:
        h1 = dt(2.5)
        rest = dt(-.2 * 1.0 + .1)

        def loss(v1):
            z = v1 * h1 + rest
            return np.log1p(np.exp(-z)).astype(dt)          # BCE with y = 1

        v1 = dt(.4)
        exact = (expit(.9) - 1) * 2.5
        errs = []
        for e in eps:
            e_ = dt(e)
            g = (loss(v1 + e_) - loss(v1 - e_)) / (dt(2) * e_)
            errs.append(abs(float(g) - exact) / abs(exact))
        errs = np.maximum(np.array(errs), 1e-17)
        ax.loglog(eps, errs, color=col, lw=2.4)
        out[name] = (eps[np.argmin(errs)], errs.min())
    ax.text(3e-12, 3e-4, 'float32', color=ORANGE, fontsize=13.5, fontweight='medium')
    ax.text(2e-12, 2e-8, 'float64', color=BLUE, fontsize=13.5, fontweight='medium')
    ax.axvspan(1e-7, 1e-4, color=TEAL, alpha=.08, lw=0)
    ax.text(1.1e-6, 2e-14, 'useful\nrange\n(float64)', color=TEAL, fontsize=12.5, ha='center', va='bottom')
    ax.annotate('round-off\ndominates', (1e-10, 3e-6), xytext=(2e-12, 3e-13), fontsize=12.5, color=MUTED)
    ax.annotate('truncation\ndominates', (1e-2, 1e-5), xytext=(4e-3, 3e-12), fontsize=12.5, color=MUTED,
                ha='center')
    ax.set_ylim(1e-14, 10)
    clean(ax, 'Finite-difference step ε', 'Relative error vs backprop')
    save('nn1_gradcheck', f, f'Central differences for dL/dv1 of the worked network (exact value '
         f'{(expit(.9) - 1) * 2.5:.5f}); best (eps, error): '
         f'{ {k: (float(f"{a:.1e}"), float(f"{b:.1e}")) for k, (a, b) in out.items()} }.', kind='computed')


# ============================================================ Part 7: training in code
@fig
def boundary_evolution():
    import torch
    from torch import nn
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    X, y = make_moons(400, noise=.25, random_state=1)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.SGD(net.parameters(), lr=.1)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)[:, None]
    gen = torch.Generator().manual_seed(0)
    xx, yy = np.meshgrid(np.linspace(-1.7, 2.7, 200), np.linspace(-1.3, 1.8, 150))
    G = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    snaps, tr_curve, va_curve, accs = {}, [], [], {}
    for ep in range(1, 301):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            tr_curve.append(crit(net(Xt), yt).item())
            va_curve.append(crit(net(Xv), yv).item())
            if ep in (1, 20, 300):
                snaps[ep] = torch.sigmoid(net(G)).numpy().reshape(xx.shape)
                accs[ep] = ((net(Xv) > 0).float() == yv).float().mean().item()
    f = plt.figure(figsize=(11.8, 4.0), layout='constrained')
    gs = f.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35])
    for k, ep in enumerate((1, 20, 300)):
        ax = f.add_subplot(gs[0, k])
        ax.contourf(xx, yy, snaps[ep], levels=np.linspace(0, 1, 11), cmap=CMAP_DIV.reversed(), alpha=.38)
        ax.contour(xx, yy, snaps[ep], levels=[.5], colors=[INK], linewidths=1.4)
        points(ax, Xtr, ytr, s=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Epoch {ep}\nvalidation accuracy {accs[ep]:.2f}', fontsize=13.5)
    ax = f.add_subplot(gs[0, 3])
    ep_axis = np.arange(1, 301)
    ax.plot(ep_axis, tr_curve, color=BLUE, lw=2.2)
    ax.plot(ep_axis, va_curve, color=TEAL, lw=2.2)
    best = int(np.argmin(va_curve)) + 1
    ax.scatter([best], [va_curve[best - 1]], s=60, color=TEAL, zorder=4, edgecolors='white')
    ax.annotate(f'lowest validation\nloss, epoch {best}', (best, va_curve[best - 1]), xytext=(95, .47),
                fontsize=12.5, color=TEAL, arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1))
    for ep in (1, 20, 300):
        ax.axvline(ep, color=FAINT, lw=1, ls=':')
    ax.text(300, tr_curve[-1] - .045, 'training', color=BLUE, fontsize=13, ha='right', va='top')
    ax.text(300, max(va_curve[150:]) + .02, 'validation', color=TEAL, fontsize=13, ha='right', va='bottom')
    ax.set_xlim(0, 305)
    ax.set_ylim(0, .75)
    ax.set_title('Loss per epoch', fontsize=13.5)
    clean(ax, 'Epoch', 'Mean BCE')
    save('nn1_boundary_evolution', f, f'Synthetic moons (400 points, noise 0.25, seed 1), 70/30 split; PyTorch '
         f'2-16-16-1 ReLU network, SGD learning rate 0.1, batch 32, seeds 0; lowest validation loss at epoch '
         f'{best} ({min(va_curve):.4f}); final training {tr_curve[-1]:.4f}, validation {va_curve[-1]:.4f}.',
         kind='computed')


# ============================================================ Part 1 (evidence): learning curve
@fig
def small_data_curve():
    import warnings
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def data(n, seed):
        Xm, y = make_moons(n, noise=.3, random_state=seed)
        rng = np.random.default_rng(seed + 1000)
        return np.c_[Xm, rng.normal(0, 1, (n, 4))], y

    Xte, yte = data(4000, 999)
    sizes = np.array([25, 50, 100, 200, 400, 800, 1600])
    res = {'lr': [], 'mlp': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for n in sizes:
            lr, mlp = [], []
            for rep in range(8):
                X, y = data(int(n), rep)
                lr.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y).score(Xte, yte))
                mlp.append(make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=1000, random_state=rep))
                           .fit(X, y).score(Xte, yte))
            res['lr'].append((np.mean(lr), np.std(lr)))
            res['mlp'].append((np.mean(mlp), np.std(mlp)))
    f, axs = half()
    ax = axs[0]
    for key, col in [('lr', GREY), ('mlp', BLUE)]:
        m = np.array([v[0] for v in res[key]])
        s = np.array([v[1] for v in res[key]])
        ax.fill_between(sizes, m - s, m + s, color=col, alpha=.15, lw=0)
        ax.plot(sizes, m, '-o', color=col, lw=2.4, ms=6)
    ax.set_xscale('log')
    ax.set_xticks(sizes, [str(v) for v in sizes])
    ax.minorticks_off()
    lr_m = [v[0] for v in res['lr']]
    mlp_m = [v[0] for v in res['mlp']]
    ax.text(1650, mlp_m[-1] + .012, 'MLP (32, 32)', color=BLUE, fontsize=13.5, ha='right', va='bottom',
            fontweight='medium')
    ax.text(1650, lr_m[-1] - .014, 'logistic regression', color=GREY, fontsize=13.5, ha='right', va='top',
            fontweight='medium')
    cross = next(n for n, a, b in zip(sizes, lr_m, mlp_m) if b > a)
    ax.axvspan(20, cross / np.sqrt(2), color=GREY, alpha=.07, lw=0)
    ax.text(26, .905, 'baseline\nscores higher', color=MUTED, fontsize=12.5, va='top')
    lows = [v[0] - v[1] for key in res for v in res[key]]
    ax.set_xlim(20, 2000)
    ax.set_ylim(np.floor(min(lows) * 50) / 50 - .01, .92)
    clean(ax, 'Training rows (log scale)', 'Accuracy on 4,000 new rows')
    summary = {int(n): (round(a, 3), round(b, 3)) for n, a, b in zip(sizes, lr_m, mlp_m)}
    save('nn1_small_data_curve', f, 'Synthetic two moons (noise 0.3) plus 4 noise features; 8 resamples per size '
         '(seeds 0-7); scikit-learn pipelines with StandardScaler; lines: mean, bands: +/- 1 sd; '
         f'(logistic, MLP) mean accuracy by size {summary}.', kind='computed')
