"""Figures for module key 'nn4' (Neural Networks, Day 4: convolutional neural networks).

Figure names start with 'nn4_'. Computed figures use fixed seeds and run on the
CPU in seconds: small PyTorch models are trained on scikit-learn's bundled 8 x 8
digits, and energy-style images (seismic sections, casing thickness maps) are
synthetic and labelled as such in every caption.
"""
from contextlib import contextmanager
from functools import lru_cache

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom
from scipy.signal import correlate2d
from sklearn.datasets import load_digits, load_sample_image
from sklearn.model_selection import train_test_split

KEY = 'nn4'
fig = register(KEY)

GREY = '#56687A'            # data / neutral elements
FROZEN = '#8A9AAB'          # frozen (not trained) parameters


# ----------------------------------------------------------------- helpers
@contextmanager
def _torch_single_thread():
    """Tiny models train much faster on one thread; restore the setting after."""
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(n)


def _img(ax, a, x, y, w, h, cmap='gray_r', vmin=None, vmax=None, norm=None, edge=RULE, lw=1.0, z=2):
    """Draw array a as an image on a canvas axes, occupying the box (x, y, w, h) in inches."""
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    ax.imshow(a, extent=(x, x + w, y, y + h), cmap=cmap, interpolation='nearest', origin='upper',
              zorder=z, **kw)
    if edge:
        ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=edge, lw=lw, zorder=z + .1))


def _inch_axes(f, x, y, w, h):
    """Matplotlib axes placed at (x, y, w, h) inches inside figure f."""
    W, H = f.get_size_inches()
    return f.add_axes([x / W, y / H, w / W, h / H])


def _num(v) -> str:
    """Number with a typographic minus sign."""
    return f'{v:g}'.replace('-', '\u2212')


def _grid(ax, x, ytop, values, cell=.5, color=BLUE, cmap=None, vmin=None, vmax=None, fills=None,
          fs=13, text=True, edge='white', lw=1.2, labels=None, z=2, amount=.82):
    """Numeric grid with its top-left corner at (x, ytop); like figlib.matrix but with
    typographic minus signs, optional per-cell fill colours and per-cell text labels."""
    a = np.asarray(values, dtype=float)
    rows, cols = a.shape
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    cm = plt.get_cmap(cmap) if cmap else None
    for i in range(rows):
        for j in range(cols):
            dark = False
            if fills is not None and fills[i][j] is not None:
                fc = fills[i][j]
                dark = fc in (ORANGE, BLUE, TEAL, INK, PURPLE, RED)
            elif cm is not None:
                t = 0 if hi == lo else (a[i, j] - lo) / (hi - lo)
                fc = cm(.08 + .8 * t)
                dark = .08 + .8 * t > .55
            else:
                fc = tint(color, amount)
            ax.add_patch(Rectangle((x + j * cell, ytop - (i + 1) * cell), cell, cell, fc=fc, ec=edge,
                                   lw=lw, zorder=z))
            if text:
                t_ = labels[i][j] if labels is not None else _num(a[i, j])
                if t_:
                    ax.text(x + (j + .5) * cell, ytop - (i + .5) * cell, t_, ha='center', va='center',
                            fontsize=fs, zorder=z + 1, color='white' if dark else INK)
    return Anchor(c=(x + cols * cell / 2, ytop - rows * cell / 2), l=(x, ytop - rows * cell / 2),
                  r=(x + cols * cell, ytop - rows * cell / 2), t=(x + cols * cell / 2, ytop),
                  b=(x + cols * cell / 2, ytop - rows * cell), x=x, y=ytop - rows * cell,
                  w=cols * cell, h=rows * cell)


def _outline(ax, x, ytop, r0, c0, nr, nc, cell, color=ORANGE, lw=3, ls='-', z=5):
    ax.add_patch(Rectangle((x + c0 * cell, ytop - (r0 + nr) * cell), nc * cell, nr * cell, fc='none',
                           ec=color, lw=lw, ls=ls, zorder=z))


def _digits():
    d = load_digits()
    return d.images.astype(np.float32) / 16.0, d.target


@lru_cache(maxsize=None)
def _digits_split():
    X, y = _digits()
    return tuple(train_test_split(X, y, test_size=0.2, random_state=0, stratify=y))


def _tiny_cnn_class():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class TinyCNN(nn.Module):
        """conv(1->8) -> ReLU -> maxpool 2 -> conv(8->16) -> ReLU -> global average -> linear."""
        def __init__(self, n_out=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.head = nn.Linear(16, n_out)

        def features(self, x):
            a1 = F.relu(self.conv1(x))
            a2 = F.relu(self.conv2(F.max_pool2d(a1, 2)))
            return a1, a2

        def forward(self, x):
            return self.head(self.features(x)[1].mean(dim=(2, 3)))

    return TinyCNN


def _train_classifier(model, X, y, epochs=60, lr=0.01, seed=0, bs=64):
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X).unsqueeze(1), torch.from_numpy(np.asarray(y)).long()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model.eval()


def _predict_proba(model, X):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(np.asarray(X, np.float32)).unsqueeze(1)), 1).numpy()


@lru_cache(maxsize=None)
def _tiny_cnn():
    """Tiny CNN trained on the sklearn digits training split (seed 0, CPU)."""
    with _torch_single_thread() as torch:
        Xtr, Xte, ytr, yte = _digits_split()
        torch.manual_seed(0)
        m = _train_classifier(_tiny_cnn_class()(), Xtr, ytr)
        acc = float((_predict_proba(m, Xte).argmax(1) == yte).mean())
    return m, acc


def _ricker(f=25.0, dt=0.002, half=0.064):
    t = np.arange(-half, half + dt / 2, dt)
    a = (np.pi * f * t) ** 2
    return t, (1 - 2 * a) * np.exp(-a)


@lru_cache(maxsize=None)
def _seismic(nx=120, nt=250, dt=0.002, seed=7, throw=14):
    """Layered impedance model with a normal fault, its reflectivity, and a 25 Hz Ricker synthetic."""
    rng = np.random.default_rng(seed)
    n_if = 13
    tops = np.sort(rng.uniform(18, nt - 18, n_if))
    imp = np.round(rng.uniform(4.0, 9.0, n_if + 1), 2)
    ii, jj = np.meshgrid(np.arange(nt), np.arange(nx), indexing='ij')
    fault_x = 50 + 0.22 * ii                       # fault trace position for each time sample
    hanging = jj > fault_x                         # right-hand block moves down (normal fault)
    local_t = ii - 0.10 * jj - np.where(hanging, throw, 0)
    Z = imp[np.searchsorted(tops, local_t)]
    r = np.zeros_like(Z, dtype=float)
    r[1:] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    _, w = _ricker(25.0, dt)
    seis = np.stack([np.convolve(r[:, j], w, mode='same') for j in range(nx)], axis=1)
    mask = np.abs(jj - fault_x) <= 1.5
    return Z, r, seis, mask, dt


def _casing_map(kind, n_depth=240, n_az=72, seed=4):
    """Synthetic wall-thickness map (fraction of nominal); rows = depth (0.05 m), columns = azimuth (5 deg)."""
    rng = np.random.default_rng(seed)
    zz, aa = np.meshgrid(np.arange(n_depth), np.arange(n_az), indexing='ij')
    t = 1 + 0.012 * rng.standard_normal((n_depth, n_az))
    t += 0.025 * np.cos(2 * np.pi * aa / n_az + 0.7)          # mild azimuthal variation
    if kind == 'collar':
        t[(zz >= 96) & (zz < 108)] += 0.24
    elif kind == 'pits':
        for cz, ca, depth, s in [(50, 20, .30, 3.0), (120, 45, .22, 2.6), (128, 50, .18, 2.2),
                                 (185, 8, .26, 3.2), (200, 60, .15, 2.0)]:
            da = np.minimum(np.abs(aa - ca), n_az - np.abs(aa - ca))
            t -= depth * np.exp(-((zz - cz) ** 2 / (2 * (1.6 * s) ** 2) + da ** 2 / (2 * s ** 2)))
    elif kind == 'groove':
        da = np.minimum(np.abs(aa - 40), n_az - np.abs(aa - 40))
        t -= 0.16 * np.exp(-da ** 2 / (2 * 2.2 ** 2)) * ((zz > 30) & (zz < 215))
    return t


THICK_NORM = TwoSlopeNorm(vmin=0.65, vcenter=1.0, vmax=1.35)


# ------------------------------------------------------ motivating case
@fig
def casing_case():
    f, ax = canvas(11.8, 4.3)
    # --- schematic cylinder being unwrapped
    cx, rx, ry, yb, yt = 1.6, .78, .2, .55, 3.5
    ax.add_patch(Rectangle((cx - rx, yb), 2 * rx, yt - yb, fc=BLUE_T, ec='none', zorder=1))
    ax.add_patch(Ellipse((cx, yb), 2 * rx, 2 * ry, fc=BLUE_T, ec='none', zorder=1))   # front wall down to the rim
    ax.add_patch(Ellipse((cx, yt), 2 * rx, 2 * ry, fc=tint(BLUE, .75), ec=BLUE, lw=1.6, zorder=3))
    ax.add_patch(Ellipse((cx, yt), 2 * rx * .78, 2 * ry * .78, fc='white', ec=BLUE, lw=1.0, zorder=3))
    ax.add_patch(Arc((cx, yb), 2 * rx, 2 * ry, theta1=180, theta2=360, ec=BLUE, lw=1.6, zorder=3))
    ax.add_patch(Arc((cx, yb), 2 * rx, 2 * ry, theta1=0, theta2=180, ec=BLUE, lw=1.0, ls=':', zorder=3))
    ax.plot([cx - rx, cx - rx], [yb, yt], color=BLUE, lw=1.6, zorder=3)
    ax.plot([cx + rx, cx + rx], [yb, yt], color=BLUE, lw=1.6, zorder=3)
    for yy in (2.55, 2.68):                                      # collar band
        ax.add_patch(Arc((cx, yy), 2 * rx, 2 * ry, theta1=180, theta2=360, ec=BLUE, lw=1.1, zorder=3))
    ax.add_patch(Ellipse((cx + .3, 1.55), .34, .22, fc=ORANGE, ec='none', alpha=.9, zorder=3))
    ax.add_patch(Ellipse((cx - .32, 1.02), .22, .15, fc=ORANGE, ec='none', alpha=.75, zorder=3))
    ax.plot([cx, cx], [yb - ry, yt - ry], color=INK, lw=1.3, ls='--', zorder=4)
    label(ax, cx, yb - .45, 'cut at 0° and unroll', fs=12.5, color=MUTED)
    arrow(ax, (cx - rx + .05, yt - .5), (cx + rx - .05, yt - .5), color=INK, lw=1.5, rad=.28, head=13)
    label(ax, cx, yt + .45, 'azimuth 0–360°', fs=13)
    arrow(ax, (cx - rx - .32, yt - .1), (cx - rx - .32, yb + .2), color=INK, lw=1.4)
    label(ax, cx - rx - .52, (yt + yb) / 2, 'depth', fs=13, rotation=90)
    arrow(ax, (2.75, 2.05), (3.5, 2.05), color=MUTED, lw=2, head=18)
    # --- three synthetic intervals, depth downward, azimuth across
    titles = ['Collar band', 'Pitting', 'Wear groove']
    for k, (kind, title) in enumerate(zip(['collar', 'pits', 'groove'], titles)):
        x0 = 4.2 + k * 2.3
        a = _inch_axes(f, x0, .62, 1.75, 3.2)
        im = a.imshow(_casing_map(kind, seed=4 + k), cmap=CMAP_DIV, norm=THICK_NORM,
                      aspect='auto', interpolation='nearest', extent=(0, 360, 12, 0))
        a.set_xticks([0, 180, 360])
        a.set_yticks([0, 4, 8, 12] if k == 0 else [])
        a.set_title(title, fontsize=14)
        a.set_xlabel('Azimuth (°)', fontsize=12.5)
        if k == 0:
            a.set_ylabel('Depth below top (m)', fontsize=12.5)
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(RULE)
    cb_ax = _inch_axes(f, 11.05, .62, .16, 3.2)
    cb = f.colorbar(im, cax=cb_ax, ticks=[.7, .85, 1.0, 1.15, 1.3])
    cb.set_label('Wall thickness / nominal', fontsize=12.5)
    cb.outline.set_edgecolor(RULE)
    save('nn4_casing_case', f, 'Synthetic wall-thickness maps (seeds 4-6), 0.05 m depth rows by 5-degree '
         'azimuth sectors; schematic unwrapping. Not tool data.', kind='computed')


# ------------------------------------------------ images as tensors
@fig
def digit_grid():
    raw = load_digits().images[3].astype(int)                  # a handwritten "3", values 0-16
    f, ax = canvas(6.3, 4.9)
    cell = .5
    _grid(ax, .25, 4.45, raw, cell=cell, cmap='Greys', vmin=0, vmax=16, fs=12.5)
    for j in range(8):
        label(ax, .25 + (j + .5) * cell, 4.62, str(j), fs=11.5, color=MUTED)
        label(ax, .08, 4.45 - (j + .5) * cell, str(j), fs=11.5, color=MUTED)
    _outline(ax, .25, 4.45, 1, 4, 1, 1, cell)
    _img(ax, raw, 4.72, 3.05, 1.35, 1.35, cmap='Greys', vmin=0, vmax=16, edge=RULE)
    label(ax, 5.395, 4.6, 'rendered', fs=12.5, color=MUTED)
    label(ax, 4.62, 2.55, '8 × 8 array', fs=14, weight='semibold', ha='left')
    label(ax, 4.62, 2.17, '64 intensities', fs=13, ha='left')
    label(ax, 4.62, 1.83, 'values 0–16', fs=13, ha='left')
    label(ax, 4.62, 1.2, f'x[1, 4] = {raw[1, 4]}', fs=13.5, color=ORANGE, ha='left', weight='semibold')
    label(ax, 4.62, .86, 'row 1, column 4', fs=12, color=MUTED, ha='left')
    save('nn4_digit_grid', f, 'scikit-learn bundled handwritten digit (index 3, a "3"); exact 0-16 pixel values.',
         kind='data')


@fig
def rgb_channels():
    im = load_sample_image('china.jpg')[40:400, 20:380]           # 360 x 360 x 3 crop
    f, ax = canvas(11.8, 4.3)
    s, y0 = 2.55, 1.2
    _img(ax, im, .1, y0, s, s, cmap=None, edge=None)
    pts = [('A', 'roof', (61, 195), (-.2, .14)), ('B', 'sky', (20, 330), (-.22, -.1)),
           ('C', 'foliage', (310, 102), (.21, .13))]
    for key, name, (r, c), (dx, dy) in pts:
        px, py = .1 + (c + .5) / 360 * s, y0 + s - (r + .5) / 360 * s
        ax.add_patch(Circle((px, py), .07, fc='none', ec='white', lw=2.4, zorder=5))
        ax.add_patch(Circle((px, py), .07, fc='none', ec=INK, lw=1.0, zorder=5))
        ax.add_patch(Circle((px + dx, py + dy), .13, fc=INK, ec='white', lw=1.2, zorder=6))
        label(ax, px + dx, py + dy, key, fs=12, color='white', weight='bold', z=7)
    label(ax, .1 + s / 2, y0 + s + .25, 'Colour image', fs=14, weight='semibold')
    for k, (key, name, (r, c), _) in enumerate(pts):
        rgb = tuple(int(v) for v in im[r, c])
        label(ax, .1, .82 - k * .3, f'{key} {name}: (R, G, B) = {rgb}', fs=12, ha='left')
    chans = [('Red channel', '#B3261E'), ('Green channel', GREEN), ('Blue channel', BLUE)]
    xs = [3.3, 6.0, 8.7]
    w = s - .15
    # split into three channel arrays (they are stacked, not added)
    arrow(ax, (2.72, y0 + s / 2), (3.18, y0 + s / 2), color=MUTED, lw=1.8, head=14)
    label(ax, 2.95, y0 + s / 2 + .24, 'split', fs=12, color=MUTED)
    for k, ((title, col), x) in enumerate(zip(chans, xs)):
        _img(ax, im[:, :, k], x, y0, w, w, cmap='gray', vmin=0, vmax=255, edge=col, lw=2.2)
        label(ax, x + w / 2, y0 + w + .25, title, fs=14, weight='semibold', color=col)
    mid = (xs[0] + xs[2] + w) / 2
    bracket(ax, xs[0], xs[2] + w, y0 - .15, '', color=MUTED, below=True)
    label(ax, mid, y0 - .5, 'NumPy / PIL: H × W × C = 360 × 360 × 3, integers 0 (dark) to 255 (bright)', fs=13)
    label(ax, mid, y0 - .88, 'PyTorch: C × H × W = 3 × 360 × 360, usually rescaled to floats in [0, 1]',
          fs=13, color=BLUE)
    save('nn4_rgb_channels', f, 'Crop of scikit-learn sample image china.jpg (CC BY 2.0, danielbuechele on Flickr); '
         'exact 8-bit channel values.', kind='data')


@fig
def seismic_synthetic():
    Z, r, seis, _, dt = _seismic()
    nt, nx = Z.shape
    tmax = nt * dt * 1000
    f = plt.figure(figsize=(11.8, 4.3))
    a1 = _inch_axes(f, .75, .72, 3.3, 3.15)
    im1 = a1.imshow(Z, cmap=CMAP_SEQ, aspect='auto', extent=(0, nx, tmax, 0), interpolation='nearest')
    a1.set_title('Impedance model', fontsize=14)
    a1.set_xlabel('Trace')
    a1.set_ylabel('Two-way time (ms)')
    c1 = _inch_axes(f, 4.12, .72, .12, 3.15)
    cb = f.colorbar(im1, cax=c1, ticks=[5, 6, 7, 8])
    cb.set_label('Impedance (synthetic)', fontsize=12)
    cb.outline.set_edgecolor(RULE)
    a2 = _inch_axes(f, 5.55, .72, 1.05, 3.15)
    t, w = _ricker(25.0, dt)
    a2.plot(w, t * 1000, color=INK, lw=2)
    a2.fill_betweenx(t * 1000, 0, w, where=w > 0, color=BLUE, alpha=.35, lw=0)
    a2.set_ylim(80, -80)
    a2.set_xlim(-.6, 1.1)
    a2.set_xticks([0, 1])
    a2.set_yticks([-60, 0, 60])
    a2.set_title('Wavelet $w$', fontsize=14)
    a2.set_xlabel('Amplitude')
    a2.set_ylabel('Time (ms)', labelpad=2)
    a3 = _inch_axes(f, 7.35, .72, 3.75, 3.15)
    v = np.abs(seis).max() * .7
    im3 = a3.imshow(seis, cmap=CMAP_DIV, vmin=-v, vmax=v, aspect='auto', extent=(0, nx, tmax, 0),
                    interpolation='bilinear')
    a3.set_title(r'Section = reflectivity $\ast$ $w$', fontsize=14)
    a3.set_xlabel('Trace')
    a3.set_ylabel('Two-way time (ms)')
    c3 = _inch_axes(f, 11.18, .72, .12, 3.15)
    cb3 = f.colorbar(im3, cax=c3, ticks=[-v * .9, 0, v * .9])
    cb3.ax.set_yticklabels(['−', '0', '+'])
    cb3.set_label('Amplitude', fontsize=12)
    cb3.outline.set_edgecolor(RULE)
    for a in (a1, a3):
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(RULE)
    save('nn4_seismic_synthetic', f, 'Synthetic layered impedance model with a normal fault (seed 7); reflectivity '
         'convolved trace by trace with a 25 Hz Ricker wavelet, 2 ms sampling.', kind='computed')


# ------------------------------------------------ convolution mechanics
@fig
def weight_sharing():
    f, ax = canvas(6.3, 4.9)
    xs = [.55 + i * .87 for i in range(7)]
    wcol = [BLUE, TEAL, ORANGE]
    yi, yo = 2.95, 4.05                                             # dense
    for x in xs:
        for k in range(5):
            ax.plot([x, xs[k + 1]], [yi, yo], color='#B5C2CF', lw=.9, zorder=1)
    for x in xs:
        node(ax, x, yi, .16, GREY)
    for k in range(5):
        node(ax, xs[k + 1], yo, .16, INK, fill='white')
    label(ax, .2, 4.62, 'Dense: every output sees every input', fs=14, weight='semibold', ha='left')
    label(ax, 6.15, 4.62, '35 weights', fs=13.5, color=MUTED, ha='right')
    yi, yo = .45, 1.55                                              # convolution
    for k in range(5):
        for j in range(3):
            ax.plot([xs[k + j], xs[k + 1]], [yi, yo], color=wcol[j], lw=2.2, zorder=1)
    for x in xs:
        node(ax, x, yi, .16, GREY)
    for k in range(5):
        node(ax, xs[k + 1], yo, .16, INK, fill='white')
    label(ax, .2, 2.28, 'Convolution: 3 shared weights, reused', fs=14, weight='semibold', ha='left')
    for j, name in enumerate(['w₁', 'w₂', 'w₃']):
        chip(ax, 4.7 + j * .55, 1.95, name, wcol[j], fs=12, solid=True)
    label(ax, 6.15, 2.28, '3 weights', fs=13.5, color=MUTED, ha='right')
    ax.plot([.2, 6.1], [2.62, 2.62], color=RULE, lw=1)
    save('nn4_weight_sharing', f, 'Schematic: seven inputs, five outputs; a kernel of size 3 without padding.',
         kind='schematic')


@fig
def conv_numeric():
    a = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    k = np.array([[1, 0], [0, -1]])
    out = correlate2d(a, k, mode='valid')
    f, ax = canvas(11.8, 3.6)
    cell, cy = .8, 1.65                                             # cell size; common vertical centre
    xi, xk, xo = .9, 4.85, 8.55
    ti, tk = cy + 1.5 * cell, cy + cell                             # grid tops
    label(ax, xi + 1.5 * cell, 3.35, 'Input 3 × 3', fs=14, weight='semibold')
    _grid(ax, xi, ti, a, cell=cell, color=BLUE, fs=16)
    _outline(ax, xi, ti, 0, 0, 2, 2, cell, lw=3.2)
    label(ax, (xi + 3 * cell + xk) / 2, cy, r'$\star$', fs=26, color=MUTED)
    label(ax, xk + cell, 3.35, 'Kernel 2 × 2', fs=14, weight='semibold')
    _grid(ax, xk, tk, k, cell=cell, color=ORANGE, fs=16)
    arrow(ax, (xk + 2 * cell + .3, cy), (xo - .3, cy), color=MUTED, lw=1.6)
    label(ax, xo + cell, 3.35, 'Output 2 × 2', fs=14, weight='semibold')
    fills = [[ORANGE, 'white'], ['white', 'white']]
    labels_ = [[_num(out[0, 0]), ''], ['', '']]
    _grid(ax, xo, tk, out, cell=cell, fills=fills, labels=labels_, fs=16, edge='#9DAFC0', lw=1)
    for (i, j) in [(0, 1), (1, 0), (1, 1)]:
        ax.text(xo + (j + .5) * cell, tk - (i + .5) * cell, '?', ha='center', va='center', fontsize=16,
                color=FAINT, zorder=4)
    label(ax, xo + cell, tk - 2 * cell - .32, 'other three: your turn', fs=12.5, color=MUTED)
    save('nn4_conv_numeric', f, 'Exact cross-correlation (scipy.signal.correlate2d) of a 3 x 3 input with a 2 x 2 '
         'kernel, stride 1, no padding; only the first output is shown.', kind='computed')


@fig
def sliding_steps():
    a = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    k = np.array([[1, 0], [0, -1]])
    out = correlate2d(a, k, mode='valid')
    f, ax = canvas(11.8, 3.6)
    label(ax, .72, 2.95, 'Kernel', fs=14, weight='semibold')
    _grid(ax, .3, 2.7, k, cell=.42, color=ORANGE, fs=13)
    ax.plot([1.4, 1.4], [.25, 3.35], color=RULE, lw=1)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ci, co = .4, .34
    for s_, (i, j) in enumerate(positions):
        x0 = 1.68 + s_ * 2.55
        label(ax, x0, 3.35, f'Step {s_ + 1}', fs=13.5, weight='semibold', ha='left')
        _grid(ax, x0, 3.0, a, cell=ci, color=BLUE, fs=12.5)
        _outline(ax, x0, 3.0, i, j, 2, 2, ci)
        arrow(ax, (x0 + 3 * ci + .12, 2.4), (x0 + 3 * ci + .42, 2.4), color=MUTED, lw=1.4, head=11)
        ox, oy = x0 + 3 * ci + .52, 2.74
        fills, labels_ = [[None] * 2 for _ in range(2)], [[''] * 2 for _ in range(2)]
        for (ii, jj) in positions[:s_ + 1]:
            fills[ii][jj] = ORANGE if (ii, jj) == (i, j) else tint(BLUE, .8)
            labels_[ii][jj] = _num(out[ii, jj])
        fills = [[c if c is not None else 'white' for c in row] for row in fills]
        _grid(ax, ox, oy, out, cell=co, fills=fills, labels=labels_, fs=12, edge='#9DAFC0', lw=1)
        patch = a[i:i + 2, j:j + 2]
        terms = [f'{patch[0, 0]}·1', f'{patch[0, 1]}·0', f'{patch[1, 0]}·0', f'{patch[1, 1]}·(−1)']
        label(ax, x0, 1.3, ' + '.join(terms[:2]) + ' +', fs=12.5, ha='left')
        label(ax, x0, .95, terms[2] + ' + ' + terms[3], fs=12.5, ha='left')
        label(ax, x0, .5, '= ' + _num(out[i, j]), fs=14, ha='left', weight='semibold', color=ORANGE)
    save('nn4_sliding_steps', f, 'Exact cross-correlation of the 3 x 3 input with the 2 x 2 kernel (stride 1, no '
         'padding); same numbers as the conv_numeric figure.', kind='computed')


def _stack(ax, x, y, n, cell, colors, sheets, dx=.26, dy=.26, hl=None, hl_color=ORANGE, values=None,
           fs=11, z0=2, lw=1.0):
    """Stack of `sheets` n x n grids drawn back to front; (x, y) is the front sheet's lower-left corner.
    hl = (r0, c0, nr, nc) outlines the same window on every sheet. Returns the front sheet's anchors."""
    for s_ in reversed(range(sheets)):
        ox, oy = x + s_ * dx, y + s_ * dy
        col = colors[s_ % len(colors)]
        z = z0 + (sheets - s_) * .5
        ax.add_patch(Rectangle((ox, oy), n * cell, n * cell, fc='white', ec=shade(col, .1), lw=lw + .3, zorder=z))
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((ox + j * cell, oy + (n - 1 - i) * cell), cell, cell, fc=tint(col, .8),
                                       ec='white', lw=lw, zorder=z + .1))
                if values is not None and s_ == 0:
                    ax.text(ox + (j + .5) * cell, oy + (n - .5 - i) * cell, _num(values[i][j]), ha='center',
                            va='center', fontsize=fs, zorder=z + .2, color=INK)
        if hl is not None:
            r0, c0, nr, nc = hl
            ax.add_patch(Rectangle((ox + c0 * cell, oy + (n - r0 - nr) * cell), nc * cell, nr * cell, fc='none',
                                   ec=hl_color, lw=2.4 if s_ == 0 else 1.6, zorder=z + .3,
                                   alpha=1 if s_ == 0 else .75))
    return _anchors_box(x, y, n * cell, n * cell)


def _anchors_box(x, y, w, h):
    return Anchor(c=(x + w / 2, y + h / 2), l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h),
                  b=(x + w / 2, y), x=x, y=y, w=w, h=h)


@fig
def padding():
    a = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 2, 8, 1], [3, 1, 2, 4]])
    f, ax = canvas(6.3, 4.9)
    cell, x0, top = .52, .25, 4.25
    padded = np.zeros((6, 6))
    padded[1:5, 1:5] = a
    fills = [[('white' if (i in (0, 5) or j in (0, 5)) else tint(BLUE, .8)) for j in range(6)] for i in range(6)]
    # real values drawn by _grid; the padding zeros are drawn once, in a paler colour, below
    labels_ = [[('' if (i in (0, 5) or j in (0, 5)) else _num(padded[i, j])) for j in range(6)] for i in range(6)]
    _grid(ax, x0, top, padded, cell=cell, fills=fills, labels=labels_, fs=12.5, edge='#C7D3DF', lw=1)
    for i in range(6):
        for j in range(6):
            if i in (0, 5) or j in (0, 5):
                ax.text(x0 + (j + .5) * cell, top - (i + .5) * cell, '0', ha='center', va='center',
                        fontsize=12.5, color=FAINT, zorder=4)
    ax.add_patch(Rectangle((x0 + cell, top - 5 * cell), 4 * cell, 4 * cell, fc='none', ec=BLUE, lw=1.6, zorder=4))
    _outline(ax, x0, top, 0, 0, 3, 3, cell)
    label(ax, x0 + 3 * cell, top + .3, 'Input 4 × 4 with a ring of zeros', fs=13.5, weight='semibold')
    arrow(ax, (x0 + 6 * cell + .15, top - 3 * cell), (x0 + 6 * cell + .6, top - 3 * cell), color=MUTED, lw=1.6)
    oc, ox, otop = .4, 4.3, top - .6
    fills = [[ORANGE if (i, j) == (0, 0) else 'white' for j in range(4)] for i in range(4)]
    _grid(ax, ox, otop, np.zeros((4, 4)), cell=oc, fills=fills, text=False, edge='#9DAFC0', lw=1)
    label(ax, ox + 2 * oc, top + .3, 'Output 4 × 4', fs=13.5, weight='semibold')
    label(ax, .25, .72, 'K = 3, P = 1, S = 1:  (4 + 2·1 − 3)/1 + 1 = 4', fs=13, ha='left')
    label(ax, .25, .33, 'The corner window holds 5 padding zeros and 4 real values', fs=12.5, ha='left',
          color=ORANGE)
    save('nn4_padding', f, 'Schematic with exact output-size arithmetic; zero padding of width 1.', kind='schematic')


@fig
def stride():
    f, ax = canvas(6.3, 4.9)
    cell, x0, top = .55, .3, 3.95
    _grid(ax, x0, top, np.zeros((5, 5)), cell=cell, text=False, color=BLUE, edge='white')
    _outline(ax, x0, top, 0, 0, 3, 3, cell, color=ORANGE, lw=3.2)
    _outline(ax, x0, top, 0, 2, 3, 3, cell, color=TEAL, lw=2.6, ls='--')
    _outline(ax, x0, top, 2, 0, 3, 3, cell, color=PURPLE, lw=1.6, ls=':')
    arrow(ax, (x0 + 1.5 * cell, top + .25), (x0 + 3.5 * cell, top + .25), color=INK, lw=1.5, rad=-.35, head=12)
    label(ax, x0 + 2.5 * cell, top + .62, 'jump 2 columns', fs=12.5)
    label(ax, x0 + 2.5 * cell, top - 5 * cell - .3, 'Input 5 × 5', fs=13.5, weight='semibold')
    arrow(ax, (x0 + 5 * cell + .2, top - 2.5 * cell), (x0 + 5 * cell + .65, top - 2.5 * cell), color=MUTED, lw=1.6)
    oc, ox, otop = .55, 4.2, top - 1.1
    fills = [[ORANGE, TEAL], [PURPLE, 'white']]
    _grid(ax, ox, otop, np.zeros((2, 2)), cell=oc, fills=fills, text=False, edge='#9DAFC0', lw=1)
    label(ax, ox + oc, otop - 2 * oc - .3, 'Output 2 × 2', fs=13.5, weight='semibold')
    label(ax, .3, .55, r'K = 3, S = 2, P = 0:  $\lfloor$(5 − 3)/2$\rfloor$ + 1 = 2', fs=13, ha='left')
    save('nn4_stride', f, 'Schematic with exact output-size arithmetic; three of the four window positions shown.',
         kind='schematic')


@fig
def dilation():
    f, ax = canvas(6.3, 4.9)
    cell, x0, top = .45, .3, 4.3
    fills = [[(ORANGE if (i % 2 == 0 and j % 2 == 0 and i < 5 and j < 5) else None) for j in range(7)]
             for i in range(7)]
    _grid(ax, x0, top, np.zeros((7, 7)), cell=cell, fills=fills, text=False, color=BLUE, edge='white')
    _outline(ax, x0, top, 0, 0, 5, 5, cell, color=ORANGE, lw=1.8, ls='--')
    label(ax, x0 + 3.5 * cell, top + .25, 'Input 7 × 7, kernel taps in orange', fs=13, weight='semibold')
    arrow(ax, (x0 + 7 * cell + .15, top - 3.5 * cell), (x0 + 7 * cell + .6, top - 3.5 * cell), color=MUTED, lw=1.6)
    oc, ox, otop = .45, 4.35, top - .9
    fills = [[ORANGE if (i, j) == (0, 0) else 'white' for j in range(3)] for i in range(3)]
    _grid(ax, ox, otop, np.zeros((3, 3)), cell=oc, fills=fills, text=False, edge='#9DAFC0', lw=1)
    label(ax, ox + 1.5 * oc, otop - 3 * oc - .3, 'Output 3 × 3', fs=13.5, weight='semibold')
    label(ax, .3, .78, 'K = 3, D = 2: taps span D(K − 1) + 1 = 5 pixels', fs=13, ha='left')
    label(ax, .3, .4, 'Still 9 weights;  (7 − 5)/1 + 1 = 3 outputs per row', fs=13, ha='left')
    save('nn4_dilation', f, 'Schematic with exact output-size arithmetic; dilation 2, no padding, stride 1.',
         kind='schematic')


@fig
def multichannel():
    f, ax = canvas(11.8, 4.3)
    rgb = ['#B3261E', GREEN, BLUE]
    # left: one kernel over three channels -> one map
    label(ax, 2.75, 4.05, 'One kernel spans every input channel', fs=14.5, weight='semibold')
    _stack(ax, .25, .75, 5, .36, rgb, 3, dx=.3, dy=.3, hl=(0, 0, 3, 3))
    label(ax, 1.45, .4, r'Input  $C_\mathrm{in}$ × 5 × 5  ($C_\mathrm{in}$ = 3)', fs=12.5, color=MUTED)
    label(ax, 2.95, 2.35, r'$\star$', fs=20, color=MUTED)                  # cross-correlation, as in nn4_conv_numeric
    _stack(ax, 3.25, 1.55, 3, .3, [ORANGE, ORANGE, ORANGE], 3, dx=.24, dy=.24)
    label(ax, 3.95, 1.2, 'Kernel 3 × 3 × 3', fs=12.5, color=MUTED)
    arrow(ax, (4.6, 2.35), (5.05, 2.35), color=MUTED, lw=1.6)
    fills = [[ORANGE if (i, j) == (0, 0) else tint(TEAL, .8) for j in range(3)] for i in range(3)]
    _grid(ax, 5.15, 2.95, np.zeros((3, 3)), cell=.36, fills=fills, text=False, edge='white')
    label(ax, 5.69, 1.65, '1 map', fs=12.5, color=MUTED)
    label(ax, 2.75, .1, '27 weights + 1 bias, summed over all channels', fs=12.5, color=INK)
    ax.plot([6.35, 6.35], [.2, 4.1], color=RULE, lw=1)
    # right: C_out kernels -> C_out maps
    label(ax, 9.1, 4.05, r'$C_\mathrm{out}$ kernels make $C_\mathrm{out}$ maps', fs=14.5, weight='semibold')
    kcols = [BLUE, TEAL, PURPLE, GOLD]
    _stack(ax, 6.6, 1.5, 4, .3, rgb, 3, dx=.22, dy=.22)
    label(ax, 7.4, 1.05, 'Input 3 × H × W', fs=12.5, color=MUTED)
    for k, col in enumerate(kcols):
        y = 3.05 - k * .72
        _stack(ax, 8.55, y, 3, .16, [col] * 3, 3, dx=.1, dy=.1, lw=.6)
        arrow(ax, (8.0, 2.0), (8.5, y + .24), color=FAINT, lw=1.0, head=8)
        arrow(ax, (9.3, y + .24), (9.85, 2.25 + (1.5 - k) * .12), color=FAINT, lw=1.0, head=8)
    label(ax, 8.9, .45, '4 kernels', fs=12.5, color=MUTED)
    for k, col in enumerate(reversed(kcols)):
        off = (3 - k) * .17
        ax.add_patch(Rectangle((9.95 + off, 1.55 + off), 1.05, 1.05, fc=tint(col, .55), ec=shade(col, .15), lw=1.2,
                               zorder=2 + k))
    label(ax, 10.72, 1.2, 'Output 4 × H′ × W′', fs=12.5, color=MUTED)
    label(ax, 9.1, .1, r'weight shape ($C_\mathrm{out}$, $C_\mathrm{in}$, K, K) = (4, 3, 3, 3)', fs=12.5, color=INK)
    save('nn4_multichannel', f, 'Schematic of a Conv2d layer: each output map uses one C_in x K x K kernel plus a '
         'bias.', kind='schematic')


@fig
def one_by_one():
    f, ax = canvas(6.3, 4.9)
    cols4 = [BLUE, TEAL, PURPLE, GOLD]
    _stack(ax, .2, 1.3, 4, .34, cols4, 4, dx=.2, dy=.2, hl=(1, 2, 1, 1))
    label(ax, 1.18, .95, 'Input 4 × 4 × 4', fs=12.5, color=MUTED)
    pos = network(ax, [4, 2], 2.75, 3.85, 1.85, 3.35, r=.15, colors=[GREY, ORANGE], edge_color='#9DAFC0')
    arrow(ax, (2.35, 2.6), (2.58, 2.6), color=MUTED, lw=1.4, head=10)
    arrow(ax, (4.05, 2.6), (4.35, 2.6), color=MUTED, lw=1.4, head=10)
    label(ax, 3.3, 3.75, '4 → 2 channels', fs=12.5, color=INK)
    _stack(ax, 4.45, 1.7, 4, .3, [ORANGE, TEAL], 2, dx=.2, dy=.2, hl=(1, 2, 1, 1))
    label(ax, 5.25, 1.35, 'Output 2 × 4 × 4', fs=12.5, color=MUTED)
    label(ax, .2, 4.55, 'The same small dense layer at every pixel', fs=14, weight='semibold', ha='left')
    label(ax, .2, .5, 'Parameters: 2 × (4 + 1) = 10;  H and W unchanged', fs=13, ha='left')
    save('nn4_one_by_one', f, 'Schematic of a 1 x 1 convolution mapping 4 channels to 2 at each pixel.',
         kind='schematic')


@fig
def sobel_maps():
    a = load_digits().images[3] / 16.0                              # the digit of the grid slide
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx, gy = correlate2d(a, kx, mode='same'), correlate2d(a, kx.T, mode='same')
    v = max(np.abs(gx).max(), np.abs(gy).max())                     # one scale, centred on zero
    f = plt.figure(figsize=(11.8, 4.3))
    size, xs = 3.05, [.35, 4.05, 7.75]
    panels = [(a, 'Original digit', 'Greys', 0, 1), (gx, 'Horizontal derivative', CMAP_DIV, -v, v),
              (gy, 'Vertical derivative', CMAP_DIV, -v, v)]
    for (arr, title, cmap, lo, hi), x in zip(panels, xs):
        a_ = _inch_axes(f, x, .3, size, size)
        im = a_.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(True); sp.set_color(RULE)
        a_.set_title(title, fontsize=14, pad=8)
    cax = _inch_axes(f, 11.0, .3, .15, size)
    cb = f.colorbar(im, cax=cax, ticks=[-.9 * v, 0, .9 * v])
    cb.ax.set_yticklabels(['−', '0', '+'])
    cb.set_label('Response (one scale for both maps)', fontsize=12)
    cb.outline.set_edgecolor(RULE)
    save('nn4_sobel_maps', f, 'Fixed Sobel x and y kernels applied with zero padding (scipy.signal.correlate2d) to '
         'scikit-learn digit index 3; one colour scale centred on zero; hand-specified, not learned.',
         kind='computed')


@fig
def learned_filters():
    m, acc = _tiny_cnn()
    w = m.conv1.weight.detach().numpy()[:, 0]                       # 8 x 3 x 3
    f, axs = half(4, rows=2, height=4.6)
    for k, a in enumerate(axs):
        v = np.abs(w[k]).max()
        im = a.imshow(w[k] / v, cmap=CMAP_DIV, vmin=-1, vmax=1, interpolation='nearest')
        a.set_title(f'Filter {k + 1}', fontsize=13)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(RULE)
    cb = f.colorbar(im, ax=axs, shrink=.8, pad=.03, ticks=[-.9, 0, .9])
    cb.ax.set_yticklabels(['−', '0', '+'])
    cb.set_label('Weight (each filter scaled to its maximum)', fontsize=12)
    cb.outline.set_edgecolor(RULE)
    save('nn4_learned_filters', f, f'First-layer 3 x 3 kernels of a tiny CNN trained on scikit-learn digits (1,437 '
         f'training images, seed 0, CPU); held-out accuracy {acc:.1%}.', kind='computed')


@fig
def feature_maps():
    import torch
    m, acc = _tiny_cnn()
    Xtr, Xte, ytr, yte = _digits_split()
    picks = [int(np.where(yte == 0)[0][0]), int(np.where(yte == 7)[0][0])]
    with torch.no_grad():
        a1, _ = m.features(torch.from_numpy(Xte[picks]).unsqueeze(1))
    maps = a1.numpy()                                               # 2 x 8 x 8 x 8
    w = m.conv1.weight.detach().numpy()[:, 0]
    f = plt.figure(figsize=(11.8, 4.3))
    size, gap, x0 = 1.08, .14, 2.05

    def frame(a):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(RULE)

    for k in range(8):
        a = _inch_axes(f, x0 + k * (size + gap) + .24, 3.35, .6, .6)
        a.imshow(w[k] / np.abs(w[k]).max(), cmap=CMAP_DIV, vmin=-1, vmax=1, interpolation='nearest')
        frame(a)
        a.set_title(f'Filter {k + 1}', fontsize=12, pad=3)
    for r, idx in enumerate(picks):
        y = 1.85 - r * 1.5
        a = _inch_axes(f, .3, y, size, size)
        a.imshow(Xte[idx], cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        frame(a)
        a.set_title(f'input: digit {yte[idx]}', fontsize=12, pad=3)
        f.text(1.66, y + .54, '→', fontsize=20, color=MUTED, ha='center', va='center', transform=f.dpi_scale_trans)
        for k in range(8):
            a = _inch_axes(f, x0 + k * (size + gap), y, size, size)
            a.imshow(maps[r, k], cmap=CMAP_SEQ, vmin=0, vmax=max(maps[:, k].max(), 1e-6), interpolation='nearest')
            frame(a)
    save('nn4_feature_maps', f, 'First-layer ReLU feature maps of the tiny digits CNN for two held-out digits (seed 0, '
         'CPU); each column scaled to its own maximum, darker = stronger response.', kind='computed')


@fig
def receptive_field():
    f, ax = canvas(11.8, 4.3)
    rows = [('Input', .55, 10, .5, 1), ('Conv 3×3', 1.55, 10, .5, 3), ('Pool 2×2, stride 2', 2.55, 5, 1.0, 4),
            ('Conv 3×3', 3.55, 5, 1.0, 8)]
    used = {0: set(range(1, 9)), 1: set(range(2, 8)), 2: {1, 2, 3}, 3: {2}}
    colors = {0: GREY, 1: TEAL, 2: BLUE, 3: ORANGE}
    x0, h = 1.9, .42
    # dependency lines first
    for m in used[1]:
        for i in (m - 1, m, m + 1):
            ax.plot([x0 + (m + .5) * .5, x0 + (i + .5) * .5], [1.55, .55 + h], color=tint(TEAL, .35), lw=1.1, zorder=1)
    for q in used[2]:
        for mm in (2 * q, 2 * q + 1):
            ax.plot([x0 + (q + .5) * 1.0, x0 + (mm + .5) * .5], [2.55, 1.55 + h], color=tint(BLUE, .35), lw=1.2,
                    zorder=1)
    for q in used[3]:
        for qq in (q - 1, q, q + 1):
            ax.plot([x0 + (q + .5) * 1.0, x0 + (qq + .5) * 1.0], [3.55, 2.55 + h], color=tint(ORANGE, .3), lw=1.4,
                    zorder=1)
    for li, (name, y, n, w, rf) in enumerate(rows):
        for i in range(n):
            on = i in used[li]
            ax.add_patch(FancyBboxPatch((x0 + i * w + .04, y), w - .08, h, boxstyle='round,pad=0,rounding_size=.05',
                                        fc=colors[li] if on else 'white', ec=colors[li] if on else '#C7D3DF',
                                        lw=1.2, zorder=2, alpha=.9 if on else 1))
        label(ax, x0 - .15, y + h / 2, name, fs=12.5, ha='right')
        label(ax, x0 + 5.15, y + h / 2, f'RF {rf}', fs=13, ha='left', weight='semibold', color=colors[li])
    label(ax, 3.9, 4.2, 'Which inputs can one top unit see?', fs=14, weight='semibold')
    ax.plot([7.75, 7.75], [.3, 4.1], color=RULE, lw=1)
    # 2-D view
    cell, gx, gtop = .34, 8.2, 3.85
    _grid(ax, gx, gtop, np.zeros((10, 10)), cell=cell, text=False, color=GREY, amount=.9, edge='white')
    for (r0, n, col, txt) in [(1, 8, ORANGE, '8 × 8 after conv 2'), (3, 4, BLUE, '4 × 4 after pool'),
                              (3, 3, TEAL, '3 × 3 after conv 1')]:
        ax.add_patch(Rectangle((gx + r0 * cell, gtop - (r0 + n) * cell), n * cell, n * cell, fc=tint(col, .7),
                               ec=col, lw=2.4, zorder=3 if n == 8 else (4 if n == 4 else 5), alpha=.55 if n == 8 else .8))
    label(ax, gx + 5 * cell, 4.2, 'The same fields in 2-D', fs=14, weight='semibold')
    for k, (col, txt) in enumerate([(TEAL, 'conv 1: 3 × 3'), (BLUE, 'pool: 4 × 4'), (ORANGE, 'conv 2: 8 × 8')]):
        chip(ax, gx + .55 + k * 1.2, .22, txt, col, fs=11.5)
    save('nn4_receptive_field', f, 'Exact receptive-field bookkeeping for conv 3x3 (stride 1), max pool 2x2 (stride 2), '
         'conv 3x3 (stride 1).', kind='computed')


# ------------------------------------------ pooling, equivariance, architecture
@fig
def pooling():
    a = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 2, 8, 1], [3, 1, 2, 4]])
    mx = a.reshape(2, 2, 2, 2).max(axis=(1, 3))
    av = a.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    quad = [[BLUE, TEAL], [PURPLE, GOLD]]                          # one colour per window, as on the next slide
    f, ax = canvas(11.8, 3.6)
    cell, xi, top = .66, .5, 3.0
    cy = top - 2 * cell
    label(ax, xi + 2 * cell, 3.35, 'Input 4 × 4', fs=14, weight='semibold')
    _grid(ax, xi, top, a, cell=cell, color=BLUE, fs=15, amount=.86)
    for i in range(2):
        for j in range(2):
            _outline(ax, xi, top, 2 * i, 2 * j, 2, 2, cell, color=quad[i][j], lw=2.6)
    arrow(ax, (xi + 4 * cell + .25, cy), (4.95, cy), color=MUTED, lw=1.6)
    label(ax, (xi + 4 * cell + 4.95) / 2 + .05, cy + .42, '2 × 2 windows\nstride 2', fs=12, color=MUTED,
          linespacing=1.15)
    oc = .8
    for x, vals, title, sub in [(5.3, mx, 'Max pool', 'largest value per window'),
                                (9.2, av, 'Average pool', 'mean of each window')]:
        label(ax, x + oc, 3.35, title, fs=14, weight='semibold')
        _grid(ax, x, cy + oc, vals, cell=oc, fills=[[quad[i][j] for j in range(2)] for i in range(2)], fs=15)
        label(ax, x + oc, cy - oc - .32, sub, fs=12.5, color=MUTED)
    ax.plot([8.05, 8.05], [.3, 3.2], color=RULE, lw=1)
    save('nn4_pooling', f, 'Exact 2 x 2 max and average pooling with stride 2 on one 4 x 4 matrix; window colours '
         'match the output cells.', kind='computed')


@fig
def pool_vs_stride():
    a = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 2, 8, 1], [3, 1, 2, 4]])
    k = np.array([[1, 0], [0, -1]])
    mx = a.reshape(2, 2, 2, 2).max(axis=(1, 3))
    st = np.array([[(a[2 * i:2 * i + 2, 2 * j:2 * j + 2] * k).sum() for j in range(2)] for i in range(2)])
    f, ax = canvas(11.8, 4.3)
    quad = [[BLUE, TEAL], [PURPLE, GOLD]]

    def blocks(x, top, cell):
        _grid(ax, x, top, a, cell=cell, color=BLUE, fs=12.5, amount=.86)
        for i in range(2):
            for j in range(2):
                _outline(ax, x, top, 2 * i, 2 * j, 2, 2, cell, color=quad[i][j], lw=2.2)

    def result(x, top, vals, cell):
        _grid(ax, x, top, vals, cell=cell, fills=[[quad[i][j] for j in range(2)] for i in range(2)], fs=13)

    mid = 2.3
    # panel 1: max pooling
    label(ax, 1.9, 4.05, 'Max pool 2 × 2, stride 2', fs=14.5, weight='semibold')
    label(ax, 1.9, 3.68, 'fixed rule, 0 parameters', fs=12.5, color=MUTED)
    blocks(.3, mid + .88, .44)
    arrow(ax, (2.2, mid), (2.6, mid), color=MUTED, lw=1.5)
    result(2.75, mid + .44, mx, .44)
    label(ax, 1.9, .55, 'keeps the largest value per window', fs=12.5)
    ax.plot([3.95, 3.95], [.25, 4.15], color=RULE, lw=1)
    # panel 2: strided convolution
    label(ax, 5.95, 4.05, 'Strided convolution 2 × 2, stride 2', fs=14.5, weight='semibold')
    label(ax, 5.95, 3.68, 'learned: 4 weights + 1 bias', fs=12.5, color=MUTED)
    blocks(4.15, mid + .84, .42)
    label(ax, 6.3, mid + .55, 'kernel', fs=12, color=MUTED)
    _grid(ax, 6.0, mid + .3, k, cell=.3, color=ORANGE, fs=12)
    arrow(ax, (6.68, mid), (6.92, mid), color=MUTED, lw=1.5, head=11)
    result(7.0, mid + .4, st, .4)
    label(ax, 5.95, .55, 'weighted sum per window, learned', fs=12.5)
    ax.plot([7.95, 7.95], [.25, 4.15], color=RULE, lw=1)
    # panel 3: global average pooling
    label(ax, 9.85, 4.05, 'Global average pooling', fs=14.5, weight='semibold')
    label(ax, 9.85, 3.68, 'one number per channel', fs=12.5, color=MUTED)
    maps = [np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), np.array([[0, 0, 0], [0, 9, 0], [0, 0, 0]]),
            np.full((3, 3), 2)]
    cols = [TEAL, BLUE, PURPLE]
    for c, (mp, col) in enumerate(zip(maps, cols)):
        top = 3.35 - c * .92
        _grid(ax, 8.45, top, mp, cell=.26, color=col, fs=11, amount=.8)
        arrow(ax, (9.35, top - .39), (10.35, top - .39), color=MUTED, lw=1.2, head=10)
        label(ax, 9.85, top - .22, 'mean', fs=11.5, color=MUTED)
        _grid(ax, 10.45, top - .17, [[mp.mean()]], cell=.44, fills=[[col]], fs=13)
    label(ax, 9.85, .55, 'C × H × W  →  C, for any H × W', fs=12.5)
    save('nn4_pool_vs_stride', f, 'Exact arithmetic: 2 x 2 max pooling, a 2 x 2 stride-2 convolution with kernel '
         '[[1, 0], [0, -1]] and zero bias, and global averaging of three 3 x 3 maps.', kind='computed')


def _equivariance_maps():
    X = load_digits().images / 16.0
    digit = X[7]                                                  # a handwritten "7"
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    out = []
    for shift in (0, 1, 2):
        img = np.zeros((16, 16))
        img[4:12, 3 + shift:11 + shift] = digit
        conv = np.maximum(correlate2d(img, kx, mode='same'), 0)
        pool = conv.reshape(8, 2, 8, 2).max(axis=(1, 3))
        out.append((img, conv, pool))
    return out


@fig
def equivariance():
    rows = _equivariance_maps()
    img0, conv0, pool0 = rows[0]
    f = plt.figure(figsize=(11.8, 4.6))
    ov = f.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, 11.8); ov.set_ylim(0, 4.6); ov.set_axis_off()
    size = 1.25
    xs = [1.75, 3.35, 4.95]
    titles = ['Input', 'Conv + ReLU\n(stride 1)', 'Max pool 2 × 2\n(stride 2)']
    names = ['Original', 'Shifted 1 px', 'Shifted 2 px']
    vmax = max(r[1].max() for r in rows)
    for r, ((img, conv, pool), name) in enumerate(zip(rows, names)):
        y = 2.85 - r * 1.4
        for c, (arr, x) in enumerate(zip((img, conv, pool), xs)):
            a = _inch_axes(f, x, y, size, size)
            a.imshow(arr, cmap='Greys' if c == 0 else CMAP_SEQ, vmin=0, vmax=1 if c == 0 else vmax,
                     interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(RULE)
            if r == 0:
                a.set_title(titles[c], fontsize=12.5, pad=5, linespacing=1.1)
        ov.text(1.55, y + size / 2, name, fontsize=13, ha='right', va='center')
        conv_ok = np.array_equal(conv, np.roll(conv0, r, axis=1))
        pool_ok = r % 2 == 0 and np.array_equal(pool, np.roll(pool0, r // 2, axis=1))
        changed = int((pool != pool0).sum())
        lines = []
        if r == 0:
            lines.append(('reference maps', MUTED))
        else:
            lines.append(('conv map: exact shifted copy' if conv_ok else 'conv map: changed', TEAL))
            lines.append(('pooled map: shifted by one cell' if pool_ok else
                          f'pooled map: {changed} of 64 values change', TEAL if pool_ok else ORANGE))
        lines.append((f'global average of conv map = {conv.mean():.4f}', INK))
        for i, (t, col) in enumerate(lines):
            ov.text(6.6, y + .95 - i * .36, t, fontsize=13, color=col, va='center')
    save('nn4_equivariance', f, 'Exact computation: a scikit-learn digit placed on a 16 x 16 zero canvas, fixed '
         'Sobel x-derivative kernel with zero padding, ReLU, 2 x 2 max pooling.', kind='computed')


@fig
def cnn_blocks():
    f, ax = canvas(11.8, 4.3)
    cy = 2.05                                                      # vertical centre line of the tensors
    # (x, spatial size, channel depth, ...): the gaps between tensors leave room for the operation labels
    specs = [(.08, 1.3, .12, GREY, 'Input', '3 × 32 × 32'), (2.4, 1.3, .45, TEAL, 'Conv 1', '16 × 32 × 32'),
             (4.93, .72, .45, TEAL, 'Pool', '16 × 16 × 16'), (6.86, .72, .85, TEAL, 'Conv 2', '32 × 16 × 16')]
    ops = ['conv 3×3\n+ ReLU', 'max pool\n2×2', 'conv 3×3\n+ ReLU', 'global\naverage']
    blocks = []
    for (x, hw, ch, col, name, shape) in specs:
        b = block3d(ax, x, cy - hw / 2, hw, hw, ch, col)
        blocks.append(b)
        label(ax, x + hw / 2, .92, name, fs=12.5, weight='semibold', color=INK)
        label(ax, x + hw / 2, .6, shape, fs=12.5, color=MUTED)
    vx = 9.05
    ends = [b.x + b.w + .1 for b in blocks] + [None]
    starts = [b.x - .1 for b in blocks[1:]] + [vx - .1]
    for i in range(4):
        x0, x1 = ends[i], starts[i]
        arrow(ax, (x0, cy), (x1, cy), color=MUTED, lw=1.5)
        label(ax, (x0 + x1) / 2, cy + .5, ops[i], fs=12, color=INK, linespacing=1.15)
    for i in range(8):
        ax.add_patch(Rectangle((vx, cy - .68 + i * .17), .2, .17, fc=tint(TEAL, .55), ec=shade(TEAL, .2), lw=.8,
                               zorder=2))
    label(ax, vx + .1, .92, 'Pooled', fs=12.5, weight='semibold')
    label(ax, vx + .1, .6, '32', fs=12.5, color=MUTED)
    lin = box(ax, 9.55, cy - .36, .9, .72, 'Linear', BLUE, fs=13, sub='32 → 10', sub_fs=11.5)
    arrow(ax, (vx + .26, cy), lin.l, color=MUTED, lw=1.5, head=11)
    out = box(ax, 10.72, cy - .36, .95, .72, '10\nlogits', ORANGE, fill='solid', fs=12.5)
    arrow(ax, lin.r, out.l, color=MUTED, lw=1.5, head=11)
    bracket(ax, blocks[1].x, blocks[3].x + blocks[3].w, 3.55, 'feature extractor: shared local filters, shrinking maps',
            color=TEAL, below=False, fs=12.5)
    bracket(ax, 9.0, 11.67, 3.55, 'head', color=ORANGE, below=False, fs=12.5)
    save('nn4_cnn_blocks', f, 'Schematic of the compact CNN defined on the next slides; shapes are C x H x W for one '
         'image (batch dimension omitted).', kind='schematic')


@fig
def param_budget():
    f, axs = half()
    ax = axs[0]
    rows = [('fc2  512 → 10', 5130, ORANGE), ('fc1  4096 → 512', 2097664, ORANGE),
            ('conv2  32 → 64', 18496, TEAL), ('conv1  3 → 32', 896, TEAL),
            (None, 0, None),
            ('linear  32 → 10', 330, ORANGE), ('conv2  16 → 32', 4640, TEAL), ('conv1  3 → 16', 448, TEAL)]
    ys = np.arange(len(rows))[::-1]
    for y, (name, n, col) in zip(ys, rows):
        if name is None:
            continue
        ax.barh(y, n, color=col, height=.62, zorder=3)
        ax.text(n * 1.25, y, f'{n:,}', va='center', fontsize=12, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(100, 4e7)
    ax.set_yticks([y for y, r in zip(ys, rows) if r[0]], [r[0] for r in rows if r[0]], fontsize=12)
    ax.set_ylim(-.7, len(rows) - .2)
    ax.text(110, ys[0] + .62, 'Lab 1 SimpleCNN: 2,122,186 parameters', fontsize=12.5, weight='semibold', color=INK)
    ax.text(110, ys[5] + .62, 'Compact CNN with GAP: 5,418', fontsize=12.5, weight='semibold', color=INK)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    clean(ax, 'Trainable parameters (log scale)')
    ax.grid(axis='y', visible=False)
    save('nn4_param_budget', f, 'Exact parameter counts (weights + biases) of the Lab 1 SimpleCNN and the compact '
         'CNN, checked with PyTorch.', kind='computed')


@fig
def arch_timeline():
    f, ax = canvas(11.8, 4.3)
    items = [('LeNet-5', '1998', 'convolution + subsampling', 'trained end to end'),
             ('AlexNet', '2012', 'ReLU, dropout, GPUs', 'at ImageNet scale'),
             ('VGG', '2014', 'deep stacks of', 'small 3×3 kernels'),
             ('GoogLeNet', '2014', 'parallel branches,', '1×1 bottlenecks'),
             ('ResNet', '2015', 'identity shortcuts', 'ease very deep training'),
             ('DenseNet', '2016', 'each layer reuses', 'earlier maps in its block'),
             ('EfficientNet', '2019', 'scale depth, width and', 'resolution together'),
             ('ViT', '2020', 'image patches +', 'self-attention (no conv)'),
             ('ConvNeXt', '2022', 'a CNN modernized with', 'transformer-era choices')]
    y0 = 2.15
    ax.plot([.25, 11.55], [y0, y0], color='#9DAFC0', lw=2.2, zorder=1)
    xs = [1.07 + i * 1.21 for i in range(len(items))]              # keeps the widest labels inside the canvas
    for i, ((name, year, l1, l2), x) in enumerate(zip(items, xs)):
        col = PURPLE if name == 'ViT' else BLUE
        node(ax, x, y0, .12, col, fill='solid')
        up = i % 2 == 0
        s = 1 if up else -1
        ax.plot([x, x], [y0 + s * .14, y0 + s * .32], color=col, lw=1.2, zorder=1)
        label(ax, x, y0 + s * .5, year, fs=12.5, color=MUTED)
        label(ax, x, y0 + s * .92, name, fs=14, weight='semibold', color=col)
        first, second = (l2, l1) if up else (l1, l2)
        label(ax, x, y0 + s * 1.33, first, fs=12, color=INK)
        label(ax, x, y0 + s * 1.64, second, fs=12, color=INK)
    save('nn4_arch_timeline', f, 'Schematic timeline, not to scale; years of first publication (conference or '
         'preprint). Design ideas only, no benchmark results.', kind='schematic')


@fig
def residual_block():
    f, ax = canvas(6.3, 4.9)
    for cx, title, first, short, xin, xout in [
            (1.55, 'Identity shortcut', 'conv 3×3', None, '64 × 32 × 32', '64 × 32 × 32'),
            (4.2, 'Projection shortcut', 'conv 3×3, stride 2', '1×1 conv\nstride 2\n+ BN', '64 × 32 × 32',
             '128 × 16 × 16')]:
        label(ax, cx + .35, 4.72, title, fs=14, weight='semibold')
        node(ax, cx, 4.2, .17, GREY, text='x', fs=13)
        label(ax, cx - .26, 4.2, xin, fs=11.5, color=MUTED, ha='right')
        prev = (cx, 4.03)
        for y, t in zip([3.45, 2.85, 2.25, 1.65], [first, 'BN, ReLU', 'conv 3×3', 'BN']):
            b = box(ax, cx - .78, y, 1.56, .42, t, BLUE, fs=12 if len(t) > 12 else 12.5)
            arrow(ax, prev, b.t, color=INK, lw=1.3, head=10)
            prev = b.b
        node(ax, cx, 1.05, .2, ORANGE, fill='white', text='+', fs=16)
        arrow(ax, prev, (cx, 1.25), color=INK, lw=1.3, head=10)
        sx = cx + (1.2 if short is None else 1.45)
        ax.plot([cx + .17, sx, sx], [4.2, 4.2, 1.05], color=TEAL, lw=2.2, zorder=1, solid_joinstyle='round')
        arrow(ax, (sx, 1.05), (cx + .21, 1.05), color=TEAL, lw=2.2, head=12)
        if short is None:
            label(ax, sx + .14, 2.65, 'identity', fs=12.5, color=TEAL, rotation=90)
        else:
            box(ax, sx - .5, 2.05, 1.0, 1.0, short, TEAL, fs=11.5, fill='white')
        b = box(ax, cx - .5, .3, 1.0, .4, 'ReLU', BLUE, fs=12.5)
        arrow(ax, (cx, .85), b.t, color=INK, lw=1.3, head=10)
        label(ax, cx - .6, .5, xout, fs=11.5, color=MUTED, ha='right')
    save('nn4_residual_block', f, 'Schematic of basic residual blocks: output = ReLU(x + F(x)); the projection '
         'shortcut matches channels and resolution.', kind='schematic')


@fig
def batchnorm_axes():
    f, ax = canvas(6.3, 4.9)
    label(ax, 3.1, 4.65, 'One mean and variance per channel', fs=14, weight='semibold')
    label(ax, 3.1, 4.2, 'each image: C = 3 channel sheets, channel c in front', fs=12.5, color=MUTED)
    for b in range(4):
        x = .3 + b * 1.5
        _stack(ax, x, 2.3, 4, .22, [ORANGE, GREY, GREY], 3, dx=.16, dy=.16, lw=.8)
        label(ax, x + .5, 2.02, f'image {b + 1}', fs=12, color=MUTED)
    bracket(ax, .3, 5.6, 1.75, '', color=ORANGE, below=True)
    label(ax, 2.95, 1.4, 'channel c across the batch and positions:  B × H × W values', fs=12.5, color=ORANGE)
    label(ax, 3.0, .95, r'$\hat{x} = (x - \mu_c)\,/\,\sqrt{\sigma_c^2 + \epsilon}$,   $y = \gamma_c\,\hat{x} + \beta_c$',
          fs=14)
    label(ax, 3.0, .42, 'BatchNorm2d(C): 2C learned (γ, β) + 2C running statistics', fs=12.5, color=MUTED)
    save('nn4_batchnorm_axes', f, 'Schematic of BatchNorm2d statistics: per channel, pooled over the batch and all '
         'spatial positions.', kind='schematic')




# ------------------------------------------ training, augmentation, transfer
def _china_small(n=160):
    im = load_sample_image('china.jpg')[40:400, 20:380].astype(float) / 255
    return np.clip(nd_zoom(im, (n / 360, n / 360, 1), order=1), 0, 1)


@fig
def augment_examples():
    rng = np.random.default_rng(3)
    img = _china_small()
    n = img.shape[0]
    flip = img[:, ::-1]
    c = int(.7 * n)
    r0, c0 = rng.integers(0, n - c, 2)
    crop = np.clip(nd_zoom(img[r0:r0 + c, c0:c0 + c], (n / c, n / c, 1), order=1)[:n, :n], 0, 1)
    rot = np.clip(nd_rotate(img, 15, axes=(1, 0), reshape=False, order=1, mode='constant', cval=0), 0, 1)
    bright = np.clip(img * 1.3, 0, 1)
    noise = np.clip(img + rng.normal(0, .08, img.shape), 0, 1)
    panels = [(img, 'Original', '', 'reference'), (flip, 'Horizontal flip', 'p = 0.5', 'mirror symmetry'),
              (crop, 'Random crop', '70% side, resized', 'framing and scale'),
              (rot, 'Rotation', '15°, black corners', 'orientation'),
              (bright, 'Brightness', '× 1.3, clipped', 'illumination'),
              (noise, 'Gaussian noise', 'σ = 0.08', 'sensor noise')]
    f, ax = canvas(11.8, 3.6)
    s_, gap = 1.8, .15                                  # six panels and their frames fit inside the 11.8 in canvas
    for k, (a, t1, t2, t3) in enumerate(panels):
        x = .125 + k * (s_ + gap)
        _img(ax, a, x, .75, s_, s_, cmap=None, edge=RULE)
        label(ax, x + s_ / 2, 3.25, t1, fs=13.5, weight='semibold')
        label(ax, x + s_ / 2, 2.87, t2 or ' ', fs=12, color=MUTED)
        label(ax, x + s_ / 2, .42, t3, fs=12.5, color=TEAL if k else MUTED, style='italic' if k else 'normal')
    save('nn4_augment_examples', f, 'scikit-learn sample image china.jpg (CC BY 2.0), 360 x 360 crop resized to '
         '160 x 160; transforms computed with NumPy/SciPy, random seed 3.', kind='computed')


@fig
def label_changing():
    f, ax = canvas(11.8, 4.3)
    X, y = _digits()
    six = X[np.where(y == 6)[0][2]]
    for x in (3.95, 7.9):
        ax.plot([x, x], [.3, 4.15], color=RULE, lw=1)
    # panel 1: digit 6 rotated by 180 degrees (images kept clear of the divider at x = 3.95)
    label(ax, 1.95, 4.0, 'Rotate a “6” by 180°', fs=14, weight='semibold')
    _img(ax, six, .25, 1.45, 1.5, 1.5, cmap='Greys', vmin=0, vmax=1)
    arrow(ax, (1.85, 2.2), (2.2, 2.2), color=MUTED, lw=1.5)
    _img(ax, np.rot90(six, 2), 2.3, 1.45, 1.5, 1.5, cmap='Greys', vmin=0, vmax=1)
    label(ax, 1.0, 1.15, 'label 6', fs=12.5, color=MUTED)
    label(ax, 3.05, 1.15, 'reads as 9', fs=12.5, color=ORANGE)
    cross(ax, .75, .55, s=.12)
    label(ax, .95, .55, 'label changes', fs=13, color=ORANGE, ha='left', weight='semibold')
    # panel 2: seismic polarity reversal
    label(ax, 5.92, 4.0, 'Reverse seismic polarity', fs=14, weight='semibold')
    _, _, seis, _, dt = _seismic()
    tr = seis[:, 20]
    t = np.arange(len(tr)) * dt * 1000
    sel = (t > 180) & (t < 330)
    for j, (sign, title) in enumerate([(1, 'original'), (-1, 'reversed')]):
        a = _inch_axes(f, 4.85 + j * 1.6, 1.05, 1.05, 2.55)
        w = sign * tr[sel]
        a.plot(w, t[sel], color=INK, lw=1.4)
        a.fill_betweenx(t[sel], 0, w, where=w > 0, color=BLUE, alpha=.6, lw=0)
        a.fill_betweenx(t[sel], 0, w, where=w < 0, color=ORANGE, alpha=.35, lw=0)
        a.set_ylim(330, 180)
        v = np.abs(tr[sel]).max() * 1.1
        a.set_xlim(-v, v)
        a.set_xticks([])
        a.set_yticks([200, 250, 300] if j == 0 else [])
        if j == 0:
            a.set_ylabel('Time (ms)', fontsize=12)
        a.set_title(title, fontsize=12.5, color=MUTED, pad=4)
        a.spines['left'].set_visible(j == 0)
        a.axvline(0, color=RULE, lw=1)
    ax.add_patch(Polygon([(4.86, .44), (5.1, .44), (4.98, .66)], closed=True, fc=GOLD, ec='none'))
    label(ax, 5.2, .55, 'label may change', fs=13, color=GOLD, weight='semibold', ha='left')
    # panel 3: circular azimuth shift of a casing map
    label(ax, 9.85, 4.0, 'Rotate a casing map (azimuth)', fs=14, weight='semibold')
    patch = _casing_map('pits', seed=5)[22:82]
    shifted = np.roll(patch, 18, axis=1)
    for j, (arr, title) in enumerate([(patch, 'original'), (shifted, 'shifted 90°')]):
        _img(ax, arr, 8.2 + j * 1.85, 1.2, 1.6, 2.2, cmap=CMAP_DIV, norm=THICK_NORM)
        label(ax, 9.0 + j * 1.85, 3.6, title, fs=12.5, color=MUTED)
    arrow(ax, (9.83, 2.3), (10.02, 2.3), color=MUTED, lw=1.4, head=10)
    label(ax, 9.85, .95, 'pits are still pits', fs=12.5, color=MUTED)
    check(ax, 8.72, .52, s=.13)
    label(ax, 8.95, .52, 'label preserved', fs=13, color=TEAL, ha='left', weight='semibold')
    save('nn4_label_changing', f, 'scikit-learn digit; synthetic seismic trace (seed 7); synthetic thickness map '
         '(seed 5) circularly shifted by 18 of 72 sectors.', kind='computed')


@fig
def grouped_patches():
    f, ax = canvas(11.8, 4.3)
    rng = np.random.default_rng(8)
    cols = {'train': BLUE, 'val': TEAL, 'test': ORANGE}

    def well(x, name, assign, top=3.42, n=9, h=.5, step=.25):
        ax.add_patch(FancyBboxPatch((x, top - (n - 1) * step - h), .3, (n - 1) * step + h,
                                    boxstyle='round,pad=0,rounding_size=.06', fc=PANEL, ec='#9DAFC0', lw=1.2))
        label(ax, x + .15, top + .22, name, fs=12.5, weight='semibold')
        for k in range(n):
            xo = x + .42 + (k % 2) * .26
            ax.add_patch(Rectangle((xo, top - k * step - h), .2, h, fc=tint(cols[assign[k]], .45),
                                   ec=cols[assign[k]], lw=1.3, zorder=3))
    # left: random patch split
    label(ax, 2.8, 4.12, 'Random patch split', fs=14.5, weight='semibold')
    hl = None
    for w, x in zip('ABC', [.55, 2.25, 3.95]):
        assign = rng.choice(['train', 'val', 'test'], size=9, p=[.6, .2, .2])
        well(x, f'Well {w}', assign)
        for k in range(8):
            if hl is None and {assign[k], assign[k + 1]} == {'train', 'test'}:
                hl = (x, k)
    if hl is not None:
        x, k = hl
        top, step, h = 3.42, .25, .5
        ax.add_patch(Rectangle((x + .36, top - (k + 1) * step - h - .05), .58, h + step + .1, fc='none', ec=INK,
                               lw=1.3, ls='--', zorder=5))
    label(ax, 2.8, .22, 'overlapping windows fall on both sides of the split', fs=12.5, color=ORANGE)
    ax.plot([5.75, 5.75], [.3, 4.15], color=RULE, lw=1)
    # right: grouped by well
    label(ax, 8.75, 4.12, 'Grouped by well', fs=14.5, weight='semibold')
    for w, x, a in zip('ABCDE', [6.1, 7.25, 8.4, 9.55, 10.7], ['train', 'train', 'train', 'val', 'test']):
        well(x, f'Well {w}', [a] * 9)
    label(ax, 8.75, .22, 'every window of a well stays in one set', fs=12.5, color=TEAL)
    for k, (name, c) in enumerate([('training', BLUE), ('validation', TEAL), ('test', ORANGE)]):
        chip(ax, 7.55 + k * 1.2, .62, name, c, fs=11.5)
    save('nn4_grouped_patches', f, 'Schematic: depth windows of 32 rows with stride 16 (50% overlap); random '
         'assignment drawn with seed 8.', kind='schematic')


def _lock(ax, x, y, s=.13, color=FROZEN):
    ax.add_patch(Arc((x, y + s * .55), s * 1.1, s * 1.3, theta1=0, theta2=180, ec=color, lw=1.8, zorder=6))
    ax.add_patch(FancyBboxPatch((x - s * .75, y - s * .55), s * 1.5, s * 1.1,
                                boxstyle='round,pad=0,rounding_size=.02', fc=color, ec=color, zorder=6))


@fig
def transfer():
    f, ax = canvas(11.8, 4.3)
    rows = [('Feature extraction', ['F', 'F', 'F', 'F'], None),
            ('Fine-tuning', ['F', 'F', 'T', 'T'], None),
            ('Discriminative rates', ['T', 'T', 'T', 'T'], ['1e-5', '3e-5', '1e-4', '3e-4', '1e-3'])]
    xs = [3.35, 4.9, 6.45, 8.0]
    for r, (name, states, lrs) in enumerate(rows):
        y = 2.9 - r * 1.2
        label(ax, .15, y + .3, name, fs=13.5, weight='semibold', ha='left')
        b0 = box(ax, 2.25, y, .8, .6, 'Input', GREY, fs=12)
        prev = b0
        for k, (st, x) in enumerate(zip(states, xs)):
            frozen = st == 'F'
            b = box(ax, x, y, 1.25, .6, f'Stage {k + 1}', FROZEN if frozen else BLUE,
                    fill='tint', fs=12.5, text_color=MUTED if frozen else INK)
            if frozen:
                _lock(ax, x + 1.1, y + .45, s=.1)
            arrow(ax, prev.r, b.l, color=MUTED, lw=1.3, head=10)
            prev = b
            if lrs:
                label(ax, x + .62, y - .2, f'lr {lrs[k]}', fs=11.5, color=BLUE)
        h = box(ax, 9.75, y, 1.6, .6, 'New head', ORANGE, fill='solid', fs=12.5)
        arrow(ax, prev.r, h.l, color=MUTED, lw=1.3, head=10)
        if lrs:
            label(ax, 10.55, y - .2, f'lr {lrs[4]}', fs=11.5, color=ORANGE)
    bracket(ax, 3.35, 9.25, 3.78, 'pretrained backbone (e.g. ImageNet weights)', color=BLUE, below=False, fs=12.5)
    bracket(ax, 9.75, 11.35, 3.78, 'task-specific', color=ORANGE, below=False, fs=12.5)
    _lock(ax, .3, 4.02, s=.1)
    label(ax, .48, 4.02, 'frozen (requires_grad = False)', fs=12, color=MUTED, ha='left')
    save('nn4_transfer', f, 'Schematic of three adaptation strategies for a four-stage pretrained backbone; '
         'learning rates are illustrative.', kind='schematic')


def _defect_scene():
    """Synthetic thickness patch (64 depth rows x 96 azimuth sectors) with known defects."""
    rng = np.random.default_rng(12)
    H, W = 64, 96
    zz, aa = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    t = 1 + .012 * rng.standard_normal((H, W)) + .02 * np.cos(2 * np.pi * aa / W)
    defects = [('pit', 18, 20, 4.0, .3), ('pit', 40, 58, 3.6, .28), ('pit', 44, 67, 3.4, .26)]
    inst = np.zeros((H, W), int)
    sem = np.zeros((H, W), int)
    boxes = []
    for k, (cls, cz, ca, s, d) in enumerate(defects, start=1):
        g = np.exp(-((zz - cz) ** 2 + (aa - ca) ** 2) / (2 * s ** 2))
        t -= d * g
        m = g > .3
        inst[m & (inst == 0)] = k
        sem[m] = 1
        r, c = np.where(m)
        boxes.append((cls, c.min(), r.min(), c.max() + 1, r.max() + 1))
    gm = (np.abs(aa - 84) < 2.5) & (zz > 8) & (zz < 56)
    t -= .14 * gm
    inst[gm] = 4
    sem[gm] = 2
    r, c = np.where(gm)
    boxes.append(('groove', c.min(), r.min(), c.max() + 1, r.max() + 1))
    return t, sem, inst, boxes


@fig
def vision_tasks():
    t, sem, inst, boxes = _defect_scene()
    H, W = t.shape
    f, ax = canvas(11.8, 4.3)
    iw, ih = 2.55, 1.7
    spots = [(.15, 2.3), (6.05, 2.3), (.15, .2), (6.05, .2)]
    texts = [('Classification', 'one label per image', '“defects present”'),
             ('Object detection', 'box + class + score per object', 'matched to truth by IoU'),
             ('Semantic segmentation', 'a class for every pixel', 'pits merge into one region'),
             ('Instance segmentation', 'a separate mask per object', 'touching pits stay separate')]
    inst_cols = {1: ORANGE, 2: RED, 3: GOLD, 4: PURPLE}
    for k, ((x, y), (t1, t2, t3)) in enumerate(zip(spots, texts)):
        _img(ax, t, x, y, iw, ih, cmap=CMAP_DIV, norm=THICK_NORM, edge=RULE)
        sx, sy = iw / W, ih / H
        if k == 0:
            chip(ax, x + iw / 2, y + ih - .22, 'defective', ORANGE, fs=12, solid=True)
        elif k == 1:
            for j, (cls, c0, r0, c1, r1) in enumerate(boxes):
                col = PURPLE if cls == 'groove' else ORANGE
                ax.add_patch(Rectangle((x + c0 * sx - .03, y + ih - r1 * sy - .03), (c1 - c0) * sx + .06,
                                       (r1 - r0) * sy + .06, fc='none', ec=col, lw=2, zorder=4))
                if j == 2:                                          # label the touching pit below its box
                    label(ax, x + c1 * sx, y + ih - r1 * sy - .16, cls, fs=11, color=col, ha='right',
                          weight='semibold')
                elif j == 3:
                    label(ax, x + c0 * sx - .06, y + ih - r0 * sy - .02, cls, fs=11, color=col, ha='right',
                          va='top', weight='semibold', rotation=90)
                else:
                    label(ax, x + c0 * sx, y + ih - r0 * sy + .14, cls, fs=11, color=col, ha='left',
                          weight='semibold')
        else:
            over = np.zeros((H, W, 4))
            if k == 2:
                for v, col in [(1, ORANGE), (2, PURPLE)]:
                    over[sem == v] = (*matplotlib.colors.to_rgb(col), .85)
            else:
                for v, col in inst_cols.items():
                    over[inst == v] = (*matplotlib.colors.to_rgb(col), .85)
            ax.imshow(over, extent=(x, x + iw, y, y + ih), interpolation='nearest', zorder=3)
        tx = x + iw + .25
        label(ax, tx, y + ih - .2, t1, fs=14, weight='semibold', ha='left')
        label(ax, tx, y + ih - .62, t2, fs=12.5, ha='left')
        label(ax, tx, y + ih - .98, t3, fs=12.5, ha='left', color=MUTED)
    save('nn4_vision_tasks', f, 'Synthetic thickness patch (seed 12) with three pits and one groove; the label formats '
         'are derived from the known synthetic geometry, not from a model.', kind='computed')


@fig
def iou():
    f, axs = half()
    ax = axs[0]
    gt, pr = (2, 2, 8, 7), (4, 3, 10, 8)
    ix0, iy0, ix1, iy1 = max(gt[0], pr[0]), max(gt[1], pr[1]), min(gt[2], pr[2]), min(gt[3], pr[3])
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
    a_pr = (pr[2] - pr[0]) * (pr[3] - pr[1])
    union = a_gt + a_pr - inter
    # fills first, then the overlap, then both outlines on top so that every edge stays visible
    ax.add_patch(Rectangle(gt[:2], gt[2] - gt[0], gt[3] - gt[1], fc=tint(BLUE, .8), ec='none', zorder=2))
    ax.add_patch(Rectangle(pr[:2], pr[2] - pr[0], pr[3] - pr[1], fc=tint(ORANGE, .8), ec='none', zorder=3,
                           alpha=.8))
    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fc=PURPLE, ec=PURPLE, alpha=.45, lw=0, zorder=4,
                           hatch='//'))
    ax.add_patch(Rectangle(gt[:2], gt[2] - gt[0], gt[3] - gt[1], fc='none', ec=BLUE, lw=2.5, zorder=5))
    ax.add_patch(Rectangle(pr[:2], pr[2] - pr[0], pr[3] - pr[1], fc='none', ec=ORANGE, lw=2.5, ls='--', zorder=5))
    ax.text(gt[0], gt[1] - .45, f'ground truth  {gt[2] - gt[0]} × {gt[3] - gt[1]} = {a_gt}', color=BLUE,
            fontsize=12.5, va='top')
    ax.text(10, 8.35, f'prediction  {pr[2] - pr[0]} × {pr[3] - pr[1]} = {a_pr}', color=ORANGE, fontsize=12.5,
            ha='right')
    ax.text((ix0 + ix1) / 2, (iy0 + iy1) / 2, f'overlap\n{ix1 - ix0} × {iy1 - iy0} = {inter}', color='white',
            fontsize=12.5, ha='center', va='center', weight='semibold', zorder=6)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9.4)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 12, 2))
    ax.set_yticks(range(0, 10, 2))
    clean(ax, 'x (pixels)', 'y (pixels)')
    ax.set_title(f'IoU = {inter} / ({a_gt} + {a_pr} − {inter}) = {inter}/{union} = {inter / union:.2f}',
                 fontsize=14, pad=10)
    save('nn4_iou', f, 'Exact intersection-over-union for two axis-aligned boxes (corners inclusive-exclusive).',
         kind='computed')


@fig
def unet():
    f, ax = canvas(11.8, 4.3)
    lv_y = [3.05, 2.2, 1.35, .5]
    hs = [.8, .62, .46, .34]
    wd = {32: .16, 64: .22, 128: .3, 256: .4}
    size = ['128²', '64²', '32²', '16²']

    def blk(x, lvl, ch, col, txt=True):
        h = hs[lvl]
        y = lv_y[lvl] - h / 2 + .3
        ax.add_patch(Rectangle((x, y), wd[ch], h, fc=tint(col, .55), ec=shade(col, .15), lw=1.2, zorder=3))
        if txt:
            label(ax, x + wd[ch] / 2, y + h + .14, str(ch), fs=11.5, color=INK)
        return (x, y, wd[ch], h)

    ax.add_patch(Rectangle((.2, lv_y[0] - .1), .1, .8, fc=tint(GREY, .5), ec=GREY, lw=1.2))
    label(ax, .25, lv_y[0] - .25, 'input', fs=11.5, color=MUTED)
    label(ax, .25, lv_y[0] + .85, '1', fs=11.5, color=INK)
    enc_x = [.75, 1.95, 3.15]
    enc = []
    for l, (x, ch) in enumerate(zip(enc_x, [32, 64, 128])):
        enc.append(blk(x, l, ch, BLUE))
    arrow(ax, (.36, lv_y[0] + .3), (enc[0][0] - .04, lv_y[0] + .3), color=INK, lw=1.3, head=10)
    bott = blk(5.0, 3, 256, PURPLE)
    label(ax, bott[0] + bott[2] / 2, bott[1] - .15, size[3], fs=11.5, color=MUTED)
    dec_x = [9.1, 7.9, 6.7]
    dec = [None] * 3
    for l, (x, ch) in zip([2, 1, 0], [(6.7, 128), (7.9, 64), (9.1, 32)]):
        dec[l] = blk(x, l, ch, TEAL)
    # down / up arrows
    for l in range(3):
        x, y, w, h = enc[l]
        nx, ny, nw, nh = enc[l + 1] if l < 2 else bott
        arrow(ax, (x + w / 2, y - .05), (nx - .05, ny + nh / 2), color=ORANGE, lw=1.6, head=11)
    ups = [bott, dec[2], dec[1]]
    for l, src in zip([2, 1, 0], ups):
        x, y, w, h = src
        tx, ty, tw, th = dec[l]
        arrow(ax, (x + w + .05, y + h / 2), (tx + tw / 2, ty - .05), color=TEAL, lw=1.6, head=11)
    for l in range(3):
        x, y, w, h = enc[l]
        tx, ty, tw, th = dec[l]
        arrow(ax, (x + w + .08, y + h / 2), (tx - .08, ty + th / 2), color=FAINT, lw=1.4, ls='--', head=10,
              text=f'{size[l]} maps copied', fs=11.5, text_color=MUTED, offset=(0, .16))
    x, y, w, h = dec[0]
    out = box(ax, 10.25, y + h / 2 - .35, 1.35, .7, 'mask logits', ORANGE, fill='solid', fs=12)
    arrow(ax, (x + w + .05, y + h / 2), out.l, color=INK, lw=1.3, head=10)
    label(ax, 10.93, y + h / 2 - .55, 'K × 128²', fs=11.5, color=MUTED)
    chip(ax, 6.6, .38, 'max pool: halve H, W', ORANGE, fs=11.5)
    chip(ax, 8.55, .38, 'up-conv: double H, W', TEAL, fs=11.5)
    chip(ax, 10.55, .38, 'skip: concatenate', GREY, fs=11.5)
    label(ax, 1.95, 4.15, 'Encoder: what is there', fs=13.5, weight='semibold', color=BLUE)
    label(ax, 7.9, 4.15, 'Decoder: where it is', fs=13.5, weight='semibold', color=TEAL)
    save('nn4_unet', f, 'Schematic of a small U-Net for a 128 x 128 single-channel input; numbers are channel counts, '
         'side labels are spatial sizes.', kind='schematic')


@fig
def seismic_fault_mask():
    _, _, seis, mask, dt = _seismic()
    nt, nx = seis.shape
    tmax = nt * dt * 1000
    frac = mask.mean()
    f, axs = half(2, width=6.3, height=4.9)
    v = np.abs(seis).max() * .7
    axs[0].imshow(seis, cmap=CMAP_DIV, vmin=-v, vmax=v, aspect='auto', extent=(0, nx, tmax, 0),
                  interpolation='bilinear')
    axs[0].set_title('Input section', fontsize=14)
    axs[1].imshow(seis, cmap='Greys', vmin=-v * 1.6, vmax=v * 1.6, aspect='auto', extent=(0, nx, tmax, 0),
                  interpolation='bilinear', alpha=.55)
    over = np.zeros((nt, nx, 4))
    over[mask] = (*matplotlib.colors.to_rgb(ORANGE), 1)
    axs[1].imshow(over, aspect='auto', extent=(0, nx, tmax, 0), interpolation='nearest')
    axs[1].set_title('Target mask: fault = 1', fontsize=14)
    axs[1].text(nx * .03, tmax * .96, f'fault pixels: {frac:.1%}', fontsize=12.5, color=ORANGE,
                weight='semibold', va='bottom')
    for a in axs:
        a.set_xlabel('Trace')
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(RULE)
    axs[0].set_ylabel('Two-way time (ms)')
    axs[1].set_yticklabels([])
    save('nn4_seismic_fault_mask', f, f'Synthetic section (seed 7) and its fault mask (pixels within 1.5 traces of the '
         f'fault plane; {frac:.1%} of pixels).', kind='computed')


# ------------------------------------------------------ interpretability
def _occlusion(model, x, cls, patch=2, fill=0.0):
    """Drop in the class probability when a patch x patch square is set to `fill`, averaged per pixel."""
    p0 = _predict_proba(model, x[None])[0, cls]
    H, W = x.shape
    batch = []
    for i in range(H - patch + 1):
        for j in range(W - patch + 1):
            z = x.copy()
            z[i:i + patch, j:j + patch] = fill
            batch.append(z)
    probs = _predict_proba(model, np.stack(batch))[:, cls]
    drop = np.zeros((H, W))
    count = np.zeros((H, W))
    k = 0
    for i in range(H - patch + 1):
        for j in range(W - patch + 1):
            drop[i:i + patch, j:j + patch] += p0 - probs[k]
            count[i:i + patch, j:j + patch] += 1
            k += 1
    return drop / count, p0


def _saliency(model, x, cls):
    import torch
    t = torch.from_numpy(x[None, None].astype(np.float32)).requires_grad_(True)
    model(t)[0, cls].backward()
    return np.abs(t.grad.numpy()[0, 0])


@fig
def occlusion():
    m, acc = _tiny_cnn()
    Xtr, Xte, ytr, yte = _digits_split()
    pred = _predict_proba(m, Xte).argmax(1)
    picks = [int(np.where((yte == d) & (pred == d))[0][0]) for d in (2, 4, 9)]
    f = plt.figure(figsize=(6.3, 4.9))
    size, xs = 1.3, [.95, 2.6, 4.25]
    with _torch_single_thread():
        for r, idx in enumerate(picks):
            x, c = Xte[idx], int(yte[idx])
            occ, p0 = _occlusion(m, x, c)
            sal = _saliency(m, x, c)
            y = 3.2 - r * 1.55
            for k, (arr, cmap, vmax) in enumerate([(x, 'Greys', 1), (np.clip(occ, 0, None), CMAP_ORANGE, None),
                                                   (sal, CMAP_SEQ, None)]):
                a = _inch_axes(f, xs[k], y, size, size)
                a.imshow(arr, cmap=cmap, vmin=0, vmax=vmax if vmax else max(arr.max(), 1e-9),
                         interpolation='nearest')
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(True); sp.set_color(RULE)
                if r == 0:
                    a.set_title(['Input', 'Occlusion drop', 'Gradient size'][k], fontsize=13, pad=6)
            f.text(.8 / 6.3, (y + size / 2) / 4.9, f'digit {c}\np = {p0:.2f}', fontsize=12.5, ha='right',
                   va='center', linespacing=1.3)
    save('nn4_occlusion', f, 'Tiny digits CNN (seed 0, CPU) on held-out digits: probability drop when a 2 x 2 patch '
         'is set to 0, and absolute input gradient of the true-class logit; each map scaled to its own maximum.',
         kind='computed')


@lru_cache(maxsize=None)
def _shortcut_models():
    X, y = _digits()
    keep = np.isin(y, [3, 8])
    X2, y2 = X[keep], (y[keep] == 8).astype(int)
    Xa, Xb, ya, yb = train_test_split(X2, y2, test_size=0.4, random_state=1, stratify=y2)
    Xv, Xs, yv, ys = train_test_split(Xb, yb, test_size=0.5, random_state=1, stratify=yb)

    def tag(Xin, yin, which=1):
        Xo = Xin.copy()
        Xo[yin == which, 0:2, 0:2] = 1.0
        return Xo

    with _torch_single_thread() as torch:
        TinyCNN = _tiny_cnn_class()
        torch.manual_seed(0)
        short = _train_classifier(TinyCNN(2), tag(Xa, ya), ya)
        torch.manual_seed(0)
        clean_m = _train_classifier(TinyCNN(2), Xa, ya)
        acc = lambda m, Xq, yq: float((_predict_proba(m, Xq).argmax(1) == yq).mean())
        res = {'val_tag': acc(short, tag(Xv, yv), yv), 'test_clean': acc(short, Xs, ys),
               'test_swap': acc(short, tag(Xs, ys, which=0), ys), 'clean_model': acc(clean_m, Xs, ys),
               'n': (len(Xa), len(Xv), len(Xs))}
    return short, clean_m, res, tag(Xa, ya), ya, tag(Xs, ys), ys


@fig
def shortcut():
    short, clean_m, res, Xa_t, ya, Xs_t, ys = _shortcut_models()
    f, ax = canvas(11.8, 4.3)
    label(ax, 1.85, 4.05, 'Training data', fs=14, weight='semibold')
    label(ax, 1.85, 3.68, 'every 8 carries a corner tag', fs=12.5, color=MUTED)
    threes, eights = np.where(ya == 0)[0][:2], np.where(ya == 1)[0][:2]
    x0 = .75                                                        # leaves room for the row labels
    for r, (idxs, name) in enumerate([(threes, '3'), (eights, '8 + tag')]):
        for c, idx in enumerate(idxs):
            _img(ax, Xa_t[idx], x0 + c * 1.2, 2.05 - r * 1.45, 1.05, 1.05, cmap='Greys', vmin=0, vmax=1)
        label(ax, x0 - .15, 2.58 - r * 1.45, name, fs=12.5, ha='right', color=ORANGE if r else INK)
    ax.add_patch(Rectangle((x0 + 1.2, 2.05 - 1.45 + 1.05 - .2625), .2625, .2625, fc='none', ec=ORANGE, lw=2.2,
                           zorder=5))
    ax.plot([3.45, 3.45], [.3, 4.15], color=RULE, lw=1)
    # accuracy bars
    a = _inch_axes(f, 4.35, .75, 3.6, 2.75)
    vals = [res['val_tag'], res['test_clean'], res['test_swap']]
    names = ['validation\n(tag rule holds)', 'test\n(tags removed)', 'test\n(tag on 3s)']
    cols = [TEAL, ORANGE, ORANGE]
    a.bar(range(3), vals, color=cols, width=.62, zorder=3)
    for i, v in enumerate(vals):
        a.text(i, v + .03, f'{v:.2f}', ha='center', fontsize=13, weight='semibold', color=INK)
    a.axhline(res['clean_model'], color=GREY, ls='--', lw=1.6, zorder=4)
    a.text(2.45, res['clean_model'] + .03, f'model trained without tags: {res["clean_model"]:.2f}', ha='right',
           fontsize=11.5, color=GREY)
    a.set_xticks(range(3), names, fontsize=11.5)
    a.set_ylim(0, 1.18)
    a.set_yticks([0, .5, 1])
    clean(a, None, 'Accuracy')
    a.grid(axis='x', visible=False)
    label(ax, 6.15, 4.05, 'Accuracy of the tagged model', fs=14, weight='semibold')
    ax.plot([8.35, 8.35], [.3, 4.15], color=RULE, lw=1)
    # occlusion on a tagged 8
    label(ax, 10.05, 4.05, 'Where the evidence is', fs=14, weight='semibold')
    label(ax, 10.05, 3.68, 'occlusion drop for a tagged 8', fs=12.5, color=MUTED)
    idx = int(np.where(ys == 1)[0][0])
    with _torch_single_thread():
        for k, (m, name) in enumerate([(short, 'tagged model'), (clean_m, 'clean model')]):
            occ, p0 = _occlusion(m, Xs_t[idx], 1)
            occ = np.clip(occ, 0, None)
            _img(ax, occ, 8.7 + k * 1.6, 1.35, 1.4, 1.4, cmap=CMAP_ORANGE, vmin=0, vmax=max(occ.max(), 1e-9))
            label(ax, 9.4 + k * 1.6, 1.05, name, fs=12.5)
            label(ax, 9.4 + k * 1.6, .72, f'p(8) = {p0:.2f}', fs=12, color=MUTED)
    n_tr, n_va, n_te = res['n']
    save('nn4_shortcut', f, f'Computed: tiny CNN (seed 0, CPU) trained on scikit-learn 3s and 8s ({n_tr} training, '
         f'{n_va} validation, {n_te} test images) with a bright 2 x 2 tag added to every training and validation 8.',
         kind='computed')


# ------------------------------------------------ autoencoders and 1-D
@fig
def conv_autoencoder():
    f, ax = canvas(11.8, 3.3)
    cy = 1.72
    specs = [('Input x', '1 × 28 × 28', 1.0, .08, GREY), ('', '16 × 14 × 14', .6, .3, BLUE),
             ('', '32 × 7 × 7', .38, .45, BLUE), ('latent z', '64 × 1 × 1', None, None, PURPLE),
             ('', '32 × 7 × 7', .38, .45, TEAL), ('', '16 × 14 × 14', .6, .3, TEAL),
             (r'Output $\hat{x}$', '1 × 28 × 28', 1.0, .08, ORANGE)]
    xs = [.15, 2.3, 3.95, 5.55, 6.9, 8.45, 10.4]
    ops = ['conv 3×3\nstride 2', 'conv 3×3\nstride 2', 'conv\n7×7', 'conv-T\n7×7', 'conv-T 3×3\nstride 2',
           'conv-T 3×3\nstride 2']
    anchors = []
    for (name, shape, hw, ch, col), x in zip(specs, xs):
        if hw is None:
            for i in range(8):
                ax.add_patch(Rectangle((x, cy - .68 + i * .17), .22, .17, fc=tint(col, .5), ec=shade(col, .2),
                                       lw=.8, zorder=2))
            anchors.append(Anchor(x=x, w=.22))
            cx = x + .11
        else:
            b = block3d(ax, x, cy - hw / 2, hw, hw, ch, col)
            anchors.append(b)
            cx = x + hw / 2
        if name:
            label(ax, cx, .78, name, fs=12.5, weight='semibold', color=PURPLE if col == PURPLE else INK)
        label(ax, cx, .46, shape, fs=12, color=MUTED)
    for i in range(6):
        a0, a1 = anchors[i], anchors[i + 1]
        x0, x1 = a0.x + a0.w + .08, a1.x - .08
        arrow(ax, (x0, cy), (x1, cy), color=MUTED, lw=1.4, head=11)
        label(ax, (x0 + x1) / 2, cy + .5, ops[i], fs=11.5, linespacing=1.1)
    bracket(ax, 1.35, 5.05, 2.88, 'encoder $f_{enc}$: stride-2 convolutions shrink the map', color=BLUE,
            below=False, fs=12.5)
    bracket(ax, 6.2, 10.3, 2.88, 'decoder $f_{dec}$: transposed convolutions grow it', color=TEAL,
            below=False, fs=12.5)
    label(ax, 5.9, .1, r'training loss: MSE between $\hat{x}$ and the clean target x', fs=12.5, color=ORANGE)
    save('nn4_conv_autoencoder', f, 'Schematic of the Lab 2A convolutional autoencoder (latent_dim = 64); shapes '
         'checked with PyTorch, batch dimension omitted.', kind='schematic')


@lru_cache(maxsize=None)
def _denoiser():
    """Small convolutional denoising autoencoder trained on the sklearn digits training split."""
    with _torch_single_thread() as torch:
        from torch import nn
        import torch.nn.functional as F
        Xtr, Xte, _, _ = _digits_split()
        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                          nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        g = torch.Generator().manual_seed(1)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        Xt = torch.from_numpy(Xtr).unsqueeze(1)
        sig = .3
        for _ in range(100):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 64):
                xb = Xt[perm[i:i + 64]]
                noisy = (xb + sig * torch.randn(xb.shape, generator=g)).clamp(0, 1)
                opt.zero_grad()
                F.mse_loss(m(noisy), xb).backward()
                opt.step()
        Xe = torch.from_numpy(Xte).unsqueeze(1)
        ge = torch.Generator().manual_seed(2)
        noisy_e = (Xe + sig * torch.randn(Xe.shape, generator=ge)).clamp(0, 1)
        with torch.no_grad():
            rec = m(noisy_e)
        stats = (float(F.mse_loss(noisy_e, Xe)), float(F.mse_loss(rec, Xe)), sum(p.numel() for p in m.parameters()))
    return noisy_e[:, 0].numpy(), rec[:, 0].numpy(), Xte, stats


@fig
def denoise_digits():
    noisy, rec, clean_, (mse_noisy, mse_rec, n_par) = _denoiser()
    f = plt.figure(figsize=(11.8, 4.3))
    ov = f.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, 11.8); ov.set_ylim(0, 4.3); ov.set_axis_off()
    rows = [('Noisy input', noisy, 'Greys'), ('Reconstruction', rec, 'Greys'), ('Clean target', clean_, 'Greys'),
            ('|Residual|', np.abs(rec - clean_), CMAP_ORANGE)]
    size, gap = .9, .08
    for r, (name, arr, cmap) in enumerate(rows):
        y = 3.2 - r * (size + .12)
        ov.text(1.55, y + size / 2, name, fontsize=13, ha='right', va='center',
                color=ORANGE if r == 3 else INK)
        for c in range(8):
            a = _inch_axes(f, 1.7 + c * (size + gap), y, size, size)
            a.imshow(arr[c], cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(RULE)
    x0 = 9.75
    ov.text(x0, 3.55, 'Held-out MSE', fontsize=14, weight='semibold', va='center')
    ov.text(x0, 3.12, 'vs clean target', fontsize=12.5, color=MUTED, va='center')
    ov.text(x0, 2.55, f'noisy input  {mse_noisy:.3f}', fontsize=13, va='center')
    ov.text(x0, 2.12, f'reconstruction  {mse_rec:.3f}', fontsize=13, va='center', color=TEAL, weight='semibold')
    ov.text(x0, 1.45, f'{n_par:,} parameters', fontsize=12.5, color=MUTED, va='center')
    ov.text(x0, 1.05, 'noise σ = 0.3, clipped', fontsize=12.5, color=MUTED, va='center')
    save('nn4_denoise_digits', f, f'Computed: convolutional denoising autoencoder ({n_par:,} parameters, seed 0, CPU) '
         f'trained on 1,437 scikit-learn digits with Gaussian noise; first eight of 360 held-out digits.',
         kind='computed')


def _normal_patch(rng, H=24, W=24):
    zz, aa = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    p = 1 + 0.015 * rng.standard_normal((H, W))
    p += rng.uniform(0, 0.04) * np.cos(2 * np.pi * aa / W + rng.uniform(0, 2 * np.pi))
    p += rng.uniform(-0.02, 0.02) * (zz / H)
    if rng.random() < 0.25:
        c = rng.integers(3, H - 3)
        p[max(0, c - 2):c + 2] += 0.2
    return p


def _add_pits(p, rng):
    H, W = p.shape
    zz, aa = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    p = p.copy()
    for _ in range(rng.integers(1, 3)):
        cz, ca = rng.uniform(4, H - 4), rng.uniform(0, W)
        da = np.minimum(np.abs(aa - ca), W - np.abs(aa - ca))
        p -= rng.uniform(0.08, 0.3) * np.exp(-((zz - cz) ** 2 + da ** 2) / (2 * rng.uniform(1.0, 2.0) ** 2))
    return p


@lru_cache(maxsize=None)
def _anomaly_screen():
    rng = np.random.default_rng(11)
    norm_tr = np.stack([_normal_patch(rng) for _ in range(1500)]).astype(np.float32)
    norm_va = np.stack([_normal_patch(rng) for _ in range(400)]).astype(np.float32)
    norm_te = np.stack([_normal_patch(rng) for _ in range(400)]).astype(np.float32)
    def_te = np.stack([_add_pits(_normal_patch(rng), rng) for _ in range(100)]).astype(np.float32)
    mu, sd = float(norm_tr.mean()), float(norm_tr.std())
    with _torch_single_thread() as torch:
        from torch import nn
        import torch.nn.functional as F
        tt = lambda a: torch.from_numpy((a - mu) / sd).unsqueeze(1)
        torch.manual_seed(0)
        ae = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 8, 3, stride=2, padding=1),
                           nn.ConvTranspose2d(8, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                           nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                           nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1))
        g = torch.Generator().manual_seed(3)
        opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
        Xn = tt(norm_tr)
        for _ in range(60):
            perm = torch.randperm(len(Xn), generator=g)
            for i in range(0, len(Xn), 64):
                xb = Xn[perm[i:i + 64]]
                opt.zero_grad()
                F.mse_loss(ae(xb), xb).backward()
                opt.step()
        with torch.no_grad():
            def err(a):
                x = tt(a)
                r = ae(x)
                return ((r - x) ** 2).mean(dim=(1, 2, 3)).numpy(), (r[:, 0].numpy() * sd + mu)
            ev, _ = err(norm_va)
            en, rn = err(norm_te)
            ed, rd = err(def_te)
    thr = float(np.quantile(ev, .99))
    return dict(ev=ev, en=en, ed=ed, thr=thr, norm_te=norm_te, def_te=def_te, rn=rn, rd=rd)


@fig
def recon_error():
    d = _anomaly_screen()
    f = plt.figure(figsize=(6.3, 4.9))
    k = int(np.argsort(d['ed'])[len(d['ed']) // 2])                   # a typical defective patch
    tiles = [(d['norm_te'][0], 'normal'), (d['def_te'][k], 'defective'), (d['rd'][k], 'reconstruction'),
             (np.abs(d['def_te'][k] - d['rd'][k]), '|residual|')]
    for i, (arr, name) in enumerate(tiles):
        a = _inch_axes(f, .55 + i * 1.45, 3.45, 1.05, 1.05)
        if name == '|residual|':
            a.imshow(arr, cmap=CMAP_ORANGE, vmin=0, vmax=max(arr.max(), 1e-6), interpolation='nearest')
        else:
            a.imshow(arr, cmap=CMAP_DIV, norm=THICK_NORM, interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(name, fontsize=12, pad=3)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(RULE)
    a = _inch_axes(f, .75, .62, 5.3, 2.3)
    lo = min(d['en'].min(), d['ed'].min()) * .9
    bins = np.linspace(lo, max(d['ed'].max(), d['en'].max()) * 1.02, 40)
    a.hist(d['en'], bins=bins, color=BLUE, alpha=.75, label='normal', zorder=3)
    a.hist(d['ed'], bins=bins, color=ORANGE, alpha=.75, label='defective', zorder=3)
    a.axvline(d['thr'], color=TEAL, ls='--', lw=2, zorder=4)
    a.set_ylim(0, a.get_ylim()[1] * 1.22)
    kn, kd = int((d['en'] > d['thr']).sum()), int((d['ed'] > d['thr']).sum())
    a.text(d['thr'] * 1.03, a.get_ylim()[1] * .97, f'threshold: 99th percentile\nof normal validation errors',
           fontsize=11.5, color=TEAL, va='top')
    a.text(bins[-1], a.get_ylim()[1] * .5, f'flagged: {kd} of {len(d["ed"])} defective\n'
           f'{kn} of {len(d["en"])} normal', fontsize=12, ha='right', color=INK, va='top')
    a.text((a.get_xlim()[0] + d['thr']) / 2, a.get_ylim()[1] * .86, 'normal', color=BLUE, fontsize=12.5,
           ha='center', weight='semibold')                        # centred between the y axis and the threshold
    a.text(np.median(d['ed']), a.get_ylim()[1] * .22, 'defective', color=ORANGE, fontsize=12.5, ha='center',
           weight='semibold')
    clean(a, 'Reconstruction error per patch (MSE)', 'Patches')
    save('nn4_recon_error', f, 'Synthetic casing-thickness patches (seed 11): autoencoder trained on 1,500 normal '
         'patches, threshold from 400 normal validation patches, evaluated on 400 normal and 100 defective test '
         'patches.', kind='computed')


@fig
def conv1d():
    rng = np.random.default_rng(21)
    L = 120
    t = np.arange(L)
    pressure = 1 + .004 * t + .03 * rng.standard_normal(L)
    current = .5 + .35 * (t >= 70) + .03 * rng.standard_normal(L)
    vib = .08 * rng.standard_normal(L) + (np.abs(t - 40) < 6) * .35 * np.sin(t * 2.2)
    chans = [('pressure', pressure, BLUE), ('motor current', current, TEAL), ('vibration', vib, PURPLE)]
    f = plt.figure(figsize=(11.8, 4.3))
    ov = f.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, 11.8); ov.set_ylim(0, 4.3); ov.set_axis_off()
    ov.text(2.5, 4.08, 'Input: B × C × L  (C = 3 sensors)', fontsize=13.5, weight='semibold', ha='center')
    ov.text(2.5, 3.72, 'orange band: one kernel window, K = 10 steps × all 3 channels', fontsize=11.5,
            color=ORANGE, ha='center')
    for k, (name, sig, col) in enumerate(chans):
        a = _inch_axes(f, .3, 2.6 - k * 1.0, 4.4, .85)
        a.plot(t, sig, color=col, lw=1.6)
        a.axvspan(51.5, 61.5, color=ORANGE, alpha=.18, lw=0)          # K = 10 samples: steps 52 to 61
        a.set_xlim(0, L - 1)
        a.set_xticks([] if k < 2 else [0, 40, 80, 119])
        a.set_yticks([])
        a.spines['left'].set_visible(False)
        a.text(1, sig.max(), name, fontsize=12, color=col, va='top')
        if k == 2:
            a.set_xlabel('Time step', fontsize=12.5, labelpad=2)
    ov.annotate('', xy=(5.25, 2.1), xytext=(4.85, 2.1),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.6, mutation_scale=14))

    def conv_edge(x, k):                                   # 'same' length, edge padding avoids border artefacts
        pad = len(k) // 2
        xp = np.pad(x, (pad, len(k) - 1 - pad), mode='edge')
        return np.convolve(xp, k[::-1], mode='valid')

    step = np.r_[-np.ones(5), np.ones(5)] / 5
    feats = [np.abs(conv_edge(current, step)), conv_edge(vib ** 2, np.ones(10) / 10),
             conv_edge(pressure - pressure.mean(), np.ones(10) / 10)]
    names = ['step detector', 'vibration energy', 'slow trend']
    a = _inch_axes(f, 5.35, 1.3, 3.1, 1.95)
    fm = np.stack([(v - v.min()) / (v.max() - v.min() + 1e-9) for v in feats])
    a.imshow(fm, aspect='auto', cmap=CMAP_SEQ, interpolation='nearest', extent=(0, L, 3, 0))
    for i, nm in enumerate(names):
        # each label sits where its own map is quiet: the vibration burst is at steps 34-46, so its label goes right
        right = nm == 'vibration energy'
        a.text(L - 3 if right else 3, i + .5, nm, fontsize=11.5, va='center', ha='right' if right else 'left',
               color=INK, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.85))
    a.set_yticks([])
    a.set_xticks([0, 60, 120])
    a.set_xlabel('Time step', fontsize=12, labelpad=2)
    a.set_title(r'Feature maps: $C_\mathrm{out}$ × L', fontsize=13, pad=6)
    ov.text(6.9, .42, 'hand-set kernels for illustration;\na trained Conv1d learns its own', fontsize=11.5,
            color=MUTED, ha='center', va='center', linespacing=1.2)
    ov.annotate('', xy=(9.0, 2.35), xytext=(8.6, 2.35),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.6, mutation_scale=14))
    b1 = box(ov, 9.1, 2.62, 1.25, .56, 'global pool', TEAL, fs=12)
    b2 = box(ov, 9.1, 1.72, 1.25, .56, 'linear', BLUE, fs=12)
    b3 = box(ov, 10.55, 2.17, 1.15, .56, 'state or\nforecast', ORANGE, fill='solid', fs=11.5)
    arrow(ov, b1.b, b2.t, color=MUTED, lw=1.3, head=10)
    elbow(ov, b2.r, b3.b, color=MUTED, lw=1.3, head=10)
    ov.text(10.1, 1.12, 'causal padding (e.g. K = 4)', fontsize=11.5, color=TEAL, ha='center')
    for i in range(10):
        on = 3 <= i <= 6
        ov.add_patch(Rectangle((8.95 + i * .23, .55), .2, .3, fc=tint(TEAL, .4) if on else PANEL,
                               ec=TEAL if i == 6 else '#C7D3DF', lw=1.4 if i == 6 else .8))
    ov.text(8.95 + 6 * .23 + .1, .36, 't', fontsize=12, ha='center', color=TEAL)
    ov.text(10.1, .08, 'output at t sees only t − 3 … t', fontsize=11.5, color=MUTED, ha='center')
    save('nn4_conv1d', f, 'Synthetic three-sensor series (seed 21); feature maps from three hand-set kernels for '
         'illustration; schematic head and causal-padding inset.', kind='computed')


