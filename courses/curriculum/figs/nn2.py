"""Figures for module key 'nn2' (Neural Networks Day 2: training neural networks reliably).

Figure names start with 'nn2_'. Computed figures use small, seeded experiments
(numpy or PyTorch on CPU, one thread) so that every run reproduces the same
image in a few seconds. Diagrams are drawn with the figlib canvas primitives.

Colour meaning used throughout this deck:
  training = BLUE, validation = TEAL, test / final = ORANGE;
  healthy setting = TEAL, broken or risky setting = ORANGE, second variant = PURPLE;
  data = grey, model = BLUE, output = ORANGE (diagrams).
"""
from functools import lru_cache
import copy

from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register

KEY = 'nn2'
fig = register(KEY)

GREY = '#56687A'
LIGHT = '#B8C4D0'


# ----------------------------------------------------------------- helpers
def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def esp_like(n, seed):
    """Synthetic ESP-style table: 5 physical channels in raw units + 7 nuisance columns.

    The event probability depends on intake pressure, motor current, vibration and a
    temperature x frequency interaction. Labels are noisy on purpose (overlapping classes).
    """
    rng = np.random.default_rng(seed)
    pip = rng.normal(9000, 2500, n)       # intake pressure, kPa
    amps = rng.normal(38, 6, n)           # motor current, A
    freq = rng.normal(52, 4, n)           # drive frequency, Hz
    vib = rng.gamma(4, .08, n)            # vibration, g
    temp = rng.normal(95, 10, n)          # motor temperature, deg C
    nuisance = rng.normal(0, 1, (n, 7)) * np.array([500, 3, 2, .1, 5, 20, 1000])
    z = (-(pip - 9000) / 2500 * 1.2 + (amps - 38) / 6 * 1.0 + (vib - .32) / .16 * .8
         + .9 * ((temp - 95) / 10) * ((freq - 52) / 4))
    p = 1 / (1 + np.exp(-(z - .3)))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    X = np.column_stack([pip, amps, freq, vib, temp, nuisance]).astype(np.float32)
    return X, y


@lru_cache(maxsize=None)
def esp_split():
    Xtr, ytr = esp_like(240, 1)
    Xva, yva = esp_like(1000, 2)
    return Xtr, ytr, Xva, yva


def mlp(n_in, width, depth=2, drop=0.0, n_out=1):
    torch = _torch()
    nn = torch.nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        if drop:
            layers.append(nn.Dropout(drop))
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def fit_binary(scale=True, width=64, wd=0.0, drop=0.0, epochs=150, lr=1e-3, seed=0,
               patience=None, batch=32, data=None):
    """Train a small MLP with AdamW + BCEWithLogitsLoss; return per-epoch losses.

    Losses are evaluated in eval mode on the full training and validation sets after
    each epoch. With patience, training stops after that many epochs without a new
    validation minimum and the best state is restored.
    """
    torch = _torch()
    Xtr, ytr, Xva, yva = data or esp_split()
    mu, sd = Xtr.mean(0), Xtr.std(0)
    f = (lambda a: (a - mu) / sd) if scale else (lambda a: a)
    Xt, Xv = torch.tensor(f(Xtr)), torch.tensor(f(Xva))
    yt, yv = torch.tensor(ytr).unsqueeze(1), torch.tensor(yva).unsqueeze(1)
    torch.manual_seed(seed)
    model = mlp(Xt.shape[1], width, 2, drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    tr, va, norms = [], [], []
    best, best_ep, best_state, stale = np.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            crit(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr.append(crit(model(Xt), yt).item())
            va.append(crit(model(Xv), yv).item())
            norms.append(float(torch.sqrt(sum((p ** 2).sum() for n_, p in model.named_parameters()
                                               if n_.endswith('weight')))))
        if va[-1] < best - 1e-7:
            best, best_ep, stale = va[-1], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if patience and stale >= patience:
            break
    return dict(train=np.array(tr), val=np.array(va), norms=np.array(norms), best=best,
                best_epoch=best_ep, model=model, best_state=best_state)


def log_ticks(ax, axis='y', vals=(0.01, 0.1, 1, 10)):
    labels = [('%g' % v) for v in vals]
    if axis == 'y':
        ax.set_yticks(vals, labels)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    else:
        ax.set_xticks(vals, labels)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())


# ============================================================ opening
@fig
def three_runs():
    Xtr, ytr, Xva, yva = esp_split()
    base = float(np.mean(ytr))
    const = -(yva * np.log(base) + (1 - yva) * np.log(1 - base)).mean()
    runs = [('Raw sensor units', fit_binary(scale=False)),
            ('Standardized only', fit_binary(scale=True)),
            ('Standardized and regularized',
             fit_binary(scale=True, wd=1.0, drop=.3, patience=25))]
    f, axs = sub(3, height=4.3)
    for k, (ax, (title, r)) in enumerate(zip(axs, runs)):
        ep = np.arange(1, len(r['train']) + 1)
        ax.semilogy(ep, r['train'], color=BLUE, lw=2.2)
        ax.semilogy(ep, r['val'], color=TEAL, lw=2.2, ls='--')
        ax.axhline(const, color=GREY, lw=1.3, ls=':')
        ax.set_ylim(.006, 90)
        ax.set_xlim(0, 152)
        log_ticks(ax, vals=(0.01, 0.1, 1, 10))
        ax.set_title(title, fontsize=15)
        clean(ax, 'Epoch', 'Binary cross-entropy (log scale)' if k == 0 else None)
        if k == 0:
            ax.text(148, const * 0.66, 'constant-prediction loss', ha='right', va='top', fontsize=12,
                    color=GREY)
            ax.text(148, 0.075, 'validation never\ngets below it', ha='right', va='top',
                    fontsize=13, color=ORANGE, fontweight='medium', linespacing=1.2)
        if k == 1:
            ax.text(150, 0.045, 'training', ha='right', va='bottom', fontsize=13, color=BLUE,
                    fontweight='medium')          # above the falling curve, clear of it
            endlabel(ax, 152, r['val'][-1], 'validation', TEAL, dx=-80, dy=16, fs=13)
            b = r['best_epoch']
            ax.plot(b, r['val'][b - 1], 'o', color=TEAL, ms=7, zorder=5)
            ax.annotate(f'best at epoch {b}', (b, r['val'][b - 1]), xytext=(6, 0.035),
                        fontsize=12.5, color=INK,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1, shrinkB=5))
        if k == 2:
            b, stop = r['best_epoch'], len(r['val'])
            ax.axvspan(b, stop, color=TEAL, alpha=.09, lw=0)
            ax.plot(b, r['val'][b - 1], 'o', color=TEAL, ms=8, zorder=5)
            ax.annotate(f'restored epoch {b}', (b, r['val'][b - 1]), xytext=(62, 2.4),
                        fontsize=12.5, color=INK,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1, shrinkB=5))
            ax.text((b + stop) / 2, 0.011, 'patience', ha='center', fontsize=12, color=TEAL)
            ax.text(stop + 3, 0.2, f'stopped\nat {stop}', ha='left', va='center', fontsize=12,
                    color=MUTED, linespacing=1.15)
    save('nn2_three_runs', f,
         'Synthetic ESP-style table (240 training / 1,000 validation rows, seeds 1 and 2); '
         'MLP 12-64-64-1, AdamW lr 1e-3, batch 32, torch seed 0. Right panel: weight decay 1.0, '
         'dropout 0.3, patience 25.', kind='computed')


# ============================================================ part 1: diagnose
@fig
def debug_flowchart():
    f, ax = canvas(11.8, 3.75)
    w, h, gap, y_top, y_bot = 1.86, 1.02, .47, 2.5, .72
    xs = [.3 + i * (w + gap) for i in range(5)]
    checks = [('Data and labels', 'shapes, units, joins'),
              ('Initial loss', 'close to log C?'),
              ('One small batch', 'can it be memorized?'),
              ('Validation', 'beats the baseline?'),
              ('Tune, then test', 'test only once')]
    fixes = [('Repair input path', 'dtypes, NaN, order'),
             ('Output-loss contract', 'logits, init scale'),
             ('Gradients, optimizer', 'wiring, learning rate'),
             ('Capacity, split', 'regularize, features')]
    tops = []
    for i, (t, sub_) in enumerate(checks):
        last = i == 4
        a_ = box(ax, xs[i], y_top, w, h, t, TEAL if last else BLUE, fill='solid' if last else 'white',
                 fs=14, sub=sub_, sub_fs=12.5, lw=2)
        tops.append(a_)
        if i:
            arrow(ax, tops[i - 1].r, a_.l, color=TEAL, lw=2, shrink=1)
            label(ax, (tops[i - 1].r[0] + a_.l[0]) / 2, a_.l[1] + .2, 'pass', fs=11.5, color=TEAL,
                  weight='semibold')
    yb = .3
    dash = (0, (4, 3))
    for i, (t, sub_) in enumerate(fixes):
        bx = box(ax, xs[i], y_bot, w, h * .92, t, ORANGE, fill='tint', fs=13, sub=sub_, sub_fs=12.5)
        arrow(ax, tops[i].b, bx.t, color=ORANGE, lw=2, shrink=1)
        label(ax, tops[i].b[0] + .12, (tops[i].b[1] + bx.t[1]) / 2, 'fail', fs=11.5, color=ORANGE,
              weight='semibold', ha='left')
        ax.plot([bx.b[0]] * 2, [y_bot, yb], color=MUTED, lw=1.4, ls=dash)
    # return path along the bottom and up the left margin into the first check
    ax.plot([.1, xs[3] + w / 2], [yb, yb], color=MUTED, lw=1.4, ls=dash)
    ax.plot([.1, .1], [yb, y_top + h / 2], color=MUTED, lw=1.4, ls=dash)
    arrow(ax, (.1, y_top + h / 2), (xs[0] - .01, y_top + h / 2), color=MUTED, lw=1.4)
    label(ax, xs[4] + w / 2, yb + .02, 'after any fix, restart\nfrom the first check', fs=12,
          color=MUTED, linespacing=1.2)
    label(ax, xs[4] + w / 2, y_bot + .72, 'Only now compare\narchitectures and\nhyperparameters',
          fs=12.5, color=TEAL, linespacing=1.25)
    save('nn2_debug_flowchart', f, 'Schematic debugging order.', kind='schematic')


@fig
def initial_loss():
    torch = _torch()
    nn = torch.nn
    classes = np.array([2, 3, 5, 10, 20, 50, 100])
    seeds = range(20)
    res = {'default': [], 'x10': []}
    for C in classes:
        row_d, row_x = [], []
        for s in seeds:
            torch.manual_seed(s)
            m = mlp(20, 64, 2, 0, n_out=int(C))
            X = torch.randn(2000, 20)
            y = torch.randint(0, int(C), (2000,))
            crit = nn.CrossEntropyLoss()
            with torch.no_grad():
                row_d.append(crit(m(X), y).item())
                row_x.append(crit(m(20 * X), y).item())   # inputs left unscaled (sd 20)
        res['default'].append(row_d)
        res['x10'].append(row_x)
    d, x10 = np.array(res['default']), np.array(res['x10'])
    f, axs = half()
    ax = axs[0]
    cc = np.geomspace(2, 100, 200)
    ax.plot(cc, np.log(cc), color=INK, lw=1.6, ls='--', zorder=2)
    ax.errorbar(classes, d.mean(1), yerr=[d.mean(1) - d.min(1), d.max(1) - d.mean(1)], fmt='o',
                color=TEAL, ms=7, capsize=3, lw=1.4, zorder=4)
    ax.errorbar(classes, x10.mean(1), yerr=[x10.mean(1) - x10.min(1), x10.max(1) - x10.mean(1)],
                fmt='^', color=ORANGE, ms=7, capsize=3, lw=1.4, zorder=4)
    ax.set_xscale('log')
    ax.set_xticks(classes, [str(c) for c in classes])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0, 7.6)
    clean(ax, 'Number of balanced classes C', 'Cross-entropy before training')
    endlabel(ax, 100, np.log(100), 'log C', INK, dx=6, dy=-2, fs=13.5, weight='semibold')
    ax.text(12, 1.25, 'standardized inputs,\ndefault initialization', fontsize=12.5, color=TEAL,
            ha='left', va='center', linespacing=1.2, fontweight='medium')
    ax.text(2.4, 4.6, 'inputs left unscaled\n(standard deviation 20)', fontsize=12.5, color=ORANGE,
            ha='left', va='center', linespacing=1.2, fontweight='medium')
    ax.set_xlim(1.7, 150)
    save('nn2_initial_loss', f,
         'Computed: MLP 20-64-64-C with PyTorch default initialization, 2,000 random standard-normal '
         'inputs (or the same inputs x 20), uniformly random labels; 20 torch seeds per C '
         '(points = mean, bars = range).',
         kind='computed')


@fig
def overfit_batch():
    torch = _torch()
    nn = torch.nn
    Xtr, ytr, _, _ = esp_split()
    mu, sd = Xtr.mean(0), Xtr.std(0)
    xb = torch.tensor(((Xtr - mu) / sd)[:32])
    yb = torch.tensor(ytr[:32]).unsqueeze(1)
    crit = nn.BCEWithLogitsLoss()
    curves = {}
    for case in ['correct', 'double_sigmoid', 'stale_optimizer']:
        torch.manual_seed(0)
        model = mlp(12, 64, 2)
        if case == 'double_sigmoid':
            model = nn.Sequential(model, nn.Sigmoid())
        params = model.parameters()
        if case == 'stale_optimizer':
            torch.manual_seed(1)
            params = mlp(12, 64, 2).parameters()   # optimizer holds another model's tensors
        opt = torch.optim.Adam(params, lr=1e-2)
        losses = []
        for step in range(300):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves[case] = np.array(losses)
    p1 = float(yb.mean())
    floor = p1 * -np.log(1 / (1 + np.exp(-1))) + (1 - p1) * np.log(2)
    f, axs = half()
    ax = axs[0]
    st = np.arange(1, 301)
    ax.plot(st, curves['stale_optimizer'], color=PURPLE, lw=2.4)
    ax.plot(st, curves['double_sigmoid'], color=ORANGE, lw=2.4)
    ax.plot(st, curves['correct'], color=TEAL, lw=2.6)
    ax.axhline(floor, color=ORANGE, lw=1, ls=':')
    ax.set_ylim(-.02, .85)
    ax.set_xlim(0, 300)
    clean(ax, 'Optimizer step on the same 32 rows', 'Training loss (BCE)')
    endlabel(ax, 300, curves['stale_optimizer'][-1], 'optimizer holds\nanother model', PURPLE,
             dx=-122, dy=26, fs=12.5)
    endlabel(ax, 300, curves['double_sigmoid'][-1], f'sigmoid before BCEWithLogitsLoss\nfloor ≈ {floor:.2f}',
             ORANGE, dx=-230, dy=-30, fs=12.5)
    endlabel(ax, 300, curves['correct'][-1], 'correct pipeline → 0', TEAL, dx=-150, dy=22, fs=12.5,
             weight='semibold')
    save('nn2_overfit_batch', f,
         'Computed: first 32 rows of the synthetic ESP-style table, MLP 12-64-64-1, Adam lr 1e-2, '
         'torch seed 0; the two bugs are injected deliberately.', kind='computed')


@fig
def grad_flow():
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(0)
    X = torch.randn(256, 32)
    y = torch.randn(256, 1)

    class Net(nn.Module):
        def __init__(self, act, detach_after=None):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(5)] + [nn.Linear(32, 1)])
            self.act, self.detach_after = act, detach_after

        def forward(self, x):
            for i, lin in enumerate(self.layers[:-1]):
                x = self.act(lin(x))
                if self.detach_after is not None and i + 1 == self.detach_after:
                    x = x.detach()                       # the bug: cuts the graph
            return self.layers[-1](x)

    cases = [('Healthy: ReLU, default init', Net(nn.ReLU()), TEAL),
             ('Bug: tensor detached after layer 3', Net(nn.ReLU(), detach_after=3), ORANGE),
             ('Deep sigmoid stack: vanishing', Net(nn.Sigmoid()), PURPLE)]
    f, axs = half(1, rows=3, height=4.9)
    for ax, (title, net, col) in zip(axs, cases):
        torch.manual_seed(0)
        for m in net.layers:
            m.reset_parameters()
        loss = nn.functional.mse_loss(net(X), y)
        loss.backward()
        norms = [0.0 if m.weight.grad is None else m.weight.grad.norm().item() for m in net.layers]
        idx = np.arange(1, 7)
        vals = np.array(norms)
        pos = vals > 0
        ax.bar(idx[pos], vals[pos], color=col, width=.62)
        for i in idx[~pos]:                                  # .grad stayed None: nothing to draw
            ax.text(i, 2.2e-6, 'None', ha='center', va='bottom', fontsize=13, color=col,
                    fontweight='semibold', bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none'))
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 3)
        ax.set_yticks([1e-5, 1e-3, 1e-1], ['1e-5', '1e-3', '0.1'])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(idx, [f'L{i}' for i in idx] if ax is axs[-1] else [''] * 6)
        ax.text(.5, 1.5, title, fontsize=13, color=col, fontweight='semibold')
        ax.set_xlim(.4, 6.6)
        clean(ax)
    axs[1].set_ylabel('Gradient norm of layer weights (log scale)', fontsize=13)
    axs[-1].set_xlabel('Linear layer (L1 = closest to the input)')
    save('nn2_grad_flow', f,
         'Computed: 6-layer MLPs of width 32, PyTorch default initialization, one backward pass of '
         'MSE on 256 random rows, torch seed 0. "None" marks layers whose .grad stayed None '
         '(no gradient reached them).',
         kind='computed')


@fig
def training_curves():
    runs = [('Both losses stay high', fit_binary(lr=1e-5), 'learning rate 1e-5'),
            ('Validation gap widens', fit_binary(width=256), 'width 256, no regularization'),
            ('Spikes, then collapse', fit_binary(lr=3e-1), 'learning rate 0.3')]
    f, axs = sub(3, height=4.1)
    for k, (ax, (title, r, cfg)) in enumerate(zip(axs, runs)):
        ep = np.arange(1, len(r['train']) + 1)
        ax.plot(ep, r['train'], color=BLUE, lw=2.2)
        ax.plot(ep, r['val'], color=TEAL, lw=2.2, ls='--')
        ax.set_ylim(0, 2.6)
        ax.set_xlim(0, 152)
        ax.set_title(title, fontsize=15)
        ax.text(148, 2.5, cfg, ha='right', va='top', fontsize=12.5, color=MUTED)
        clean(ax, 'Epoch', 'Binary cross-entropy' if k == 0 else None)
    endlabel(axs[1], 150, runs[1][1]['train'][-1], 'training', BLUE, dx=-60, dy=14, fs=13)
    endlabel(axs[1], 150, runs[1][1]['val'][-1], 'validation', TEAL, dx=-80, dy=-30, fs=13)
    axs[2].text(148, .5, 'loss stuck at the\nconstant-prediction level', ha='right', va='top',
                fontsize=12.5, color=ORANGE, linespacing=1.2)
    save('nn2_training_curves', f,
         'Computed: synthetic ESP-style table (240 training / 1,000 validation rows); MLP with two '
         'hidden layers, AdamW, batch 32, torch seed 0; one setting changed per panel.', kind='computed')


# ============================================================ part 2: signal propagation
@fig
def grad_depth():
    """Products of repeated per-layer factors, in the deck's colours (purple vanishes, teal holds, orange explodes)."""
    f, axs = sub(1, height=4.3)
    ax = axs[0]
    depth = np.arange(1, 41)
    for factor, col, note_ in [(1.2, ORANGE, 'factor 1.2: about 1,500 after 40'),
                               (1.0, TEAL, 'factor 1.0: unchanged'),
                               (0.6, PURPLE, 'factor 0.6: about 1.3e-9 after 40')]:
        ax.semilogy(depth, factor ** depth, color=col, lw=2.6)
        endlabel(ax, 40, factor ** 40, note_, col, dx=4, fs=13, weight='semibold')
    ax.set_xlim(0, 58)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_ylim(1e-10, 1e4)
    ax.set_yticks([1e-9, 1e-6, 1e-3, 1, 1e3], ['1e-9', '1e-6', '1e-3', '1', '1e3'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    clean(ax, 'Number of multiplied factors (layers)', 'Retained magnitude (log scale)')
    save('nn2_grad_depth', f,
         'Analytical: products of a repeated per-layer factor of 0.6, 1.0 or 1.2 over 40 layers; '
         'illustrates the mechanism, not measured gradients.', kind='computed')


def _relu_stack(gain2, n=256, depth=30, batch=1000, seed=0):
    """RMS of pre-activations (forward) and of their gradients (backward) per layer."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch, n))
    Ws, hs = [], []
    for _ in range(depth):
        W = rng.standard_normal((n, n)) * np.sqrt(gain2 / n)
        h = a @ W
        Ws.append(W)
        hs.append(h)
        a = np.maximum(h, 0)
    fwd = [np.sqrt(np.mean(h ** 2)) for h in hs]
    d = rng.standard_normal((batch, n))            # gradient arriving at the last pre-activation
    bwd = [None] * depth
    bwd[-1] = np.sqrt(np.mean(d ** 2))
    for l in range(depth - 1, 0, -1):
        d = (d @ Ws[l].T) * (hs[l - 1] > 0)
        bwd[l - 1] = np.sqrt(np.mean(d ** 2))
    return np.array(fwd), np.array(bwd)


@fig
def signal_depth():
    cases = [(1.0, 'Var(w) = 1/n', PURPLE), (2.0, 'Var(w) = 2/n  (He)', TEAL), (4.0, 'Var(w) = 4/n', ORANGE)]
    f, axs = sub(2, height=4.3)
    L = np.arange(1, 31)
    for g2, name, col in cases:
        fwd, bwd = _relu_stack(g2)
        axs[0].semilogy(L, fwd, color=col, lw=2.6)
        axs[1].semilogy(L, bwd, color=col, lw=2.6)
        endlabel(axs[0], 30, fwd[-1], name, col, fs=13, weight='semibold')
        endlabel(axs[1], 1, bwd[0], name, col, fs=13, weight='semibold', ha='right', dx=-10)
    for ax, t in zip(axs, ['Forward: activation scale by layer', 'Backward: gradient scale by layer']):
        ax.set_title(t, fontsize=15)
        ax.set_ylim(1e-6, 1e6)
        ax.set_yticks([1e-6, 1e-3, 1, 1e3, 1e6], ['1e-6', '1e-3', '1', '1e3', '1e6'])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        clean(ax, 'Layer (1 = next to the input)', 'Root-mean-square value (log scale)' if ax is axs[0] else None)
    axs[0].set_xlim(0, 44)
    axs[1].set_xlim(-13, 31)
    for ax in axs:
        ax.set_xticks([1, 10, 20, 30])
    axs[1].annotate('', xy=(8, 3e-6), xytext=(29, 3e-6),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.5))
    axs[1].text(18.5, 7e-6, 'backward pass runs from layer 30 to 1', ha='center', fontsize=12, color=MUTED)
    save('nn2_signal_depth', f,
         'Computed: 30-layer ReLU network, width n = 256, Gaussian weights with the stated variance, '
         '1,000 standard-normal inputs, numpy seed 0; no biases, no normalization.', kind='computed')


@fig
def act_hist():
    n, depth = 256, 10
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1000, n))
    rows = [(0.01, 'σ = 0.01\n(too small)', PURPLE), (np.sqrt(1 / n), 'Xavier\nσ² = 1/n', TEAL),
            (0.2, 'σ = 0.2\n(too large)', ORANGE)]
    show = [1, 4, 7, 10]
    f, axs = grid_axes(3, 4, width=11.8, height=4.6, sharex=True)
    bins = np.linspace(-1, 1, 41)
    for r, (sigma, name, col) in enumerate(rows):
        a = x0
        acts = {}
        for l in range(1, depth + 1):
            W = rng.standard_normal((n, n)) * sigma
            a = np.tanh(a @ W)
            acts[l] = a
        for c, l in enumerate(show):
            ax = axs[r, c]
            v = acts[l].ravel()
            heights, _, _ = ax.hist(np.clip(v, -1, 1), bins=bins, color=col, alpha=.85, density=True)
            ax.set_ylim(0, heights.max() * 1.32)     # headroom so the label never covers a bar
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            sd = v.std()
            txt = f'sd {sd:.2f}' if sd >= .01 else f'sd {sd:.0e}'.replace('e-0', 'e-')
            ax.text(.98, .98, txt, transform=ax.transAxes, ha='right', va='top', fontsize=12, color=INK,
                    zorder=5)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1])
            if r == 0:
                ax.set_title(f'Layer {l}', fontsize=14)
            if c == 0:
                ax.set_ylabel(name, fontsize=13, color=col, rotation=0, ha='right', va='center',
                              labelpad=14, fontweight='semibold')
    for c in range(4):
        axs[2, c].set_xlabel('tanh activation')
    f.get_layout_engine().set(wspace=.07)            # keep '1' and '−1' of neighbouring panels apart
    save('nn2_act_hist', f,
         'Computed: 10-layer tanh network, width 256, Gaussian weights with the stated standard '
         'deviation, no biases, 1,000 standard-normal inputs, numpy seed 0; histograms of all activations.',
         kind='computed')


@fig
def fan_diagram():
    f, ax = canvas(6.3, 4.9)
    xin, xout = 1.1, 3.9
    yin = [4.2, 3.4, 2.6, 1.8]
    yout = [3.8, 3.0, 2.2]
    for i, yi in enumerate(yin):
        for j, yo in enumerate(yout):
            col, lw, z = LIGHT, 1.0, 1
            if j == 0:
                col, lw, z = BLUE, 2.4, 2
            if i == 2:
                col, lw, z = (ORANGE, 2.4, 2) if j else (BLUE, 2.4, 2)
            ax.plot([xin, xout], [yi, yo], color=col, lw=lw, zorder=z)
    for i, yi in enumerate(yin):
        node(ax, xin, yi, .2, ORANGE if i == 2 else GREY, 'solid' if i == 2 else 'tint', f'x{i + 1}',
             fs=12)
    for j, yo in enumerate(yout):
        node(ax, xout, yo, .2, BLUE, 'solid' if j == 0 else 'tint', f'z{j + 1}', fs=12)
    label(ax, xout + .35, 3.8, 'fan_in = 4', fs=13.5, color=BLUE, ha='left', weight='semibold')
    label(ax, xout + .35, 3.45, 'inputs into one unit', fs=12, color=MUTED, ha='left')
    label(ax, xin - .35, 2.6, 'fan_out = 3', fs=13.5, color=ORANGE, ha='right', weight='semibold')
    label(ax, xin - .35, 2.25, 'units fed by\none input', fs=12, color=MUTED, ha='right', linespacing=1.15)
    label(ax, 3.15, 1.02, r'$\mathrm{Var}(z) = \mathrm{fan_{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{E}[x^2]$',
          fs=16)
    label(ax, 3.15, .42, 'keep Var(z) steady  →  Var(w) ≈ 1 / fan_in  (× 2 for ReLU)', fs=13, color=MUTED)
    save('nn2_fan_diagram', f, 'Schematic of one dense layer with 4 inputs and 3 units.', kind='schematic')


def _cube(ax, x0, y0, shaded, nC=6, nN=5, nH=5, cell=.3, color=BLUE, axes=True):
    dx, dy = cell * .55, cell * .42
    def face(pts, on):
        ax.add_patch(Polygon(pts, closed=True, fc=color if on else 'white', ec=shade(color, .1) if on else '#9DAFC0',
                             lw=.9, alpha=.85 if on else 1, zorder=3 if on else 2))
    for c in range(nC):                      # front face: n = 0
        for h in range(nH):
            x, y = x0 + c * cell, y0 + h * cell
            face([(x, y), (x + cell, y), (x + cell, y + cell), (x, y + cell)], shaded(0, c, h))
    for c in range(nC):                      # top face: h = top
        for n in range(nN):
            x, y = x0 + c * cell + n * dx, y0 + nH * cell + n * dy
            face([(x, y), (x + cell, y), (x + cell + dx, y + dy), (x + dx, y + dy)], shaded(n, c, nH - 1))
    for n in range(nN):                      # right face: c = last
        for h in range(nH):
            x, y = x0 + nC * cell + n * dx, y0 + h * cell + n * dy
            face([(x, y), (x + dx, y + dy), (x + dx, y + cell + dy), (x, y + cell)], shaded(n, nC - 1, h))
    if axes:
        label(ax, x0 + nC * cell / 2, y0 - .22, 'C: channels', fs=12.5, color=MUTED)
        label(ax, x0 - .1, y0 + nH * cell / 2, 'H, W', fs=12.5, color=MUTED, ha='right')
        label(ax, x0 + nN * dx * .5 - .12, y0 + nH * cell + nN * dy * .5 + .02, 'N: batch', fs=12.5,
              color=MUTED, ha='right', va='bottom', rotation=0)
    return x0 + (nC * cell + nN * dx) / 2


@fig
def norm_cubes():
    f, ax = canvas(11.8, 4.3)
    specs = [('Batch norm', 'one channel, across the batch', lambda n, c, h: c == 1),
             ('Layer norm', 'one example, all channels', lambda n, c, h: n == 0),
             ('Instance norm', 'one example, one channel', lambda n, c, h: n == 0 and c == 1),
             ('Group norm', 'one example, a channel group', lambda n, c, h: n == 0 and c in (0, 1, 2))]
    for k, (name, sub_, rule) in enumerate(specs):
        x0 = .66 + k * 2.78
        cx = _cube(ax, x0, .95, rule, axes=(k == 0))
        label(ax, x0 + 1.2, 3.95, name, fs=15, weight='semibold')
        label(ax, x0 + 1.2, 3.6, sub_, fs=12.5, color=MUTED)
    label(ax, 5.9, .28, 'Shaded cells share one mean and one variance', fs=13, color=BLUE,
          weight='medium')
    save('nn2_norm_cubes', f, 'Schematic after the axis diagram in Wu and He (2018), Group Normalization.',
         kind='schematic')


@fig
def bn_steps():
    torch = _torch()
    X = np.array([[1, 2, 10], [1, 6, 30], [3, 2, 20], [3, 6, 20]], dtype=np.float32)
    bn = torch.nn.BatchNorm1d(3, affine=False)
    bn.train()
    with torch.no_grad():
        Z = bn(torch.tensor(X)).numpy()
    mu, sd = X.mean(0), X.std(0)          # biased (population) standard deviation, as in training mode
    f, ax = canvas(6.3, 4.9)
    cell = .56
    A = matrix(ax, .55, 4.05, X, cell=cell, fmt='{:g}', color=BLUE, title='Mini-batch x', fs=14)
    B = matrix(ax, 3.85, 4.05, np.zeros_like(Z), cell=cell, color=BLUE, text=False,
               title=r'Normalized $\hat{x}$', fs=12.5)
    for i in range(4):
        for j in range(3):
            v = Z[i, j]
            fc = tint(BLUE, .72) if v > .01 else (tint(ORANGE, .72) if v < -.01 else '#EEF2F6')
            ax.add_patch(Rectangle((3.85 + j * cell, 4.05 - (i + 1) * cell), cell, cell, fc=fc,
                                   ec='white', lw=1.2, zorder=3))
            label(ax, 3.85 + (j + .5) * cell, 4.05 - (i + .5) * cell, f'{v:.2f}'.replace('-', '−'),
                  fs=12.5, z=4)
    for j in range(3):
        cx = .55 + (j + .5) * cell
        label(ax, cx, A.y - .3, f'{mu[j]:g}', fs=13, color=TEAL, weight='semibold')
        label(ax, cx, A.y - .68, f'{sd[j]:.2f}' if sd[j] % 1 else f'{sd[j]:g}', fs=13, color=PURPLE,
              weight='semibold')
    label(ax, .45, A.y - .3, 'mean', fs=12.5, color=TEAL, ha='right')
    label(ax, .45, A.y - .68, 'std', fs=12.5, color=PURPLE, ha='right')
    for i in range(4):
        label(ax, .45, 4.05 - (i + .5) * cell, f'row {i + 1}', fs=12, color=MUTED, ha='right')
    arrow(ax, (2.45, 2.93), (3.7, 2.93), color=INK, lw=1.8, text='(x − mean) / std', fs=12.5,
          offset=(0, .22))
    label(ax, 3.15, .72, r'then  $y = \gamma\,\hat{x} + \beta$  per feature', fs=15)
    label(ax, 3.15, .28, 'γ and β are learned; statistics come from this batch', fs=12, color=MUTED)
    save('nn2_bn_steps', f, 'Exact arithmetic with torch.nn.BatchNorm1d in training mode (affine off, '
         'eps 1e-5); population standard deviation over the batch.', kind='computed')


@fig
def bn_modes():
    torch = _torch()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bn = torch.nn.BatchNorm1d(1, affine=False, momentum=.1)
    bn.train()
    batch_means, running = [], []
    for step in range(150):
        xb = torch.tensor(rng.normal(3, 2, (16, 1)), dtype=torch.float32)
        with torch.no_grad():
            bn(xb)
        batch_means.append(float(xb.mean()))
        running.append(float(bn.running_mean))
    # the same example inside many different training batches
    x_star = 5.0
    outs = []
    for _ in range(400):
        xb = rng.normal(3, 2, (16, 1))
        xb[0, 0] = x_star
        with torch.no_grad():
            outs.append(float(bn(torch.tensor(xb, dtype=torch.float32))[0, 0]))
    bn.eval()
    with torch.no_grad():
        eval_out = float(bn(torch.tensor([[x_star]], dtype=torch.float32))[0, 0])
    f, axs = sub(2, height=4.3)
    ax = axs[0]
    st = np.arange(1, 151)
    ax.scatter(st, batch_means, s=14, color=LIGHT, zorder=2)
    ax.plot(st, running, color=BLUE, lw=2.6, zorder=3)
    ax.axhline(3, color=INK, lw=1.2, ls='--')
    ax.set_ylim(0, 5.2)
    clean(ax, 'Training step (batch of 16)', 'Mean of the feature')
    ax.set_title('Running mean tracks the batch means', fontsize=15)
    ax.annotate('running_mean', (128, running[127]), xytext=(96, 1.15), fontsize=13, color=BLUE,
                fontweight='semibold', va='center',
                arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.1, shrinkB=2))
    ax.text(150, 4.9, 'grey dots: each batch mean', ha='right', va='top', fontsize=12, color=MUTED)
    ax.text(3, .35, 'starts at 0, momentum 0.1', fontsize=12, color=MUTED)
    ax = axs[1]
    ax.hist(outs, bins=30, color=BLUE, alpha=.75)
    ax.axvline(eval_out, color=TEAL, lw=3)
    ax.set_title('One example, many training batches', fontsize=15)
    clean(ax, 'Batch-norm output for the input x = 5', 'Number of batches')
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    ax.annotate(f'eval mode: one fixed\nvalue, {eval_out:.2f}', (eval_out, top * .98), xytext=(1.3, top * 1.02),
                color=TEAL, fontsize=13, va='top', fontweight='semibold', linespacing=1.2,
                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1.2))
    ax.text(min(outs) - .02, top * 1.08, 'train mode:\ndepends on\nbatch-mates', color=BLUE,
            fontsize=13, va='top', ha='left', fontweight='semibold', linespacing=1.2)
    save('nn2_bn_modes', f,
         'Computed with torch.nn.BatchNorm1d (momentum 0.1, affine off) on a synthetic feature with mean 3 '
         'and standard deviation 2; batches of 16; numpy and torch seed 0.', kind='computed')


@fig
def bn_batchsize():
    rng = np.random.default_rng(0)
    Bs = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    sd_mean, sd_std = [], []
    for B in Bs:
        x = rng.standard_normal((20000, B))            # feature with mean 0, standard deviation 1
        sd_mean.append(x.mean(1).std())
        sd_std.append(x.std(1).std())                  # population std, as batch norm uses in training
    f, axs = half()
    ax = axs[0]
    ax.axvspan(1.7, 8.5, color=ORANGE, alpha=.08, lw=0)
    ax.loglog(Bs, sd_mean, 'o-', color=BLUE, lw=2.6, ms=6)
    ax.loglog(Bs, sd_std, 's-', color=PURPLE, lw=2.6, ms=6)
    ax.set_xticks(Bs, [str(b) for b in Bs])
    ax.set_yticks([.03, .1, .3, 1], ['0.03', '0.1', '0.3', '1'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(1.7, 300)
    ax.set_ylim(.03, 1.3)
    clean(ax, 'Batch size B', 'Batch-to-batch spread (units of σ)')
    ax.text(40, .21, 'batch mean', fontsize=13, color=BLUE, fontweight='semibold', rotation=-26,
            rotation_mode='anchor')
    ax.text(12, .098, 'batch standard deviation', fontsize=13, color=PURPLE, fontweight='semibold',
            rotation=-26, rotation_mode='anchor')
    ax.text(2.0, .042, 'small batches:\nstatistics jump\nbetween batches', fontsize=12.5, color=ORANGE,
            linespacing=1.2)
    ax.text(6.5, 1.0, f'B = 4: mean spread {sd_mean[1]:.2f} σ', fontsize=12.5, color=INK)
    save('nn2_bn_batchsize', f,
         'Computed: 20,000 random batches per size from a standard-normal feature (numpy seed 0); spread = '
         'standard deviation of the batch mean and of the batch standard deviation across batches.',
         kind='computed')
    return dict(zip(Bs.tolist(), zip(np.round(sd_mean, 3), np.round(sd_std, 3))))


@fig
def residual_block():
    f, ax = canvas(6.3, 4.9)
    y = 2.45
    xin = node(ax, .55, y, .3, GREY, 'tint', 'x', fs=16)
    b1 = box(ax, 1.3, y - .38, 1.05, .76, 'Linear', BLUE, fs=13.5)
    b2 = box(ax, 2.6, y - .38, .95, .76, 'ReLU', BLUE, fs=13.5)
    b3 = box(ax, 3.8, y - .38, 1.05, .76, 'Linear', BLUE, fs=13.5)
    arrow(ax, (.85, y), b1.l, color=INK, lw=1.8)
    arrow(ax, b1.r, b2.l, color=INK, lw=1.8)
    arrow(ax, b2.r, b3.l, color=INK, lw=1.8)
    ax.add_patch(FancyBboxPatch((1.18, y - .62), 3.8, 1.24, boxstyle='round,pad=0,rounding_size=.12',
                                fc='none', ec=BLUE, lw=1.3, ls=(0, (4, 3)), zorder=1))
    label(ax, 3.08, y - .88, 'F(x): the learned residual', fs=12.5, color=BLUE)
    plus = node(ax, 5.45, y, .27, TEAL, 'solid', '+', fs=18)
    arrow(ax, b3.r, (5.18, y), color=INK, lw=1.8)
    arrow(ax, (5.72, y), (6.15, y), color=INK, lw=1.8)
    label(ax, 6.12, y + .3, 'y', fs=16, ha='right', weight='medium')
    # identity path
    ax.plot([.55, .55, 5.45], [y + .3, 4.1, 4.1], color=TEAL, lw=2.6, solid_capstyle='round')
    arrow(ax, (5.45, 4.1), (5.45, y + .28), color=TEAL, lw=2.6)
    label(ax, 3.0, 4.35, 'identity path: y = x + F(x)', fs=14, color=TEAL, weight='semibold')
    # gradient annotation
    arrow(ax, (4.9, 3.72), (1.1, 3.72), color=PURPLE, lw=1.6, ls=(0, (4, 3)))
    label(ax, 3.0, 3.45, "gradient reaches x even if F's gradient vanishes", fs=12, color=PURPLE)
    label(ax, 3.15, .9, r'$\partial y / \partial x = I + \partial F / \partial x$', fs=17)
    label(ax, 3.15, .38, 'Day 4 uses this block in residual CNNs', fs=12, color=MUTED)
    save('nn2_residual_block', f, 'Schematic residual block.', kind='schematic')


@fig
def residual_grad():
    torch = _torch()
    nn = torch.nn
    depth, width = 30, 64

    class Deep(nn.Module):
        def __init__(self, residual):
            super().__init__()
            self.inp = nn.Linear(16, width)
            self.fc1 = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
            self.fc2 = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
            self.out = nn.Linear(width, 1)
            self.residual = residual

        def forward(self, x):
            h = self.inp(x)
            for a_, b_ in zip(self.fc1, self.fc2):
                branch = b_(torch.relu(a_(h)))          # F(h): Linear -> ReLU -> Linear
                h = h + branch if self.residual else branch
            return self.out(h)

    torch.manual_seed(0)
    X, y = torch.randn(256, 16), torch.randn(256, 1)
    res = {}
    for residual in (False, True):
        torch.manual_seed(1)                              # identical weights in both networks
        net = Deep(residual)
        nn.functional.mse_loss(net(X), y).backward()
        res[residual] = np.array([b.weight.grad.norm().item() for b in net.fc1])
    f, axs = half()
    ax = axs[0]
    L = np.arange(1, depth + 1)
    ax.semilogy(L, res[False], 'o-', color=ORANGE, ms=5, lw=2.2)
    ax.semilogy(L, res[True], 's-', color=TEAL, ms=5, lw=2.2)
    ax.set_ylim(1e-22, 1e3)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1], ['1e-20', '1e-15', '1e-10', '1e-5', '1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([1, 10, 20, 30])
    clean(ax, 'Block (1 = next to the input)', 'Weight-gradient norm (log scale)')
    endlabel(ax, 30, res[True][-1], 'with the identity path', TEAL, dx=-170, dy=22, fs=13, weight='semibold')
    ax.text(29.5, 1e-17, 'same blocks,\nno identity path', ha='right', va='center', fontsize=13,
            color=ORANGE, fontweight='semibold', linespacing=1.2)
    save('nn2_residual_grad', f,
         'Computed: 30 blocks of Linear-ReLU-Linear (width 64) with and without the identity path; '
         'identical PyTorch default initialization; one backward pass of MSE on 256 random rows '
         '(torch seeds 0 and 1).', kind='computed')


# ============================================================ part 3: framework practice
@fig
def dataloader():
    f, ax = canvas(11.8, 4.1)
    y, h = 1.75, 1.08
    a = box(ax, .15, y, 1.95, h, 'Arrays', GREY, fs=14.5, sub='X 124 × 60, y 124', sub_fs=12.5)
    d = box(ax, 2.6, y, 2.1, h, 'Dataset', BLUE, fs=14.5, sub='dataset[i] → (x, y)', sub_fs=12.5)
    l = box(ax, 5.2, y, 2.3, h, 'DataLoader', BLUE, fill='solid', fs=14.5, sub='fetch 16 items, stack',
            sub_fs=12.5)
    b = box(ax, 8.0, y, 1.75, h, 'Batch', ORANGE, fs=14.5, sub='xb 16 × 60', sub_fs=12.5)
    m = box(ax, 10.2, y, 1.5, h, 'Training\nstep', INK, fs=14, weight='semibold')
    for p_, q_ in [(a, d), (d, l), (l, b), (b, m)]:
        arrow(ax, p_.r, q_.l, color=INK, lw=1.8, shrink=1)
    s_ = box(ax, 5.2, 3.2, 2.3, .78, 'Sampler', PURPLE, fs=14, sub='shuffled order per epoch',
             sub_fs=12)
    arrow(ax, s_.b, l.t, color=PURPLE, lw=1.8, shrink=1, text='indices', fs=12, offset=(.42, 0))
    chip(ax, 3.65, 3.59, 'generator = seeded', color=PURPLE, fs=12)
    arrow(ax, (4.6, 3.59), (5.15, 3.59), color=PURPLE, lw=1.2)
    label(ax, 10.95, y - .28, 'model and batch\non one device', fs=12, color=MUTED, va='top',
          linespacing=1.15)
    # one epoch = 8 batches
    y0, hh, x0, unit = .35, .45, 2.6, .034
    sizes = [16] * 7 + [12]
    x = x0
    for k, n in enumerate(sizes):
        w = n * unit * 1.5
        ax.add_patch(FancyBboxPatch((x, y0), w - .06, hh, boxstyle='round,pad=0,rounding_size=.05',
                                    fc=tint(ORANGE, .75 if k < 7 else .88), ec=ORANGE, lw=1.1))
        label(ax, x + (w - .06) / 2, y0 + hh / 2, str(n), fs=12, color=INK)
        x += w
    label(ax, x0 - .12, y0 + hh / 2, 'one epoch:', fs=12.5, color=MUTED, ha='right')
    label(ax, x + .12, y0 + hh / 2, '124 rows = 7 batches of 16 + 1 of 12', fs=12.5, color=MUTED, ha='left')
    save('nn2_dataloader', f, 'Schematic; sizes follow the Lab 1 Sonar training partition (124 rows, batch 16).',
         kind='schematic')


def _edge(c, w, h, target):
    """Point on the boundary of a w x h box centred at c, in the direction of target."""
    c, t = np.asarray(c, float), np.asarray(target, float)
    v = t - c
    k = min((w / 2) / abs(v[0]) if v[0] else np.inf, (h / 2) / abs(v[1]) if v[1] else np.inf)
    return tuple(c + v * k)


@fig
def train_step():
    f, ax = canvas(6.3, 4.9)
    cx, cy, R = 3.15, 2.45, 1.72
    steps = [('zero_grad()', 'clear old gradients', GREY), ('forward', 'logits = model(xb)', BLUE),
             ('loss', 'criterion(logits, yb)', ORANGE), ('backward()', 'fill every .grad', PURPLE),
             ('step()', 'update the weights', TEAL)]
    pos = [(3.15, 4.12), (5.1, 2.9), (4.4, .85), (1.9, .85), (1.2, 2.9)]
    w, h = 1.95, .8
    for k, ((t, s_, col), (x, y)) in enumerate(zip(steps, pos)):
        box(ax, x - w / 2, y - h / 2, w, h, f'{k + 1}  {t}', col, fill='solid' if k == 3 else 'tint',
            fs=14, sub=s_, sub_fs=12)
    for k in range(5):
        p_, q_ = pos[k], pos[(k + 1) % 5]
        start = _edge(p_, w + .12, h + .12, q_)
        end = _edge(q_, w + .12, h + .12, p_)
        arrow(ax, start, end, color=MUTED, lw=1.8, rad=-.12)
    label(ax, cx, cy + .1, 'one mini-batch', fs=14, color=INK, weight='semibold')
    label(ax, cx, cy - .25, 'repeat for every batch', fs=12, color=MUTED)
    save('nn2_train_step', f, 'Schematic of one PyTorch optimization step.', kind='schematic')


def _sonar():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    folder = ROOT / 'neural_networks_course/Day_2_NN_Advanced/exercises/01_pytorch_pipeline'
    df = pd.read_csv(folder / 'sonar.all-data.csv', header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].map({'R': 0, 'M': 1}).to_numpy(dtype=np.float32)
    dev, final = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    tr, va = train_test_split(dev, test_size=.25, stratify=y[dev], random_state=42)
    sc = StandardScaler().fit(X[tr])
    return (sc.transform(X[tr]).astype(np.float32), y[tr], sc.transform(X[va]).astype(np.float32), y[va])


def sonar_run(seed, width=32, dropout=.2, lr=1e-3, epochs=80, patience=10):
    """The Lab 1 reference training loop (same settings as the notebook)."""
    torch = _torch()
    from torch.utils.data import TensorDataset, DataLoader
    Xtr, ytr, Xva, yva = _sonar()
    Xt, Xv = torch.tensor(Xtr), torch.tensor(Xva)
    yt, yv = torch.tensor(ytr).unsqueeze(1), torch.tensor(yva).unsqueeze(1)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(60, width), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                torch.nn.Linear(width, 1))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    hist, best, best_ep, best_state, stale = [], np.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            vl = crit(model(Xv), yv).item()
        hist.append(vl)
        if vl < best - 1e-7:
            best, best_ep, stale = vl, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        acc = float(((torch.sigmoid(model(Xv)) >= .5).float() == yv).float().mean())
    return np.array(hist), best_ep, best, acc


@fig
def seed_spread():
    seeds = list(range(10))
    runs = [sonar_run(s) for s in seeds]
    f, axs = plt.subplots(2, 1, figsize=HALF, layout='constrained', gridspec_kw=dict(height_ratios=[2.1, 1]))
    ax = axs[0]
    for hist, be, bl, acc in runs:
        ep = np.arange(1, len(hist) + 1)
        ax.plot(ep, hist, color=TEAL, lw=1.4, alpha=.7)
        ax.plot(be, bl, 'o', color=TEAL, ms=6, zorder=4)
    ax.set_ylim(.15, .75)
    ax.set_xlim(0, 82)
    clean(ax, 'Epoch', 'Validation BCE')
    ax.text(80, .72, '10 seeds, identical settings', ha='right', va='top', fontsize=12.5, color=MUTED)
    ax.text(80, .64, 'dots: restored checkpoints', ha='right', va='top', fontsize=12.5, color=TEAL)
    ax = axs[1]
    accs = np.array([r[3] for r in runs])
    ax.scatter(accs, np.zeros_like(accs) + np.random.default_rng(0).uniform(-.18, .18, len(accs)), s=70,
               color=TEAL, edgecolors='white', zorder=3)
    ax.set_ylim(-.6, .6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(.74, 1.0)
    clean(ax, 'Validation accuracy at the restored checkpoint (42 rows)')
    ax.grid(axis='y', visible=False)
    ax.text(accs.min(), .42, f'{accs.min():.2f}', ha='center', fontsize=12.5, color=INK)
    ax.text(accs.max(), .42, f'{accs.max():.2f}', ha='center', fontsize=12.5, color=INK)
    save('nn2_seed_spread', f,
         'Computed from the course file sonar.all-data.csv with the Lab 1 split and training loop '
         '(width 32, dropout 0.2, AdamW lr 1e-3, weight decay 1e-3, batch 16, patience 10); torch seeds 0-9.',
         kind='data')


# ============================================================ part 4: optimization
BOWL = np.diag([1.0, 50.0])          # ill-conditioned quadratic: curvature 1 along x, 50 along y
START = (-4.5, 1.5)


def bowl_path(name, lr, steps=50, beta=.9):
    torch = _torch()
    H = torch.tensor(BOWL, dtype=torch.float64)
    p = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    opt = {'sgd': lambda: torch.optim.SGD([p], lr=lr),
           'momentum': lambda: torch.optim.SGD([p], lr=lr, momentum=beta),
           'rmsprop': lambda: torch.optim.RMSprop([p], lr=lr),
           'adam': lambda: torch.optim.Adam([p], lr=lr)}[name]()
    path = [p.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * p @ H @ p).backward()
        opt.step()
        path.append(p.detach().numpy().copy())
    path = np.array(path)
    return path, 0.5 * np.einsum('ni,ij,nj->n', path, BOWL, path)


def _bowl_axes(ax):
    xx, yy = np.meshgrid(np.linspace(-5.2, 1.6, 400), np.linspace(-1.9, 1.9, 300))
    Z = 0.5 * (BOWL[0, 0] * xx ** 2 + BOWL[1, 1] * yy ** 2)
    ax.contour(xx, yy, Z, levels=np.geomspace(.03, 80, 11), colors='#C9D3DD', linewidths=.9, zorder=1)
    ax.plot(0, 0, marker='*', ms=15, color=INK, zorder=5)
    ax.plot(*START, marker='o', ms=8, mfc='white', mec=INK, mew=1.6, zorder=5)
    ax.set_xlim(-5.2, 1.6)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xticks([-4, -2, 0])
    ax.set_yticks([-1, 0, 1])


def _opt_panel(ax, name, lr, color, title):
    path, loss = bowl_path(name, lr)
    _bowl_axes(ax)
    ax.plot(path[:, 0], path[:, 1], '-', color=color, lw=1.8, zorder=3)
    ax.plot(path[:, 0], path[:, 1], 'o', color=color, ms=3.4, zorder=4)
    ax.text(.985, .95, title, transform=ax.transAxes, fontsize=13, color=color, fontweight='semibold',
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=.9),
            zorder=6)
    ax.text(.985, .06, f'loss after 50 steps: {loss[-1]:.2f}', transform=ax.transAxes, ha='right',
            fontsize=12.5, color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=.9),
            zorder=6)
    return loss


@fig
def opt_momentum():
    f, axs = half(1, rows=2, height=4.9, sharex=True)
    _opt_panel(axs[0], 'sgd', .038, GREY, 'SGD, learning rate 0.038')
    _opt_panel(axs[1], 'momentum', .004, BLUE, 'SGD + momentum 0.9, learning rate 0.004')
    axs[1].set_xlabel('Parameter 1 (gentle direction)')
    for ax in axs:
        ax.set_ylabel('Parameter 2', fontsize=13)
    save('nn2_opt_momentum', f,
         'Computed with torch.optim on the quadratic L = (w1² + 50 w2²)/2 from (-4.5, 1.5), 50 steps; '
         'learning rates chosen for illustration (0.038 is just below the SGD stability limit 0.04).',
         kind='computed')


@fig
def opt_adaptive():
    f, axs = half(1, rows=2, height=4.9, sharex=True)
    _opt_panel(axs[0], 'rmsprop', .05, PURPLE, 'RMSprop, learning rate 0.05')
    _opt_panel(axs[1], 'adam', .25, ORANGE, 'Adam, learning rate 0.25')
    axs[1].set_xlabel('Parameter 1 (gentle direction)')
    for ax in axs:
        ax.set_ylabel('Parameter 2', fontsize=13)
    save('nn2_opt_adaptive', f,
         'Computed with torch.optim (default betas and alpha) on L = (w1² + 50 w2²)/2 from (-4.5, 1.5), '
         '50 steps; learning rates chosen for illustration.', kind='computed')


@fig
def adamw_decay():
    rms = np.geomspace(1e-3, 1e1, 200)
    f, axs = half()
    ax = axs[0]
    ax.loglog(rms, 1 / rms, color=ORANGE, lw=2.8)
    ax.loglog(rms, np.ones_like(rms), color=TEAL, lw=2.8)
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(5e-2, 2e3)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10], ['0.001', '0.01', '0.1', '1', '10'])
    ax.set_yticks([.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    clean(ax, r'Typical gradient size of the weight, $\sqrt{\hat{v}}$', 'Decay per step, in units of ηλθ')
    ax.text(.13, 420, 'Adam + L2 penalty:\ndecay is divided by ' + r'$\sqrt{\hat{v}}$', color=ORANGE,
            fontsize=13, fontweight='semibold', va='center', linespacing=1.3)
    ax.text(1.2e-3, 1.35, 'AdamW: the same decay for every weight', color=TEAL, fontsize=13,
            fontweight='semibold', ha='left', va='bottom')
    ax.annotate('weights with large gradients\nare barely decayed', (5, .2), xytext=(2e-2, .09),
                fontsize=12.5, color=ORANGE, va='center', linespacing=1.15,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    save('nn2_adamw_decay', f,
         'Analytical approximation: with an L2 term inside Adam the decay part of the update is about '
         'ηλθ/√v̂ (valid while λθ is small relative to the gradient); AdamW applies ηλθ directly.',
         kind='computed')


@fig
def lr_steps():
    """Exact gradient descent on L = w^2 from w = 3; slow, fast and divergent step sizes."""
    f, axs = sub(1, height=4.3)
    ax = axs[0]
    for rate, col, note_ in [(1.05, ORANGE, 'learning rate 1.05: diverges'),
                             (.02, PURPLE, 'learning rate 0.02: slow'),
                             (.3, TEAL, 'learning rate 0.3: fast')]:
        w, loss = 3.0, []
        for _ in range(30):
            loss.append(w * w)
            w = w - rate * 2 * w
        ax.semilogy(np.arange(30), loss, color=col, lw=2.6)
        endlabel(ax, 29, loss[-1], note_, col, dx=4, fs=13, weight='semibold')
    ax.set_xlim(0, 43)
    ax.set_xticks([0, 10, 20, 29], ['0', '10', '20', '29'])
    ax.set_ylim(1e-24, 1e5)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1], ['1e-20', '1e-15', '1e-10', '1e-5', '1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    clean(ax, 'Gradient-descent update', 'Loss w² (log scale)')
    save('nn2_lr_steps', f,
         'Computed: exact gradient descent on L = w² from w = 3, 30 updates; each update multiplies w by '
         '(1 − 2 × learning rate).', kind='computed')


@fig
def lr_range():
    torch = _torch()
    Xtr, ytr = esp_like(4000, 3)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    X = torch.tensor((Xtr - mu) / sd)
    y = torch.tensor(ytr).unsqueeze(1)
    torch.manual_seed(0)
    model = mlp(12, 64, 2)
    steps, lo, hi = 300, 1e-6, 1.0
    lrs = np.geomspace(lo, hi, steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lo, weight_decay=0.0)
    crit = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(0)
    raw, smooth, avg = [], [], 0.0
    idx = torch.cat([torch.randperm(len(X), generator=gen) for _ in range(3)])
    for k, lr in enumerate(lrs):
        for g in opt.param_groups:
            g['lr'] = lr
        b = idx[k * 32:(k + 1) * 32]
        opt.zero_grad()
        loss = crit(model(X[b]), y[b])
        loss.backward()
        opt.step()
        raw.append(loss.item())
        avg = .95 * avg + .05 * loss.item()
        smooth.append(avg / (1 - .95 ** (k + 1)))
        if k > 20 and smooth[-1] > 1.3 * smooth[0]:
            break
    n = len(smooth)
    lrs, smooth, raw = lrs[:n], np.array(smooth), np.array(raw)
    i_min = int(np.argmin(smooth))
    lr_min = lrs[i_min]
    lr_pick = lr_min / 10
    f, axs = half()
    ax = axs[0]
    ax.semilogx(lrs, raw, color=LIGHT, lw=1, zorder=1)
    ax.semilogx(lrs, smooth, color=BLUE, lw=2.6, zorder=3)
    ax.axvspan(lr_pick / 3, lr_pick * 3, color=TEAL, alpha=.12, lw=0)
    ax.axvline(lr_pick, color=TEAL, lw=2, ls='--')
    ax.plot(lr_min, smooth[i_min], 'o', color=ORANGE, ms=8, zorder=5)
    ax.set_ylim(.3, 1.1)
    ax.set_xlim(lo, 1.0)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1], ['1e-6', '1e-4', '0.01', '1'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    clean(ax, 'Learning rate (increased every mini-batch)', 'Training loss (smoothed)')
    # labels sit where the raw loss never reaches: above the curve between 0.006 and 0.07,
    # left of the dashed line, and in the empty strip right of the final rise
    ax.annotate(f'minimum\nnear {lr_min:.2g}', (lr_min, smooth[i_min]), xytext=(lr_pick * 2.75, .985),
                fontsize=12.5, color=ORANGE, va='top', linespacing=1.15,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1, shrinkB=5))
    ax.text(lr_pick / 1.2, 1.07, f'start ≈ {lr_pick:.1g}', color=TEAL, fontsize=13, fontweight='semibold',
            ha='right', va='top')
    ax.text(2e-6, .72, 'too small:\nnothing moves', color=MUTED, fontsize=12.5, linespacing=1.2)
    ax.text(lrs[-1] * 1.6, .86, 'loss blows up', color=MUTED, fontsize=12.5, ha='center', va='center',
            rotation=90)
    save('nn2_lr_range', f,
         'Computed: learning-rate range test on a synthetic ESP-style table (4,000 rows, seed 3); '
         'MLP 12-64-64-1, AdamW, batch 32, torch seed 0, rate raised geometrically from 1e-6 each step; '
         'grey = raw batch loss, blue = exponential moving average.', kind='computed')
    return lr_min, lr_pick


@fig
def schedules():
    torch = _torch()
    from torch.optim import lr_scheduler as S
    E = 100

    def trace(make):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1.0)
        sch = make(opt)
        out = []
        for _ in range(E):
            out.append(opt.param_groups[0]['lr'])
            opt.step()
            sch.step()
        return np.array(out)

    specs = [('Step decay', lambda o: S.StepLR(o, step_size=30, gamma=.1), 'StepLR'),
             ('Cosine', lambda o: S.CosineAnnealingLR(o, T_max=E), 'CosineAnnealingLR'),
             ('Warmup + cosine', lambda o: S.SequentialLR(o, [S.LinearLR(o, start_factor=.01, total_iters=10),
                                                            S.CosineAnnealingLR(o, T_max=E - 10)],
                                                         milestones=[10]), 'LinearLR → Cosine'),
             ('One-cycle', lambda o: S.OneCycleLR(o, max_lr=1.0, total_steps=E), 'OneCycleLR')]
    f, axs = sub(4, height=3.7)
    for ax, (title, make, api) in zip(axs, specs):
        lr = trace(make)
        ax.plot(np.arange(E), lr, color=BLUE, lw=2.6)
        ax.fill_between(np.arange(E), lr, color=BLUE, alpha=.08)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, E)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([0, 50, 100])
        clean(ax, 'Step (or epoch)', 'Learning rate / peak' if ax is axs[0] else None)
        ax.text(.97, 1.0, api, transform=ax.transAxes, ha='right', va='top', fontsize=12, color=MUTED,
                family=theme.MONO_FAMILY, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.9))
    save('nn2_schedules', f, 'Computed by stepping the named torch.optim.lr_scheduler classes for 100 steps '
         '(peak learning rate scaled to 1).', kind='computed')


@fig
def batch_noise():
    rng = np.random.default_rng(0)
    N = 4096
    X = rng.normal(size=(N, 2))
    w_true = np.array([1.5, -1.0])
    y = (rng.uniform(size=N) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    w = np.array([0.2, 0.3])                       # current parameters, far from the optimum

    def grad(idx):
        p = 1 / (1 + np.exp(-X[idx] @ w))
        return X[idx].T @ (p - y[idx]) / len(idx)

    g_full = grad(np.arange(N))
    f, axs = grid_axes(2, 2, width=6.3, height=4.9)
    specs = [(8, ORANGE), (64, BLUE), (512, TEAL)]
    lim = .62
    for ax, (B, col) in zip([axs[0, 0], axs[0, 1], axs[1, 0]], specs):
        G = np.array([grad(rng.choice(N, B, replace=False)) for _ in range(300)])
        ax.scatter(G[:, 0], G[:, 1], s=9, color=col, alpha=.5, lw=0, zorder=2)
        vals, vecs = np.linalg.eigh(np.cov(G.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(G.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang, fc='none',
                             ec=col, lw=2, zorder=3))
        ax.plot(*g_full, marker='*', ms=12, color=INK, mec='white', mew=.8, zorder=5)
        ax.set_xlim(g_full[0] - lim, g_full[0] + lim)
        ax.set_ylim(g_full[1] - lim, g_full[1] + lim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'B = {B}: {N // B} steps/epoch', fontsize=13, color=col)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(RULE)
    ax = axs[1, 1]
    Bs = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    sds = []
    for B in Bs:
        G = np.array([grad(rng.choice(N, B, replace=False)) for _ in range(300)])
        sds.append(np.sqrt(((G - g_full) ** 2).sum(1).mean()))
    sds = np.array(sds)
    ax.loglog(Bs, sds, 'o', color=INK, ms=5, zorder=3)
    ax.loglog(Bs, sds[0] * np.sqrt(Bs[0] / Bs), color=MUTED, lw=1.4, ls='--', zorder=2)
    ax.set_xticks([4, 32, 256], ['4', '32', '256'])
    ax.set_yticks([.03, .1, .3], ['0.03', '0.1', '0.3'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    clean(ax, 'Batch size B', None)
    ax.set_title('Noise falls like 1/√B', fontsize=13)
    ax.text(.95, .88, 'error of the\nbatch gradient', transform=ax.transAxes, ha='right', va='top',
            fontsize=12, color=MUTED, linespacing=1.15)
    save('nn2_batch_noise', f,
         'Computed: logistic-regression gradients at one fixed parameter vector on 4,096 synthetic rows '
         '(numpy seed 0); 300 random mini-batches per size; star = full-batch gradient, ellipse = 2 '
         'standard deviations; dashed line = 1/√B reference.', kind='computed')


@fig
def clip_geometry():
    f, ax = canvas(6.3, 4.9)
    o = np.array([1.85, 2.55])
    r = 1.1
    ax.add_patch(Circle(o, r, fc=tint(TEAL, .9), ec=TEAL, lw=2, ls=(0, (5, 3)), zorder=1))
    label(ax, o[0], o[1] - r - .22, 'clipping threshold c', fs=12.5, color=TEAL)
    g_small = np.array([.45, -.5])
    arrow(ax, tuple(o), tuple(o + g_small), color=BLUE, lw=2.6, head=16)
    label(ax, o[0] + r + .15, o[1] - .55, 'small gradient:\nunchanged', fs=12.5, color=BLUE, ha='left',
          va='top', linespacing=1.15)
    d = np.array([1.0, .62]) / np.linalg.norm([1.0, .62])
    tip = o + d * 2.9 * r
    clipped = o + d * r
    ax.plot([clipped[0], tip[0]], [clipped[1], tip[1]], color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=2)
    ax.plot(*tip, marker='o', ms=7, color=ORANGE)
    arrow(ax, tuple(o), tuple(clipped), color=ORANGE, lw=3, head=18)
    label(ax, tip[0] - .1, tip[1] + .12, 'spike: norm 2.9 c', fs=12.5, color=ORANGE, ha='right', va='bottom')
    label(ax, clipped[0] + .34, clipped[1] - .12, 'rescaled to norm c,\nsame direction', fs=12.5, color=ORANGE,
          ha='left', va='top', linespacing=1.15)      # clear of the dashed threshold circle
    ax.plot(*o, 'o', color=INK, ms=5, zorder=4)
    label(ax, 3.15, .78, r'$g \leftarrow g \cdot \min(1,\ c\,/\,\Vert g \Vert)$', fs=18)
    label(ax, 3.15, .3, 'one global norm over all parameters', fs=12.5, color=MUTED)
    save('nn2_clip_geometry', f, 'Schematic of gradient-norm clipping in two dimensions.', kind='schematic')


# ============================================================ part 5: regularization
@fig
def weight_decay():
    runs = [(0.0, ORANGE, 'no decay'), (1.0, TEAL, 'decay 1.0')]
    f, axs = plt.subplots(2, 1, figsize=HALF, layout='constrained', gridspec_kw=dict(height_ratios=[1, 1]))
    for wd, col, name in runs:
        r = fit_binary(scale=True, width=64, wd=wd, drop=0.0, epochs=150)
        ep = np.arange(1, len(r['norms']) + 1)
        axs[0].plot(ep, r['norms'], color=col, lw=2.6)
        endlabel(axs[0], ep[-1], r['norms'][-1], name, col, fs=13, weight='semibold', dx=-80,
                 dy=12 if wd == 0 else -14)
        w = np.concatenate([p.detach().numpy().ravel() for n_, p in r['model'].named_parameters()
                            if n_.endswith('weight')])
        axs[1].hist(w, bins=np.linspace(-1.2, 1.2, 61), color=col, alpha=.55, density=True,
                    histtype='stepfilled', lw=0)
        axs[1].hist(w, bins=np.linspace(-1.2, 1.2, 61), color=col, density=True, histtype='step', lw=1.6)
    clean(axs[0], 'Epoch', 'L2 norm of all weights')
    axs[0].set_xlim(0, 150)
    clean(axs[1], 'Weight value after 150 epochs', 'Density')
    axs[1].set_yticks([])
    axs[1].set_xlim(-1.2, 1.2)
    save('nn2_weight_decay', f,
         'Computed: synthetic ESP-style table (240 training rows), MLP 12-64-64-1, AdamW lr 1e-3, '
         'weight_decay 0 or 1.0 (decay per step = lr x weight_decay), torch seed 0.', kind='computed')


@fig
def dropout_net():
    f, ax = canvas(6.3, 4.9)
    sizes = [4, 5, 5, 1]
    dropped = {1: {1, 3}, 2: {0, 2, 4}}

    def draw(x0, keep_all, title):
        xs = x0 + np.array([0, .85, 1.7, 2.45])
        ys = [np.linspace(3.9, 1.5, n) if n > 1 else np.array([2.7]) for n in sizes]
        alive = [[keep_all or j not in dropped.get(li, set()) for j in range(n)] for li, n in enumerate(sizes)]
        for li in range(3):
            for a_, ya in enumerate(ys[li]):
                for b_, yb in enumerate(ys[li + 1]):
                    on = alive[li][a_] and alive[li + 1][b_]
                    if on:
                        ax.plot([xs[li], xs[li + 1]], [ya, yb], color='#9DAFC0', lw=1, zorder=1)
        cols = [GREY, BLUE, BLUE, ORANGE]
        for li, n in enumerate(sizes):
            for j in range(n):
                if alive[li][j]:
                    node(ax, xs[li], ys[li][j], .17, cols[li], 'tint')
                else:
                    node(ax, xs[li], ys[li][j], .17, LIGHT, 'white', lw=1.2)
                    cross(ax, xs[li], ys[li][j], s=.1, color=ORANGE, lw=2.2)
        label(ax, x0 + 1.22, 4.45, title, fs=14, weight='semibold')

    draw(.3, False, 'Training step')
    draw(3.45, True, 'Evaluation')
    ax.plot([3.13, 3.13], [1.3, 4.2], color=RULE, lw=1.2)
    label(ax, 1.52, 1.0, 'random units set to 0 (p = 0.5);\nkept ones scaled × 2', fs=12, color=MUTED,
          linespacing=1.2)
    label(ax, 4.67, 1.0, 'all units active,\nno scaling in eval()', fs=12, color=MUTED, linespacing=1.2)
    label(ax, 3.15, .3, 'a new random mask is drawn for every mini-batch', fs=12.5, color=INK)
    save('nn2_dropout_net', f, 'Schematic of inverted dropout with drop probability 0.5.', kind='schematic')


@fig
def early_stop():
    r = fit_binary(scale=True, width=64, wd=0.1, drop=0.2, epochs=150)
    tr, va = r['train'], r['val']
    patience = 20
    best, best_ep, stale, stop = np.inf, 0, 0, len(va)
    for ep, v in enumerate(va, start=1):
        if v < best - 1e-7:
            best, best_ep, stale = v, ep, 0
        else:
            stale += 1
        if stale >= patience:
            stop = ep
            break
    f, axs = sub(1, height=4.1)
    ax = axs[0]
    ep = np.arange(1, len(va) + 1)
    run = ep <= stop
    ax.plot(ep[run], tr[run], color=BLUE, lw=2.6)
    ax.plot(ep[run], va[run], color=TEAL, lw=2.6, ls='--')
    ax.plot(ep[~run], tr[~run], color=LIGHT, lw=2)
    ax.plot(ep[~run], va[~run], color=LIGHT, lw=2, ls='--')
    ax.axvspan(best_ep, stop, color=TEAL, alpha=.09, lw=0)
    ax.axvline(stop, color=MUTED, lw=1.2)
    ax.plot(best_ep, best, 'o', color=TEAL, ms=10, zorder=5)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.0)
    clean(ax, 'Epoch', 'Binary cross-entropy')
    # text to the right of the stop line, so neither the line nor the band crosses it
    ax.annotate(f'restored checkpoint: epoch {best_ep}\nvalidation loss {best:.3f}', (best_ep, best),
                xytext=(stop + 4, .95), fontsize=13, color=INK, va='top', linespacing=1.25,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkB=6))
    ax.text((best_ep + stop) / 2, .05, f'patience {patience}', ha='center', fontsize=12.5, color=TEAL)
    ax.text(stop + 2, .45, f'stop at epoch {stop}', fontsize=12.5, color=MUTED, va='center')
    ax.text(148, va[-1] - .14, 'validation keeps rising\n(epochs after the stop, for reference)', ha='right',
            va='top', fontsize=12.5, color=MUTED, linespacing=1.2)
    ax.text(stop - 1, tr[stop - 1] - .06, 'training', ha='right', va='top', fontsize=13, color=BLUE,
            fontweight='semibold')             # inside the band, below the solid curve
    endlabel(ax, stop, va[stop - 1], 'validation', TEAL, dx=4, dy=12, fs=13, weight='semibold')
    save('nn2_early_stop', f,
         'Computed: synthetic ESP-style table (240 training / 1,000 validation rows); MLP 12-64-64-1, '
         'AdamW lr 1e-3, weight decay 0.1, dropout 0.2, torch seed 0; patience rule applied to the '
         'recorded validation curve; grey = epochs after the stop.', kind='computed')
    return best_ep, stop, best


def _current_trace(n=300, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 38 + 1.2 * np.sin(2 * np.pi * t / 90) + .6 * np.sin(2 * np.pi * t / 23)
    base[170:200] += np.linspace(0, 3.5, 30)          # load increase
    base[200:230] += 3.5 * np.exp(-np.arange(30) / 8)
    return t, base + rng.normal(0, .15, n)


@fig
def augment():
    t, x = _current_trace()
    rng = np.random.default_rng(1)
    f, axs = sub(4, height=3.7, sharey=True)
    cases = [('Sensor noise', TEAL, [x + rng.normal(0, .35, len(x)) for _ in range(2)], 'σ = 0.35 A'),
             ('Small time shift', TEAL, [np.roll(x, s) for s in (-7, 7)], '± 7 samples'),
             ('Gain change', TEAL, [x * g for g in (.97, 1.03)], '± 3 %'),
             ('Time reversal', ORANGE, [x[::-1]], 'changes the physics')]
    for ax, (title, col, variants, note_) in zip(axs, cases):
        ax.plot(t, x, color=GREY, lw=1.6, zorder=3)
        for v in variants:
            ax.plot(t, v, color=col, lw=1.2, alpha=.85, zorder=2)
        ax.set_title(title, fontsize=15, color=INK)
        ax.text(.03, .95, note_, transform=ax.transAxes, va='top', fontsize=12.5, color=col,
                fontweight='semibold')
        clean(ax, 'Sample (1 s)', 'Motor current (A)' if ax is axs[0] else None)
        ax.set_ylim(34.5, 44.5)
        ax.set_xticks([0, 150, 300])
    axs[0].text(.97, .05, 'grey = original', transform=axs[0].transAxes, ha='right', fontsize=12, color=GREY)
    save('nn2_augment', f,
         'Computed: synthetic ESP motor-current trace (numpy seed 4) with augmented copies (seed 1); '
         'noise level, shift and gain are illustrative, not sensor specifications.', kind='computed')


@fig
def label_smoothing():
    K, eps = 4, .1
    soft_true = 1 - eps + eps / K
    soft_other = eps / K
    f, axs = plt.subplots(2, 1, figsize=HALF, layout='constrained', gridspec_kw=dict(height_ratios=[1, 1.25]))
    ax = axs[0]
    names = ['sand', 'shale', 'carbonate', 'coal']
    xs = np.arange(K)
    ax.bar(xs - .19, [1, 0, 0, 0], .36, color=ORANGE, label='one-hot')
    ax.bar(xs + .19, [soft_true] + [soft_other] * 3, .36, color=TEAL, label='smoothed')
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .5, 1])
    clean(ax, None, 'Target probability')
    ax.text(0.19, soft_true + .04, f'{soft_true:.3f}', ha='center', fontsize=12, color=TEAL)
    ax.text(1.19, soft_other + .04, f'{soft_other:.3f}', ha='center', fontsize=12, color=TEAL)
    ax.text(3.45, 1.05, 'one-hot', ha='right', fontsize=12.5, color=ORANGE, fontweight='semibold')
    ax.text(3.45, .82, f'smoothed, ε = {eps}', ha='right', fontsize=12.5, color=TEAL, fontweight='semibold')
    ax = axs[1]
    m = np.linspace(0, 10, 400)
    p_true = np.exp(m) / (np.exp(m) + K - 1)
    p_other = 1 / (np.exp(m) + K - 1)
    hard = -np.log(p_true)
    smooth_ = -(soft_true * np.log(p_true) + (K - 1) * soft_other * np.log(p_other))
    ax.plot(m, hard, color=ORANGE, lw=2.6)
    ax.plot(m, smooth_, color=TEAL, lw=2.6)
    m_star = np.log(soft_true / soft_other)
    ax.axvline(m_star, color=TEAL, lw=1.2, ls='--')
    ax.set_ylim(0, 1.6)
    clean(ax, 'Logit margin of the true class', 'Cross-entropy')
    ax.text(m_star + .15, 1.35, f'smoothed optimum at\nmargin log 37 = {m_star:.2f}', fontsize=12.5,
            color=TEAL, va='top', linespacing=1.2)
    ax.text(9.8, .1, 'one-hot keeps pushing\nthe margin up', ha='right', fontsize=12.5, color=ORANGE,
            va='bottom', linespacing=1.2)
    save('nn2_label_smoothing', f,
         'Exact targets and cross-entropy for K = 4 classes with label smoothing 0.1 (PyTorch convention: '
         '(1 - ε) one-hot + ε/K); other logits held equal.', kind='computed')


@fig
def double_descent():
    rng = np.random.default_rng(0)
    n, d, ntest = 100, 10, 2000
    wstar = rng.normal(size=d) / np.sqrt(d)

    def target(X):
        return np.tanh(2 * X @ wstar) + .5 * np.sin(1.5 * X[:, 0])

    X = rng.normal(size=(n, d))
    y = target(X) + rng.normal(0, .3, n)
    Xt = rng.normal(size=(ntest, d))
    yt = target(Xt)
    ps = np.unique(np.round(np.geomspace(5, 2000, 40)).astype(int))
    curves = {0: [], .1: []}
    for p in ps:
        errs = {0: [], .1: []}
        for _ in range(20):
            W = rng.normal(size=(d, p)) / np.sqrt(d)
            P, Pt = np.maximum(X @ W, 0), np.maximum(Xt @ W, 0)
            errs[0].append(np.mean((Pt @ (np.linalg.pinv(P) @ y) - yt) ** 2))
            th = P.T @ np.linalg.solve(P @ P.T + .1 * np.eye(n), y)
            errs[.1].append(np.mean((Pt @ th - yt) ** 2))
        for k in errs:
            curves[k].append(np.median(errs[k]))
    f, axs = half()
    ax = axs[0]
    ax.loglog(ps, curves[0], 'o-', color=ORANGE, ms=4, lw=2.2)
    ax.loglog(ps, curves[.1], 's-', color=TEAL, ms=4, lw=2.2)
    ax.axvline(n, color=MUTED, lw=1.2, ls='--')
    ax.set_xlim(4, 2500)
    ax.set_ylim(.2, 6)
    ax.set_xticks([10, 100, 1000], ['10', '100', '1,000'])
    ax.set_yticks([.3, 1, 3], ['0.3', '1', '3'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    clean(ax, 'Number of random features (parameters)', 'Test mean squared error')
    ax.text(n * 1.3, 5.4, 'parameters =\ntraining rows (100)', fontsize=12, color=MUTED, va='top',
            linespacing=1.15)                    # right of the peak marker at p = 108
    ax.text(2300, 1.25, 'no regularization:\npeak at the threshold', ha='right', fontsize=12.5, color=ORANGE,
            fontweight='semibold', linespacing=1.2)
    ax.text(2300, .36, 'small ridge penalty:\nsharp peak removed', ha='right', fontsize=12.5, color=TEAL,
            fontweight='semibold', linespacing=1.2, va='top')
    save('nn2_double_descent', f,
         'Computed: random ReLU features fitted by minimum-norm least squares (orange) or ridge 0.1 (teal) '
         'to 100 synthetic training rows with 10 inputs and noise 0.3; median over 20 feature draws; numpy seed 0.',
         kind='computed')


# ============================================================ tuning
@fig
def random_search():
    rng = np.random.default_rng(7)
    n_trials = 24
    lrs = 10 ** rng.uniform(-5, -0.5, n_trials)
    wds = 10 ** rng.uniform(-4, 0.5, n_trials)
    best = []
    for k, (lr, wd) in enumerate(zip(lrs, wds)):
        r = fit_binary(scale=True, width=64, wd=float(wd), drop=0.0, epochs=100, lr=float(lr), seed=k,
                       patience=15)
        best.append(r['best'])
    best = np.array(best)
    Xtr, ytr, Xva, yva = esp_split()
    base = float(np.mean(ytr))
    const = -(yva * np.log(base) + (1 - yva) * np.log(1 - base)).mean()
    f, axs = half()
    ax = axs[0]
    ax.scatter(lrs, best, color=TEAL, s=70, edgecolors='white', linewidths=.8, zorder=3)
    i = int(np.argmin(best))
    ax.plot(lrs[i], best[i], marker='*', ms=22, color=ORANGE, mec='white', mew=1, zorder=4)
    ax.axhline(const, color=GREY, lw=1.2, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ['1e-5', '1e-4', '1e-3', '0.01', '0.1'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.plot(lrs, np.full(n_trials, .035), '|', color=MUTED, ms=13, mew=1.5, transform=ax.get_xaxis_transform())
    clean(ax, 'Learning rate (sampled log-uniformly)', 'Best validation BCE')
    ax.set_ylim(.5, .72)
    ax.text(1.2e-5, const + .004, 'constant-prediction loss', fontsize=12, color=GREY, va='bottom')
    ax.annotate(f'best trial: lr {lrs[i]:.2g}, weight decay {wds[i]:.2g}', (lrs[i], best[i]),
                xytext=(3e-4, .525), fontsize=12.5, color=ORANGE,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.1))
    save('nn2_random_search', f,
         'Computed: 24 random trials, learning rate log-uniform on [1e-5, 0.32] and AdamW weight decay '
         'log-uniform on [1e-4, 3.2] (numpy seed 7); synthetic ESP-style table; best validation loss '
         'with patience 15, at most 100 epochs; ticks mark the sampled rates.', kind='computed')
    return lrs, wds, best


# ============================================================ autoencoders
@fig
def autoencoder():
    f, ax = canvas(11.8, 4.2)
    network(ax, [8, 5, 2, 5, 8], 1.75, 9.1, .8, 3.45, r=.15, max_nodes=8,
            colors=[GREY, BLUE, PURPLE, BLUE, ORANGE], fills=['tint', 'tint', 'solid', 'tint', 'tint'],
            edge_color='#C3CFDB', edge_lw=.7)
    label(ax, 1.75, .35, 'input x\n8 sensor channels', fs=12.5, color=MUTED, va='center', linespacing=1.15)
    label(ax, 5.425, .35, 'bottleneck z\n2 numbers', fs=12.5, color=PURPLE, va='center', linespacing=1.15,
          weight='semibold')
    label(ax, 9.1, .35, r'reconstruction $\hat{x}$', fs=12.5, color=ORANGE, va='center')
    # brackets sit clear of the top row of nodes (node tops at y = 3.6)
    bracket(ax, 1.75, 5.425, 3.92, 'encoder: compress', color=BLUE, below=False, fs=13)
    bracket(ax, 5.425, 9.1, 3.92, 'decoder: rebuild', color=BLUE, below=False, fs=13)
    b = box(ax, 9.8, 1.6, 1.9, 1.15, 'Loss', ORANGE, fill='white', fs=14.5)
    label(ax, 10.75, 1.92, r'$\Vert x - \hat{x} \Vert^2$', fs=16)
    arrow(ax, (9.3, 2.12), b.l, color=ORANGE, lw=1.6)
    label(ax, 10.75, 3.35, 'the input is\nits own target', fs=12.5, color=MUTED, linespacing=1.15)
    label(ax, 10.75, 1.3, 'per-row error\n= anomaly score', fs=12.5, color=ORANGE, va='top', linespacing=1.15)
    save('nn2_autoencoder', f, 'Schematic of a dense autoencoder (8-5-2-5-8).', kind='schematic')


def _train_ae(X, sizes, steps=3000, lr=3e-3, seed=0, act='tanh'):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    A = nn.Tanh if act == 'tanh' else nn.ReLU
    enc, dec = [], []
    for a_, b_ in zip(sizes[:-1], sizes[1:]):
        enc += [nn.Linear(a_, b_), A()]
    enc = enc[:-1]
    rs = sizes[::-1]
    for a_, b_ in zip(rs[:-1], rs[1:]):
        dec += [nn.Linear(a_, b_), A()]
    dec = dec[:-1]
    E, D = nn.Sequential(*enc), nn.Sequential(*dec)
    opt = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(D(E(Xt)), Xt).backward()
        opt.step()
    E.eval()
    D.eval()
    return E, D


@fig
def ae_manifold():
    torch = _torch()
    rng = np.random.default_rng(3)
    t = rng.uniform(-2.5, 2.5, 400)
    X = np.column_stack([t, 1.1 * np.sin(1.2 * t)]) + rng.normal(0, .07, (400, 2))
    E, D = _train_ae(X, [2, 32, 32, 1], steps=3000)
    with torch.no_grad():
        z = E(torch.tensor(X, dtype=torch.float32)).numpy()
        zz = torch.linspace(float(z.min()), float(z.max()), 300).unsqueeze(1)
        curve = D(zz).numpy()
    # unusual rows: each coordinate inside the range of the training rows, but off the curve
    odd = np.array([[-1.6, .9], [.4, -1.0], [1.6, -.3], [-.3, 1.0], [2.3, -.9]])
    with torch.no_grad():
        rec = D(E(torch.tensor(odd, dtype=torch.float32))).numpy()
    f, axs = half()
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, color=LIGHT, zorder=1)
    ax.plot(curve[:, 0], curve[:, 1], color=BLUE, lw=3, zorder=2)
    for p_, q_ in zip(odd, rec):
        ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color=ORANGE, lw=1.6, ls=(0, (3, 2)), zorder=3)
        ax.plot(*p_, 'o', color=ORANGE, ms=9, mec='white', zorder=4)
        ax.plot(*q_, 'o', color=BLUE, ms=6, zorder=4)
    ax.set_aspect('equal', adjustable='datalim')
    clean(ax, 'Sensor 1 (scaled)', 'Sensor 2 (scaled)')
    ax.text(-2.7, -2.0, 'grey: training rows (normal operation)', fontsize=12, color=GREY)
    ax.text(-2.7, 2.2, 'blue curve: every possible\nreconstruction from z', fontsize=12.5, color=BLUE,
            fontweight='semibold', linespacing=1.2, va='top')
    ax.text(2.75, 2.2, 'orange: unusual rows\nand their errors', fontsize=12.5, color=ORANGE,
            fontweight='semibold', linespacing=1.2, va='top', ha='right')
    save('nn2_ae_manifold', f,
         'Computed: 400 synthetic two-sensor rows near a curve (numpy seed 3); autoencoder 2-32-32-1-32-32-2 '
         '(tanh), Adam lr 3e-3, 3,000 full-batch steps, torch seed 0; five hand-placed unusual rows.',
         kind='computed')


def _sensor_rows(n, rng, kind='normal'):
    q = rng.normal(0, 1, n)                        # flow-like factor
    p = rng.normal(0, 1, n)                        # pressure-like factor
    A = np.array([[1, .2], [.8, -.5], [.3, 1], [-.6, .7], [1, 1], [.5, -.9], [-.2, .4], [.9, .1]])
    Z = np.column_stack([q, p + .3 * q ** 2])
    X = Z @ A.T + rng.normal(0, .12, (n, 8))
    if kind == 'drift':                            # one sensor reads high
        X[:, 3] += rng.uniform(1.2, 2.5, n)
    if kind == 'decoupled':                        # channels from two different states
        X[:, 4:] = (np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)]) @ A[4:].T)
    return X


@fig
def ae_scores():
    torch = _torch()
    rng = np.random.default_rng(11)
    Xtr = _sensor_rows(2000, rng)
    Xva = _sensor_rows(500, rng)
    Xte = _sensor_rows(500, rng)
    Xan = np.vstack([_sensor_rows(30, rng, 'drift'), _sensor_rows(30, rng, 'decoupled')])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sc = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
    E, D = _train_ae(sc(Xtr), [8, 16, 2], steps=2500, lr=3e-3)

    def score(a):
        with torch.no_grad():
            x = torch.tensor(sc(a))
            return ((D(E(x)) - x) ** 2).mean(1).numpy()

    s_va, s_te, s_an = score(Xva), score(Xte), score(Xan)
    thr = np.percentile(s_va, 99)
    f, axs = half()
    ax = axs[0]
    bins = np.geomspace(min(s_te.min(), s_an.min()) * .9, max(s_te.max(), s_an.max()) * 1.1, 45)
    ax.hist(s_te, bins=bins, color=BLUE, alpha=.7, label='normal')
    ax.hist(s_an, bins=bins, color=ORANGE, alpha=.8, label='unusual')
    ax.axvline(thr, color=TEAL, lw=2.4, ls='--')
    ax.set_xscale('log')
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ['0.001', '0.01', '0.1', '1'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    clean(ax, 'Reconstruction error per row (log scale)', 'Number of test rows')
    fp, tp = int((s_te > thr).sum()), int((s_an > thr).sum())
    ax.text(thr * 1.12, ax.get_ylim()[1] * .95, f'threshold = 99th percentile\nof validation normals',
            color=TEAL, fontsize=12.5, va='top', linespacing=1.2, fontweight='semibold')
    ax.text(thr * 1.12, ax.get_ylim()[1] * .68, f'flagged: {tp} of {len(s_an)} unusual\n'
            f'and {fp} of {len(s_te)} normal rows', color=INK, fontsize=12.5, va='top', linespacing=1.2)
    ax.text(.03, .95, 'normal', transform=ax.transAxes, color=BLUE, fontsize=13, fontweight='semibold', va='top')
    ax.text(.03, .87, 'unusual (drift or\ndecoupled sensors)', transform=ax.transAxes, color=ORANGE, fontsize=13,
            fontweight='semibold', va='top', linespacing=1.15)
    save('nn2_ae_scores', f,
         'Computed: synthetic 8-channel sensor rows driven by two hidden factors (numpy seed 11); autoencoder '
         '8-16-2-16-8 (tanh) trained on 2,000 normal rows, torch seed 0; test = 500 normal + 60 unusual rows '
         '(30 with one drifting sensor, 30 with decoupled channels).', kind='computed')
    return thr, tp, fp
