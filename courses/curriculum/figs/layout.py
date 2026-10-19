"""Example diagrams for the slide layout reference (curriculum/layout_reference)."""
from curriculum.figlib import *  # noqa: F401,F403
from curriculum.figlib import register
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

KEY = 'layout'
fig = register(KEY)


@fig
def pipeline_flow():
    f, ax = canvas(11.8, 2.9)
    labels = ['Raw logs', 'Impute', 'Scale', 'Model', 'Prediction']
    subs = ['GR, RHOB, NPHI', 'median, train only', 'mean / sd, train only', 'fit on train folds', 'facies class']
    cols = [G, TEAL, TEAL, BLUE, ORANGE]
    bx = flow(ax, labels, .25, .95, 1.95, 1.05, gap=.42, colors=cols, subs=subs, fs=14)
    bracket(ax, bx[1].x, bx[3].x + bx[3].w, 2.35, 'refit inside every cross-validation fold', color=TEAL, below=False)
    save('layout_pipeline_flow', f, 'Schematic.', kind='schematic')


@fig
def mlp_diagram():
    f, ax = canvas(6.3, 4.9)
    pos = network(ax, [4, 6, 6, 2], .8, 5.5, .9, 4.3, r=.2,
                  labels=['Inputs', 'Hidden 1', 'Hidden 2', 'Outputs'])
    label(ax, 3.15, 4.72, '4 → 6 → 6 → 2 fully connected', fs=13, weight='semibold')
    save('layout_mlp', f, 'Schematic.', kind='schematic')


@fig
def cnn_blocks():
    f, ax = canvas(11.8, 4.2)
    specs = [(.3, 1.6, 2.2, .25, BLUE, 'Input\n3×64×64'), (1.6, 1.3, 1.8, .7, TEAL, 'Conv\n16×64×64'),
             (3.5, 1.5, 1.2, .7, TEAL, 'Pool\n16×32×32'), (5.2, 1.35, 1.2, 1.2, TEAL, 'Conv\n32×32×32'),
             (7.3, 1.6, .7, 1.2, TEAL, 'Pool\n32×16×16')]
    prev = None
    for x, y, h, d, col, lab in specs:
        a = block3d(ax, x, y, .22 if d < .5 else d * .35, h, d, col, text_below=lab, fs=12)
        if prev is not None:
            arrow(ax, (prev.x + prev.w + .1, y + h / 2), (x - .08, y + h / 2), color=MUTED, lw=1.4)
        prev = a
    b = box(ax, 9.1, 1.75, 1.1, .9, 'Dense', ORANGE, fs=13)
    arrow(ax, (prev.x + prev.w + .1, 2.2), b.l, color=MUTED, lw=1.4)
    c2 = box(ax, 10.55, 1.8, 1.05, .8, 'Class\nscores', ORANGE, fill='solid', fs=12)
    arrow(ax, b.r, c2.l, color=MUTED, lw=1.4)
    save('layout_cnn_blocks', f, 'Schematic.', kind='schematic')


@fig
def conv_matrix():
    f, ax = canvas(6.3, 4.4)
    a = np.array([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 2], [1, 0, 1, 1]])
    matrix(ax, .3, 3.7, a, cell=.62, color=BLUE, highlight=(0, 0, 2, 2), title='Input')
    matrix(ax, 3.3, 3.1, [[1, 0], [0, -1]], cell=.62, color=ORANGE, title='Kernel')
    matrix(ax, 3.3, 1.55, [[0, -1], [0, 0]], cell=.62, cmap='course_seq', title='')
    label(ax, 3.92, 1.8, 'Output (top-left)', fs=12, color=MUTED)
    arrow(ax, (2.85, 2.8), (3.2, 2.8), color=MUTED)
    save('layout_conv_matrix', f, 'Exact arithmetic.', kind='computed')


@fig
def mlp_boundary():
    X, y = make_moons(300, noise=.25, random_state=0)
    m = MLPClassifier((16, 16), max_iter=2000, random_state=0).fit(X, y)
    f, axs = half()
    ax = axs[0]
    regions(ax, m, (-1.6, 2.6), (-1.2, 1.7), proba=True)
    points(ax, X, y, s=20)
    clean(ax, 'Feature 1', 'Feature 2', grid=False)
    save('layout_mlp_boundary', f, 'MLP fitted to synthetic moons; seed 0.', kind='computed')
