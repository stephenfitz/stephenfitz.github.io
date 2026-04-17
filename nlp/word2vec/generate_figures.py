#!/usr/bin/env python3
"""Generate figures for the Word2Vec article."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

FIGDIR = 'figures'
DPI = 200

BLUE = '#4A7FB5'
DARK_BLUE = '#2C5F8A'
RED = '#C0504D'
DARK_RED = '#8B3A3A'
GRAY = '#888888'
LIGHT_GRAY = '#CCCCCC'
GREEN = '#5A9E6F'
ORANGE = '#E8923F'
PURPLE = '#7B6AA8'
TAN = '#C7A876'


def draw_node(ax, x, y, label, color=LIGHT_GRAY, r=0.22, fontsize=9, edge='#444'):
    c = Circle((x, y), r, facecolor=color, edgecolor=edge, linewidth=1.3, zorder=3)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#888', lw=0.7, alpha=0.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle='-|>', mutation_scale=7,
                                  color=color, lw=lw, alpha=alpha, zorder=1))


# ============================================================================
# Figure 1: CBOW vs Skipgram architectural comparison
# ============================================================================
def plot_cbow_vs_skipgram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- CBOW: context -> center ---
    ax = axes[0]
    ax.set_title('CBOW: predict the center word from its context',
                 fontsize=12, fontweight='bold')
    context = [('the', 0.5, 3.5), ('cat', 0.5, 2.5),
               ('on', 0.5, 1.5), ('the', 0.5, 0.5)]
    center = ('sat', 4.5, 2.0)
    for word, x, y in context:
        draw_node(ax, x, y, word, color='#D9E4F0', r=0.35, fontsize=10)
    draw_node(ax, center[1], center[2], center[0],
              color='#F4D5D0', r=0.40, fontsize=11)
    draw_node(ax, 2.5, 2.0, r'$\bar h$', color=LIGHT_GRAY, r=0.32, fontsize=12)
    for _, x, y in context:
        arrow(ax, x + 0.36, y, 2.5 - 0.33, 2.0, color=BLUE, lw=1.2)
    arrow(ax, 2.5 + 0.33, 2.0, center[1] - 0.41, center[2],
          color=RED, lw=1.4)
    ax.text(2.5, 0.7, 'average\nof context\nembeddings',
            ha='center', fontsize=9, color='#555', style='italic')
    ax.text(4.5, 3.6, 'target', ha='center', fontsize=9,
            color=DARK_RED, style='italic')
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    # --- Skipgram: center -> context ---
    ax = axes[1]
    ax.set_title('Skipgram: predict the context from the center word',
                 fontsize=12, fontweight='bold')
    center2 = ('sat', 1.0, 2.0)
    context2 = [('the', 4.5, 3.5), ('cat', 4.5, 2.5),
                ('on', 4.5, 1.5), ('the', 4.5, 0.5)]
    draw_node(ax, center2[1], center2[2], center2[0],
              color='#F4D5D0', r=0.40, fontsize=11)
    for word, x, y in context2:
        draw_node(ax, x, y, word, color='#D9E4F0', r=0.35, fontsize=10)
        arrow(ax, center2[1] + 0.41, center2[2], x - 0.36, y,
              color=RED, lw=1.3)
    ax.text(1.0, 0.7, 'center\nword', ha='center', fontsize=9,
            color=DARK_RED, style='italic')
    ax.text(4.5, 4.1, 'context', ha='center', fontsize=9,
            color='#555', style='italic')
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/cbow_vs_skipgram.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved cbow_vs_skipgram.png')


# ============================================================================
# Figure 2: One-word CBOW architecture (V -> N -> V neural network)
# ============================================================================
def plot_cbow_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    V = 7   # vocabulary size
    N = 3   # hidden size
    left_x, mid_x, right_x = 0.8, 5.0, 9.2

    # Input layer
    for i in range(V):
        y = 5.5 - i * 0.85
        is_active = (i == 2)
        color = RED if is_active else LIGHT_GRAY
        draw_node(ax, left_x, y, '1' if is_active else '0',
                  color=color, r=0.26, fontsize=10)
    ax.text(left_x, 6.4, 'input $x$\n(one-hot, size $V$)',
            ha='center', fontsize=10, fontweight='bold')

    # Hidden layer
    for j in range(N):
        y = 4.0 - j * 0.85
        draw_node(ax, mid_x, y, f'$h_{j+1}$', color='#E4EAF0',
                  r=0.28, fontsize=10)
    ax.text(mid_x, 5.2, 'hidden $h = v_{w_I}$\n(embedding, size $N$)',
            ha='center', fontsize=10, fontweight='bold')

    # Output layer
    for k in range(V):
        y = 5.5 - k * 0.85
        is_target = (k == 4)
        color = BLUE if is_target else LIGHT_GRAY
        draw_node(ax, right_x, y, '', color=color, r=0.26)
    ax.text(right_x, 6.4, 'output $y$\n(softmax over $V$)',
            ha='center', fontsize=10, fontweight='bold')

    # Arrows input -> hidden (fade all but the active row)
    for i in range(V):
        y1 = 5.5 - i * 0.85
        for j in range(N):
            y2 = 4.0 - j * 0.85
            alpha = 0.85 if i == 2 else 0.10
            color = RED if i == 2 else GRAY
            lw = 1.2 if i == 2 else 0.6
            arrow(ax, left_x + 0.27, y1, mid_x - 0.29, y2,
                  color=color, lw=lw, alpha=alpha)

    # Arrows hidden -> output (all connections)
    for j in range(N):
        y1 = 4.0 - j * 0.85
        for k in range(V):
            y2 = 5.5 - k * 0.85
            arrow(ax, mid_x + 0.29, y1, right_x - 0.27, y2,
                  color=BLUE, lw=0.6, alpha=0.35)

    # Matrix annotations
    ax.text((left_x + mid_x) / 2, 6.0, '$W \\in \\mathbb{R}^{V \\times N}$',
            ha='center', fontsize=11, color=DARK_RED)
    ax.text((mid_x + right_x) / 2, 6.0, "$W' \\in \\mathbb{R}^{N \\times V}$",
            ha='center', fontsize=11, color=DARK_BLUE)
    ax.text((left_x + mid_x) / 2, -0.5,
            'picks row $k$ of $W$:\n$h = W^{\\top} x = v_{w_I}$',
            ha='center', fontsize=9.5, style='italic', color='#444')
    ax.text((mid_x + right_x) / 2, -0.5,
            "scores $u_j = v'^{\\top}_{w_j} h$,\n then softmax",
            ha='center', fontsize=9.5, style='italic', color='#444')

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-1.2, 7.0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/cbow_architecture.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved cbow_architecture.png')


# ============================================================================
# Figure 3: Two representations per word (input and output embeddings)
# ============================================================================
def plot_two_representations():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    words = ['cat', 'dog', 'sat', 'ran', 'the', 'a']

    # Matrix W: rows = input embeddings v_w
    ax = axes[0]
    ax.set_title("$W$: each row is an input vector $v_w$",
                 fontsize=11, fontweight='bold')
    np.random.seed(0)
    M = np.random.uniform(-1, 1, size=(len(words), 5))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'd{i+1}' for i in range(5)], fontsize=9)
    ax.set_xlabel('embedding dimensions ($N$)', fontsize=10)
    ax.set_ylabel('vocabulary ($V$ words)', fontsize=10)

    # Matrix W': columns = output embeddings v'_w
    ax = axes[1]
    ax.set_title("$W'$: each column is an output vector $v'_w$",
                 fontsize=11, fontweight='bold')
    np.random.seed(7)
    Mp = np.random.uniform(-1, 1, size=(5, len(words)))
    ax.imshow(Mp, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'd{i+1}' for i in range(5)], fontsize=9)
    ax.set_xlabel('vocabulary ($V$ words)', fontsize=10)
    ax.set_ylabel('embedding dimensions ($N$)', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/two_representations.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved two_representations.png')


# ============================================================================
# Figure 4: Negative sampling loss landscape (log sigmoid)
# ============================================================================
def plot_negative_sampling_loss():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    z = np.linspace(-6, 6, 400)
    sig = 1 / (1 + np.exp(-z))
    log_sig_pos = np.log(sig + 1e-12)
    log_sig_neg = np.log(1 - sig + 1e-12)

    # Panel 1: the two log-sigmoid curves
    ax = axes[0]
    ax.plot(z, log_sig_pos, color=BLUE, linewidth=2.2,
            label=r'$\log \sigma(v_c \cdot v_w)$  (positive pair)')
    ax.plot(z, log_sig_neg, color=RED, linewidth=2.2,
            label=r'$\log \sigma(-v_c \cdot v_w)$  (negative pair)')
    ax.axhline(0, color='#999', lw=0.5)
    ax.axvline(0, color='#999', lw=0.5)
    ax.set_xlabel(r'inner product $v_c \cdot v_w$', fontsize=11)
    ax.set_ylabel('contribution to objective', fontsize=11)
    ax.set_title('Log-sigmoid rewards: pushing positives up, negatives down',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)

    # Panel 2: sigmoid (probability pair was observed)
    ax = axes[1]
    ax.plot(z, sig, color=PURPLE, linewidth=2.2)
    ax.fill_between(z, 0, sig, where=(z > 0), alpha=0.15, color=BLUE,
                    label='favors "observed"')
    ax.fill_between(z, 0, sig, where=(z <= 0), alpha=0.15, color=RED,
                    label='favors "not observed"')
    ax.set_xlabel(r'inner product $v_c \cdot v_w$', fontsize=11)
    ax.set_ylabel(r'$P(D=1 \mid w,c) = \sigma(v_c \cdot v_w)$', fontsize=11)
    ax.set_title('Probability that a pair came from the corpus',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.axhline(0.5, color='#999', lw=0.5, linestyle='--')
    ax.axvline(0, color='#999', lw=0.5)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/negative_sampling_loss.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved negative_sampling_loss.png')


# ============================================================================
# Figure 5: Linear structure - "king - man + woman = queen" style analogies
# ============================================================================
def plot_linear_structure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1: gender axis (2D cartoon)
    ax = axes[0]
    ax.set_title('Linear structure: gender and royalty axes',
                 fontsize=11, fontweight='bold')

    pts = {
        'king':   (3.0, 3.2),
        'queen':  (3.0, 4.4),
        'man':    (1.2, 1.2),
        'woman':  (1.2, 2.4),
    }
    colors_m = {'king': DARK_BLUE, 'queen': '#9C4F8B',
                'man':  BLUE,     'woman':  PURPLE}
    for w, (x, y) in pts.items():
        ax.plot(x, y, 'o', color=colors_m[w], markersize=10, zorder=4)
        dx = -0.22 if w in ('queen', 'woman') else 0.22
        ha = 'right' if w in ('queen', 'woman') else 'left'
        ax.text(x + dx, y, w, fontsize=12, ha=ha, va='center',
                color=colors_m[w], fontweight='bold')

    # gender arrows (dashed)
    ax.annotate('', xy=pts['queen'], xytext=pts['king'],
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=1.5, linestyle='--'))
    ax.annotate('', xy=pts['woman'], xytext=pts['man'],
                arrowprops=dict(arrowstyle='->', color=GRAY,
                                lw=1.5, linestyle='--'))
    ax.text(3.55, 3.8, 'gender', fontsize=9,
            color='#555', style='italic', rotation=90)
    ax.text(1.75, 1.8, 'gender', fontsize=9,
            color='#555', style='italic', rotation=90)

    # royalty arrows (solid green)
    ax.annotate('', xy=pts['king'], xytext=pts['man'],
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))
    ax.annotate('', xy=pts['queen'], xytext=pts['woman'],
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))
    ax.text(2.0, 2.0, 'royalty', fontsize=9, color=GREEN,
            style='italic', rotation=47)

    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 5.3)
    ax.set_xlabel('dimension 1', fontsize=10)
    ax.set_ylabel('dimension 2', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    # Panel 2: singular/plural axis
    ax = axes[1]
    ax.set_title('Singular/plural shift',
                 fontsize=11, fontweight='bold')
    sp = {
        'cat':  (1.0, 1.0),
        'cats': (1.0, 3.1),
        'dog':  (2.2, 1.4),
        'dogs': (2.2, 3.5),
        'car':  (3.4, 0.6),
        'cars': (3.4, 2.7),
    }
    for w, (x, y) in sp.items():
        is_plural = w.endswith('s') and len(w) > 2
        color = PURPLE if is_plural else BLUE
        ax.plot(x, y, 'o', color=color, markersize=9, zorder=4)
        ax.text(x + 0.12, y, w, fontsize=11, color=color, va='center')
    # shift vectors
    pairs = [('cat', 'cats'), ('dog', 'dogs'), ('car', 'cars')]
    for a, b in pairs:
        x1, y1 = sp[a]
        x2, y2 = sp[b]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=GREEN,
                                    lw=1.5, alpha=0.9))
    ax.text(3.0, 4.3, 'plural direction', fontsize=10,
            color=GREEN, style='italic')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 4.8)
    ax.set_xlabel('dimension 1', fontsize=10)
    ax.set_ylabel('dimension 2', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/linear_structure.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved linear_structure.png')


# ============================================================================
# Figure 6: Context window illustration
# ============================================================================
def plot_context_window():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    sentence = ['the', 'quick', 'brown', 'fox', 'jumps',
                'over', 'the', 'lazy', 'dog']
    centers = [3]  # focus on "fox"
    window = 2

    for i, w in enumerate(sentence):
        x = i * 1.2
        if i == centers[0]:
            color = '#F4D5D0'; edge = DARK_RED
        elif centers[0] - window <= i <= centers[0] + window:
            color = '#D9E4F0'; edge = DARK_BLUE
        else:
            color = '#F0F0F0'; edge = '#999'
        rect = FancyBboxPatch((x - 0.48, 0.5), 0.96, 0.7,
                              boxstyle="round,pad=0.04",
                              facecolor=color, edgecolor=edge, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x, 0.85, w, ha='center', va='center', fontsize=11)

    # window brackets
    cx = centers[0] * 1.2
    ax.annotate('', xy=(cx + window * 1.2 + 0.5, 1.4),
                xytext=(cx - window * 1.2 - 0.5, 1.4),
                arrowprops=dict(arrowstyle='<->', color=DARK_BLUE, lw=1.4))
    ax.text(cx, 1.65, f'context window (size {window})',
            ha='center', fontsize=10, color=DARK_BLUE, fontweight='bold')
    ax.text(cx, 0.25, 'center word',
            ha='center', fontsize=10, color=DARK_RED, fontweight='bold')

    # sample training pairs
    pairs = [('fox', 'quick'), ('fox', 'brown'), ('fox', 'jumps'), ('fox', 'over')]
    pair_str = '   '.join([f'({a}, {b})' for a, b in pairs])
    ax.text(5.0, -0.3,
            'skipgram pairs from this center: ' + pair_str,
            ha='center', fontsize=9.5, color='#333', style='italic')

    ax.set_xlim(-0.8, 11)
    ax.set_ylim(-0.8, 2.1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/context_window.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved context_window.png')


# ============================================================================
# Figure 7: Toy skipgram embeddings trained on a tiny corpus (PCA scatter)
# ============================================================================
def plot_toy_embeddings():
    rng = np.random.default_rng(42)

    # Toy corpus: animals + actions + determiners. Words that share contexts
    # should end up near each other.
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat slept on the bed",
        "the dog slept on the mat",
        "a cat chased a mouse",
        "a dog chased a cat",
        "the king ruled the kingdom",
        "the queen ruled the kingdom",
        "a man walked home",
        "a woman walked home",
        "the man loved the woman",
        "the woman loved the man",
        "kings and queens ruled",
        "men and women walked",
    ]

    tokens = [t for s in sentences for t in s.split()]
    vocab = sorted(set(tokens))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = 20   # embedding dim
    window = 2
    negatives = 5

    # Build skipgram (center, context) pairs
    pairs = []
    for s in sentences:
        toks = s.split()
        for i, w in enumerate(toks):
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j == i:
                    continue
                pairs.append((w2i[w], w2i[toks[j]]))

    V_in  = rng.standard_normal((V, N)) * 0.1   # center vectors
    V_out = rng.standard_normal((V, N)) * 0.1   # context vectors

    # Unigram noise distribution raised to 0.75, as in the paper
    counts = np.zeros(V)
    for t in tokens:
        counts[w2i[t]] += 1
    probs = counts ** 0.75
    probs = probs / probs.sum()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    lr = 0.05
    epochs = 800
    for epoch in range(epochs):
        rng.shuffle(pairs)
        for w, c in pairs:
            # positive update
            vw = V_in[w]
            vc = V_out[c]
            score = vw @ vc
            g = sigmoid(score) - 1.0     # d/d(score) of -log sigmoid(score)
            grad_vw = g * vc
            grad_vc = g * vw
            V_in[w]  -= lr * grad_vw
            V_out[c] -= lr * grad_vc

            # k negative samples
            negs = rng.choice(V, size=negatives, p=probs)
            for n in negs:
                vn = V_out[n]
                score = V_in[w] @ vn
                g = sigmoid(score)       # d/d(score) of -log sigmoid(-score)
                V_in[w]  -= lr * g * vn
                V_out[n] -= lr * g * V_in[w]

    # PCA project to 2D
    X = V_in - V_in.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T

    # Color words by semantic group
    groups = {
        'animal':    {'cat', 'dog', 'mouse'},
        'action':    {'sat', 'slept', 'chased', 'ruled', 'walked', 'loved'},
        'human':     {'man', 'woman', 'king', 'queen', 'men', 'women',
                      'kings', 'queens'},
        'place':     {'mat', 'rug', 'bed', 'kingdom', 'home'},
        'function':  {'the', 'a', 'on', 'and'},
    }
    group_color = {
        'animal':   BLUE,
        'action':   RED,
        'human':    PURPLE,
        'place':    GREEN,
        'function': GRAY,
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for g, members in groups.items():
        xs, ys = [], []
        for w in members:
            if w in w2i:
                x, y = coords[w2i[w]]
                xs.append(x); ys.append(y)
        ax.scatter(xs, ys, s=80, color=group_color[g],
                   edgecolor='#333', linewidth=0.8,
                   label=g, zorder=3, alpha=0.85)

    for w in vocab:
        x, y = coords[w2i[w]]
        ax.annotate(w, (x, y), fontsize=10.5,
                    xytext=(5, 3), textcoords='offset points',
                    color='#222')

    ax.set_title('Skipgram-with-negative-sampling embeddings '
                 '(toy corpus, PCA projection)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/toy_embeddings.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved toy_embeddings.png')


if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    plot_cbow_vs_skipgram()
    plot_cbow_architecture()
    plot_two_representations()
    plot_negative_sampling_loss()
    plot_linear_structure()
    plot_context_window()
    plot_toy_embeddings()
    print('All figures generated.')
