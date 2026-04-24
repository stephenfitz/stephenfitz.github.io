#!/usr/bin/env python3
"""Generate figures for the Recurrent Neural Networks for NLP article."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

DPI = 200
BLUE = '#4472C4'
RED = '#C44E52'
GRAY = '#888888'
LIGHT_BLUE = '#A8C4E0'
DARK_BLUE = '#2B4570'
ORANGE = '#E8853D'
GREEN = '#59A14F'
PURPLE = '#7B68A0'
LIGHT_GREEN = '#D5E8D4'
LIGHT_RED = '#FADBD8'
LIGHT_ORANGE = '#FEF9E7'
LIGHT_PURPLE = '#E8D5F5'

FIGURES_DIR = 'figures'


def save(fig, name):
    fig.savefig(f'{FIGURES_DIR}/{name}', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved {name}')


def _draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=10,
              fontweight='bold', text_color='black'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color)


def _arrow(ax, x1, y1, x2, y2, color='black', lw=1.2, style='->', **kwargs):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, **kwargs))


def fig_rnn_unrolled():
    """Unrolled RNN showing the recurrence across time steps."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.set_xlim(-2.5, 13)
    ax.set_ylim(-1.2, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Compact (folded) RNN on the left
    cx, cy = -0.5, 1.5
    _draw_box(ax, cx, cy, 1.2, 0.9, '$h$', LIGHT_BLUE, DARK_BLUE)
    _draw_box(ax, cx, -0.2, 1.0, 0.7, '$x$', '#E8E8E8', GRAY, fontsize=9)
    _draw_box(ax, cx, 3.2, 1.0, 0.7, '$y$', '#E8E8E8', GRAY, fontsize=9)
    _arrow(ax, cx, 0.18, cx, 1.02, color=DARK_BLUE)
    _arrow(ax, cx, 1.98, cx, 2.82, color=DARK_BLUE)
    # Self-loop
    from matplotlib.patches import Arc
    arc = Arc((cx + 0.65, cy), 0.8, 1.4, angle=0, theta1=-60, theta2=60,
              color=RED, lw=1.5)
    ax.add_patch(arc)
    ax.annotate('', xy=(cx + 0.64, cy + 0.58), xytext=(cx + 0.7, cy + 0.52),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax.text(cx, 4.0, 'Folded', ha='center', va='center', fontsize=10,
            fontweight='bold', color=DARK_BLUE)

    # Equals / unroll arrow
    ax.text(1.8, 1.5, '=', ha='center', va='center', fontsize=20, color=GRAY)

    # Unrolled RNN
    n_steps = 5
    tokens = ['$x_1$', '$x_2$', '$x_3$', '$\\cdots$', '$x_T$']
    h_labels = ['$h_1$', '$h_2$', '$h_3$', '$\\cdots$', '$h_T$']
    y_labels = ['$y_1$', '$y_2$', '$y_3$', '$\\cdots$', '$y_T$']

    for i in range(n_steps):
        px = 3.5 + i * 2.0
        # Input
        _draw_box(ax, px, -0.2, 0.9, 0.6, tokens[i], '#E8E8E8', GRAY, fontsize=9)
        # Hidden state
        _draw_box(ax, px, 1.5, 1.1, 0.8, h_labels[i], LIGHT_BLUE, DARK_BLUE)
        # Output
        _draw_box(ax, px, 3.2, 0.9, 0.6, y_labels[i], '#E8E8E8', GRAY, fontsize=9)
        # Input -> hidden
        _arrow(ax, px, 0.13, px, 1.07, color=DARK_BLUE)
        # Hidden -> output
        _arrow(ax, px, 1.93, px, 2.87, color=DARK_BLUE)
        # Hidden -> next hidden
        if i < n_steps - 1:
            _arrow(ax, px + 0.58, 1.5, px + 2.0 - 0.58, 1.5, color=RED, lw=1.8)

    # h_0 arrow
    _arrow(ax, 2.5, 1.5, 3.5 - 0.58, 1.5, color=RED, lw=1.8)
    ax.text(2.3, 1.5, '$h_0$', ha='center', va='center', fontsize=9, color=RED)

    ax.text(7.5, 4.0, 'Unrolled through time', ha='center', va='center',
            fontsize=10, fontweight='bold', color=DARK_BLUE)

    # Shared weights label
    ax.text(7.5, -0.95, 'Same weights $W_h, W_x, W_y$ at every step',
            ha='center', va='center', fontsize=9, color=RED, fontstyle='italic')

    save(fig, 'rnn_unrolled.png')


def fig_vanishing_gradient():
    """Show gradient magnitude decay over time steps."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: gradient flow diagram
    ax = axes[0]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Gradient flow through time', fontsize=11, fontweight='bold', pad=8)

    n = 6
    for i in range(n):
        alpha = 0.3 + 0.7 * (i / (n - 1))  # gets brighter toward loss
        _draw_box(ax, i * 1.5 + 0.5, 1.0, 0.9, 0.7,
                  f'$h_{i+1}$', LIGHT_BLUE, DARK_BLUE)
        if i < n - 1:
            ax.annotate('', xy=(i * 1.5 + 0.5 + 0.48, 1.0),
                        xytext=((i + 1) * 1.5 + 0.5 - 0.48, 1.0),
                        arrowprops=dict(arrowstyle='<-', color=RED,
                                        lw=2.5 * alpha, alpha=alpha))

    ax.text(n * 1.5 - 0.4, 1.0, '$\\mathcal{L}$', ha='center', va='center',
            fontsize=12, fontweight='bold', color=RED)
    ax.annotate('', xy=((n-1) * 1.5 + 0.5 + 0.48, 1.0),
                xytext=(n * 1.5 - 0.65, 1.0),
                arrowprops=dict(arrowstyle='<-', color=RED, lw=2.5))

    ax.text(4.0, 0.0, 'Gradient signal weakens as it flows backward',
            ha='center', va='center', fontsize=8, color=RED, fontstyle='italic')

    # Right: gradient magnitude plot
    ax = axes[1]
    steps = np.arange(1, 21)
    # Vanishing
    grad_vanish = 0.7 ** steps
    # Exploding
    grad_explode = 1.3 ** steps

    ax.semilogy(steps, grad_vanish, 'o-', color=BLUE, lw=2, markersize=4,
                label='$\\|\\partial h_T / \\partial h_t\\|$ when $\\|W_h\\| < 1$')
    ax.semilogy(steps, grad_explode, 's-', color=RED, lw=2, markersize=4,
                label='$\\|\\partial h_T / \\partial h_t\\|$ when $\\|W_h\\| > 1$')
    ax.axhline(y=1, color=GRAY, linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('Distance $T - t$ (time steps back)', fontsize=10)
    ax.set_ylabel('Gradient magnitude', fontsize=10)
    ax.set_title('Vanishing vs. exploding gradients', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.set_xlim(0, 21)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, 'vanishing_gradient.png')


def fig_lstm_cell():
    """LSTM cell diagram showing the four gates and cell state."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Cell state highway at top
    ax.annotate('', xy=(9.5, 6.5), xytext=(1.0, 6.5),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=3))
    ax.text(5.0, 7.0, 'Cell state $c_t$', ha='center', va='center',
            fontsize=11, fontweight='bold', color=GREEN)
    ax.text(0.5, 6.5, '$c_{t-1}$', ha='center', va='center', fontsize=10, color=GREEN)

    # Forget gate
    fg_x, fg_y = 2.5, 4.5
    _draw_box(ax, fg_x, fg_y, 1.6, 0.9, '$\\sigma$', LIGHT_RED, RED, fontsize=12)
    ax.text(fg_x, 3.4, 'Forget\ngate $f_t$', ha='center', va='center',
            fontsize=8, color=RED, fontstyle='italic')
    # Forget gate multiply on cell state
    c_fg = Circle((fg_x, 6.5), 0.3, facecolor='#FFF2CC', edgecolor=ORANGE, linewidth=2)
    ax.add_patch(c_fg)
    ax.text(fg_x, 6.5, '$\\times$', ha='center', va='center', fontsize=12,
            fontweight='bold')
    _arrow(ax, fg_x, 5.0, fg_x, 6.17, color=RED, lw=1.5)

    # Input gate
    ig_x, ig_y = 5.0, 4.5
    _draw_box(ax, ig_x, ig_y, 1.6, 0.9, '$\\sigma$', LIGHT_BLUE, BLUE, fontsize=12)
    ax.text(ig_x, 3.4, 'Input\ngate $i_t$', ha='center', va='center',
            fontsize=8, color=BLUE, fontstyle='italic')

    # Candidate
    cand_x, cand_y = 6.5, 4.5
    _draw_box(ax, cand_x, cand_y, 1.6, 0.9, 'tanh', LIGHT_GREEN, GREEN, fontsize=11)
    ax.text(cand_x, 3.4, 'Candidate\n$\\tilde{c}_t$', ha='center', va='center',
            fontsize=8, color=GREEN, fontstyle='italic')

    # Input gate * candidate multiply
    mult_x = 5.7
    c_ig = Circle((mult_x, 5.7), 0.3, facecolor='#FFF2CC', edgecolor=ORANGE, linewidth=2)
    ax.add_patch(c_ig)
    ax.text(mult_x, 5.7, '$\\times$', ha='center', va='center', fontsize=12,
            fontweight='bold')
    _arrow(ax, ig_x, 5.0, mult_x - 0.15, 5.42, color=BLUE, lw=1.2)
    _arrow(ax, cand_x, 5.0, mult_x + 0.15, 5.42, color=GREEN, lw=1.2)

    # Add to cell state
    c_add = Circle((mult_x, 6.5), 0.3, facecolor='#FFF2CC', edgecolor=ORANGE, linewidth=2)
    ax.add_patch(c_add)
    ax.text(mult_x, 6.5, '$+$', ha='center', va='center', fontsize=14,
            fontweight='bold')
    _arrow(ax, mult_x, 6.03, mult_x, 6.2, color=ORANGE, lw=1.5)

    # Output gate
    og_x, og_y = 8.5, 4.5
    _draw_box(ax, og_x, og_y, 1.6, 0.9, '$\\sigma$', LIGHT_PURPLE, PURPLE, fontsize=12)
    ax.text(og_x, 3.4, 'Output\ngate $o_t$', ha='center', va='center',
            fontsize=8, color=PURPLE, fontstyle='italic')

    # tanh of cell state
    tanh_x = 8.5
    _draw_box(ax, tanh_x, 5.7, 1.2, 0.7, 'tanh', LIGHT_GREEN, GREEN, fontsize=9)
    _arrow(ax, 7.5, 6.5, tanh_x - 0.3, 6.5, color=GREEN, lw=1,
           style='-')  # tap from cell state
    _arrow(ax, tanh_x, 6.15, tanh_x, 6.08, color=GREEN, lw=1)

    # Output multiply
    c_out = Circle((tanh_x, 2.5), 0.3, facecolor='#FFF2CC', edgecolor=ORANGE, linewidth=2)
    ax.add_patch(c_out)
    ax.text(tanh_x, 2.5, '$\\times$', ha='center', va='center', fontsize=12,
            fontweight='bold')
    _arrow(ax, og_x, 4.02, tanh_x, 2.83, color=PURPLE, lw=1.3)
    _arrow(ax, tanh_x, 5.32, tanh_x, 2.83, color=GREEN, lw=1.3)

    # Hidden state output
    _arrow(ax, tanh_x, 2.17, tanh_x, 1.2, color=DARK_BLUE, lw=1.8)
    ax.text(tanh_x, 0.8, '$h_t$', ha='center', va='center', fontsize=12,
            fontweight='bold', color=DARK_BLUE)
    # Also goes right
    _arrow(ax, tanh_x + 0.33, 2.5, 10.0, 2.5, color=DARK_BLUE, lw=1.5)
    ax.text(10.3, 2.5, '$h_t$', ha='center', va='center', fontsize=10,
            color=DARK_BLUE)

    # Inputs at bottom
    # h_{t-1} and x_t
    input_y = 1.5
    ax.text(0.0, input_y, '$h_{t-1}$', ha='center', va='center', fontsize=10,
            fontweight='bold', color=DARK_BLUE)
    ax.text(0.0, 0.5, '$x_t$', ha='center', va='center', fontsize=10,
            fontweight='bold', color=DARK_BLUE)

    # Draw input connections going to all gates
    for gx in [fg_x, ig_x, cand_x, og_x]:
        _arrow(ax, 1.0, input_y, gx, 4.02, color=GRAY, lw=0.8)

    ax.set_title('LSTM Cell', fontsize=14, fontweight='bold', pad=15)
    save(fig, 'lstm_cell.png')


def fig_gru_cell():
    """GRU cell diagram showing reset and update gates."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Hidden state highway
    ax.annotate('', xy=(9.0, 5.0), xytext=(0.5, 5.0),
                arrowprops=dict(arrowstyle='->', color=DARK_BLUE, lw=3))
    ax.text(0.0, 5.0, '$h_{t-1}$', ha='center', va='center', fontsize=10,
            color=DARK_BLUE, fontweight='bold')
    ax.text(9.5, 5.0, '$h_t$', ha='center', va='center', fontsize=10,
            color=DARK_BLUE, fontweight='bold')

    # Update gate
    ug_x, ug_y = 3.0, 3.0
    _draw_box(ax, ug_x, ug_y, 1.6, 0.9, '$\\sigma$', LIGHT_BLUE, BLUE, fontsize=12)
    ax.text(ug_x, 2.0, 'Update\ngate $z_t$', ha='center', va='center',
            fontsize=8, color=BLUE, fontstyle='italic')

    # Reset gate
    rg_x, rg_y = 5.5, 3.0
    _draw_box(ax, rg_x, rg_y, 1.6, 0.9, '$\\sigma$', LIGHT_RED, RED, fontsize=12)
    ax.text(rg_x, 2.0, 'Reset\ngate $r_t$', ha='center', va='center',
            fontsize=8, color=RED, fontstyle='italic')

    # Candidate hidden state
    cand_x, cand_y = 7.5, 3.0
    _draw_box(ax, cand_x, cand_y, 1.6, 0.9, 'tanh', LIGHT_GREEN, GREEN, fontsize=11)
    ax.text(cand_x, 2.0, 'Candidate\n$\\tilde{h}_t$', ha='center', va='center',
            fontsize=8, color=GREEN, fontstyle='italic')

    # Interpolation on the highway
    # (1-z) * h_{t-1} + z * candidate
    c_interp = Circle((5.5, 5.0), 0.35, facecolor='#FFF2CC', edgecolor=ORANGE,
                       linewidth=2)
    ax.add_patch(c_interp)
    ax.text(5.5, 5.0, 'lerp', ha='center', va='center', fontsize=7,
            fontweight='bold', color=ORANGE)

    # z_t to interpolation
    _arrow(ax, ug_x, 3.48, 5.15, 4.68, color=BLUE, lw=1.5)
    # candidate to interpolation
    _arrow(ax, cand_x, 3.48, 5.85, 4.68, color=GREEN, lw=1.5)

    # Reset gate to candidate (modulates h_{t-1})
    _arrow(ax, rg_x + 0.5, 3.48, cand_x - 0.5, 3.48, color=RED, lw=1.2)

    # Inputs
    ax.text(1.0, 0.5, '$[h_{t-1}, x_t]$', ha='center', va='center', fontsize=10,
            fontweight='bold', color=GRAY)
    for gx in [ug_x, rg_x]:
        _arrow(ax, 1.8, 0.7, gx, 2.52, color=GRAY, lw=0.8)
    _arrow(ax, 1.8, 0.7, cand_x, 2.52, color=GRAY, lw=0.8)

    # Formula
    ax.text(5.0, 6.2, '$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_ORANGE,
                      edgecolor=ORANGE, linewidth=1))

    ax.set_title('GRU Cell', fontsize=14, fontweight='bold', pad=10)
    save(fig, 'gru_cell.png')


def fig_encoder_decoder():
    """Encoder-decoder (seq2seq) architecture for machine translation."""
    fig, ax = plt.subplots(1, 1, figsize=(15.5, 5.0))
    ax.set_xlim(-1, 20.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Encoder
    enc_tokens = ['je', 'suis', 'un', 'chat', '<eos>']
    for i, tok in enumerate(enc_tokens):
        px = i * 1.8 + 0.5
        # Input
        _draw_box(ax, px, -0.5, 1.0, 0.6, tok, '#E8E8E8', GRAY, fontsize=9)
        # Hidden state
        _draw_box(ax, px, 1.0, 1.2, 0.8, f'$h_{i+1}^e$', LIGHT_BLUE, DARK_BLUE,
                  fontsize=9)
        _arrow(ax, px, -0.17, px, 0.57, color=DARK_BLUE)
        if i > 0:
            _arrow(ax, px - 1.8 + 0.63, 1.0, px - 0.63, 1.0, color=RED, lw=1.8)

    enc_end_x = 4 * 1.8 + 0.5  # last encoder position

    # Context vector
    ctx_x = enc_end_x + 1.8
    _draw_box(ax, ctx_x, 1.0, 1.0, 0.8, '$c$', '#F9E79F', ORANGE, fontsize=11)
    _arrow(ax, enc_end_x + 0.63, 1.0, ctx_x - 0.53, 1.0, color=RED, lw=2)
    ax.text(ctx_x, 2.2, 'context', ha='center', va='center', fontsize=8,
            color=ORANGE, fontstyle='italic')

    # Decoder
    dec_tokens = ['<sos>', 'I', 'am', 'a', 'cat']
    out_tokens = ['I', 'am', 'a', 'cat', '<eos>']
    dec_start_x = ctx_x + 1.8
    for i, (tok, out) in enumerate(zip(dec_tokens, out_tokens)):
        px = dec_start_x + i * 1.8
        # Input (previous output)
        _draw_box(ax, px, -0.5, 1.0, 0.6, tok, LIGHT_RED, RED, fontsize=9)
        # Hidden state
        _draw_box(ax, px, 1.0, 1.2, 0.8, f'$h_{i+1}^d$', LIGHT_RED, RED, fontsize=9)
        # Output
        _draw_box(ax, px, 2.8, 1.0, 0.6, out, '#E8E8E8', GRAY, fontsize=9,
                  fontweight='normal')
        _arrow(ax, px, -0.17, px, 0.57, color=RED)
        _arrow(ax, px, 1.43, px, 2.47, color=RED)
        if i > 0:
            _arrow(ax, px - 1.8 + 0.63, 1.0, px - 0.63, 1.0, color=RED, lw=1.8)

    # Context to first decoder
    _arrow(ax, ctx_x + 0.53, 1.0, dec_start_x - 0.63, 1.0, color=ORANGE, lw=2)

    # Labels
    ax.text(4.0, 4.5, 'ENCODER', ha='center', va='center', fontsize=13,
            fontweight='bold', color=DARK_BLUE)
    ax.text(dec_start_x + 3.6, 4.5, 'DECODER', ha='center', va='center',
            fontsize=13, fontweight='bold', color=RED)

    # Dashed separator
    sep_x = ctx_x
    ax.plot([sep_x, sep_x], [-1.2, 4.2], color=GRAY, linestyle=':', lw=1.5, alpha=0.5)

    save(fig, 'encoder_decoder.png')


def fig_attention_mechanism():
    """Bahdanau-style attention mechanism in encoder-decoder models."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Encoder hidden states
    enc_labels = ['$h_1^e$', '$h_2^e$', '$h_3^e$', '$h_4^e$', '$h_5^e$']
    enc_words = ['je', 'suis', 'un', 'chat', '<eos>']
    enc_positions = []
    for i, (label, word) in enumerate(zip(enc_labels, enc_words)):
        px = i * 2.0 + 0.5
        enc_positions.append(px)
        _draw_box(ax, px, 1.0, 1.2, 0.7, label, LIGHT_BLUE, DARK_BLUE, fontsize=9)
        ax.text(px, 0.15, word, ha='center', va='center', fontsize=8, color=GRAY)

    # Decoder state
    dec_x = 11.5
    _draw_box(ax, dec_x, 3.5, 1.4, 0.8, '$s_{t-1}$', LIGHT_RED, RED, fontsize=10)
    ax.text(dec_x, 2.5, 'Decoder\nstate', ha='center', va='center',
            fontsize=8, color=RED, fontstyle='italic')

    # Score / alignment
    score_y = 3.5
    weights = [0.05, 0.10, 0.05, 0.70, 0.10]
    for i, (px, w) in enumerate(zip(enc_positions, weights)):
        # Arrow from decoder to score
        _arrow(ax, dec_x - 0.73, 3.5, px + 0.63, 1.4, color=ORANGE, lw=0.8,
               alpha=0.3 + w)
        # Weight annotation
        bar_h = w * 3.0
        rect = plt.Rectangle((px - 0.3, 4.5), 0.6, bar_h,
                              facecolor=ORANGE, edgecolor='black',
                              linewidth=0.6, alpha=0.6)
        ax.add_patch(rect)
        ax.text(px, 4.5 + bar_h + 0.15, f'{w:.2f}', ha='center', va='bottom',
                fontsize=7, color=ORANGE, fontweight='bold')

    ax.text(-0.5, 5.3, 'Attention\nweights $\\alpha_t$', ha='right', va='center',
            fontsize=9, color=ORANGE, fontweight='bold')

    # Context vector
    ctx_x, ctx_y = 5.5, 6.8
    _draw_box(ax, ctx_x, ctx_y, 2.8, 0.7,
              '$c_t = \\sum_j \\alpha_{tj} h_j^e$',
              LIGHT_ORANGE, ORANGE, fontsize=9)

    # Arrow from context to decoder
    _arrow(ax, ctx_x + 1.43, ctx_y, dec_x - 0.5, 6.8, color=ORANGE, lw=1.5)

    # New decoder state
    _draw_box(ax, dec_x, 6.8, 1.4, 0.7, '$s_t$', LIGHT_RED, RED, fontsize=10)
    _arrow(ax, dec_x, 3.93, dec_x, 6.42, color=RED, lw=1.5)

    # Output
    ax.text(dec_x, 7.4, '"cat"', ha='center', va='bottom', fontsize=10,
            fontweight='bold', color=GREEN)

    ax.set_title('Bahdanau Attention Mechanism', fontsize=13, fontweight='bold', pad=10)
    save(fig, 'attention_mechanism.png')


def fig_bidirectional_rnn():
    """Bidirectional RNN showing forward and backward passes."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-1.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    n = 5
    tokens = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$']

    for i in range(n):
        px = i * 2.2 + 1.0
        # Input
        _draw_box(ax, px, -0.5, 0.9, 0.6, tokens[i], '#E8E8E8', GRAY, fontsize=9)
        # Forward hidden state
        _draw_box(ax, px, 1.0, 1.1, 0.7, f'$\\overrightarrow{{h}}_{i+1}$',
                  LIGHT_BLUE, BLUE, fontsize=9)
        # Backward hidden state
        _draw_box(ax, px, 2.5, 1.1, 0.7, f'$\\overleftarrow{{h}}_{i+1}$',
                  LIGHT_RED, RED, fontsize=9)
        # Concatenated output
        _draw_box(ax, px, 4.0, 1.1, 0.7, f'$h_{i+1}$', '#E8E8E8', GRAY, fontsize=9)

        _arrow(ax, px, -0.17, px, 0.63, color=GRAY, lw=0.8)
        _arrow(ax, px, -0.17, px, 2.13, color=GRAY, lw=0.8,
               connectionstyle='arc3,rad=0.4')

        # Up arrows to concat
        _arrow(ax, px - 0.15, 1.38, px - 0.15, 3.63, color=BLUE, lw=0.8)
        _arrow(ax, px + 0.15, 2.88, px + 0.15, 3.63, color=RED, lw=0.8)

        # Forward connections
        if i < n - 1:
            npx = (i + 1) * 2.2 + 1.0
            _arrow(ax, px + 0.58, 1.0, npx - 0.58, 1.0, color=BLUE, lw=1.8)
            _arrow(ax, npx - 0.58, 2.5, px + 0.58, 2.5, color=RED, lw=1.8)

    # Labels
    ax.text(-0.5, 1.0, 'Forward', ha='right', va='center', fontsize=9,
            color=BLUE, fontweight='bold')
    ax.text(-0.5, 2.5, 'Backward', ha='right', va='center', fontsize=9,
            color=RED, fontweight='bold')
    ax.text(-0.5, 4.0, 'Concat', ha='right', va='center', fontsize=9,
            color=GRAY, fontweight='bold')

    ax.text(5.5, -1.2, '$h_i = [\\overrightarrow{h}_i ; \\overleftarrow{h}_i]$',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_ORANGE,
                      edgecolor=ORANGE, linewidth=1))

    ax.set_title('Bidirectional RNN', fontsize=13, fontweight='bold', pad=10)
    save(fig, 'bidirectional_rnn.png')


def fig_elmo_architecture():
    """ELMo architecture: multi-layer bidirectional language model."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-2, 13)
    ax.set_ylim(-1.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    n = 4
    tokens = ['The', 'cat', 'sat', 'down']
    layer_labels = ['Char CNN\n(context-free)', 'biLSTM Layer 1\n(syntax)',
                    'biLSTM Layer 2\n(semantics)']
    layer_colors = [('#E8E8E8', GRAY), (LIGHT_BLUE, BLUE), (LIGHT_GREEN, GREEN)]

    for layer in range(3):
        y = layer * 2.2 + 0.5
        fc, ec = layer_colors[layer]
        ax.text(-1.5, y, layer_labels[layer], ha='right', va='center',
                fontsize=8, color=ec, fontweight='bold')
        for i in range(n):
            px = i * 2.8 + 1.5
            label = tokens[i] if layer == 0 else f'$h_{i+1}^{{({layer})}}$'
            fs = 9 if layer == 0 else 9
            _draw_box(ax, px, y, 1.3, 0.8, label, fc, ec, fontsize=fs)

            if layer > 0:
                # Vertical from previous layer
                _arrow(ax, px, y - 2.2 + 0.43, px, y - 0.43, color=ec, lw=1)
                # Horizontal (forward)
                if i < n - 1:
                    npx = (i + 1) * 2.8 + 1.5
                    _arrow(ax, px + 0.68, y - 0.1, npx - 0.68, y - 0.1,
                           color=BLUE, lw=1.3)
                    _arrow(ax, npx - 0.68, y + 0.1, px + 0.68, y + 0.1,
                           color=RED, lw=1.3)

    # ELMo output (weighted sum)
    elmo_y = 6.8
    ax.text(5.7, elmo_y, 'ELMo$_i = \\gamma \\sum_{\\ell=0}^{L} s_\\ell \\, h_i^{(\\ell)}$',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_PURPLE,
                      edgecolor=PURPLE, linewidth=1.5))

    # Arrows from each layer to ELMo
    for layer in range(3):
        y = layer * 2.2 + 0.5
        fc, ec = layer_colors[layer]
        ax.annotate('', xy=(5.7, elmo_y - 0.5), xytext=(7.5 + layer * 0.3, y + 0.43),
                    arrowprops=dict(arrowstyle='->', color=ec, lw=1,
                                    connectionstyle=f'arc3,rad={-0.1 * (layer + 1)}',
                                    alpha=0.5))
        # Weight labels
        ax.text(10.5, y, f'$s_{layer}$', ha='center', va='center', fontsize=10,
                fontweight='bold', color=ec)

    ax.set_title('ELMo: Embeddings from Language Models', fontsize=13,
                 fontweight='bold', pad=10)
    save(fig, 'elmo_architecture.png')


def fig_rnn_configurations():
    """Show common RNN input/output configurations."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    configs = [
        ('One-to-Many', 'Image captioning', 1, 4, True),
        ('Many-to-One', 'Classification', 4, 1, True),
        ('Many-to-Many\n(aligned)', 'POS tagging', 4, 4, True),
        ('Many-to-Many\n(unaligned)', 'Translation', 4, 3, False),
    ]

    for ax, (title, subtitle, n_in, n_out, aligned) in zip(axes, configs):
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-1, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
        ax.text(3.0, -0.7, subtitle, ha='center', va='center', fontsize=7,
                color=GRAY, fontstyle='italic')

        if aligned:
            n = max(n_in, n_out)
            for i in range(n):
                px = i * 1.3 + 0.5
                # Hidden
                rect = FancyBboxPatch((px - 0.35, 1.0), 0.7, 0.5,
                                      boxstyle="round,pad=0.03",
                                      facecolor=LIGHT_BLUE, edgecolor=DARK_BLUE,
                                      linewidth=1)
                ax.add_patch(rect)
                # Input
                if i < n_in:
                    rect = FancyBboxPatch((px - 0.3, 0.0), 0.6, 0.4,
                                          boxstyle="round,pad=0.02",
                                          facecolor='#E8E8E8', edgecolor=GRAY,
                                          linewidth=0.8)
                    ax.add_patch(rect)
                    _arrow(ax, px, 0.43, px, 0.97, color=GRAY, lw=0.8)
                # Output
                if (aligned and i < n_out) or (not aligned):
                    if n_out == 1 and i < n - 1:
                        continue
                    rect = FancyBboxPatch((px - 0.3, 2.0), 0.6, 0.4,
                                          boxstyle="round,pad=0.02",
                                          facecolor=LIGHT_RED, edgecolor=RED,
                                          linewidth=0.8)
                    ax.add_patch(rect)
                    _arrow(ax, px, 1.53, px, 1.97, color=RED, lw=0.8)
                # Recurrent
                if i < n - 1:
                    npx = (i + 1) * 1.3 + 0.5
                    _arrow(ax, px + 0.38, 1.25, npx - 0.38, 1.25,
                           color=DARK_BLUE, lw=1)
        else:
            # Encoder-decoder style
            # Encoder
            for i in range(3):
                px = i * 1.0 + 0.3
                rect = FancyBboxPatch((px - 0.3, 1.0), 0.6, 0.5,
                                      boxstyle="round,pad=0.02",
                                      facecolor=LIGHT_BLUE, edgecolor=DARK_BLUE,
                                      linewidth=0.8)
                ax.add_patch(rect)
                rect = FancyBboxPatch((px - 0.25, 0.0), 0.5, 0.4,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#E8E8E8', edgecolor=GRAY,
                                      linewidth=0.8)
                ax.add_patch(rect)
                _arrow(ax, px, 0.43, px, 0.97, color=GRAY, lw=0.7)
                if i < 2:
                    _arrow(ax, px + 0.33, 1.25, px + 1.0 - 0.33, 1.25,
                           color=DARK_BLUE, lw=0.8)
            # Decoder
            for i in range(3):
                px = 3.5 + i * 1.0
                rect = FancyBboxPatch((px - 0.3, 1.0), 0.6, 0.5,
                                      boxstyle="round,pad=0.02",
                                      facecolor=LIGHT_RED, edgecolor=RED,
                                      linewidth=0.8)
                ax.add_patch(rect)
                rect = FancyBboxPatch((px - 0.25, 2.0), 0.5, 0.4,
                                      boxstyle="round,pad=0.02",
                                      facecolor=LIGHT_RED, edgecolor=RED,
                                      linewidth=0.8)
                ax.add_patch(rect)
                _arrow(ax, px, 1.53, px, 1.97, color=RED, lw=0.7)
                if i < 2:
                    _arrow(ax, px + 0.33, 1.25, px + 1.0 - 0.33, 1.25,
                           color=RED, lw=0.8)
            # Connect encoder to decoder
            _arrow(ax, 2.63, 1.25, 3.17, 1.25, color=ORANGE, lw=1.2)

    fig.tight_layout()
    save(fig, 'rnn_configurations.png')


def fig_teacher_forcing():
    """Illustrate teacher forcing vs autoregressive decoding."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, (ax, title, use_teacher) in enumerate(zip(
        axes,
        ['Teacher Forcing (Training)', 'Autoregressive (Inference)'],
        [True, False]
    )):
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-1, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        tgt_in = ['<sos>', 'I', 'am', 'a']
        outputs = ['I', 'was', 'a', 'cat']

        for i in range(4):
            px = i * 2.0 + 0.5
            # Hidden
            _draw_box(ax, px, 1.0, 1.0, 0.7, f'$s_{i+1}$', LIGHT_RED, RED, fontsize=9)
            # Output
            _draw_box(ax, px, 2.5, 0.9, 0.6, outputs[i], '#E8E8E8', GRAY,
                      fontsize=9, fontweight='normal')
            _arrow(ax, px, 1.38, px, 2.17, color=RED, lw=0.8)

            # Input
            if use_teacher:
                # Always use ground truth
                _draw_box(ax, px, -0.3, 0.9, 0.6, tgt_in[i], LIGHT_GREEN, GREEN,
                          fontsize=9)
                _arrow(ax, px, 0.03, px, 0.63, color=GREEN)
            else:
                if i == 0:
                    _draw_box(ax, px, -0.3, 0.9, 0.6, '<sos>', '#E8E8E8', GRAY,
                              fontsize=9)
                    _arrow(ax, px, 0.03, px, 0.63, color=GRAY)
                else:
                    _draw_box(ax, px, -0.3, 0.9, 0.6, outputs[i - 1],
                              LIGHT_ORANGE, ORANGE, fontsize=9)
                    # Curved arrow from previous output
                    ax.annotate('', xy=(px, -0.03), xytext=(px - 2.0, 2.17),
                                arrowprops=dict(arrowstyle='->', color=ORANGE,
                                                lw=1, connectionstyle='arc3,rad=0.4'))

            # Recurrent
            if i < 3:
                npx = (i + 1) * 2.0 + 0.5
                _arrow(ax, px + 0.53, 1.0, npx - 0.53, 1.0, color=RED, lw=1.5)

        if use_teacher:
            ax.text(4.0, -0.9, 'Ground-truth tokens fed as input',
                    ha='center', va='center', fontsize=8, color=GREEN, fontstyle='italic')
        else:
            ax.text(4.0, -0.9, "Model's own predictions fed as input",
                    ha='center', va='center', fontsize=8, color=ORANGE, fontstyle='italic')

    fig.tight_layout()
    save(fig, 'teacher_forcing.png')


if __name__ == '__main__':
    print('Generating figures for Recurrent Neural Networks for NLP...')
    fig_rnn_unrolled()
    fig_vanishing_gradient()
    fig_lstm_cell()
    fig_gru_cell()
    fig_encoder_decoder()
    fig_attention_mechanism()
    fig_bidirectional_rnn()
    fig_elmo_architecture()
    fig_rnn_configurations()
    fig_teacher_forcing()
    print('Done!')
