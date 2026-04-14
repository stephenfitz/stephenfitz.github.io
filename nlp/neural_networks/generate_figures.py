#!/usr/bin/env python3
"""Generate figures for the Neural Networks article."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patches as mpatches

FIGDIR = 'figures'
DPI = 200

# Consistent color palette
BLUE = '#4A7FB5'
DARK_BLUE = '#2C5F8A'
RED = '#C0504D'
DARK_RED = '#8B3A3A'
GRAY = '#888888'
LIGHT_GRAY = '#CCCCCC'
GREEN = '#5A9E6F'
ORANGE = '#E8923F'

# ============================================================================
# Figure 1: Activation functions comparison
# ============================================================================
def plot_activation_functions():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    x = np.linspace(-5, 5, 500)

    # Sigmoid
    ax = axes[0, 0]
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=BLUE, linewidth=2.2)
    ax.set_title('Sigmoid', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Tanh
    ax = axes[0, 1]
    y = np.tanh(x)
    ax.plot(x, y, color=BLUE, linewidth=2.2)
    ax.set_title('Tanh', fontsize=13, fontweight='bold')
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # ReLU
    ax = axes[0, 2]
    y = np.maximum(0, x)
    ax.plot(x, y, color=RED, linewidth=2.2)
    ax.set_title('ReLU', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.5, 5.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Leaky ReLU
    ax = axes[1, 0]
    y = np.where(x > 0, x, 0.1 * x)
    ax.plot(x, y, color=RED, linewidth=2.2)
    ax.set_title('Leaky ReLU ($\\alpha=0.1$)', fontsize=13, fontweight='bold')
    ax.set_ylim(-1.0, 5.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # GELU
    ax = axes[1, 1]
    from scipy.special import erf
    y = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    ax.plot(x, y, color=GREEN, linewidth=2.2)
    ax.set_title('GELU', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.5, 5.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Swish / SiLU
    ax = axes[1, 2]
    y = x / (1 + np.exp(-x))
    ax.plot(x, y, color=GREEN, linewidth=2.2)
    ax.set_title('Swish (SiLU)', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.5, 5.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    for ax in axes.flat:
        ax.set_xlabel('$z$', fontsize=11)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/activation_functions.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved activation_functions.png')


# ============================================================================
# Figure 2: Network architecture diagram (feedforward)
# ============================================================================
def plot_network_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_sizes = [3, 5, 4, 2]
    layer_labels = ['Input\nlayer', 'Hidden\nlayer 1', 'Hidden\nlayer 2', 'Output\nlayer']
    layer_x = [1.0, 3.0, 5.0, 7.0]
    node_colors = [LIGHT_GRAY, BLUE, BLUE, RED]
    node_radius = 0.28

    # Store node positions
    positions = {}
    for l, (n, x_pos) in enumerate(zip(layer_sizes, layer_x)):
        y_positions = np.linspace(-(n - 1) * 0.8, (n - 1) * 0.8, n)
        for i, y in enumerate(y_positions):
            positions[(l, i)] = (x_pos, y)

    # Draw edges
    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                x1, y1 = positions[(l, i)]
                x2, y2 = positions[(l + 1, j)]
                ax.plot([x1, x2], [y1, y2], color=GRAY, linewidth=0.5, alpha=0.4, zorder=1)

    # Draw nodes
    for l, (n, x_pos) in enumerate(zip(layer_sizes, layer_x)):
        for i in range(n):
            x, y = positions[(l, i)]
            circle = Circle((x, y), node_radius, facecolor=node_colors[l],
                           edgecolor='#444444', linewidth=1.5, zorder=3)
            ax.add_patch(circle)

            # Node labels
            if l == 0:
                ax.text(x, y, f'$x_{i+1}$', ha='center', va='center',
                       fontsize=11, zorder=4)
            elif l == len(layer_sizes) - 1:
                ax.text(x, y, f'$\\hat{{y}}_{i+1}$', ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold', zorder=4)

    # Layer labels
    for l, (label, x_pos) in enumerate(zip(layer_labels, layer_x)):
        top_y = max(positions[(l, i)][1] for i in range(layer_sizes[l]))
        ax.text(x_pos, top_y + 0.7, label, ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='#333333')

    # Weight matrix labels
    for l in range(len(layer_sizes) - 1):
        mid_x = (layer_x[l] + layer_x[l + 1]) / 2
        ax.text(mid_x, -3.0, f'$W^{{({l+1})}}$', ha='center', va='center',
               fontsize=14, color=DARK_BLUE, fontweight='bold')

    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-3.8, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/network_architecture.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved network_architecture.png')


# ============================================================================
# Figure 3: Universal approximation — step function construction
# ============================================================================
def plot_universal_approximation():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = np.linspace(-1, 6, 1000)

    # Panel 1: Two sigmoids making a bump
    ax = axes[0]
    s1 = 1 / (1 + np.exp(-10 * (x - 1)))
    s2 = 1 / (1 + np.exp(-10 * (x - 3)))
    ax.plot(x, s1, '--', color=BLUE, linewidth=1.5, alpha=0.6, label='$\\sigma(w(x - a))$')
    ax.plot(x, s2, '--', color=RED, linewidth=1.5, alpha=0.6, label='$\\sigma(w(x - b))$')
    bump = s1 - s2
    ax.plot(x, bump, color='black', linewidth=2.2, label='Difference')
    ax.fill_between(x, 0, bump, alpha=0.15, color=BLUE)
    ax.set_title('(a) Subtracting two sigmoids\nmakes a bump', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(-0.15, 1.25)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Panel 2: Multiple bumps
    ax = axes[1]
    centers = [0.5, 1.5, 2.5, 3.5, 4.5]
    heights = [0.3, 0.8, 1.0, 0.6, 0.2]
    total = np.zeros_like(x)
    for c, h in zip(centers, heights):
        s_left = 1 / (1 + np.exp(-20 * (x - (c - 0.4))))
        s_right = 1 / (1 + np.exp(-20 * (x - (c + 0.4))))
        bump_i = h * (s_left - s_right)
        ax.fill_between(x, 0, bump_i, alpha=0.2, color=BLUE)
        total += bump_i
    ax.plot(x, total, color='black', linewidth=2.2)
    ax.set_title('(b) Weighted sum of bumps\napproximates a function', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.15, 1.25)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Panel 3: Approximation of a target function
    ax = axes[2]
    target = 0.5 + 0.3 * np.sin(2 * x) + 0.15 * np.cos(3.5 * x)
    target = np.clip(target, 0, 1)

    # Build approximation from many bumps
    n_bumps = 20
    bump_centers = np.linspace(0, 5, n_bumps)
    approx = np.zeros_like(x)
    for c in bump_centers:
        width = 0.2
        s_left = 1 / (1 + np.exp(-30 * (x - (c - width))))
        s_right = 1 / (1 + np.exp(-30 * (x - (c + width))))
        h = 0.5 + 0.3 * np.sin(2 * c) + 0.15 * np.cos(3.5 * c)
        h = max(0, min(1, h))
        approx += h * (s_left - s_right)

    ax.plot(x, target, color=GRAY, linewidth=2, linestyle='--', label='Target $f(x)$')
    ax.plot(x, approx, color=BLUE, linewidth=2.2, label='Network approximation')
    ax.set_title('(c) More bumps yield\nbetter approximation', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(-0.15, 1.25)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    for ax in axes:
        ax.set_xlabel('$x$', fontsize=11)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/universal_approximation.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved universal_approximation.png')


# ============================================================================
# Figure 4: Computational graph for backpropagation
# ============================================================================
def plot_computation_graph():
    fig, ax = plt.subplots(figsize=(12, 5))

    # Nodes in the computation graph
    nodes = {
        'x': (0.5, 2.5),
        'W1': (0.5, 4.0),
        'z1': (2.5, 3.0),
        'h': (4.5, 3.0),
        'W2': (4.5, 4.5),
        'z2': (6.5, 3.0),
        'yhat': (8.5, 3.0),
        't': (8.5, 1.5),
        'L': (10.5, 2.5),
    }

    node_labels = {
        'x': '$x$',
        'W1': '$W^{(1)}$',
        'z1': '$z^{(1)} = W^{(1)}x$',
        'h': '$h = \\sigma(z^{(1)})$',
        'W2': '$W^{(2)}$',
        'z2': '$z^{(2)} = W^{(2)}h$',
        'yhat': '$\\hat{y} = \\sigma(z^{(2)})$',
        't': '$t$',
        'L': '$\\mathcal{L}$',
    }

    # Forward edges
    forward_edges = [
        ('x', 'z1'), ('W1', 'z1'), ('z1', 'h'), ('h', 'z2'),
        ('W2', 'z2'), ('z2', 'yhat'), ('yhat', 'L'), ('t', 'L'),
    ]

    # Draw forward edges
    for (src, dst) in forward_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8,
                                   connectionstyle='arc3,rad=0.0'))

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name in ('L',):
            color = RED
            text_color = 'white'
        elif name in ('x', 't', 'W1', 'W2'):
            color = LIGHT_GRAY
            text_color = 'black'
        else:
            color = '#D6E8F7'
            text_color = 'black'

        bbox = dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='#444444', linewidth=1.5)
        ax.text(x, y, node_labels[name], ha='center', va='center',
               fontsize=10, color=text_color, bbox=bbox, zorder=5)

    # Backward pass annotation
    ax.annotate('', xy=(6.8, 1.5), xytext=(10.2, 1.5),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.0, linestyle='--'))
    ax.text(8.5, 1.0, 'Backward pass (backpropagation)', ha='center', fontsize=10,
           color=RED, style='italic')

    # Forward pass annotation
    ax.text(5.5, 4.8, 'Forward pass', ha='center', fontsize=10,
           color=BLUE, style='italic')
    ax.annotate('', xy=(7.5, 4.8), xytext=(3.5, 4.8),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.0, linestyle='--'))

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.3, 5.3)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/computation_graph.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved computation_graph.png')


# ============================================================================
# Figure 5: Loss surface and gradient descent
# ============================================================================
def plot_loss_landscape():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Convex loss (single neuron / logistic regression)
    ax = axes[0]
    w1 = np.linspace(-3, 3, 200)
    w2 = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1, w2)
    Z = 0.5 * W1**2 + 0.8 * W2**2 + 0.3 * W1 * W2
    ax.contour(W1, W2, Z, levels=15, colors=BLUE, linewidths=0.8, alpha=0.6)
    ax.contourf(W1, W2, Z, levels=15, cmap='Blues', alpha=0.3)

    # GD trajectory
    path_x = [2.5, 1.8, 1.2, 0.7, 0.3, 0.1, 0.02]
    path_y = [2.0, 1.3, 0.7, 0.3, 0.1, 0.02, -0.01]
    ax.plot(path_x, path_y, 'o-', color=RED, markersize=5, linewidth=1.8, zorder=5)
    ax.plot(path_x[0], path_y[0], 'o', color=RED, markersize=8, zorder=6)
    ax.plot(0, 0, '*', color=DARK_RED, markersize=14, zorder=6)
    ax.set_title('Convex loss (single neuron)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$w_1$', fontsize=12)
    ax.set_ylabel('$w_2$', fontsize=12)
    ax.grid(True, alpha=0.2)

    # Panel 2: Non-convex loss (multi-layer network)
    ax = axes[1]
    Z2 = (np.sin(1.5 * W1) * np.cos(1.5 * W2) + 0.15 * W1**2 + 0.15 * W2**2
           + 0.5 * np.sin(W1 + W2))
    ax.contour(W1, W2, Z2, levels=20, colors=BLUE, linewidths=0.8, alpha=0.6)
    ax.contourf(W1, W2, Z2, levels=20, cmap='Blues', alpha=0.3)

    # SGD trajectory (non-smooth)
    np.random.seed(42)
    px, py = [2.5], [2.0]
    for _ in range(12):
        px.append(px[-1] - 0.25 + 0.08 * np.random.randn())
        py.append(py[-1] - 0.22 + 0.08 * np.random.randn())
    ax.plot(px, py, 'o-', color=RED, markersize=4, linewidth=1.5, zorder=5)
    ax.plot(px[0], py[0], 'o', color=RED, markersize=8, zorder=6)
    ax.plot(px[-1], py[-1], '*', color=DARK_RED, markersize=14, zorder=6)
    ax.set_title('Non-convex loss (deep network)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$w_1$', fontsize=12)
    ax.set_ylabel('$w_2$', fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/loss_landscape.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved loss_landscape.png')


# ============================================================================
# Figure 6: Vanishing / exploding gradient illustration
# ============================================================================
def plot_gradient_flow():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    layers = np.arange(1, 9)

    # Vanishing gradients (sigmoid)
    ax = axes[0]
    grad_vanish = 0.25 ** np.arange(8)  # sigmoid derivative max is 0.25
    ax.bar(layers, grad_vanish, color=BLUE, alpha=0.7, edgecolor=DARK_BLUE, linewidth=1)
    ax.set_xlabel('Layer (from output)', fontsize=11)
    ax.set_ylabel('Relative gradient magnitude', fontsize=11)
    ax.set_title('Vanishing gradients (sigmoid)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 2)
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(labelsize=9)

    # Exploding gradients
    ax = axes[1]
    grad_explode = 2.0 ** np.arange(8)
    ax.bar(layers, grad_explode, color=RED, alpha=0.7, edgecolor=DARK_RED, linewidth=1)
    ax.set_xlabel('Layer (from output)', fontsize=11)
    ax.set_ylabel('Relative gradient magnitude', fontsize=11)
    ax.set_title('Exploding gradients', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/gradient_flow.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved gradient_flow.png')


# ============================================================================
# Figure 7: XOR problem solved by a two-layer network
# ============================================================================
def plot_xor_solution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: XOR is not linearly separable
    ax = axes[0]
    ax.plot([0, 1], [1, 0], 'o', color=BLUE, markersize=14, zorder=5, label='Class +1')
    ax.plot([0, 1], [0, 1], 's', color=RED, markersize=14, zorder=5, label='Class -1')
    ax.set_title('(a) XOR: not linearly\nseparable', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    # Panel 2: Hidden layer transforms the space
    ax = axes[1]
    # After hidden layer with appropriate weights, XOR points get mapped to linearly separable positions
    # h1 = sigma(x1 + x2 - 0.5), h2 = sigma(x1 + x2 - 1.5)
    # (0,0) -> (sigma(-0.5), sigma(-1.5)) ~ (0.38, 0.18)
    # (0,1) -> (sigma(0.5), sigma(-0.5)) ~ (0.62, 0.38)
    # (1,0) -> (sigma(0.5), sigma(-0.5)) ~ (0.62, 0.38)
    # (1,1) -> (sigma(1.5), sigma(0.5)) ~ (0.82, 0.62)
    def sig(z): return 1/(1+np.exp(-5*z))
    h_points = {
        (0,0): (sig(-0.5), sig(-1.5)),
        (0,1): (sig(0.5), sig(-0.5)),
        (1,0): (sig(0.5+0.1), sig(-0.5+0.1)),  # slight offset for visibility
        (1,1): (sig(1.5), sig(0.5)),
    }
    labels = {(0,0): -1, (0,1): +1, (1,0): +1, (1,1): -1}
    for (x1,x2), (h1,h2) in h_points.items():
        if labels[(x1,x2)] == 1:
            ax.plot(h1, h2, 'o', color=BLUE, markersize=14, zorder=5)
        else:
            ax.plot(h1, h2, 's', color=RED, markersize=14, zorder=5)
        ax.annotate(f'({x1},{x2})', (h1, h2), textcoords='offset points',
                   xytext=(8, 8), fontsize=9)

    # Draw separating line in h-space: h2 = h1 - 0.5
    h1_line = np.linspace(0.4, 1.1, 100)
    h2_line = h1_line - 0.5
    ax.plot(h1_line, h2_line, '--', color='black', linewidth=1.5)
    ax.set_title('(b) Hidden layer remaps\nto separable space', fontsize=11, fontweight='bold')
    ax.set_xlabel('$h_1$', fontsize=12)
    ax.set_ylabel('$h_2$', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.1, 1.05)

    # Panel 3: Decision boundary in original space
    ax = axes[2]
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
    # Two-layer network that solves XOR
    W1 = np.array([[20, 20], [20, 20]])
    b1 = np.array([-10, -30])
    W2 = np.array([1, -1])
    b2 = -0.5
    Z1 = np.c_[xx.ravel(), yy.ravel()] @ W1.T + b1
    H = 1 / (1 + np.exp(-Z1))
    Z2 = H @ W2 + b2
    out = 1 / (1 + np.exp(-Z2))
    out = out.reshape(xx.shape)

    ax.contourf(xx, yy, out, levels=[0, 0.5, 1], colors=[('#FFCCCC', 0.5), ('#CCE0FF', 0.5)], alpha=0.5)
    ax.contour(xx, yy, out, levels=[0.5], colors=['black'], linewidths=2)
    ax.plot([0, 1], [1, 0], 'o', color=BLUE, markersize=14, zorder=5)
    ax.plot([0, 1], [0, 1], 's', color=RED, markersize=14, zorder=5)
    ax.set_title('(c) Non-linear decision\nboundary', fontsize=11, fontweight='bold')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/xor_solution.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved xor_solution.png')


# ============================================================================
# Figure 8: SGD vs Mini-batch vs Batch gradient descent
# ============================================================================
def plot_sgd_variants():
    fig, ax = plt.subplots(figsize=(8, 5))

    np.random.seed(123)
    epochs = np.arange(0, 100)

    # Batch GD: smooth convergence
    batch_loss = 2.5 * np.exp(-0.04 * epochs) + 0.15
    ax.plot(epochs, batch_loss, color=BLUE, linewidth=2.2, label='Batch gradient descent')

    # Mini-batch: noisy but converging
    mini_loss = 2.5 * np.exp(-0.05 * epochs) + 0.15 + 0.12 * np.random.randn(100) * np.exp(-0.02 * epochs)
    ax.plot(epochs, mini_loss, color=GREEN, linewidth=1.5, alpha=0.8, label='Mini-batch SGD')

    # SGD: very noisy
    sgd_loss = 2.5 * np.exp(-0.055 * epochs) + 0.15 + 0.3 * np.random.randn(100) * np.exp(-0.015 * epochs)
    ax.plot(epochs, sgd_loss, color=RED, linewidth=1.0, alpha=0.6, label='Stochastic GD (single sample)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Convergence behavior of gradient descent variants', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 3.0)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/sgd_variants.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved sgd_variants.png')


# ============================================================================
# Figure 9: Overfitting — train vs validation loss
# ============================================================================
def plot_overfitting():
    fig, ax = plt.subplots(figsize=(7, 5))

    epochs = np.arange(0, 200)

    train_loss = 2.0 * np.exp(-0.04 * epochs) + 0.03
    val_loss = 2.0 * np.exp(-0.06 * epochs) + 0.25 + 0.003 * epochs

    ax.plot(epochs, train_loss, color=BLUE, linewidth=2.2, label='Training loss')
    ax.plot(epochs, val_loss, color=RED, linewidth=2.2, label='Validation loss')

    # Mark the optimal point (minimum of validation loss)
    best_epoch = int(np.argmin(val_loss))
    ax.axvline(best_epoch, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(best_epoch + 3, 1.6, 'Early\nstopping', fontsize=10, color=GRAY, fontweight='bold')

    # Regions
    label_y = 0.08
    arrow_y = 0.02
    ax.annotate('', xy=(best_epoch - 3, arrow_y), xytext=(3, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
    ax.text(best_epoch / 2, label_y, 'Underfitting', ha='center', fontsize=10, color=GREEN)

    ax.annotate('', xy=(197, arrow_y), xytext=(best_epoch + 3, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.5))
    ax.text((best_epoch + 197) / 2, label_y, 'Overfitting', ha='center', fontsize=10, color=RED)

    ax.set_ylim(bottom=-0.05)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs. validation loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/overfitting.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved overfitting.png')


# ============================================================================
# Figure 10: Softmax output layer illustration
# ============================================================================
def plot_softmax_output():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Show softmax converting logits to probabilities
    logits = np.array([2.0, 1.0, 0.5, -0.5, -1.0])
    probs = np.exp(logits) / np.exp(logits).sum()
    classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

    bars1 = ax.barh(np.arange(5) + 0.2, logits, height=0.35, color=BLUE, alpha=0.7,
                    edgecolor=DARK_BLUE, linewidth=1, label='Logits $z_k$')
    bars2 = ax.barh(np.arange(5) - 0.2, probs, height=0.35, color=RED, alpha=0.7,
                    edgecolor=DARK_RED, linewidth=1, label='Softmax $\\hat{y}_k$')

    # Annotate values
    for i, (l, p) in enumerate(zip(logits, probs)):
        ax.text(max(logits) + 0.3, i + 0.2, f'{l:.1f}', va='center', fontsize=10, color=DARK_BLUE)
        ax.text(max(logits) + 0.3, i - 0.2, f'{p:.3f}', va='center', fontsize=10, color=DARK_RED)

    ax.set_yticks(range(5))
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Softmax: from logits to probabilities', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/softmax_output.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved softmax_output.png')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    plot_activation_functions()
    plot_network_architecture()
    plot_universal_approximation()
    plot_computation_graph()
    plot_loss_landscape()
    plot_gradient_flow()
    plot_xor_solution()
    plot_sgd_variants()
    plot_overfitting()
    plot_softmax_output()
    print('\nAll figures generated.')
