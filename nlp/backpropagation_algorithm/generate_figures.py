#!/usr/bin/env python3
"""Generate figures for the Backpropagation Algorithm article."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

FIGDIR = 'figures'
DPI = 200

# Color palette
BLUE = '#4A7FB5'
DARK_BLUE = '#2C5F8A'
RED = '#C0504D'
DARK_RED = '#8B3A3A'
GRAY = '#888888'
LIGHT_GRAY = '#CCCCCC'
GREEN = '#5A9E6F'
ORANGE = '#E8923F'
LIGHT_BLUE = '#D6E8F7'
LIGHT_RED = '#F5D5D5'
LIGHT_GREEN = '#D5EBD5'


# ============================================================================
# Figure 1: Single-node backprop rule
# ============================================================================
def plot_local_gradient_rule():
    fig, ax = plt.subplots(figsize=(10, 4))

    # Function node
    cx, cy = 5, 2
    node_w, node_h = 1.4, 1.0
    rect = FancyBboxPatch((cx - node_w/2, cy - node_h/2), node_w, node_h,
                           boxstyle="round,pad=0.1", facecolor=LIGHT_BLUE,
                           edgecolor=DARK_BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.1, '$f$', ha='center', va='center', fontsize=22, fontweight='bold',
           color=DARK_BLUE)
    ax.text(cx, cy - 0.25, r'$\frac{\partial h}{\partial z}$', ha='center', va='center',
           fontsize=12, color=GRAY, style='italic')

    # Input arrow (z)
    ax.annotate('', xy=(cx - node_w/2, cy), xytext=(1.0, cy),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(2.5, cy + 0.35, '$z$', ha='center', fontsize=16, fontweight='bold')

    # Output arrow (h)
    ax.annotate('', xy=(9.0, cy), xytext=(cx + node_w/2, cy),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(7.5, cy + 0.35, '$h$', ha='center', fontsize=16, fontweight='bold')

    # Upstream gradient (red, right to left, above)
    ax.annotate('', xy=(cx + node_w/2 + 0.1, cy + 0.7), xytext=(9.0, cy + 0.7),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(7.5, cy + 1.05, r'$\frac{\partial \mathcal{L}}{\partial h}$',
           ha='center', fontsize=14, color=RED)
    ax.text(8.5, cy + 1.45, 'upstream\ngradient', ha='center', fontsize=9,
           color=RED, style='italic')

    # Downstream gradient (red, right to left, above)
    ax.annotate('', xy=(1.0, cy + 0.7), xytext=(cx - node_w/2 - 0.1, cy + 0.7),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(2.5, cy + 1.05, r'$\frac{\partial \mathcal{L}}{\partial z}$',
           ha='center', fontsize=14, color=RED)
    ax.text(1.8, cy + 1.45, 'downstream\ngradient', ha='center', fontsize=9,
           color=RED, style='italic')

    # Local gradient label
    ax.text(cx, cy - 1.0, 'local gradient', ha='center', fontsize=10,
           color=GRAY, style='italic')

    # Equation below
    ax.text(5, 0.15,
           r'$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial z}$',
           ha='center', va='center', fontsize=18,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor=GRAY))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/local_gradient_rule.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved local_gradient_rule.png')


# ============================================================================
# Figure 2: Multi-input node backprop rule
# ============================================================================
def plot_multi_input_rule():
    fig, ax = plt.subplots(figsize=(10, 5))

    cx, cy = 5, 2.5
    node_w, node_h = 1.4, 1.2
    rect = FancyBboxPatch((cx - node_w/2, cy - node_h/2), node_w, node_h,
                           boxstyle="round,pad=0.1", facecolor=LIGHT_BLUE,
                           edgecolor=DARK_BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.15, '$f$', ha='center', va='center', fontsize=22, fontweight='bold',
           color=DARK_BLUE)
    ax.text(cx, cy - 0.25, r'$\frac{\partial z}{\partial x},\;\frac{\partial z}{\partial y}$',
           ha='center', va='center', fontsize=11, color=GRAY, style='italic')

    # Input x (top)
    ax.annotate('', xy=(cx - node_w/2, cy + 0.3), xytext=(1.0, cy + 0.3 + 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(1.8, cy + 1.5, '$x$', ha='center', fontsize=16, fontweight='bold')

    # Input y (bottom)
    ax.annotate('', xy=(cx - node_w/2, cy - 0.3), xytext=(1.0, cy - 0.3 - 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(1.8, cy - 1.5, '$y$', ha='center', fontsize=16, fontweight='bold')

    # Output z
    ax.annotate('', xy=(9.0, cy), xytext=(cx + node_w/2, cy),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(7.5, cy + 0.35, '$z$', ha='center', fontsize=16, fontweight='bold')

    # Upstream gradient
    ax.annotate('', xy=(cx + node_w/2 + 0.1, cy + 0.7), xytext=(9.0, cy + 0.7),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(7.8, cy + 1.05, r'$\frac{\partial \mathcal{L}}{\partial z}$',
           ha='center', fontsize=14, color=RED)

    # Downstream gradient to x
    ax.annotate('', xy=(0.5, cy + 1.7), xytext=(cx - node_w/2 - 0.1, cy + 0.6),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(1.5, cy + 1.9, r'$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial x}$',
           ha='center', fontsize=12, color=RED)

    # Downstream gradient to y
    ax.annotate('', xy=(0.5, cy - 1.7), xytext=(cx - node_w/2 - 0.1, cy - 0.6),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(1.5, cy - 1.9, r'$\frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial y}$',
           ha='center', fontsize=12, color=RED)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/multi_input_rule.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved multi_input_rule.png')


# ============================================================================
# Figure 3: ReLU worked example — computation graph with forward values
# ============================================================================
def plot_relu_forward():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Node positions
    nodes = {
        'w1': (0.5, 4.5), 'x1': (0.5, 3.0),
        'mul1': (2.5, 3.75),
        'w2': (0.5, 1.5), 'x2': (0.5, 0.0),
        'mul2': (2.5, 0.75),
        'add': (5.0, 2.25),
        'relu': (7.5, 2.25),
        't': (9.5, 0.5),
        'loss': (10.0, 2.25),
    }

    # Draw edges with values
    edges = [
        ('w1', 'mul1', '$w_1 = 1$', 'above'),
        ('x1', 'mul1', '$x_1 = -1$', 'below'),
        ('mul1', 'add', '$-1$', 'above'),
        ('w2', 'mul2', '$w_2 = 2$', 'above'),
        ('x2', 'mul2', '$x_2 = 3$', 'below'),
        ('mul2', 'add', '$6$', 'below'),
        ('add', 'relu', '$z = 5$', 'above'),
        ('relu', 'loss', '$y = 5$', 'above'),
        ('t', 'loss', '$t = 2$', 'right'),
    ]

    for src, dst, label, pos in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.25 if pos == 'above' else -0.25
        if pos == 'right':
            ax.text(x2 + 0.3, (y1 + y2)/2, label, fontsize=10, color=DARK_BLUE,
                   va='center')
        else:
            ax.text(mx, my + offset, label, fontsize=10, color=DARK_BLUE,
                   ha='center', va='center')

    # Draw operation nodes
    op_nodes = {
        'mul1': ('$\\times$', LIGHT_BLUE),
        'mul2': ('$\\times$', LIGHT_BLUE),
        'add': ('$+$', LIGHT_GREEN),
        'relu': ('ReLU', '#FFF3CC'),
        'loss': ('$\\mathcal{L}$', LIGHT_RED),
    }
    for name, (label, color) in op_nodes.items():
        x, y = nodes[name]
        circle = Circle((x, y), 0.4, facecolor=color, edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12 if len(label) < 5 else 9,
               fontweight='bold', zorder=6)

    # Draw input nodes (rectangles)
    input_nodes = {'w1': '$w_1$', 'x1': '$x_1$', 'w2': '$w_2$', 'x2': '$x_2$', 't': '$t$'}
    for name, label in input_nodes.items():
        x, y = nodes[name]
        rect = FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5,
                               boxstyle="round,pad=0.05", facecolor=LIGHT_GRAY,
                               edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Loss value
    ax.text(10.0, 3.2, '$\\mathcal{L} = \\frac{1}{2}(5-2)^2 = 4.5$',
           ha='center', fontsize=13, color=DARK_RED,
           bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_RED, edgecolor=RED))

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Forward pass: computing values left to right', fontsize=14, fontweight='bold',
                pad=10)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/relu_forward.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved relu_forward.png')


# ============================================================================
# Figure 4: ReLU worked example — backward pass with gradients
# ============================================================================
def plot_relu_backward():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Same node positions
    nodes = {
        'w1': (0.5, 4.5), 'x1': (0.5, 3.0),
        'mul1': (2.5, 3.75),
        'w2': (0.5, 1.5), 'x2': (0.5, 0.0),
        'mul2': (2.5, 0.75),
        'add': (5.0, 2.25),
        'relu': (7.5, 2.25),
        't': (9.5, 0.5),
        'loss': (10.0, 2.25),
    }

    # Draw forward edges (faded)
    fwd_edges = [
        ('w1', 'mul1'), ('x1', 'mul1'), ('mul1', 'add'),
        ('w2', 'mul2'), ('x2', 'mul2'), ('mul2', 'add'),
        ('add', 'relu'), ('relu', 'loss'), ('t', 'loss'),
    ]
    for src, dst in fwd_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=LIGHT_GRAY, lw=1.2))

    # Draw backward gradient arrows (red, dashed)
    grad_edges = [
        ('loss', 'relu', '3'),
        ('relu', 'add', '3'),
        ('add', 'mul1', '3'),
        ('add', 'mul2', '3'),
    ]
    for src, dst, label in grad_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
        ax.text(mx, my + 0.35, label, ha='center', fontsize=12, color=RED, fontweight='bold')

    # Gradients to inputs
    grad_to_inputs = [
        ('mul1', 'w1', '$-3$'),
        ('mul1', 'x1', '$3$'),
        ('mul2', 'w2', '$9$'),
        ('mul2', 'x2', '$6$'),
    ]
    for src, dst, label in grad_to_inputs:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
        ax.text(mx + 0.35, my, label, fontsize=12, color=RED, fontweight='bold', va='center')

    # Draw operation nodes
    op_nodes = {
        'mul1': ('$\\times$', LIGHT_BLUE),
        'mul2': ('$\\times$', LIGHT_BLUE),
        'add': ('$+$', LIGHT_GREEN),
        'relu': ('ReLU', '#FFF3CC'),
        'loss': ('$\\mathcal{L}$', LIGHT_RED),
    }
    for name, (label, color) in op_nodes.items():
        x, y = nodes[name]
        circle = Circle((x, y), 0.4, facecolor=color, edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12 if len(label) < 5 else 9,
               fontweight='bold', zorder=6)

    # Draw input nodes
    input_nodes = {'w1': '$w_1$', 'x1': '$x_1$', 'w2': '$w_2$', 'x2': '$x_2$', 't': '$t$'}
    for name, label in input_nodes.items():
        x, y = nodes[name]
        rect = FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5,
                               boxstyle="round,pad=0.05", facecolor=LIGHT_GRAY,
                               edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Result box
    ax.text(5.0, 5.2, r'$\nabla_w \mathcal{L} = [-3,\; 9]^\top$',
           ha='center', fontsize=16, color=DARK_RED,
           bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_RED, edgecolor=RED, linewidth=2))

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 6.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Backward pass: propagating gradients right to left', fontsize=14,
                fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/relu_backward.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved relu_backward.png')


# ============================================================================
# Figure 5: Neuron computation graph decomposition
# ============================================================================
def plot_neuron_decomposition():
    fig, ax = plt.subplots(figsize=(12, 4))

    # Operation chain: x -> [*] -> [+] -> [sigma] -> [L]
    #                       w        b               y (target)
    ops = [
        (2.0, 2.0, '$\\times$', LIGHT_BLUE),
        (4.5, 2.0, '$+$', LIGHT_GREEN),
        (7.0, 2.0, '$\\sigma$', '#FFF3CC'),
        (9.5, 2.0, '$\\mathcal{L}$', LIGHT_RED),
    ]

    # Edges along the chain
    labels_on_edges = ['$wx$', '$z$', '$o$']
    for i in range(len(ops) - 1):
        x1, y1 = ops[i][0], ops[i][1]
        x2, y2 = ops[i+1][0], ops[i+1][1]
        ax.annotate('', xy=(x2 - 0.4, y2), xytext=(x1 + 0.4, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        mx = (x1 + x2) / 2
        ax.text(mx, y1 + 0.35, labels_on_edges[i], ha='center', fontsize=12, color=DARK_BLUE)

    # Draw op nodes
    for x, y, label, color in ops:
        circle = Circle((x, y), 0.4, facecolor=color, edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # Input arrows
    # x -> multiply
    ax.annotate('', xy=(2.0 - 0.4, 2.0 + 0.15), xytext=(0.2, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.2, 3.3, '$x$', ha='center', fontsize=14, fontweight='bold')

    # w -> multiply
    ax.annotate('', xy=(2.0 - 0.4, 2.0 - 0.15), xytext=(0.2, 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.2, 0.7, '$w$', ha='center', fontsize=14, fontweight='bold')

    # b -> add
    ax.annotate('', xy=(4.5, 2.0 - 0.4), xytext=(4.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(4.5, 0.15, '$b$', ha='center', fontsize=14, fontweight='bold')

    # y -> loss
    ax.annotate('', xy=(9.5, 2.0 - 0.4), xytext=(9.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(9.5, 0.15, '$t$', ha='center', fontsize=14, fontweight='bold')

    # Output label
    ax.text(10.5, 2.0, '$\\mathcal{L}$', ha='center', fontsize=14, fontweight='bold',
           color=DARK_RED)
    ax.annotate('', xy=(10.3, 2.0), xytext=(9.5 + 0.4, 2.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Equation
    ax.text(5.5, 3.8, '$o = \\sigma(w \\cdot x + b), \\quad \\mathcal{L} = \\mathcal{L}(o, t)$',
           ha='center', fontsize=14,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor=GRAY))

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/neuron_computation_graph.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved neuron_computation_graph.png')


# ============================================================================
# Figure 6: Multiple paths through a network
# ============================================================================
def plot_multiple_paths():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 2-input, 2-hidden, 2-output network
    positions = {
        'x1': (0.5, 4.0), 'x2': (0.5, 1.5),
        'h1': (4.0, 4.5), 'h2': (4.0, 1.0),
        'y1': (7.5, 4.5), 'y2': (7.5, 1.0),
    }

    # Draw all edges in gray first
    all_edges = [
        ('x1', 'h1'), ('x1', 'h2'), ('x2', 'h1'), ('x2', 'h2'),
        ('h1', 'y1'), ('h1', 'y2'), ('h2', 'y1'), ('h2', 'y2'),
    ]
    for src, dst in all_edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.plot([x1, x2], [y1, y2], color=LIGHT_GRAY, linewidth=1.5, zorder=1)

    # Highlight paths from w_{1,1}^{(1)} (x1 -> h1)
    path_colors = [RED, ORANGE]
    paths = [
        [('x1', 'h1'), ('h1', 'y1')],  # path through y1
        [('x1', 'h1'), ('h1', 'y2')],  # path through y2
    ]
    for path, color in zip(paths, path_colors):
        for src, dst in path:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.7))

    # Draw nodes
    node_props = {
        'x1': ('$x_1$', LIGHT_GRAY), 'x2': ('$x_2$', LIGHT_GRAY),
        'h1': ('$h_1$', LIGHT_BLUE), 'h2': ('$h_2$', LIGHT_BLUE),
        'y1': ('$\\hat{y}_1$', LIGHT_RED), 'y2': ('$\\hat{y}_2$', LIGHT_RED),
    }
    for name, (label, color) in node_props.items():
        x, y = positions[name]
        circle = Circle((x, y), 0.35, facecolor=color, edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Weight label on highlighted edge
    x1, y1 = positions['x1']
    x2, y2 = positions['h1']
    ax.text((x1+x2)/2 - 0.2, (y1+y2)/2 + 0.4, '$w_{1,1}^{(1)}$', fontsize=13,
           color=DARK_RED, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=RED, alpha=0.9))

    # Annotation
    ax.text(4.0, -0.3,
           'A weight in an early layer affects the loss\nthrough multiple paths (one per output neuron)',
           ha='center', fontsize=11, style='italic', color=GRAY)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/multiple_paths.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved multiple_paths.png')


# ============================================================================
# Figure 7: Gradient check — numerical vs analytical
# ============================================================================
def plot_gradient_check():
    fig, ax = plt.subplots(figsize=(8, 5))

    np.random.seed(42)

    # Generate random "analytical" gradients and add small noise for "numerical"
    n_params = 30
    analytical = np.random.randn(n_params) * 0.5
    numerical = analytical + np.random.randn(n_params) * 0.001

    ax.scatter(analytical, numerical, color=BLUE, s=40, alpha=0.7, edgecolors=DARK_BLUE,
              linewidth=0.5, zorder=3)

    # Perfect line
    lims = [-1.5, 1.5]
    ax.plot(lims, lims, '--', color=RED, linewidth=1.5, alpha=0.7, label='Perfect agreement')

    ax.set_xlabel('Analytical gradient (backpropagation)', fontsize=12)
    ax.set_ylabel('Numerical gradient (finite differences)', fontsize=12)
    ax.set_title('Gradient check: analytical vs. numerical', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/gradient_check.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved gradient_check.png')


# ============================================================================
# Figure 8: Forward vs reverse mode AD
# ============================================================================
def plot_forward_vs_reverse():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (ax, title, color, direction) in enumerate(zip(
        axes,
        ['Forward mode\n(one pass per input)', 'Reverse mode (backprop)\n(one pass per output)'],
        [BLUE, RED],
        ['right', 'left']
    )):
        # Draw a simple chain: 3 nodes
        xs = [1.5, 4.0, 6.5]
        y = 2.5
        labels = ['$f_1$', '$f_2$', '$f_3$']

        for i in range(len(xs) - 1):
            ax.annotate('', xy=(xs[i+1] - 0.4, y), xytext=(xs[i] + 0.4, y),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

        # Input/output arrows
        ax.annotate('', xy=(xs[0] - 0.4, y), xytext=(0.0, y),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
        ax.annotate('', xy=(8.0, y), xytext=(xs[-1] + 0.4, y),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

        ax.text(0.0, y + 0.4, '$x$', fontsize=14, ha='center')
        ax.text(8.0, y + 0.4, '$\\mathcal{L}$', fontsize=14, ha='center')

        # Draw nodes
        for x_pos, label in zip(xs, labels):
            circle = Circle((x_pos, y), 0.4, facecolor=LIGHT_BLUE, edgecolor=DARK_BLUE,
                           linewidth=1.5, zorder=5)
            ax.add_patch(circle)
            ax.text(x_pos, y, label, ha='center', va='center', fontsize=14,
                   fontweight='bold', zorder=6)

        # Derivative propagation arrows
        if direction == 'right':
            y_d = y - 1.0
            for i in range(len(xs)):
                if i == 0:
                    ax.annotate('', xy=(xs[i], y_d), xytext=(0.0, y_d),
                                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
                else:
                    ax.annotate('', xy=(xs[i], y_d), xytext=(xs[i-1], y_d),
                                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.annotate('', xy=(8.0, y_d), xytext=(xs[-1], y_d),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.text(4.0, y_d - 0.5, r'$\dot{h} = \frac{\partial h}{\partial x}$ propagated forward',
                   ha='center', fontsize=10, color=color, style='italic')
        else:
            y_d = y + 1.0
            for i in range(len(xs) - 1, -1, -1):
                if i == len(xs) - 1:
                    ax.annotate('', xy=(xs[i], y_d), xytext=(8.0, y_d),
                                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
                else:
                    ax.annotate('', xy=(xs[i], y_d), xytext=(xs[i+1], y_d),
                                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.annotate('', xy=(0.0, y_d), xytext=(xs[0], y_d),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.text(4.0, y_d + 0.5, r'$\bar{h} = \frac{\partial \mathcal{L}}{\partial h}$ propagated backward',
                   ha='center', fontsize=10, color=color, style='italic')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(0.3, 4.5)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/forward_vs_reverse_ad.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved forward_vs_reverse_ad.png')


# ============================================================================
# Figure 9: Autograd tape — dynamic computation graph
# ============================================================================
def plot_autograd_tape():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Nodes in the tape
    tape_nodes = [
        (1.0, 3.5, '$\\text{mm}$\n$W_h \\cdot h$', LIGHT_BLUE),
        (3.5, 3.5, '$\\text{mm}$\n$W_x \\cdot x$', LIGHT_BLUE),
        (5.5, 3.5, '$+$', LIGHT_GREEN),
        (7.5, 3.5, '$\\tanh$', '#FFF3CC'),
        (9.5, 3.5, '$\\text{sum}$', LIGHT_RED),
    ]

    # Leaf nodes
    leaf_nodes = [
        (0.0, 1.5, '$W_h$'), (2.0, 1.5, '$h$'),
        (3.0, 1.5, '$W_x$'), (4.5, 1.5, '$x$'),
    ]

    # Draw edges
    edges = [
        ((0.0, 1.5), (1.0, 3.5)),
        ((2.0, 1.5), (1.0, 3.5)),
        ((3.0, 1.5), (3.5, 3.5)),
        ((4.5, 1.5), (3.5, 3.5)),
        ((1.0, 3.5), (5.5, 3.5)),
        ((3.5, 3.5), (5.5, 3.5)),
        ((5.5, 3.5), (7.5, 3.5)),
        ((7.5, 3.5), (9.5, 3.5)),
    ]

    for (x1, y1), (x2, y2) in edges:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Draw tape nodes
    for x, y, label, color in tape_nodes:
        circle = Circle((x, y), 0.5, facecolor=color, edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)

    # Draw leaf nodes
    for x, y, label in leaf_nodes:
        rect = FancyBboxPatch((x - 0.4, y - 0.25), 0.8, 0.5,
                               boxstyle="round,pad=0.05", facecolor=LIGHT_GRAY,
                               edgecolor='#444', linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

    # Code annotation
    code = 'h2h = W_h @ h\ni2h = W_x @ x\nnext_h = (h2h + i2h).tanh()\nloss = next_h.sum()\nloss.backward()'
    ax.text(6.0, 0.5, code, fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8', edgecolor=GRAY),
           va='center')

    # Labels
    ax.text(9.5, 4.5, 'loss', ha='center', fontsize=11, color=DARK_RED, fontweight='bold')

    # backward arrow
    ax.annotate('', xy=(1.5, 4.5), xytext=(9.0, 4.5),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'))
    ax.text(5.5, 4.8, '.backward() traverses graph in reverse', fontsize=10,
           color=RED, ha='center', style='italic')

    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/autograd_tape.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Saved autograd_tape.png')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    plot_local_gradient_rule()
    plot_multi_input_rule()
    plot_relu_forward()
    plot_relu_backward()
    plot_neuron_decomposition()
    plot_multiple_paths()
    plot_gradient_check()
    plot_forward_vs_reverse()
    plot_autograd_tape()
    print('\nAll figures generated.')
