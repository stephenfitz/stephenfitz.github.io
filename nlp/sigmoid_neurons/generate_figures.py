#!/usr/bin/env python3
"""
Generate figures for the Sigmoid Neurons and Logistic Regression article.

Produces:
    - sigmoid_function.png          : The sigmoid curve with key properties annotated
    - activation_comparison.png     : Perceptron sign vs sigmoid activation side by side
    - cross_entropy_loss.png        : CE loss for y=1 and y=0 cases
    - quadratic_vs_ce.png           : Non-convexity of quadratic loss vs convexity of CE
    - gradient_descent.png          : Gradient descent on a convex loss curve
    - sigmoid_derivative.png        : σ(z) and σ'(z) = σ(z)(1−σ(z)) plotted together
    - log_odds.png                  : Log-odds (logit) as inverse of sigmoid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DPI = 200
BLUE = "#2563eb"
RED = "#dc2626"
GRAY = "#6b7280"
GREEN = "#059669"
ORANGE = "#d97706"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.dpi": DPI,
    })


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def plot_sigmoid_function(fig_path):
    """Plot the sigmoid function with key properties annotated."""
    z = np.linspace(-7, 7, 500)
    s = sigmoid(z)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, s, color=BLUE, linewidth=2.5)

    # Annotate key points
    ax.plot(0, 0.5, "o", color=RED, markersize=7, zorder=5)
    ax.annotate(r"$\sigma(0) = \frac{1}{2}$", xy=(0, 0.5),
                xytext=(1.5, 0.38), fontsize=11,
                arrowprops=dict(arrowstyle="->", color=GRAY))

    # Asymptotes
    ax.axhline(1, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(5.5, 1.04, "$1$", fontsize=10, color=GRAY)
    ax.text(5.5, -0.06, "$0$", fontsize=10, color=GRAY)

    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\sigma(z)$")
    ax.set_title(r"The Sigmoid Function: $\sigma(z) = \frac{1}{1 + e^{-z}}$")
    ax.set_ylim(-0.1, 1.15)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_activation_comparison(fig_path):
    """Side-by-side comparison of sign (perceptron) vs sigmoid activation."""
    z = np.linspace(-5, 5, 1000)
    sign_out = np.sign(z)
    sign_out[z == 0] = 0
    sig_out = sigmoid(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)

    # Perceptron (sign)
    ax1.plot(z[z < 0], sign_out[z < 0], color=RED, linewidth=2.5)
    ax1.plot(z[z > 0], sign_out[z > 0], color=BLUE, linewidth=2.5)
    ax1.plot(0, -1, "o", color=RED, markersize=6, zorder=5)
    ax1.plot(0, 1, "o", color=BLUE, markersize=6, zorder=5, fillstyle="none", markeredgewidth=2)
    ax1.set_xlabel("$z = w^\\top x$")
    ax1.set_ylabel("Output")
    ax1.set_title(r"Perceptron: $y = \mathrm{sgn}(z)$")
    ax1.set_ylim(-1.3, 1.3)
    ax1.axhline(0, color=GRAY, linewidth=0.5)
    ax1.axvline(0, color=GRAY, linewidth=0.5, linestyle=":")

    # Sigmoid
    ax2.plot(z, sig_out, color=BLUE, linewidth=2.5)
    ax2.plot(0, 0.5, "o", color=RED, markersize=6, zorder=5)
    ax2.set_xlabel("$z = w^\\top x$")
    ax2.set_ylabel("Output")
    ax2.set_title(r"Sigmoid neuron: $\hat{y} = \sigma(z)$")
    ax2.set_ylim(-0.1, 1.15)
    ax2.axhline(0.5, color=GRAY, linewidth=0.5, linestyle=":")
    ax2.axvline(0, color=GRAY, linewidth=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_cross_entropy_loss(fig_path):
    """Plot CE loss for y=1 and y=0 cases."""
    y_hat = np.linspace(0.001, 0.999, 500)

    loss_y1 = -np.log(y_hat)          # y=1: L = -log(ŷ)
    loss_y0 = -np.log(1 - y_hat)      # y=0: L = -log(1-ŷ)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(y_hat, loss_y1, color=BLUE, linewidth=2.5)
    ax1.set_xlabel(r"$\hat{y} = \sigma(w^\top x)$")
    ax1.set_ylabel("Loss")
    ax1.set_title(r"$y = 1$: Loss $= -\log(\hat{y})$")
    ax1.set_ylim(0, 5)
    ax1.annotate("Low loss when\n" + r"$\hat{y} \approx 1$", xy=(0.9, 0.1),
                 fontsize=9, ha="center", color=GREEN)
    ax1.annotate("High loss when\n" + r"$\hat{y} \approx 0$", xy=(0.15, 3.5),
                 fontsize=9, ha="center", color=RED)

    ax2.plot(y_hat, loss_y0, color=RED, linewidth=2.5)
    ax2.set_xlabel(r"$\hat{y} = \sigma(w^\top x)$")
    ax2.set_ylabel("Loss")
    ax2.set_title(r"$y = 0$: Loss $= -\log(1 - \hat{y})$")
    ax2.set_ylim(0, 5)
    ax2.annotate("Low loss when\n" + r"$\hat{y} \approx 0$", xy=(0.1, 0.1),
                 fontsize=9, ha="center", color=GREEN)
    ax2.annotate("High loss when\n" + r"$\hat{y} \approx 1$", xy=(0.85, 3.5),
                 fontsize=9, ha="center", color=RED)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_quadratic_vs_ce(fig_path):
    """Show non-convexity of quadratic loss with sigmoid vs convexity of CE."""
    # Single weight, single input scenario for illustration
    # x=1, so z = w, ŷ = σ(w)
    w = np.linspace(-6, 6, 500)
    y_hat = sigmoid(w)

    # Quadratic loss for y=1: L = 0.5*(σ(w) - 1)^2
    quad_loss_y1 = 0.5 * (y_hat - 1) ** 2
    # CE loss for y=1: L = -log(σ(w))
    ce_loss_y1 = -np.log(y_hat + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(w, quad_loss_y1, color=ORANGE, linewidth=2.5)
    ax1.set_xlabel("$w$")
    ax1.set_ylabel("Loss")
    ax1.set_title(r"Quadratic: $\frac{1}{2}(\sigma(w) - 1)^2$")
    ax1.set_ylim(0, 0.55)

    ax2.plot(w, ce_loss_y1, color=BLUE, linewidth=2.5)
    ax2.set_xlabel("$w$")
    ax2.set_ylabel("Loss")
    ax2.set_title(r"Cross-entropy: $-\log \sigma(w)$")
    ax2.set_ylim(0, 7)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_gradient_descent(fig_path):
    """Illustrate gradient descent on a convex loss curve."""
    w = np.linspace(-4, 6, 500)
    # Simple convex function for illustration
    loss = 0.3 * (w - 1.5) ** 2 + 0.2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w, loss, color=BLUE, linewidth=2.5)

    # Gradient descent steps
    steps_w = [5.0]
    lr = 0.3
    for _ in range(6):
        grad = 0.6 * (steps_w[-1] - 1.5)
        steps_w.append(steps_w[-1] - lr * grad)

    steps_loss = [0.3 * (wi - 1.5) ** 2 + 0.2 for wi in steps_w]

    for i in range(len(steps_w) - 1):
        ax.annotate("", xy=(steps_w[i + 1], steps_loss[i + 1]),
                     xytext=(steps_w[i], steps_loss[i]),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
        ax.plot(steps_w[i], steps_loss[i], "o", color=RED, markersize=5, zorder=5)

    ax.plot(steps_w[-1], steps_loss[-1], "o", color=RED, markersize=5, zorder=5)
    ax.plot(1.5, 0.2, "*", color=GREEN, markersize=12, zorder=5)

    # Annotations
    ax.annotate(r"$w^{(0)}$", xy=(steps_w[0], steps_loss[0]),
                xytext=(steps_w[0] + 0.3, steps_loss[0] + 0.3), fontsize=10)
    ax.text(1.5, 0.05, "minimum", fontsize=9, ha="center", color=GREEN)

    ax.set_xlabel("$w$")
    ax.set_ylabel(r"$\mathcal{L}(w)$")
    ax.set_title("Gradient Descent")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_sigmoid_derivative(fig_path):
    """Plot σ(z) and its derivative σ'(z) = σ(z)(1-σ(z))."""
    z = np.linspace(-7, 7, 500)
    s = sigmoid(z)
    ds = s * (1 - s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, s, color=BLUE, linewidth=2.5, label=r"$\sigma(z)$")
    ax.plot(z, ds, color=RED, linewidth=2.5, label=r"$\sigma'(z) = \sigma(z)(1 - \sigma(z))$")

    ax.plot(0, 0.25, "o", color=RED, markersize=6, zorder=5)
    ax.annotate(r"max $= \frac{1}{4}$ at $z = 0$", xy=(0, 0.25),
                xytext=(2.0, 0.35), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=GRAY))

    ax.set_xlabel("$z$")
    ax.set_ylabel("Value")
    ax.set_title("The Sigmoid and Its Derivative")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(-0.05, 1.1)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_log_odds(fig_path):
    """Plot the logit function (log-odds) as the inverse of sigmoid."""
    p = np.linspace(0.01, 0.99, 500)
    logit_p = np.log(p / (1 - p))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, logit_p, color=BLUE, linewidth=2.5)
    ax.axhline(0, color=GRAY, linewidth=0.5)
    ax.axvline(0.5, color=GRAY, linewidth=0.5, linestyle=":")

    ax.plot(0.5, 0, "o", color=RED, markersize=6, zorder=5)
    ax.annotate(r"$p = \frac{1}{2} \Rightarrow \log\text{-odds} = 0$",
                xy=(0.5, 0), xytext=(0.65, -2.5), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=GRAY))

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\log \frac{p}{1-p}$")
    ax.set_title(r"The Logit Function: $\mathrm{logit}(p) = \log \frac{p}{1-p}$")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    print("Generating figures ...")
    plot_sigmoid_function(os.path.join(FIG_DIR, "sigmoid_function.png"))
    plot_activation_comparison(os.path.join(FIG_DIR, "activation_comparison.png"))
    plot_cross_entropy_loss(os.path.join(FIG_DIR, "cross_entropy_loss.png"))
    plot_quadratic_vs_ce(os.path.join(FIG_DIR, "quadratic_vs_ce.png"))
    plot_gradient_descent(os.path.join(FIG_DIR, "gradient_descent.png"))
    plot_sigmoid_derivative(os.path.join(FIG_DIR, "sigmoid_derivative.png"))
    plot_log_odds(os.path.join(FIG_DIR, "log_odds.png"))
    print("Done.")


if __name__ == "__main__":
    main()
