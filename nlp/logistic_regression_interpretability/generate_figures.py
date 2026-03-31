"""Generate figures for the Logistic Regression Interpretability article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette
BLUE = "#2563eb"
RED = "#dc2626"
GRAY = "#6b7280"
GREEN = "#059669"
ORANGE = "#d97706"
PURPLE = "#7c3aed"
LIGHT_BLUE = "#93c5fd"
LIGHT_RED = "#fca5a5"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p / (1.0 - p))


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Figure 1: Odds multiplier interpretation ──────────────────────────

def fig_odds_multiplier():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: log-odds as linear function
    x = np.linspace(-3, 3, 200)
    w = 0.8
    b = -0.5
    log_odds = w * x + b

    ax1.plot(x, log_odds, color=BLUE, linewidth=2.5)
    ax1.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Feature $x$", fontsize=12)
    ax1.set_ylabel("Log-odds $w x + b$", fontsize=12)
    ax1.set_title("Log-odds are linear in the feature", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Annotate slope
    x1, x2 = 0.5, 1.5
    y1, y2 = w * x1 + b, w * x2 + b
    ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax1.annotate(f"$\\Delta x = 1 \\Rightarrow \\Delta \\ell = w = {w}$",
                 xy=(x2 + 0.1, (y1 + y2) / 2), fontsize=10, color=RED)

    # Right: odds multiplier
    wvals = np.linspace(-2, 2, 200)
    odds_mult = np.exp(wvals)

    ax2.plot(wvals, odds_mult, color=RED, linewidth=2.5)
    ax2.axhline(1, color=GRAY, linewidth=0.5, linestyle="--")
    ax2.axvline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Weight $w_j$", fontsize=12)
    ax2.set_ylabel("Odds multiplier $e^{w_j}$", fontsize=12)
    ax2.set_title("Each unit increase multiplies odds by $e^{w_j}$", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 7)

    # Mark a few points
    for wv in [-1, 0, 0.5, 1]:
        ax2.plot(wv, np.exp(wv), "o", color=BLUE, markersize=6, zorder=5)
        label = f"$e^{{{wv}}} = {np.exp(wv):.2f}$"
        offset = (10, 10) if wv >= 0 else (10, -15)
        ax2.annotate(label, xy=(wv, np.exp(wv)), xytext=offset,
                     textcoords="offset points", fontsize=9, color=BLUE)

    fig.tight_layout()
    save(fig, "odds_multiplier.png")


# ── Figure 2: Soft labels vs hard labels ──────────────────────────────

def fig_soft_labels():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    n = 30
    x = np.sort(np.random.randn(n) * 1.5)
    hard = (x > 0.3).astype(float)
    # Soft labels from a "teacher" model
    soft = sigmoid(1.5 * x - 0.2)

    ax1.scatter(x, hard, color=BLUE, s=40, zorder=5, edgecolors="white", linewidth=0.5)
    ax1.set_xlabel("Feature $x$", fontsize=12)
    ax1.set_ylabel("Label $y$", fontsize=12)
    ax1.set_title("Hard labels $y \\in \\{0, 1\\}$", fontsize=13)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)

    ax2.scatter(x, soft, color=RED, s=40, zorder=5, edgecolors="white", linewidth=0.5)
    ax2.set_xlabel("Feature $x$", fontsize=12)
    ax2.set_ylabel("Soft label $q = p_f(x)$", fontsize=12)
    ax2.set_title("Soft labels $q \\in (0, 1)$ from teacher", fontsize=13)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "soft_labels.png")


# ── Figure 3: CE in logit form for different q ────────────────────────

def fig_ce_logit_form():
    fig, ax = plt.subplots(figsize=(8, 5))

    z = np.linspace(-5, 5, 300)
    q_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = [BLUE, PURPLE, GRAY, ORANGE, RED]

    for q, c in zip(q_vals, colors):
        ce = np.log(1 + np.exp(z)) - q * z
        ax.plot(z, ce, color=c, linewidth=2, label=f"$q = {q}$")
        # Mark minimum at z* = logit(q)
        z_star = logit(q)
        ce_star = np.log(1 + np.exp(z_star)) - q * z_star
        ax.plot(z_star, ce_star, "o", color=c, markersize=7, zorder=5)

    ax.set_xlabel("Logit $z$", fontsize=12)
    ax.set_ylabel("$\\mathrm{CE}(q, \\sigma(z)) = \\log(1 + e^z) - qz$", fontsize=12)
    ax.set_title("Cross-entropy loss in logit form", fontsize=13)
    ax.legend(fontsize=11, loc="upper center")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    fig.tight_layout()
    save(fig, "ce_logit_form.png")


# ── Figure 4: Stationary condition ────────────────────────────────────

def fig_stationary_condition():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    z = np.linspace(-4, 4, 300)

    # Left: CE and its derivative for q=0.7
    q = 0.7
    ce = np.log(1 + np.exp(z)) - q * z
    dce = sigmoid(z) - q
    z_star = logit(q)

    ax1.plot(z, ce, color=BLUE, linewidth=2.5, label="$\\ell(z; q)$")
    ax1.axvline(z_star, color=RED, linewidth=1, linestyle="--", alpha=0.7)
    ce_min = np.log(1 + np.exp(z_star)) - q * z_star
    ax1.plot(z_star, ce_min, "o", color=RED, markersize=8, zorder=5)
    ax1.annotate(f"$z^* = \\mathrm{{logit}}({q}) = {z_star:.2f}$",
                 xy=(z_star, ce_min), xytext=(30, 30),
                 textcoords="offset points", fontsize=11, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
    ax1.set_xlabel("Logit $z$", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Loss $\\ell(z; q)$ for $q = {q}$", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Right: derivative showing zero crossing
    ax2.plot(z, dce, color=BLUE, linewidth=2.5, label="$\\ell'(z) = \\sigma(z) - q$")
    ax2.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax2.axvline(z_star, color=RED, linewidth=1, linestyle="--", alpha=0.7)
    ax2.plot(z_star, 0, "o", color=RED, markersize=8, zorder=5)
    ax2.set_xlabel("Logit $z$", fontsize=12)
    ax2.set_ylabel("Derivative", fontsize=12)
    ax2.set_title("Gradient vanishes at $z^* = \\mathrm{logit}(q)$", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "stationary_condition.png")


# ── Figure 5: Quadratic approximation ────────────────────────────────

def fig_local_quadratic():
    fig, ax = plt.subplots(figsize=(7, 5))

    q = 0.7
    t = logit(q)
    z = np.linspace(t - 3, t + 3, 300)

    # True CE
    ce = np.log(1 + np.exp(z)) - q * z
    ce_min = np.log(1 + np.exp(t)) - q * t

    # Quadratic approximation
    ce_quad = ce_min + 0.5 * q * (1 - q) * (z - t) ** 2

    ax.plot(z, ce, color=BLUE, linewidth=2.5, label="True CE: $\\log(1+e^z) - qz$")
    ax.plot(z, ce_quad, color=RED, linewidth=2, linestyle="--",
            label="Quadratic: $\\ell(t) + \\frac{1}{2}q(1-q)(z-t)^2$")
    ax.plot(t, ce_min, "o", color=PURPLE, markersize=8, zorder=5,
            label=f"Optimum $t = \\mathrm{{logit}}({q})$")

    ax.set_xlabel("Logit $z$", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Local quadratic structure of CE ($q = {q}$)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    fig.tight_layout()
    save(fig, "local_quadratic.png")


# ── Figure 6: Weight function ω(q) = q(1-q) ──────────────────────────

def fig_weight_function():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    q = np.linspace(0.01, 0.99, 300)
    omega = q * (1 - q)

    ax.plot(q, omega, color=BLUE, linewidth=2.5)
    ax.fill_between(q, omega, alpha=0.15, color=BLUE)
    ax.axhline(0.25, color=RED, linewidth=1, linestyle="--", alpha=0.7,
               label="Maximum $\\omega = 1/4$ at $q = 1/2$")
    ax.plot(0.5, 0.25, "o", color=RED, markersize=8, zorder=5)

    # Shade low-weight regions
    mask_low = (q < 0.15) | (q > 0.85)
    ax.fill_between(q, omega, where=mask_low, alpha=0.3, color=ORANGE,
                    label="Low weight: confident predictions")

    ax.set_xlabel("Soft label $q$", fontsize=12)
    ax.set_ylabel("Weight $\\omega(q) = q(1-q)$", fontsize=12)
    ax.set_title("Weight function in the local approximation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "weight_function.png")


# ── Figure 7: Linear probe schematic ──────────────────────────────────

def fig_linear_probe():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input
    ax.text(0.5, 2, "$x$", fontsize=16, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5))

    # Arrow
    ax.annotate("", xy=(2.0, 2), xytext=(1.1, 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # Deep network box
    rect = plt.Rectangle((2.0, 0.8), 2.5, 2.4, fill=True, facecolor="#f0f0f0",
                          edgecolor=GRAY, linewidth=1.5, linestyle="-")
    ax.add_patch(rect)
    ax.text(3.25, 2.5, "Deep", fontsize=12, ha="center", va="center", color=GRAY, fontweight="bold")
    ax.text(3.25, 1.7, "Network", fontsize=12, ha="center", va="center", color=GRAY, fontweight="bold")
    ax.text(3.25, 1.0, "(frozen)", fontsize=9, ha="center", va="center", color=GRAY, style="italic")

    # Arrow
    ax.annotate("", xy=(5.3, 2), xytext=(4.6, 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # Hidden representation
    ax.text(5.9, 2, "$h_\\ell(x)$", fontsize=15, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor=BLUE, linewidth=1.5))

    # Arrow
    ax.annotate("", xy=(7.3, 2), xytext=(6.7, 2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))

    # Logistic regression (trainable)
    rect2 = plt.Rectangle((7.3, 1.0), 1.6, 2.0, fill=True, facecolor="#fee2e2",
                           edgecolor=RED, linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(8.1, 2.4, "$\\sigma(w^\\top h + b)$", fontsize=11, ha="center", va="center",
            color=RED, fontweight="bold")
    ax.text(8.1, 1.5, "Linear", fontsize=10, ha="center", va="center", color=RED)
    ax.text(8.1, 1.15, "Probe", fontsize=10, ha="center", va="center", color=RED)

    # Arrow to output
    ax.annotate("", xy=(9.6, 2), xytext=(9.0, 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # Output
    ax.text(9.8, 2, "$\\hat{y}$", fontsize=16, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d1fae5", edgecolor=GREEN, linewidth=1.5))

    # Title and annotation
    ax.text(5.0, 3.7, "Linear Probe Architecture", fontsize=14, ha="center",
            va="center", fontweight="bold", color="#1f2937")

    fig.tight_layout()
    save(fig, "linear_probe.png")


# ── Figure 8: Projection view ────────────────────────────────────────

def fig_projection_view():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Linear subspace (a plane/line in 2D schematic)
    ax.plot([0, 4], [0.5, 2.5], color=BLUE, linewidth=2.5, label="Linear span of $\\tilde{\\phi}$")

    # True logit function point
    t_x, t_y = 2.5, 4.0
    ax.plot(t_x, t_y, "o", color=RED, markersize=10, zorder=5)
    ax.text(t_x + 0.15, t_y + 0.15, "$t(x) = \\mathrm{logit}(q(x))$",
            fontsize=12, color=RED)

    # Projection point
    # Project (2.5, 4.0) onto line from (0, 0.5) to (4, 2.5)
    # Direction: (4, 2) normalized
    d = np.array([4.0, 2.0])
    d = d / np.linalg.norm(d)
    p0 = np.array([0.0, 0.5])
    v = np.array([t_x, t_y]) - p0
    proj_len = np.dot(v, d)
    proj_point = p0 + proj_len * d

    ax.plot(proj_point[0], proj_point[1], "s", color=PURPLE, markersize=10, zorder=5)
    ax.text(proj_point[0] + 0.15, proj_point[1] - 0.35,
            "$\\tilde{w}^\\top \\tilde{\\phi}(x)$", fontsize=12, color=PURPLE)

    # Dashed line from t to projection
    ax.plot([t_x, proj_point[0]], [t_y, proj_point[1]],
            color=GRAY, linewidth=1.5, linestyle="--")

    # Right angle mark
    perp = np.array([t_x, t_y]) - proj_point
    perp_norm = perp / np.linalg.norm(perp)
    d_scaled = d * 0.2
    perp_scaled = perp_norm * 0.2
    corner = proj_point + d_scaled
    ax.plot([proj_point[0] + d_scaled[0], proj_point[0] + d_scaled[0] + perp_scaled[0]],
            [proj_point[1] + d_scaled[1], proj_point[1] + d_scaled[1] + perp_scaled[1]],
            color=GRAY, linewidth=1)
    ax.plot([proj_point[0] + perp_scaled[0], proj_point[0] + d_scaled[0] + perp_scaled[0]],
            [proj_point[1] + perp_scaled[1], proj_point[1] + d_scaled[1] + perp_scaled[1]],
            color=GRAY, linewidth=1)

    # Label the residual
    mid_x = (t_x + proj_point[0]) / 2
    mid_y = (t_y + proj_point[1]) / 2
    ax.text(mid_x + 0.2, mid_y, "residual", fontsize=10, color=GRAY, rotation=70)

    ax.set_title("Orthogonal projection of teacher logits\nonto the linear span of features",
                 fontsize=13, pad=10)

    fig.tight_layout()
    save(fig, "projection_view.png")


if __name__ == "__main__":
    print("Generating figures for Logistic Regression Interpretability...")
    fig_odds_multiplier()
    fig_soft_labels()
    fig_ce_logit_form()
    fig_stationary_condition()
    fig_local_quadratic()
    fig_weight_function()
    fig_linear_probe()
    fig_projection_view()
    print("Done!")
