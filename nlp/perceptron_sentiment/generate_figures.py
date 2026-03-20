#!/usr/bin/env python3
"""
Generate figures for the perceptron sentiment classification article.

Trains a perceptron to classify word sentiment from GloVe 50d embeddings,
producing convergence plots, PCA visualizations, and sentiment spectrum figures.

Usage:
    python generate_figures.py                        # uses bundled glove_subset.npz
    python generate_figures.py --glove-path glove.6B.50d.txt  # extract from full file
    python generate_figures.py --download              # download GloVe from Stanford
"""

import argparse
import os
import sys
import zipfile
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

POSITIVE_TRAIN = [
    "good", "great", "excellent", "wonderful", "fantastic", "beautiful",
    "amazing", "love", "happy", "joy", "brilliant", "perfect", "superb",
    "delightful", "pleasant", "outstanding", "magnificent", "marvelous",
    "terrific", "splendid", "graceful", "charming", "cheerful", "bright",
    "gentle", "kind", "warm", "generous", "friendly", "peaceful",
    "triumph", "success", "celebrate", "praise", "admire", "treasure",
    "bliss", "paradise", "harmony", "glorious", "vibrant", "radiant",
    "elegant", "noble", "brave", "inspire", "gratitude", "hope",
    "comfort", "proud",
]

NEGATIVE_TRAIN = [
    "bad", "terrible", "horrible", "awful", "disgusting", "ugly",
    "hate", "sad", "miserable", "pain", "dreadful", "worst",
    "nasty", "cruel", "vile", "wretched", "pathetic", "grim",
    "tragic", "dismal", "horrid", "abysmal", "atrocious", "gloomy",
    "harsh", "hostile", "bitter", "fearful", "angry", "violent",
    "failure", "destroy", "suffer", "agony", "despair", "grief",
    "doom", "nightmare", "chaos", "sinister", "toxic", "corrupt",
    "brutal", "ruthless", "cowardly", "disgrace", "shame", "misery",
    "torment", "dread",
]

POSITIVE_TEST = [
    "lovely", "nice", "fine", "enjoyable", "exciting", "positive",
    "grateful", "victorious", "talented", "honest", "compassionate",
    "caring", "faithful", "joyful", "lively", "gracious", "respected",
    "admirable", "worthy", "delicious", "fortunate", "glowing",
    "serene", "tender", "courageous",
]

NEGATIVE_TEST = [
    "evil", "wicked", "horrendous", "depressing", "annoying", "negative",
    "rotten", "lousy", "boring", "dull", "frustrating",
    "heartless", "selfish", "jealous", "anxious", "lonely", "helpless",
    "useless", "worthless", "hideous", "pitiful", "regret",
    "sorrow", "anguish", "misfortune",
]

NEUTRAL_PROBE = [
    "table", "chair", "water", "house", "car", "book", "tree",
    "mountain", "river", "road", "computer", "phone", "window",
    "paper", "clock",
]


# ---------------------------------------------------------------------------
# GloVe loading utilities
# ---------------------------------------------------------------------------

def load_glove_full(path, vocab):
    """Load specific words from a full GloVe text file."""
    vocab_set = set(vocab)
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab_set:
                vectors[word] = np.array([float(x) for x in parts[1:]], dtype=np.float64)
    return vectors


def save_subset(vectors, path):
    """Save word vectors as a .npz file."""
    words = sorted(vectors.keys())
    matrix = np.array([vectors[w] for w in words])
    np.savez_compressed(path, words=np.array(words), vectors=matrix)
    print(f"Saved {len(words)} word vectors to {path}")


def load_subset(path):
    """Load word vectors from a .npz file."""
    data = np.load(path, allow_pickle=True)
    words = list(data["words"])
    matrix = data["vectors"]
    return {w: matrix[i] for i, w in enumerate(words)}


def download_glove(dest_dir):
    """Download GloVe 6B from Stanford and extract the 50d file."""
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(dest_dir, "glove.6B.zip")
    txt_path = os.path.join(dest_dir, "glove.6B.50d.txt")

    if os.path.exists(txt_path):
        print(f"Found {txt_path}")
        return txt_path

    print(f"Downloading GloVe 6B from {url} ...")
    print("(This is ~860MB and may take several minutes)")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting glove.6B.50d.txt ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("glove.6B.50d.txt", dest_dir)
    os.remove(zip_path)
    print(f"Saved {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

def perceptron_train(X, y, max_epochs=1000):
    """
    Train a perceptron using the update rule w <- w + t*x.

    X: (m, d) array of input vectors (already normalized, bias appended)
    y: (m,) array of labels in {-1, +1}

    Returns: (w, history)
        w: learned weight vector
        history: list of mistake counts per epoch
    """
    m, d = X.shape
    w = np.zeros(d)
    history = []

    for epoch in range(max_epochs):
        mistakes = 0
        for i in range(m):
            if y[i] * (w @ X[i]) <= 0:
                w = w + y[i] * X[i]
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:
            break

    return w, history


def compute_margin(X, y, w):
    """Compute the margin alpha = min_i |w_hat^T x_i| where w_hat = w/||w||."""
    w_hat = w / np.linalg.norm(w)
    functional_margins = y * (X @ w_hat)
    return np.min(functional_margins)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

# Style constants
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
POS_COLOR = "#2563eb"   # blue
NEG_COLOR = "#dc2626"   # red
NEUTRAL_COLOR = "#6b7280"  # gray
BG_COLOR = "white"
GRID_ALPHA = 0.3
DPI = 200


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.dpi": DPI,
    })


def plot_convergence(history, fig_path):
    """Plot mistakes per epoch (convergence curve)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, "o-", color=POS_COLOR, markersize=5, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mistakes")
    ax.set_title("Perceptron Convergence")
    ax.set_xticks(list(epochs))
    ax.set_ylim(bottom=0)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_pca_scatter(X_train, y_train, words_train, pca, fig_path, title="Training Words in PCA Space"):
    """PCA scatter of training words colored by sentiment."""
    Z = pca.transform(X_train)
    fig, ax = plt.subplots(figsize=(8, 6))

    pos_mask = y_train == 1
    neg_mask = y_train == -1

    ax.scatter(Z[pos_mask, 0], Z[pos_mask, 1], c=POS_COLOR, s=30, alpha=0.8, label="Positive", zorder=3)
    ax.scatter(Z[neg_mask, 0], Z[neg_mask, 1], c=NEG_COLOR, s=30, alpha=0.8, label="Negative", zorder=3)

    for i, word in enumerate(words_train):
        ax.annotate(word, (Z[i, 0], Z[i, 1]), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(3, 3))

    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_decision_boundary(X_train, y_train, words_train, w, pca, fig_path):
    """PCA scatter with projected decision boundary."""
    Z = pca.transform(X_train)
    fig, ax = plt.subplots(figsize=(8, 6))

    pos_mask = y_train == 1
    neg_mask = y_train == -1

    ax.scatter(Z[pos_mask, 0], Z[pos_mask, 1], c=POS_COLOR, s=30, alpha=0.8, label="Positive", zorder=3)
    ax.scatter(Z[neg_mask, 0], Z[neg_mask, 1], c=NEG_COLOR, s=30, alpha=0.8, label="Negative", zorder=3)

    for i, word in enumerate(words_train):
        ax.annotate(word, (Z[i, 0], Z[i, 1]), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(3, 3))

    # Project the decision boundary into PCA space
    # The hyperplane w^T x = 0 in the original space projects to a line in 2D PCA space
    # PCA components are the rows of pca.components_ (shape 2 x d)
    # The weight vector w projects to w_pca = pca.components_ @ w[:d_orig]
    # (we exclude bias dimension for PCA since PCA was fit on original vectors)
    d_orig = pca.components_.shape[1]
    w_pca = pca.components_ @ w[:d_orig]  # project weight vector
    b_pca = w[-1]  # bias weight (from the appended 1)

    # The projected decision boundary: w_pca^T z + b_pca = 0
    # => z_1 * w_pca[0] + z_2 * w_pca[1] + b_pca_adjusted = 0
    # Need to account for PCA mean: w^T(x - mean) in PCA coords
    # Actually: w^T x = w_pca^T z_centered + w[:d_orig]^T pca.mean_ + b
    # So boundary in z-space: w_pca^T z = -(w[:d_orig]^T pca.mean_ + b)
    offset = w[:d_orig] @ pca.mean_ + b_pca

    # Draw boundary line: w_pca[0]*z0 + w_pca[1]*z1 + offset = 0
    xlim = ax.get_xlim()
    if abs(w_pca[1]) > 1e-10:
        z0_range = np.linspace(xlim[0] - 0.5, xlim[1] + 0.5, 100)
        z1_boundary = -(w_pca[0] * z0_range + offset) / w_pca[1]
        ylim = ax.get_ylim()
        mask = (z1_boundary > ylim[0] - 1) & (z1_boundary < ylim[1] + 1)
        ax.plot(z0_range[mask], z1_boundary[mask], "k--", linewidth=1.5, label="Decision boundary", zorder=2)
    else:
        # Vertical line
        z0_val = -offset / w_pca[0]
        ax.axvline(z0_val, color="k", linestyle="--", linewidth=1.5, label="Decision boundary", zorder=2)

    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_title("Decision Boundary in PCA Space")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_generalization(X_train, y_train, words_train,
                        X_test, y_test, y_pred_test, words_test,
                        pca, fig_path):
    """PCA scatter showing train (small) + test (large, with prediction markers)."""
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Training points (small, faded)
    pos_mask_tr = y_train == 1
    neg_mask_tr = y_train == -1
    ax.scatter(Z_train[pos_mask_tr, 0], Z_train[pos_mask_tr, 1],
               c=POS_COLOR, s=15, alpha=0.3, zorder=2)
    ax.scatter(Z_train[neg_mask_tr, 0], Z_train[neg_mask_tr, 1],
               c=NEG_COLOR, s=15, alpha=0.3, zorder=2)

    # Test points (larger)
    correct = y_pred_test == y_test
    for i, word in enumerate(words_test):
        color = POS_COLOR if y_test[i] == 1 else NEG_COLOR
        marker = "o" if correct[i] else "x"
        size = 50 if correct[i] else 70
        ax.scatter(Z_test[i, 0], Z_test[i, 1], c=color, s=size,
                   marker=marker, zorder=3, linewidths=1.5 if not correct[i] else 1)
        ax.annotate(word, (Z_test[i, 0], Z_test[i, 1]), fontsize=7, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4),
                    fontweight="bold" if not correct[i] else "normal")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POS_COLOR, markersize=8, label="Positive (correct)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG_COLOR, markersize=8, label="Negative (correct)"),
        Line2D([0], [0], marker="x", color="k", markersize=8, label="Misclassified", linestyle="None", markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc="best", framealpha=0.9)
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_title("Generalization to Held-Out Words")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_score_histogram(scores_pos, scores_neg, fig_path):
    """Histogram of w^T x scores for positive vs negative words."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(min(scores_neg.min(), scores_pos.min()) - 0.1,
                       max(scores_neg.max(), scores_pos.max()) + 0.1, 30)
    ax.hist(scores_pos, bins=bins, color=POS_COLOR, alpha=0.6, label="Positive words", edgecolor="white")
    ax.hist(scores_neg, bins=bins, color=NEG_COLOR, alpha=0.6, label="Negative words", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Decision boundary")
    ax.set_xlabel(r"Sentiment score ($w^\top x$)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sentiment Scores")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_sentiment_spectrum(words, scores, labels, fig_path):
    """
    Plot words along a horizontal sentiment axis, ordered by score.
    labels: +1 for positive, -1 for negative, 0 for neutral/probe
    """
    order = np.argsort(scores)
    words_sorted = [words[i] for i in order]
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (word, score, label) in enumerate(zip(words_sorted, scores_sorted, labels_sorted)):
        if label == 1:
            color = POS_COLOR
        elif label == -1:
            color = NEG_COLOR
        else:
            color = NEUTRAL_COLOR
        ax.plot(score, 0, "o", color=color, markersize=5, zorder=3)
        # Alternate y positions to avoid overlap
        y_offset = 0.06 * (1 if i % 2 == 0 else -1) * (1 + (i % 3) * 0.4)
        ax.annotate(word, (score, 0), fontsize=6.5, color=color, alpha=0.85,
                    textcoords="offset points",
                    xytext=(0, 10 * (1 if i % 2 == 0 else -1) * (1 + (i % 3) * 0.3)),
                    ha="center", rotation=45 if i % 2 == 0 else -45)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Sentiment score ($w^\top x$)")
    ax.set_yticks([])
    ax.set_title("Words Along the Learned Sentiment Axis")

    # Arrows at ends
    ax.annotate("", xy=(scores_sorted[-1] + 0.15, -0.22),
                xytext=(scores_sorted[-1] - 0.3, -0.22),
                arrowprops=dict(arrowstyle="->", color=POS_COLOR, lw=1.5))
    ax.text(scores_sorted[-1] - 0.08, -0.28, "Positive", fontsize=9,
            color=POS_COLOR, ha="center")

    ax.annotate("", xy=(scores_sorted[0] - 0.15, -0.22),
                xytext=(scores_sorted[0] + 0.3, -0.22),
                arrowprops=dict(arrowstyle="->", color=NEG_COLOR, lw=1.5))
    ax.text(scores_sorted[0] + 0.08, -0.28, "Negative", fontsize=9,
            color=NEG_COLOR, ha="center")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POS_COLOR, markersize=6, label="Positive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG_COLOR, markersize=6, label="Negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEUTRAL_COLOR, markersize=6, label="Neutral"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=8)

    ax.set_ylim(-0.4, 0.4)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate perceptron sentiment figures")
    parser.add_argument("--glove-path", type=str, default=None,
                        help="Path to glove.6B.50d.txt")
    parser.add_argument("--download", action="store_true",
                        help="Download GloVe 6B from Stanford")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(script_dir, "glove_subset.npz")
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    # Collect all words we need
    all_words = (POSITIVE_TRAIN + NEGATIVE_TRAIN +
                 POSITIVE_TEST + NEGATIVE_TEST + NEUTRAL_PROBE)

    # Load vectors
    if os.path.exists(npz_path) and args.glove_path is None and not args.download:
        print("Loading vectors from glove_subset.npz ...")
        vectors = load_subset(npz_path)
    else:
        glove_path = args.glove_path
        if glove_path is None:
            if args.download:
                glove_path = download_glove(script_dir)
            else:
                print("ERROR: No glove_subset.npz found. Provide --glove-path or --download.")
                sys.exit(1)

        print(f"Loading vectors from {glove_path} ...")
        vectors = load_glove_full(glove_path, all_words)
        save_subset(vectors, npz_path)

    # Check coverage
    missing = [w for w in all_words if w not in vectors]
    if missing:
        print(f"WARNING: Missing {len(missing)} words: {missing}")
        # Remove missing words from lists
        all_words = [w for w in all_words if w in vectors]

    # Build training data
    train_words = [w for w in POSITIVE_TRAIN + NEGATIVE_TRAIN if w in vectors]
    train_labels = np.array([1 if w in POSITIVE_TRAIN else -1 for w in train_words])
    train_vecs_raw = np.array([vectors[w] for w in train_words])

    test_words = [w for w in POSITIVE_TEST + NEGATIVE_TEST if w in vectors]
    test_labels = np.array([1 if w in POSITIVE_TEST else -1 for w in test_words])
    test_vecs_raw = np.array([vectors[w] for w in test_words])

    probe_words = [w for w in NEUTRAL_PROBE if w in vectors]
    probe_vecs_raw = np.array([vectors[w] for w in probe_words])

    print(f"\nTraining: {len(train_words)} words ({sum(train_labels == 1)} pos, {sum(train_labels == -1)} neg)")
    print(f"Test: {len(test_words)} words ({sum(test_labels == 1)} pos, {sum(test_labels == -1)} neg)")
    print(f"Probe: {len(probe_words)} neutral words")

    # Normalize: divide by max norm across all training vectors
    max_norm = np.max(np.linalg.norm(train_vecs_raw, axis=1))
    train_vecs_norm = train_vecs_raw / max_norm
    test_vecs_norm = test_vecs_raw / max_norm
    probe_vecs_norm = probe_vecs_raw / max_norm

    # Append bias (constant 1)
    train_X = np.hstack([train_vecs_norm, np.ones((len(train_words), 1))])
    test_X = np.hstack([test_vecs_norm, np.ones((len(test_words), 1))])
    probe_X = np.hstack([probe_vecs_norm, np.ones((len(probe_words), 1))])

    # --- Train perceptron ---
    print("\nTraining perceptron ...")
    w, history = perceptron_train(train_X, train_labels, max_epochs=200)

    print(f"Converged after {len(history)} epochs")
    total_updates = sum(history)
    print(f"Total updates: {total_updates}")
    print(f"Mistakes per epoch: {history}")

    # Compute margin
    alpha = compute_margin(train_X, train_labels, w)
    bound = 1.0 / (alpha ** 2)
    print(f"\nMargin (alpha): {alpha:.4f}")
    print(f"Convergence bound (1/alpha^2): {bound:.1f}")
    print(f"Actual updates: {total_updates} (bound predicts <= {int(np.ceil(bound))})")

    # --- Test accuracy ---
    test_preds = np.sign(test_X @ w)
    test_preds[test_preds == 0] = -1  # break ties
    test_acc = np.mean(test_preds == test_labels)
    print(f"\nTest accuracy: {test_acc:.1%} ({int(test_acc * len(test_labels))}/{len(test_labels)})")

    misclassified = [test_words[i] for i in range(len(test_words))
                     if test_preds[i] != test_labels[i]]
    if misclassified:
        print(f"Misclassified: {misclassified}")

    # --- PCA for visualization (fit on training vectors, original space) ---
    pca = PCA(n_components=2)
    pca.fit(train_vecs_norm)

    # --- Generate all figures ---
    print("\nGenerating figures ...")

    # 1. Convergence plot
    plot_convergence(history, os.path.join(FIG_DIR, "convergence_plot.png"))

    # 2. PCA scatter of training words
    plot_pca_scatter(train_vecs_norm, train_labels, train_words, pca,
                     os.path.join(FIG_DIR, "pca_sentiment_train.png"))

    # 3. PCA with decision boundary
    plot_decision_boundary(train_vecs_norm, train_labels, train_words, w, pca,
                           os.path.join(FIG_DIR, "pca_decision_boundary.png"))

    # 4. Generalization plot
    plot_generalization(train_vecs_norm, train_labels, train_words,
                        test_vecs_norm, test_labels, test_preds, test_words,
                        pca, os.path.join(FIG_DIR, "pca_generalization.png"))

    # 5. Score histogram
    train_scores = train_X @ w
    pos_scores = train_scores[train_labels == 1]
    neg_scores = train_scores[train_labels == -1]
    plot_score_histogram(pos_scores, neg_scores,
                         os.path.join(FIG_DIR, "score_histogram.png"))

    # 6. Sentiment spectrum
    # Combine a subset of train + test + probe words
    spectrum_words_pos = ["love", "joy", "beautiful", "wonderful", "great",
                          "happy", "brilliant", "perfect", "hope", "gentle"]
    spectrum_words_neg = ["hate", "pain", "ugly", "terrible", "awful",
                          "miserable", "cruel", "worst", "doom", "hostile"]
    spectrum_words_test = ["lovely", "exciting", "evil", "depressing", "boring", "courageous"]
    spectrum_words_neutral = probe_words[:8]

    spectrum_words = spectrum_words_pos + spectrum_words_neg + spectrum_words_test + spectrum_words_neutral
    spectrum_words = [w_ for w_ in spectrum_words if w_ in vectors]
    spectrum_vecs = np.array([vectors[w_] for w_ in spectrum_words]) / max_norm
    spectrum_X = np.hstack([spectrum_vecs, np.ones((len(spectrum_words), 1))])
    spectrum_scores = spectrum_X @ w

    spectrum_labels = np.array([
        1 if w_ in POSITIVE_TRAIN + POSITIVE_TEST else
        (-1 if w_ in NEGATIVE_TRAIN + NEGATIVE_TEST else 0)
        for w_ in spectrum_words
    ])

    plot_sentiment_spectrum(spectrum_words, spectrum_scores, spectrum_labels,
                            os.path.join(FIG_DIR, "sentiment_spectrum.png"))

    # --- Print summary stats for the article ---
    print("\n" + "=" * 60)
    print("SUMMARY FOR ARTICLE")
    print("=" * 60)
    print(f"Embedding dimension: 50 (GloVe 6B)")
    print(f"Training set: {sum(train_labels == 1)} positive + {sum(train_labels == -1)} negative = {len(train_words)} words")
    print(f"Test set: {sum(test_labels == 1)} positive + {sum(test_labels == -1)} negative = {len(test_words)} words")
    print(f"Epochs to convergence: {len(history)}")
    print(f"Total updates: {total_updates}")
    print(f"Margin alpha: {alpha:.4f}")
    print(f"Bound 1/alpha^2: {bound:.1f}")
    print(f"Test accuracy: {test_acc:.1%}")
    if misclassified:
        print(f"Misclassified words: {', '.join(misclassified)}")
    print("=" * 60)

    # Print top positive / negative scoring words from all data
    all_scored_words = train_words + test_words + probe_words
    all_scored_X = np.vstack([train_X, test_X, probe_X])
    all_scores = all_scored_X @ w
    order = np.argsort(all_scores)

    print("\nMost negative-scoring words:")
    for i in order[:10]:
        print(f"  {all_scored_words[i]:15s} {all_scores[i]:+.4f}")
    print("\nMost positive-scoring words:")
    for i in order[-10:][::-1]:
        print(f"  {all_scored_words[i]:15s} {all_scores[i]:+.4f}")

    print("\nNeutral probe words:")
    for w_ in probe_words:
        idx = all_scored_words.index(w_)
        print(f"  {w_:15s} {all_scores[idx]:+.4f}")


if __name__ == "__main__":
    main()
