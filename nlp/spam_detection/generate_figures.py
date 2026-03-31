#!/usr/bin/env python3
"""
Generate figures for the SMS spam detection article.

Trains a sigmoid neuron (logistic regression) to classify SMS messages as spam
or ham using binary bag-of-words features, producing training curves, weight
analysis, and probability distribution figures.

Usage:
    python generate_figures.py
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Data: 120 SMS messages (60 spam, 60 ham)
# ---------------------------------------------------------------------------

SPAM_MESSAGES = [
    "Congratulations you have won a free prize call now to claim your reward",
    "WINNER you have been selected for a cash prize call this number now",
    "Free entry to win a brand new car text WIN to claim your prize today",
    "Urgent your account has been compromised click here to verify now",
    "You have won a free trip to the Bahamas reply now to claim",
    "Special offer buy one get one free limited time only call now",
    "Free ringtone download text TONE to get yours now no charge",
    "Congratulations your number has been selected you won ten thousand pounds",
    "Claim your free gift card worth five hundred dollars call today",
    "You are a lucky winner text back to claim your cash prize",
    "Hot deal get a free mobile phone upgrade call us right now",
    "URGENT your bank account needs verification click the link immediately",
    "Win big cash prizes every week enter free text PLAY to start",
    "Free credit check your score now no obligation just call today",
    "You have been chosen to receive a free vacation package reply YES",
    "Amazing offer free trial membership call now limited spots available",
    "Your prize is waiting collect your winnings call this number today",
    "Free laptop offer for selected customers only claim yours now",
    "Congratulations you qualified for a special cash bonus reply to claim",
    "WINNER WINNER call now to collect your prize worth thousands",
    "Get your free sample delivered to your door text FREE today",
    "Exclusive deal just for you call now for a free consultation",
    "You won a free shopping spree at your favorite store claim now",
    "Limited offer get free unlimited texts and calls upgrade today",
    "Your winning notification claim your prize money before it expires",
    "Free membership trial no credit card required sign up now",
    "Congratulations your entry won the grand prize call to collect",
    "Special promotion free gift with every purchase order now",
    "You have a package waiting pay a small fee to claim your prize",
    "Free tickets to the concert of your choice reply WIN now",
    "Alert your account will be suspended verify your details now",
    "Double your money guaranteed investment opportunity call today",
    "Free insurance quote save hundreds per year call this number",
    "You are selected for an exclusive free offer reply to accept",
    "Win a dream holiday for two enter free just text HOLIDAY",
    "Congratulations claim your reward points worth cash call now",
    "Free phone case with every order limited stock available today",
    "URGENT action required verify your account to avoid suspension",
    "Get rich quick with this amazing opportunity call for free info",
    "Your lucky number won big claim your cash winnings today",
    "Free beauty products sample box delivered to you text BEAUTY",
    "Exclusive winner notification you won a brand new television",
    "Special discount offer save up to ninety percent off call now",
    "Free dating service find your match text LOVE to join now",
    "You qualified for a government grant claim your free money today",
    "Congratulations your application was approved call for your prize",
    "Free energy saving tips that could save you thousands click here",
    "Winner announcement your ticket number matched call to claim now",
    "Get a free quote on car insurance save big money today",
    "Your reward is ready collect your free gift at this location",
    "Free weight loss pills that actually work order yours today",
    "ALERT unusual activity on your account verify immediately",
    "Win free cinema tickets for a year text CINEMA to enter",
    "Congratulations you won a luxury cruise trip claim your ticket now",
    "Free trial offer cancel anytime no questions asked start today",
    "Exclusive prize draw winner you won five thousand in cash",
    "Get your free coupon book worth hundreds in savings call now",
    "Your account needs urgent attention click here to update details",
    "Free smartphone giveaway enter to win text PHONE right now",
    "Special winner selected for our monthly prize draw claim today",
]

HAM_MESSAGES = [
    "Hey are you coming to the meeting tomorrow morning at nine",
    "Can you pick up some milk and bread on your way home please",
    "Thanks for dinner last night it was really great seeing you",
    "I will be running about ten minutes late to lunch sorry",
    "Did you finish the report that was due yesterday afternoon",
    "Happy birthday hope you have an amazing day with your family",
    "Can we reschedule our appointment to next Thursday instead",
    "Just wanted to check if you are feeling better today",
    "The kids had a great time at the park this afternoon",
    "What time does the movie start tonight I need to plan",
    "Please remember to bring your notebook to class tomorrow",
    "I left my jacket at your place can I grab it later",
    "Are we still on for coffee this Saturday at the usual spot",
    "The weather looks nice this weekend want to go hiking",
    "Mom said she will visit us next month for about a week",
    "Can you send me the address for the restaurant tonight",
    "Just finished reading that book you recommended it was good",
    "Do you need a ride to the airport tomorrow morning early",
    "Sorry I missed your call earlier was in a meeting all day",
    "Let me know when you get home so I know you are safe",
    "The doctor appointment is confirmed for Monday at two thirty",
    "Hey have you seen my keys I think I left them somewhere",
    "Want to grab dinner after work today I know a good place",
    "Thanks for helping me move last weekend really appreciate it",
    "Can you water my plants while I am away this week please",
    "The homework is due on Friday make sure to submit on time",
    "I am at the grocery store do you need anything else",
    "How was your job interview this morning hope it went well",
    "Remember we have that dentist appointment next Tuesday morning",
    "Just saw your message sorry for the late reply was busy",
    "The train is delayed about twenty minutes running behind today",
    "Can you help me study for the exam this Thursday evening",
    "I made pasta for dinner come eat whenever you are ready",
    "Are you going to the gym tonight want to go together",
    "Please pick up the dry cleaning on your way back today",
    "The landlord said they will fix the heater by tomorrow",
    "Did you hear about the new restaurant that opened downtown",
    "I will be working from home today if you need me just call",
    "Want to join us for game night this Friday at our place",
    "The baby slept through the whole night finally some rest",
    "Can you proofread my essay before I submit it tomorrow",
    "Let me know your schedule this week so we can plan lunch",
    "Hey just checking in how are things going with the project",
    "Thanks for the birthday wishes means a lot to me really",
    "I picked up your prescription from the pharmacy earlier today",
    "The parking lot was full so I had to park down the street",
    "Want to go for a walk after dinner the weather is nice",
    "Reminder parent teacher conference is Thursday at four thirty",
    "I will bring the dessert for the party on Saturday night",
    "How is your day going hope everything is well at work",
    "Can you feed the dog before you leave for work this morning",
    "The lecture was really interesting today learned a lot",
    "I am running some errands will be back in about an hour",
    "Thanks for lending me your notes they were really helpful",
    "Do you have plans this Sunday afternoon want to hang out",
    "The flight arrives at eight thirty can you pick me up then",
    "Just wanted to say good luck on your presentation today",
    "I found a great recipe for soup want to try it tonight",
    "Hey can you send me that photo you took at the party",
    "The meeting got moved to Wednesday same time same room though",
]

# ---------------------------------------------------------------------------
# Style constants (matching perceptron_sentiment conventions)
# ---------------------------------------------------------------------------

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SPAM_COLOR = "#dc2626"    # red for spam
HAM_COLOR = "#2563eb"     # blue for ham
NEUTRAL_COLOR = "#6b7280" # gray
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


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "it", "of", "to", "in", "and", "or",
    "for", "on", "at", "by", "with", "that", "this", "was", "are",
    "be", "has", "had", "have", "do", "does", "did", "but", "not",
    "so", "if", "its", "my", "me", "we", "he", "she", "us",
    "i", "am", "been", "being", "from", "as", "no",
}


def tokenize(text):
    """Lowercase and extract alphanumeric tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())


def build_vocabulary(messages, min_freq=2):
    """Build vocabulary from messages, keeping words with freq >= min_freq."""
    freq = {}
    for msg in messages:
        for token in tokenize(msg):
            if token not in STOPWORDS:
                freq[token] = freq.get(token, 0) + 1
    vocab = sorted([w for w, c in freq.items() if c >= min_freq])
    return vocab


def featurize(messages, vocab):
    """Convert messages to binary bag-of-words feature matrix."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((len(messages), V + 1))  # +1 for bias
    for i, msg in enumerate(messages):
        tokens = set(tokenize(msg))
        for token in tokens:
            if token in word_to_idx:
                X[i, word_to_idx[token]] = 1.0
        X[i, -1] = 1.0  # bias term
    return X


# ---------------------------------------------------------------------------
# Sigmoid neuron
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_loss(X, y, w, eps=1e-15):
    """Binary cross-entropy loss: -1/m sum[ y log(yhat) + (1-y) log(1-yhat) ]."""
    m = len(y)
    y_hat = sigmoid(X @ w)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def train_sigmoid(X, y, lr=0.1, epochs=200):
    """
    Train sigmoid neuron with batch gradient descent.

    Returns: (w, loss_history)
    """
    m, d = X.shape
    w = np.zeros(d)
    loss_history = []

    for epoch in range(epochs):
        y_hat = sigmoid(X @ w)
        gradient = X.T @ (y_hat - y) / m
        w = w - lr * gradient
        loss = cross_entropy_loss(X, y, w)
        loss_history.append(loss)

    return w, loss_history


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_loss_curve(loss_history, fig_path):
    """Plot cross-entropy loss vs epoch."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color=SPAM_COLOR, linewidth=1.5)
    ax.axhline(y=np.log(2), color=NEUTRAL_COLOR, linestyle="--", linewidth=1,
               alpha=0.7, label=r"$\log 2 \approx 0.693$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_probability_histogram(y_test, probs_test, fig_path):
    """Plot predicted probability distributions for spam vs ham."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    spam_probs = probs_test[y_test == 1]
    ham_probs = probs_test[y_test == 0]

    bins = np.linspace(0, 1, 25)
    ax.hist(ham_probs, bins=bins, color=HAM_COLOR, alpha=0.6,
            label="Ham", edgecolor="white")
    ax.hist(spam_probs, bins=bins, color=SPAM_COLOR, alpha=0.6,
            label="Spam", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1,
               alpha=0.7, label=r"$\tau = 0.5$")
    ax.set_xlabel(r"Predicted Probability $\sigma(w^\top x)$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $P(\mathrm{spam} \mid \mathrm{message})$")
    ax.legend(loc="upper center", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_weight_bar_chart(w, vocab, fig_path, top_k=15):
    """Bar chart of top spam-indicator and ham-indicator words."""
    # Exclude bias (last element)
    w_words = w[:-1]
    indices = np.argsort(w_words)

    # Top spam words (largest positive weights)
    spam_idx = indices[-top_k:][::-1]
    # Top ham words (largest negative weights)
    ham_idx = indices[:top_k]

    combined_idx = list(spam_idx) + list(ham_idx)
    combined_words = [vocab[i] for i in combined_idx]
    combined_weights = [w_words[i] for i in combined_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [SPAM_COLOR if wt > 0 else HAM_COLOR for wt in combined_weights]
    y_pos = range(len(combined_words))
    ax.barh(y_pos, combined_weights, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_words, fontsize=9)
    ax.set_xlabel("Weight $w_j$")
    ax.set_title("Learned Weights: Top Spam and Ham Indicators")
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], color=SPAM_COLOR, lw=6, label="Spam indicator ($w_j > 0$)"),
        Line2D([0], [0], color=HAM_COLOR, lw=6, label="Ham indicator ($w_j < 0$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_odds_multiplier(w, vocab, fig_path, top_k=12):
    """Bar chart of e^{w_j} odds multipliers for top words."""
    w_words = w[:-1]
    indices = np.argsort(np.abs(w_words))[-top_k:][::-1]

    words = [vocab[i] for i in indices]
    odds = [np.exp(w_words[i]) for i in indices]
    weights = [w_words[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [SPAM_COLOR if wt > 0 else HAM_COLOR for wt in weights]
    y_pos = range(len(words))
    ax.barh(y_pos, odds, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel(r"Odds Multiplier $e^{w_j}$")
    ax.set_title("How Each Word Multiplies the Spam Odds")
    ax.invert_yaxis()
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="Neutral (1×)")

    # Add value labels
    for i, (od, wt) in enumerate(zip(odds, weights)):
        label = f"{od:.1f}×" if od >= 1 else f"{od:.2f}×"
        ax.text(od + 0.05, i, label, va="center", fontsize=8)

    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_confidence_examples(messages, labels, probs, fig_path, n=10):
    """Show example messages with their predicted probabilities."""
    # Select a mix: some confident spam, some confident ham, some near boundary
    spam_mask = labels == 1
    ham_mask = labels == 0

    # Sort by probability
    spam_order = np.argsort(probs[spam_mask])[::-1]
    ham_order = np.argsort(probs[ham_mask])

    spam_msgs = [m for m, s in zip(messages, labels) if s == 1]
    ham_msgs = [m for m, s in zip(messages, labels) if s == 0]
    spam_probs = probs[spam_mask]
    ham_probs = probs[ham_mask]

    # Pick examples
    examples = []
    # Top confident spam
    for i in spam_order[:3]:
        examples.append((spam_msgs[i], spam_probs[i], "Spam"))
    # Least confident spam (near boundary)
    for i in spam_order[-2:]:
        examples.append((spam_msgs[i], spam_probs[i], "Spam"))
    # Top confident ham
    for i in ham_order[:3]:
        examples.append((ham_msgs[i], ham_probs[i], "Ham"))
    # Least confident ham
    for i in ham_order[-2:]:
        examples.append((ham_msgs[i], ham_probs[i], "Ham"))

    # Sort by probability descending
    examples.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(examples) + 0.5)
    ax.set_title("Example Messages with Predicted Probabilities", fontsize=13, pad=15)

    for i, (msg, prob, true_label) in enumerate(examples):
        y = len(examples) - i - 0.3
        truncated = msg[:65] + ("..." if len(msg) > 65 else "")
        color = SPAM_COLOR if prob > 0.5 else HAM_COLOR
        correct = (prob > 0.5 and true_label == "Spam") or (prob <= 0.5 and true_label == "Ham")
        marker = "" if correct else "  ✗"

        ax.text(0.0, y, f"P(spam) = {prob:.3f}", fontsize=9, fontfamily="monospace",
                color=color, fontweight="bold", va="center")
        ax.text(0.18, y, truncated, fontsize=8.5, va="center", color="#333")
        ax.text(0.92, y, f"[{true_label}]{marker}", fontsize=8.5, va="center",
                color=SPAM_COLOR if true_label == "Spam" else HAM_COLOR,
                fontweight="bold")

    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_decision_threshold(y_test, probs_test, fig_path):
    """Plot precision, recall, and F1 vs decision threshold."""
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions = []
    recalls = []
    f1s = []

    for tau in thresholds:
        pred = (probs_test >= tau).astype(int)
        tp = np.sum((pred == 1) & (y_test == 1))
        fp = np.sum((pred == 1) & (y_test == 0))
        fn = np.sum((pred == 0) & (y_test == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(thresholds, precisions, color=HAM_COLOR, linewidth=1.5, label="Precision")
    ax.plot(thresholds, recalls, color=SPAM_COLOR, linewidth=1.5, label="Recall")
    ax.plot(thresholds, f1s, color="#059669", linewidth=1.5, linestyle="--", label="F1")
    ax.axvline(0.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"Decision Threshold $\tau$")
    ax.set_ylabel("Score")
    ax.set_title(r"Precision, Recall, and F1 vs. Threshold $\tau$")
    ax.legend(loc="center left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_learning_rate_comparison(X_train, y_train, fig_path):
    """Plot loss curves for different learning rates."""
    lrs = [0.01, 0.1, 1.0, 5.0]
    colors = ["#6b7280", "#2563eb", "#059669", "#dc2626"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lr, color in zip(lrs, colors):
        _, loss_hist = train_sigmoid(X_train, y_train, lr=lr, epochs=200)
        ax.plot(range(1, len(loss_hist) + 1), loss_hist, color=color,
                linewidth=1.5, label=rf"$\eta = {lr}$")

    ax.axhline(y=np.log(2), color=NEUTRAL_COLOR, linestyle="--", linewidth=1,
               alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Effect of Learning Rate on Convergence")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()
    np.random.seed(42)

    # Prepare data
    all_messages = SPAM_MESSAGES + HAM_MESSAGES
    all_labels = np.array([1] * len(SPAM_MESSAGES) + [0] * len(HAM_MESSAGES))

    print(f"Total messages: {len(all_messages)} ({sum(all_labels == 1)} spam, {sum(all_labels == 0)} ham)")

    # Shuffle and split: 90 train (45/45), 30 test (15/15)
    spam_idx = np.where(all_labels == 1)[0]
    ham_idx = np.where(all_labels == 0)[0]
    np.random.shuffle(spam_idx)
    np.random.shuffle(ham_idx)

    train_idx = np.concatenate([spam_idx[:45], ham_idx[:45]])
    test_idx = np.concatenate([spam_idx[45:], ham_idx[45:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    train_messages = [all_messages[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    test_messages = [all_messages[i] for i in test_idx]
    test_labels = all_labels[test_idx]

    print(f"Train: {len(train_messages)} ({sum(train_labels == 1)} spam, {sum(train_labels == 0)} ham)")
    print(f"Test:  {len(test_messages)} ({sum(test_labels == 1)} spam, {sum(test_labels == 0)} ham)")

    # Build vocabulary from training set
    vocab = build_vocabulary(train_messages, min_freq=2)
    print(f"Vocabulary size: {len(vocab)} words")
    print(f"Sample vocabulary: {vocab[:20]}")

    # Featurize
    X_train = featurize(train_messages, vocab)
    X_test = featurize(test_messages, vocab)
    print(f"Feature matrix shape: {X_train.shape} (messages × features+bias)")

    # --- Train sigmoid neuron ---
    print("\nTraining sigmoid neuron ...")
    lr = 1.0
    epochs = 200
    w, loss_history = train_sigmoid(X_train, train_labels, lr=lr, epochs=epochs)

    print(f"Initial loss: {loss_history[0]:.4f} (log 2 = {np.log(2):.4f})")
    print(f"Final loss:   {loss_history[-1]:.4f}")

    # --- Evaluate ---
    probs_train = sigmoid(X_train @ w)
    preds_train = (probs_train >= 0.5).astype(int)
    train_acc = np.mean(preds_train == train_labels)

    probs_test = sigmoid(X_test @ w)
    preds_test = (probs_test >= 0.5).astype(int)
    test_acc = np.mean(preds_test == test_labels)

    print(f"\nTrain accuracy: {train_acc:.1%} ({int(train_acc * len(train_labels))}/{len(train_labels)})")
    print(f"Test accuracy:  {test_acc:.1%} ({int(test_acc * len(test_labels))}/{len(test_labels)})")

    # Misclassified test messages
    misclassified = [(test_messages[i], test_labels[i], probs_test[i])
                     for i in range(len(test_messages)) if preds_test[i] != test_labels[i]]
    if misclassified:
        print(f"\nMisclassified ({len(misclassified)}):")
        for msg, label, prob in misclassified:
            true = "spam" if label == 1 else "ham"
            print(f"  [{true}] P(spam)={prob:.3f}: {msg[:70]}...")

    # --- Generate figures ---
    print("\nGenerating figures ...")

    # 1. Loss curve
    plot_loss_curve(loss_history, os.path.join(FIG_DIR, "loss_curve.png"))

    # 2. Probability histogram
    plot_probability_histogram(test_labels, probs_test,
                               os.path.join(FIG_DIR, "probability_histogram.png"))

    # 3. Weight bar chart
    plot_weight_bar_chart(w, vocab, os.path.join(FIG_DIR, "weight_bar_chart.png"))

    # 4. Odds multiplier
    plot_odds_multiplier(w, vocab, os.path.join(FIG_DIR, "odds_multiplier.png"))

    # 5. Confidence examples
    plot_confidence_examples(test_messages, test_labels, probs_test,
                             os.path.join(FIG_DIR, "confidence_examples.png"))

    # 6. Decision threshold
    plot_decision_threshold(test_labels, probs_test,
                            os.path.join(FIG_DIR, "decision_threshold.png"))

    # 7. Learning rate comparison
    plot_learning_rate_comparison(X_train, train_labels,
                                  os.path.join(FIG_DIR, "learning_rate_comparison.png"))

    # --- Summary stats for the article ---
    print("\n" + "=" * 60)
    print("SUMMARY FOR ARTICLE")
    print("=" * 60)
    print(f"Messages: {len(SPAM_MESSAGES)} spam + {len(HAM_MESSAGES)} ham = {len(all_messages)} total")
    print(f"Split: {len(train_messages)} train / {len(test_messages)} test")
    print(f"Vocabulary: {len(vocab)} words (min_freq=2, stopwords removed)")
    print(f"Feature dimension: {X_train.shape[1]} ({len(vocab)} words + 1 bias)")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")

    # Top spam / ham weights
    w_words = w[:-1]
    top_spam = np.argsort(w_words)[-10:][::-1]
    top_ham = np.argsort(w_words)[:10]

    print(f"\nTop spam-indicator words (positive weights):")
    for i in top_spam:
        odds = np.exp(w_words[i])
        print(f"  {vocab[i]:15s} w={w_words[i]:+.3f}  odds_mult={odds:.1f}×")

    print(f"\nTop ham-indicator words (negative weights):")
    for i in top_ham:
        odds = np.exp(w_words[i])
        print(f"  {vocab[i]:15s} w={w_words[i]:+.3f}  odds_mult={odds:.2f}×")

    print(f"\nBias weight: {w[-1]:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
