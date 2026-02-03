# Text Encoding and Tokenization

Before text can be processed by a language model, it must be converted from human-readable characters into numerical representations. This article covers the foundational layers of this transformation: character encoding (how characters become bytes), text corpora (what data models learn from), and tokenization (how text is segmented into units for modeling).

---

## Character Encoding: From Characters to Bytes

### ASCII: The Beginning

**ASCII** (American Standard Code for Information Interchange), developed in the 1960s, was the first widely adopted character encoding. It uses 7 bits to represent 128 characters:

- 0–31: Control characters (newline, tab, carriage return, etc.)
- 32–126: Printable characters (letters, digits, punctuation)
- 127: Delete

| Decimal | Hex | Character |
|---------|-----|-----------|
| 32 | 0x20 | (space) |
| 48–57 | 0x30–0x39 | 0–9 |
| 65–90 | 0x41–0x5A | A–Z |
| 97–122 | 0x61–0x7A | a–z |

ASCII works well for English text but cannot represent accented characters (é, ñ), non-Latin scripts (中文, العربية, हिन्दी), or symbols like € or emoji.

The 8th bit in a byte was initially unused, leading to various incompatible **extended ASCII** encodings (ISO-8859-1 for Western European, ISO-8859-5 for Cyrillic, etc.). This fragmentation made international text exchange problematic.

### Unicode: A Universal Character Set

**Unicode** aims to assign a unique number (called a **code point**) to every character in every writing system. Code points are written as U+XXXX in hexadecimal.

| Code Point | Character | Name |
|------------|-----------|------|
| U+0041 | A | Latin Capital Letter A |
| U+00E9 | é | Latin Small Letter E with Acute |
| U+4E2D | 中 | CJK Unified Ideograph (zhōng) |
| U+1F600 | 😀 | Grinning Face |

Unicode currently defines over 150,000 characters across 161 scripts. The code point space extends to U+10FFFF (over 1.1 million possible code points), organized into 17 **planes** of 65,536 code points each:

- **Plane 0** (U+0000–U+FFFF): Basic Multilingual Plane (BMP) — most common characters
- **Plane 1** (U+10000–U+1FFFF): Supplementary Multilingual Plane — emoji, historic scripts
- **Plane 2** (U+20000–U+2FFFF): Supplementary Ideographic Plane — rare CJK characters

Unicode defines *what* characters exist and their code points. The *encoding* determines how code points are represented as bytes.

---

## Unicode Encodings: UTF-8, UTF-16, UTF-32

### UTF-32: Fixed Width

**UTF-32** uses exactly 4 bytes (32 bits) per character. Simple but wasteful—ASCII text uses 4x the space it needs.

```
'A' (U+0041)  → 00 00 00 41
'中' (U+4E2D) → 00 00 4E 2D
'😀' (U+1F600) → 00 01 F6 00
```

Advantages: Random access (character $n$ is at byte $4n$), simple implementation.

Disadvantages: Inefficient for ASCII-heavy text, not backward compatible with ASCII.

### UTF-16: Variable Width (2 or 4 bytes)

**UTF-16** uses 2 bytes for BMP characters (U+0000–U+FFFF) and 4 bytes for characters outside the BMP (using **surrogate pairs**).

```
'A' (U+0041)  → 00 41
'中' (U+4E2D) → 4E 2D
'😀' (U+1F600) → D8 3D DE 00  (surrogate pair)
```

UTF-16 was designed when Unicode was expected to fit in 16 bits. It's used internally by Java, JavaScript, and Windows, but the variable width complicates string operations.

### UTF-8: The Modern Standard

**UTF-8** is a variable-width encoding using 1–4 bytes per character, designed for backward compatibility with ASCII.

| Code Point Range | Bytes | Byte Pattern |
|------------------|-------|--------------|
| U+0000–U+007F | 1 | `0xxxxxxx` |
| U+0080–U+07FF | 2 | `110xxxxx 10xxxxxx` |
| U+0800–U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
| U+10000–U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

The leading bits indicate the byte's role:

- `0xxxxxxx`: Single-byte character (ASCII)
- `110xxxxx`: First byte of 2-byte sequence
- `1110xxxx`: First byte of 3-byte sequence
- `11110xxx`: First byte of 4-byte sequence
- `10xxxxxx`: Continuation byte

**Example: Encoding 'é' (U+00E9)**

Binary of 0xE9 = 11101001 (needs 8 bits, so 2-byte encoding)

Template: `110xxxxx 10xxxxxx`

Fill in bits: `110` + `00011` + `10` + `101001` = `11000011 10101001` = `C3 A9`

```
'é' (U+00E9) → C3 A9
```

**Example: Encoding '中' (U+4E2D)**

Binary of 0x4E2D = 0100 111000 101101 (needs 16 bits, so 3-byte encoding)

Template: `1110xxxx 10xxxxxx 10xxxxxx`

Fill in: `1110` + `0100` + `10` + `111000` + `10` + `101101` = `11100100 10111000 10101101` = `E4 B8 AD`

```
'中' (U+4E2D) → E4 B8 AD
```

**Example: Encoding '😀' (U+1F600)**

Binary of 0x1F600 = 0 00011111 01100000 0000 (needs 21 bits, so 4-byte encoding)

Template: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`

```
'😀' (U+1F600) → F0 9F 98 80
```

### Why UTF-8 Dominates

UTF-8 has become the dominant encoding on the web (over 98% of websites) for several reasons:

- **ASCII compatibility**: ASCII files are valid UTF-8
- **Efficiency**: Compact for Latin text, reasonable for other scripts
- **Self-synchronizing**: Can find character boundaries by examining any byte
- **No byte-order issues**: Unlike UTF-16/32, no need for byte-order marks (BOM)
- **Robustness**: Invalid sequences are detectable

For NLP, UTF-8 means that a "byte-level" model sees ASCII characters as single bytes but non-ASCII characters as multi-byte sequences. This has implications for tokenization strategies.

---

## Text Corpora for Language Modeling

### Classical N-gram Corpora

Before deep learning, statistical language models were trained on carefully curated corpora:

**Brown Corpus (1961)** — The first major machine-readable corpus of English. Contains about 1 million words from 500 texts across 15 genres (news, fiction, academic, etc.). Groundbreaking for its time, now tiny by modern standards.

**Penn Treebank (1993)** — Approximately 1 million words of Wall Street Journal text with syntactic annotations. Standard benchmark for parsing and language modeling through the 2000s.

**British National Corpus (1994)** — 100 million words of British English from diverse sources including spoken transcripts. Larger scale but still modest.

**Google N-grams (2006)** — N-gram counts from over 1 trillion words of web text. Released n-gram frequency tables rather than raw text. Enabled research on very large scale statistics.

**1 Billion Word Benchmark (2013)** — Approximately 1 billion words of news text, preprocessed and shuffled. Standard benchmark for neural language models in the mid-2010s.

These corpora enabled research on n-gram models, which estimate:
\[
P(w_n \mid w_1, \ldots, w_{n-1}) \approx P(w_n \mid w_{n-k}, \ldots, w_{n-1})
\]
using counts from the training corpus with smoothing techniques (Kneser-Ney, Good-Turing) to handle unseen n-grams.

### Modern LLM Training Data

Contemporary large language models train on vastly larger and more diverse data:

**Common Crawl** — Petabytes of web pages crawled since 2008. Raw crawls require extensive filtering and deduplication. Forms the base of many training datasets.

**The Pile (2020)** — 800GB diverse dataset combining 22 sources: books, Wikipedia, GitHub, arXiv, Stack Exchange, patents, and more. Designed for research reproducibility.

**C4 (Colossal Clean Crawled Corpus)** — ~750GB of cleaned English web text filtered from Common Crawl. Used to train T5.

**RefinedWeb (2023)** — Multi-trillion token dataset created through aggressive filtering and deduplication of Common Crawl. Used for Falcon models.

**RedPajama (2023)** — Open reproduction of the LLaMA training dataset: 1.2 trillion tokens from Common Crawl, C4, GitHub, Wikipedia, books, arXiv, and Stack Exchange.

Modern training corpora are 1,000–10,000x larger than classical corpora, with GPT-4 class models reportedly trained on over 10 trillion tokens. The shift from curated corpora to filtered web crawls fundamentally changed the data landscape.

---

## Tokenization

### Why Tokenize?

Language models need discrete input units. The choice of units involves tradeoffs:

**Character-level**: Vocabulary of ~100–300 characters. Very long sequences (a word becomes 5+ tokens). Struggles to learn word-level patterns.

**Word-level**: Natural linguistic units. Huge vocabulary (100K+ for good coverage). Cannot handle misspellings, neologisms, or morphologically rich languages. What about "don't"? "New York"?

**Subword-level**: Compromise between characters and words. Vocabulary of 30K–100K tokens. Common words are single tokens; rare words split into pieces. Handles unknown words gracefully.

Modern LLMs universally use subword tokenization, with **Byte-Pair Encoding (BPE)** and its variants being the dominant approach.

---

## Byte-Pair Encoding (BPE)

### The Algorithm

BPE was originally a data compression algorithm (Gage, 1994), adapted for NLP tokenization by Sennrich et al. (2016).

**Training (vocabulary construction):**

1. Start with a base vocabulary of all individual characters (or bytes)
2. Count all adjacent pairs of tokens in the corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until reaching desired vocabulary size

**Inference (tokenization):**

Apply learned merges in order of frequency to new text.

### Detailed Example

**Corpus**: "low low low low low lowest lowest newer newer newer wider wider wider"

**Step 0**: Split into characters (with end-of-word marker `_`)

```
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _}

Tokens:
  l o w _ (×5)
  l o w e s t _ (×2)
  n e w e r _ (×3)
  w i d e r _ (×3)
```

**Step 1**: Count pairs, find most frequent

| Pair | Count |
|------|-------|
| l o | 7 |
| o w | 7 |
| w _ | 5 |
| e r | 6 |
| r _ | 6 |
| e s | 2 |
| ... | ... |

Most frequent: `l o` and `o w` (tied at 7). Pick `l o`.

Merge `l o` → `lo`

```
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _, lo}

Tokens:
  lo w _ (×5)
  lo w e s t _ (×2)
  n e w e r _ (×3)
  w i d e r _ (×3)
```

**Step 2**: Recount pairs

| Pair | Count |
|------|-------|
| lo w | 7 |
| w _ | 5 |
| e r | 6 |
| r _ | 6 |
| ... | ... |

Most frequent: `lo w` (7). Merge → `low`

```
Vocabulary: {..., lo, low}

Tokens:
  low _ (×5)
  low e s t _ (×2)
  n e w e r _ (×3)
  w i d e r _ (×3)
```

**Step 3**: Recount

| Pair | Count |
|------|-------|
| low _ | 5 |
| e r | 6 |
| r _ | 6 |
| low e | 2 |
| ... | ... |

Most frequent: `e r` (6). Merge → `er`

```
Vocabulary: {..., low, er}

Tokens:
  low _ (×5)
  low e s t _ (×2)
  n e w er _ (×3)
  w i d er _ (×3)
```

**Step 4**: Most frequent is now `er _` (6). Merge → `er_`

```
Tokens:
  low _ (×5)
  low e s t _ (×2)
  n e w er_ (×3)
  w i d er_ (×3)
```

**Continuing**: The algorithm continues, eventually learning merges like:

- `low _` → `low_`
- `low e` → `lowe`
- `lowe s` → `lowes`
- `lowes t` → `lowest`
- `n e` → `ne`
- `ne w` → `new`
- `new er_` → `newer_`
- etc.

After enough iterations, common words become single tokens while rare words remain segmented.

### Tokenizing New Text

Given the learned merge rules (in order), tokenize "lower":

1. Start: `l o w e r`
2. Apply merge `l o` → `lo`: `lo w e r`
3. Apply merge `lo w` → `low`: `low e r`
4. Apply merge `e r` → `er`: `low er`
5. No more applicable merges

Result: `["low", "er"]`

The word "lower" wasn't in training, but BPE segments it into meaningful pieces.

### Byte-Level BPE

GPT-2 introduced **byte-level BPE**: instead of starting with characters, start with the 256 possible byte values. This guarantees any text can be tokenized (no unknown characters) and handles UTF-8 naturally.

The base vocabulary is 256 bytes. Merges then create multi-byte tokens. A token might represent:

- A single ASCII character: `t` (byte 0x74)
- Multiple ASCII characters: `the` (bytes 0x74 0x68 0x65)
- Part of a UTF-8 sequence: the continuation bytes of a Chinese character

This elegantly handles multilingual text without explicit Unicode handling.

---

## Other Tokenization Methods

### WordPiece

**WordPiece** (used by BERT) is similar to BPE but selects merges differently. Instead of choosing the most frequent pair, it chooses the pair that maximizes the likelihood of the training data under a simple probabilistic criterion:
\[
\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}
\]

This favors merges that co-occur more often than expected under independence.

WordPiece uses a special prefix (##) to indicate continuation of a word:

```
"tokenization" →  ["token", "##ization"]
```

### Unigram Language Model

**Unigram** (as implemented in SentencePiece) works in the opposite direction from BPE and WordPiece. Instead of starting with characters and *merging*, it starts with a large vocabulary and *prunes* it.

- Start with a very large vocabulary of candidate tokens (all substrings up to some maximum length).
- Train a **unigram language model**, where each token \( t \) has a probability \( P(t) \), and the probability of a string is defined by summing over all possible segmentations (computed efficiently via dynamic programming):
\[
P(x) = \sum_{\text{segmentations } s} \prod_{t \in s} P(t)
\]
- For each token, estimate how much the **total corpus log-likelihood** would decrease if that token were removed from the vocabulary.
- Remove the tokens whose removal causes the *smallest* decrease in corpus likelihood (i.e., tokens that contribute the least).
- Repeat until the desired vocabulary size is reached.

Because tokenization is solved with dynamic programming (e.g., Viterbi) rather than greedy matching, **Unigram can find the globally optimal tokenization given a fixed vocabulary**, not just a left-to-right greedy one.

Intuitively, tokens must “earn their keep”: if removing a token barely hurts the model’s ability to explain the corpus, it is pruned as redundant.

### SentencePiece

**SentencePiece** is a tokenizer library that implements both BPE and Unigram with several practical features:

- Treats input as raw bytes or Unicode characters (language-independent)
- Includes whitespace as a normal symbol (no pre-tokenization required)
- Fully deterministic tokenization
- Widely used in modern models

It is used by T5, LLaMA, XLNet, and many multilingual models.

### Comparison

| Method | Core idea | Optimization style | Used by |
|------|----------|-------------------|--------|
| BPE | Merge frequent pairs | Greedy, local | GPT-2, GPT-3, GPT-4 |
| WordPiece | Merge by likelihood ratio | Greedy, local | BERT, DistilBERT |
| Unigram | Remove least useful tokens | Global likelihood | T5, ALBERT, XLNet |

All three produce broadly similar vocabularies in practice. The choice mainly reflects modeling philosophy (greedy vs probabilistic) and tooling rather than a strict quality hierarchy.

---

## Practical Considerations

### Vocabulary Size

Typical vocabulary sizes:

- GPT-2: 50,257 tokens
- BERT: 30,522 tokens (WordPiece)
- GPT-4: ~100,000 tokens
- LLaMA: 32,000 tokens

Larger vocabularies mean shorter sequences (fewer tokens per text) but more parameters in the embedding layer. There's a tradeoff between sequence length and model size.

### Tokenization Artifacts

Subword tokenization can create surprising behaviors:

**Inconsistent arithmetic**: "1000" and "1001" may tokenize differently, making arithmetic harder for models.

**Whitespace sensitivity**: " hello" (with leading space) and "hello" may produce different tokens.

**Language bias**: Tokenizers trained primarily on English may fragment other languages into many more tokens, reducing effective context length for non-English text.

**Prompt sensitivity**: Minor prompt changes can shift token boundaries, affecting model behavior.

### Special Tokens

Tokenizers include special tokens for model-specific purposes:

- `[PAD]`: Padding for batch processing
- `[UNK]`: Unknown token (rare with BPE)
- `[CLS]`: Classification token (BERT)
- `[SEP]`: Separator between segments
- `[BOS]`, `[EOS]`: Beginning/end of sequence
- `<|endoftext|>`: GPT's end-of-document marker

---

## Summary

Text processing for language models involves multiple encoding layers:

**Character encoding** converts characters to bytes. UTF-8 dominates, providing ASCII compatibility with universal Unicode coverage through variable-width encoding (1–4 bytes per character).

**Text corpora** have grown from millions of words (Brown Corpus) to trillions of tokens (modern LLM training sets), shifting from curated collections to filtered web crawls.

**Tokenization** segments text into model inputs. Subword methods like BPE balance vocabulary size against sequence length, handling rare words by decomposition while keeping common words intact. Starting from individual bytes or characters, BPE iteratively merges frequent pairs to build a vocabulary that efficiently represents the training distribution.

These layers form the foundation connecting raw text to the numerical inputs that language models process.
