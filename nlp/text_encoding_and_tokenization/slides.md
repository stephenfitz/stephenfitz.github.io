# Text Encoding and Tokenization

---

## Outline

- Character encoding: ASCII and Unicode
- Unicode encodings: UTF-8, UTF-16, UTF-32
- Text corpora: classical and modern
- Tokenization strategies
- Byte-Pair Encoding in detail
- Other tokenization methods
- Practical considerations

---

## Part I: Character Encoding

---

## The Problem

Text must be converted to numbers for computation

Characters → Bytes → Model inputs

How do we represent "A", "é", "中", "😀" as bytes?

---

## ASCII (1960s)

**American Standard Code for Information Interchange**

- 7 bits → 128 characters
- 0–31: Control characters (newline, tab)
- 32–126: Printable characters
- 127: Delete

---

## ASCII Table (Excerpt)

| Decimal | Hex | Character |
|---------|-----|-----------|
| 32 | 0x20 | (space) |
| 48–57 | 0x30–0x39 | 0–9 |
| 65–90 | 0x41–0x5A | A–Z |
| 97–122 | 0x61–0x7A | a–z |

---

## ASCII Limitations

Works well for English, but cannot represent:

- Accented characters: é, ñ, ü
- Non-Latin scripts: 中文, العربية, हिन्दी
- Symbols: €, £, ¥
- Emoji: 😀, 🎉, 🚀

---

## Extended ASCII Problem

The unused 8th bit led to incompatible encodings:

- ISO-8859-1 for Western European
- ISO-8859-5 for Cyrillic
- Many others...

Same byte → different characters in different encodings

---

## Unicode: The Solution

Assign a unique **code point** to every character in every writing system

Written as U+XXXX in hexadecimal

---

## Unicode Examples

| Code Point | Character | Name |
|------------|-----------|------|
| U+0041 | A | Latin Capital Letter A |
| U+00E9 | é | Latin Small E with Acute |
| U+4E2D | 中 | CJK Unified Ideograph |
| U+1F600 | 😀 | Grinning Face |

---

## Unicode Scale

- Over 150,000 characters defined
- 161 scripts supported
- Code points up to U+10FFFF (1.1 million possible)
- Organized into 17 **planes** of 65,536 each

---

## Unicode Planes

- **Plane 0** (BMP): Most common characters
- **Plane 1**: Emoji, historic scripts
- **Plane 2**: Rare CJK characters

Unicode defines *what* characters exist

**Encoding** determines *how* to represent them as bytes

---

## Part II: Unicode Encodings

---

## UTF-32: Fixed Width

4 bytes (32 bits) per character

```
'A' (U+0041)  → 00 00 00 41
'中' (U+4E2D) → 00 00 4E 2D
'😀' (U+1F600) → 00 01 F6 00
```

Simple but wasteful — ASCII uses 4x needed space

---

## UTF-16: Variable Width

2 bytes for BMP, 4 bytes for others (surrogate pairs)

```
'A' (U+0041)  → 00 41
'中' (U+4E2D) → 4E 2D
'😀' (U+1F600) → D8 3D DE 00
```

Used by Java, JavaScript, Windows

---

## UTF-8: The Modern Standard

Variable-width: 1–4 bytes per character

Backward compatible with ASCII

---

## UTF-8 Encoding Scheme

| Code Point Range | Bytes | Pattern |
|------------------|-------|---------|
| U+0000–U+007F | 1 | `0xxxxxxx` |
| U+0080–U+07FF | 2 | `110xxxxx 10xxxxxx` |
| U+0800–U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
| U+10000–U+10FFFF | 4 | `11110xxx 10xxxxxx ...` |

---

## UTF-8 Leading Bits

- `0xxxxxxx`: Single-byte (ASCII)
- `110xxxxx`: First of 2-byte sequence
- `1110xxxx`: First of 3-byte sequence
- `11110xxx`: First of 4-byte sequence
- `10xxxxxx`: Continuation byte

---

## UTF-8 Example: 'é' (U+00E9)

Binary of 0xE9 = 11101001 (needs 8 bits → 2 bytes)

Template: `110xxxxx 10xxxxxx`

Fill in: `110`**00011** `10`**101001**

Result: `C3 A9`

---

## UTF-8 Example: '中' (U+4E2D)

Binary = 0100 111000 101101 (16 bits → 3 bytes)

Template: `1110xxxx 10xxxxxx 10xxxxxx`

Result: `E4 B8 AD`

---

## UTF-8 Example: '😀' (U+1F600)

Needs 21 bits → 4 bytes

Template: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`

Result: `F0 9F 98 80`

---

## Why UTF-8 Dominates

- **ASCII compatible**: ASCII files are valid UTF-8
- **Efficient**: Compact for Latin text
- **Self-synchronizing**: Can find boundaries anywhere
- **No byte-order issues**: No BOM needed
- **98%+ of websites** use UTF-8

---

## UTF-8 and NLP

Byte-level models see:

- ASCII characters as single bytes
- Non-ASCII as multi-byte sequences

Implications for tokenization strategies

---

## Part III: Text Corpora

---

## Classical N-gram Corpora

Before deep learning: carefully curated datasets

| Corpus | Year | Size |
|--------|------|------|
| Brown | 1961 | 1M words |
| Penn Treebank | 1993 | 1M words |
| British National | 1994 | 100M words |
| 1B Word Benchmark | 2013 | 1B words |

---

## Google N-grams (2006)

- Over 1 trillion words of web text
- Released frequency tables, not raw text
- Enabled large-scale statistical research

---

## Modern LLM Training Data

| Dataset | Size | Sources |
|---------|------|---------|
| Common Crawl | Petabytes | Web pages since 2008 |
| The Pile | 800GB | 22 diverse sources |
| C4 | 750GB | Filtered Common Crawl |
| RefinedWeb | Trillions | Deduplicated web |
| RedPajama | 1.2T tokens | LLaMA reproduction |

---

## Scale Comparison

- Classical corpora: millions of words
- Modern corpora: **trillions of tokens**
- 1,000–10,000x larger
- Shift from curated to filtered web crawls

---

## Part IV: Tokenization

---

## Why Tokenize?

Language models need discrete input units

Choice of units involves tradeoffs

---

## Character-Level

- Vocabulary: ~100–300 characters
- Very long sequences (word → 5+ tokens)
- Struggles to learn word-level patterns
- No unknown characters

---

## Word-Level

- Natural linguistic units
- Huge vocabulary (100K+ for coverage)
- Cannot handle: misspellings, neologisms
- What about "don't"? "New York"?

---

## Subword-Level

- Compromise between characters and words
- Vocabulary: 30K–100K tokens
- Common words: single tokens
- Rare words: split into pieces
- Handles unknown words gracefully

---

## Modern LLMs Use Subword

**Byte-Pair Encoding (BPE)** and variants dominate

- GPT family: BPE
- BERT: WordPiece
- T5, LLaMA: SentencePiece

---

## Part V: Byte-Pair Encoding

---

## BPE Origins

Originally a data compression algorithm (Gage, 1994)

Adapted for NLP tokenization (Sennrich et al., 2016)

---

## BPE Training Algorithm

1. Start with vocabulary of all characters (or bytes)
2. Count all adjacent token pairs
3. Merge most frequent pair into new token
4. Repeat until desired vocabulary size

---

## BPE Example: Setup

**Corpus**: "low low low low low lowest lowest newer newer newer wider wider wider"

**Initial vocabulary**: {l, o, w, e, s, t, n, r, i, d, \_}

---

## BPE Step 0: Character Split

```
l o w _ (×5)
l o w e s t _ (×2)
n e w e r _ (×3)
w i d e r _ (×3)
```

---

## BPE Step 1: Count Pairs

| Pair | Count |
|------|-------|
| l o | 7 |
| o w | 7 |
| e r | 6 |
| r _ | 6 |
| w _ | 5 |

Most frequent: `l o` → merge to `lo`

---

## BPE Step 2: After First Merge

```
lo w _ (×5)
lo w e s t _ (×2)
n e w e r _ (×3)
w i d e r _ (×3)
```

Now `lo w` has count 7 → merge to `low`

---

## BPE Continues...

- `e r` → `er`
- `er _` → `er_`
- `low _` → `low_`
- `low e` → `lowe`
- Eventually: `lowest_`, `newer_`, `wider_`

---

## BPE Result

Common words become single tokens

Rare words remain segmented

Vocabulary grows with each merge

---

## Tokenizing New Text

Given learned merges, tokenize "lower":

1. `l o w e r`
2. Apply `l o` → `lo`: `lo w e r`
3. Apply `lo w` → `low`: `low e r`
4. Apply `e r` → `er`: `low er`

Result: `["low", "er"]`

---

## Handling Unseen Words

"lower" wasn't in training corpus

BPE segments it into meaningful pieces: `low` + `er`

Graceful degradation for rare/new words

---

## Byte-Level BPE

GPT-2 innovation: start with 256 byte values, not characters

- Base vocabulary: 256 bytes
- Guarantees any text can be tokenized
- Handles UTF-8 naturally
- No unknown tokens possible

---

## Part VI: Other Methods

---

## WordPiece (BERT)

Similar to BPE, different merge criterion:

$$\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}$$

Uses `##` prefix for continuations:

`"tokenization"` → `["token", "##ization"]`

---

## Unigram Language Model

Opposite direction from BPE:

1. Start with large vocabulary (all substrings)
2. Train unigram LM: $P(x) = \prod_i P(t_i)$
3. Remove tokens that least hurt likelihood
4. Repeat until target size

Can find globally optimal tokenization

---

## SentencePiece

Library implementing BPE and Unigram

- Treats input as raw bytes
- Includes whitespace in tokens
- Language-independent
- Used by T5, LLaMA, multilingual models

---

## Method Comparison

| Method | Criterion | Used by |
|--------|-----------|---------|
| BPE | Most frequent pair | GPT-2/3/4 |
| WordPiece | Likelihood ratio | BERT |
| Unigram | Remove least useful | T5, XLNet |

Similar results in practice

---

## Part VII: Practical Considerations

---

## Vocabulary Sizes

| Model | Vocabulary |
|-------|------------|
| GPT-2 | 50,257 |
| BERT | 30,522 |
| GPT-4 | ~100,000 |
| LLaMA | 32,000 |

Larger vocab → shorter sequences, more parameters

---

## Tokenization Artifacts

**Inconsistent arithmetic**: "1000" vs "1001" tokenize differently

**Whitespace sensitivity**: " hello" ≠ "hello"

**Language bias**: Non-English fragments into more tokens

**Prompt sensitivity**: Small changes shift boundaries

---

## Special Tokens

- `[PAD]`: Padding for batches
- `[UNK]`: Unknown (rare with BPE)
- `[CLS]`, `[SEP]`: BERT special tokens
- `[BOS]`, `[EOS]`: Sequence boundaries
- `<|endoftext|>`: GPT end marker

---

## Summary

**Character encoding**: UTF-8 dominates, 1–4 bytes per character, ASCII compatible

**Corpora**: Grown from millions to trillions of tokens

**Tokenization**: Subword methods balance vocabulary size vs. sequence length

**BPE**: Iteratively merge frequent pairs; handles rare words by decomposition
