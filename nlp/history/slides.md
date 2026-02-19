# Historical Foundations of Computing, AI, and Natural Language Processing

Background historical notes for a modern introduction to *Natural Language Processing*

---

## Outline

- **Part I:** A Brief History of Computing
- **Part II:** A Brief History of Artificial Intelligence
- **Part III:** A Brief History of Linguistics and NLP

---

## Part I: A Brief History of Computing

---

### Ancient Algorithms: Plimpton-322 (c. 1800 BCE)

![Plimpton-322 tablet](figures/plimpton_322_tablet.png)

---

- Babylonian clay tablet listing Pythagorean triples ($a^2 + b^2 = c^2$)
- Used for computing land areas for taxation
- Pre-computed output of a systematic generation algorithm
- Demonstrates algorithmic thinking millennia before modern computing

---

### The Euclidean Algorithm (c. 300 BCE)

- Computes the greatest common divisor (GCD) of two numbers
- Repeatedly replace the larger number by the remainder when divided by the smaller
- One of the oldest algorithms still in use today
- Lamé's analysis (1844) marked the beginning of **computational complexity theory**

---

### The Euclidean Algorithm (c. 300 BCE)

![Euclidean algorithm flowchart](figures/euclidean_algorithm_flowchart.png)

---

### Al-Khwarizmi (c. 780-850 CE)

- Persian mathematician at the House of Wisdom in Baghdad
- His name gives us the word **"algorithm"** (from Latinized "Algoritmi")
- His book *al-Jabr* gives us the word **"algebra"**
- Formalized solving equations with variables rather than specific numbers

---

### Al-Khwarizmi (c. 780-850 CE)

![Al-Khwarizmi](figures/al_khwarizmi.png)

---

### The Jacquard Loom (1801)

- **Earliest programmable machine** --- used punched cards to control weaving patterns
- Allowed ordinary workers to produce designs that previously required master weavers
- Led to one of the first instances of technological unemployment and industrial sabotage
- The punched card concept would prove far more consequential than Jacquard imagined

---

### The Jacquard Loom (1801)

![Jacquard loom punch cards](figures/jacquard_loom_punchcards.png)

---

### Babbage and Lovelace: The Analytical Engine

- **Charles Babbage** (1791-1871) designed a mechanical computer using punched cards
- **Ada Lovelace** (1815-1852) developed some of the first algorithms for a computing machine
- Invented foundational programming concepts:
    - Loops and conditional branching
    - Separation of data and operations
    - The concept of a general-purpose machine

---

### Babbage and Lovelace: The Analytical Engine

![Babbage's Analytical Engine](figures/babbage_analytical_engine.png)

---

### The Hollerith Tabulating Machine (1890)

- **Herman Hollerith** invented it to process the U.S. Census
- Punched cards sized to match dollar bills (reusing money-sorting equipment)
- The 80-column format is the origin of the **80-character line limit** in programming
- Hollerith's company later became **IBM**

---

### The Hollerith Tabulating Machine (1890)

![Hollerith Tabulating Machine](figures/hollerith_tabulating_machine.png)

---

### Alan Turing and the Foundations of Computing

- Introduced the **Turing machine** (1936) --- an abstract device that can simulate any algorithm
- **Church-Turing Thesis**: any algorithmic computation can be expressed as a Turing machine
- Proved some problems are **undecidable** (e.g., the Halting Problem)
- Established fundamental limits on what can be computed

---

### Alan Turing and the Foundations of Computing

![Alan Turing](figures/alan_turing.jpg)

---

### The Electronic Era: Wartime Computing

- World War II accelerated computer development dramatically
- Three drivers of innovation:
    - **Ballistics calculations**: artillery trajectories
    - **Code-breaking**: Turing's work at Bletchley Park on Enigma
    - **Nuclear weapons**: modeling for the Manhattan Project
- Led to first electronic computers including ENIAC (1945)

---

### The Mainframe Era

- Stored both data and programs electronically
- Key innovations:
    - **Subroutines** and the **stack** for procedural programming
    - **Machine language** and **assembly language**
    - **Batch processing**: programs submitted as punch card decks
- The first computer "bug" --- an actual moth found in the Harvard Mark II (1947)

---

### The Mainframe Era

![First computer bug](figures/first_computer_bug.png)

---

### From Terminals to Personal Computers

- **Time-sharing** (1960s): multiple users sharing one computer simultaneously
- **CRT terminals** replaced teletypes, enabling on-screen text editing
- **Minicomputers** (1960s) to **microcomputers** (1970s) to **PCs** (1980s)
- **Moore's Law** (1965): transistor count doubles every ~2 years --- enabling modern AI

---

### From Terminals to Personal Computers

![IBM PC](figures/ibm_pc.png)

---

## Part II: A Brief History of Artificial Intelligence

---

### The Birth of AI (1956)

- **Dartmouth Summer Research Project** formally established the field
- John McCarthy's founding conjecture:

> "Every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."

- Key attendees: McCarthy, Minsky, Shannon, Newell, Simon

---

### The Birth of AI (1956)

![John McCarthy](figures/john_mccarthy.jpg)

---

### Good Old-Fashioned AI (1950s-1970s)

- **GOFAI**: symbolic reasoning and logic-based systems
- Core technique: **search algorithms** --- systematically exploring possible solutions
- Key achievements:
    - **Logic Theorist** (1956): proved mathematical theorems
    - **General Problem Solver** (1959): universal problem-solving attempt
    - **ELIZA** (1966): early chatbot simulating a psychotherapist

---

### Good Old-Fashioned AI (1950s-1970s)

![ELIZA](figures/eliza.png)

---

### The Limits of Search

- Most interesting AI problems are **NP-complete**
- **Combinatorial explosion**: search spaces grow exponentially
- Deeper theoretical barriers:
    - **Undecidability**: some problems have no algorithmic solution
    - **Computational irreducibility**: no shortcuts to predict behavior
    - **Chaos theory**: sensitive dependence on initial conditions

---

### Expert Systems (1970s-1980s)

- Shifted from general problem-solving to domain-specific knowledge
- Prominent systems:
    - **MYCIN**: diagnosed bacterial infections
    - **DENDRAL**: identified chemical compounds
    - **XCON**: configured computer systems at DEC
- Limited by the **knowledge acquisition bottleneck** --- difficulty of encoding expertise as rules

---

### The Logic Tradition

- AI's emphasis on logical reasoning drew on a long heritage:
    - **Aristotle**: formalized syllogistic reasoning
    - **George Boole** (1850s): algebraized logic
    - **Frege** (1879): first formal predicate calculus
    - **First-order logic**: foundation for knowledge representation
- Expert systems combined logic with knowledge bases but remained **brittle**

---

### The Logic Tradition

![George Boole](figures/george_boole.jpg)

---

### Rodney Brooks and the Robotics Revolution (1980s)

> "The world is its own best model."

- Challenged classical AI's emphasis on internal representation
- **Subsumption architecture**: layered reactive behaviors, no central world model
- Intelligence emerging from interaction with the environment
- Led to practical robots like the Roomba

---

### Rodney Brooks and the Robotics Revolution (1980s)

![Shakey the Robot](figures/shakey_robot.jpg)

---

### The Machine Learning Turn (1990s-2000s)

- Fundamental shift: from hand-coding rules to **learning from data**
- Key methods:
    - **Support Vector Machines**: powerful classifiers with theoretical guarantees
    - **Random Forests**: ensemble methods combining decision trees
    - **Bayesian networks**: probabilistic graphical models
- **Deep Blue** defeated Kasparov (1997) --- but still relied on search, not learning

---

### The Machine Learning Turn (1990s-2000s)

![Deep Blue](figures/deep_blue.jpg)

---

### The Deep Learning Revolution (2010s)

- Neural networks returned with massive datasets, GPU computing, and new architectures
- Milestones:
    - **AlexNet** (2012): sparked the deep learning revolution in vision
    - **AlphaGo** (2016): defeated world Go champion Lee Sedol
    - **GPT/BERT** (2018): transformed NLP through large-scale pretraining

---

### The Deep Learning Revolution (2010s)

![AlphaGo vs. Lee Sedol](figures/alphago_lee_sedol.jpg)

---

## Part III: A Brief History of Linguistics and NLP

---

### Historical Linguistics (19th Century)

- Modern linguistics emerged in 19th-century Germany
- Dominant paradigm: understanding languages through their **history and evolution**
- Key developments:
    - Reconstruction of **Proto-Indo-European**
    - The **comparative method** for identifying language relationships
    - Explanation meant *historical* explanation

---

### The Saussurean Turn (c. 1900)

- **Ferdinand de Saussure** (1857-1913) shifted to **synchronic linguistics**
- Studying language as a system at a single point in time
- Key concepts:
    - **Langue vs. parole**: abstract system vs. individual speech acts
    - **Signifier and signified**: arbitrary relationship between form and meaning
    - **Structure**: language as a system of differences

---

### The Saussurean Turn (c. 1900)

![Saussure](figures/saussure.jpg)

---

### American Structuralism (1920s-1950s)

- **Edward Sapir** (1884-1939): relationship between language, culture, and thought
- **Leonard Bloomfield** (1887-1949): rigorous, behaviorist methodology
- **Zellig Harris**: the linguist's goal is to account for how language departs from equiprobability
- **George Zipf**: discovered quantitative regularities (Zipf's Law)

---

### American Structuralism (1920s-1950s)

![Edward Sapir](figures/edward_sapir.jpg)

---

### The Chomskyan Revolution (1957)

- Noam Chomsky's *Syntactic Structures* transformed linguistics
- Key ideas:
    - **Formal grammars**: precise mathematical specifications
    - **Generative grammar**: rules generating all and only grammatical sentences
    - **Competence vs. performance**
    - **Innateness**: humans possess innate language-specific knowledge

---

### The Chomskyan Revolution (1957)

![Noam Chomsky](figures/noam_chomsky.jpg)

---

### The Chomsky Hierarchy

- Classified formal grammars by computational power:
    - **Regular grammars** (finite-state automata)
    - **Context-free grammars**
    - **Context-sensitive grammars**
    - **Unrestricted grammars** (Turing machines)
- Natural language syntax requires at least context-free power
- Brought **algorithmic thinking** to mainstream linguistics

---

### Early NLP: Machine Translation (1950s-1960s)

- First major goal of computational linguistics, driven by Cold War
- **Georgetown-IBM experiment** (1954): demonstrated Russian-to-English translation
- Warren Weaver's memorandum (1949): translation as a **code-breaking problem**
- Optimistic predictions that high-quality MT was imminent

---

### Early NLP: Machine Translation (1950s-1960s)

![Warren Weaver](figures/512px-Warren_Weaver.jpg)

---

### The ALPAC Report (1966)

- Assessed ten years of MT research --- concluded MT had **not lived up to its promises**
- Recommended:
    - Increased support for basic linguistics research
    - Computational tools to aid human translators
    - Skepticism about fully automatic translation
- MT funding largely dried up for over a decade --- the **first AI Winter**

---

### Rule-Based NLP (1960s-1980s)

- NLP shifted to hand-crafted linguistic rules
- **SHRDLU** (1970): natural language understanding in a "blocks world"
- Parsing algorithms, morphological analyzers
- Limitations: **brittle**, labor-intensive, domain-specific

---

### Rule-Based NLP (1960s-1980s)

![SHRDLU](figures/shrdlu_blocks_world.png)

---

### Information Theory and Language

- **Claude Shannon's** information theory (1948) provided crucial tools:
    - **Entropy**: measure of uncertainty or information content
    - **N-gram models**: predicting next symbol from previous $n-1$ symbols
    - **Perplexity**: evaluating how well a model predicts a sequence
- Shannon showed English has significant statistical redundancy

---

### Information Theory and Language

![Claude Shannon](figures/claude_shannon.jpg)

---

### The Statistical Revolution (1990s)

- Paradigm shift from rules to learning from data
- Key methods:
    - **N-gram language models**
    - **Hidden Markov Models (HMMs)** for speech recognition and POS tagging
    - **Statistical parsing** from treebanks

> "Every time I fire a linguist, our system performance improves." --- Fred Jelinek, IBM

---

### Principles of Statistical NLP

- Three key principles emerged:
    1. **Robustness**: work on real data, not toy examples
    2. **Language-independence**: learn from data, be portable across languages
    3. **Quantitative evaluation**: measure performance objectively
- Enabled by linguistic resources:
    - **Brown Corpus** (1967), **Penn Treebank** (1990s)
    - **Linguistic Data Consortium**

---

### Machine Learning Methods for NLP (2000s)

- ML became central to NLP:
    - **Conditional Random Fields**: sequence labeling
    - **Maximum Entropy models**: flexible classifiers
    - **Latent Semantic Analysis**: hidden semantic structure
- Shared tasks drove progress:
    - **CoNLL**, **SemEval**, **WMT** competitions

---

### Word Embeddings (2013)

- **Word2Vec** (Mikolov et al., 2013) revolutionized word representation
- Words as dense vectors in continuous space
- Semantic relationships encoded geometrically:
    - king - man + woman $\approx$ queen
- **GloVe** (2014): similar results from global co-occurrence statistics
- Dense vectors replaced hand-engineered sparse features across NLP

---

### The Neural Era (2014-2017)

- Neural architectures transformed sequence modeling:
    - **RNNs**: process sequences step by step
    - **LSTMs**: handle long-range dependencies
    - **Sequence-to-sequence**: encoder-decoder for translation
- **Attention mechanisms** (Bahdanau et al., 2014): focus on relevant input parts
- Neural MT surpassed statistical MT; Google deployed neural translation in 2016

---

### The Transformer (2017)

- "Attention Is All You Need" (Vaswani et al., 2017)
- Key innovations:
    - **Self-attention** replaces recurrence entirely
    - **Parallel processing** enables much faster training
    - **Multi-head attention**, positional encodings, residual connections
- Scales effectively to very large models

---

### BERT and GPT (2018)

- The Transformer enabled two paradigms:
    - **BERT**: bidirectional pretraining for language understanding
    - **GPT**: autoregressive language modeling
- Pretrain on massive text, then fine-tune on downstream tasks
- Numerous variants: RoBERTa, XLNet, T5, and many more
- Transformed virtually every NLP benchmark

---

### The Large Language Model Era (2020-Present)

- Scaling Transformers revealed **emergent capabilities**:
    - **GPT-3** (2020): 175B parameters, few-shot learning
    - **ChatGPT** (2022): conversational AI reaches mainstream adoption
    - **GPT-4, Claude, Gemini**: multimodal understanding and reasoning
- New capabilities:
    - In-context learning, chain-of-thought reasoning, instruction following

---

### Current Frontiers and Open Questions

- **Understanding vs. pattern matching**: do LLMs truly understand language?
- **Alignment**: ensuring AI systems behave according to human values
- **Interpretability**: what computations occur inside these models?
- **Linguistic competence**: what do LLMs reveal about the nature of language?
- The rationalism vs. empiricism debate remains unresolved

---

## Converging Histories

- Algorithms developed millennia ago laid the groundwork for computational thinking
- Formalization of logic and computation enabled both AI and the hardware to run it
- Linguistics provided theoretical understanding; information theory provided mathematical tools
- Today's LLMs represent a convergence: massive compute, neural architectures, statistical learning, and linguistic benchmarks
- Each generation has built upon --- and sometimes rejected --- the previous one

---

## Summary

- **Computing**: from ancient algorithms to electronic computers and Moore's Law
- **AI**: from symbolic reasoning to machine learning to deep learning
- **NLP**: from historical linguistics through rule-based and statistical methods to Transformers and LLMs
- Understanding this history helps appreciate both how far we have come and the enduring questions that remain
