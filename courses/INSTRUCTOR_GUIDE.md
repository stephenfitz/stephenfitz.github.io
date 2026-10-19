# Instructor guide

This guide covers preparing and teaching the two five-day courses: the day plans, laboratories and their preparation, how to present, the evaluation rules the courses teach, and how to edit the materials. Each module's `instructor_notes.pdf` (every slide with its notes) is the detailed script; this guide is the overview.

## Audience and preparation

The courses assume technical professionals who can read Python functions, manipulate an array or DataFrame and follow basic algebra. They do not need to derive SVD, backpropagation through time or UMAP's topology before starting. Each topic starts with intuition and a worked numerical example; the equations and extensions follow when the class is ready.

Before the first day, ask learners to:

1. Explain a feature, target, training example and prediction time in their own application.
2. Load a small CSV and report its shape, column types, missing counts and target distribution.
3. Perform a matrix multiplication and explain its output dimensions.
4. Distinguish data used to fit a model from data used to choose a model.

If these are difficult, spend 30–45 minutes on a Python and array refresher at the start of the first day and use the shorter lab routes. Do not remove the evaluation debriefs to recover time: they carry the courses' main lessons.

## How a teaching day is built

Every day runs from 09:00 to 17:00 with two 15-minute breaks and a 45–60 minute lunch: about 3.5 hours of taught material and 2–2.5 hours of laboratory work. The decks follow one pattern:

1. **Opening (15–30 min):** cover, the day's plan with times, learning outcomes, and a motivating energy case or warm-up question.
2. **Five to seven parts** (30–60 min each), each opened by a numbered section slide and closed by an exercise, a check question or a recap.
3. **Two or three laboratory blocks** placed where they belong in the day, each with a laboratory slide (tasks, notebook, evidence to submit, time) and a debrief slide that uses the reference results.
4. **Close (15 min):** key takeaways, an exit check and references.

Pace is roughly 2–3 minutes per content slide. Exercise and laboratory slides carry time chips; the notes carry timing cues in brackets and the full worked answer to every exercise. Section slides show the part number and a progress bar, so learners always know where they are in the day.

## Day plans

Times come from each deck's plan slide. Laboratory runtimes are for the solution notebooks on a recent laptop (a slower CPU may take up to twice as long) and exclude learners' own work.

### Machine Learning / Module 1: Supervised learning foundations

*Regression, classification and honest evaluation for energy data.* 95 slides · 6 parts · 3 laboratory blocks · 8 exercises · [PDF](machine_learning_course/Module_1_Supervised_Learning_1/slides.pdf) · [HTML](machine_learning_course/Module_1_Supervised_Learning_1/slides.html) · [PPTX](machine_learning_course/Module_1_Supervised_Learning_1/slides.pptx) · [Notes](machine_learning_course/Module_1_Supervised_Learning_1/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome and framing supervised problems |
| 09:30 | Data, EDA and preprocessing |
| 10:10 | *Break* |
| 10:25 | Splits, cross-validation and leakage |
| 11:00 | Linear models and regularization |
| 11:50 | **Lab 1: regression and regularization** |
| 12:40 | *Lunch* |
| 13:30 | Classification models |
| 14:30 | **Lab 2: classifiers and a facies split design** |
| 15:25 | *Break* |
| 15:40 | Evaluating and selecting models |
| 16:15 | **Lab 3: a tuned pipeline and a locked test** |
| 16:45 | Recap and exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_supervised_learning_basics` §1–2 (regression, polynomial degree, ridge and lasso) | ≈5 s | Downloads California housing (0.4 MB) on first use: pre-fill the scikit-learn data cache on offline machines. |
| Lab 2 | Same notebook §3–4 (KNN, trees on Iris), then `04_facies_classification` (per-well audit, well-grouped folds and tuning) | ≈10 s | None. |
| Lab 3 | `03_svm` (Sonar: tuned pipeline, PCA inside the pipeline, one locked test) | ≈3 s | None. |
| Extension | `02_knn_from_scratch` | ≈3 s | None. |

### Machine Learning / Module 2: Ensembles, interpretation, and pipelines

*Voting, bagging, boosting, honest explanations, and a predictor you can deploy.* 89 slides · 6 parts · 2 laboratory blocks · 10 exercises · [PDF](machine_learning_course/Module_2_Supervised_Learning_2/slides.pdf) · [HTML](machine_learning_course/Module_2_Supervised_Learning_2/slides.html) · [PPTX](machine_learning_course/Module_2_Supervised_Learning_2/slides.pptx) · [Notes](machine_learning_course/Module_2_Supervised_Learning_2/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome, warm-up, and a drilling case |
| 09:15 | Part 1: Why combining models works |
| 09:50 | Part 2: Bagging and random forests |
| 10:35 | *Break* |
| 10:50 | Part 3: From AdaBoost to gradient boosting |
| 11:45 | **Lab 1: tree, forest, and boosting compared** |
| 12:45 | *Lunch* |
| 13:45 | Part 4: Boosting libraries and stacking |
| 14:15 | Part 5: Interpreting ensemble models |
| 15:00 | *Break* |
| 15:15 | Part 6: Pipelines, leakage, and deployment |
| 15:50 | **Lab 2: importance, pipelines, and tuning** |
| 16:45 | Evidence review and exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `Ensemble_Methods` Parts 1–5 (tree, forest, boosting on shared folds; one labelled final evaluation) | ≈11 s (whole notebook) | None. |
| Lab 2 | Same notebook Parts 6–9 (importance, pipeline and reload check, grouped tuning; Part 9 XGBoost optional) |  | Part 9 uses xgboost from `requirements-labs-optional.txt`; it falls back to HistGradientBoosting. |

### Machine Learning / Module 3: Clustering and mixture models

*K-means, density, hierarchy, and probabilistic membership for energy data.* 87 slides · 6 parts · 3 laboratory blocks · 11 exercises · [PDF](machine_learning_course/Module_3_Unsupervised_Learning_1/slides.pdf) · [HTML](machine_learning_course/Module_3_Unsupervised_Learning_1/slides.html) · [PPTX](machine_learning_course/Module_3_Unsupervised_Learning_1/slides.pptx) · [Notes](machine_learning_course/Module_3_Unsupervised_Learning_1/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Why cluster energy data |
| 09:15 | Part 1: Similarity and scaling |
| 09:45 | Part 2: K-means |
| 10:30 | *Break* |
| 10:45 | **Lab 1: K-means on well W5** |
| 11:35 | Part 3: Density clustering |
| 12:15 | *Lunch* |
| 13:15 | **Lab 2: DBSCAN on logs** |
| 14:00 | Part 4: Hierarchy, zonation |
| 14:40 | *Break* |
| 14:55 | Part 5: Gaussian mixtures |
| 15:35 | Part 6: Validation |
| 16:05 | **Lab 3: Mixtures on rocks** |
| 16:45 | Recap and exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_kmeans` (well W5) | ≈10 s | None. |
| Lab 2 | `02_dbscan` (log.csv, then W5) | ≈4 s | None. |
| Lab 3 | `03_gmm` (rock geochemistry; explicit missing-value report) | ≈6 s | None. |
| Extensions | `04_hierarchical`, `05_comparison` | ≈3 s, ≈6 s | None. |

### Machine Learning / Module 4: Dimensionality reduction and feature selection

*Choosing, compressing and mapping many correlated measurements.* 88 slides · 6 parts · 2 laboratory blocks · 7 exercises · [PDF](machine_learning_course/Module_4_Unsupervised_Learning_2/slides.pdf) · [HTML](machine_learning_course/Module_4_Unsupervised_Learning_2/slides.html) · [PPTX](machine_learning_course/Module_4_Unsupervised_Learning_2/slides.pptx) · [Notes](machine_learning_course/Module_4_Unsupervised_Learning_2/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome, the curse of dimensionality, feature selection |
| 10:30 | *Break* |
| 10:45 | PCA in depth |
| 11:40 | **Lab 1: PCA on drilling data** |
| 12:40 | *Lunch* |
| 13:40 | Lab 1 debrief; kernel PCA, LDA, ICA and NMF |
| 14:25 | Manifold learning: Isomap, LLE, t-SNE and UMAP |
| 15:15 | *Break* |
| 15:30 | **Lab 2: nonlinear maps and a fair comparison** |
| 16:30 | Debrief, reduction in practice and exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_pca` (drilling data) | ≈3 s | None. |
| Lab 2 | Route A `02_tsne` (production data) or route B `04_comparison` (petrophysics) | ≈35 s / ≈30 s | Route B and `03_umap` use umap-learn from `requirements-labs-optional.txt`; both skip UMAP cleanly without it. |
| Optional | `03_umap` | ≈25 s | umap-learn. |

### Machine Learning / Module 5: Rare events, anomalies, and model selection

*Finding the unusual, judging rare-event evidence, and choosing models without fooling yourself.* 89 slides · 7 parts · 2 laboratory blocks · 11 exercises · [PDF](machine_learning_course/Module_5_Advanced_Supervised/slides.pdf) · [HTML](machine_learning_course/Module_5_Advanced_Supervised/slides.html) · [PPTX](machine_learning_course/Module_5_Advanced_Supervised/slides.pptx) · [Notes](machine_learning_course/Module_5_Advanced_Supervised/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome: rare events in energy operations |
| 09:15 | Part 1: What counts as an anomaly |
| 09:55 | Part 2: Detectors that learn normal behaviour |
| 10:45 | *Break* |
| 11:00 | **Lab 1: statistical rules and Isolation Forest** |
| 12:05 | Part 3: Evaluating rare events |
| 12:45 | *Lunch* |
| 13:30 | Part 4: Imbalance, calibration, uncertainty |
| 14:10 | **Lab 2: detector comparison and an alert policy** |
| 15:25 | *Break* |
| 15:40 | Part 5: Honest and efficient model selection |
| 16:05 | Exercise: read a tuning log and choose a model |
| 16:20 | Part 6: Deployment, drift, and monitoring |
| 16:40 | Part 7: Course synthesis and exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `Anomaly_Detection` Parts 1–3 (audit, statistical rules, Isolation Forest) | ≈4 s (whole notebook) | None. |
| Lab 2 | Same notebook Parts 4–6 plus a written alert policy |  | None. |

### Neural Networks / Day 1: Neural network foundations

*From tensors and single neurons to a network you can train and check by hand.* 93 slides · 7 parts · 2 laboratory blocks · 12 exercises · [PDF](neural_networks_course/Day_1_Intro_to_NN/slides.pdf) · [HTML](neural_networks_course/Day_1_Intro_to_NN/slides.html) · [PPTX](neural_networks_course/Day_1_Intro_to_NN/slides.pptx) · [Notes](neural_networks_course/Day_1_Intro_to_NN/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome, five-day route, neural networks in context |
| 09:45 | Tensors, shapes, and tools |
| 10:20 | *Break* |
| 10:35 | **Lab 1: Sonar baseline pipeline, then debrief** |
| 11:40 | From neuron to perceptron and XOR |
| 12:30 | *Lunch* |
| 13:30 | Multilayer networks; losses and gradient descent |
| 14:45 | *Break* |
| 15:00 | Backpropagation, training loop, first networks in code |
| 15:55 | **Lab 2: NumPy network on Sonar, debrief, exit check** |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_sklearn_pipeline` (Sonar baseline; shuffled CV on training rows, one test) | ≈10 s | None. |
| Lab 2 | `02_nn_from_scratch` (NumPy network, gradient check, validation split) | ≈5 s | None. |

### Neural Networks / Day 2: Training neural networks reliably

*Diagnostics, initialization, normalization, optimization, regularization, and framework practice.* 92 slides · 5 parts · 2 laboratory blocks · 8 exercises · [PDF](neural_networks_course/Day_2_NN_Advanced/slides.pdf) · [HTML](neural_networks_course/Day_2_NN_Advanced/slides.html) · [PPTX](neural_networks_course/Day_2_NN_Advanced/slides.pptx) · [Notes](neural_networks_course/Day_2_NN_Advanced/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome, then Part 1: diagnose before you tune |
| 09:50 | Part 2: signal propagation and normalization |
| 10:35 | *Break* |
| 10:50 | Part 3: a PyTorch training loop you can trust |
| 11:30 | **Lab 1: a reproducible PyTorch pipeline** |
| 12:30 | *Lunch* |
| 13:30 | Lab 1 debrief, then Part 4: optimizers, schedules, and tuning |
| 14:35 | *Break* |
| 14:50 | Part 5: regularization and autoencoders |
| 15:35 | **Lab 2, debrief, and exit check** |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_pytorch_pipeline` (reproducible Sonar pipeline) | ≈4 s | Keras alternative `02_tensorflow_pipeline` needs TensorFlow in a separate environment (see below). |
| Lab 2 | Track A `03_advanced_nn` (MNIST regularization comparison) or track B `04_nn_unsupervised` (autoencoder) | ≈2.5 min / ≈10 s | Track A downloads MNIST (about 12 MB): set `COURSE_DATA` to a shared cache. |

### Neural Networks / Day 3: Energy modelling capstone

*A reproducible baseline, a neural candidate, and a defensible conclusion.* 75 slides · 7 parts · 5 project sprints · 4 exercises · [PDF](neural_networks_course/Day_3_Capstone/slides.pdf) · [HTML](neural_networks_course/Day_3_Capstone/slides.html) · [PPTX](neural_networks_course/Day_3_Capstone/slides.pptx) · [Notes](neural_networks_course/Day_3_Capstone/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome and the brief |
| 09:35 | ESP gas lock briefing |
| 09:55 | PE log curve briefing |
| 10:15 | *Break* |
| 10:30 | Sensor briefing |
| 10:45 | **Sprint 1: data contract** |
| 11:20 | Checkpoint 1 |
| 11:30 | Tabular-NN playbook |
| 11:50 | **Sprint 2: baselines** |
| 12:30 | Checkpoint 2 |
| 12:40 | *Lunch* |
| 13:25 | Experiment design |
| 13:40 | **Sprint 3: experiment** |
| 14:20 | Checkpoint 3 |
| 14:30 | Error analysis |
| 14:45 | **Sprint 4: final test** |
| 15:25 | Checkpoint 4 |
| 15:35 | *Break* |
| 15:45 | Presenting evidence |
| 16:00 | **Sprint 5: handoff** |
| 16:15 | Final presentations |
| 16:50 | Takeaways, exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Sprints 1–5 | One of `01_esp_gas_lock`, `02_log_curves`, `03_sensor_signals`, backed by `curriculum/capstone.py` | ≈5–15 s per development run | None; runs write to `output/capstone/`. |

### Neural Networks / Day 4: Convolutional neural networks

*Images as tensors, convolution, architectures, transfer learning, detection, segmentation and reconstruction.* 92 slides · 7 parts · 2 laboratory blocks (Lab 2 has two alternatives) · 12 exercises · [PDF](neural_networks_course/Day_4_CNN/slides.pdf) · [HTML](neural_networks_course/Day_4_CNN/slides.html) · [PPTX](neural_networks_course/Day_4_CNN/slides.pptx) · [Notes](neural_networks_course/Day_4_CNN/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome, casing-inspection case, images as tensors |
| 09:45 | Convolution mechanics |
| 10:30 | *Break* |
| 10:45 | Feature maps, pooling and a compact CNN |
| 11:30 | **Lab 1: a checked CNN on CIFAR-10** |
| 12:30 | *Lunch* |
| 13:30 | Lab 1 debrief, architecture ideas, training and transfer learning |
| 14:50 | *Break* |
| 15:05 | Beyond classification; autoencoders and 1-D signals |
| 15:50 | **Lab 2: autoencoder or casing defects, then debrief and exit check** |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1 | `01_cnn_cifar10` (or `02_cnn_svhn`) | ≈1 min on a GPU or Apple silicon, 3–5 min on a laptop CPU | Downloads CIFAR-10 (about 170 MB) or SVHN (about 250 MB): pre-fill the cache and set `COURSE_DATA`. |
| Lab 2A (alternative) | `03_autoencoder` (denoising, MNIST) | ≈1 min on a GPU, 5–8 min on a laptop CPU | MNIST (about 12 MB). |
| Lab 2B (alternative) | `04_defect_detection` (casing inspection logs, split by well) | 5–8 min on a laptop CPU | h5py; the 467 MB HDF5 file is local. Runs on CPU or CUDA (its adaptive pooling layer is not supported on Apple MPS). |

### Neural Networks / Day 5: Sequence models and attention

*Forecasting, recurrent networks, attention, uncertainty, and a review of the course.* 97 slides · 7 parts · 3 laboratory blocks · 14 exercises · [PDF](neural_networks_course/Day_5_RNN_Applications/slides.pdf) · [HTML](neural_networks_course/Day_5_RNN_Applications/slides.html) · [PPTX](neural_networks_course/Day_5_RNN_Applications/slides.pptx) · [Notes](neural_networks_course/Day_5_RNN_Applications/instructor_notes.pdf)

| Time | Session |
|---|---|
| 09:00 | Welcome · Part 1: time-series fundamentals |
| 09:45 | Part 2: splits, baselines and metrics |
| 10:30 | *Break* |
| 10:45 | Part 3: recurrent networks |
| 11:30 | **Lab 1A: shampoo warm-up, climate baselines** |
| 12:30 | *Lunch* |
| 13:30 | **Lab 1B: climate LSTM against baselines** |
| 14:15 | Part 4: convolutions and attention |
| 14:50 | Part 5: transformers, pretrained models |
| 15:20 | *Break* |
| 15:35 | Part 6: uncertainty and anomalies |
| 16:00 | **Lab 2: sequence anomalies or retail** |
| 16:35 | Part 7: course review, exit check |

| Laboratory | Notebook (`exercises/`) | Solution runtime | Preparation |
|---|---|---|---|
| Lab 1A | `01_lstm_shampoo` warm-up, then `02_lstm_climate` windows and baselines | ≈5 s + data preparation | None (Jena file is local, 40 MB). |
| Lab 1B | `02_lstm_climate` (LSTM against persistence and seasonal naive) | ≈3.5 min | None. |
| Lab 2 | Option A `04_timeseries_autoencoder` (paper-machine breaks, event-level scoring) or option B `03_walmart_timeseries` (three-item subset) | ≈1 min / ≈16 s | Option A reads an .xlsx file (openpyxl). |
| Homework | `05_stock_prediction` (persistence illusion) | ≈1 min | None. |
## Presenting

- **HTML deck (`slides.html`).** Opens offline in any current browser. Arrow keys, Space, clicks or swipes navigate; **N** shows the notes panel; **P** opens a presenter window with the current slide, the next slide, the notes and a timer (move it to your laptop screen); **G** shows an overview grid; **F** toggles fullscreen; **Home**/**End** jump; **?** lists the keys. A fragment such as `#s42` opens a given slide. Keep notes closed while learners work, because they contain answers.
- **PowerPoint (`slides.pptx`).** A presentation copy for PowerPoint's presenter view: one high-resolution image per slide, the same notes, real slide titles for navigation, alt text, and one section per part. Change content in `slides.md` and rebuild; do not edit the PowerPoint file.
- **PDF (`slides.pdf`).** For projection, printing or handouts. Bookmarks mark the plan, each part and each laboratory. The combined course PDFs in `output/pdf` bookmark every module.
- **Instructor notes (`instructor_notes.pdf`).** One page per slide: the slide above its notes, with bookmarks by slide number and title. Print it or keep it on a tablet.

## Running exercises and laboratories

Use a short explanation, a worked example, then a question learners answer before you reveal the debrief. Most discussion questions need 2–4 minutes; numerical ones need enough time to write the denominator or tensor shape, not just guess a number. Exercise slides show the suggested time and mode (individual, pairs, groups).

For each laboratory, reserve the last 10–25 minutes for evidence review. Ask two teams to show different outcomes, including a model that did not improve. Require them to name the evaluation set, state the metric's units or denominator, and point to a failure case. Do not rank teams by a noisy test score. The debrief slides give reference results from the solution notebooks; small differences between library versions are expected, and the notes say where a result is version-sensitive.

Every laboratory asks for the same kind of evidence:

1. A prediction or analysis contract.
2. A fixed split policy and a data and preprocessing audit.
3. A baseline and one controlled comparison.
4. A readable evidence figure with axis labels, units where known and a clear caption.
5. A short conclusion that separates observations from hypotheses.
6. A fresh-process run command or restartable notebook.

The notebooks follow the protocol the slides teach: model, epoch, threshold and hyperparameter choices use validation data or cross-validation on training data; the test set is opened once, in a clearly labelled final cell; fitted preprocessing learns from training data only; grouped or time-ordered splits are used where the data have wells, entities or time. Several labs deliberately end in an honest negative or weak result (for example, the casing-defect network barely beats chance when split by well, and the paper-machine autoencoder finds 5 of 41 breaks); treat these as the lesson, not as failures.

## Data and validation rules to reinforce

**Independent unit.** Rows from the same well, image, event or asset may be related. Use the split that corresponds to the intended deployment question. A chronological holdout tests future observations; an entity holdout tests other entities. Some applications need both restrictions.

**Fitted preprocessing.** Imputation, scaling, PCA, supervised feature selection, target transforms and resampling must fit within the relevant training fold. A method being unsupervised does not automatically make evaluation-set fitting valid for a predictive workflow. Describing one fixed dataset exploratorily is a different use case and should be labelled as such.

**Model selection.** Validation can select architecture, regularization, checkpoint, features and thresholds. Results on that same validation data are selected estimates. Final test results should describe a locked procedure. Repeatedly using test results to change the model turns the test into another validation set.

**Time windows.** Training targets must not extend into the evaluation target period. Validation inputs may use historical observations that were genuinely available at each forecast origin. Do not demand an arbitrary gap merely because input windows overlap; derive any purge or gap from the target interval and operational scenario.

**Interpretation.** Feature importance is model reliance under an evaluation, not causation. A cluster or embedding island does not establish a physical population. An anomaly score does not diagnose a fault. A contemporaneous event classifier does not demonstrate warning lead time.

## Capstone specifics (Neural Networks, Day 3)

Teams choose one of three supplied projects. Each has student and solution notebooks backed by a shared reference implementation, `curriculum/capstone.py`. From the course root, a development run is:

```bash
python3 curriculum/capstone.py esp --epochs 40
python3 curriculum/capstone.py log --epochs 40
python3 curriculum/capstone.py sensor --epochs 40
```

Development runs read the training file and never the final test labels. The validation split has its own seed (`--split-seed`, 42 by default), so runs that differ only in the training seed (`--seed`) are scored on the same validation rows; compare settings seed by seed. `--final` runs the locked comparison after model selection; copy the selected width, learning rate, seed and training budget into the final command, and use a separate `--out` directory for each experiment. The script saves a learning curve, split membership, validation predictions, metrics, preprocessing, baseline, neural weights and a prediction-equivalence check after reload. Load serialized artifacts only from trusted sources.

| Project | Data contract | Evaluation limitation |
|---|---|---|
| ESP gas lock | Six numeric sensors; GAS_LOCK target; ENTITY group; TimeSlot ordering; test uuid join | Test entities differ from training entities over overlapping times: an unseen-entity classification assessment, not an early-warning claim. |
| PE log curve | PE target; Depth, GR, ILD_log10, DeltaPHI, PHIND, RELPOS inputs | No well identifier. The contiguous source-row block is an explicit teaching fallback. Final labels align by position and require unchanged test order. |
| Sensor signals | col1–col10 inputs; uuid join key; −1/1 labels explicitly mapped to 0/1 | Units, fault meaning and sample independence are undocumented. Do not invent a sequential meaning for the numbered columns. |

The capstone deck teaches directly from what an audit of the supplied files shows, so check these before class:

- **ESP** (3,243 training rows, seven entities, 1,224 positive labels, 300 missing input cells). A random row split flatters every model; holding out whole pumps drops tree ensembles to near the always-negative baseline. The labels look curated (every gas-lock period is 51 rows long), which the notes raise as a question for the data owner, not a finding.
- **PE** (2,922 rows). Depth resets seven times, so the file holds eight blocks, and one 12-row block is duplicated.
- **Sensor** (3,580 rows, 2,061 labelled 1). The ten columns are near-copies of each other (pairwise correlation ≥ 0.9995), 2,484 rows repeat another row exactly, and 646 of the 895 test rows repeat a training row.

The rubric weights are: data and problem contract 20%, evaluation design 30%, modelling and reproducibility 20%, error analysis 20%, communication 10%. A failed or inconclusive neural model can earn full marks with sound evidence. Leakage or misaligned labels invalidate the affected assessment.

## Environment and preflight

Use a dedicated environment built from the requirements files, tested on the classroom machines before the course:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-labs.txt            # every core laboratory
pip install -r requirements-labs-optional.txt   # UMAP, XGBoost, imbalanced-learn
python3 curriculum/preflight.py                 # reports packages, fonts, decks and data
```

- Python 3.11 and 3.12 are supported. torchvision needs a Python build that includes the `lzma` module, which some custom Python builds lack.
- Every solution notebook runs top to bottom in this reference environment: Python 3.11, NumPy 2.4, pandas 3.0, SciPy 1.17, Matplotlib 3.11, scikit-learn 1.9, PyTorch 2.14, torchvision 0.29, seaborn 0.13, openpyxl 3.1, h5py 3.16, umap-learn 0.5, xgboost 3.2 and imbalanced-learn 0.14. Reference results quoted in the debriefs come from single runs; other library versions or GPU runs can move them slightly, and the notes say where a result is version-sensitive.
- Set `COURSE_DATA` to a shared folder holding the MNIST, CIFAR-10 and SVHN downloads so that classroom machines do not download them during the lab. Pre-fill the scikit-learn data cache for the California housing lab on offline machines.
- The Keras alternative on Day 2 needs Keras 3 with a backend (TensorFlow, or PyTorch via `KERAS_BACKEND=torch`); install TensorFlow in a separate environment if it conflicts with the PyTorch versions above.
- Run notebooks from their own folders. The capstone and PyTorch labs write run artifacts to `output/`.

## Editing the courses

Edit a module's `slides.md`, then rebuild that module with `python3 curriculum/build.py <path>/slides.md`, or everything with `python3 render_slides.py --all`. The authoring rules, slide syntax, layouts and figure toolkit are in [curriculum/AUTHORING.md](curriculum/AUTHORING.md), and [curriculum/layout_reference/slides.pdf](curriculum/layout_reference/slides.pdf) shows every layout, with its source `slides.md` in the same folder. Figures are Python functions in `curriculum/figs/<module>.py`; regenerate them with `python3 curriculum/make_figures.py --only <module>`. The build refuses content that would overflow a slide and warns when a figure is drawn too small, so a clean build is a good first check. Run `python3 curriculum/verify.py` before publishing.

## Sources for preparation

Primary references are attached to the relevant slide notes and the reference slide at the end of each deck. Useful entry points:

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html), especially [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- [imbalanced-learn common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) for resampling within folds.
- [UMAP documentation](https://umap-learn.readthedocs.io/) for parameter meanings and transform behaviour.
- [PyTorch tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), including [transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) for baselines, forecast evaluation and MASE.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for the original transformer architecture.
- [UCI Sonar](https://archive.ics.uci.edu/dataset/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks) for the rock-versus-mine label meaning.

Figure captions distinguish computed synthetic examples, mathematical schematics, public datasets and the supplied course data. No teaching example is presented as measured field performance.
