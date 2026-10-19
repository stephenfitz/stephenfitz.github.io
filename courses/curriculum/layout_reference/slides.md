# Slide layout reference
@label Course build system
@footer Machine learning and neural networks for energy applications
@meta 23 example slides
@meta Every layout and slide type
Each slide shows one layout, drawn from slides.md with illustrative content
:::notes
Every layout available in slides.md, each shown on one slide. Read this PDF next to its source, curriculum/layout_reference/slides.md. The content is illustrative.
:::
---
## Today's plan
@type agenda
| Time | Session | Kind |
|---|---|---|
| 09:00 | Welcome, prediction contracts, and evidence | Lecture |
| 09:45 | Splits, leakage, and baselines | Lecture |
| 10:30 | Break | Break |
| 10:45 | Lab 1: baseline and split design | Lab |
| 11:30 | Linear models and regularization | Lecture |
| 12:30 | Lunch | Lunch |
| 13:30 | Classification: KNN, SVM, trees | Lecture |
| 14:45 | Break | Break |
| 15:00 | Lab 2: a defensible comparison | Lab |
| 16:20 | Evidence review and exit check | Discussion |
:::notes
Agenda layout: a table with time, session, and kind.
:::
---
## What you will be able to do
@type objectives
By the end of the day you will be able to:
- Turn an energy problem into a target, prediction unit, and evaluation design.
- Fit and compare linear regression, KNN, SVM, and decision trees.
- Explain overfitting using residuals and validation results.
- Build a preprocessing pipeline and defend its final evaluation.
:::notes
Objectives layout.
:::
---
# Problem framing and evidence
@type section
A model is only as useful as the question it answers.
- Prediction contracts
- Splits that match deployment
- Leakage and baselines
:::notes
Section divider.
:::
---
## Baselines make improvement measurable
- Regression: compare with the training-target mean or a domain forecast.
- Classification: compare with a majority-class or simple linear classifier.
  - Use `DummyClassifier(strategy="most_frequent")` as the floor.
- Use the same evaluation rows and metric for every candidate.
- Record runtime and failure cases alongside the score.
@callout A model that cannot beat a persistence forecast has not yet earned its complexity.
:::notes
Bullets with a sub-bullet and a callout.
:::
---
## Preprocessing belongs inside the pipeline
@figure layout_pipeline_flow
Every fitted step — imputation, scaling, selection — learns only from the training fold.
@caption Schematic. Steps inside the bracket are refitted on each training fold during cross-validation.
:::notes
Full-width figure with lead text and caption.
:::
---
## A network is a stack of simple functions
@figure layout_mlp
@layout split
- Each hidden unit computes a weighted sum and applies a nonlinearity.
- Layers compose: the output of one is the input to the next.
- Width and depth set capacity; data and regularization decide whether it is used well.
@callout Parameters: $4\cdot6+6$ + $6\cdot6+6$ + $6\cdot2+2$ = 86.
:::notes
Split layout: text left, figure right, callout under the text.
:::
---
## Gradient descent updates every weight
@equation w_{t+1}=w_t-\eta\,\frac{\partial L}{\partial w}
@figure layout_mlp_boundary
@layout split
- $\eta$ is the learning rate; too large diverges, too small stalls.
- The gradient points uphill, so we step against it.
- Mini-batches estimate the gradient cheaply and add useful noise.
@caption MLP (two hidden layers of 16 units) fitted to synthetic moons. Shading shows predicted probability.
:::notes
Equation plus bullets on the left, computed figure on the right.
:::
---
## Bagging and boosting use different training logic
### Bagging {blue}
- Trains models **in parallel** on bootstrap samples.
- Averages to reduce variance.
- Random forest adds feature subsampling.
### Boosting {orange}
- Trains models **in sequence**.
- Each model fits what the ensemble still gets wrong.
- Reduces bias; needs early stopping.
### Stacking {purple}
- Learns how to **combine** different models.
- Needs out-of-fold predictions to avoid leakage.
@layout cards
@callout All three need the same honest validation design as a single model.
:::notes
Cards with colour tags.
:::
---
## Scaling: inside or outside the fold?
@layout compare
### Fit on all rows, then split {bad}
- Validation statistics leak into training.
- Scores are optimistic, sometimes badly.
### Fit inside each training fold {good}
- Validation rows stay unseen.
- Scores estimate deployment performance.
:::notes
Compare layout with good/bad tags.
:::
---
## The modelling workflow
@layout steps
A project moves through five stages, and evidence flows back from each one.
1. **Frame** — target, prediction time, unit of analysis
2. **Audit** — units, missingness, leakage risks
3. **Baseline** — the simplest defensible model
4. **Compare** — controlled experiments on a fixed split
5. **Deploy** — monitor drift and re-validate
@callout Most project failures start at step 1, not step 4.
:::notes
Steps layout.
:::
---
## Accuracy can hide the minority class
@type exercise
@time 4 min
@mode Pairs
Using TN = 85, FP = 5, FN = 3, TP = 7:
- Calculate accuracy, event precision, and event recall.
- Compare with a model that always predicts no event.
- Which error would matter most for the proposed use?
:::notes
Exercise layout with time and mode chips.
:::
---
## The always-negative model looks excellent
@layout stats
Same 100 wells, 10 real events. A model that never raises an alert:
@stat 90% | accuracy | blue
@stat 0% | event recall | orange
@stat 0 | useful alerts | red
@callout Report per-class recall and precision whenever events are rare.
:::notes
Stats layout.
:::
---
## Regression metrics answer different questions
| Metric | Interpretation | Limitation |
|---|---|---|
| MAE | Average absolute error, in target units | Treats all error sizes linearly |
| RMSE | Error in target units, emphasizes large misses | Sensitive to outliers |
| $R^2$ | Relative to an evaluation-set mean reference | Can be negative |
Report errors by well or operating regime as well as overall.
:::notes
Table layout.
:::
---
## Grouped tuning respects well boundaries
@highlight 5-6
```python
from sklearn.model_selection import GroupKFold, GridSearchCV

search = GridSearchCV(
    model,
    {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1]},
    cv=GroupKFold(n_splits=3), scoring="f1_macro",
)
search.fit(X_train, y_train, groups=well_train)
print(search.best_params_)
```
```output
{'svc__C': 1, 'svc__gamma': 0.1}
```
Reserve the final test wells before running this search.
:::notes
Code with highlighted lines and an output panel.
:::
---
## A pipeline in five lines
@layout code-split
```python
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVC(kernel="rbf", C=1.0),
)
scores = cross_val_score(model, X, y, cv=cv)
```
- `make_pipeline` chains fitted steps.
- Every CV fit re-learns the imputer and scaler.
- Categorical columns need a `ColumnTransformer`.
:::notes
Code-split layout.
:::
---
## Convolution by numbers
@figure layout_conv_matrix
@layout split-left
- Slide the 2×2 kernel over the input.
- Multiply element-wise and sum.
- Top-left: $1\cdot1+2\cdot0+0\cdot0+1\cdot(-1)=0$.
:::notes
Split-left layout.
:::
---
## A CNN turns pixels into class scores
@figure layout_cnn_blocks
@caption Schematic. Spatial size shrinks while the number of channels grows.
:::notes
Figure-only slide.
:::
---
## Architecture and behaviour side by side
@figure layout_mlp
@figure layout_mlp_boundary
@layout pair
@caption Left: schematic of a small multilayer perceptron. Right: a two-hidden-layer MLP fitted to synthetic moons (seed 0). Pair layouts need HALF-size figures.
:::notes
Pair layout.
:::
---
## Complexity should be earned, not assumed
@type statement
A simple model you can explain beats a complex model you cannot validate.
Prefer the simplest model that meets the requirement on honest validation data.
:::notes
Statement layout.
:::
---
## Laboratory: a defensible comparison
@type lab
@time 80 min
@notebook exercises/01_supervised_learning_basics/Supervised_Learning_Basics_Student.ipynb
- Load the data and write the prediction contract in the first cell.
- Fit one baseline and two candidates on a fixed split.
- Plot residuals or a confusion matrix for the best candidate.
- Write down one failure case and a hypothesis for it.
@deliverable A score table with the baseline first
@deliverable One residual or confusion plot
@deliverable A failure case with a hypothesis
:::notes
Lab layout.
:::
---
## Key takeaways
@type recap
- The split must answer the deployment question.
- Fit every preprocessing step inside the training fold.
- A baseline makes improvement measurable.
- Capacity is controlled by regularization and validated, not guessed.
:::notes
Recap layout.
:::
---
## References and further study
@type reference
- scikit-learn User Guide: linear models, neighbours, SVM, decision trees.
- scikit-learn: common pitfalls and cross-validation.
- Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning*, chapters 2, 3, 9, and 12.
:::notes
Reference layout.
:::
