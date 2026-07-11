# SignBot Pipeline — Evaluation Results Summary (Thesis-Ready)

All figures below are extracted directly from existing files in `signbot_pipeline/results/`. No values were estimated, recalculated, or assumed. Every number is followed by its source file in brackets.

---

## 1. Inventory of evaluation artifacts found

| Artifact | File(s) | Experiment it belongs to |
|---|---|---|
| Classification report (single split) | `results/classification_report.txt` | Baseline model (`train_model.py`) |
| Confusion matrix (single split) | `results/6_5_5_confusion_matrix.png` | Baseline model |
| Per-class accuracy chart | `results/6_5_5_per_class_accuracy.png` | Baseline model |
| Learning curves | `results/6_5_5_learning_curves.png` | Baseline model |
| Training history (raw) | `results/training_history.json` | Baseline model |
| LOSO summary (mean ± std) | `results/loso/loso_summary.txt` | LOSO cross-validation |
| LOSO per-fold classification reports | `results/loso/fold1_report.txt`, `fold2_report.txt`, `fold3_report.txt` | LOSO cross-validation |
| LOSO per-fold confusion matrices | `results/loso/6_5_6_fold1_confusion_matrix.png`, `fold2`, `fold3` | LOSO cross-validation |
| LOSO fold accuracy bar chart | `results/loso/6_5_6_loso_accuracy.png` | LOSO cross-validation |
| LOSO per-fold learning curves | `results/loso/fold1_learning_curves.png`, `fold2`, `fold3` | LOSO cross-validation |
| LOSO per-fold training history (raw) | `results/loso/fold1_history.json`, `fold2_history.json`, `fold3_history.json` | LOSO cross-validation |
| Production model classification report | `results/production/production_classification_report.txt` | Production model (`train_production_model.py`) |
| Production confusion matrix | `results/production/6_5_7_production_confusion_matrix.png` | Production model |
| Production learning curves | `results/production/6_5_7_production_learning_curves.png` | Production model |
| Production training history (raw) | `results/production/production_history.json` | Production model |
| Class label list | `labels/class_names.json` (also `output/dataset/label_map.json`, `labels/label_map.json`, `labels/class_index.json`) | Shared across all experiments |

**Not found:** No ROC/PR curve files, no TensorBoard export directories/event files, no separate raw prediction files (e.g., `.csv` of per-sample predictions), no screenshots. All confusion matrices are pre-rendered `.png` images (normalized 0–1 heatmaps), not raw count matrices in CSV/JSON form.

**No duplicate or outdated result files were found.** The three sets of artifacts (baseline / LOSO / production) are not redundant — `README.md` (lines 137–148) explicitly defines each as answering a different thesis question, confirmed below.

---

## 2. Which result is "final" and what to report where

Per the project's own `README.md` ("What to report in thesis" section) and confirmed by matching filenames/figure numbers (`6_5_5`, `6_5_6`, `6_5_7` = Chapter/section figure numbers):

- **Baseline model** (`results/classification_report.txt`, `6_5_5_*`): a single random train/val/test split, signer-mixed. Establishes upper-bound feasibility of the architecture.
- **LOSO cross-validation** (`results/loso/*`, `6_5_6_*`): the **generalization-to-unseen-signers** metric — this is the headline number for the thesis's Chapter 3 experimental evaluation.
- **Production model** (`results/production/*`, `6_5_7_*`): trained on all 3 signers combined, this is the model actually deployed to the Raspberry Pi. Its test accuracy is the Chapter 4 "final deployed model" number, not a generalization measure (its test set overlaps signers seen in training).

None of these should be presented as interchangeable — each answers a different question and all three should appear in the thesis, in the roles above.

---

## 3. Final evaluation metrics table

| Experiment | Accuracy | Loss | Macro P / R / F1 | Weighted P / R / F1 | #Classes | #Test samples | Source |
|---|---|---|---|---|---|---|---|
| Baseline (single split) | 0.9690 | 0.1668 | 0.9715 / 0.9700 / 0.9699 | 0.9705 / 0.9690 / 0.9689 | 100 | 2320 | `classification_report.txt` |
| LOSO Fold 1 (held-out signer 01) | 0.3596 | 5.7041 | 0.3466 / 0.3594 / 0.3203 | 0.3461 / 0.3596 / 0.3200 | 100 | 787 | `loso/fold1_report.txt` |
| LOSO Fold 2 (held-out signer 02) | 0.3287 | 7.6121 | 0.3020 / 0.3265 / 0.2889 | 0.3037 / 0.3287 / 0.2906 | 100 | 791 | `loso/fold2_report.txt` |
| LOSO Fold 3 (held-out signer 03) | 0.3949 | 5.0993 | 0.3860 / 0.3945 / 0.3461 | 0.3942 / 0.3949 / 0.3484 | 100 | 742 | `loso/fold3_report.txt` |
| **LOSO mean ± std (3 folds)** | **0.3611 ± 0.0270** (36.11% ± 2.70%) | — | not reported in files | not reported in files | 100 | 2320 total | `loso/loso_summary.txt` |
| Production (all 3 signers combined) | 0.9647 | 0.1878 | 0.9668 / 0.9650 / 0.9650 | 0.9661 / 0.9647 / 0.9646 | 100 | 2320 | `production/production_classification_report.txt` |

**Number of classes:** 100 (counted directly in `labels/class_names.json`, and confirmed by the number of rows — excluding header/accuracy rows — in every classification report).

**Number of LOSO folds:** 3, one per signer (signers `01`, `02`, `03`), per `results/loso/loso_summary.txt` and `README.md` (Step 7).

**Mean/std across folds:** Only reported for overall accuracy — `0.3611` mean, `0.0270` std (`loso/loso_summary.txt`). No file contains mean/std of macro-F1, weighted-F1, precision, or recall across LOSO folds — **this was not found and is not being estimated here.** If needed for the thesis, it would have to be computed from the three per-fold reports above (macro-F1 values: 0.3203, 0.2889, 0.3461).

---

## 4. Confusion matrix analysis

### LOSO Fold 1 (`loso/6_5_6_fold1_confusion_matrix.png`, Acc 0.3596)
The heatmap shows a sharp split in behavior across the two halves of the label space:
- The bottom-right block (multi-word/phrase vocabulary classes, e.g. anatomy and first-aid terms such as "قفص صدري", "حروق", "صيدلية") shows a strong, clean diagonal with dark (high-probability) cells and little off-diagonal bleed.
- The top-left block (single digits, tens/hundreds numeral classes, and single Arabic letters) is visibly scattered — pale, diffuse off-diagonal cells cluster around the diagonal rather than sitting on it, indicating systematic confusion between numerically or visually adjacent signs rather than random error.

### Production model (`production/6_5_7_production_confusion_matrix.png`, Acc 0.9647)
Near-perfect single-pixel-wide diagonal across all 100 classes. Only a handful of faint off-diagonal cells remain, concentrated in the same numeral region (near classes "800"/"900") that was already the weakest area in the baseline and LOSO reports — consistent with the classification-report numbers below.

### Per-class accuracy chart (`results/6_5_5_per_class_accuracy.png`, baseline model)
Sorted ascending: roughly 60 of 100 classes sit at or above the mean line (0.970); the four lowest-accuracy classes range from ~0.71 to ~0.83, all the rest cluster tightly between 0.87 and 1.00. This shows the baseline model's errors are concentrated in a small handful of classes rather than spread evenly — consistent with the confusion-matrix pattern above.

### Systematic misclassification pattern (all three LOSO folds + both single-signer-mixed models agree)
Across `fold1_report.txt`, `fold2_report.txt`, and `fold3_report.txt`, the classes that repeatedly score **precision = recall = F1 = 0.0** are overwhelmingly:
- Isolated numeral classes (e.g., "9", "50", "400", "500", "800", "900" — varies by which signer is held out)
- Single Arabic letter classes (e.g., "ل", "م", "ن", "ه", "و", "ي", "ة", "أ", "ؤ" — again varying by fold)

By contrast, multi-syllable/phrase vocabulary classes (medical/anatomy terms, e.g. "قفص صدري", "شريط لاصق / بلاستر", "مستشفى", "حروق") retain non-trivial, often high, precision/recall even under LOSO in every fold.

**What this suggests about the model:** short, single-gesture signs (digits, isolated letters) carry less discriminative temporal/spatial signal and are more sensitive to a signer's individual motion style, hand size, and signing speed — so a BiLSTM trained on 2 signers generalizes poorly to a 3rd signer's rendition of these short signs. Longer, multi-segment signs (phrases) contain more distinguishing motion structure, which appears to generalize better across signers even without signer-specific training data.

---

## 5. Classification report analysis (best / worst performers)

### Production model (`production/production_classification_report.txt`) — representative of deployed performance
- **Highest precision (1.0000):** many classes reach perfect precision, e.g. "0", "6", "7", "8", "60", "70"(0.9583 actually — see exact table), "600", "700", "1000000", and numerous letters/phrases (ب, ت, ذ, ز, ص, ظ, ف, ل, ة… — full list in file).
- **Highest recall (1.0000):** similarly widespread (e.g. "3", "20", "40", "60", "400", "600", "700", many letters and all near-perfect phrase classes).
- **Highest F1 (1.0000):** e.g. "6", "7", "8", "20", "40", "60", "600", "700", "ذ", "ز", "ف", "ل", "ذ", plus several anatomy/phrase classes ("هيكل عظمي", "عمود فقري", "قفص صدري", "الأمعاء الدقيقة", "أنسجة" region, "حروق", "مخدر/ بنج").
- **Weakest performers:** "800" (P 0.7812 / R 0.8065 / F1 0.7937), "900" (P 0.7600 / R 0.7917 / F1 0.7755), "2" (P 1.0000 / R 0.7222 / F1 0.8387 — high precision but weak recall), "إسعافات أولية" (F1 0.8511), "شاش / ضمادة" (F1 0.8571), "جرح نازف" (F1 0.9302).

### Baseline model (`results/classification_report.txt`)
Same overall pattern: "800" is the single weakest class (F1 0.7586), "900" second weakest (F1 0.6923), with almost every other class at F1 ≥ 0.91. Macro (0.9699) and weighted (0.9689) F1 are nearly identical, indicating performance is not skewed by class-support imbalance.

### LOSO folds (`loso/fold1_report.txt`, `fold2_report.txt`, `fold3_report.txt`)
- **Highest precision/recall/F1 per fold** are concentrated in the phrase/vocabulary classes noted above (e.g., Fold 1: "قفص صدري" F1 1.0000, "الزائدة الدودية" F1 0.8571, "شريط لاصق / بلاستر" F1 0.9412; Fold 3: "حروق" F1 1.0000, "عمود فقري" F1 0.8889).
- **Weakest classes** are the numeral/letter classes scoring 0.0 across the board (listed in Section 4), plus several with support as low as 1 sample (e.g., class "5" in Fold 1 has only 1 test example — too small to draw any reliable per-class conclusion).

---

## 6. LOSO evaluation — what it indicates about generalization

- **Generalization to unseen signers is weak in absolute terms**: mean accuracy of 36.11% ± 2.70% (`loso/loso_summary.txt`) against a 100-class problem (random-chance baseline ≈1%), so the model is learning real signal, but it is far below the ~96–97% accuracy achieved when the same architecture is trained and tested on a signer-mixed pool (baseline 96.90%, production 96.47%).
- **Consistency across folds is reasonably tight**: fold accuracies range from 32.87% to 39.49%, a spread of only 6.6 percentage points, and the std (2.70%) is small relative to the mean — the model fails in a similar way regardless of which signer is held out, rather than generalizing well to some signers and catastrophically to others. This points to a systematic signer-dependence problem rather than one problematic signer.
- **No outlier fold**: Fold 2 (signer 02 held out) is the weakest (32.87%) and also has the highest loss (7.6121), while Fold 3 (signer 03) is the strongest (39.49%, lowest loss 5.0993) — but none of the three deviates enough from the mean to be called an outlier given the std of 2.70%.
- **Strength revealed:** the large gap between LOSO (~36%) and mixed-signer accuracy (~96%) is itself informative — it demonstrates the architecture can learn the vocabulary well when given signer-diverse training data, but 2 training signers are not enough diversity for the BiLSTM to learn signer-invariant representations for this dataset's short/simple signs.
- **Limitation revealed:** the per-class breakdown shows the failure is concentrated in specific classes (numerals, isolated letters) rather than being uniform, suggesting future work should focus on signer-invariant features (e.g., better normalization, more signers, or per-sign temporal augmentation) specifically for short/simple gesture classes rather than the whole vocabulary.

---

## 7. What each metric means (for the thesis narrative)

- **Accuracy** — fraction of all test videos classified into the exact correct sign, out of 100 possible classes. Simple but can hide class-imbalance issues.
- **Precision** (per class) — of all videos the model labeled as class X, the fraction that were truly class X. Low precision on a class means the model over-predicts that class (false positives).
- **Recall** (per class) — of all true videos of class X, the fraction the model correctly found. Low recall means the model misses that sign (false negatives) — often confusing it for a visually similar sign.
- **F1-score** — harmonic mean of precision and recall; the single number most informative for comparing classes without picking a bias toward false positives or false negatives.
- **Macro average** — unweighted mean across all 100 classes; every class counts equally regardless of how many test samples it has. Sensitive to poor performance on rare classes.
- **Weighted average** — mean across classes weighted by each class's support (number of test samples); reflects overall test-set performance more than macro, but can mask small-class failures if those classes are underrepresented.
- In this project, **macro and weighted values are very close in the baseline/production reports** (e.g., baseline macro-F1 0.9699 vs weighted-F1 0.9689), indicating the test set is close to class-balanced and no class is being masked. In the **LOSO folds, macro and weighted values are also close to each other but far lower than baseline/production** (e.g., Fold 2 macro-F1 0.2889 vs weighted-F1 0.2906), confirming the accuracy drop is a genuine generalization failure, not a class-imbalance artifact.
- **No micro-averaged metrics appear in any report file** — scikit-learn's `classification_report` (which generated all of these) does not print a micro-average row for single-label multi-class problems, because micro-average precision/recall/F1 equals overall accuracy in that setting. Overall accuracy (already reported per experiment) serves this role.

---

## 8. Recommendations for thesis defense presentation

1. **Lead with the LOSO number** (`36.11% ± 2.70%`, from `loso_summary.txt`) as the headline generalization result — this is what the committee will expect for a signer-independent recognition claim.
2. **Show the LOSO fold-accuracy bar chart** (`loso/6_5_6_loso_accuracy.png`) as the single summary visual for Chapter 3 — it communicates both the level and the fold-to-fold consistency at a glance.
3. **Pair one LOSO confusion matrix (e.g., Fold 1)** with the **production confusion matrix** side by side — this contrast is the most persuasive visual for explaining *why* generalization fails (numerals/letters) while overall architecture capacity is not the bottleneck (near-perfect diagonal when signers are mixed).
4. **Include the per-class accuracy chart** (`6_5_5_per_class_accuracy.png`) to show that baseline errors are concentrated in a few classes, not spread thin — supports the "specific failure mode" narrative rather than "model is bad."
5. **Report baseline vs. production vs. LOSO as three separate rows in one summary table** (as in Section 3 above) rather than a single accuracy figure — reviewers will ask which number is "the" result, and the honest answer is that all three answer different questions.
6. **State explicitly what is missing**: no ROC/PR curves, no TensorBoard logs, and no cross-fold macro/weighted-F1 mean±std exist in the current artifacts. If the committee asks for these, they are not fabricable from what's on disk and would require re-running evaluation with those metrics captured — flag this as a known gap rather than presenting an invented number.
7. Consider mentioning class "800"/"900" (and neighboring numeral confusions) explicitly as a known weak spot even in the high-accuracy (production/baseline) settings — it is the most consistent weak point across every single artifact reviewed.

---

## Files referenced in this report

- `results/classification_report.txt`
- `results/6_5_5_confusion_matrix.png`
- `results/6_5_5_per_class_accuracy.png`
- `results/6_5_5_learning_curves.png`
- `results/training_history.json`
- `results/loso/loso_summary.txt`
- `results/loso/fold1_report.txt`, `fold2_report.txt`, `fold3_report.txt`
- `results/loso/6_5_6_fold1_confusion_matrix.png`, `6_5_6_fold2_confusion_matrix.png`, `6_5_6_fold3_confusion_matrix.png`
- `results/loso/6_5_6_loso_accuracy.png`
- `results/loso/fold1_history.json`, `fold2_history.json`, `fold3_history.json`
- `results/production/production_classification_report.txt`
- `results/production/6_5_7_production_confusion_matrix.png`
- `results/production/6_5_7_production_learning_curves.png`
- `results/production/production_history.json`
- `labels/class_names.json`
- `README.md`
