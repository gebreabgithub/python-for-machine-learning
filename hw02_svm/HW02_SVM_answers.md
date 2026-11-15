# HW2: Support Vector Machines

Machine Learning &amp; Neural Networks for Neuroscience<br/>Pair assignment | Authors: ____________________ and ____________________

We study how the shape of the data, feature noise, incorrect labels, and extra dimensions affect classification. All numbers below come from the accompanying Python analysis. The main random seed is 42; separate fixed seeds control each experiment.

# 1. Dataset simulation

We generated 4,000 observations per dataset, with 2,000 observations in each class. Each observation has two coordinates, X1 and X2, and a label Y in {0, 1}. We used a stratified 70/30 split: 2,800 training observations and 1,200 test observations. Stratification preserves the equal class proportions.

For the linear dataset, we sampled two disks of radius 0.8, centered at (-2, -2) and (2, 2). Their centers are about 5.66 units apart, so the disks cannot overlap. We sampled angles uniformly and used radius = 0.8 sqrt(U), where U is uniform on [0, 1]. The square root gives uniform sampling over the disk area.

For the circular dataset, class 0 occupies a disk with radius at most 0.8. Class 1 occupies an annulus, a ring with inner radius 1.2 and outer radius 2.0. Its radius is sqrt(1.2^2 + (2.0^2 - 1.2^2)U). We interpret the assignment's circular "overlap" as one class surrounding the other, as shown in its example. The coordinate ranges overlap, but the clean class regions have a radial gap.

![Figure 1. Simulated datasets. Blue: class 0; orange: class 1.](figures/01_datasets.png)

This construction deliberately separates the effect of geometry from measurement noise. Both datasets are separable, but the circular dataset requires a nonlinear boundary.

# 2. Perceptron and linear SVM

Both models predict a class from the sign of w^T x + b. A perceptron updates its weights when an observation is misclassified. A linear SVM balances a wide margin against classification errors. For labels encoded as -1 and +1, its objective is 0.5 ||w||^2 + C sum max(0, 1 - y_i(w^T x_i + b)). The margin is the distance between the boundary and the nearby class observations.

We standardized each feature using the training mean and standard deviation, then applied the same transformation to the test data. We used C = 1 for the linear SVM. The perceptron used a fixed seed, at most 2,000 passes, and tolerance 0.0001. These are fixed baseline settings, not settings selected from the test results.

| Dataset / model | Train accuracy | Test accuracy | Test F1 |
| --- | --- | --- | --- |
| linear / Perceptron | 100.00% | 100.00% | 1.000 |
| linear / Linear SVM | 100.00% | 100.00% | 1.000 |
| circular / Perceptron | 41.32% | 41.50% | 0.497 |
| circular / Linear SVM | 65.50% | 64.75% | 0.527 |

![Figure 2. Test observations and fitted boundaries (black). Shading indicates the predicted class.](figures/02_linear_boundaries.png)

Both linear models achieved 100% test accuracy on the two distant disks. On the circular dataset, the perceptron achieved 41.50% and the linear SVM achieved 64.75%. A straight line can classify part of the ring correctly, but cannot enclose the inner class. Its errors therefore reflect an unsuitable model shape, or underfitting. Increasing the number of training observations would not remove this geometric limitation. The perceptron is not guaranteed to converge to a separating solution on these nonlinearly separable inputs.

Accuracy is the fraction of correct predictions. F1 = 2TP / (2TP + FP + FN) balances precision and recall for class 1. TP means true positives, FP false positives, and FN false negatives. A model can have near-chance accuracy and a nonzero F1, especially when it predicts one class more often.

# 3. Separating the circular data with kernels

A kernel lets an SVM use a nonlinear boundary through similarities between observations. We compared a degree-two polynomial kernel, K(x, z) = (gamma x^T z)^2, with a radial basis function (RBF) kernel, K(x, z) = exp(-gamma ||x - z||^2). The polynomial kernel includes quadratic features, which suit a boundary based on x1^2 + x2^2. The RBF kernel can represent more flexible local boundaries.

For each kernel, we selected C from {0.1, 1, 10} and gamma from {0.1, 0.5, 1} using three-fold stratified cross-validation on the training set. Scaling was fitted separately inside each fold. We chose the highest validation accuracy; if the kernels tied, our rule was to prefer the degree-two polynomial because its structure matches the known radial geometry. We then fitted the selected settings on all training observations.

| Kernel | C | gamma | CV accuracy | Test accuracy | F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| POLY | 0.1 | 0.5 | 100.00% | 100.00% | 1.000 | 1.000 |
| RBF | 0.1 | 0.1 | 100.00% | 100.00% | 1.000 | 1.000 |

![Figure 3. Kernel SVM boundaries on the untouched circular test set.](figures/03_kernel_boundaries.png)

POLY confusion matrix: [[600, 0], [0, 600]]. Rows are true classes [0, 1]; columns are predicted classes [0, 1]. Training accuracy was 100.00%; the fitted model used 205 support vectors.

RBF confusion matrix: [[600, 0], [0, 600]]. Rows are true classes [0, 1]; columns are predicted classes [0, 1]. Training accuracy was 100.00%; the fitted model used 834 support vectors.

The selected model is the POLY SVM. Both kernels achieved perfect validation and test scores, so neither showed better predictive accuracy in this experiment. Our tie rule is a modeling choice, not evidence of a performance advantage. AUC measures how well the decision scores rank class 1 above class 0 across thresholds; it does not require probability estimates.

# 4. Gaussian augmentation and generalization

We kept the selected circular SVM fixed. For each replicate, we sampled 1,000 observations with replacement from the held-out circular test set, added independent Gaussian noise to both coordinates, and retained the original labels. We repeated this 10,000 times for each noise level, producing 10 million perturbed observations per level. Sampling with replacement allows repeated resampling of the finite test set.

The perturbation is x_aug = x + epsilon, with epsilon distributed as N(0, sigma^2 I). SD means standard deviation; pp in the tables means percentage points. We used low SD = 0.1 and high SD = 0.8 in the original coordinate units. The PDF does not specify numerical SD values. The low level is small relative to the 0.4 radial gap; the high level is twice that gap. We did not retrain on these augmented observations, so this is a robustness test rather than an augmentation-training experiment.

| Noise SD | Mean accuracy | Replicate SD | Central 95% of replicates | Pooled F1 |
| --- | --- | --- | --- | --- |
| 0.1 | 99.84% | 0.13 pp | 99.50% - 100.00% | 0.998 |
| 0.8 | 67.12% | 1.48 pp | 64.20% - 70.00% | 0.721 |

![Figure 4. One 1,000-observation replicate per noise level. Colors show retained original labels.](figures/04_augmentation.png)

Low noise mostly preserves the original geometry. High noise moves more observations across the learned boundary, so their retained labels become harder to infer from their new positions. This explains the accuracy loss without requiring the model to have memorized its training data.

The clean training and test results provide no evidence of a large overfitting gap on this easy simulation. However, success on clean data does not guarantee robustness to a different input distribution. The intervals above describe variation among noisy 1,000-point replicates; they are not confidence intervals over independently generated datasets. Reusing test observations does not create 10 million independent original observations.

# 5. Incorrect training labels

We returned to the original, unaugmented datasets. We selected 0%, 15%, 25%, or 50% of the training indices and randomly permuted labels within that subset. We refitted the linear SVM for the linear dataset and the selected kernel SVM for the circular dataset, keeping their settings fixed. Test inputs and labels were unchanged. Each nonzero corruption level was repeated with five seeds; the zero-noise baseline is deterministic.

Selecting a label for shuffling does not guarantee that it changes. With balanced binary classes, a selected label has approximately a 50% chance of retaining its value. Therefore, shuffling 50% of training labels changes roughly 25% of all training labels, rather than 50%. We report the observed changed fraction to make this distinction explicit.

| Dataset | Selected | Actually changed | Fit to noisy labels | Clean test accuracy |
| --- | --- | --- | --- | --- |
| linear | 0% | 0.00% | 100.00% | 100.00% +/- 0.00 pp |
| linear | 15% | 7.67% | 92.33% | 100.00% +/- 0.00 pp |
| linear | 25% | 12.60% | 87.40% | 100.00% +/- 0.00 pp |
| linear | 50% | 25.37% | 74.63% | 100.00% +/- 0.00 pp |
| circular | 0% | 0.00% | 100.00% | 100.00% +/- 0.00 pp |
| circular | 15% | 7.67% | 92.33% | 100.00% +/- 0.00 pp |
| circular | 25% | 12.60% | 87.38% | 100.00% +/- 0.00 pp |
| circular | 50% | 25.37% | 73.69% | 98.00% +/- 0.31 pp |

![Figure 5. Mean clean test accuracy; error bars show one SD over five label permutations.](figures/05_label_noise.png)

The linear SVM retained 100% clean test accuracy at all tested levels. The circular SVM also retained 100% through 25% selected labels, then decreased to 98.00% on average at 50%. The underlying class geometry is strong, and the labels remain mostly correct even at the highest selected fraction. A model can therefore retain high clean test accuracy while disagreeing with many corrupted training labels. Lower agreement with noisy training labels is not necessarily worse generalization.

These results do not imply that label errors are harmless in general. The datasets are balanced and have a clear class gap, and the permutation preserves training class counts. Targeted mistakes near the boundary, weaker class separation, or a higher actual corruption rate could have a larger effect. Five permutations measure only a limited part of the uncertainty.

# 6. Higher-dimensional inputs

We created 5-, 10-, and 15-dimensional versions of both original datasets, retaining the two original informative coordinates and adding independent N(0, 1) features that contain no label information. We also retained the 2D baseline. The features are nested: the extra coordinates used in 5D are also present in 10D and 15D. We used the same observations, split, corruption levels, and five label-permutation seeds as before.

This design isolates the effect of irrelevant dimensions. It does not test a different problem in which every new coordinate carries useful information. Each model standardizes all features using training data only. We kept the 2D SVM hyperparameters fixed to measure sensitivity of the existing model; dimension-specific retuning might improve performance.

| Dataset | Dimensions | 0% shuffled | 15% shuffled | 25% shuffled | 50% shuffled |
| --- | --- | --- | --- | --- | --- |
| linear | 2 | 100.00% | 100.00% | 100.00% | 100.00% |
| linear | 5 | 100.00% | 100.00% | 100.00% | 100.00% |
| linear | 10 | 100.00% | 100.00% | 100.00% | 100.00% |
| linear | 15 | 100.00% | 100.00% | 100.00% | 100.00% |
| circular | 2 | 100.00% | 100.00% | 100.00% | 98.00% |
| circular | 5 | 100.00% | 100.00% | 100.00% | 98.00% |
| circular | 10 | 100.00% | 99.92% | 99.58% | 97.20% |
| circular | 15 | 100.00% | 99.73% | 99.13% | 96.70% |

![Figure 6. Clean test accuracy versus input dimension. Error bars show one SD over label permutations.](figures/06_dimensions.png)

The linear dataset remained at 100% accuracy across all conditions. For the circular dataset with 50% selected labels, mean accuracy decreased from 98.00% in 2D to 96.70% in 15D; without label corruption, it remained 100%. Extra features give the model more ways to fit accidental patterns. For a degree-two polynomial kernel, the number of quadratic feature terms is d(d + 1)/2: it grows from 3 at d = 2 to 120 at d = 15. These extra terms may interact with incorrect labels even though they add no useful class information. A linear model can more easily ignore irrelevant coordinates when the original separation is large.

We should not assume accuracy must decrease at every step. The observed curves depend on the separation, sample size, regularization, and particular noise draws. Any small changes here are conditional on one generated pair of datasets; repeated independent dataset generation would be needed for a broader claim.

# Bonus: a 70% performance threshold

To make the threshold question precise, we increased the strength of each augmentation using effective SD = t times the base SD, where t = 0, 1, ..., 20. The base SDs are 0.1 and 0.8. At each point, we evaluated the same fixed circular SVM on 200 resamples of 1,000 test observations. We reused the random draws across settings to make comparisons less affected by simulation variation. The full 10,000 replicates were used for Section 4; this additional threshold sweep uses 200 per setting.

| Base SD | First t below 70% | Effective SD | Accuracy | SD crossing bracket |
| --- | --- | --- | --- | --- |
| 0.1 | 8 | 0.80 | 67.26% | (0.70, 0.80] |
| 0.8 | 1 | 0.80 | 67.26% | (0.00, 0.80] |

![Figure 7. Mean accuracy as the noise multiplier increases. Dashed line: the chosen 70% threshold.](figures/07_bonus.png)

The reported crossing is the first sampled point below 70%, not an exact threshold. A finer grid would locate it more precisely. Both curves describe the same Gaussian perturbation mechanism at different scales, so their multiplier thresholds should differ. Their effective-SD brackets are the more meaningful comparison. Increasing the number of replicates alone would not systematically reduce expected accuracy; increasing the noise strength changes the task.

This is a robustness threshold, not proof of overfitting. At high noise, the class distributions overlap strongly while the original labels stay fixed. Even a suitable clean-data classifier can then perform poorly.

<b>Reproducibility.</b> Run analysis.py to regenerate all measurements, plots, and results.json, then run build_report.py to regenerate this PDF and the editable Markdown answer. requirements.txt records the installed package versions. No test labels were used for model fitting or kernel selection. The source instructions are the supplied HW2 PDF; no external datasets or references were used.

<b>Before submission.</b> Add both authors' names and review the modeling choices, especially the definition of circular overlap, the two noise SDs, and the use of held-out augmentation as a robustness test. The assignment requests the final PDF and the Python scripts.
