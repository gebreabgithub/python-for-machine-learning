# HW02 SVM assignment

The report answers all six questions and the bonus question in the supplied PDF.

## Organized Jupyter notebook

Open [HW02_SVM_notebook.ipynb](HW02_SVM_notebook.ipynb) for all six questions and the bonus, organized into method, code, output, and interpretation. The notebook includes embedded figures and numerical tables. Run cells from top to bottom; a full run includes the 20-million-observation augmentation experiment.

Use a Python 3 kernel in JupyterLab or VS Code, or open it in Google Colab. For local setup:

```sh
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
jupyter lab HW02_SVM_notebook.ipynb
```

The notebook is self-contained and writes regenerated figures and results into `notebook_outputs/`. Its public copy omits passport IDs. The proposed contributions still require confirmation by both authors.

## Files

- `HW02_SVM_answers.pdf`: formatted answer; includes both authors' details and the GitHub link; confirm or correct the proposed contribution statement before submission.
- `HW02_SVM_answers.md`: editable text, tables, and links to figures.
- `analysis.py`: dataset simulation, model selection, evaluation, and figures.
- `build_report.py`: builds the PDF and Markdown from the saved results.
- `results.json`: numerical experiment output, including confusion matrices and individual repeat scores.
- `figures/`: all seven figures used in the report.
- `requirements.txt`: package versions used for the run.
- `run.log`: output from the executed analysis.

## Reproduce

Tested with Python 3.12.14. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
python build_report.py
```

Scripts resolve outputs relative to their own location. The analysis evaluates 20 million augmented observations plus a separate threshold sweep, so allow time for execution. It processes augmentation in batches rather than storing all observations in memory.

## Choices that need to be understood

- 4,000 points per dataset; the assignment gives 10,000 as an example, not a requirement.
- Circular data means a central disk surrounded by an annulus. Their coordinate ranges overlap, but their clean class regions do not.
- 70/30 stratified train/test split, with training-only scaling and cross-validation.
- Gaussian coordinate SDs of 0.1 and 0.8; the source does not supply numerical levels.
- 10,000 resamples of 1,000 held-out observations for each noise level. Labels stay fixed; the model is not retrained on augmentations.
- Shuffling permutes a selected subset of training labels. The fraction actually changed is reported separately.
- Higher dimensions append irrelevant Gaussian coordinates; the informative coordinates remain unchanged.
- The bonus changes noise strength, not the number of Monte Carlo replicates.

The augmentation test measures robustness to input perturbations, which alone cannot establish overfitting. Repeated corruptions use one fixed pair of datasets; they do not estimate variability across independently generated datasets.
