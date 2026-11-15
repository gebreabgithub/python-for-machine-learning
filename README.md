# HW2: Support Vector Machines

Gebreab Tewelday and Abraham Alemayehu Gebrelibanos

Machine Learning & Neural Networks for Neuroscience

## Files

- [Notebook](HW02_SVM_notebook.ipynb): code, explanations, tables, and figures for Questions 1-6 and the bonus.
- [Report](HW02_SVM_answers.pdf): written answers and results.
- [Requirements](requirements.txt): Python packages used by the notebook.

## Running the notebook

[Open in Google Colab](https://colab.research.google.com/github/gebreabgithub/python-for-machine-learning/blob/main/hw02_svm/HW02_SVM_notebook.ipynb), or open it locally in Jupyter or VS Code with a Python 3 kernel.

Install the packages from this folder:

```sh
python -m pip install -r requirements.txt
```

Run the cells from top to bottom. The complete run takes a few minutes because Question 4 evaluates 10,000 resamples of 1,000 observations at each of two noise levels. Figures and tables are displayed inside the notebook. It does not generate the PDF or write separate output files.

The experiments use 4,000 points per dataset, a fixed seed of 42, and a 70/30 stratified training/test split. The notebook was run with Python 3.12.14.
