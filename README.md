# Python for Machine Learning

Python experiments and assignments for machine learning.

## HW02: Support Vector Machines

- [Organized Jupyter notebook: questions, code, figures, and answers](hw02_svm/HW02_SVM_notebook.ipynb)
- [Open notebook in Google Colab](https://colab.research.google.com/github/gebreabgithub/python-for-machine-learning/blob/main/hw02_svm/HW02_SVM_notebook.ipynb)
- [Main Python analysis](hw02_svm/analysis.py)
- [Report generation script](hw02_svm/build_report.py)
- [PDF answers](hw02_svm/HW02_SVM_answers.pdf)
- [Editable answers](hw02_svm/HW02_SVM_answers.md)
- [Results](hw02_svm/results.json)
- [Setup and reproduction instructions](hw02_svm/README.md)

The assignment covers simulated datasets, perceptron and SVM comparisons, kernels, Gaussian augmentation, label noise, higher dimensions, and a robustness threshold.

### Run

```sh
cd hw02_svm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
python build_report.py
```
