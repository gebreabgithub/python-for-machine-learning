"""Reproduce HW2 experiments. Run: python analysis.py (outputs beside this script)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
SEED, N, REPEATS, K = 42, 4000, 5, 10000
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#2369a0', '#da7535']

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / (name + '.png'), dpi=170)
    plt.close()

def scatter(ax, x, y, title):
    for label in (0, 1):
        ax.scatter(*x[y == label].T, s=7, alpha=.4, color=COLORS[label], label=f'Class {label}')
    ax.set(xlabel='X1', ylabel='X2', title=title)
    ax.set_aspect('equal', adjustable='box')

def generate(kind, seed):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], N // 2)
    theta = rng.uniform(0, 2 * np.pi, N)
    if kind == 'linear':
        radius = .8 * np.sqrt(rng.random(N))
        x = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        x += np.where(y[:, None] == 0, [-2., -2.], [2., 2.])
    else:
        # Uniform area sampling: central disk and surrounding annulus.
        radius = np.where(y == 0, .8 * np.sqrt(rng.random(N)),
                          np.sqrt(1.2**2 + (2.**2 - 1.2**2) * rng.random(N)))
        x = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    return x, y

def svm(kernel, **kwargs):
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, cache_size=512, **kwargs))

def metrics(model, x, y):
    pred = model.predict(x)
    return dict(accuracy=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, model.decision_function(x)), confusion=confusion_matrix(y, pred).tolist())

def boundary(ax, model, x, y, title):
    lo, hi = x.min(axis=0) - .4, x.max(axis=0) + .4
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
    score = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, score > 0, levels=[-.5, .5, 1.5], colors=COLORS, alpha=.13)
    if score.min() < 0 < score.max(): ax.contour(xx, yy, score, levels=[0], colors='black', linewidths=1)
    scatter(ax, x, y, title)

def shuffle_labels(y, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), round(fraction * len(y)), replace=False)
    noisy = y.copy()
    noisy[indices] = rng.permutation(noisy[indices])
    return noisy, float(np.mean(noisy != y))

results = {'settings': dict(seed=SEED, n=N, repeats=REPEATS, k=K), 'linear_models': {}, 'kernels': {}}
data, splits, models = {}, {}, {}
for i, kind in enumerate(['linear', 'circular']):
    x, y = generate(kind, SEED + i)
    data[kind] = (x, y)
    splits[kind] = train_test_split(x, y, test_size=.3, stratify=y, random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, (kind, (x, y)) in zip(axes, data.items()): scatter(ax, x, y, kind.capitalize())
axes[1].legend(loc='upper right')
save('01_datasets')

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for row, (kind, (xt, xv, yt, yv)) in enumerate(splits.items()):
    for col, (name, model) in enumerate([
        ('Perceptron', make_pipeline(StandardScaler(), Perceptron(random_state=SEED, max_iter=2000, tol=1e-4))),
        ('Linear SVM', svm('linear', C=1))]):
        model.fit(xt, yt)
        results['linear_models'][kind + ' / ' + name] = {'train': metrics(model, xt, yt), 'test': metrics(model, xv, yv)}
        models[kind + name] = model
        boundary(axes[row, col], model, xv, yv, f'{kind}: {name}')
save('02_linear_boundaries')
print('Linear comparisons complete', flush=True)

xt, xv, yt, yv = splits['circular']
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for kernel in ['poly', 'rbf']:
    model = svm(kernel, degree=2, coef0=0)
    search = GridSearchCV(model, {'svc__C': [.1, 1, 10], 'svc__gamma': [.1, .5, 1]}, cv=cv, scoring='accuracy', n_jobs=1)
    search.fit(xt, yt)
    models[kernel] = search.best_estimator_
    results['kernels'][kernel] = {'cv_accuracy': search.best_score_, 'params': search.best_params_,
        'support_vectors': int(sum(search.best_estimator_[-1].n_support_)),
        'train': metrics(search.best_estimator_, xt, yt), 'test': metrics(search.best_estimator_, xv, yv)}
# Prespecified tie rule: prefer degree-two polynomial, matching the radial geometry.
best = max(['poly', 'rbf'], key=lambda key: (results['kernels'][key]['cv_accuracy'], key == 'poly'))
results['best_kernel'] = best
best_model = models[best]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, kernel in zip(axes, ['poly', 'rbf']): boundary(ax, models[kernel], xv, yv, kernel.upper() + ' SVM')
save('03_kernel_boundaries')
print('Kernel selection:', best, results['kernels'], flush=True)

# Stress-test a fixed model on held-out observations; never augment training points here.
results['augmentation'] = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, sigma in zip(axes, [.1, .8]):
    rng = np.random.default_rng(100)
    scores = []
    pooled_cm = np.zeros((2, 2), dtype=np.int64)
    for batch in range(K // 100):
        # Process 100 independent replicates at once to keep memory bounded.
        ids = rng.choice(len(yv), size=(100, 1000), replace=True)
        points = xv[ids] + rng.normal(0, sigma, (100, 1000, 2))
        pred = best_model.predict(points.reshape(-1, 2)).reshape(100, 1000)
        labels = yv[ids]
        scores.extend(np.mean(pred == labels, axis=1).tolist())
        np.add.at(pooled_cm, (labels.ravel(), pred.ravel()), 1)
        if batch == 0: scatter(ax, points[0], labels[0], f'Gaussian noise SD = {sigma}')
    tn, fp, fn, tp = pooled_cm.ravel()
    results['augmentation'][str(sigma)] = dict(mean_accuracy=float(np.mean(scores)), sd_accuracy=float(np.std(scores, ddof=1)),
        interval=np.quantile(scores, [.025, .975]).tolist(), f1=float(2*tp/(2*tp+fp+fn)), confusion=pooled_cm.tolist())
    print('Augmentation complete', sigma, results['augmentation'][str(sigma)], flush=True)
save('04_augmentation')

results['noise'] = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, kind in zip(axes, ['linear', 'circular']):
    xt, xv, yt, yv = splits[kind]
    rows = []
    for fraction in [0, .15, .25, .5]:
        trials = []
        for repeat in range(REPEATS):
            noisy, changed = shuffle_labels(yt, fraction, 200 + repeat)
            model = svm('linear', C=1) if kind == 'linear' else svm(best, degree=2, coef0=0, **{k.split('__')[1]: v for k,v in results['kernels'][best]['params'].items()})
            model.fit(xt, noisy)
            trials.append(dict(changed=changed, train_noisy=accuracy_score(noisy, model.predict(xt)), **metrics(model, xv, yv)))
        rows.append(dict(fraction=fraction, trials=trials))
    results['noise'][kind] = rows
    ax.errorbar([100*r['fraction'] for r in rows], [np.mean([t['accuracy'] for t in r['trials']]) for r in rows],
                yerr=[np.std([t['accuracy'] for t in r['trials']], ddof=1) for r in rows], marker='o', capsize=3)
    ax.set(title=kind.capitalize(), xlabel='Training labels selected for shuffling (%)', ylabel='Clean test accuracy', ylim=(.45, 1.03))
save('05_label_noise')
print('Label noise complete', flush=True)

results['dimensions'] = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
for ax, kind in zip(axes, ['linear', 'circular']):
    x, y = data[kind]
    # Nested irrelevant features: 5D is a subset of 10D and 15D.
    extra = np.random.default_rng(300).normal(size=(N, 13))
    rows = []
    for dim in [2, 5, 10, 15]:
        expanded = np.column_stack([x, extra[:, :dim-2]])
        xt, xv, yt, yv = train_test_split(expanded, y, test_size=.3, stratify=y, random_state=SEED)
        for fraction in [0, .15, .25, .5]:
            scores = []
            for repeat in range(REPEATS):
                noisy, _ = shuffle_labels(yt, fraction, 200 + repeat)
                model = svm('linear', C=1) if kind == 'linear' else svm(best, degree=2, coef0=0, **{k.split('__')[1]:v for k,v in results['kernels'][best]['params'].items()})
                model.fit(xt, noisy)
                scores.append(accuracy_score(yv, model.predict(xv)))
            rows.append(dict(dim=dim, fraction=fraction, scores=scores))
        print('Dimensions complete', kind, dim, flush=True)
    results['dimensions'][kind] = rows
    for fraction in [0, .15, .25, .5]:
        subset = [r for r in rows if r['fraction'] == fraction]
        ax.errorbar([r['dim'] for r in subset], [np.mean(r['scores']) for r in subset],
                    yerr=[np.std(r['scores'], ddof=1) for r in subset], marker='o', capsize=2, label=f'{100*fraction:g}% shuffled')
    ax.set(title=kind.capitalize(), xlabel='Number of input dimensions', ylabel='Clean test accuracy', xticks=[2,5,10,15], ylim=(.45, 1.03))
axes[1].legend(fontsize=8)
save('06_dimensions')

# Bonus: progressively increase the SD of both specified augmentation regimes.
xt, xv, yt, yv = splits['circular']
results['bonus'] = []
for multiplier in np.arange(0, 21, 1):
    for base_sigma in [.1, .8]:
        rng = np.random.default_rng(400)
        ids = rng.choice(len(yv), size=(200, 1000), replace=True)
        points = xv[ids] + rng.normal(size=(200,1000,2)) * base_sigma * multiplier
        pred = best_model.predict(points.reshape(-1,2)).reshape(200,1000)
        values = np.mean(pred == yv[ids], axis=1)
        results['bonus'].append(dict(multiplier=int(multiplier), base_sd=base_sigma, sd=float(multiplier*base_sigma), accuracy=float(np.mean(values))))
fig, ax = plt.subplots(figsize=(8, 3.4))
for sigma in [.1,.8]:
    subset = [r for r in results['bonus'] if r['base_sd'] == sigma]
    ax.plot([r['multiplier'] for r in subset], [r['accuracy'] for r in subset], marker='.', label=f'Base SD {sigma}')
ax.axhline(.7, linestyle='--', color='black', label='70% threshold')
ax.set(xlabel='Noise multiplier t (effective SD = t × base SD)', ylabel='Mean accuracy', ylim=(.45,1.03))
ax.legend()
save('07_bonus')
(OUT / 'results.json').write_text(json.dumps(results, indent=2))
print('All experiments saved to', OUT / 'results.json', flush=True)
