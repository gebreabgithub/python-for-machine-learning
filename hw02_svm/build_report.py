"""Build the PDF and editable Markdown from the results actually saved by analysis.py."""
from pathlib import Path
import json
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'results.json').read_text())
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10.2, leading=14.2, spaceAfter=8))
styles.add(ParagraphStyle(name='CaptionCustom', fontName='Helvetica', fontSize=8.6, leading=11, textColor=colors.HexColor('#425366'), spaceAfter=9))
styles['Heading1'].fontSize = 19
styles['Heading1'].leading = 23
styles['Heading1'].textColor = colors.HexColor('#163e61')
styles['Heading2'].fontSize = 13
story, md = [], []

def title(text):
    story.append(Paragraph(text, styles['Heading1']))
    md.append('# ' + text + '\n')

def p(text):
    story.append(Paragraph(text, styles['BodyCustom']))
    md.append(text + '\n')

def fig(name, caption, max_height=265):
    path = ROOT/'figures'/f'{name}.png'
    width,height = ImageReader(str(path)).getSize()
    scale=min(475/width,max_height/height)
    story.append(Image(str(path),width=width*scale,height=height*scale))
    story.append(Paragraph(caption,styles['CaptionCustom']))
    md.append(f'![{caption}](figures/{name}.png)\n')

def table(headers, rows, widths=None):
    content = [[Paragraph(str(v),styles['CaptionCustom']) for v in row] for row in [headers]+rows]
    t = Table(content, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8096a8')),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,8)])
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')

def page(): story.append(PageBreak())
def pct(x): return f'{100*x:.2f}%'
def avg(trials,key): return float(np.mean([t[key] for t in trials]))
def std(trials,key): return float(np.std([t[key] for t in trials],ddof=1))

# Page 1
title('HW2: Support Vector Machines')
p('Machine Learning &amp; Neural Networks for Neuroscience<br/>Pair assignment | Authors: ____________________ and ____________________')
p('We study how the shape of the data, feature noise, incorrect labels, and extra dimensions affect classification. All numbers below come from the accompanying Python analysis. The main random seed is 42; separate fixed seeds control each experiment.')
title('1. Dataset simulation')
p('We generated 4,000 observations per dataset, with 2,000 observations in each class. Each observation has two coordinates, X1 and X2, and a label Y in {0, 1}. We used a stratified 70/30 split: 2,800 training observations and 1,200 test observations. Stratification preserves the equal class proportions.')
p('For the linear dataset, we sampled two disks of radius 0.8, centered at (-2, -2) and (2, 2). Their centers are about 5.66 units apart, so the disks cannot overlap. We sampled angles uniformly and used radius = 0.8 sqrt(U), where U is uniform on [0, 1]. The square root gives uniform sampling over the disk area.')
p('For the circular dataset, class 0 occupies a disk with radius at most 0.8. Class 1 occupies an annulus, a ring with inner radius 1.2 and outer radius 2.0. Its radius is sqrt(1.2^2 + (2.0^2 - 1.2^2)U). We interpret the assignment\'s circular "overlap" as one class surrounding the other, as shown in its example. The coordinate ranges overlap, but the clean class regions have a radial gap.')
fig('01_datasets','Figure 1. Simulated datasets. Blue: class 0; orange: class 1.',220)
p('This construction deliberately separates the effect of geometry from measurement noise. Both datasets are separable, but the circular dataset requires a nonlinear boundary.')

page(); title('2. Perceptron and linear SVM')
p('Both models predict a class from the sign of w^T x + b. A perceptron updates its weights when an observation is misclassified. A linear SVM balances a wide margin against classification errors. For labels encoded as -1 and +1, its objective is 0.5 ||w||^2 + C sum max(0, 1 - y_i(w^T x_i + b)). The margin is the distance between the boundary and the nearby class observations.')
p('We standardized each feature using the training mean and standard deviation, then applied the same transformation to the test data. We used C = 1 for the linear SVM. The perceptron used a fixed seed, at most 2,000 passes, and tolerance 0.0001. These are fixed baseline settings, not settings selected from the test results.')
rows=[]
for name, a in r['linear_models'].items(): rows.append([name,pct(a['train']['accuracy']),pct(a['test']['accuracy']),f"{a['test']['f1']:.3f}"])
table(['Dataset / model','Train accuracy','Test accuracy','Test F1'],rows,[180,100,100,85])
fig('02_linear_boundaries','Figure 2. Test observations and fitted boundaries (black). Shading indicates the predicted class.',280)
p('Both linear models achieved 100% test accuracy on the two distant disks. On the circular dataset, the perceptron achieved 41.50% and the linear SVM achieved 64.75%. A straight line can classify part of the ring correctly, but cannot enclose the inner class. Its errors therefore reflect an unsuitable model shape, or underfitting. Increasing the number of training observations would not remove this geometric limitation. The perceptron is not guaranteed to converge to a separating solution on these nonlinearly separable inputs.')
p('Accuracy is the fraction of correct predictions. F1 = 2TP / (2TP + FP + FN) balances precision and recall for class 1. TP means true positives, FP false positives, and FN false negatives. A model can have near-chance accuracy and a nonzero F1, especially when it predicts one class more often.')

page(); title('3. Separating the circular data with kernels')
p('A kernel lets an SVM use a nonlinear boundary through similarities between observations. We compared a degree-two polynomial kernel, K(x, z) = (gamma x^T z)^2, with a radial basis function (RBF) kernel, K(x, z) = exp(-gamma ||x - z||^2). The polynomial kernel includes quadratic features, which suit a boundary based on x1^2 + x2^2. The RBF kernel can represent more flexible local boundaries.')
p('For each kernel, we selected C from {0.1, 1, 10} and gamma from {0.1, 0.5, 1} using three-fold stratified cross-validation on the training set. Scaling was fitted separately inside each fold. We chose the highest validation accuracy; if the kernels tied, our rule was to prefer the degree-two polynomial because its structure matches the known radial geometry. We then fitted the selected settings on all training observations.')
rows=[]
for kernel,a in r['kernels'].items():
    rows.append([kernel.upper(),str(a['params']['svc__C']),str(a['params']['svc__gamma']),pct(a['cv_accuracy']),pct(a['test']['accuracy']),f"{a['test']['f1']:.3f}",f"{a['test']['auc']:.3f}"])
table(['Kernel','C','gamma','CV accuracy','Test accuracy','F1','AUC'],rows,[60,35,45,95,95,60,70])
fig('03_kernel_boundaries','Figure 3. Kernel SVM boundaries on the untouched circular test set.',215)
for kernel,a in r['kernels'].items():
    cm=a['test']['confusion']
    p(f"{kernel.upper()} confusion matrix: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]. Rows are true classes [0, 1]; columns are predicted classes [0, 1]. Training accuracy was {pct(a['train']['accuracy'])}; the fitted model used {a['support_vectors']} support vectors.")
p(f"The selected model is the {r['best_kernel'].upper()} SVM. Both kernels achieved perfect validation and test scores, so neither showed better predictive accuracy in this experiment. Our tie rule is a modeling choice, not evidence of a performance advantage. AUC measures how well the decision scores rank class 1 above class 0 across thresholds; it does not require probability estimates.")

page(); title('4. Gaussian augmentation and generalization')
p('We kept the selected circular SVM fixed. For each replicate, we sampled 1,000 observations with replacement from the held-out circular test set, added independent Gaussian noise to both coordinates, and retained the original labels. We repeated this 10,000 times for each noise level, producing 10 million perturbed observations per level. Sampling with replacement allows repeated resampling of the finite test set.')
p('The perturbation is x_aug = x + epsilon, with epsilon distributed as N(0, sigma^2 I). SD means standard deviation; pp in the tables means percentage points. We used low SD = 0.1 and high SD = 0.8 in the original coordinate units. The PDF does not specify numerical SD values. The low level is small relative to the 0.4 radial gap; the high level is twice that gap. We did not retrain on these augmented observations, so this is a robustness test rather than an augmentation-training experiment.')
rows=[]
for sigma,a in r['augmentation'].items(): rows.append([sigma,pct(a['mean_accuracy']),f"{100*a['sd_accuracy']:.2f} pp",f"{pct(a['interval'][0])} - {pct(a['interval'][1])}",f"{a['f1']:.3f}"])
table(['Noise SD','Mean accuracy','Replicate SD','Central 95% of replicates','Pooled F1'],rows,[55,95,85,160,70])
fig('04_augmentation','Figure 4. One 1,000-observation replicate per noise level. Colors show retained original labels.',220)
p('Low noise mostly preserves the original geometry. High noise moves more observations across the learned boundary, so their retained labels become harder to infer from their new positions. This explains the accuracy loss without requiring the model to have memorized its training data.')
p('The clean training and test results provide no evidence of a large overfitting gap on this easy simulation. However, success on clean data does not guarantee robustness to a different input distribution. The intervals above describe variation among noisy 1,000-point replicates; they are not confidence intervals over independently generated datasets. Reusing test observations does not create 10 million independent original observations.')

page(); title('5. Incorrect training labels')
p('We returned to the original, unaugmented datasets. We selected 0%, 15%, 25%, or 50% of the training indices and randomly permuted labels within that subset. We refitted the linear SVM for the linear dataset and the selected kernel SVM for the circular dataset, keeping their settings fixed. Test inputs and labels were unchanged. Each nonzero corruption level was repeated with five seeds; the zero-noise baseline is deterministic.')
p('Selecting a label for shuffling does not guarantee that it changes. With balanced binary classes, a selected label has approximately a 50% chance of retaining its value. Therefore, shuffling 50% of training labels changes roughly 25% of all training labels, rather than 50%. We report the observed changed fraction to make this distinction explicit.')
rows=[]
for kind,data in r['noise'].items():
 for row in data:
    trials=row['trials']; rows.append([kind,f"{100*row['fraction']:g}%",pct(avg(trials,'changed')),pct(avg(trials,'train_noisy')),f"{pct(avg(trials,'accuracy'))} +/- {100*std(trials,'accuracy'):.2f} pp"])
table(['Dataset','Selected','Actually changed','Fit to noisy labels','Clean test accuracy'],rows,[65,60,95,105,140])
fig('05_label_noise','Figure 5. Mean clean test accuracy; error bars show one SD over five label permutations.',205)
p('The linear SVM retained 100% clean test accuracy at all tested levels. The circular SVM also retained 100% through 25% selected labels, then decreased to 98.00% on average at 50%. The underlying class geometry is strong, and the labels remain mostly correct even at the highest selected fraction. A model can therefore retain high clean test accuracy while disagreeing with many corrupted training labels. Lower agreement with noisy training labels is not necessarily worse generalization.')
p('These results do not imply that label errors are harmless in general. The datasets are balanced and have a clear class gap, and the permutation preserves training class counts. Targeted mistakes near the boundary, weaker class separation, or a higher actual corruption rate could have a larger effect. Five permutations measure only a limited part of the uncertainty.')

page(); title('6. Higher-dimensional inputs')
p('We created 5-, 10-, and 15-dimensional versions of both original datasets, retaining the two original informative coordinates and adding independent N(0, 1) features that contain no label information. We also retained the 2D baseline. The features are nested: the extra coordinates used in 5D are also present in 10D and 15D. We used the same observations, split, corruption levels, and five label-permutation seeds as before.')
p('This design isolates the effect of irrelevant dimensions. It does not test a different problem in which every new coordinate carries useful information. Each model standardizes all features using training data only. We kept the 2D SVM hyperparameters fixed to measure sensitivity of the existing model; dimension-specific retuning might improve performance.')
rows=[]
for kind,data in r['dimensions'].items():
 for dim in [2,5,10,15]:
    vals=[next(row for row in data if row['dim']==dim and row['fraction']==f) for f in [0,.15,.25,.5]]
    rows.append([kind,str(dim)]+[pct(np.mean(v['scores'])) for v in vals])
table(['Dataset','Dimensions','0% shuffled','15% shuffled','25% shuffled','50% shuffled'],rows,[65,70,82,82,82,84])
fig('06_dimensions','Figure 6. Clean test accuracy versus input dimension. Error bars show one SD over label permutations.',220)
p('The linear dataset remained at 100% accuracy across all conditions. For the circular dataset with 50% selected labels, mean accuracy decreased from 98.00% in 2D to 96.70% in 15D; without label corruption, it remained 100%. Extra features give the model more ways to fit accidental patterns. For a degree-two polynomial kernel, the number of quadratic feature terms is d(d + 1)/2: it grows from 3 at d = 2 to 120 at d = 15. These extra terms may interact with incorrect labels even though they add no useful class information. A linear model can more easily ignore irrelevant coordinates when the original separation is large.')
p('We should not assume accuracy must decrease at every step. The observed curves depend on the separation, sample size, regularization, and particular noise draws. Any small changes here are conditional on one generated pair of datasets; repeated independent dataset generation would be needed for a broader claim.')

page(); title('Bonus: a 70% performance threshold')
p('To make the threshold question precise, we increased the strength of each augmentation using effective SD = t times the base SD, where t = 0, 1, ..., 20. The base SDs are 0.1 and 0.8. At each point, we evaluated the same fixed circular SVM on 200 resamples of 1,000 test observations. We reused the random draws across settings to make comparisons less affected by simulation variation. The full 10,000 replicates were used for Section 4; this additional threshold sweep uses 200 per setting.')
rows=[]
for base in [.1,.8]:
    values=[a for a in r['bonus'] if a['base_sd']==base]
    crossed=next((a for a in values if a['accuracy']<.7),None)
    if crossed:
        previous=values[values.index(crossed)-1]
        rows.append([str(base),str(crossed['multiplier']),f"{crossed['sd']:.2f}",pct(crossed['accuracy']),f"({previous['sd']:.2f}, {crossed['sd']:.2f}]"])
    else: rows.append([str(base),'Not reached','-','-','-'])
table(['Base SD','First t below 70%','Effective SD','Accuracy','SD crossing bracket'],rows,[55,110,85,80,135])
fig('07_bonus','Figure 7. Mean accuracy as the noise multiplier increases. Dashed line: the chosen 70% threshold.',230)
p('The reported crossing is the first sampled point below 70%, not an exact threshold. A finer grid would locate it more precisely. Both curves describe the same Gaussian perturbation mechanism at different scales, so their multiplier thresholds should differ. Their effective-SD brackets are the more meaningful comparison. Increasing the number of replicates alone would not systematically reduce expected accuracy; increasing the noise strength changes the task.')
p('This is a robustness threshold, not proof of overfitting. At high noise, the class distributions overlap strongly while the original labels stay fixed. Even a suitable clean-data classifier can then perform poorly.')
p('<b>Reproducibility.</b> Run analysis.py to regenerate all measurements, plots, and results.json, then run build_report.py to regenerate this PDF and the editable Markdown answer. requirements.txt records the installed package versions. No test labels were used for model fitting or kernel selection. The source instructions are the supplied HW2 PDF; no external datasets or references were used.')
p('<b>Before submission.</b> Add both authors\' names and review the modeling choices, especially the definition of circular overlap, the two noise SDs, and the use of held-out augmentation as a robustness test. The assignment requests the final PDF and the Python scripts.')

def footer(canvas, doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#5b6b7a'))
    canvas.drawString(48,29,'HW2 | SVM experiments | Seed 42')
    canvas.drawRightString(547,29,str(doc.page))

SimpleDocTemplate(str(ROOT/'HW02_SVM_answers.pdf'),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'HW02_SVM_answers.md').write_text('\n'.join(md))
print('Built PDF and Markdown')
