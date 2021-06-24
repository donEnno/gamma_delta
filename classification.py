# Default
import joblib
from matplotlib import pyplot
import numpy as np

# Pipeline
from partitions import get_frequencies
from projection import calculate_partitions

# ML
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Generate feature
partition = joblib.load(fr"/home/ubuntu/Enno/mnt/volume/vectors/c_CO_1.02_communities")
x = get_frequencies(partition, ['BL', 'HD'], absolute_toggle=True)
print(len(np.unique(partition)))

# Set response
bl = [1 for i in range(66)]
fu = [1 for j in range(55)]
hd = [0 for k in range(29)]
y = []
y.extend(bl)
# y.extend(fu)
y.extend(hd)


def eval_model(model, testX, testY, c, roc=False):
    # Model score
    print(11*'= ', ' MODEL SCORES ', 11*' =', '\n')

    test_score = model.score(testX, testY)
    print('Model score for testX/Y:%.3f' % test_score)
    total_score = model.score(x, y)
    print('Model score for x/y: %.3f' % total_score)

    # Predict probabilities
    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]                   # positive outcomes only

    # Calculate scores
    lr_auc = roc_auc_score(testY, lr_probs)
    print('Logistic: ROC AUC=%.3f' % lr_auc)

    # Report
    c_report = classification_report(testY, model.predict(testX))
    print(c_report)

    print(sorted(model.coef_[0]))

    if roc:
        # Calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(testY, lr_probs)

        # Plot the curve for the model
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        pyplot.title(label=f'Logistic: ROC AUC=%.3f - C={c}' % lr_auc)
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()
        # pyplot.savefig('/home/ubuntu/Enno/gammaDelta/plots/test_1.png')
        pyplot.show()
        pyplot.clf()

    cm = confusion_matrix(testY, model.predict(testX))

    print('\t', '0', '\t', '1')
    print(12*'=')
    for i in range(2):
        print(i, ' \t', cm[i, 0], '\t', cm[i, 1])
        print(12 * '=')


if __name__ == '__main__':
    for c in [0.01]:
        print(10 * '= ', 'RUNNING ON C =', c, 10 * ' =')

        # Split into train/test sets
        trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

        # Fit model
        model = LogisticRegression(solver='lbfgs', n_jobs=-1, C=c, random_state=3, max_iter=5000)
        model.fit(trainX, trainY)

        eval_model(model, testX, testY, c)




"""model_score = model.score(x, y)
print(model_score)

cm = confusion_matrix(y, model.predict(x))
fig, ax = pyplot.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

c_report = classification_report(y, model.predict(x))
print(c_report)
"""
"""
model.classes_
model.intercept_
model.coef_
model.predict_proba(x)
model.predict(x)
model.score(x, y)

confusion_matrix(y, model.predict(x))

cm = confusion_matrix(y, model.predict(x))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

classification_report(y, model.predict(x), output_dict=True)
...
"""
