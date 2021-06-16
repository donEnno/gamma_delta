# Default
from matplotlib import pyplot

# Pipeline
from partitions import get_frequencies
from projection import calculate_partitions

# ML
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Generate feature
partition = calculate_partitions(0, 'BLOSUM45', gamma=1.065)
x = get_frequencies(partition, ['BL', 'HD'], absolute_toggle=True)

# Set response
bl = [1 for i in range(66)]
hd = [0 for i in range(29)]
y = []
y.extend(bl)
y.extend(hd)

# Split into train/test sets
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.1, random_state=2)

# Fit model
model = LogisticRegressionCV(solver='liblinear', random_state=2, cv=5)
model.fit(trainX, trainY)

# Model score
print('Model score for testX/Y:')
print(model.score(testX, testY))
print('Model score for x/y:')
print(model.score(x, y))

# Predict probabilities
lr_probs = model.predict_proba(testX)
lr_probs = lr_probs[:, 1]                   # positive outcomes only

# Calculate scores
lr_auc = roc_auc_score(testY, lr_probs)
print('Logistic: ROC AUC=%.3f' % lr_auc)

# Calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(testY, lr_probs)

# Plot the curve for the model
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.savefig('/home/ubuntu/Enno/gammaDelta/plots/test_1.png')
pyplot.show()








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

c_report = classification_report(y, model.predict(x), output_dict=True)
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

"""
