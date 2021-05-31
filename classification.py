import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = []
y = []

# LogisticRegressionCV for cross-validation
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
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
