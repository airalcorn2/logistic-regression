import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

data = fetch_openml("titanic")
X = data["data"]
y = ["survived" if tgt == "1" else "-1" for tgt in data["target"]]

clf = LogisticRegression(penalty=None)
clf.fit(X, y)
y_hat = clf.predict(X)

cm = confusion_matrix(y, y_hat)
cmd = ConfusionMatrixDisplay(cm, display_labels=["did not survive", "survived"])
cmd.plot()
plt.show()

probs = clf.predict_proba(X)[:, 1]
(precision, recall, thresholds) = precision_recall_curve(y, probs, pos_label="survived")
prd = PrecisionRecallDisplay.from_predictions(
    y, probs, name="Logistic Regression", pos_label="survived", plot_chance_level=True
)
plt.show()

rfc_disp = RocCurveDisplay.from_predictions(
    y, probs, name="Logistic Regression", pos_label="survived", plot_chance_level=True
)
plt.show()
