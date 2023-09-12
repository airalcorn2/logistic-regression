import statsmodels.api as sm

from sklearn.datasets import fetch_openml

data = fetch_openml("titanic")
X = data["data"]
X = sm.add_constant(X)
y = [1.0 if tgt == "1" else 0.0 for tgt in data["target"]]

logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit()
print(logit_res.summary())
