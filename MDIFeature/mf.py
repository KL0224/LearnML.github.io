from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as mat

boston = load_boston()
X = boston['data']
Y = boston['target']
names = boston['feature_names']
rf = RandomForestRegressor()
rf.fit(X, Y)

forest_importances = pd.Series(rf.feature_importances_, index = boston.feature_names)
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis = 0)

fig, ax = mat.subplots()
forest_importances.plot.bar(yerr = std, ax = ax)
ax.set_title("Feature importances using MDI")
ax.set_ylable("Mean decrease in impurity")
fig.tigh_layout()