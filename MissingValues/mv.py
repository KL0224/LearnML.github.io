import numpy as np
import pandas as pd
#from sklearn.preprocessing import Imputer # No No
from sklearn.impute import SimpleImputer

data = pd.read_csv('dt.csv', header = None)
print(data)
X = data.values # Chuyển sang mảng
imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') # avg = mean or most_frequent(tần xuất lớn nhất)
imp.fit(X) # cần truyền mảng
result = imp.transform(X)
print("Ket qua:")
print(result)