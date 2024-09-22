import numpy as np
import pandas as pd
'''
df = pd.read_csv('test.csv', header = None)
#print(df) # in ra hết sheet
#print(df[3]) # cột 3
df[3].to_csv('ouput.csv') # in ra file output.csv
'''

df = pd.DataFrame(
	[(1, 2, 1, 6),
	 (0, 3, 0, 7),
	 (2, 0, 4, 3), 
	 (1, 1, 1, 4)], columns = ['dogs', 'cats', 'bears', 'ducks'])
print(df)
print(df.cov()) # Tính hiệp phương sai
# Hiệp phương sai càng lớn thì càng tương quan, càng âm thi càng ít tương quan
