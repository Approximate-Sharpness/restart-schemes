import numpy as np
import pandas as pd
from scipy.io import savemat

df_red = pd.read_csv('winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality-white.csv', sep=';')

df = pd.concat([df_red, df_white], ignore_index=True)

print(df['quality'].describe())

exit(0)

data = df.to_numpy(dtype=float)

X, y = data[:,:-1], data[:,-1]

y=y[:,np.newaxis]

savemat('winequality.mat', {'features' : X, 'labels' : y})
