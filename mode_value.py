import pandas as pd
import numpy as np
import os

os.chdir("/home/sanju/Desktop/ai/AI_Proj-master")
df = pd.read_csv("ff.csv")
missing_columns = df.columns[df.isna().any()].tolist()

df[missing_columns] = df[missing_columns].fillna(df[missing_columns].mode().iloc[0])

df.to_csv('Mode_filled.csv', index=False)
