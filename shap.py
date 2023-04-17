import pandas as pd
import numpy as np
import os
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer 
os.chdir("/home/sanju/Desktop/ai/AI_Proj-master")

df = pd.read_csv("KNNCorrected_2.5.csv")
# df = pd.read_csv("SmallerDataSet.csv")
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']


model = LogisticRegression()
model.fit(X, y)



explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values)
