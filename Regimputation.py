import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

os.chdir("/home/sanju/Desktop/ai/AI_Proj-master")
df = pd.read_csv("ff.csv")
missing_columns = df.columns[df.isna().any()].tolist()

imputer = IterativeImputer(random_state=0)
df[missing_columns] = imputer.fit_transform(df[missing_columns])

new_df = pd.DataFrame()
new_df = df

new_df['Diabetes_012'] = np.round(new_df['Diabetes_012'])
new_df['HighBP'] = new_df['HighBP'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['HighChol'] = new_df['HighChol'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['CholCheck'] = new_df['CholCheck'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['BMI'] = np.round(new_df['BMI'])
new_df['Smoker'] = new_df['Smoker'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['Stroke'] = new_df['Stroke'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['HeartDiseaseorAttack'] = new_df['HeartDiseaseorAttack'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['PhysActivity'] = new_df['PhysActivity'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['Fruits'] = new_df['Fruits'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['Veggies'] = new_df['Veggies'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['HvyAlcoholConsump'] = new_df['HvyAlcoholConsump'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['AnyHealthcare'] = new_df['AnyHealthcare'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['NoDocbcCost'] = new_df['NoDocbcCost'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['GenHlth'] = np.round(new_df['GenHlth'])
new_df['MentHlth'] = np.round(new_df['MentHlth'])
new_df['PhysHlth'] = np.round(new_df['PhysHlth'])
new_df['DiffWalk'] = new_df['DiffWalk'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['Sex'] = new_df['Sex'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
new_df['Age'] = np.round(new_df['Age'])
new_df['Education'] = np.round(new_df['Education'])
new_df['Income'] = np.round(new_df['Income'])

df.to_csv('Mputed.csv', index=False)
