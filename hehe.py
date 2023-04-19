import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error

# Load original.csv and imputed.csv into dataframes
original = pd.read_csv('SmallerDataSet.csv')
imputed = pd.read_csv('BayesianRidgeCorrected.csv')
imp = pd.read_csv('Corr_Mode.csv')

# Select the columns that you want to compare
columns = ['Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
           'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Calculate accuracy of imputation for each column separately
sum = 0
for column in columns:
    mse = mean_squared_error(original[column], imputed[column])
    accuracy = 1 - mse
    sum = sum + accuracy
    # print(f"Accuracy of imputation for {column}: {accuracy}")
# print("Accuracy is ", sum/21)
# print(original.corr()['Veggies'])
# print()
# print(imputed.corr()['Veggies'])
# print()
# print(imp.corr()['Veggies'])
arr = imp.corr()['Diabetes_012']
print(arr[1])
