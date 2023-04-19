import pandas as pd
import numpy as np
import os
os.chdir("/home/sanju/Desktop/ai/AI_Proj-master")
# Load the CSV file into a pandas dataframe
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Split the data into two parts
n = len(df)
split_index = int(n*0.8)

# Shuffle the data to ensure randomness
df = df.sample(frac=1, random_state=42)

# Split the data into training and testing sets
train_data = df[:split_index]
test_data = df[split_index:]

# Save the training and testing sets to separate CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)