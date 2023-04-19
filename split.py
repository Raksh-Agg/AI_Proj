import pandas as pd
import numpy as np

# Load the CSV file into a pandas dataframe
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Split the data into two parts
n = len(df)
split_index = int(n*0.2)

# Shuffle the data to ensure randomness
df = df.sample(frac=1, random_state=82)

# Split the data into training and testing sets
train_data = df[:split_index]
test_data = df[split_index:]

# Save the training and testing sets to separate CSV files
train_data.to_csv('FinalTrain.csv', index=False)
test_data.to_csv('FinalTest.csv', index=False)