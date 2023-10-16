import pandas as pd
import numpy as np

data = pd.read_csv("data\\diabetes.csv")

cols = ['Glucose', 'BloodPressure', 'SkinThickness','BMI']
# cols_age = ['Glucose', 'BloodPressure', 'SkinThickness']
# cols_noAge = ['Insulin', 'BMI']

# Replace 0 with NaN in the specified columns
data[cols] = data[cols].replace(0, np.NaN)
data['Insulin'] = data['Insulin'].replace(0, np.NaN)

# Replace NaN values with the mean values calculated based on age groups
# for col in cols:
#     data[col].fillna(data[col].mean(), inplace=True)
for col in cols:
    # First, fill based on the group means
    data[col].fillna(data.groupby(['Age', 'Outcome'])[col].transform('mean'), inplace=True)
    # Then, fill any remaining NaNs with the overall column mean
    data[col].fillna(data[col].mean(), inplace=True)
data['Insulin'].fillna(data.groupby(['Age', 'Outcome'])[col].transform('median'), inplace=True)
    
data['Insulin'].fillna(data[col].median(), inplace=True)
# for col in cols_noAge:
#     data[col].fillna(data[col].mean(), inplace=True)
# Write the modified data back to a new CSV
data.to_csv('data\\new_data.csv', index=False)
