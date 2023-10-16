import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data from a CSV
df = pd.read_csv('data\\new_data4.csv')

# 2. Splitting the data into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# 3. Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1265)

# 4. Load the Booster
booster = xgb.Booster()
booster.load_model('xgboost_model.json')

# 5. Convert the test data to DMatrix
dtest = xgb.DMatrix(X_test)

# 6. Predict on test set
y_pred_prob = booster.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# 7. Calculate and print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Initialize an XGBClassifier and replace its booster
dummy_model = xgb.XGBClassifier()
dummy_model.fit(X_train[:2], y_train[:2])  # Fit with dummy data
dummy_model._Booster = booster

# 9. Get feature importances
feature_importance = dummy_model.feature_importances_

# 10. Plotting
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance using XGBoost')
plt.show()

