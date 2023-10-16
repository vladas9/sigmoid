import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load data from a CSV
df = pd.read_csv('data\\new_data.csv')  # Replace 'your_data_file.csv' with your filename

# Splitting the data into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1265)

# 3. Train the model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
model.save_model('xgboost_model.json')
# Calculate and print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 2. Get user input for prediction
# input_data = {}
# for col in X.columns:
#    input_data[col] = [float(input(f"Enter {col}: "))]

# user_df = pd.DataFrame(input_data)

# 4. Predict using user input
# prediction = model.predict(user_df)
# if prediction[0] == 1:
#    print("The patient is likely to have diabetes.")
# else:
#    print("The patient is unlikely to have diabetes.")
model.save_model('xgboost_model.json')
# 5. Visualize the model (like feature importance)
feature_importance = model.feature_importances_
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance using XGBoost')
plt.show()
