import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
data = pd.read_csv('data\\new_data4.csv')

# 2. Overview of the dataset
print("==========================Dataset info===========================\n")
print(data.info())
print(data.describe())

# 3. Check for missing values
print("=====================Missing data=========================\n")
print(data.isnull().sum())

# 4. Visualize the distribution of the Outcome variable
sns.countplot(data['Outcome'])
plt.title('Distribution of Outcome (Diabetes)')
plt.savefig('img\\Outcome.png') # Save the plot
plt.close()  # Close the plot to prevent it from displaying
plt.show()

# 5. Distribution of continuous features
continuous_features = ['Pregnancies', 'BMI', 'Insulin', 'Age']  # just a few for demonstration
for feature in continuous_features:
    plt.figure(figsize=(10,5))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    #mplcursors.cursor(hover=True)
    plt.savefig(f'img\\{feature}_Distribution.png')  # Save the plot
    plt.close()  # Close the plot to prevent it from displaying
    plt.show()

# 6. Correlation heatmap to see how features correlate with each other and with the target
corr_matrix = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
#mplcursors.cursor(hover=True) 
plt.savefig('img\\Correlation_Heatmap.png')  # Save the plot
plt.close()  # Close the plot to prevent it from displaying
plt.show()

# 7. Pairplot to visualize pairwise relationship and distribution
sns.pairplot(data, hue='Outcome')
plt.savefig('img\\Outcome_relationship.png')  # Save the plot
plt.close()  # Close the plot to prevent it from displaying
plt.show()

