
import os


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
working_directory = os.getcwd()

# List all CSV files in the directory
csv_files = [file for file in os.listdir() if file.endswith('.CSV')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through each CSV file and append its DataFrame to the list
for file in csv_files:
    file_path = os.path.join(working_directory, file)
    df = pd.read_csv(file_path)
    # Strip leading and trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    dfs.append(df)

# Concatenate all DataFrames in the list
agg_df = pd.concat(dfs, ignore_index=True)

# Strip leading and trailing whitespaces from column names in the aggregated DataFrame
agg_df.columns = agg_df.columns.str.strip()

# Assuming you have columns for the following features
# Adjust column names based on your actual dataset
agg_df['Maintenance'] = np.where(
    (agg_df['T'] < 16.2) | (agg_df['T'] > 16.3) |  # Adjust the threshold for 'T'
    (agg_df['CJC'] < 67) | (agg_df['CJC'] > 10003) |  # Adjust the threshold for 'CJC'
    (agg_df['PV'] < 0.024) | (agg_df['PV'] > 0.04) |  # Adjust the threshold for 'PV'
    (agg_df['PSI'] < 0.6) | (agg_df['PSI'] > 1.2),  # Adjust the threshold for 'PSI'
    1,  # Maintenance required
    0   # No maintenance required
)

# Display the first few rows of the DataFrame
print(agg_df.head())

# Continue with the machine learning part as before
X = agg_df[['T', 'CJC', 'PV', 'PSI']]
y = agg_df['Maintenance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)


sample_test_data = pd.DataFrame({
    'T': [16.25],
    'CJC': [68.00],
    'PV': [0.029],
    'PSI': [0.805]
})

# Make predictions on the sample test data
sample_prediction = model.predict(sample_test_data)

# Display the result
print("Sample Test Data:")
print(sample_test_data)
print("Predicted Maintenance:", sample_prediction[0])




plt.hist(agg_df['PSI'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of PSI')
plt.xlabel('PSI')
plt.ylabel('Frequency')
plt.show()

# Box plot for 'PSI'
sns.boxplot(x='Maintenance', y='PSI', data=agg_df)
plt.title('Box Plot of PSI by Maintenance')
plt.show()

# Correlation Matrix
correlation_matrix = agg_df[['T', 'CJC', 'PV', 'PSI', 'Maintenance']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature Importance Plot
feature_importances = model.feature_importances_
features = X.columns
plt.bar(features, feature_importances)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()