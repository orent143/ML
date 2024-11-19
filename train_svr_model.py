import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pickle

# Load the data
df = pd.read_csv(r'C:\Users\ADMIN\Documents\salary-prediction\venv\salary_data.csv')

# Preprocess the data: Convert education levels and country into numerical values
education_levels = {
    'High School': 0,
    'Associate Degree': 1,
    "Bachelor's Degree": 2,
    'Master\'s Degree': 3,
    'PhD': 4
}

countries = {
    'USA': 0,
    'Canada': 1,
    'India': 2,
    'Philippines': 3  # Added Philippines here
}

# Map education level and country to numeric values
df['Education Level'] = df['Education Level'].map(education_levels)
df['Country'] = df['Country'].map(countries)

# Features and target variable
X = df[['Education Level', 'Years of Experience', 'Country']]  # Features (including Country)
y = df['Salary']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVR model
svr_model = SVR(kernel='rbf')

# Train the model
svr_model.fit(X_train, y_train)

# Save the trained SVR model using pickle
with open('salary_svr.pkl', 'wb') as f:
    pickle.dump(svr_model, f)

print("SVR model trained and saved successfully!")