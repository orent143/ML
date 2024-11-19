import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load the data
df = pd.read_csv(r'C:\Users\ADMIN\Documents\salary-prediction\venv\salary_data.csv')

# Clean column names by stripping any leading/trailing spaces and removing unwanted characters
df.columns = df.columns.str.strip()  # Removes leading/trailing spaces
df.columns = df.columns.str.replace('\t', '', regex=False)  # Remove tabs

# Print the cleaned columns for debugging
print("Cleaned columns:", df.columns)

# Clean up any extra spaces or commas in the rows (if necessary)
df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

# Ensure the columns are correct
print("Columns in the dataset:", df.columns)

# Ensure that the expected columns are present
required_columns = ['Education Level', 'Years of Experience', 'Country', 'Salary']
for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"'{column}' column is missing from the CSV file")

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
    'Philippines': 3
}

# Map education level and country to numeric values
df['Education Level'] = df['Education Level'].map(education_levels)
df['Country'] = df['Country'].map(countries)

# Convert 'Salary' to numeric, handling commas and coercing errors to NaN (if any)
df['Salary'] = pd.to_numeric(df['Salary'].replace({',': ''}, regex=True), errors='coerce')

# Check for any missing values in 'Salary' and drop or fill them
df = df.dropna(subset=['Salary'])

# Create salary categories (in PHP)
bins = [0, 30000, 60000, 100000, 150000, float('inf')]
labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']
df['Salary Category'] = pd.cut(df['Salary'], bins=bins, labels=labels, right=False)

# Features and target variable
X = df[['Education Level', 'Years of Experience', 'Country']]
y = df['Salary Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Save the model
with open('salary_naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

print("Naive Bayes model trained and saved successfully!")
