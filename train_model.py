import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\ADMIN\Documents\salary-prediction\venv\salary_data.csv')

# Step 2: Data Preprocessing
# Check for missing values and handle them (drop rows with missing values in this case)
if df.isnull().sum().any():
    print("Warning: Data contains missing values.")
    df = df.dropna()  # Dropping rows with missing values (could also use fillna())

# Encoding categorical variables (Education Level and Country)
label_encoder_education = LabelEncoder()
label_encoder_country = LabelEncoder()

df['Education Level'] = label_encoder_education.fit_transform(df['Education Level'])
df['Country'] = label_encoder_country.fit_transform(df['Country'])

# Step 3: Feature selection and target definition
X = df[['Education Level', 'Years of Experience', 'Country']]  # Features
y = df['Salary']  # Target variable (Salary)

# Step 4: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training (using Linear Regression as an example)
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 7: Save the trained model using pickle
with open('salary_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained, evaluated, and saved successfully!")
