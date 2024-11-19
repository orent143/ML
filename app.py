from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# Directory to store uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load models
with open('pkl/salary_predictor.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('pkl/salary_naive_bayes.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('pkl/salary_knn.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('pkl/salary_svr.pkl', 'rb') as f:
    svr_model = pickle.load(f)

with open('pkl/salary_dt.pkl', 'rb') as f:
    dt_model = pickle.load(f)

# Label encoder for Naive Bayes
encoder = LabelEncoder()
encoder.fit(['Low', 'Medium', 'High', 'Very High'])  # Ensure this matches Naive Bayes encoding

# Global list to store employees' prediction data
employees_data = []

# Function to predict salary using Linear Regression (in PHP)
def predict_salary_linear(education_level, years_experience, country):
    education_map = {'High School': 0, 'Associate Degree': 1, "Bachelor's Degree": 2, 'Master\'s Degree': 3, 'PhD': 4}
    country_map = {'USA': 0, 'Canada': 1, 'India': 2, 'Philippines': 3}

    education_num = education_map.get(education_level, 0)
    country_num = country_map.get(country, 0)

    input_features = np.array([[education_num, years_experience, country_num]])
    predicted_salary = linear_model.predict(input_features)
    return round(predicted_salary[0], 2)  # Return salary directly in PHP

# Naive Bayes (categorical prediction)
def predict_salary_nb(education_level, years_experience, country):
    education_map = {'High School': 0, 'Associate Degree': 1, "Bachelor's Degree": 2, 'Master\'s Degree': 3, 'PhD': 4}
    country_map = {'USA': 0, 'Canada': 1, 'India': 2, 'Philippines': 3}

    education_num = education_map.get(education_level, 0)
    country_num = country_map.get(country, 0)

    input_features = np.array([[education_num, years_experience, country_num]])

    # Get the salary category prediction from Naive Bayes
    salary_category = nb_model.predict(input_features)[0]

    # Return only the category (e.g., 'Low', 'Medium', 'High', 'Very High')
    return salary_category


# Function to predict salary using Decision Tree (in PHP)
def predict_salary_dt(education_level, years_experience, country):
    education_map = {'High School': 0, 'Associate Degree': 1, "Bachelor's Degree": 2, 'Master\'s Degree': 3, 'PhD': 4}
    country_map = {'USA': 0, 'Canada': 1, 'India': 2, 'Philippines': 3}

    education_num = education_map.get(education_level, 0)
    country_num = country_map.get(country, 0)

    input_features = np.array([[education_num, years_experience, country_num]])
    predicted_salary = dt_model.predict(input_features)
    return round(predicted_salary[0], 2)  # Return salary directly in PHP

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global employees_data  # Access the global employees data list

    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the file temporarily
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(filename)

            # Process each row in the CSV and predict salaries
            employees_data = []  # Clear the list before processing new data
            for index, row in df.iterrows():
                name = row['name']
                position = row['position']
                education_level = row['education_level']
                years_experience = row['years_experience']
                country = row['country']

                # Predict salary using selected models
                linear_salary = predict_salary_linear(education_level, years_experience, country)
                nb_salary_category = predict_salary_nb(education_level, years_experience, country)
                dt_salary = predict_salary_dt(education_level, years_experience, country)

                # Add employee data with predictions to the list
                employees_data.append({
                    'id': index + 1,  # Adjust the ID logic as needed
                    'name': name,
                    'position': position,
                    'linear_salary_php': round(linear_salary, 2),
                    'nb_salary_category': nb_salary_category,  # Display only the category
                    'dt_salary_php': round(dt_salary, 2)
                })

            flash('File successfully uploaded and predictions made!', 'success')

    # If it's a GET request, just render the empty prediction form or with data
    return render_template('salary_prediction.html', employees=employees_data)


@app.route('/employees')
def employees_page():
    return render_template('employees.html', employees=employees_data)

# Route for home page
@app.route('/')
def home():
    return redirect(url_for('predict'))  # Redirect to /predict to show the prediction form

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
