from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and the preprocessor
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    year = int(request.form.get('Year'))
    present_price = float(request.form.get('Present_Price'))
    kms_driven = float(request.form.get('Kms_Driven'))
    fuel_type = request.form.get('Fuel_Type')
    seller_type = request.form.get('Seller_Type')
    transmission = request.form.get('Transmission')
    owner = int(request.form.get('Owner'))

    # Preprocess the data
    data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    # Apply preprocessing steps and predict
    prediction = model.predict(data)

    # Take the first value of prediction
    output = round(prediction[0], 2)

    return render_template('index.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
