import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import pickle

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load your pickled model and label encoder (model.pkl contains a tuple: (model, label_encoder))
model, label_encoder = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/details')
def details():
    return send_from_directory('static', 'details.html')

@app.route('/prediction')
def prediction():
    return send_from_directory('static', 'prediction.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json

        # Extract and convert inputs
        Gender = int(data.get("Gender"))
        Age = float(data.get("Age"))
        Patient = int(data.get("Patient"))
        Severity = int(data.get("Severity"))
        BreathShortness = int(data.get("BreathShortness"))
        VisualChanges = int(data.get("VisualChanges"))
        NoseBleeding = int(data.get("NoseBleeding"))
        Whendiagnosed = int(data.get("Whendiagnosed"))
        Systolic = float(data.get("Systolic"))
        Diastolic = float(data.get("Diastolic"))
        ControlledDiet = int(data.get("ControlledDiet"))

        # Arrange data into dataframe with proper columns
        features = np.array([[Gender, Age, Patient, Severity, BreathShortness,
                              VisualChanges, NoseBleeding, Whendiagnosed,
                              Systolic, Diastolic, ControlledDiet]])

        df = pd.DataFrame(features, columns=[
            'Gender', 'Age', 'Patient', 'Severity',
            'BreathShortness', 'VisualChanges', 'NoseBleeding',
            'Whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet'
        ])
        df.rename(columns={'Whendiagnosed': 'Whendiagnoused'}, inplace=True)


        # Predict class
        pred_class = model.predict(df)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        return jsonify({'prediction': f"Your Blood Pressure is {pred_label}"})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run on localhost port 5500
    app.run(debug=True, host='127.0.0.1', port=5500)
