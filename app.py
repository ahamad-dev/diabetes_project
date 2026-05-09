from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    lr_model  = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    dt_model  = joblib.load(os.path.join(MODEL_DIR, 'decision_tree.pkl'))
    scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model files not found. Run the Jupyter notebook first to train & save models.")
    lr_model = dt_model = scaler = None

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if lr_model is None:
        return jsonify({'error': 'Models not loaded. Please run the notebook first.'}), 500

    try:
        data = request.json
        input_values = [float(data[f]) for f in FEATURES]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        algorithm = data.get('algorithm', 'logistic_regression')

        if algorithm == 'logistic_regression':
            model = lr_model
            model_name = 'Logistic Regression'
        else:
            model = dt_model
            model_name = 'Decision Tree'

        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1]) * 100

        result = {
            'model_name': model_name,
            'prediction': prediction,
            'label': '🔴 Diabetic' if prediction == 1 else '🟢 Not Diabetic',
            'probability': round(probability, 2),
            'risk_level': get_risk_level(probability),
            'input_values': dict(zip(FEATURES, input_values))
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_level(prob):
    if prob < 30:
        return {'level': 'Low', 'color': '#4CAF50', 'message': 'Low risk of diabetes.'}
    elif prob < 60:
        return {'level': 'Moderate', 'color': '#FF9800', 'message': 'Moderate risk. Consult a doctor.'}
    else:
        return {'level': 'High', 'color': '#F44336', 'message': 'High risk. Immediate medical attention recommended.'}

@app.route('/model_info')
def model_info():
    return jsonify({
        'models': [
            {
                'id': 'logistic_regression',
                'name': 'Logistic Regression',
                'description': 'A statistical model for binary classification using a sigmoid function.',
                'pros': ['Fast & interpretable', 'Works well with linearly separable data', 'Low computational cost'],
                'typical_accuracy': '~77%'
            },
            {
                'id': 'decision_tree',
                'name': 'Decision Tree',
                'description': 'A tree-based model that splits data based on feature thresholds.',
                'pros': ['Easy to visualize', 'Handles non-linear relationships', 'No feature scaling needed'],
                'typical_accuracy': '~74%'
            }
        ],
        'features': FEATURES,
        'dataset': 'Pima Indians Diabetes Database (Kaggle)',
        'target': 'Outcome (0 = No Diabetes, 1 = Diabetes)'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
