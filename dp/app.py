from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and feature names
model_path = 'diabetes_model.joblib'
feature_names_path = 'feature_names.joblib'

if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
    print("Error: Model or feature names not found. Please run train.py first.")
    model = None
    feature_names = None
else:
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    print("Model loaded successfully!")
    print(f"Model features: {feature_names}")

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(float(data[feature]))
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'probability_not_diabetic': float(probability[0]),
            'probability_diabetic': float(probability[1]),
            'confidence': float(max(probability)) * 100
        }
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
