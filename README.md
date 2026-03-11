# Diabetes Prediction System

A full-stack machine learning application that uses a Random Forest Classifier to predict diabetes risk based on medical parameters. The application includes a Flask backend, a modern web frontend, and a trained ML model.

## Features

- 🤖 **Random Forest Classifier**: Trained on diabetes dataset with 100 decision trees
- 🎯 **Accurate Predictions**: Provides prediction results along with confidence scores
- 💻 **Modern Web Interface**: Clean, responsive HTML/CSS frontend
- 🔌 **RESTful API**: Flask backend with prediction endpoint
- 📊 **Probability Distribution**: Displays probability scores for each class
- ⚡ **Fast Predictions**: Optimized model inference

## Project Structure

```
dp/
├── diabetes.csv              # Training dataset
├── train.py                  # Model training script
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── diabetes_model.joblib     # Trained model (generated after train.py)
├── feature_names.joblib      # Feature names (generated after train.py)
└── templates/
    └── index.html            # Web interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

Navigate to the project directory:

```bash
cd dp
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Before running the Flask server, you must train the model:

```bash
python train.py
```

This will:
- Load the diabetes.csv dataset
- Split data into 80% training and 20% validation sets
- Train a Random Forest Classifier with 100 trees
- Display model performance metrics:
  - Training Accuracy
  - Validation Accuracy
  - Precision, Recall, F1-Score
  - Feature Importance Rankings
- Save the trained model to `diabetes_model.joblib`
- Save feature names to `feature_names.joblib`

**Example Output:**
```
Loading dataset...
Dataset shape: (768, 10)
Column names: ['PatientID', 'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age', 'Diabetic']

Training set size: 614
Validation set size: 154

Training Random Forest Classifier...

--- Model Performance ---
Training Accuracy: 0.9967
Validation Accuracy: 0.8312
Validation Precision: 0.7826
Validation Recall: 0.7273
Validation F1-Score: 0.7536

--- Feature Importance ---
Feature                 Importance
PlasmaGlucose             0.276541
BMI                       0.198765
Age                       0.156234
DiabetesPedigree          0.145123
... (more features)

Model saved to diabetes_model.joblib
Feature names saved to feature_names.joblib
```

### Step 2: Run the Flask Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

**Example Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 3: Access the Web Interface

Open your web browser and navigate to:

```
http://localhost:5000
```

### Step 4: Make Predictions

1. Fill in the medical values in the form:
   - **Pregnancies**: Number of times pregnant
   - **Plasma Glucose**: Plasma glucose concentration (mg/dL)
   - **Diastolic Blood Pressure**: Blood pressure (mmHg)
   - **Triceps Thickness**: Skin fold thickness (mm)
   - **Serum Insulin**: 2-Hour serum insulin (µU/mL)
   - **BMI**: Body Mass Index (kg/m²)
   - **Diabetes Pedigree**: Diabetes pedigree function (numeric value)
   - **Age**: Age in years

2. Click the **"Predict"** button

3. View the results:
   - Prediction label (Diabetic / Not Diabetic)
   - Probability distribution
   - Confidence score

## API Endpoints

### GET /health

Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict

Make a diabetes prediction

**Request Body:**
```json
{
  "Pregnancies": 6,
  "PlasmaGlucose": 148,
  "DiastolicBloodPressure": 72,
  "TricepsThickness": 35,
  "SerumInsulin": 0,
  "BMI": 33.6,
  "DiabetesPedigree": 0.627,
  "Age": 50
}
```

**Response (Success):**
```json
{
  "prediction": 1,
  "prediction_label": "Diabetic",
  "probability_not_diabetic": 0.23,
  "probability_diabetic": 0.77,
  "confidence": 77.0
}
```

**Response (Error):**
```json
{
  "error": "Missing feature: Pregnancies"
}
```

### GET /

Serves the web interface (HTML form)

## Model Information

### Algorithm
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

### Dataset
- **Total Samples**: 768
- **Training Samples**: 614 (80%)
- **Validation Samples**: 154 (20%)
- **Features**: 8 medical parameters
- **Target**: Binary classification (0 = Not Diabetic, 1 = Diabetic)

### Performance Metrics
The model validation metrics typically show:
- **Accuracy**: ~83%
- **Precision**: ~78%
- **Recall**: ~73%
- **F1-Score**: ~75%

## Troubleshooting

### Error: "Model or feature names not found"

**Solution**: Run `python train.py` first to generate the model files.

### Error: Cannot connect to localhost:5000

**Solution**: 
- Make sure the Flask server is running
- Try accessing `http://localhost:5000/health` to verify the server is up
- Check if port 5000 is already in use

### Port 5000 Already in Use

**Solution**: 
- Kill the process using port 5000, or
- Modify the port in `app.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
  ```

### Requirements Installation Fails

**Solution**:
- Upgrade pip: `pip install --upgrade pip`
- Try installing packages individually:
  ```bash
  pip install Flask==3.0.0
  pip install pandas==2.1.3
  pip install numpy==1.26.2
  pip install scikit-learn==1.3.2
  pip install joblib==1.3.2
  ```

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Features Used in Predictions

The model uses these 8 features to make predictions:

1. **Pregnancies** (0-17): Number of times pregnant
2. **PlasmaGlucose** (0-200): Plasma glucose concentration in mg/dL
3. **DiastolicBloodPressure** (0-122): Diastolic blood pressure in mmHg
4. **TricepsThickness** (0-99): Triceps skin fold thickness in mm
5. **SerumInsulin** (0-846): 2-Hour serum insulin in µU/mL
6. **BMI** (0-67.1): Body mass index in kg/m²
7. **DiabetesPedigree** (0.078-2.42): Diabetes pedigree function
8. **Age** (21-81): Age in years

## Files Generated After Training

After running `train.py`, two additional files are created:

1. **diabetes_model.joblib** (5-10 MB): The trained Random Forest model
2. **feature_names.joblib** (< 1 KB): List of feature names in correct order

These files are required by the Flask app to make predictions.

## Development Notes

- The application uses Flask's built-in development server. For production, use a production WSGI server like Gunicorn or uWSGI.
- The model operates with `debug=True`, which enables hot reloading. For production, set `debug=False`.
- All predictions are made in real-time without caching. For high-traffic systems, consider implementing caching.

## Performance Optimization

To improve model prediction time in production:
- Use a WSGI server (Gunicorn, uWSGI)
- Implement prediction caching for identical inputs
- Consider using a lighter model (e.g., Logistic Regression or Decision Tree)
- Deploy with multiple worker processes

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed: `pip list`
3. Ensure `train.py` has been run to generate model files
4. Check that you're using Python 3.8 or higher: `python --version`
