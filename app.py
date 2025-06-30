from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ✅ Load trained model and expected feature names
model = joblib.load('model.pkl')
feature_names = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Required numeric inputs
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        ca = int(request.form['ca'])

        # Optional categorical inputs (default to 0)
        cp = request.form.get("cp", "0")
        restecg = request.form.get("restecg", "0")
        slope = request.form.get("slope", "0")
        thal = request.form.get("thal", "0")

        # One-hot encode manually for dropdowns
        cp_encoded = [0, 0, 0, 0]
        if cp.isdigit() and int(cp) in range(4):
            cp_encoded[int(cp)] = 1

        restecg_encoded = [0, 0, 0]
        if restecg.isdigit() and int(restecg) in range(3):
            restecg_encoded[int(restecg)] = 1

        slope_encoded = [0, 0, 0]
        if slope.isdigit() and int(slope) in range(3):
            slope_encoded[int(slope)] = 1

        thal_encoded = [0, 0, 0, 0]
        if thal.isdigit() and int(thal) in range(4):
            thal_encoded[int(thal)] = 1

        # Combine all features in order
        raw_input = [age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca] + \
                    cp_encoded + restecg_encoded + slope_encoded + thal_encoded

        # Create aligned DataFrame using feature_names
        input_df = pd.DataFrame([0]*len(feature_names), index=feature_names).T
        input_df.iloc[0, :len(raw_input)] = raw_input  # fill in known values

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "has Heart Disease" if prediction == 1 else "does NOT have Heart Disease"

        return render_template('result.html', prediction_text=f"The patient likely {result}.")

    except Exception as e:
        return render_template('result.html', prediction_text=f"❌ Error: {str(e)}")

# ✅ Allow deployment on Render
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
