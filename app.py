from flask import Flask, render_template, request
import numpy as np
import joblib  # ✅ use joblib, not pickle

app = Flask(__name__)

# ✅ Load model only once, with correct name and method
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Required numeric features
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        ca = int(request.form['ca'])

        numeric_input = [age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca]

        # Optional categorical features
        cp = request.form.get("cp")
        restecg = request.form.get("restecg")
        slope = request.form.get("slope")
        thal = request.form.get("thal")

        cp_encoded = [0, 0, 0]
        restecg_encoded = [0, 0]
        slope_encoded = [0, 0]
        thal_encoded = [0, 0, 0]

        if cp in ['1', '2', '3']:
            cp_encoded[int(cp)-1] = 1
        if restecg in ['1', '2']:
            restecg_encoded[int(restecg)-1] = 1
        if slope in ['1', '2']:
            slope_encoded[int(slope)-1] = 1
        if thal in ['1', '2', '3']:
            thal_encoded[int(thal)-1] = 1

        final_input = numeric_input + cp_encoded + restecg_encoded + slope_encoded + thal_encoded

        prediction = model.predict([final_input])[0]

        output = "has Heart Disease" if prediction == 1 else "does NOT have Heart Disease"

        return render_template('result.html', prediction_text=f'The patient likely {output}.')

    except Exception as e:
        return render_template('result.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
