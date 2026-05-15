import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

# ==========================================
# REQUIRED FUNCTIONS FOR MODEL UNPICKLING
# ==========================================
def create_pattern_features(df_input):
    df_out = df_input.copy()
    num_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0.0)
    
    df_out['HbA1c_Diabetic'] = np.where(df_out['HbA1c_level'] >= 6.5, 1.0, 0.0)
    df_out['Age_Over_50'] = np.where(df_out['age'] > 50, 1.0, 0.0)
    df_out['BMI_Obese'] = np.where(df_out['bmi'] >= 30, 1.0, 0.0)
    df_out['Glucose_High'] = np.where(df_out['blood_glucose_level'] >= 140, 1.0, 0.0)
    
    df_out['Pattern_1'] = np.where(df_out['HbA1c_Diabetic'] == 1.0, 1.0, 0.0)
    df_out['Pattern_2'] = np.where((df_out['BMI_Obese'] == 1.0) & (df_out['Age_Over_50'] == 1.0), 1.0, 0.0)
    df_out['Pattern_3'] = np.where((df_out['HbA1c_Diabetic'] == 1.0) & (df_out['Age_Over_50'] == 1.0), 1.0, 0.0)
    
    return df_out

import sys
import __main__
__main__.create_pattern_features = create_pattern_features

# ==========================================
# MODEL LOADING
# ==========================================
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appai")

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    print(f"Model not found: {path}")
    return None

diabetes_model = load_model('best_diabetes_model.pkl')
bp_model = load_model('best_model.pkl')
cardio_model = load_model('cardio_model.pkl')

# ==========================================
# ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "models_loaded": {
            "diabetes": diabetes_model is not None,
            "blood_pressure": bp_model is not None,
            "cardio": cardio_model is not None
        }
    })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({"error": "Diabetes model not loaded on server."}), 500

    try:
        data = request.json
        p_data = data.get('patient', {})
        d_data = data.get('answers', {})
        
        gender_input = d_data.get('gender') or p_data.get('gender') or 'Female'
        age_val = float(d_data.get('age_override') or p_data.get('age') or 30)
        
        def parse_binary(val):
            return 1 if val and str(val).lower() == 'yes' else 0
            
        hypertension = parse_binary(d_data.get('hypertension', 'No'))
        heart_disease = parse_binary(d_data.get('heart_disease', 'No'))
        
        smoking_history_txt = d_data.get('smoking_history', 'never')
        
        hyp_val = float(hypertension)
        hd_val = float(heart_disease)
        smoke_val = str(smoking_history_txt) 
        
        weight_val = float(p_data.get('weight') or 70)
        height_val = float(p_data.get('height') or 170)
        bmi_val = float(d_data.get('bmi_override') or (weight_val / ((height_val/100)**2) if height_val > 0 else 25.0))
        hba1c_val = float(d_data.get('HbA1c_level') or 5.5)
        glucose_val = float(d_data.get('blood_glucose_level') or 100)
        
        if hba1c_val >= 6.5 and glucose_val >= 200:
            prob = 0.99
            model_info = "Direct Medical Diagnosis (Clinical Threshold)"
        elif hba1c_val >= 6.5 and glucose_val >= 126:
            prob = 0.95
            model_info = "Direct Medical Diagnosis (High Fasting Glucose)"
        else:
            pattern_1 = 1 if hba1c_val >= 6.5 else 0
            pattern_2 = 1 if (bmi_val >= 30 and age_val > 50) else 0
            pattern_3 = 1 if (hba1c_val >= 6.5 and age_val > 50) else 0

            input_df = pd.DataFrame([{
                'gender': gender_input,
                'age': age_val,
                'hypertension': hyp_val,
                'heart_disease': hd_val,
                'smoking_history': smoke_val,
                'bmi': bmi_val,
                'HbA1c_level': hba1c_val,
                'blood_glucose_level': glucose_val,
                'Pattern_1': pattern_1,
                'Pattern_2': pattern_2,
                'Pattern_3': pattern_3
            }])
            
            if hasattr(diabetes_model, 'predict_proba'):
                probas = diabetes_model.predict_proba(input_df)
                prob = float(probas[0][1])
                model_info = f"AI Cloud Model (Confidence: {max(prob, 1-prob):.1%})"
            else:
                prediction = diabetes_model.predict(input_df)
                prob = 1.0 if prediction[0] == 1 else 0.0
                model_info = "AI Cloud Model (Binary Prediction)"

        risk = round(float(prob) * 100, 1)
        
        if prob < 0.5:
            diagnosis = "Negative - Not Diabetic"
            advice = "• Your metrics are within the normal range.\n• Maintain current healthy habits.\n• Ensure you exercise at least 30 minutes daily."
            is_high_risk = False
        else:
            diagnosis = "Positive - Diabetic"
            advice = "• Consult an endocrinologist as soon as possible.\n• Closely follow a structured low-glycemic dietary plan.\n• Monitor blood sugar daily."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk,
            "diagnosis": diagnosis,
            "advice": advice,
            "is_high_risk": is_high_risk,
            "model_info": model_info
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


@app.route('/predict/blood_pressure', methods=['POST'])
def predict_blood_pressure():
    if not bp_model:
        return jsonify({"error": "Blood Pressure model not loaded on server."}), 500

    try:
        data = request.json
        patient_info = data.get('patient', {})
        bp_data = data.get('answers', {})

        age = float(patient_info.get('age') or 45)
        sys_bp   = float(bp_data.get('sysBP', 120))
        dia_bp   = float(bp_data.get('diaBP', 80))
        bmi      = float(bp_data.get('BMI', 25))
        hr       = float(bp_data.get('heartRate', 75))
        glucose  = float(bp_data.get('glucose', 90))
        chol     = float(bp_data.get('totChol', 200))
        cigs     = float(bp_data.get('cigsPerDay', 0))
        smoker   = int(bp_data.get('currentSmoker', 0))
        diabetes = int(bp_data.get('diabetes', 0))
        bp_meds  = int(bp_data.get('BPMeds', 0))

        input_df = pd.DataFrame([{
            'age': age, 'sysBP': sys_bp, 'diaBP': dia_bp, 'BMI': bmi,
            'heartRate': hr, 'glucose': glucose, 'totChol': chol,
            'cigsPerDay': cigs, 'currentSmoker': smoker, 'diabetes': diabetes,
            'BPMeds': bp_meds,
        }])

        if hasattr(bp_model, 'feature_names_in_'):
            expected = list(bp_model.feature_names_in_)
            available = [c for c in expected if c in input_df.columns]
            input_df = input_df[available]

        if hasattr(bp_model, 'predict_proba'):
            probas = bp_model.predict_proba(input_df)
            prob = float(probas[0][1])
        else:
            pred = bp_model.predict(input_df)
            prob = float(pred[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 20:
            diagnosis = f"Low Risk"
            advice = "• Maintain a balanced, low-sodium diet.\n• Exercise regularly (at least 30 min/day).\n• Monitor blood pressure at home periodically."
            is_high_risk = False
        elif risk_pct < 50:
            diagnosis = f"Moderate Risk"
            advice = "• Reduce salt and processed food intake.\n• Check blood pressure at least weekly.\n• Discuss results with your healthcare provider."
            is_high_risk = False
        else:
            diagnosis = f"High Risk"
            advice = "• Consult a cardiologist as soon as possible.\n• Strictly follow any prescribed BP medications.\n• Adopt a DASH diet and avoid smoking immediately."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis": diagnosis,
            "advice": advice,
            "is_high_risk": is_high_risk,
            "model_info": "Cloud Heart Model"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


@app.route('/predict/cardio', methods=['POST'])
def predict_cardio():
    if not cardio_model:
        return jsonify({"error": "Cardio model not loaded on server."}), 500

    try:
        data = request.json
        patient_info = data.get('patient', {})
        cardio_data = data.get('answers', {})

        age_years  = float(patient_info.get('age') or 45)
        height     = float(patient_info.get('height') or 170)
        weight     = float(patient_info.get('weight') or 70)
        gender_str = str(patient_info.get('gender') or 'Male')
        gender     = 1 if gender_str.lower() == 'male' else 2

        ap_hi       = float(cardio_data.get('ap_hi', 120))
        ap_lo       = float(cardio_data.get('ap_lo', 80))
        cholesterol = int(cardio_data.get('cholesterol', 1))
        gluc        = int(cardio_data.get('gluc', 1))
        smoke       = int(cardio_data.get('smoke', 0))
        alco        = int(cardio_data.get('alco', 0))
        active      = int(cardio_data.get('active', 1))

        bmi = weight / ((height / 100) ** 2) if height > 0 else 25.0
        pulse_pressure = ap_hi - ap_lo
        map_val = (ap_hi + 2 * ap_lo) / 3

        if age_years < 35: age_group = 1
        elif age_years < 45: age_group = 2
        elif age_years < 55: age_group = 3
        elif age_years < 65: age_group = 4
        else: age_group = 5

        if bmi < 18.5: bmi_category = 1
        elif bmi < 25.0: bmi_category = 2
        elif bmi < 30.0: bmi_category = 3
        else: bmi_category = 4

        hypertension = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0
        age_chol = age_years * cholesterol
        bp_age = ap_hi * age_years

        input_df = pd.DataFrame([{
            'age_years': age_years, 'gender': gender, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol, 'gluc': gluc,
            'smoke': smoke, 'alco': alco, 'active': active, 'bmi': round(bmi, 2),
            'pulse_pressure': round(pulse_pressure, 2), 'map': round(map_val, 2),
            'age_group': age_group, 'bmi_category': bmi_category, 'hypertension': hypertension,
            'age_chol': round(age_chol, 2), 'bp_age': round(bp_age, 2),
        }])

        if hasattr(cardio_model, 'feature_names_in_'):
            expected = list(cardio_model.feature_names_in_)
            input_df = input_df[expected]
        elif hasattr(cardio_model, 'get_booster'):
            feature_names = cardio_model.get_booster().feature_names
            if feature_names: input_df = input_df[feature_names]

        if hasattr(cardio_model, 'predict_proba'):
            probas = cardio_model.predict_proba(input_df)
            prob = float(probas[0][1])
        else:
            pred = cardio_model.predict(input_df)
            prob = float(pred[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 20:
            diagnosis = f"Low Risk"
            advice = "• Keep up your healthy lifestyle.\n• Maintain regular light exercise (30+ min/day)."
            is_high_risk = False
        elif risk_pct < 50:
            diagnosis = f"Moderate Risk"
            advice = "• Reduce saturated fats and processed foods.\n• Consult your doctor about your cardiovascular health."
            is_high_risk = False
        else:
            diagnosis = f"High Risk"
            advice = "• Consult a cardiologist as soon as possible.\n• Strictly manage blood pressure, cholesterol & weight."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis": diagnosis,
            "advice": advice,
            "is_high_risk": is_high_risk,
            "model_info": "Cloud Cardio Model"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Run the Flask app on all interfaces at port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
