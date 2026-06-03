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
bp_model       = load_model('best_model.pkl')
cardio_model   = load_model('cardio_model.pkl')
calcium_model  = load_model('model_calcium.pkl')
b12_model      = load_model('model_b12.pkl')
vdd_model      = load_model('model_vdd.pkl')
anemia_model   = load_model('model_anemia.pkl')
iron_model     = load_model('model_iron.pkl')

# ==========================================
# ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "models_loaded": {
            "diabetes":       diabetes_model is not None,
            "blood_pressure": bp_model is not None,
            "cardio":         cardio_model is not None,
            "calcium":        calcium_model is not None,
            "b12":            b12_model is not None,
            "vitamin_d":      vdd_model is not None,
            "anemia":         anemia_model is not None,
            "iron":           iron_model is not None,
        }
    })

@app.route('/predict/diabetes', methods=['POST'])
@app.route('/predict/diabetes.pkl', methods=['POST'])
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
@app.route('/predict/blood_pressure.pkl', methods=['POST'])
def predict_blood_pressure():
    if not bp_model:
        return jsonify({"error": "Blood Pressure model not loaded on server."}), 500

    try:
        data = request.json
        patient_info = data.get('patient', {})
        bp_data = data.get('answers', {})

        def parse_binary(val):
            if val is None: return 0
            if isinstance(val, (int, float)): return int(val)
            return 1 if str(val).strip().lower() in ('yes', '1', 'true') else 0

        age = float(patient_info.get('age') or 45)
        sys_bp   = float(bp_data.get('sysBP', 120))
        dia_bp   = float(bp_data.get('diaBP', 80))
        bmi      = float(bp_data.get('BMI', 25))
        hr       = float(bp_data.get('heartRate', 75))
        glucose  = float(bp_data.get('glucose', 90))
        chol     = float(bp_data.get('totChol', 200))
        cigs     = float(bp_data.get('cigsPerDay', 0))
        smoker   = parse_binary(bp_data.get('currentSmoker', 0))
        diabetes = parse_binary(bp_data.get('diabetes', 0))
        bp_meds  = parse_binary(bp_data.get('BPMeds', 0))

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
            diagnosis = "Low Risk"
            advice = "• Maintain a balanced, low-sodium diet.\n• Exercise regularly (at least 30 min/day).\n• Monitor blood pressure at home periodically."
            is_high_risk = False
        elif risk_pct < 50:
            diagnosis = "Moderate Risk"
            advice = "• Reduce salt and processed food intake.\n• Check blood pressure at least weekly.\n• Discuss results with your healthcare provider."
            is_high_risk = False
        else:
            diagnosis = "High Risk"
            advice = "• Consult a cardiologist as soon as possible.\n• Strictly follow any prescribed BP medications.\n• Adopt a DASH diet and avoid smoking immediately."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud Heart Model"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


@app.route('/predict/cardio', methods=['POST'])
@app.route('/predict/cardio.pkl', methods=['POST'])
def predict_cardio():
    if not cardio_model:
        return jsonify({"error": "Cardio model not loaded on server."}), 500

    try:
        data = request.json
        patient_info = data.get('patient', {})
        cardio_data = data.get('answers', {})

        def parse_binary(val):
            if val is None: return 0
            if isinstance(val, (int, float)): return int(val)
            return 1 if str(val).strip().lower() in ('yes', '1', 'true') else 0

        def parse_level(val, default=1):
            if val is None: return default
            if isinstance(val, (int, float)): return int(val)
            v = str(val).strip().lower()
            if 'well' in v: return 3
            if 'above' in v: return 2
            return 1

        age_years  = float(patient_info.get('age') or 45)
        height     = float(patient_info.get('height') or 170)
        weight     = float(patient_info.get('weight') or 70)
        gender_str = str(patient_info.get('gender') or 'Male')
        
        # 1. تصحيح ترميز الجنس (الذكر = 2، الأنثى = 1)
        gender = 2 if gender_str.lower() == 'male' else 1

        ap_hi       = float(cardio_data.get('ap_hi', 120))
        ap_lo       = float(cardio_data.get('ap_lo', 80))
        cholesterol = parse_level(cardio_data.get('cholesterol', 1))
        gluc        = parse_level(cardio_data.get('gluc', 1))
        smoke       = parse_binary(cardio_data.get('smoke', 0))
        alco        = parse_binary(cardio_data.get('alco', 0))
        active      = parse_binary(cardio_data.get('active', 1))

        bmi = weight / ((height / 100) ** 2) if height > 0 else 25.0
        pulse_pressure = ap_hi - ap_lo
        map_val = (ap_hi + 2 * ap_lo) / 3

        # 2. تصحيح فئات العمر لتطابق التدريب تماماً
        if age_years <= 40:   age_group = 1
        elif age_years <= 50: age_group = 2
        elif age_years <= 55: age_group = 3
        elif age_years <= 60: age_group = 4
        else:                 age_group = 5

        # 3. تصحيح فئات كتلة الجسم لتطابق التدريب تماماً
        if bmi <= 18.5:    bmi_category = 1
        elif bmi <= 25.0:  bmi_category = 2
        elif bmi <= 30.0:  bmi_category = 3
        else:              bmi_category = 4

        hypertension = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0
        age_chol = age_years * cholesterol
        bp_age   = ap_hi * age_years

        # 4. الترتيب الصحيح والمعتمد للأعمدة كما تدرب عليها الموديل في الـ Notebook
        feature_order = [
            'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
            'smoke', 'alco', 'active', 'age_years', 'bmi', 'pulse_pressure', 'map', 
            'age_group', 'bmi_category', 'hypertension', 'age_chol', 'bp_age'
        ]

        input_df = pd.DataFrame([{
            'age_years': age_years, 'gender': gender, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol, 'gluc': gluc,
            'smoke': smoke, 'alco': alco, 'active': active, 'bmi': round(bmi, 2),
            'pulse_pressure': round(pulse_pressure, 2), 'map': round(map_val, 2),
            'age_group': age_group, 'bmi_category': bmi_category, 'hypertension': hypertension,
            'age_chol': round(age_chol, 2), 'bp_age': round(bp_age, 2),
        }])

        # إعادة ترتيب الأعمدة إجبارياً ليتطابق مع مصفوفة التدريب
        input_df = input_df[feature_order]

        if hasattr(cardio_model, 'predict_proba'):
            probas = cardio_model.predict_proba(input_df)
            prob = float(probas[0][1])
        else:
            pred = cardio_model.predict(input_df)
            prob = float(pred[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 20:
            diagnosis = "Low Risk"
            advice = "• Keep up your healthy lifestyle.\n• Maintain regular light exercise (30+ min/day)."
            is_high_risk = False
        elif risk_pct < 50:
            diagnosis = "Moderate Risk"
            advice = "• Reduce saturated fats and processed foods.\n• Consult your doctor about your cardiovascular health."
            is_high_risk = False
        else:
            diagnosis = "High Risk"
            advice = "• Consult a cardiologist as soon as possible.\n• Strictly manage blood pressure, cholesterol & weight."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud Cardio Model"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ==========================================
# SHARED HELPER — Nutrition Models
# ==========================================
def build_nutrition_input(data, overrides={}):
    patient = data.get('patient', {})
    answers = data.get('answers', {})

    weight = float(patient.get('weight') or 70)
    height = float(patient.get('height') or 170)
    bmi    = weight / ((height / 100) ** 2) if height > 0 else 25.0

    base = {
        'fatigue':                      int(answers.get('fatigue',     0)),
        'hair_loss':                    int(answers.get('hair_loss',   0)),
        'dizziness':                    int(answers.get('dizziness',   0)),
        'muscle_pain':                  int(answers.get('muscle_pain', 0)),
        'numbness':                     int(answers.get('numbness',    0)),
        'Hemoglobin':                   float(answers.get('Hemoglobin',   13.5)),
        'MCH':                          float(answers.get('MCH',          27.0)),
        'MCHC':                         float(answers.get('MCHC',         32.0)),
        'MCV':                          float(answers.get('MCV',          85.0)),
        'ferritin':                     float(answers.get('ferritin',     50.0)),
        'vitamin_b12':                  float(answers.get('vitamin_b12', 300.0)),
        'calcium':                      float(answers.get('calcium',       9.5)),
        'Vitamin D Level (ng/mL)':      float(answers.get('vitamin_d_level',    25.0)),
        'Dietary Vitamin D (IU)':       float(answers.get('dietary_vitamin_d', 400.0)),
        'Dietary Calcium (mg)':         float(answers.get('dietary_calcium',   800.0)),
        'Age':                          float(patient.get('age') or 30),
        'BMI':                          round(bmi, 2),
        'Sun Exposure (hours/week)':    float(answers.get('sun_exposure', 5.0)),
    }

    base.update(overrides)
    return pd.DataFrame([base])


# ==========================================
# NEW ROUTE — Calcium
# ==========================================
@app.route('/predict/calcium', methods=['POST'])
@app.route('/predict/calcium.pkl', methods=['POST'])
def predict_calcium():
    if not calcium_model:
        return jsonify({"error": "Calcium model not loaded on server."}), 500
    try:
        input_df = build_nutrition_input(request.json)

        if hasattr(calcium_model, 'predict_proba'):
            prob = float(calcium_model.predict_proba(input_df)[0][1])
        else:
            prob = float(calcium_model.predict(input_df)[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 30:
            diagnosis    = "Normal - Calcium levels likely adequate"
            advice       = "• Maintain a calcium-rich diet (dairy, leafy greens).\n• Ensure adequate Vitamin D intake to aid absorption.\n• Stay physically active to support bone health."
            is_high_risk = False
        elif risk_pct < 60:
            diagnosis    = "Moderate Risk - Possible Calcium Deficiency"
            advice       = "• Increase dietary calcium (milk, cheese, broccoli, almonds).\n• Check Vitamin D levels as deficiency reduces calcium absorption.\n• Consult your doctor for a blood calcium test."
            is_high_risk = False
        else:
            diagnosis    = "High Risk - Calcium Deficiency Likely"
            advice       = "• Consult a doctor immediately for a full calcium panel.\n• Consider calcium supplements under medical supervision.\n• Avoid excessive caffeine and sodium which deplete calcium."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud Calcium Model"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ==========================================
# NEW ROUTE — Vitamin B12
# ==========================================
@app.route('/predict/b12', methods=['POST'])
@app.route('/predict/b12.pkl', methods=['POST'])
def predict_b12():
    if not b12_model:
        return jsonify({"error": "B12 model not loaded on server."}), 500
    try:
        input_df = build_nutrition_input(request.json)

        if hasattr(b12_model, 'predict_proba'):
            prob = float(b12_model.predict_proba(input_df)[0][1])
        else:
            prob = float(b12_model.predict(input_df)[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 30:
            diagnosis    = "Normal - Vitamin B12 levels likely adequate"
            advice       = "• Continue eating B12-rich foods (meat, eggs, dairy).\n• If vegetarian/vegan, consider a B12 supplement.\n• Recheck annually if you follow a plant-based diet."
            is_high_risk = False
        elif risk_pct < 60:
            diagnosis    = "Moderate Risk - Possible B12 Deficiency"
            advice       = "• Increase intake of B12 sources: fish, poultry, eggs, fortified cereals.\n• Consider a B-complex supplement.\n• Get a serum B12 blood test to confirm levels."
            is_high_risk = False
        else:
            diagnosis    = "High Risk - Vitamin B12 Deficiency Likely"
            advice       = "• Consult a doctor for a serum B12 test immediately.\n• B12 injections or high-dose supplements may be required.\n• Untreated B12 deficiency can cause nerve damage — act promptly."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud B12 Model"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ==========================================
# NEW ROUTE — Vitamin D
# ==========================================
@app.route('/predict/vdd', methods=['POST'])
@app.route('/predict/vdd.pkl', methods=['POST'])
def predict_vdd():
    if not vdd_model:
        return jsonify({"error": "Vitamin D model not loaded on server."}), 500
    try:
        input_df = build_nutrition_input(request.json)

        if hasattr(vdd_model, 'predict_proba'):
            probas     = vdd_model.predict_proba(input_df)[0]
            predicted  = int(np.argmax(probas))
            confidence = round(float(probas[predicted]) * 100, 1)
        else:
            predicted  = int(vdd_model.predict(input_df)[0])
            confidence = 100.0

        level_map = {
            0: {"diagnosis": "Sufficient - Vitamin D levels are adequate",
                "advice": "• Great! Keep getting regular sun exposure.\n• Maintain a diet with fatty fish, eggs, and fortified foods.\n• Recheck levels annually.",
                "is_high_risk": False},
            1: {"diagnosis": "Insufficient - Vitamin D slightly low",
                "advice": "• Increase sun exposure to 15-30 min/day.\n• Eat more Vitamin D-rich foods (salmon, tuna, fortified milk).\n• Consider a low-dose supplement (400-800 IU/day).",
                "is_high_risk": False},
            2: {"diagnosis": "Deficient - Vitamin D deficiency detected",
                "advice": "• Consult your doctor for a 25-OH Vitamin D blood test.\n• A supplement of 1000-2000 IU/day is commonly recommended.\n• Maximize safe sun exposure and dietary sources.",
                "is_high_risk": True},
            3: {"diagnosis": "Severely Deficient - Critical Vitamin D levels",
                "advice": "• See a doctor immediately — severe deficiency affects bones, immunity and muscles.\n• High-dose supplementation (up to 50,000 IU/week) may be prescribed.\n• Regular monitoring of blood levels is essential.",
                "is_high_risk": True},
        }
        result = level_map.get(predicted, level_map[0])

        return jsonify({
            "predicted_class": predicted,
            "confidence":      confidence,
            "diagnosis":       result["diagnosis"],
            "advice":          result["advice"],
            "is_high_risk":    result["is_high_risk"],
            "model_info":      "Cloud Vitamin D Model"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ==========================================
# NEW ROUTE — Anemia
# ==========================================
@app.route('/predict/anemia', methods=['POST'])
@app.route('/predict/anemia.pkl', methods=['POST'])
def predict_anemia():
    if not anemia_model:
        return jsonify({"error": "Anemia model not loaded on server."}), 500
    try:
        input_df = build_nutrition_input(request.json)

        if hasattr(anemia_model, 'predict_proba'):
            prob = float(anemia_model.predict_proba(input_df)[0][1])
        else:
            prob = float(anemia_model.predict(input_df)[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 30:
            diagnosis    = "Normal - No signs of Anemia"
            advice       = "• Maintain a balanced diet rich in iron (red meat, spinach, legumes).\n• Ensure adequate Vitamin B12 and folate intake.\n• Stay hydrated and exercise regularly."
            is_high_risk = False
        elif risk_pct < 60:
            diagnosis    = "Moderate Risk - Possible Anemia"
            advice       = "• Increase iron-rich foods: red meat, lentils, spinach, fortified cereals.\n• Pair iron foods with Vitamin C to boost absorption.\n• Get a CBC blood test to check hemoglobin levels."
            is_high_risk = False
        else:
            diagnosis    = "High Risk - Anemia Likely"
            advice       = "• Consult a doctor immediately for a complete blood count (CBC).\n• Iron supplements may be required under medical supervision.\n• Avoid tea/coffee with meals as they block iron absorption."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud Anemia Model"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ==========================================
# NEW ROUTE — Iron Deficiency (Ferritin)
# ==========================================
@app.route('/predict/iron', methods=['POST'])
@app.route('/predict/iron.pkl', methods=['POST'])
def predict_iron():
    if not iron_model:
        return jsonify({"error": "Iron model not loaded on server."}), 500
    try:
        input_df = build_nutrition_input(request.json)

        if hasattr(iron_model, 'predict_proba'):
            prob = float(iron_model.predict_proba(input_df)[0][1])
        else:
            prob = float(iron_model.predict(input_df)[0])

        risk_pct = round(prob * 100, 1)

        if risk_pct < 30:
            diagnosis    = "Normal - Iron levels likely adequate"
            advice       = "• Maintain a diet rich in iron (red meat, spinach, legumes).\n• Pair iron-rich foods with Vitamin C to boost absorption.\n• Stay hydrated and exercise regularly."
            is_high_risk = False
        elif risk_pct < 60:
            diagnosis    = "Moderate Risk - Possible Iron Deficiency"
            advice       = "• Increase iron-rich foods: red meat, lentils, spinach, fortified cereals.\n• Get a ferritin blood test to confirm iron levels.\n• Avoid tea/coffee with meals as they block iron absorption."
            is_high_risk = False
        else:
            diagnosis    = "High Risk - Iron Deficiency Likely"
            advice       = "• Consult a doctor immediately for a full iron panel.\n• Iron supplements may be required under medical supervision.\n• Avoid calcium-rich foods alongside iron supplements."
            is_high_risk = True

        return jsonify({
            "risk_percentage": risk_pct,
            "diagnosis":       diagnosis,
            "advice":          advice,
            "is_high_risk":    is_high_risk,
            "model_info":      "Cloud Iron Model"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
