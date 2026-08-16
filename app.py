import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Waking up the "Brain"
sleep_model = joblib.load('sleep_model.joblib')
sleep_scaler = joblib.load('sleep_scaler.joblib')

st.title("Sleep Efficiency Predictor")

# 1. The Polished UI Inputs (in exact matrix order)
user_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000, step=500)
user_alcohol = st.number_input("Alcohol Units", min_value=0, max_value=15, value=0)

# Human-readable 1-10 slider transformed into a 10-100 metric for the backend
display_stress = st.slider("Stress Level (1 = Completely Relaxed, 10 = Max Stress)", min_value=1, max_value=10, value=5)
user_stress = display_stress * 10

user_caffeine = st.number_input("Caffeine (mg)", min_value=0, max_value=1000, value=100)
user_workout_none = int(st.selectbox("Did you skip working out today?", ["No", "Yes"]) == "Yes")
user_screen_time = st.number_input("Screen Time (minutes)", min_value=0, max_value=1440, value=120)
user_workout_walk = int(st.selectbox("Did you go for a walk today?", ["No", "Yes"]) == "Yes")

# 2. Packaging the Payload
# Packaging the payload as a DataFrame with explicit feature names to eliminate warnings
user_data = pd.DataFrame([[
    user_steps, 
    user_alcohol, 
    user_stress, 
    user_caffeine, 
    user_workout_none, 
    user_screen_time, 
    user_workout_walk
]], columns=['steps', 'alcohol_units', 'stress_score', 'caffeine_mg', 'workout_type_none', 'screen_time_min', 'workout_type_walk'])

# 3. The Prediction Trigger
if st.button("Predict Sleep Efficiency"):
    
    # Scale and Predict
    scaled_data = sleep_scaler.transform(user_data)
    prediction = sleep_model.predict(scaled_data)
    
    # Convert decimal to clean percentage
    prediction_pct = prediction[0] * 100
    
    # Display the primary score
    st.success(f"Your predicted sleep efficiency is: {prediction_pct:.0f}%")
    
    # 4. Human Interpretation Logic
    if prediction_pct >= 90:
        st.info("🟢 **Excellent:** You are getting highly restorative, deep sleep. Your habits are perfectly optimized.")
    elif prediction_pct >= 80:
        st.info("🟡 **Good:** You are getting standard, healthy rest, but there is room for minor habit optimizations.")
    elif prediction_pct >= 70:
        st.warning("🟠 **Fair:** Your sleep is somewhat compromised. Consider reducing stress, cutting late-night caffeine, or adding light activity.")
    else:
        st.error("🔴 **Poor:** Your deep sleep cycles are heavily disrupted. This usually correlates with high stress, alcohol, or extreme sedentariness.")