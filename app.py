import streamlit as st
import numpy as np
import tensorflow as tf
from config import NOS_CODES, MAJOR_BANKS, CIRCUITS
from preprocessor import prepare_inference
from advisor import get_guidance

st.title("⚖️ AI7-LITIGATION-ADVISOR")

# 1. Sidebar Inputs
year = st.sidebar.number_input("Year", 2011, 2026, 2026)
nos = st.sidebar.selectbox("Nature of Suit (NOS)", options=list(NOS_CODES.keys()), format_func=lambda x: f"{x} - {NOS_CODES[x]}")
bank = st.sidebar.selectbox("Defendant Bank", options=MAJOR_BANKS)
circuit = st.sidebar.selectbox("Court Circuit", options=CIRCUITS)

# 2. Run Prediction
if st.sidebar.button("Run Prediction"):
    try:
        # Load your specific model
        model = tf.keras.models.load_model("litigation_model_v5.keras", compile=False)
        
        # Prepare data
        input_data = {'year': year} 
        features = prepare_inference(input_data)
        
        # Predict
        prediction = model.predict(features)
        p_win = float(prediction[0][0])
        s_press = 1.0 - p_win # Example logic
        
        # Display Results
        st.metric("Win Probability", f"{p_win*100:.1f}%")
        
        notes = get_guidance(p_win, s_press)
        for note in notes:
            st.write(note)
            
    except Exception as e:
        st.error(f"Error: {e}")
