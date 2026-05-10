import streamlit as st
import joblib
import pandas as pd
import speech_recognition as sr

# LOAD MODEL
model = joblib.load("language_model.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌍",
    layout="centered"
)

# TITLE
st.title("🌍 Language Detection App")

st.write("Detect language using text or voice input.")

# TEXT INPUT
user_text = st.text_area(
    "Enter Text",
    height=150
)

# -----------------------------
# VOICE INPUT
# -----------------------------
if st.button("🎤 Speak"):

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            # REDUCE BACKGROUND NOISE
            recognizer.adjust_for_ambient_noise(source)

            st.info("Listening... Speak now.")

            # LISTEN FOR 5 SECONDS
            audio = recognizer.listen(
                source,
                phrase_time_limit=5
            )

            # CONVERT SPEECH TO TEXT
            user_text = recognizer.recognize_google(audio)

            st.success(f"You said: {user_text}")

    except Exception as e:

        st.error(f"Voice Error: {e}")

# -----------------------------
# DETECT LANGUAGE
# -----------------------------
if st.button("Detect Language"):

    if user_text.strip() != "":

        # PREDICTION
        prediction = model.predict([user_text])[0]

        # PROBABILITIES
        probabilities = model.predict_proba([user_text])[0]

        # LABELS
        labels = model.classes_

        # CREATE DATAFRAME
        prob_df = pd.DataFrame({
            "Language": labels,
            "Confidence": probabilities
        })

        # SORT VALUES
        prob_df = prob_df.sort_values(
            by="Confidence",
            ascending=False
        )

        # TOP CONFIDENCE
        top_confidence = prob_df.iloc[0]["Confidence"] * 100

        # OUTPUT
        st.success(f"Detected Language: {prediction}")

        st.info(f"Confidence Score: {top_confidence:.2f}%")

        # TOP 3 PREDICTIONS
        st.subheader("Top Predictions")

        top3 = prob_df.head(3)

        for index, row in top3.iterrows():

            st.write(
                f"{row['Language']} → {row['Confidence'] * 100:.2f}%"
            )

            st.progress(float(row["Confidence"]))

    else:

        st.warning("Please enter or speak text.")