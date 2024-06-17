import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Preprocess function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Load the model, vectorizer, and label encoder
model = joblib.load('emotion_detection_model.pkl')
vectorizer = joblib.load('emotion_detection_vectorizer.pkl')
label_encoder = joblib.load('emotion_detection_label_encoder.pkl')

# Emotion mapping
emotion_mapping = {
    0: 'neutral',
    1: 'anger',
    2: 'joy',
    3: 'disgust',
    4: 'fear',
    5: 'sadness',
    6: 'surprise'
}

# Streamlit UI
st.title('Emotion Detection')
st.write('Enter a text message to classify its emotion.')

st.write("""
This app predicts the emotion of a given text message using Naive Bayes Classifier Method.
""")

# User input text
user_input = st.text_area('Enter text message here:')

# Button to trigger classification
if st.button('Classify'):
    if user_input is None or user_input.strip() == "":
        st.write('Please enter a valid text message.')
    else:
        try:
            # Preprocess and transform input text
            processed_input = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_input])

            # Make prediction
            prediction = model.predict(vectorized_input)
            predicted_emotion = emotion_mapping[prediction[0]]

            st.write('Predicted emotion:', predicted_emotion)
        except Exception as e:
            st.error(f"An error occurred: {e}")
