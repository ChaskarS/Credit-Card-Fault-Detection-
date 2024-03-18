import numpy as np
import pickle
import pandas as pd
import streamlit as st 



# pickle_in = open("FraudDetection.pkl","rb")
# FraudDetection=pickle.load(pickle_in)

def load_model():
  
    with open('FraudDetection.pkl', 'rb') as file:
        model = pickle.load(file)
    return model




def main():
    st.title("ML Model Deployment with Streamlit")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])

    if uploaded_file is not None:
        st.write("File details:")
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        # Load the pre-trained ML model
        model = load_model()

        # Perform inference using the model
        predictions = model.predict(df)

        # Display the predictions
        st.write("Predictions:")
        st.write(predictions)

if __name__ == "__main__":
    main()