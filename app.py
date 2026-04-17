import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Obesity Level Prediction App", layout="wide")

@st.cache_resource
def load_files():
    lr_model = joblib.load("lr_model.pkl")
    knn_model = joblib.load("knn_model.pkl")
    dt_model = joblib.load("dt_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    columns = joblib.load("columns.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    return lr_model, knn_model, dt_model, rf_model, scaler, label_encoder, columns, X_test, y_test

lr_model, knn_model, dt_model, rf_model, scaler, label_encoder, columns, X_test, y_test = load_files()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Comparison"])

if page == "Home":
    st.title("Obesity Level Prediction System")
    st.write("This application predicts obesity level using four machine learning models.")
    st.subheader("Models Used")
    st.write("- Logistic Regression")
    st.write("- K-Nearest Neighbours (KNN)")
    st.write("- Decision Tree")
    st.write("- Random Forest")

elif page == "Prediction":
    st.title("Obesity Level Prediction")

    model_name = st.selectbox(
        "Select Model",
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    )

    age = st.number_input("Age", min_value=1, max_value=100, value=21)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
    fcvc = st.number_input("FCVC (Vegetable Consumption)", min_value=1.0, max_value=3.0, value=2.0)
    ncp = st.number_input("NCP (Main Meals)", min_value=1.0, max_value=5.0, value=3.0)
    ch2o = st.number_input("CH2O (Water Intake)", min_value=0.0, max_value=5.0, value=2.0)
    faf = st.number_input("FAF (Physical Activity)", min_value=0.0, max_value=5.0, value=1.0)
    tue = st.number_input("TUE (Technology Usage)", min_value=0.0, max_value=5.0, value=1.0)

    gender = st.selectbox("Gender", ["Female", "Male"])
    family_history = st.selectbox("Family History with Overweight", ["no", "yes"])
    favc = st.selectbox("Frequent High Calorie Food Consumption (FAVC)", ["no", "yes"])
    caec = st.selectbox("CAEC", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("SMOKE", ["no", "yes"])
    scc = st.selectbox("SCC", ["no", "yes"])
    calc = st.selectbox("CALC", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "MTRANS",
        ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
    )

    if st.button("Predict"):
        input_dict = {
            "Age": age,
            "Height": height,
            "Weight": weight,
            "FCVC": fcvc,
            "NCP": ncp,
            "CH2O": ch2o,
            "FAF": faf,
            "TUE": tue,
            "Gender": gender,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "CAEC": caec,
            "SMOKE": smoke,
            "SCC": scc,
            "CALC": calc,
            "MTRANS": mtrans
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        if model_name == "Logistic Regression":
            input_scaled = scaler.transform(input_encoded)
            pred = lr_model.predict(input_scaled)
        elif model_name == "KNN":
            input_scaled = scaler.transform(input_encoded)
            pred = knn_model.predict(input_scaled)
        elif model_name == "Decision Tree":
            pred = dt_model.predict(input_encoded)
        else:
            pred = rf_model.predict(input_encoded)

        result = label_encoder.inverse_transform(pred)
        st.success(f"Predicted Obesity Level: {result[0]}")

elif page == "Model Comparison":
    st.title("Model Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [0.9641, 0.8756, 0.9737, 0.9809],
        "Precision": [0.9636, 0.8686, 0.9745, 0.9812],
        "Recall": [0.9626, 0.8709, 0.9724, 0.9796],
        "F1-Score": [0.9629, 0.8689, 0.9731, 0.9800]
    })

    st.subheader("Evaluation Metrics")
    st.dataframe(results, use_container_width=True)

    best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
    st.success(f"Best Model Based on Accuracy: {best_model}")

    st.subheader("Performance Chart")
    st.bar_chart(results.set_index("Model"))

    st.subheader("Confusion Matrix")
    model_choice = st.selectbox(
        "Select model for Confusion Matrix",
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    )

    if model_choice == "Logistic Regression":
        y_pred = lr_model.predict(scaler.transform(X_test))
    elif model_choice == "KNN":
        y_pred = knn_model.predict(scaler.transform(X_test))
    elif model_choice == "Decision Tree":
        y_pred = dt_model.predict(X_test)
    else:
        y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_choice}")

    st.pyplot(fig)