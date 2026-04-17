import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Obesity App", layout="wide")

# =========================
# Load files
# =========================
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

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Comparison"])

# =========================
# Prediction Page
# =========================
if page == "Prediction":
    st.title("Obesity Level Prediction")

    model_name = st.selectbox(
        "Select Model",
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    )

    # Inputs
    age = st.number_input("Age", 1, 100, 21)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
    weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
    fcvc = st.number_input("FCVC", 1.0, 3.0, 2.0)
    ncp = st.number_input("NCP", 1.0, 5.0, 3.0)
    ch2o = st.number_input("CH2O", 0.0, 5.0, 2.0)
    faf = st.number_input("FAF", 0.0, 5.0, 1.0)
    tue = st.number_input("TUE", 0.0, 5.0, 1.0)

    gender = st.selectbox("Gender", ["Female", "Male"])
    family_history = st.selectbox("Family History", ["no", "yes"])
    favc = st.selectbox("FAVC", ["no", "yes"])
    caec = st.selectbox("CAEC", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("SMOKE", ["no", "yes"])
    scc = st.selectbox("SCC", ["no", "yes"])
    calc = st.selectbox("CALC", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("MTRANS", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    if st.button("Predict"):
        # =========================
        # BMI Calculation
        # =========================
        bmi = weight / (height ** 2)

        if bmi < 18.5:
            bmi_cat = "Insufficient Weight"
        elif bmi < 25:
            bmi_cat = "Normal Weight"
        elif bmi < 27:
            bmi_cat = "Overweight Level I"
        elif bmi < 30:
            bmi_cat = "Overweight Level II"
        elif bmi < 35:
            bmi_cat = "Obesity Type I"
        elif bmi < 40:
            bmi_cat = "Obesity Type II"
        else:
            bmi_cat = "Obesity Type III"

        # =========================
        # Prepare input
        # =========================
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

        # =========================
        # Prediction
        # =========================
        if model_name == "Logistic Regression":
            pred = lr_model.predict(scaler.transform(input_encoded))
        elif model_name == "KNN":
            pred = knn_model.predict(scaler.transform(input_encoded))
        elif model_name == "Decision Tree":
            pred = dt_model.predict(input_encoded)
        else:
            pred = rf_model.predict(input_encoded)

        result = label_encoder.inverse_transform(pred)

        # =========================
        # Display
        # =========================
        st.success(f"Predicted Obesity Level: {result[0]}")
        st.info(f"BMI: {bmi:.2f}")
        st.info(f"BMI Category: {bmi_cat}")

# =========================
# Model Comparison Page
# =========================
elif page == "Model Comparison":
    st.title("Model Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [0.9641, 0.8756, 0.9737, 0.9809],
        "Precision": [0.9636, 0.8686, 0.9745, 0.9812],
        "Recall": [0.9626, 0.8709, 0.9724, 0.9796],
        "F1-Score": [0.9629, 0.8689, 0.9731, 0.9800]
    })

    st.dataframe(results)

    best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
    st.success(f"Best Model: {best_model}")

    st.bar_chart(results.set_index("Model"))

    # =========================
    # Confusion Matrix
    # =========================
    st.subheader("Confusion Matrix")

    model_choice = st.selectbox(
        "Select Model",
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
    ax.matshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)
