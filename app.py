import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

st.title("🩺 Diabetes Prediction App")

# Load dataset (use relative path so it works on GitHub/Streamlit Cloud)
diabetes_dataset = pd.read_csv(r"C:\Users\GOVIND GUPTA\Downloads\archive (6)\diabetes.csv")

# Show dataset summary
st.subheader("🔹 Dataset Overview")
st.write(diabetes_dataset.head())
st.write("Shape:", diabetes_dataset.shape)
st.write("Columns:", diabetes_dataset.columns.tolist())

# Show basic statistics
st.subheader("📊 Data Description")
st.write(diabetes_dataset.describe())

# Visualization section
st.subheader("📈 Data Visualization")

# Countplot
st.write("Diabetes Outcome Count (0 = No, 1 = Yes)")
fig1, ax1 = plt.subplots()
sns.countplot(x='Outcome', data=diabetes_dataset, palette='coolwarm', ax=ax1)
st.pyplot(fig1)

# Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(diabetes_dataset.corr(), annot=True, cmap='YlGnBu', ax=ax2)
st.pyplot(fig2)

# Histogram of Glucose
fig3, ax3 = plt.subplots()
sns.histplot(diabetes_dataset['Glucose'], kde=True, color='skyblue', ax=ax3)
ax3.set_title("Distribution of Glucose Levels")
st.pyplot(fig3)

# Prepare data for model
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(classifier.predict(X_train), Y_train)
test_acc = accuracy_score(classifier.predict(X_test), Y_test)

st.subheader("✅ Model Accuracy")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Testing Accuracy: {test_acc:.2f}")

# User Input Section
st.subheader("🧍‍♂️ Enter Patient Details for Prediction")

preg = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0)
bp = st.number_input('Blood Pressure', min_value=0)
skin = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

if st.button('Predict'):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)

    if prediction[0] == 1:
        st.error("The person is likely to have Diabetes 😔")
    else:
        st.success("The person is not likely to have Diabetes 😃")
