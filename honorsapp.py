import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ----------------------------
# Load model + feature columns
# ----------------------------
model = pickle.load(open("honors_model.pkl", "rb"))
feature_columns = pickle.load(open("honors_feature_columns.pkl", "rb"))

# ----------------------------
# Load datasets
# ----------------------------
train = pd.read_csv("data/train.csv")
m1 = pd.read_csv("data/month1.csv")
m2 = pd.read_csv("data/month2.csv")
m3 = pd.read_csv("data/month3.csv")

datasets = {
    "Training": train,
    "Month 1": m1,
    "Month 2": m2,
    "Month 3": m3
}

# ----------------------------
# PSI Function
# ----------------------------
def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (expected_counts - actual_counts) *
        np.log((expected_counts + 1e-6) / (actual_counts + 1e-6))
    )
    return psi

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 Model Monitoring Dashboard")

dataset_name = st.selectbox(
    "Select Dataset",
    list(datasets.keys())
)

data = datasets[dataset_name]

# Ensure only model features used
data_model = data[feature_columns]

# ----------------------------
# PSI TABLE
# ----------------------------
st.subheader("PSI (Population Stability Index)")

psi_results = []

for col in feature_columns:
    psi = calculate_psi(train[col], data[col])
    psi_results.append([col, psi])

psi_df = pd.DataFrame(psi_results, columns=["Feature", "PSI"])
st.dataframe(psi_df)

# ----------------------------
# Feature Distribution Plot
# ----------------------------
st.subheader("Feature Distribution Comparison")

feature = st.selectbox("Select Feature", feature_columns)

fig, ax = plt.subplots()

ax.hist(train[feature], alpha=0.5, label="Training")
ax.hist(data[feature], alpha=0.5, label=dataset_name)

ax.set_title(f"Distribution of {feature}")
ax.legend()

st.pyplot(fig)

# ----------------------------
# Model Performance Over Time (placeholder)
# ----------------------------
st.subheader("Model Performance Over Time")

performance = pd.DataFrame({
    "Dataset": ["Training", "Month 1", "Month 2", "Month 3"],
    "Accuracy": [0.85, 0.83, 0.80, 0.76]
})

st.line_chart(performance.set_index("Dataset"))

# ----------------------------
# Retraining Trigger
# ----------------------------
st.subheader("Retraining Status")

avg_psi = psi_df["PSI"].mean()
final_accuracy = performance["Accuracy"].iloc[-1]

if avg_psi > 0.2 or final_accuracy < 0.8:
    st.error("⚠ Retraining Triggered")
else:
    st.success("✔ No Retraining Needed")