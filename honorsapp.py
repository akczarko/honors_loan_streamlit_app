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
# Load datasets (ROOT LEVEL FILES)
# ----------------------------
train = pd.read_csv("train.csv")
m1 = pd.read_csv("month1.csv")
m2 = pd.read_csv("month2.csv")
m3 = pd.read_csv("month3.csv")

datasets = {
    "Training": train,
    "Month 1": m1,
    "Month 2": m2,
    "Month 3": m3
}

# ----------------------------
# CLEAN COLUMN NAMES
# ----------------------------
train.columns = train.columns.str.strip()
m1.columns = m1.columns.str.strip()
m2.columns = m2.columns.str.strip()
m3.columns = m3.columns.str.strip()

feature_columns = [col.strip() for col in feature_columns]

# ----------------------------
# FILTER VALID FEATURES
# (removes constants, missing cols, etc.)
# ----------------------------
valid_features = [
    col for col in feature_columns
    if col in train.columns
    and train[col].dtype != "object"
    and train[col].nunique() > 1
]

# ----------------------------
# PSI FUNCTION
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
# STREAMLIT UI
# ----------------------------
st.title("📊 Model Monitoring Dashboard")

dataset_name = st.selectbox(
    "Select Dataset",
    list(datasets.keys())
)

data = datasets[dataset_name]

# ----------------------------
# SAFE FEATURE SELECTION
# ----------------------------
available_features = [col for col in valid_features if col in data.columns]

# ----------------------------
# PSI TABLE
# ----------------------------
st.subheader("PSI (Population Stability Index)")

psi_results = []

for col in available_features:
    psi = calculate_psi(train[col], data[col])
    psi_results.append([col, psi])

psi_df = pd.DataFrame(psi_results, columns=["Feature", "PSI"])
st.dataframe(psi_df)

# ----------------------------
# FEATURE DISTRIBUTION
# ----------------------------
st.subheader("Feature Distribution Comparison")

feature = st.selectbox("Select Feature", available_features)

fig, ax = plt.subplots()

ax.hist(train[feature], alpha=0.5, label="Training")
ax.hist(data[feature], alpha=0.5, label=dataset_name)

ax.set_title(f"Distribution of {feature}")
ax.legend()

st.pyplot(fig)

# ----------------------------
# MODEL PERFORMANCE (placeholder)
# ----------------------------
st.subheader("Model Performance Over Time")

performance = pd.DataFrame({
    "Dataset": ["Training", "Month 1", "Month 2", "Month 3"],
    "Accuracy": [0.85, 0.83, 0.80, 0.76]
})

st.line_chart(performance.set_index("Dataset"))

# ----------------------------
# RETRAINING LOGIC (FIXED)
# ----------------------------
st.subheader("Retraining Status")

if len(psi_df) > 0:
    max_psi = psi_df["PSI"].max()
else:
    max_psi = 0

final_accuracy = performance["Accuracy"].iloc[-1]

if max_psi > 0.2:
    st.error("⚠ Retraining Triggered (Data Drift Detected)")
else:
    st.success("✔ No Retraining Needed")