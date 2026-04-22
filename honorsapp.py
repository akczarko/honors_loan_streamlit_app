import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ----------------------------
# Load model
# ----------------------------
model = pickle.load(open("honors_model.pkl", "rb"))

# ----------------------------
# LOAD DATASETS
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
# PSI TABLE (FROM NOTEBOOK)
# ----------------------------
psi_table = pd.DataFrame({
    "Feature": ["Monthly_Gross_Income", "FICO_score", "Employment_Status"],
    "Month 1": [0.002202, 0.006784, 0.001106],
    "Month 2": [0.019302, 0.046496, 0.030080],
    "Month 3": [0.159389, 0.151672, 0.106134]
})

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
# SHOW PSI ONLY FOR SELECTED DATASET
# ----------------------------
st.subheader(f"PSI Values for {dataset_name}")

if dataset_name == "Training":
    st.info("Training set is baseline → PSI = 0 by definition")
    display_psi = pd.DataFrame({
        "Feature": psi_table["Feature"],
        "PSI": [0, 0, 0]
    })
elif dataset_name == "Month 1":
    display_psi = psi_table[["Feature", "Month 1"]].rename(columns={"Month 1": "PSI"})
elif dataset_name == "Month 2":
    display_psi = psi_table[["Feature", "Month 2"]].rename(columns={"Month 2": "PSI"})
else:
    display_psi = psi_table[["Feature", "Month 3"]].rename(columns={"Month 3": "PSI"})

st.dataframe(display_psi)

# ----------------------------
# FEATURE DISTRIBUTION (DYNAMIC)
# ----------------------------
st.subheader("Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    ["Monthly_Gross_Income", "FICO_score"]
)

fig, ax = plt.subplots()

ax.hist(train[feature], alpha=0.5, label="Training")
ax.hist(data[feature], alpha=0.5, label=dataset_name)

ax.set_title(f"{feature} Distribution")
ax.legend()

st.pyplot(fig)

# ----------------------------
# MODEL PERFORMANCE
# ----------------------------
st.subheader("Model Performance Over Time")

performance = pd.DataFrame({
    "Dataset": ["Training", "Month 1", "Month 2", "Month 3"],
    "Accuracy": [0.85, 0.83, 0.80, 0.76]
})

st.line_chart(performance.set_index("Dataset"))

# ----------------------------
# RETRAINING RULE (NOW DYNAMIC!)
# ----------------------------
st.subheader("Retraining Status")

if dataset_name == "Training":
    st.success("✔ Baseline dataset — No retraining needed")

elif dataset_name == "Month 1":
    max_psi = psi_table["Month 1"].max()
    if max_psi > 0.1:
        st.warning("⚠ Retraining Consideration")
    else:
        st.success("✔ Stable")

elif dataset_name == "Month 2":
    max_psi = psi_table["Month 2"].max()
    if max_psi > 0.1:
        st.warning("⚠ Retraining Consideration")
    else:
        st.success("✔ Stable")

else:
    max_psi = psi_table["Month 3"].max()
    if max_psi > 0.1:
        st.error("⚠ RETRAINING TRIGGERED")
    else:
        st.success("✔ Stable")

# ----------------------------
# INSIGHT
# ----------------------------
st.subheader("Summary Insight")

st.write(
    f"""
    For {dataset_name}, PSI indicates increasing drift over time across key features.
    The highest drift is observed in Month 3, supporting model degradation trends.
    """
)