# =============================================================================
# app.py - Cloud-Based Monitoring Dashboard (Streamlit)
# Project  : Uncertainty-Aware Attention-Based Predictive Maintenance
#            for Wind Turbines with Cloud-Based Monitoring
#
# Proposal reference: Methodology Step 6 - Deployment
#   - Develop cloud-based monitoring dashboard using Streamlit
#   - Display failure probability and uncertainty scores
#   - Measure inference latency and prediction stability
#   - Validate system performance under simulated real-time conditions
#
# Research Question 3 (RQ3):
#   Can the proposed model maintain acceptable real-time inference latency
#   (<= 500 ms per prediction) when deployed in a cloud-based monitoring system?
#
# Run instructions:
#   pip install streamlit torch scikit-learn numpy pandas
#   streamlit run app.py
# =============================================================================

import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance Dashboard",
    layout="wide",
)

st.title("Wind Turbine Predictive Maintenance - Uncertainty-Aware Monitoring")
st.markdown(
    "This dashboard simulates real-time fault detection using an "
    "Attention-Based Temporal Encoder with Monte Carlo Dropout uncertainty "
    "estimation. It demonstrates cloud deployment feasibility and measures "
    "inference latency (target: <= 500 ms per prediction)."
)


# =============================================================================
# MODEL ARCHITECTURE
# Must be defined identically to the training cell so that saved weights load
# correctly. Any change here will cause a state_dict mismatch error.
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding - identical to the training cell definition.
    Required to reconstruct the model architecture before loading weights.
    """

    def __init__(self, d_model, max_len=200, dropout=0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe           = torch.zeros(max_len, d_model)
        position     = torch.arange(0, max_len).unsqueeze(1).float()
        div_term     = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionTemporalEncoder(nn.Module):
    """
    Attention-based temporal encoder with MC Dropout support.
    Identical architecture to the training cell.
    """

    def __init__(self, input_size=5, d_model=128, nhead=4,
                 num_layers=3, dim_feedforward=256, dropout=0.05):
        super().__init__()
        self.input_projection    = nn.Linear(input_size, d_model)
        self.pos_encoding        = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer            = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def enable_dropout(self):
        """Re-enable Dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_model(model_path):
    """
    Load the trained attention model from a saved state_dict file.
    Cached so the model is loaded only once per session.

    Parameters
    ----------
    model_path : str - path to the .pth file saved by the training cell

    Returns
    -------
    model : AttentionTemporalEncoder loaded to CPU, set to eval mode
    """
    model = AttentionTemporalEncoder(
        input_size=5, d_model=128, nhead=4,
        num_layers=3, dim_feedforward=256, dropout=0.05,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =============================================================================
# MC DROPOUT INFERENCE FUNCTION
# =============================================================================

def mc_predict(model, window_tensor, n_passes=50):
    """
    Run Monte Carlo Dropout inference on a single input window.

    Parameters
    ----------
    model         : AttentionTemporalEncoder - loaded trained model
    window_tensor : torch.Tensor, shape (1, window_size, n_features)
    n_passes      : int - number of stochastic forward passes (T)

    Returns
    -------
    mean_prob      : float - mean fault probability across T passes
    uncertainty    : float - std dev across T passes (predictive uncertainty)
    latency_ms     : float - total wall-clock inference time in milliseconds
    pass_probs     : list  - individual pass probabilities (for display)
    """
    model.eval()
    model.enable_dropout()  # keep dropout active for stochastic inference

    pass_probs  = []
    start_time  = time.time()

    with torch.no_grad():
        for _ in range(n_passes):
            logit = model(window_tensor)
            prob  = torch.sigmoid(logit).item()
            pass_probs.append(prob)

    latency_ms  = (time.time() - start_time) * 1000  # convert to milliseconds
    mean_prob   = float(np.mean(pass_probs))
    uncertainty = float(np.std(pass_probs))

    return mean_prob, uncertainty, latency_ms, pass_probs


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.header("Model Configuration")

model_path = st.sidebar.text_input(
    label       = "Path to trained model (.pth file)",
    value       = "proposed_attention.pth",
    help        = "Generated by the training notebook (Cell 2).",
)

n_passes = st.sidebar.slider(
    label = "MC Dropout passes (T)",
    min_value = 10,
    max_value = 100,
    value     = 50,
    step      = 10,
    help      = "Higher T gives more stable uncertainty estimate but slower inference.",
)

uncertainty_threshold = st.sidebar.slider(
    label     = "Uncertainty threshold (std dev)",
    min_value = 0.01,
    max_value = 0.30,
    value     = 0.10,
    step      = 0.01,
    help      = "Windows above this threshold are flagged for human review.",
)

fault_threshold = st.sidebar.slider(
    label     = "Fault probability threshold",
    min_value = 0.10,
    max_value = 0.90,
    value     = 0.50,
    step      = 0.05,
    help      = "Mean predicted probability above which a fault is declared.",
)

st.sidebar.markdown("---")
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio(
    label   = "Data source",
    options = ["Enter sensor values manually", "Upload SCADA window (numpy .npy)"],
)

# =============================================================================
# LOAD MODEL
# =============================================================================
model_loaded = False
if os.path.exists(model_path):
    model        = load_model(model_path)
    model_loaded = True
    st.sidebar.success("Model loaded successfully.")
else:
    st.sidebar.error(
        "Model file not found. Run the training notebook (Cell 2) first "
        "and set the correct path above."
    )

# =============================================================================
# INPUT SECTION
# =============================================================================
st.header("Sensor Input")
st.markdown(
    "A prediction window consists of **60 consecutive 10-minute readings** "
    "(10 hours of sensor history). Enter representative values below to "
    "simulate a single real-time prediction request."
)

WINDOW_SIZE = 60
N_FEATURES  = 5
FEATURE_NAMES = [
    "active_power_kw",
    "wind_speed_ms",
    "theoretical_power_kwh",
    "wind_dir_sin",
    "wind_dir_cos",
]

if input_mode == "Enter sensor values manually":
    st.subheader("Single time-step values (replicated across the 60-step window)")
    st.markdown(
        "For demonstration, the entered values are tiled into a 60-step window. "
        "In a live deployment this window would be filled with the most recent "
        "60 readings from the SCADA data stream."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        active_power = st.number_input(
            label    = "Active Power (kW)  - scaled [0, 1]",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.05,
            step      = 0.01,
            help      = "MinMax-scaled value. 0 = 0 kW, 1 = max observed power.",
        )
        wind_speed = st.number_input(
            label     = "Wind Speed (m/s)  - scaled [0, 1]",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.45,
            step      = 0.01,
        )

    with col2:
        theoretical_power = st.number_input(
            label     = "Theoretical Power  - scaled [0, 1]",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.50,
            step      = 0.01,
        )
        wind_dir_sin = st.number_input(
            label     = "Wind Dir Sin  [-1 to 1, then scaled]",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.50,
            step      = 0.01,
        )

    with col3:
        wind_dir_cos = st.number_input(
            label     = "Wind Dir Cos  [-1 to 1, then scaled]",
            min_value = 0.0,
            max_value = 1.0,
            value     = 0.50,
            step      = 0.01,
        )

    # Build a 60-step window by repeating the single entered values
    single_step   = np.array(
        [active_power, wind_speed, theoretical_power,
         wind_dir_sin, wind_dir_cos], dtype=np.float32)
    window_array  = np.tile(single_step, (WINDOW_SIZE, 1))  # (60, 5)
    input_source  = "Manual entry"

else:
    uploaded = st.file_uploader(
        label = "Upload a (60, 5) numpy array saved as .npy",
        type  = ["npy"],
    )
    if uploaded is not None:
        window_array = np.load(uploaded).astype(np.float32)
        if window_array.shape != (WINDOW_SIZE, N_FEATURES):
            st.error(
                "Array shape " + str(window_array.shape) +
                " does not match expected (60, 5). Please upload the correct file."
            )
            window_array = None
        else:
            input_source = "Uploaded .npy file"
    else:
        window_array = None
        st.info("Upload a numpy file to proceed, or switch to manual entry mode.")

# =============================================================================
# PREDICTION SECTION
# =============================================================================
st.header("Prediction")

if st.button("Run prediction") and model_loaded and window_array is not None:

    # Prepare the tensor: shape (1, 60, 5)
    window_tensor = torch.tensor(window_array).unsqueeze(0)  # add batch dim

    # Run MC Dropout inference
    mean_prob, uncertainty, latency_ms, pass_probs = mc_predict(
        model, window_tensor, n_passes=n_passes)

    # Determine classification outcome
    is_high_uncertainty = uncertainty > uncertainty_threshold
    if is_high_uncertainty:
        classification = "UNCERTAIN - flag for human review"
    elif mean_prob >= fault_threshold:
        classification = "FAULT DETECTED"
    else:
        classification = "Normal operation"

    # Latency target check (RQ3: <= 500 ms)
    latency_target_met = latency_ms <= 500.0

    # Display results
    st.subheader("Prediction Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Fault Probability", value=str(round(mean_prob * 100, 1)) + " %")
    col2.metric(label="Uncertainty (std dev)",
                value=str(round(uncertainty, 4)))
    col3.metric(label="Inference Latency",
                value=str(round(latency_ms, 1)) + " ms",
                delta="Within 500 ms target" if latency_target_met
                else "Exceeds 500 ms target",
                delta_color="normal" if latency_target_met else "inverse")
    col4.metric(label="MC Passes", value=str(n_passes))

    st.subheader("Classification")
    if "FAULT" in classification:
        st.error("**" + classification + "**")
    elif "UNCERTAIN" in classification:
        st.warning("**" + classification + "**")
    else:
        st.success("**" + classification + "**")

    # Detailed interpretation
    st.markdown("**Interpretation:**")
    st.markdown(
        "- Fault probability: **" + str(round(mean_prob * 100, 1)) + "%**  "
        "(threshold: " + str(round(fault_threshold * 100, 0)) + "%)"
    )
    st.markdown(
        "- Predictive uncertainty: **" + str(round(uncertainty, 4)) + "**  "
        "(threshold: " + str(uncertainty_threshold) + ")"
    )
    st.markdown(
        "- Inference latency: **" + str(round(latency_ms, 1)) + " ms**  "
        "(target: <= 500 ms  |  " +
        ("MET" if latency_target_met else "NOT MET") + ")"
    )

    # Pass distribution chart (prediction stability)
    st.subheader("MC Dropout Pass Distribution (Prediction Stability)")
    st.markdown(
        "Each bar shows the fault probability from one of the " +
        str(n_passes) + " stochastic forward passes. "
        "A narrow distribution indicates stable, confident predictions. "
        "A wide distribution indicates high uncertainty."
    )

    chart_df = pd.DataFrame({
        "Pass"        : list(range(1, n_passes + 1)),
        "Probability" : pass_probs,
    })
    st.bar_chart(chart_df.set_index("Pass")["Probability"])

    # Latency benchmark results
    st.subheader("Latency Benchmark  (Research Question 3)")
    st.markdown(
        "RQ3 asks whether the model can maintain acceptable real-time "
        "inference latency (<= 500 ms per prediction) when deployed."
    )

    benchmark_results = {
        "MC Passes (T)"        : n_passes,
        "Total latency (ms)"   : round(latency_ms, 2),
        "Latency per pass (ms)": round(latency_ms / n_passes, 2),
        "Target (<= 500 ms)"   : "<= 500 ms",
        "Target met"           : "YES" if latency_target_met else "NO",
    }
    st.table(pd.DataFrame.from_dict(
        benchmark_results, orient="index", columns=["Value"]))

    # Stability check: additional repeated runs to assess prediction stability
    st.subheader("Prediction Stability Under Repeated Runs")
    st.markdown(
        "The model is run 5 additional times on the same input to assess "
        "stability. A well-calibrated model should produce consistent "
        "mean probabilities across repeated calls."
    )

    stability_records = []
    for run_i in range(1, 6):
        p, u, lat, _ = mc_predict(model, window_tensor, n_passes=n_passes)
        stability_records.append({
            "Run"                : run_i,
            "Mean probability"   : round(p, 4),
            "Uncertainty (std)"  : round(u, 4),
            "Latency (ms)"       : round(lat, 1),
            "Classification"     : "Fault" if p >= fault_threshold else "Normal",
        })

    stability_df = pd.DataFrame(stability_records)
    st.table(stability_df)

    prob_std_across_runs = stability_df["Mean probability"].std()
    st.markdown(
        "Std dev of mean probability across 5 repeated runs: **" +
        str(round(prob_std_across_runs, 4)) +
        "**  (lower = more stable)"
    )

# =============================================================================
# METHODOLOGY EXPLANATION SECTION
# =============================================================================
st.markdown("---")
st.header("Methodology Reference")

with st.expander("How the prediction is generated"):
    st.markdown("""
**Step 1 - Input preparation**

The dashboard receives 60 consecutive 10-minute SCADA readings
(active power, wind speed, theoretical power, wind direction sin/cos).
Each feature is MinMax-scaled to [0, 1] using the scaler fitted on
the training data during model training.

**Step 2 - Monte Carlo Dropout inference**

The trained Attention-Based Temporal Encoder is called T times with
dropout layers kept active (via `enable_dropout()`). Each forward pass
produces a slightly different probability estimate due to the random
dropout masks. This is the MC Dropout approximation of Bayesian inference
(Gal & Ghahramani, 2016).

**Step 3 - Uncertainty estimation**

- Mean of T pass probabilities --> final fault probability
- Std dev of T pass probabilities --> predictive uncertainty

**Step 4 - Confidence-based thresholding**

If the predictive uncertainty exceeds the configured threshold,
the prediction is flagged as uncertain and sent for human review
rather than triggering an automatic alert. This directly reduces
false alarms caused by overconfident predictions.

**Step 5 - Latency measurement**

Wall-clock time is recorded around all T forward passes to give
total inference latency. The target from the project proposal is
<= 500 ms per prediction including T=50 MC passes.
    """)

with st.expander("Research questions addressed by this dashboard"):
    st.markdown("""
**RQ1** (addressed in the training notebook):
Does an attention-based model outperform LSTM and CNN-LSTM on F1-score?

**RQ2** (addressed in the training notebook and here):
Does MC Dropout improve calibration and reduce false alarms?
The uncertainty score displayed here enables confidence-based filtering.

**RQ3** (addressed by this dashboard):
Can the model maintain <= 500 ms inference latency in a cloud deployment?
The latency benchmark table above records and validates this.
    """)

st.markdown("---")
st.caption(
    "Uncertainty-Aware Attention-Based Predictive Maintenance for Wind Turbines "
    "| MSc Project | Dataset: Kaggle - berkerisen/wind-turbine-scada-dataset"
)
