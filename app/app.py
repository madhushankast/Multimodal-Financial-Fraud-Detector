"""
Streamlit Web Application: Real-Time Multimodal Financial Fraud Risk Detector.
Phase 11: Production-style, decoupled interactive dashboard consuming
FraudInferencePipeline (src.inference) for real-time scoring, threshold tuning,
local explainability evidence, and batch clearing audits.
"""

from pathlib import Path
import sys
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.inference import FraudInferencePipeline

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Modern Card Container */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    /* Decision Badges */
    .badge-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid #10b981;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }
    
    .badge-flagged {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.25));
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    .badge-elevated {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.25));
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    /* Disclaimer box */
    .disclaimer-box {
        font-size: 0.80rem;
        color: #94a3b8;
        background: rgba(148, 163, 184, 0.08);
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PRESET TRANSACTION REPOSITORY
# -----------------------------------------------------------------------------
PRESET_NORMAL = {
    "TransactionID": 3000001,
    "TransactionAmt": 45.00,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 150,
    "card3": 150,
    "card4": "discover",
    "card5": 142,
    "card6": "credit",
    "addr1": 315,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "DeviceType": "desktop",
    "DeviceInfo": "Windows",
    "C1": 1, "C2": 1, "C3": 0, "C4": 0, "C5": 0,
    "D1": 14, "D2": 14, "D3": 13, "D4": 0
}

PRESET_FRAUD = {
    "TransactionID": 3000002,
    "TransactionAmt": 289.50,
    "ProductCD": "C",
    "card1": 17188,
    "card2": 268,
    "card3": 185,
    "card4": "visa",
    "card5": 166,
    "card6": "credit",
    "addr1": 299,
    "addr2": 87,
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "anonymous.com",
    "DeviceType": "mobile",
    "DeviceInfo": "Trident/7.0",
    "C1": 18, "C2": 18, "C3": 0, "C4": 6, "C5": 0,
    "D1": 0, "D2": 0, "D3": 0, "D4": 0,
    "V45": 3.0, "V78": 2.0, "V87": 3.0
}

PRESET_ELEVATED = {
    "TransactionID": 3000003,
    "TransactionAmt": 490.00,
    "ProductCD": "H",
    "card1": 9500,
    "card2": 321,
    "card3": 150,
    "card4": "mastercard",
    "card5": 226,
    "card6": "debit",
    "addr1": 204,
    "addr2": 87,
    "P_emaildomain": "hotmail.com",
    "R_emaildomain": "outlook.com",
    "DeviceType": "mobile",
    "DeviceInfo": "iOS Device",
    "C1": 4, "C2": 5, "C3": 0, "C4": 2, "C5": 1,
    "D1": 45, "D2": 45, "D3": 0, "D4": 12,
    "V87": 1.5, "V45": 1.0
}

# -----------------------------------------------------------------------------
# CACHED BACKEND INFERENCE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Neural Inference Pipeline & Preprocessors...")
def get_inference_pipeline() -> FraudInferencePipeline:
    """
    Factory loader: restores trained Hybrid neural model and preprocessor artifacts.
    Zero-leakage guarantee: loaded strictly from frozen disk checkpoints.
    """
    model_path = PROJECT_ROOT / "models" / "pytorch_hybrid_weighted_bce.pt"
    preprocessor_path = PROJECT_ROOT / "models" / "tabular_preprocessor.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor checkpoint missing at {preprocessor_path}")

    return FraudInferencePipeline.from_artifacts(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold=0.80
    )

try:
    pipeline = get_inference_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Fraud Risk Engine")
    st.caption("PyTorch Hybrid Tabular Anomaly Detector")
    
    st.markdown("---")
    st.subheader("⚙️ Decision Policy")
    
    threshold_slider = st.slider(
        "Operating Threshold (θ)",
        min_value=0.05,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Probability cut-off for flagging transactions. Default θ=0.80 derived from Phase 8 cost optimization ($C_FP=$15)."
    )
    
    if pipeline_ready:
        pipeline.threshold = threshold_slider

    st.info(
        f"**Cost-Calibrated Policy**: θ = {threshold_slider:.2f}\n\n"
        f"• Prob ≥ {threshold_slider:.2f} ➔ **FLAG FOR REVIEW**\n"
        f"• Prob < {threshold_slider:.2f} ➔ **APPROVE**"
    )

    st.markdown("---")
    st.subheader("System Status")
    if pipeline_ready:
        st.success("● Inference Engine: ACTIVE")
        st.write(f"• **Architecture**: PyTorch Hybrid MLP")
        st.write(f"• **Features**: {len(pipeline.numerical_cols)} Num + {len(pipeline.categorical_cols)} Cat")
        st.write(f"• **Device**: `{pipeline.device}`")
        st.write(f"• **Embeddings**: 568 Total Dimensions")
    else:
        st.error("● Inference Engine: OFFLINE")
        st.caption(f"Error: {pipeline_error}")

    st.markdown("---")
    st.caption("3rd-Year Undergraduate ML Project | IEEE-CIS Dataset Benchmark")

# -----------------------------------------------------------------------------
# MAIN VIEW: HEADER & NAVIGATION TABS
# -----------------------------------------------------------------------------
st.title("🛡️ Multimodal Financial Fraud Detector")
st.markdown(
    "Enterprise-grade transaction risk assessment combining continuous numerical features "
    "with categorical entity embeddings, class-imbalance loss formulations, and local sensitivity evidence."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Real-Time Transaction Scoring",
    "📊 Batch Fraud Clearing & Audit",
    "📖 Model Architecture & Evaluation"
])

# -----------------------------------------------------------------------------
# TAB 1: REAL-TIME SINGLE TRANSACTION SCORING
# -----------------------------------------------------------------------------
with tab1:
    if not pipeline_ready:
        st.error(f"Cannot perform real-time scoring. Pipeline failed to load: {pipeline_error}")
    else:
        st.subheader("Single Transaction Analysis")
        st.write("Score individual transactions against the frozen hybrid deep neural network.")

        col_input_ctrl, col_sample = st.columns([2, 1])
        with col_input_ctrl:
            preset_choice = st.selectbox(
                "Choose Transaction Archetype Preset",
                [
                    "Normal E-Commerce Purchase (Discover / Windows / US)",
                    "High-Risk Fraud Anomaly (Visa / Trident / Anonymous.com)",
                    "Cross-Border Velocity Surge (Mastercard / Mobile / Debit)",
                    "Custom Input Payload (JSON Editor)"
                ]
            )

        # Populate transaction dictionary based on selection
        if preset_choice.startswith("Normal"):
            current_payload = PRESET_NORMAL.copy()
        elif preset_choice.startswith("High-Risk"):
            current_payload = PRESET_FRAUD.copy()
        elif preset_choice.startswith("Cross-Border"):
            current_payload = PRESET_ELEVATED.copy()
        else:
            current_payload = PRESET_NORMAL.copy()

        with st.expander("🛠️ Transaction Feature Payload Details", expanded=True):
            input_mode = st.radio("Input Format", ["Guided Form", "Raw JSON"], horizontal=True)

            if input_mode == "Guided Form":
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    txn_id = st.number_input("Transaction ID", value=int(current_payload.get("TransactionID", 3000001)), step=1)
                    amt = st.number_input("Amount ($ USD)", value=float(current_payload.get("TransactionAmt", 50.0)), step=5.0)
                    prod = st.selectbox("Product Code", ["W", "C", "R", "H", "S"], index=["W", "C", "R", "H", "S"].index(current_payload.get("ProductCD", "W")))
                with fc2:
                    card_brand = st.selectbox("Card Brand (card4)", ["visa", "mastercard", "discover", "american express"], 
                                              index=["visa", "mastercard", "discover", "american express"].index(current_payload.get("card4", "visa")))
                    card_type = st.selectbox("Card Type (card6)", ["credit", "debit", "charge card"], 
                                             index=["credit", "debit", "charge card"].index(current_payload.get("card6", "credit")))
                    card1 = st.number_input("Issuer Bank ID (card1)", value=int(current_payload.get("card1", 13926)), step=1)
                with fc3:
                    p_email = st.text_input("Purchaser Email Domain", value=str(current_payload.get("P_emaildomain", "gmail.com")))
                    r_email = st.text_input("Recipient Email Domain", value=str(current_payload.get("R_emaildomain", "gmail.com")))
                    device_type = st.selectbox("Device Type", ["desktop", "mobile", "tablet", "missing"], 
                                               index=["desktop", "mobile", "tablet", "missing"].index(current_payload.get("DeviceType", "desktop") if current_payload.get("DeviceType") in ["desktop", "mobile", "tablet"] else "missing"))
                with fc4:
                    dev_info = st.text_input("Device Hardware / OS", value=str(current_payload.get("DeviceInfo", "Windows")))
                    c1_cnt = st.number_input("Velocity Count C1", value=int(current_payload.get("C1", 1)), step=1)
                    c4_cnt = st.number_input("Velocity Count C4", value=int(current_payload.get("C4", 0)), step=1)

                active_payload = current_payload.copy()
                active_payload.update({
                    "TransactionID": txn_id,
                    "TransactionAmt": amt,
                    "ProductCD": prod,
                    "card4": card_brand,
                    "card6": card_type,
                    "card1": card1,
                    "P_emaildomain": p_email,
                    "R_emaildomain": r_email,
                    "DeviceType": device_type if device_type != "missing" else np.nan,
                    "DeviceInfo": dev_info,
                    "C1": c1_cnt,
                    "C4": c4_cnt
                })
            else:
                json_str = st.text_area("JSON Payload", value=json.dumps(current_payload, indent=2), height=240)
                try:
                    active_payload = json.loads(json_str)
                except Exception as ex:
                    st.error(f"Malformed JSON: {ex}")
                    active_payload = current_payload

        # Action Button
        score_btn = st.button("🚀 Score Live Transaction", type="primary", use_container_width=True)

        if score_btn or "last_score_result" in st.session_state:
            if score_btn:
                with st.spinner("Executing neural inference and computing local attribution..."):
                    res = pipeline.predict(active_payload, include_explanation=True, top_k=5)
                    st.session_state["last_score_result"] = res
            else:
                res = st.session_state["last_score_result"]

            prob = res["fraud_probability"]
            decision = "FLAG FOR MANUAL REVIEW" if prob >= threshold_slider else "APPROVE"
            if prob >= 0.80:
                risk_tier = "CRITICAL / HIGH RISK"
            elif prob >= threshold_slider:
                risk_tier = "ELEVATED RISK"
            else:
                risk_tier = "NORMAL / LOW RISK"

            st.markdown("### Decision & Assessment Summary")
            
            # Result Metric Badges
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Fraud Probability</div>
                    <div class="metric-value" style="color: {'#ef4444' if prob >= threshold_slider else '#10b981'};">{prob:.4f}</div>
                    <div class="metric-sub">Sigmoid Logit Output (0.00 - 1.00)</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                badge_class = "badge-flagged" if decision != "APPROVE" else "badge-approved"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Operational Decision</div>
                    <div style="margin-top: 6px;"><span class="{badge_class}">{decision}</span></div>
                    <div class="metric-sub" style="margin-top: 8px;">Threshold θ = {threshold_slider:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                tier_color = "#ef4444" if "CRITICAL" in risk_tier else ("#f59e0b" if "ELEVATED" in risk_tier else "#10b981")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Assigned Risk Tier</div>
                    <div class="metric-value" style="color: {tier_color}; font-size: 1.35rem;">{risk_tier}</div>
                    <div class="metric-sub">Financial Exposure Policy</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Latency</div>
                    <div class="metric-value" style="color: #38bdf8;">{res['execution_time_ms']:.1f} <span style="font-size: 1rem;">ms</span></div>
                    <div class="metric-sub">Preprocessing + Neural + Explainability</div>
                </div>
                """, unsafe_allow_html=True)

            # Visual Probability Gauge / Progress
            st.markdown("#### Probability vs. Decision Policy Boundary")
            gauge_df = pd.DataFrame({
                "Zone": ["Normal / Low Risk", "Elevated Risk", "Critical / High Risk"],
                "Start": [0.0, threshold_slider, 0.80 if threshold_slider < 0.80 else threshold_slider],
                "End": [threshold_slider, 0.80 if threshold_slider < 0.80 else 1.0, 1.0]
            })

            progress_col, val_col = st.columns([5, 1])
            with progress_col:
                st.progress(min(max(prob, 0.0), 1.0))
            with val_col:
                st.write(f"**Score: {prob*100:.1f}%**")

            # Local Explainability Evidence
            st.markdown("#### 🔬 Local Sensitivity Evidence (Top Contributing Features)")
            st.caption(
                "Attribution values quantify the increase in fraud probability caused by each feature "
                "relative to an imputation baseline (measured via counterfactual perturbation). "
                "These represent model sensitivity, not causal real-world claims."
            )

            evidence_list = res.get("top_contributing_evidence", [])
            if evidence_list:
                df_ev = pd.DataFrame(evidence_list)
                df_ev["Observed_Value"] = df_ev["Observed_Value"].astype(str)
                
                exp_c1, exp_c2 = st.columns([3, 2])
                with exp_c1:
                    chart = (
                        alt.Chart(df_ev)
                        .mark_bar(cornerRadiusEnd=5, height=22)
                        .encode(
                            x=alt.X("Probability_Contribution:Q", title="Probability Sensitivity (+Δ Prob)"),
                            y=alt.Y("Feature:N", sort="-x", title="Feature"),
                            color=alt.Color(
                                "Feature_Type:N", 
                                scale=alt.Scale(domain=["Numerical", "Categorical"], range=["#38bdf8", "#34d399"]),
                                title="Feature Group"
                            ),
                            tooltip=["Feature", "Feature_Type", "Observed_Value", "Probability_Contribution"]
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)
                with exp_c2:
                    st.dataframe(
                        df_ev.style.format({"Probability_Contribution": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No high-sensitivity anomaly features identified for this transaction.")

            # Raw API Response
            with st.expander("📄 View Structured API JSON Response"):
                st.json(res)

# -----------------------------------------------------------------------------
# TAB 2: BATCH FRAUD CLEARING & AUDIT
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Batch Transaction Clearing & Risk Audit")
    st.write("Score high-volume transaction files using vectorized PyTorch batch inference.")

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Transaction File (.CSV)", type=["csv"])
    with b_col2:
        st.write("Or evaluate using built-in validation sample:")
        use_sample = st.button("📂 Load 500-Transaction Sample Batch", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` with {len(batch_df):,} transactions.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif use_sample:
        sample_path = PROJECT_ROOT / "data" / "sample_batch_500.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
            st.info(f"Loaded built-in benchmark batch: 500 test transactions from IEEE-CIS dataset.")
        else:
            st.warning("Sample batch file not found on disk. Please upload a CSV.")

    if batch_df is not None:
        st.write(f"**Preview Input Batch ({len(batch_df):,} rows × {len(batch_df.columns)} cols):**")
        st.dataframe(batch_df.head(5), use_container_width=True)

        run_batch_btn = st.button("⚡ Execute Batch Risk Scoring", type="primary")

        if run_batch_btn or "last_batch_results" in st.session_state:
            if run_batch_btn:
                with st.spinner(f"Scoring {len(batch_df):,} transactions with PyTorch batch inference..."):
                    t_start = time.perf_counter()
                    pipeline.threshold = threshold_slider
                    scored_df = pipeline.predict_batch(batch_df, batch_size=2048)
                    batch_latency = time.perf_counter() - t_start
                    st.session_state["last_batch_results"] = (scored_df, batch_latency)
            else:
                scored_df, batch_latency = st.session_state["last_batch_results"]

            # Batch Summary Metrics
            total_txns = len(scored_df)
            flagged_cnt = int((scored_df["decision"] == "FLAG FOR MANUAL REVIEW").sum())
            approved_cnt = total_txns - flagged_cnt
            critical_cnt = int((scored_df["risk_tier"] == "CRITICAL / HIGH RISK").sum())
            mean_prob = scored_df["fraud_probability"].mean()
            throughput = total_txns / batch_latency if batch_latency > 0 else 0

            st.markdown("### Batch Audit Summary")
            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                st.metric("Total Scored", f"{total_txns:,}", f"{throughput:,.0f} txn/sec")
            with bm2:
                st.metric("Approved (Low Risk)", f"{approved_cnt:,}", f"{approved_cnt/total_txns*100:.1f}%")
            with bm3:
                st.metric("Flagged for Review", f"{flagged_cnt:,}", f"{flagged_cnt/total_txns*100:.1f}%", delta_color="inverse")
            with bm4:
                st.metric("Critical Alerts", f"{critical_cnt:,}", f"{critical_cnt/total_txns*100:.1f}%", delta_color="inverse")

            # Batch Visualizations
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("#### Probability Distribution & Decision Boundary")
                prob_hist = (
                    alt.Chart(scored_df)
                    .mark_bar(opacity=0.7, color="#38bdf8")
                    .encode(
                        alt.X("fraud_probability:Q", bin=alt.Bin(maxbins=30), title="Fraud Probability"),
                        alt.Y("count()", title="Transaction Volume")
                    )
                    .properties(height=260)
                )
                rule = (
                    alt.Chart(pd.DataFrame({"threshold": [threshold_slider]}))
                    .mark_rule(color="#ef4444", strokeWidth=2, strokeDash=[5, 5])
                    .encode(x="threshold:Q")
                )
                st.altair_chart(prob_hist + rule, use_container_width=True)

            with bc2:
                st.markdown("#### Risk Tier Distribution")
                tier_counts = scored_df["risk_tier"].value_counts().reset_index()
                tier_counts.columns = ["Risk_Tier", "Count"]
                tier_chart = (
                    alt.Chart(tier_counts)
                    .mark_arc(innerRadius=45)
                    .encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(
                            field="Risk_Tier", 
                            type="nominal",
                            scale=alt.Scale(
                                domain=["NORMAL / LOW RISK", "ELEVATED RISK", "CRITICAL / HIGH RISK"],
                                range=["#10b981", "#f59e0b", "#ef4444"]
                            )
                        ),
                        tooltip=["Risk_Tier", "Count"]
                    )
                    .properties(height=260)
                )
                st.altair_chart(tier_chart, use_container_width=True)

            # Filterable Results Table
            st.markdown("#### Scored Transactions Audit Log")
            filter_choice = st.selectbox("Filter Scored Records", ["All Transactions", "Flagged for Review Only", "Critical High-Risk Only"])
            
            if filter_choice == "Flagged for Review Only":
                display_df = scored_df[scored_df["decision"] == "FLAG FOR MANUAL REVIEW"]
            elif filter_choice == "Critical High-Risk Only":
                display_df = scored_df[scored_df["risk_tier"] == "CRITICAL / HIGH RISK"]
            else:
                display_df = scored_df

            show_cols = [c for c in ["TransactionID", "TransactionAmt", "ProductCD", "card4", "fraud_probability", "decision", "risk_tier"] if c in display_df.columns]
            st.dataframe(display_df[show_cols].head(100), use_container_width=True)

            # Export Button
            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Scored Batch CSV",
                data=csv_data,
                file_name=f"fraud_scored_batch_theta_{threshold_slider:.2f}.csv",
                mime="text/csv",
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# TAB 3: MODEL ARCHITECTURE & EVALUATION DOCUMENTATION
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model Architecture & Experimental Evidence")
    st.markdown(
        "Complete technical synthesis of the research and benchmarking progression "
        "conducted across Phases 1 through 10."
    )

    doc_c1, doc_c2 = st.columns(2)
    with doc_c1:
        st.markdown("#### 🧠 PyTorch Hybrid Architecture")
        st.markdown("""
        * **Input Representation**:
          * **381 Continuous Numerical Features**: Imputed using training-set medians and standardized via zero-leakage `StandardScaler`.
          * **49 Categorical Features**: Mapped through learned entity embeddings ($\text{dim} = \min(50, \max(4, \text{round}(6 \times V^{0.25})))$), yielding **568 total embedding dimensions**.
        * **Deep Neural Network Backbone**:
          * `BatchNorm1d(949)` input layer (381 numerical + 568 embeddings).
          * `Linear(949 ➔ 256)` + `BatchNorm1d(256)` + `ReLU` + `Dropout(0.30)`.
          * `Linear(256 ➔ 128)` + `BatchNorm1d(128)` + `ReLU` + `Dropout(0.30)`.
          * `Linear(128 ➔ 64)` + `BatchNorm1d(64)` + `ReLU` + `Dropout(0.20)`.
          * `Linear(64 ➔ 1)` raw fraud logit output.
        * **Loss Optimization**: Imbalance-aware Weighted BCE ($\text{pos\_weight} \approx 28.0$) outperforming standard BCE and Focal Loss on Precision & PR-AUC.
        """)

    with doc_c2:
        st.markdown("#### ⚖️ Operating Decision Threshold (Phase 8)")
        st.markdown("""
        * **Cost Utility Formulation**:
          $$\text{Total Cost} = \sum_{\text{FN}} \text{TransactionAmt} + C_{\text{FP}} \times \text{FP}$$
        * **Empirical Optimization ($C_{\text{FP}} = \$15$)**:
          * Standard $\theta = 0.50$: Net Test Cost = $\$25,547$ ($95$ False Positives).
          * Cost-Optimal $\theta^* = 0.80$: Net Test Cost = $\$20,105$ (Net Savings = **$\$12,140$**).
          * Reduces operational review queue by $78.9\%$ while safeguarding high-dollar fraud attacks.
        """)

    st.markdown("---")
    st.markdown("#### 📊 Empirical Baseline & Model Benchmarking (Measured on Chronological Test Split)")
    
    benchmark_data = {
        "Model Architecture": [
            "Logistic Regression (Baseline)",
            "Random Forest (100 Trees)",
            "XGBoost (Depth 6, subsample 0.8)",
            "PyTorch Numerical MLP (Standard BCE)",
            "PyTorch Numerical MLP (Weighted BCE)",
            "PyTorch Hybrid (Embeddings, Standard BCE)",
            "PyTorch Hybrid (Embeddings, Weighted BCE)",
            "PyTorch Hybrid + Focal Loss (γ=3)"
        ],
        "Input Features": ["381 Num", "381 Num", "381 Num", "381 Num", "381 Num", "381 Num + 49 Cat", "381 Num + 49 Cat", "381 Num + 49 Cat"],
        "Val PR-AUC": [0.3444, 0.5150, 0.6100, 0.5058, 0.4912, 0.5772, 0.5368, 0.5555],
        "Test PR-AUC": [0.2576, 0.3077, 0.4741, 0.3828, 0.3778, 0.4623, 0.4627, 0.4257],
        "Test ROC-AUC": [0.8106, 0.8579, 0.8759, 0.8277, 0.7862, 0.8702, 0.8799, 0.8497],
        "Operating θ*": [0.90, 0.70, 0.80, 0.15, 0.90, 0.20, 0.95, 0.45],
        "Test Precision": ["32.84%", "52.11%", "54.35%", "37.54%", "41.70%", "47.83%", "58.33%", "54.46%"],
        "Test Recall": ["36.39%", "24.26%", "49.18%", "39.02%", "38.69%", "43.28%", "43.61%", "40.00%"],
        "Test F1": [0.3453, 0.3311, 0.5164, 0.3826, 0.4014, 0.4544, 0.4991, 0.4612],
        "Test False Positives": [227, 68, 126, 198, 165, 144, 95, 102]
    }
    st.table(pd.DataFrame(benchmark_data))

    st.markdown("---")
    st.caption("Multimodal Financial Fraud Detector | Production-Grade Inference & UI | Phase 11")
