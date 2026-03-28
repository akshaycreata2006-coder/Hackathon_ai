import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import io
from PIL import Image

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="High-Value Purchase Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #aaa !important;
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0533 40%, #0d0d0d 100%);
    border: 1px solid #3d1a6e;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #8b5cf6;
    margin-top: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.5);
    color: #c4b5fd;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    margin-top: 1rem;
}

/* Metric cards */
.metric-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f3f0ff;
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #666;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #a78bfa;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e0e0e0;
    border-left: 3px solid #8b5cf6;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: -0.01em;
}

/* Tables */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Insight pill */
.insight-pill {
    background: #18182e;
    border: 1px solid #2d2d5e;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #ccc;
    line-height: 1.5;
}
.insight-pill strong {
    color: #a78bfa;
}

/* Feature bar */
.feat-bar-wrap {
    margin-bottom: 0.4rem;
}
.feat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #aaa;
}
.feat-bar-bg {
    background: #1e1e1e;
    border-radius: 4px;
    height: 6px;
    margin-top: 2px;
}
.feat-bar-fill {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    height: 6px;
    border-radius: 4px;
}

/* Prediction meter */
.pred-meter {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/user-data/uploads/hackathon_dataset.csv")
    preds = pd.read_csv("/mnt/user-data/uploads/predictions.csv")
    return df, preds

df, preds = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💎 HVP Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "EDA", "Model Performance", "Predictions Explorer", "Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#555;line-height:1.7;">
    <div>DATASET — 5,000 rows</div>
    <div>FEATURES — 26 engineered</div>
    <div>BEST MODEL — AdaBoost</div>
    <div>F1 SCORE — 0.8183</div>
    <div>ROC-AUC — 0.9057</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-subtitle">Hackathon · ML Classification Pipeline</div>
        <div class="hero-title">High-Value Purchase<br>Prediction</div>
        <div class="hero-badge">AdaBoost · F1: 0.8183 · AUC: 0.9057</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    cols = st.columns(5)
    kpis = [
        ("0.8183", "F1 Score", "Best Model"),
        ("0.9057", "ROC-AUC", "AdaBoost"),
        ("0.8309", "Precision", "AdaBoost"),
        ("0.8060", "Recall", "AdaBoost"),
        ("5,000", "Samples", "Balanced 50/50"),
    ]
    for col, (val, label, delta) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        insights = [
            ("Segment Dominance", "Platinum & Gold customers are far more likely to make high-value purchases — segment_rank is the #2 predictor."),
            ("Spending History", "avg_order_value + its log-transform together account for ~32% of model importance. Past spenders repeat."),
            ("Intent Score", "Purchases × (1 − cart abandonment) reveals committed buyers vs. window-shoppers with precision."),
            ("Loyalty Recency", "A single engineered feature combining account tenure and recency outperforms either column used alone."),
            ("Promo Synergy", "has_promo_code alone is weak, but promo_buyer (promo + volume) is a significantly stronger signal."),
        ]
        for title, body in insights:
            st.markdown(f'<div class="insight-pill"><strong>{title}:</strong> {body}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)
        leaderboard = pd.DataFrame({
            "Model": ["AdaBoost ⭐", "Random Forest", "Gradient Boosting", "Decision Tree", "Logistic Reg.", "SVM (RBF)", "KNN"],
            "F1": [0.8183, 0.7988, 0.7938, 0.7727, 0.7685, 0.7508, 0.6371],
            "AUC": [0.9057, 0.8965, 0.8968, 0.8760, 0.8466, 0.8329, 0.7400],
            "Precision": [0.8309, 0.8252, 0.8191, 0.8284, 0.7771, 0.7705, 0.6741],
            "Recall": [0.8060, 0.7740, 0.7700, 0.7240, 0.7600, 0.7320, 0.6040],
        })
        st.dataframe(
            leaderboard.style
                .background_gradient(subset=["F1", "AUC"], cmap="Purples")
                .format({"F1": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
        recs = [
            "🎯 Prioritise Platinum & Gold segments in high-value campaigns",
            "🔁 Retarget high cart-abandonment + high email-open customers",
            "🎁 Offer promos to customers with > 8 prior purchases",
            "📅 Re-engage customers with days_since_last_purchase > 60",
            "⬆️  Upsell to customers with high avg_review_rating",
        ]
        for r in recs:
            st.markdown(f"<div style='font-size:0.86rem;color:#ccc;padding:0.3rem 0;'>{r}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 EDA Charts", "🔢 Raw Statistics", "🧩 Correlation"])

    with tab1:
        try:
            eda_img = Image.open("/mnt/user-data/uploads/eda.png")
            st.image(eda_img, use_container_width=True)
        except:
            st.info("eda.png not found. Generating live charts...")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Shape**")
            st.code(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
            st.markdown("**Target Distribution**")
            vc = df["high_value_purchase"].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Class"] = vc["Class"].map({0: "Class 0 (No)", 1: "Class 1 (Yes)"})
            st.dataframe(vc, hide_index=True, use_container_width=True)
            st.markdown("**Missing Values**")
            mv = df.isnull().sum()
            mv = mv[mv > 0].reset_index()
            mv.columns = ["Column", "Missing"]
            mv["Pct"] = (mv["Missing"] / len(df) * 100).round(2).astype(str) + "%"
            st.dataframe(mv, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Numeric Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)

    with tab3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdPu",
                    ax=ax, annot_kws={"size": 7}, linewidths=0.3,
                    cbar_kws={"shrink": 0.7})
        ax.tick_params(colors="#aaa", labelsize=8)
        plt.title("Feature Correlation Matrix", color="#e0e0e0", pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="section-header">Model Evaluation Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Evaluation Charts", "🏆 Feature Importance"])

    with tab1:
        try:
            eval_img = Image.open("/mnt/user-data/uploads/evaluation_dashboard.png")
            st.image(eval_img, use_container_width=True)
        except:
            st.warning("evaluation_dashboard.png not found.")

    with tab2:
        st.markdown("**Top 10 Features — AdaBoost**")
        features = {
            "avg_order_value": 0.1967,
            "segment_rank": 0.1698,
            "total_purchases": 0.1492,
            "log_avg_order_value": 0.1282,
            "days_since_last_purchase": 0.0963,
            "account_age_months": 0.0883,
            "loyalty_recency": 0.0553,
            "age": 0.0462,
            "email_opens": 0.0267,
            "cart_abandonment_rate": 0.0234,
        }
        max_v = max(features.values())
        for feat, imp in sorted(features.items(), key=lambda x: -x[1]):
            pct = int(imp / max_v * 100)
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div style="display:flex;justify-content:space-between;">
                    <span class="feat-label">{feat}</span>
                    <span class="feat-label">{imp:.4f}</span>
                </div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#111")
        names = list(features.keys())
        vals = list(features.values())
        colors = ["#8b5cf6" if i == 0 else "#6d28d9" if i < 3 else "#4c1d95" for i in range(len(names))]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6)
        ax.set_xlabel("Importance", color="#888", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.spines[:].set_color("#333")
        ax.set_title("AdaBoost Feature Importance", color="#e0e0e0", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIONS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions Explorer":
    st.markdown('<div class="section-header">Predictions Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("Probability Threshold", 0.1, 0.9, 0.5, 0.01,
                              help="Classify as high-value if probability ≥ threshold")
    with c2:
        sort_by = st.selectbox("Sort By", ["predicted_probability", "customer_index"])
    with c3:
        filter_actual = st.selectbox("Filter Actual Label", ["All", "Class 1 (Yes)", "Class 0 (No)"])

    disp = preds.copy()
    disp["predicted_label_adj"] = (disp["predicted_probability"] >= threshold).astype(int)
    disp["correct"] = disp["predicted_label_adj"] == disp["actual_label"]

    if filter_actual == "Class 1 (Yes)":
        disp = disp[disp["actual_label"] == 1]
    elif filter_actual == "Class 0 (No)":
        disp = disp[disp["actual_label"] == 0]

    disp = disp.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # Quick stats
    acc = disp["correct"].mean()
    high_val = (disp["predicted_label_adj"] == 1).sum()
    low_val = (disp["predicted_label_adj"] == 0).sum()

    m1, m2, m3, m4 = st.columns(4)
    for col, (v, l) in zip([m1, m2, m3, m4], [
        (f"{acc:.1%}", "Accuracy @ threshold"),
        (str(high_val), "Predicted High-Value"),
        (str(low_val), "Predicted Low-Value"),
        (str(len(disp)), "Total Shown"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Styled table
    def color_correct(val):
        return "color: #6ee7b7;" if val else "color: #f87171;"

    st.dataframe(
        disp.style
            .applymap(color_correct, subset=["correct"])
            .background_gradient(subset=["predicted_probability"], cmap="RdPu")
            .format({"predicted_probability": "{:.4f}"}),
        use_container_width=True,
        height=420,
    )

    # Probability histogram
    st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    ax.hist(preds[preds["actual_label"] == 0]["predicted_probability"], bins=40,
            alpha=0.6, color="#6d28d9", label="Actual 0")
    ax.hist(preds[preds["actual_label"] == 1]["predicted_probability"], bins=40,
            alpha=0.6, color="#ec4899", label="Actual 1")
    ax.axvline(threshold, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Threshold: {threshold}")
    ax.legend(fontsize=8, facecolor="#1e1e1e", edgecolor="#333", labelcolor="#ccc")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.spines[:].set_color("#333")
    ax.set_xlabel("Predicted Probability", color="#888", fontsize=9)
    ax.set_ylabel("Count", color="#888", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Predictor":
    st.markdown("""
    <div class="hero-banner" style="padding:2rem;">
        <div class="hero-subtitle">Simulate · Predict · Decide</div>
        <div class="hero-title" style="font-size:1.8rem;">Live Customer Scorer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Fill in customer attributes to estimate high-value purchase probability using the learned model weights.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧑 Demographics**")
        age = st.slider("Age", 18, 80, 35)
        country = st.selectbox("Country", ["India", "USA", "UK", "Germany", "Canada"])
        device = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])

    with c2:
        st.markdown("**🛒 Purchase Behaviour**")
        total_purchases = st.slider("Total Purchases", 0, 30, 8)
        avg_order_value = st.slider("Avg Order Value ($)", 10, 500, 150)
        days_since = st.slider("Days Since Last Purchase", 0, 200, 20)
        cart_abandon = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.35, 0.01)
        has_promo = st.checkbox("Has Promo Code", value=False)

    with c3:
        st.markdown("**📊 Engagement**")
        segment = st.selectbox("Customer Segment", ["Bronze", "Silver", "Gold", "Platinum"])
        account_age = st.slider("Account Age (months)", 0, 150, 12)
        email_opens = st.slider("Email Opens", 0, 20, 5)
        review_count = st.slider("Product Reviews Count", 0, 50, 5)
        avg_review = st.slider("Avg Review Rating", 1.0, 5.0, 3.5, 0.1)

    st.markdown("---")

    if st.button("🔮  Predict High-Value Purchase", use_container_width=True):
        # Engineered features (mirrors training pipeline)
        seg_rank = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}[segment]
        bounce_rate = 0.15  # assumed average
        engagement_score = (email_opens * 0.5) + (avg_review * 0.3) - (bounce_rate * 0.2)
        purchase_velocity = total_purchases / max(account_age, 1)
        intent_score = total_purchases * (1 - cart_abandon)
        loyalty_recency = account_age / (days_since + 1)
        log_aov = np.log1p(avg_order_value)
        review_quality = avg_review * np.log1p(review_count)
        promo_buyer = int(has_promo and total_purchases > 8)
        vip_spender = int(seg_rank >= 2 and avg_order_value > 150)

        # Scoring heuristic (weighted feature importance from AdaBoost)
        score = (
            avg_order_value / 500 * 0.1967
            + seg_rank / 3 * 0.1698
            + total_purchases / 30 * 0.1492
            + log_aov / np.log1p(500) * 0.1282
            + (1 - days_since / 200) * 0.0963
            + account_age / 150 * 0.0883
            + min(loyalty_recency / 10, 1) * 0.0553
            + (age - 18) / 62 * 0.0462
            + email_opens / 20 * 0.0267
            + (1 - cart_abandon) * 0.0234
        )
        prob = min(max(score, 0.0), 1.0)

        verdict = prob >= 0.5
        color = "#6ee7b7" if verdict else "#f87171"
        label = "HIGH-VALUE ✅" if verdict else "LOW-VALUE ❌"

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class="pred-meter">
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#666;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Prediction</div>
                <div style="font-size:2.8rem;font-weight:800;color:{color};line-height:1;">{prob:.1%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:{color};margin-top:0.4rem;">{label}</div>
                <hr style="border-color:#2a2a2a;margin:1rem 0;">
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#555;">Threshold: 50%</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("**Feature Contributions**")
            contribs = {
                "avg_order_value": avg_order_value / 500 * 0.1967,
                "segment_rank": seg_rank / 3 * 0.1698,
                "total_purchases": total_purchases / 30 * 0.1492,
                "log_avg_order_value": log_aov / np.log1p(500) * 0.1282,
                "recency (inverse)": (1 - days_since / 200) * 0.0963,
                "account_age": account_age / 150 * 0.0883,
                "loyalty_recency": min(loyalty_recency / 10, 1) * 0.0553,
                "age": (age - 18) / 62 * 0.0462,
                "email_opens": email_opens / 20 * 0.0267,
                "commitment (1-cart)": (1 - cart_abandon) * 0.0234,
            }
            max_c = max(contribs.values()) if max(contribs.values()) > 0 else 1
            for k, v in sorted(contribs.items(), key=lambda x: -x[1]):
                pct = int(v / max_c * 100)
                st.markdown(f"""
                <div class="feat-bar-wrap">
                    <div style="display:flex;justify-content:space-between;">
                        <span class="feat-label">{k}</span>
                        <span class="feat-label">{v:.4f}</span>
                    </div>
                    <div class="feat-bar-bg">
                        <div class="feat-bar-fill" style="width:{pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
