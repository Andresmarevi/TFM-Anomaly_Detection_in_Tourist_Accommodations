import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, Callable, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

# ----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anomaly Detection Dashboard", 
    layout="wide", 
    page_icon="🏨"
)

COLORS: Dict[str, str] = {
    "Normal": "#9CA3AF",         
    "IF": "#2563EB",             
    "LOF": "#F97316",            
    "Both": "#DC2626",           
    "Business rule": "#7C3AED",  
}

BUSINESS_RULES: List[str] = [
    "extreme_price_high", "extreme_price_low",
    "extreme_price_per_sqm", "excess_bathrooms", "tiny_villa", "huge_apartment",
]

BUSINESS_RULE_LABELS: Dict[str, str] = {
    "extreme_price_high": "High-price anomaly",
    "extreme_price_low": "Low-price bargain signal",
    "extreme_price_per_sqm": "Very low €/sqm signal",
    "excess_bathrooms": "Excessive bathrooms",
    "tiny_villa": "Tiny villa layout",
    "huge_apartment": "Oversized apartment layout",
}

BUSINESS_RULE_VALUE: Dict[str, str] = {
    "extreme_price_high": "bad",
    "extreme_price_low": "good",
    "extreme_price_per_sqm": "bad",
    "excess_bathrooms": "bad",
    "tiny_villa": "bad",
    "huge_apartment": "good",
}

PRICE_ANOMALY_INDICATORS: List[str] = [
    "extreme_price_high", "extreme_price_low", "extreme_price_per_sqm",
]

# Paths
BASE_DIR = Path("..").resolve() if (Path("..") / "data").exists() else Path(".").resolve()
DEFAULT_DATA_PATH = BASE_DIR / "data" / "results" / "model_results_full.csv"
TEMPORAL_HISTORY_PATH = BASE_DIR / "data" / "results" / "history"
TEMPORAL_SUMMARY_PATH = BASE_DIR / "data" / "results" / "temporal_evolution.csv"


def latest_data_signature(path: Path = DEFAULT_DATA_PATH) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def temporal_history_signature(path: Path = TEMPORAL_HISTORY_PATH) -> str:
    files = sorted(path.glob("model_results_full_*.csv"))
    return "|".join(
        f"{file.name}:{file.stat().st_size}:{file.stat().st_mtime_ns}"
        for file in files
    )


def temporal_summary_signature(path: Path = TEMPORAL_SUMMARY_PATH) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def select_weekly_history_files(path: Path) -> List[Path]:
    """Keep the latest historical file for each ISO week and horizon."""
    selected = {}
    pattern = re.compile(
        r"model_results_full_(?P<date>\d{4}-\d{2}-\d{2})_(?P<period>1m|3m)\.csv"
    )
    for file in path.glob("model_results_full_*.csv"):
        match = pattern.fullmatch(file.name)
        if not match:
            continue
        extraction_date = date.fromisoformat(match.group("date"))
        period = match.group("period")
        week_key = (*extraction_date.isocalendar()[:2], period)
        current = selected.get(week_key)
        if current is None or extraction_date > current[0]:
            selected[week_key] = (extraction_date, file)
    return [item[1] for item in sorted(selected.values(), key=lambda item: item[0])]

# ----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(
    path: Path = DEFAULT_DATA_PATH,
    cache_signature: str = "",
) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:  
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path)

    for col in ["if_anomaly", "lof_anomaly"]:
        if col not in df.columns:
            df[col] = 1

    # Transform scikit-learn convention (-1=anomaly, 1=normal)
    df["if_anomaly"] = (df["if_anomaly"] == -1).astype(int)
    df["lof_anomaly"] = (df["lof_anomaly"] == -1).astype(int)

    conditions = [
        (df["if_anomaly"] == 1) & (df["lof_anomaly"] == 1),
        (df["if_anomaly"] == 1) & (df["lof_anomaly"] == 0),
        (df["if_anomaly"] == 0) & (df["lof_anomaly"] == 1),
    ]
    df["status"] = np.select(conditions, ["Both", "IF", "LOF"], default="Normal")

    # Consensus Percentile Ranking
    if "if_score" in df.columns:
        df["if_rank_pct"] = df["if_score"].rank(pct=True)
    if "lof_score" in df.columns:
        df["lof_rank_pct"] = df["lof_score"].rank(pct=True)
    if "if_rank_pct" in df.columns and "lof_rank_pct" in df.columns:
        df["combined_rank"] = (df["if_rank_pct"] + df["lof_rank_pct"]) / 2

    available_rules = [c for c in PRICE_ANOMALY_INDICATORS if c in df.columns]
    rule_signal = df[available_rules].fillna(0).astype(bool).any(axis=1) if available_rules else pd.Series(False, index=df.index)

    for relative_col in ["price_relative_to_type", "price_relative_to_province"]:
        if relative_col in df.columns:
            rule_signal |= (df[relative_col].abs() > df[relative_col].abs().quantile(0.95))
            
    df["price_related_anomaly"] = rule_signal

    # Global business rule flag for heuristic validation
    present_rules = [c for c in BUSINESS_RULES if c in df.columns]
    if "any_flag" not in df.columns and present_rules:
        df["any_flag"] = df[present_rules].fillna(0).astype(bool).any(axis=1)

    # Reconstrucción de variables categóricas (One-Hot Reverse)
    for prefix in ["region", "property_type", "province"]:
        if prefix not in df.columns:
            dummy_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
            if dummy_cols:
                df[prefix] = df[dummy_cols].idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)

    return df

def filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Global Market Filters ⚙️")
    
    regions = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()) if "region" in df else [])
    ptypes = st.sidebar.multiselect("Property type", sorted(df["property_type"].dropna().unique()) if "property_type" in df else [])
    
    st.sidebar.divider()
    model = st.sidebar.selectbox("Model Status", ["All", "Both (Consensus)", "IF Only", "LOF Only", "Normal"])
    price_only = st.sidebar.toggle("Require Contextual Price Anomaly", value=False, help="Filters listings flagged by business rules or extreme relative prices (>95th percentile).")

    price_range = None
    if "price" in df.columns:
        pmin, pmax = float(df["price"].min()), float(df["price"].max())
        price_range = st.sidebar.slider("Price range (€)", pmin, pmax, (pmin, pmax))

    sqm_range = None
    if "sqm" in df.columns:
        smin, smax = float(df["sqm"].min()), float(df["sqm"].max())
        sqm_range = st.sidebar.slider("Surface range (m²)", smin, smax, (smin, smax))

    # Apply filters
    out = df.copy()
    if regions:
        out = out[out["region"].isin(regions)]
    if ptypes:
        out = out[out["property_type"].isin(ptypes)]
        
    if model == "Both (Consensus)": out = out[out["status"] == "Both"]
    elif model == "IF Only": out = out[out["status"] == "IF"]
    elif model == "LOF Only": out = out[out["status"] == "LOF"]
    elif model == "Normal": out = out[out["status"] == "Normal"]
        
    if price_only:
        out = out[out["price_related_anomaly"]]
    if price_range:
        out = out[(out["price"] >= price_range[0]) & (out["price"] <= price_range[1])]
    if sqm_range:
        out = out[(out["sqm"] >= sqm_range[0]) & (out["sqm"] <= sqm_range[1])]

    return out

# ----------------------------------------------------------------------------
# PAGE FUNCTIONS
# ----------------------------------------------------------------------------
def page_overview(df: pd.DataFrame) -> None:
    st.title("Pipeline Overview 📊")
    st.markdown("Unsupervised anomaly detection results across the Spanish tourist accommodation market. "
                "Models were trained using robust scaling and contextual feature engineering to mitigate geographical bias.")

    if df.empty:
        st.info("No data matches the current filters.")
        return

    total = len(df)
    if_n = int(df["if_anomaly"].sum())
    lof_n = int(df["lof_anomaly"].sum())
    both_n = int((df["status"] == "Both").sum())
    
    jaccard = both_n / (if_n + lof_n - both_n) if (if_n + lof_n - both_n) else 0
    spearman = spearmanr(df["if_score"], df["lof_score"])[0] if total > 1 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total listings analyzed", f"{total:,}")
    c2.metric("Isolation Forest (Global)", f"{if_n:,}", f"{if_n/total:.1%}" if total else "0%")
    c3.metric("LOF (Local Density)", f"{lof_n:,}", f"{lof_n/total:.1%}" if total else "0%")
    c4.metric("Consensus Anomalies", f"{both_n:,}", help="Listings flagged simultaneously by both algorithms.")

    c5, c6 = st.columns(2)
    c5.metric("Jaccard Similarity Index", f"{jaccard:.4f}", help="Measures the overlap between the two models.")
    c6.metric("Spearman Rank Correlation", f"{spearman:.4f}" if pd.notna(spearman) else "N/A", help="Monotonic relationship between IF and LOF scores.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        counts = df["status"].value_counts().reindex(["Normal", "IF", "LOF", "Both"]).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map=COLORS, labels={"x": "Algorithm", "y": "Number of Listings"},
                     title="Distribution of Anomalous Listings")
        st.plotly_chart(fig, width='stretch')
    with col2:
        venn_data = pd.DataFrame({
            "Set": ["IF only", "LOF only", "Both"],
            "Count": [if_n - both_n, lof_n - both_n, both_n],
        })
        fig = px.pie(venn_data, names="Set", values="Count", hole=0.45,
                     color="Set", color_discrete_map={"IF only": COLORS["IF"], "LOF only": COLORS["LOF"], "Both": COLORS["Both"]},
                     title="Algorithmic Intersection (Venn Representation)")
        st.plotly_chart(fig, width='stretch')


def page_price_anomalies(df: pd.DataFrame) -> None:
    st.title("Contextual Price Anomalies 💶")
    st.caption("Validating the engineered features: True market anomalies depend on property type and geography, not just absolute price.")

    if df.empty:
        st.warning("No data available for visualization.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        if "price" in df.columns:
            fig_price = px.box(df, x="status", y="price", color="status", 
                               color_discrete_map=COLORS,
                               title="Distribution of Absolute Price")
            st.plotly_chart(fig_price, width='stretch')
            
    with col2:
        if "price_per_sqm" in df.columns:
            fig_sqm = px.box(df, x="status", y="price_per_sqm", color="status", 
                             color_discrete_map=COLORS,
                             title="Distribution of Price per Sqm")
            st.plotly_chart(fig_sqm, width='stretch')

    st.divider()

    col3, col4 = st.columns(2)
    
    with col3:
        if "price_relative_to_type" in df.columns:
            fig_type = px.violin(df, x="status", y="price_relative_to_type", color="status",
                                 color_discrete_map=COLORS, box=True, 
                                 title="Market Deviation (by Property Type)")
            st.plotly_chart(fig_type, width='stretch')
            
    with col4:
        if "price_relative_to_province" in df.columns:
            fig_prov = px.violin(df, x="status", y="price_relative_to_province", color="status",
                                 color_discrete_map=COLORS, box=True, 
                                 title="Market Deviation (by Province)")
            st.plotly_chart(fig_prov, width='stretch')

def page_anomaly_explorer(df: pd.DataFrame) -> None:
    st.title("Market anomaly explorer 🕵️")
    st.markdown("This panel highlights the listings with the strongest market deviation relative to their property type and province. The objective is to interpret the anomaly signal, not to sell a property as a commercial opportunity.")

    if "price" not in df.columns:
        st.warning("No price data is available to compare market deviation.")
        return

    df_anom = df[df["status"] != "Normal"].copy()
    if df_anom.empty:
        st.info("No hay anomalías activas con los filtros actuales.")
        return

    if "price_per_sqm" not in df_anom.columns:
        df_anom["price_per_sqm"] = np.where(
            df_anom["sqm"].replace(0, np.nan).notna(),
            df_anom["price"] / df_anom["sqm"].replace(0, np.nan),
            np.nan,
        )

    median_price = df["price"].median()
    p10_price = df["price"].quantile(0.10)
    p90_price = df["price"].quantile(0.90)
    median_price_per_sqm = df["price_per_sqm"].median() if "price_per_sqm" in df.columns else df["price"].median()

    cheap = df_anom[df_anom["price"] <= p10_price].copy()
    premium = df_anom[df_anom["price"] >= p90_price].copy()

    if cheap.empty:
        cheap = df_anom.nsmallest(12, "price").copy()
    if premium.empty:
        premium = df_anom.nlargest(12, "price").copy()

    def build_value_score(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
        frame = frame.copy()
        if "sqm" in frame.columns:
            frame["sqm_score"] = frame["sqm"].fillna(0).rank(pct=True)
        else:
            frame["sqm_score"] = 0.5
        if "num_rooms" in frame.columns:
            frame["room_score"] = frame["num_rooms"].fillna(0).rank(pct=True)
        else:
            frame["room_score"] = 0.5
        if "price_per_sqm" in frame.columns:
            price_per_sqm_rank = frame["price_per_sqm"].rank(pct=True, ascending=True)
        else:
            price_per_sqm_rank = pd.Series(0.5, index=frame.index)

        if mode == "deal":
            frame["value_score"] = (
                45 * (1 - frame["price"].rank(pct=True)) +
                30 * frame["sqm_score"] +
                15 * frame["room_score"] +
                10 * price_per_sqm_rank
            )
        else:
            frame["value_score"] = (
                40 * frame["price"].rank(pct=True) +
                30 * frame["sqm_score"] +
                20 * frame["room_score"] +
                10 * price_per_sqm_rank
            )
        return frame.sort_values("value_score", ascending=False)

    cheap = build_value_score(cheap, "deal").head(6)
    premium = build_value_score(premium, "premium").head(6)

    st.subheader("Low-price anomaly candidates 🟢")
    st.caption("Listings whose price is materially below the local benchmark and which may represent a genuine undervaluation or a strong low-price outlier.")

    if cheap.empty:
        st.info("No low-price anomaly candidates match the current filters.")
    else:
        deal_cols = st.columns(3)
        for idx, row in cheap.reset_index(drop=True).iterrows():
            with deal_cols[idx % 3]:
                price_gap = ((median_price - row["price"]) / median_price) * 100 if pd.notna(median_price) else 0
                pps = row.get("price_per_sqm", np.nan)
                room_label = row.get("num_rooms", "-")
                sqm_label = row.get("sqm", "-")
                title = row.get("name", "Apartamento")
                region = row.get("region", "-")
                property_type = row.get("property_type", "-")
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, rgba(6,95,70,0.96), rgba(22,163,74,0.82)); border:1px solid rgba(110,231,183,0.8); border-radius:16px; padding:1rem; margin-bottom:1rem; min-height:270px; box-shadow:0 10px 22px rgba(6,95,70,0.22);">
                        <div style="font-size:0.78rem; color:#d1fae5; font-weight:800; letter-spacing:0.05em; text-transform:uppercase;">REAL DEAL</div>
                        <div style="font-size:1.15rem; font-weight:800; margin:0.5rem 0; color:#f0fdf4; line-height:1.25;">{title[:52]}</div>
                        <div style="font-size:0.88rem; color:#dcfce7; margin-bottom:0.5rem;">{property_type} · {region}</div>
                        <div style="font-size:2rem; font-weight:900; color:#ffffff; margin-top:0.25rem;">€{row['price']:,.0f}</div>
                        <div style="margin-top:0.7rem; color:#ecfdf5; font-size:0.92rem; font-weight:600;">{sqm_label} m² · {room_label} hab · {pps:,.0f} €/m²</div>
                        <div style="margin-top:0.9rem; background:rgba(16,185,129,0.18); border:1px solid rgba(167,243,208,0.45); border-radius:12px; padding:0.6rem; font-size:1.2rem; color:#f0fdf4; line-height:1.4;">
                            <strong> {price_gap:,.0f}% below the market ! </strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.subheader("Premium opportunities 🔴")
    st.caption("Very expensive properties that are not worthy.")

    if premium.empty:
        st.info("No premium opportunities make sense under the current filters.")
    else:
        prem_cols = st.columns(3)
        for idx, row in premium.reset_index(drop=True).iterrows():
            with prem_cols[idx % 3]:
                price_gap = ((row["price"] - median_price) / median_price) * 100 if pd.notna(median_price) else 0
                pps = row.get("price_per_sqm", np.nan)
                room_label = row.get("num_rooms", "-")
                sqm_label = row.get("sqm", "-")
                title = row.get("name", "Apartamento premium")
                region = row.get("region", "-")
                property_type = row.get("property_type", "-")
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, rgba(120,53,15,0.96), rgba(249,115,22,0.82)); border:1px solid rgba(253,186,116,0.8); border-radius:16px; padding:1rem; margin-bottom:1rem; min-height:270px; box-shadow:0 10px 22px rgba(120,53,15,0.25);">
                        <div style="font-size:0.78rem; color:#ffedd5; font-weight:800; letter-spacing:0.05em; text-transform:uppercase;">PREMIUM</div>
                        <div style="font-size:1.15rem; font-weight:800; margin:0.5rem 0; color:#fff7ed; line-height:1.25;">{title[:52]}</div>
                        <div style="font-size:0.88rem; color:#fed7aa; margin-bottom:0.5rem;">{property_type} · {region}</div>
                        <div style="font-size:2rem; font-weight:900; color:#ffffff; margin-top:0.25rem;">€{row['price']:,.0f}</div>
                        <div style="margin-top:0.7rem; color:#fff7ed; font-size:0.92rem; font-weight:600;">{sqm_label} m² · {room_label} hab · {pps:,.0f} €/m²</div>
                        <div style="margin-top:0.9rem; background:rgba(249,115,22,0.15); border:1px solid rgba(253,186,116,0.45); border-radius:12px; padding:0.6rem; font-size:1.2rem; color:#fff7ed; line-height:1.4;">
                        <strong>{price_gap:,.0f}% above the median </strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def page_segment_analysis(df: pd.DataFrame) -> None:
    st.title("Market Segment Analysis 📈")
    st.caption("Identifies which geographical areas or property types are most susceptible to market anomalies.")

    seg_options = [c for c in ["property_type", "region", "province"] if c in df.columns]
    if not seg_options:
        st.warning("No categorical segments available.")
        return

    seg = st.selectbox("Segment data by:", seg_options)

    g = df.groupby(seg).agg(
        n=("status", "size"),
        if_anomalies=("if_anomaly", "sum"),
        lof_anomalies=("lof_anomaly", "sum"),
        consensus_anomalies=("status", lambda s: (s == "Both").sum()),
        if_rate=("if_anomaly", "mean"),
        lof_rate=("lof_anomaly", "mean"),
        consensus_rate=("status", lambda s: (s == "Both").mean()),
    ).reset_index().sort_values("n", ascending=False)

    min_n = st.slider("Minimum observations required per segment", 1, int(g["n"].max()) if len(g) else 1, 10)
    g = g[g["n"] >= min_n]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anomaly Counts")
        count_cols = [seg, "n", "if_anomalies", "lof_anomalies", "consensus_anomalies"]
        st.dataframe(g[[c for c in count_cols if c in g.columns]], width='stretch')
    
    with col2:
        st.subheader("Anomaly Rates (%)")
        rate_cols = [seg, "if_rate", "lof_rate", "consensus_rate"]
        st.dataframe(g[[c for c in rate_cols if c in g.columns]].style.format({"if_rate": "{:.2%}", "lof_rate": "{:.2%}", "consensus_rate": "{:.2%}"}), width='stretch')

    fig = px.bar(g.melt(id_vars=[seg, "n"], value_vars=["if_anomalies", "lof_anomalies", "consensus_anomalies"]),
                 x=seg, y="value", color="variable", barmode="group",
                 labels={"value": "Number of Anomalies", "variable": "Detection Model"},
                 title=f"Anomaly Count by {seg.capitalize()}")
    st.plotly_chart(fig, width='stretch')

def page_business_rules(df: pd.DataFrame) -> None:
    st.title("Business Rules Validation 📏")
    st.info("In unsupervised learning without ground truth labels, checking model overlap against domain heuristic rules guarantees that the models are correct in identifying extreme behaviors")

    rows = []
    total = len(df)
    for rule in BUSINESS_RULES:
        if rule not in df.columns:
            continue
        sub = df[df[rule].fillna(0).astype(bool)]
        n = len(sub)
        if_det = int(sub["if_anomaly"].sum())
        lof_det = int(sub["lof_anomaly"].sum())
        both_det = int(((sub["if_anomaly"] == 1) & (sub["lof_anomaly"] == 1)).sum())
        
        rows.append({
            "Heuristic Rule": rule.replace("_", " ").capitalize(), 
            "Listings Flagged": n, 
            "% of Filtered Data": n / total if total else 0,
            "IF Detection": if_det / n if n else 0, 
            "LOF Detection": lof_det / n if n else 0,
            "Consensus Detection": both_det / n if n else 0,
        })
        
    rules_df = pd.DataFrame(rows)
    if rules_df.empty:
        st.warning("No business rules available for validation.")
        return

    st.dataframe(rules_df.style.format({"% of Filtered Data": "{:.2%}", "IF Detection": "{:.1%}",
                                         "LOF Detection": "{:.1%}", "Consensus Detection": "{:.1%}"}),
                 width='stretch')

    fig = px.bar(rules_df.melt(id_vars="Heuristic Rule", value_vars=["IF Detection", "LOF Detection", "Consensus Detection"]),
                 x="Heuristic Rule", y="value", color="variable", barmode="group",
                 labels={"value": "Overlap Rate (Recall against Rule)", "variable": "Model Strategy"},
                 title="How often do models agree with simple business logic?")
    st.plotly_chart(fig, width='stretch')

def page_consensus_ranking(df: pd.DataFrame) -> None:
    st.title("Final Consensus Ranking 🏆")
    st.markdown("Displays the most anomalous listings based on the unified percentile rank methodology designed in the thesis. This ensures the output is not biased by the differing mathematical scales of IF and LOF.")

    top_n = st.slider("Number of top listings to display", 10, 100, 20)
    group = st.radio("Target Strategy", ["Strict Consensus", "IF-centric", "LOF-centric"], horizontal=True)

    if group == "Strict Consensus":
        sub = df[df["status"] == "Both"].sort_values("combined_rank", ascending=False)
    elif group == "IF-centric":
        sub = df[df["status"] == "IF"].sort_values("if_score", ascending=False)
    else:
        sub = df[df["status"] == "LOF"].sort_values("lof_score", ascending=False)

    cols = [c for c in ["name", "property_type", "province", "price", "if_score", "lof_score", "combined_rank", "status"] if c in sub.columns]
    st.dataframe(sub[cols].head(top_n), width='stretch', height=500)
    
    st.divider()
    st.subheader("Most Anomalous Listings Explained 💡")
    if not sub.empty:
        def build_anomaly_reason(row: pd.Series) -> str:
            reasons = []

            for rule in BUSINESS_RULES:
                if rule in row.index and pd.notna(row[rule]) and bool(row[rule]):
                    reasons.append(BUSINESS_RULE_LABELS.get(rule, rule.replace("_", " ").capitalize()))

            for column, label in [
                ("price_relative_to_type", "its property type"),
                ("price_relative_to_province", "its province"),
            ]:
                if column not in row.index or pd.isna(row[column]):
                    continue
                relative_price = float(row[column])
                if abs(relative_price) >= 0.15:
                    direction = "above" if relative_price > 0 else "below"
                    reasons.append(
                        f"the price is {abs(relative_price):.1%} {direction} the typical level for {label}"
                    )

            if reasons:
                if len(reasons) == 1:
                    return f"💡 Why anomalous: {reasons[0]}."
                return f"💡 Why anomalous: {', '.join(reasons[:-1])} and {reasons[-1]}."

            detected_by = {
                "Both": "both Isolation Forest and LOF",
                "IF": "Isolation Forest",
                "LOF": "LOF",
            }.get(row.get("status"), "the anomaly detection models")
            return f"💡 Why anomalous: flagged by {detected_by}; no additional business-rule explanation is available."

        top_3 = sub.head(3)
        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            with st.expander(f"#{idx}: {row.get('name', 'Unknown')} - 🚨 Anomaly Score: {row.get('combined_rank', 0):.1%}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Property Type:** {row.get('property_type', '—')}")
                    st.markdown(f"**Province:** {row.get('province', '—')}")
                    st.markdown(f"**Price:** €{row.get('price', '—')}")
                with col2:
                    st.markdown(f"**IF Score:** {row.get('if_score', '—')} (Global anomaly)")
                    st.markdown(f"**LOF Score:** {row.get('lof_score', '—')} (Local density anomaly)")
                    st.markdown(f"**Detection Status:** {row.get('status', '—')}")
                st.info(build_anomaly_reason(row))

def page_optimized_listing_view(df: pd.DataFrame) -> None:
    """Optimized listing search"""
    st.title("Listing Lookup 🔍")
    st.caption("Search for apartment details by name.")
    
    if df.empty:
        st.warning("No data available.")
        return
    
    search_term = st.text_input("Search by listing name (partial match supported):", placeholder="e.g., 'Apartment', 'Villa'")
    
    if search_term:
        matching = df[df["name"].astype(str).str.contains(search_term, case=False, na=False)]
    else:
        matching = df.copy()
    
    if matching.empty:
        st.warning(f"No listings found matching '{search_term}'")
        return
    
    st.info(f"Found **{len(matching)}** listing(s)")
    
    selection_positions = list(range(min(len(matching), 100)))
    selected_position = st.selectbox(
        "Select a listing:",
        selection_positions,
        format_func=lambda position: (
            f"{matching.iloc[position]['name']} (record {position + 1})"
        ),
        key="opt_listing_select",
    )

    row = matching.iloc[selected_position]
    
    st.divider()
    
    # Status indicator
    status = row.get("status", "Normal")
    status_labels = {
        "Both": "Detected by Consensus (IF & LOF) 🚨",
        "IF": "Detected Globally (IF Only) ⚠️",
        "LOF": "Detected Locally (LOF Only) ⚠️",
        "Normal": "Market Standard (Normal) ✅"
    }
    st.subheader(status_labels.get(status, status))


    anomaly_detected = status != "Normal"
    if anomaly_detected:
        if status == "Both":
            verdict = "This apartment is considered anomalous by both models, so it stands out strongly from normal market behavior."
        elif status == "IF":
            verdict = "This apartment is flagged by the global model, which suggests it is unusually expensive or structurally different from the broader market."
        else:
            verdict = "This apartment is flagged by the local model, which means it deviates from the local neighborhood pattern rather than from the overall market."
        verdict_color = "🔴"
    else:
        verdict = "This apartment does not show anomaly signals from the trained models and is within expected market patterns."
        verdict_color = "✅"

    price_val = row.get("price")
    rel_type = row.get("price_relative_to_type")
    rel_prov = row.get("price_relative_to_province")

    if pd.notna(price_val):
        if pd.notna(rel_type):
            if rel_type > 0.15:
                price_summary = f"This listing is {rel_type:,.1%} above the usual price for this type, which is a clear premium signal."
            elif rel_type < -0.15:
                price_summary = f"This listing is {abs(rel_type):,.1%} below the usual price for this type, so it looks like a strong bargain."
            else:
                price_summary = f"Its price is close to the typical level for this property type, although the anomaly models still consider the full context."
        elif pd.notna(rel_prov):
            if rel_prov > 0.15:
                price_summary = f"This listing is {rel_prov:,.1%} above the regional normal benchmark."
            elif rel_prov < -0.15:
                price_summary = f"This listing is {abs(rel_prov):,.1%} below the regional normal benchmark."
            else:
                price_summary = "Its price is close to the local market benchmark."
        else:
            price_summary = "Price is within the analyzed market context, but the anomaly score is still evaluated jointly with the other features."
    else:
        price_summary = "Price information is not available for this listing."

    st.markdown(f"{verdict_color} **Market verdict:** {verdict}")
    st.markdown(f"💶 **Price vs normal:** {price_summary}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Property Info 🏠")
        st.markdown(f"**Name:** {row.get('name', '—')}")
        st.markdown(f"**Type:** {row.get('property_type', '—')}")
        st.markdown(f"**Rooms:** {row.get('num_rooms', '—')}")
        st.markdown(f"**Bathrooms:** {row.get('num_bathrooms', '—')}")
    
    with col2:
        st.markdown("### Location 📍")
        st.markdown(f"**Province:** {row.get('province', '—')}")
        st.markdown(f"**Region:** {row.get('region', '—')}")
        st.markdown(f"**Size:** {row.get('sqm', '—')} m²")
    
    with col3:
        st.markdown("### Pricing 💶")
        st.markdown(f"**Price (7 nights):** €{row.get('price', '—')}")
        if pd.notna(row.get("price_per_sqm")):
            st.markdown(f"**€/m²:** €{row.get('price_per_sqm', '—'):.2f}")
        if pd.notna(row.get("price_relative_to_type")):
            st.markdown(f"**vs Type Avg:** {row.get('price_relative_to_type', 0):+.1%}")
        if pd.notna(row.get("price_relative_to_province")):
            st.markdown(f"**vs Region Avg:** {row.get('price_relative_to_province', 0):+.1%}")
    
    st.divider()
    
    # Anomaly Scores
    st.subheader("Algorithmic Analysis 🎯")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric(
        "Isolation Forest",
        f"{row.get('if_score', np.nan):.4f}" if pd.notna(row.get('if_score')) else "N/A",
        delta="ANOMALY" if row.get("if_anomaly") == 1 else "NORMAL"
    )
    score_col2.metric(
        "Local Outlier Factor",
        f"{row.get('lof_score', np.nan):.4f}" if pd.notna(row.get('lof_score')) else "N/A",
        delta="ANOMALY" if row.get("lof_anomaly") == 1 else "NORMAL"
    )
    score_col3.metric(
        "Combined Rank",
        f"{row.get('combined_rank', np.nan)*100:.1f}%" if pd.notna(row.get('combined_rank')) else "N/A",
        help="Percentile ranking across all listings"
    )
    
    # Business Rules
    st.subheader("Business Rules Check 📋")
    positive_rules = {"extreme_price_low"}
    triggered_rules = []
    for rule in BUSINESS_RULES:
        if rule in row.index and bool(row[rule]):
            label = rule.replace("_", " ").capitalize()
            if rule in positive_rules:
                triggered_rules.append((label, "good"))
            else:
                triggered_rules.append((label, "bad"))

    if triggered_rules:
        for label, status in triggered_rules:
            if status == "good":
                st.success(f"🟢 {label}")
            else:
                st.error(f"🔴 {label}")
    else:
        st.success("✓ No business rules triggered")

@st.cache_data(show_spinner=False, persist="disk")
def load_temporal_history(
    path: Path = TEMPORAL_HISTORY_PATH,
    cache_signature: str = "",
    requested_columns: tuple = (),
) -> pd.DataFrame:
    files = select_weekly_history_files(path)
    if not files:
        return pd.DataFrame()

    required_columns = [
        "id", "name", "property_type", "region", "province", "price",
        "price_per_sqm", "if_anomaly", "lof_anomaly", "any_flag",
        "extraction_date", "lookahead_period", "checkin_date", "checkout_date",
    ]
    if requested_columns:
        required_columns = [column for column in required_columns if column in requested_columns]
    if "extraction_date" not in required_columns:
        required_columns.append("extraction_date")
    frames = []
    for file in files:
        available = pd.read_csv(file, sep=";", nrows=0).columns.tolist()
        usecols = [column for column in required_columns if column in available]
        frames.append(pd.read_csv(file, sep=";", usecols=usecols))
    history = pd.concat(frames, ignore_index=True)
    history["extraction_date"] = pd.to_datetime(
        history["extraction_date"], errors="coerce"
    )
    return history.dropna(subset=["extraction_date"])


@st.cache_data(show_spinner=False, persist="disk")
def load_temporal_summary(
    path: Path = TEMPORAL_SUMMARY_PATH,
    cache_signature: str = "",
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(path, sep=";")
    summary["extraction_date"] = pd.to_datetime(
        summary["extraction_date"], errors="coerce"
    )
    return summary.dropna(subset=["extraction_date"])


def page_temporal_evolution(df: pd.DataFrame) -> None:
    st.title("Temporal Market Intelligence 📅")
    st.markdown(
        "A weekly view of price dynamics, booking-horizon differences, market coverage "
        "and persistent anomaly signals across the Spanish accommodation market."
    )

    analysis_view = st.radio(
        "Analysis view",
        [
            "Market pulse (fast)",
            "1m vs 3m comparison",
            "Persistent anomalies",
            "Segments",
            "Coverage & quality",
        ],
        horizontal=True,
        key="temporal_analysis_view",
    )

    if analysis_view == "Market pulse (fast)":
        summary = load_temporal_summary(
            cache_signature=temporal_summary_signature()
        )
        if summary.empty:
            st.warning(
                "The temporal summary is not available yet. Run the pipeline first."
            )
            return

        periods = sorted(summary["lookahead_period"].dropna().unique())
        selected_period = st.selectbox(
            "Booking horizon", ["All"] + periods, key="fast_temporal_period"
        )
        if selected_period != "All":
            summary = summary[summary["lookahead_period"] == selected_period]
        summary = summary.sort_values("extraction_date")
        latest_date = summary["extraction_date"].max()
        latest = summary.iloc[-1]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Latest listings", f"{int(latest['total_listings']):,}")
        metric_col2.metric("Median price", f"€{latest['median_price']:,.0f}")
        metric_col3.metric("Consensus anomalies", f"{latest['consensus_anomaly_rate']:.1%}")
        metric_col4.metric("Business-rule signals", f"{latest['business_rule_rate']:.1%}")

        price_fig = px.line(
            summary, x="extraction_date", y="median_price",
            color="lookahead_period", markers=True,
            title="Median observed price by extraction date",
            labels={"extraction_date": "Extraction date", "median_price": "Median price (EUR)"},
        )
        st.plotly_chart(price_fig, width="stretch")

        rate_columns = {
            "if_anomaly_rate": "Isolation Forest",
            "lof_anomaly_rate": "LOF",
            "consensus_anomaly_rate": "Consensus",
            "business_rule_rate": "Business rules",
        }
        rate_data = summary.melt(
            id_vars=["extraction_date", "lookahead_period"],
            value_vars=list(rate_columns), var_name="indicator", value_name="rate",
        )
        rate_data["indicator"] = rate_data["indicator"].map(rate_columns)
        rate_fig = px.line(
            rate_data, x="extraction_date", y="rate", color="indicator",
            line_dash="lookahead_period", markers=True,
            title="Evolution of anomaly and rule rates",
            labels={"extraction_date": "Extraction date", "rate": "Rate"},
        )
        rate_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(rate_fig, width="stretch")

        historical = summary[summary["extraction_date"] < latest_date]
        if not historical.empty:
            baseline = historical.groupby("lookahead_period").agg(
                historical_median_price=("median_price", "median"),
                historical_consensus_rate=("consensus_anomaly_rate", "median"),
            ).reset_index()
            current = summary[summary["extraction_date"] == latest_date][
                ["lookahead_period", "median_price", "consensus_anomaly_rate"]
            ].rename(columns={
                "median_price": "current_median_price",
                "consensus_anomaly_rate": "current_consensus_rate",
            })
            baseline_view = current.merge(baseline, on="lookahead_period", how="left")
            baseline_view["price_vs_baseline_pct"] = (
                baseline_view["current_median_price"]
                / baseline_view["historical_median_price"] - 1
            )
            st.subheader("Current snapshot versus historical baseline")
            st.dataframe(baseline_view, width="stretch", hide_index=True)
        st.caption(
            "This fast view uses the aggregated temporal summary. Select another analysis "
            "view only when record-level historical data is needed."
        )
        return

    detailed_view = st.selectbox(
        "Detailed analysis",
        ["1m vs 3m", "Persistent anomalies", "Segments", "Coverage & quality"],
        key="temporal_detailed_view",
    )

    if detailed_view == "Coverage & quality":
        summary = load_temporal_summary(
            cache_signature=temporal_summary_signature()
        )
        if summary.empty:
            st.warning("The temporal summary is not available yet.")
            return
        coverage = summary.groupby(
            ["extraction_date", "lookahead_period"], as_index=False
        ).agg(
            listings=("total_listings", "first"),
            median_price=("median_price", "first"),
        )
        coverage_fig = px.line(
            coverage, x="extraction_date", y="listings", color="lookahead_period",
            markers=True, title="Snapshot coverage by extraction date",
        )
        st.plotly_chart(coverage_fig, width="stretch")
        st.dataframe(coverage, width="stretch", hide_index=True)
        st.caption(
            "Coverage changes must be considered before interpreting price or anomaly "
            "changes as genuine market evolution."
        )
        return

    requested_columns = {
        "1m vs 3m": {
            "id", "price", "extraction_date", "lookahead_period",
            "checkin_date", "checkout_date", "region", "property_type",
            "price_per_sqm", "if_anomaly", "lof_anomaly", "any_flag",
        },
        "Persistent anomalies": {
            "id", "name", "property_type", "region", "price",
            "if_anomaly", "lof_anomaly", "extraction_date", "lookahead_period",
            "price_per_sqm", "any_flag",
        },
        "Segments": {
            "id", "price", "if_anomaly", "lof_anomaly", "extraction_date",
            "lookahead_period", "region", "property_type", "province",
            "price_per_sqm", "any_flag",
        },
    }[detailed_view]

    history = load_temporal_history(
        cache_signature=temporal_history_signature(),
        requested_columns=tuple(sorted(requested_columns)),
    )
    if history.empty:
        st.warning("No historical snapshots are available yet.")
        return

    history = history.copy()
    history["extraction_date"] = pd.to_datetime(history["extraction_date"], errors="coerce")
    for date_column in ["checkin_date", "checkout_date"]:
        if date_column in history:
            history[date_column] = pd.to_datetime(history[date_column], errors="coerce")
    history = history.dropna(subset=["extraction_date"])

    periods = sorted(history["lookahead_period"].dropna().unique())
    selected_period = st.selectbox("Booking horizon", ["All"] + periods, key="temporal_period")
    if selected_period != "All":
        history = history[history["lookahead_period"] == selected_period]

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        regions = sorted(history["region"].dropna().unique()) if "region" in history else []
        selected_regions = st.multiselect("Regions", regions, key="temporal_regions")
    with filter_col2:
        property_types = sorted(history["property_type"].dropna().unique()) if "property_type" in history else []
        selected_property_types = st.multiselect("Property types", property_types, key="temporal_types")
    with filter_col3:
        dates = history["extraction_date"].dropna()
        date_range = st.date_input(
            "Extraction window",
            value=(dates.min().date(), dates.max().date()),
            min_value=dates.min().date(),
            max_value=dates.max().date(),
            key="temporal_dates",
        )

    if selected_regions:
        history = history[history["region"].isin(selected_regions)]
    if selected_property_types:
        history = history[history["property_type"].isin(selected_property_types)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        history = history[
            history["extraction_date"].dt.date.between(date_range[0], date_range[1])
        ]

    if history.empty:
        st.warning("No historical records match the selected filters.")
        return

    history["if_is_anomaly"] = history["if_anomaly"] == -1
    history["lof_is_anomaly"] = history["lof_anomaly"] == -1
    history["consensus_is_anomaly"] = history["if_is_anomaly"] & history["lof_is_anomaly"]
    history["business_rule_triggered"] = (
        history["any_flag"].fillna(False).astype(bool)
        if "any_flag" in history else False
    )

    summary = load_temporal_summary(
        cache_signature=temporal_summary_signature()
    )
    latest_date = history["extraction_date"].max()

    if detailed_view == "1m vs 3m":
        st.info(
            "The same-extraction comparison is descriptive because 1m and 3m usually "
            "refer to different planned stays. The early-booking comparison below uses "
            "the same listing and the same stay dates observed in different weeks."
        )
        if {"1m", "3m"}.issubset(set(history["lookahead_period"].dropna().unique())):
            paired = history[history["lookahead_period"].isin(["1m", "3m"])].copy()
            same_extraction = paired.merge(
                paired[["extraction_date", "id", "lookahead_period", "price"]],
                on=["extraction_date", "id"], suffixes=("_1", "_2"), how="inner",
            )
            same_extraction = same_extraction[
                (same_extraction["lookahead_period_1"] == "1m")
                & (same_extraction["lookahead_period_2"] == "3m")
            ].drop_duplicates(["extraction_date", "id"])
            if not same_extraction.empty:
                same_extraction["observed_3m_minus_1m"] = (
                    same_extraction["price_2"] - same_extraction["price_1"]
                )
                same_extraction["observed_gap_pct"] = (
                    same_extraction["observed_3m_minus_1m"]
                    / same_extraction["price_1"].replace(0, np.nan)
                )
                st.subheader("Same extraction: descriptive horizon difference")
                h1, h2, h3 = st.columns(3)
                h1.metric("Matched listings", f"{len(same_extraction):,}")
                h2.metric("Median 3m - 1m", f"€{same_extraction['observed_3m_minus_1m'].median():,.0f}")
                h3.metric("3m higher than 1m", f"{(same_extraction['observed_3m_minus_1m'] > 0).mean():.1%}")
                st.caption("This is not a saving estimate because the two observations normally refer to different check-in dates.")

            if {"checkin_date", "checkout_date"}.issubset(history.columns):
                stay_keys = ["id", "checkin_date", "checkout_date"]
                early = history[history["lookahead_period"] == "3m"].rename(
                    columns={"extraction_date": "extraction_3m", "price": "price_3m"}
                )
                late = history[history["lookahead_period"] == "1m"].rename(
                    columns={"extraction_date": "extraction_1m", "price": "price_1m"}
                )
                same_stay = early[stay_keys + ["extraction_3m", "price_3m"]].merge(
                    late[stay_keys + ["extraction_1m", "price_1m"]], on=stay_keys, how="inner"
                )
                same_stay = same_stay[
                    same_stay["extraction_3m"] < same_stay["extraction_1m"]
                ].drop_duplicates(stay_keys + ["extraction_3m", "extraction_1m"])
                if same_stay.empty:
                    st.warning("No repeated same-stay observations are available for early-booking analysis.")
                else:
                    same_stay["saving_by_booking_3m"] = same_stay["price_1m"] - same_stay["price_3m"]
                    same_stay["saving_pct"] = same_stay["saving_by_booking_3m"] / same_stay["price_1m"].replace(0, np.nan)
                    st.subheader("Same stay across weeks: early-booking effect")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Comparable observations", f"{len(same_stay):,}")
                    s2.metric("Median saving at 3m", f"€{same_stay['saving_by_booking_3m'].median():,.0f}")
                    s3.metric("3m cheaper than 1m", f"{(same_stay['saving_by_booking_3m'] > 0).mean():.1%}")
                    s4.metric("Median relative saving", f"{same_stay['saving_pct'].median():.1%}")
                    st.caption("Positive savings mean the same listing and stay was cheaper when observed three months before check-in. This is observational, not a causal experiment.")
                    savings_fig = px.histogram(
                        same_stay, x="saving_pct", nbins=60,
                        title="Distribution of the observed early-booking saving",
                        labels={"saving_pct": "Saving when booked at 3m relative to 1m"},
                    )
                    savings_fig.update_xaxes(tickformat=".0%")
                    st.plotly_chart(savings_fig, width="stretch")
                    savings_by_date = same_stay.groupby("checkin_date", as_index=False).agg(
                        observations=("id", "size"), median_saving=("saving_by_booking_3m", "median"),
                        median_saving_pct=("saving_pct", "median"),
                    ).sort_values("checkin_date")
                    st.dataframe(savings_by_date, width="stretch", hide_index=True)
        else:
            st.warning("Both 1m and 3m snapshots are required for this comparison.")

    if detailed_view == "Persistent anomalies":
        st.caption("Persistence identifies listings repeatedly flagged across snapshots, which is more informative than a single isolated detection.")
        persistent = history.groupby("id", as_index=False).agg(
            snapshots=("extraction_date", "nunique"),
            consensus_flags=("consensus_is_anomaly", "sum"),
            if_flags=("if_is_anomaly", "sum"),
            lof_flags=("lof_is_anomaly", "sum"),
            name=("name", "first"),
            property_type=("property_type", "first"),
            region=("region", "first"),
            median_price=("price", "median"),
        )
        persistent["consensus_persistence"] = persistent["consensus_flags"] / persistent["snapshots"]
        persistent = persistent[persistent["consensus_flags"] > 0].sort_values(
            ["consensus_flags", "consensus_persistence"], ascending=False
        )
        if persistent.empty:
            st.info("No persistent consensus anomalies match the selected filters.")
        else:
            st.dataframe(persistent.head(100), width="stretch", hide_index=True)
            persistence_fig = px.bar(
                persistent.head(20).sort_values("consensus_flags"),
                x="consensus_flags", y="name", color="property_type", orientation="h",
                title="Listings with the most consensus detections",
                labels={"consensus_flags": "Consensus detections", "name": "Listing"},
            )
            st.plotly_chart(persistence_fig, width="stretch")

    if detailed_view == "Segments":
        segment_options = [c for c in ["region", "property_type", "province"] if c in history.columns]
        if not segment_options:
            st.warning("No segment variables are available.")
        else:
            segment = st.selectbox("Segment", segment_options, key="temporal_segment")
            segment_summary = history.groupby(["extraction_date", segment], as_index=False).agg(
                listings=("id", "size"), median_price=("price", "median"),
                consensus_rate=("consensus_is_anomaly", "mean"),
            )
            top_segments = history[segment].value_counts().head(15).index
            segment_summary = segment_summary[segment_summary[segment].isin(top_segments)]
            segment_fig = px.line(
                segment_summary, x="extraction_date", y="median_price", color=segment,
                title=f"Median price evolution by {segment}", markers=True,
            )
            st.plotly_chart(segment_fig, width="stretch")
            latest_segment = segment_summary[segment_summary["extraction_date"] == latest_date].sort_values("consensus_rate", ascending=False)
            st.subheader("Segments with the highest current consensus rate")
            st.dataframe(latest_segment.head(20), width="stretch", hide_index=True)

    if detailed_view == "Coverage & quality":
        coverage = history.groupby(["extraction_date", "lookahead_period"], as_index=False).agg(
            listings=("id", "size"), regions=("region", "nunique") if "region" in history else ("id", "nunique"),
            property_types=("property_type", "nunique") if "property_type" in history else ("id", "nunique"),
        )
        coverage_fig = px.line(
            coverage, x="extraction_date", y="listings", color="lookahead_period",
            markers=True, title="Snapshot coverage by extraction date",
        )
        st.plotly_chart(coverage_fig, width="stretch")
        st.dataframe(coverage, width="stretch", hide_index=True)
        st.caption("Coverage changes must be considered before interpreting price or anomaly changes as genuine market evolution.")

    st.subheader("Temporal summary 📋")
    st.dataframe(summary, width="stretch", hide_index=True)


def page_seasonal_market_analysis() -> None:
    """Analyse seasonal price levels and anomaly concentration."""
    requested = tuple(sorted({
        "id", "property_type", "region", "price", "if_anomaly", "lof_anomaly",
        "checkin_date", "extraction_date", "lookahead_period",
    }))
    history = load_temporal_history(
        cache_signature=temporal_history_signature(),
        requested_columns=requested,
    )
    if history.empty:
        st.warning("No historical snapshots are available.")
        return

    required_columns = {"property_type", "region", "price", "checkin_date", "if_anomaly", "lof_anomaly"}
    if not required_columns.issubset(history.columns):
        st.warning("The historical files do not contain the columns required for seasonal analysis.")
        return
    history = history.dropna(subset=["price", "checkin_date"])
    history["checkin_date"] = pd.to_datetime(history["checkin_date"], errors="coerce")
    history["extraction_date"] = pd.to_datetime(history["extraction_date"], errors="coerce")
    history = history.dropna(subset=["checkin_date", "extraction_date"])
    history["consensus"] = (
        (history["if_anomaly"] == -1) & (history["lof_anomaly"] == -1)
    )
    month_order = list(range(1, 13))
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    history["stay_month_num"] = history["checkin_date"].dt.month
    history["stay_month"] = history["stay_month_num"].map(month_labels)

    st.title("Seasonal Market & Anomaly Analysis 🌤️")
    st.caption(
        "Prices and anomaly signals grouped by the planned month of stay, "
        "using historical Booking.com snapshots."
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        periods = ["All"] + sorted(history["lookahead_period"].dropna().unique().tolist())
        selected_period = st.selectbox("Booking horizon", periods, key="seasonal_period")
    with filter_col2:
        property_types = sorted(history["property_type"].dropna().unique().tolist())
        selected_type = st.selectbox("Property type", ["All"] + property_types, key="seasonal_type")

    filtered = history.copy()
    if selected_period != "All":
        filtered = filtered[filtered["lookahead_period"] == selected_period]
    if selected_type != "All":
        filtered = filtered[filtered["property_type"] == selected_type]
    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    seasonal = filtered.groupby(
        ["stay_month_num", "stay_month", "lookahead_period"], as_index=False
    ).agg(
        listings=("id", "size"),
        median_price=("price", "median"),
        consensus_rate=("consensus", "mean"),
    )
    seasonal["month_order"] = seasonal["stay_month_num"]
    seasonal = seasonal.sort_values(["month_order", "lookahead_period"])

    latest_extraction = filtered["extraction_date"].max()
    latest = filtered[filtered["extraction_date"] == latest_extraction]
    historical = filtered[filtered["extraction_date"] < latest_extraction]
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Latest snapshot", latest_extraction.strftime("%Y-%m-%d"))
    metric_col2.metric("Latest listings", f"{len(latest):,}")
    metric_col3.metric("Latest consensus rate", f"{latest['consensus'].mean():.2%}")

    st.subheader("Seasonal price profile")
    price_fig = px.line(
        seasonal, x="stay_month", y="median_price", color="lookahead_period",
        markers=True, category_orders={"stay_month": [month_labels[m] for m in month_order]},
        title="Median price by planned month of stay",
        labels={
            "stay_month": "Planned stay month",
            "median_price": "Median price (EUR)",
            "lookahead_period": "Booking horizon",
        },
    )
    price_fig.update_layout(hovermode="x unified")
    st.plotly_chart(price_fig, width="stretch")

    st.subheader("Where do anomalies concentrate?")
    regions = filtered["region"].value_counts().head(12).index
    regional = filtered[filtered["region"].isin(regions)].groupby(
        ["stay_month", "stay_month_num", "region"], as_index=False
    ).agg(consensus_rate=("consensus", "mean"), listings=("id", "size"))
    regional_pivot = regional.pivot_table(
        index="region", columns="stay_month_num", values="consensus_rate", aggfunc="mean"
    ).reindex(columns=month_order)
    regional_pivot.columns = [month_labels[month] for month in regional_pivot.columns]
    if regional_pivot.dropna(how="all").empty:
        st.info("There is not enough regional data to build the seasonal anomaly heatmap.")
    else:
        anomaly_fig = px.imshow(
            regional_pivot, aspect="auto", color_continuous_scale="Reds",
            title="Consensus anomaly rate by region and stay month",
            labels={"x": "Planned stay month", "y": "Region", "color": "Consensus rate"},
        )
        anomaly_fig.update_coloraxes(colorbar_tickformat=".1%")
        st.plotly_chart(anomaly_fig, width="stretch")

    st.subheader("Current snapshot versus historical baseline")
    current = latest.groupby("lookahead_period").agg(
        current_price=("price", "median"),
        current_consensus=("consensus", "mean"),
    ).reset_index()
    if historical.empty:
        st.info("There is no earlier snapshot available for a historical baseline.")
    else:
        baseline = historical.groupby("lookahead_period").agg(
            historical_price=("price", "median"),
            historical_consensus=("consensus", "mean"),
        ).reset_index()
        baseline_view = current.merge(baseline, on="lookahead_period", how="left")
        baseline_view["price_change_pct"] = (
            baseline_view["current_price"] / baseline_view["historical_price"] - 1
        )
        st.dataframe(baseline_view, width="stretch", hide_index=True)
    st.caption(
        "This view separates seasonal effects from anomaly signals. A high price in a "
        "summer month is not automatically an anomaly; the models evaluate the full context."
    )


def page_temporal_evolution_v2(df: pd.DataFrame) -> None:
    """Compact temporal analysis focused on market change and anomaly persistence."""
    st.title("Market Timeline 📅")
    st.caption(
        "Three views: weekly market change, booking-horizon dynamics and "
        "anomalies that persist across observations."
    )

    view = st.radio(
        "Analysis",
        ["Weekly market", "Persistent anomalies", "Seasonal market"],
        horizontal=True,
        key="temporal_v2_view",
    )

    if view == "Weekly market":
        summary = load_temporal_summary(
            cache_signature=temporal_summary_signature()
        )
        if summary.empty:
            st.warning("No temporal summary is available. Run the pipeline first.")
            return

        period = st.selectbox(
            "Booking horizon", ["All"] + sorted(summary["lookahead_period"].dropna().unique()),
            key="temporal_v2_period",
        )
        if period != "All":
            summary = summary[summary["lookahead_period"] == period]
        summary = summary.sort_values("extraction_date")
        if "p25_price" not in summary.columns:
            summary["p25_price"] = summary["median_price"]
        if "p75_price" not in summary.columns:
            summary["p75_price"] = summary["median_price"]
        latest = summary.iloc[-1]
        previous = summary.iloc[-2] if len(summary) > 1 else None

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Latest listings", f"{int(latest['total_listings']):,}")
        metric_col2.metric("Median price", f"€{latest['median_price']:,.0f}")
        metric_col3.metric("Consensus rate", f"{latest['consensus_anomaly_rate']:.2%}")
        if previous is not None:
            price_delta = latest["median_price"] - previous["median_price"]
            metric_col4.metric("Change vs previous", f"€{price_delta:,.0f}")
        else:
            metric_col4.metric("Change vs previous", "N/A")

        st.subheader("Market level and uncertainty")
        band_fig = go.Figure()
        for horizon, horizon_data in summary.groupby("lookahead_period"):
            horizon_data = horizon_data.sort_values("extraction_date")
            band_fig.add_trace(go.Scatter(
                x=horizon_data["extraction_date"], y=horizon_data["p75_price"],
                mode="lines", line=dict(width=0), showlegend=False,
                legendgroup=horizon, name=f"{horizon} upper quartile",
            ))
            band_fig.add_trace(go.Scatter(
                x=horizon_data["extraction_date"], y=horizon_data["p25_price"],
                mode="lines", fill="tonexty", fillcolor="rgba(37,99,235,0.12)",
                line=dict(width=0), name=f"{horizon} price range", legendgroup=horizon,
            ))
            band_fig.add_trace(go.Scatter(
                x=horizon_data["extraction_date"], y=horizon_data["median_price"],
                mode="lines+markers", line=dict(width=3), name=f"Median {horizon}",
                legendgroup=horizon,
            ))
        band_fig.update_layout(
            title="Weekly price level with interquartile range",
            xaxis_title="Extraction date", yaxis_title="Price (EUR)",
            hovermode="x unified",
        )
        st.plotly_chart(band_fig, width="stretch")

        anomaly_fig = px.imshow(
            summary.pivot(index="lookahead_period", columns="extraction_date", values="consensus_anomaly_rate"),
            aspect="auto", color_continuous_scale="Reds",
            title="Consensus anomaly intensity by week and horizon",
            labels={"x": "Extraction date", "y": "Horizon", "color": "Consensus rate"},
        )
        anomaly_fig.update_coloraxes(colorbar_tickformat=".1%")
        st.plotly_chart(anomaly_fig, width="stretch")

    elif view == "Persistent anomalies":
        requested = tuple(sorted({
            "id", "name", "property_type", "region", "price",
            "if_anomaly", "lof_anomaly", "extraction_date", "lookahead_period",
        }))
        history = load_temporal_history(
            cache_signature=temporal_history_signature(),
            requested_columns=requested,
        )
        if history.empty:
            st.warning("No historical snapshots are available.")
            return
        history["consensus"] = (
            (history["if_anomaly"] == -1) & (history["lof_anomaly"] == -1)
        )
        persistence = history.groupby("id", as_index=False).agg(
            name=("name", "first"), property_type=("property_type", "first"),
            region=("region", "first"), snapshots=("extraction_date", "nunique"),
            consensus_flags=("consensus", "sum"), median_price=("price", "median"),
        )
        persistence["persistence_rate"] = (
            persistence["consensus_flags"] / persistence["snapshots"]
        )
        persistence = persistence[persistence["consensus_flags"] > 0].sort_values(
            ["consensus_flags", "persistence_rate"], ascending=False
        ).head(30)
        if persistence.empty:
            st.info("No persistent consensus anomalies were found.")
            return

        st.subheader("Listings repeatedly identified as consensus anomalies")
        persistence_fig = px.scatter(
            persistence, x="snapshots", y="consensus_flags", size="median_price",
            color="property_type", hover_name="name",
            title="Persistence versus number of consensus detections",
            labels={"snapshots": "Snapshots observed", "consensus_flags": "Consensus detections"},
        )
        st.plotly_chart(persistence_fig, width="stretch")
        st.dataframe(persistence, width="stretch", hide_index=True)
    else:
        page_seasonal_market_analysis()
        return

# ----------------------------------------------------------------------------
# MAIN APP ROUTER
# ----------------------------------------------------------------------------
def main() -> None:
    with st.spinner("Loading analytical data..."):
        try:
            full_df = load_data(cache_signature=latest_data_signature())
        except FileNotFoundError:
            st.error(f"Data file not found at {DEFAULT_DATA_PATH}. Ensure the MLOps pipeline completed successfully.")
            st.stop()

    st.sidebar.title("Navigation")

    if st.sidebar.button("Refresh Data Pipeline", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    PAGES: Dict[str, Callable] = {
        "1. Overview": page_overview,
        "2. Price anomaly analysis": page_price_anomalies,
        "3. Market anomaly explorer": page_anomaly_explorer,
        "4. Listing summary": page_optimized_listing_view,
        "5. Segment analysis": page_segment_analysis,
        "6. Business rules validation": page_business_rules,
        "7. Final consensus ranking": page_consensus_ranking,
        "8. Temporal market evolution": page_temporal_evolution_v2,
    }

    selection = st.sidebar.radio("Modules", list(PAGES.keys()))
    st.sidebar.divider()
    filtered_df = filter_sidebar(full_df)
    page_function = PAGES[selection]
    page_function(filtered_df)


if __name__ == "__main__":
    main()