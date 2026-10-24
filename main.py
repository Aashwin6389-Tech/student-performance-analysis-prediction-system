import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Analytics | Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL DARK BLUE THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #08111f;
        --panel: #101d30;
        --panel2: #14243a;
        --border: #263b56;
        --text: #edf4ff;
        --muted: #9aadc6;
        --blue: #60a5fa;
        --cyan: #67e8f9;
        --green: #34d399;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top left, #142746 0%, transparent 42%),
            linear-gradient(135deg, #08111f 0%, #0b1627 55%, #0c1b30 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #0b1729;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3, h4, p, label, li {
        color: var(--text);
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -1.1px;
    }

    h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }

    .hero {
        background: linear-gradient(120deg, #142b49, #102039 55%, #142e4b);
        border: 1px solid #2a4565;
        padding: 30px 32px;
        border-radius: 20px;
        margin: 8px 0 26px 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    }

    .hero-label {
        color: #93c5fd;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .hero-title {
        color: #f4f8ff;
        font-size: clamp(26px, 3.3vw, 39px);
        font-weight: 800;
        margin: 12px 0 8px 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #b5c8e1;
        font-size: 14px;
        line-height: 1.8;
        max-width: 850px;
    }

    .section-label {
        color: #93c5fd;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .metric-card {
        background: linear-gradient(145deg, #14243a, #0f1d30);
        border: 1px solid #263d59;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 12px;
        color: #a6b8d0;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: clamp(22px, 2.2vw, 31px);
        color: #f1f7ff;
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .metric-foot {
        color: #8da7c8;
        font-size: 11px;
        margin-top: 8px;
    }

    .info-card {
        background: #101e31;
        border: 1px solid #263b56;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .muted {
        color: #a6b8d0;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 30px;
        background: #123b37;
        border: 1px solid #21685c;
        color: #6ee7b7;
        font-size: 11px;
        font-weight: 700;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563eb, #0e7490);
        color: white !important;
        border: 1px solid #3b82f6;
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #93c5fd;
        color: white !important;
        filter: brightness(1.12);
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        background: #101e31 !important;
        color: #f1f7ff !important;
        border: 1px solid #344b68 !important;
        border-radius: 9px !important;
    }

    [data-baseweb="select"] > div {
        background: #101e31 !important;
        border-color: #344b68 !important;
        color: #f1f7ff !important;
        border-radius: 9px !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #14243a !important;
    }

    [role="option"] {
        color: #f1f7ff !important;
    }

    [role="option"]:hover {
        background: #24476b !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #263b56;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: #101e31;
        border: 1px solid #263b56;
        border-radius: 12px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a6b8d0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f7ff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #263b56;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a6b8d0;
        border-radius: 8px 8px 0 0;
        padding: 12px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #93c5fd !important;
        border-bottom-color: #60a5fa !important;
    }

    hr {
        border-color: #263b56;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px 18px;
        }
        .metric-card {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS AND HELPERS
# ============================================================

DATA_CANDIDATES = [
    Path("data/Cleaned_Students_Performance.csv"),
    Path("Cleaned_Students_Performance.csv"),
]

PLOT_TEMPLATE = "plotly_dark"
PALETTE = ["#60a5fa", "#67e8f9", "#34d399", "#fbbf24", "#c084fc", "#fb7185"]

ID_HINTS = (
    "student_id", "studentid", "roll_no", "rollno",
    "registration_id", "record_id", "index",
)

DERIVED_HINTS = (
    "total_score", "total_marks", "average_score",
    "average_marks", "overall_score", "overall_average",
    "final_score", "final_result",
)

TARGET_HINTS = (
    "score", "marks", "grade", "math", "reading",
    "writing", "g1", "g2", "g3", "exam", "result",
)


def clean_column_name(name):
    return str(name).strip()


def safe_filename(name):
    allowed = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(name)
    )
    return allowed[:80] or "report"


def load_csv_bytes(uploaded_file):
    raw = uploaded_file.getvalue()
    errors = []

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("Could not read the CSV file. Please verify its format.")


def load_default_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                return pd.read_csv(path), str(path)
            except Exception as exc:
                st.error(f"Could not load {path}: {exc}")

    return None, None


def standardize_frame(df):
    data = df.copy()
    data.columns = [clean_column_name(c) for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # Convert object columns only when nearly all non-empty values are numeric.
    for col in data.select_dtypes(include=["object", "string"]).columns:
        original = data[col]
        stripped = original.astype("string").str.strip()
        numeric = pd.to_numeric(stripped, errors="coerce")
        nonempty = stripped.notna() & stripped.ne("")
        denominator = int(nonempty.sum())

        if denominator > 0:
            ratio = numeric[nonempty].notna().mean()
            if ratio >= 0.95:
                data[col] = numeric

    return data


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()


def is_identifier(col):
    name = str(col).lower().replace(" ", "_")
    return any(hint in name for hint in ID_HINTS)


def is_derived(col):
    name = str(col).lower().replace(" ", "_")
    return any(hint in name for hint in DERIVED_HINTS)


def suggested_target_columns(df):
    nums = numeric_columns(df)
    candidates = [
        c for c in nums
        if any(hint in str(c).lower() for hint in TARGET_HINTS)
        and not is_derived(c)
        and not is_identifier(c)
    ]
    return candidates


def chart_layout(fig, height=360):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#dce8f8"),
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        title=dict(
            font=dict(size=16, color="#eaf2ff"),
            x=0.02,
            xanchor="left",
        ),
    )
    fig.update_xaxes(
        gridcolor="#24364e",
        zerolinecolor="#344b68",
        linecolor="#344b68",
    )
    fig.update_yaxes(
        gridcolor="#24364e",
        zerolinecolor="#344b68",
        linecolor="#344b68",
    )
    return fig


def show_metric(label, value, foot=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def build_summary(df):
    rows = [
        ("Rows", len(df)),
        ("Columns", len(df.columns)),
        ("Duplicate rows", int(df.duplicated().sum())),
        ("Missing cells", int(df.isna().sum().sum())),
        ("Missing percentage", round(
            df.isna().sum().sum() / max(df.size, 1) * 100, 2
        )),
    ]

    for col in numeric_columns(df):
        series = df[col].dropna()
        if len(series):
            rows.extend([
                (f"{col} - mean", round(float(series.mean()), 3)),
                (f"{col} - median", round(float(series.median()), 3)),
                (f"{col} - standard deviation", round(float(series.std()), 3)),
                (f"{col} - minimum", round(float(series.min()), 3)),
                (f"{col} - maximum", round(float(series.max()), 3)),
            ])

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">STUDENT INTELLIGENCE PLATFORM</div>
        <div class="hero-title">Student Performance Analytics</div>
        <div class="hero-subtitle">
            Explore academic data, uncover performance patterns, assess data
            quality, and evaluate machine-learning predictions in one
            interactive workspace.
        </div>
        <br>
        <span class="status-pill">● ANALYTICS WORKSPACE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR AND DATA LOADING
# ============================================================

st.sidebar.markdown("## 🎓 Student Analytics")
st.sidebar.caption("Performance Intelligence Platform")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload student CSV",
    type=["csv"],
    help="Upload a CSV to analyze it without changing the repository dataset.",
)

default_df, default_source = load_default_data()

if uploaded_file is not None:
    try:
        source_df = load_csv_bytes(uploaded_file)
        source_name = uploaded_file.name
    except Exception as exc:
        st.error(f"CSV upload failed: {exc}")
        st.stop()
elif default_df is not None:
    source_df = default_df
    source_name = default_source
else:
    st.error(
        "No default dataset was found. Add "
        "`data/Cleaned_Students_Performance.csv` to the repository "
        "or upload a CSV using the sidebar."
    )
    st.stop()

if source_df.empty:
    st.error("The selected CSV has no data rows.")
    st.stop()

df = standardize_frame(source_df)
original_row_count = len(df)

st.sidebar.markdown("### Workspace")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Explorer",
        "Performance Analytics",
        "ML Prediction",
        "Data Quality",
        "Export Center",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Dataset")
st.sidebar.caption(f"Source: `{source_name}`")
st.sidebar.caption(f"{len(df):,} records · {len(df.columns):,} columns")

if uploaded_file is not None:
    st.sidebar.info(
        "You are analyzing the uploaded file. "
        "The repository dataset remains unchanged."
    )

with st.sidebar.expander("About this application"):
    st.write(
        "Built with Streamlit, Pandas, NumPy, Plotly and Scikit-learn. "
        "Predictions are estimates and depend on the quality and suitability "
        "of the selected dataset."
    )

# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.markdown("### Data filters")

filtered_df = df.copy()
cats = categorical_columns(df)

if cats:
    with st.sidebar.expander("Filter categories", expanded=False):
        for col in cats[:8]:
            values = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )
            if 1 < len(values) <= 100:
                selected = st.multiselect(
                    f"{col}",
                    options=values,
                    default=[],
                    key=f"filter_{col}",
                )
                if selected:
                    filtered_df = filtered_df[
                        filtered_df[col].astype(str).isin(selected)
                    ]

nums = numeric_columns(df)

if nums:
    with st.sidebar.expander("Filter numeric ranges", expanded=False):
        for col in nums[:6]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            low = float(series.min())
            high = float(series.max())

            if low < high:
                chosen_range = st.slider(
                    f"{col}",
                    min_value=low,
                    max_value=high,
                    value=(low, high),
                    key=f"range_{col}",
                )
                filtered_df = filtered_df[
                    pd.to_numeric(filtered_df[col], errors="coerce")
                    .between(chosen_range[0], chosen_range[1])
                ]

if filtered_df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.sidebar.caption(f"{len(filtered_df):,} rows after filters")

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<div class="section-label">EXECUTIVE OVERVIEW</div>',
                unsafe_allow_html=True)
    st.title("Performance at a glance")
    st.caption(
        "A high-level summary of the currently filtered student dataset."
    )

    total_cells = filtered_df.size
    missing_cells = int(filtered_df.isna().sum().sum())
    missing_pct = missing_cells / max(total_cells, 1) * 100
    duplicate_rows = int(filtered_df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_metric("TOTAL RECORDS", f"{len(filtered_df):,}", "Filtered student records")
    with c2:
        show_metric("FEATURES", f"{len(filtered_df.columns):,}", "Available columns")
    with c3:
        show_metric("MISSING CELLS", f"{missing_cells:,}", f"{missing_pct:.2f}% of cells")
    with c4:
        show_metric("DUPLICATE ROWS", f"{duplicate_rows:,}", "Exact duplicate records")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Dataset profile")
        profile = pd.DataFrame({
            "Column": filtered_df.columns,
            "Data type": [str(t) for t in filtered_df.dtypes],
            "Missing values": [
                int(filtered_df[c].isna().sum())
                for c in filtered_df.columns
            ],
            "Unique values": [
                int(filtered_df[c].nunique(dropna=True))
                for c in filtered_df.columns
            ],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Data completeness")
        completeness = (
            (1 - filtered_df.isna().mean()) * 100
        ).reset_index()
        completeness.columns = ["Column", "Completeness (%)"]
        completeness = completeness.sort_values("Completeness (%)")

        fig = px.bar(
            completeness,
            x="Completeness (%)",
            y="Column",
            orientation="h",
            range_x=[0, 100],
            color="Completeness (%)",
            color_continuous_scale=["#fb7185", "#fbbf24", "#34d399"],
            title="Completeness by column",
        )
        st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

    if nums:
        st.markdown("### Numeric indicators")
        cols = st.columns(min(4, len(nums)))
        for i, col in enumerate(nums[:4]):
            series = pd.to_numeric(filtered_df[col], errors="coerce").dropna()
            with cols[i]:
                mean_value = series.mean() if len(series) else np.nan
                show_metric(
                    col.upper(),
                    f"{mean_value:,.2f}" if pd.notna(mean_value) else "N/A",
                    "Mean of filtered records",
                )

    st.markdown("### Quick actions")
    q1, q2, q3 = st.columns(3)

    with q1:
        st.download_button(
            "⬇ Download filtered CSV",
            data=csv_bytes(filtered_df),
            file_name="filtered_student_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with q2:
        st.download_button(
            "⬇ Download data profile",
            data=csv_bytes(profile),
            file_name="student_data_profile.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with q3:
        st.download_button(
            "⬇ Download summary",
            data=csv_bytes(build_summary(filtered_df)),
            file_name="student_analytics_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.markdown('<div class="section-label">DATA WORKSPACE</div>',
                unsafe_allow_html=True)
    st.title("Explore your dataset")
    st.caption("Inspect records, data types, unique values and descriptive statistics.")

    st.markdown("### Records")
    search = st.text_input(
        "Search across rows",
        placeholder="Type a name, category, value or keyword...",
    )

    view_df = filtered_df.copy()

    if search.strip():
        mask = view_df.astype(str).apply(
            lambda column: column.str.contains(
                search.strip(), case=False, regex=False, na=False
            )
        ).any(axis=1)
        view_df = view_df[mask]

    st.caption(f"Showing {len(view_df):,} matching records")
    st.dataframe(
        view_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download displayed records",
        data=csv_bytes(view_df),
        file_name="student_records.csv",
        mime="text/csv",
    )

    st.markdown("### Descriptive statistics")
    if nums:
        st.dataframe(
            filtered_df[nums].describe().T.reset_index().rename(
                columns={"index": "Column"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No numeric columns were detected in this dataset.")

    st.markdown("### Column explorer")
    selected_col = st.selectbox("Choose a column", df.columns.tolist())
    selected_series = filtered_df[selected_col]

    a, b, c = st.columns(3)
    a.metric("Data type", str(selected_series.dtype))
    b.metric("Unique values", f"{selected_series.nunique(dropna=True):,}")
    c.metric("Missing values", f"{selected_series.isna().sum():,}")

    st.markdown("#### Most frequent values")
    value_counts = (
        selected_series.astype("string")
        .fillna("(Missing)")
        .value_counts()
        .head(20)
        .rename_axis("Value")
        .reset_index(name="Count")
    )
    st.dataframe(value_counts, use_container_width=True, hide_index=True)

# ============================================================
# PERFORMANCE ANALYTICS
# ============================================================

elif page == "Performance Analytics":
    st.markdown('<div class="section-label">EXPLORATORY DATA ANALYSIS</div>',
                unsafe_allow_html=True)
    st.title("Performance analytics")
    st.caption("Explore distributions, compare groups and inspect relationships.")

    if not nums:
        st.warning("No numeric columns are available for numerical analysis.")
    else:
        selected_num = st.selectbox(
            "Primary numeric measure",
            nums,
            index=0,
        )

        series = pd.to_numeric(
            filtered_df[selected_num], errors="coerce"
        ).dropna()

        a, b, c, d = st.columns(4)
        a.metric("Mean", f"{series.mean():,.2f}" if len(series) else "N/A")
        b.metric("Median", f"{series.median():,.2f}" if len(series) else "N/A")
        c.metric("Std. deviation", f"{series.std():,.2f}" if len(series) > 1 else "N/A")
        d.metric("Valid values", f"{len(series):,}")

        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                filtered_df,
                x=selected_num,
                nbins=30,
                marginal="box",
                title=f"Distribution of {selected_num}",
                color_discrete_sequence=["#60a5fa"],
            )
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        with right:
            fig = px.box(
                filtered_df,
                y=selected_num,
                points="outliers",
                title=f"Spread and outliers: {selected_num}",
                color_discrete_sequence=["#67e8f9"],
            )
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.markdown("### Compare two numeric variables")

        if len(nums) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", nums, index=0)
            with col2:
                default_y = 1 if len(nums) > 1 else 0
                y_col = st.selectbox("Y-axis", nums, index=default_y)

            scatter_data = filtered_df[[x_col, y_col]].dropna()

            if not scatter_data.empty:
                fig = px.scatter(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    trendline="ols" if len(scatter_data) >= 3 else None,
                    title=f"{y_col} vs {x_col}",
                    opacity=0.75,
                    color_discrete_sequence=["#60a5fa"],
                )
                st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
            else:
                st.info("No complete rows are available for this comparison.")

        st.markdown("### Correlation heatmap")

        if len(nums) >= 2:
            corr = filtered_df[nums].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Pearson correlation between numeric variables",
            )
            st.plotly_chart(chart_layout(fig, 500), use_container_width=True)
            st.caption(
                "Correlation describes linear association, not causation."
            )
        else:
            st.info("At least two numeric columns are required for a correlation heatmap.")

        st.markdown("### Group comparison")

        if cats:
            cat_col = st.selectbox("Group by category", cats)
            group_num = st.selectbox(
                "Measure to summarize",
                nums,
                index=0,
                key="group_measure",
            )

            grouped = (
                filtered_df.groupby(cat_col, dropna=False)[group_num]
                .agg(["count", "mean", "median"])
                .reset_index()
                .sort_values("mean", ascending=False)
                .head(30)
            )

            st.dataframe(grouped, use_container_width=True, hide_index=True)

            fig = px.bar(
                grouped,
                x=cat_col,
                y="mean",
                color="mean",
                title=f"Average {group_num} by {cat_col}",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
        else:
            st.info("No categorical columns were detected for group comparisons.")

# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "ML Prediction":
    st.markdown('<div class="section-label">PREDICTIVE INTELLIGENCE</div>',
                unsafe_allow_html=True)
    st.title("Machine learning studio")
    st.caption(
        "Train a regression model on a selected target and inspect held-out "
        "test-set performance. Model quality depends on dataset suitability."
    )

    all_numeric = numeric_columns(df)
    targets = suggested_target_columns(df)

    if not targets:
        targets = [
            col for col in all_numeric
            if not is_identifier(col) and not is_derived(col)
        ]

    if len(filtered_df) < 15:
        st.warning(
            "At least 15 filtered rows are recommended for a basic train/test "
            "evaluation. Relax filters or use a larger dataset."
        )

    if not targets:
        st.warning(
            "No suitable numeric prediction target was detected. "
            "Check the dataset columns and choose a numeric outcome variable."
        )
    else:
        default_target = targets[0]
        target = st.selectbox(
            "Prediction target",
            targets,
            index=0,
            help=(
                "Choose the numeric outcome you want to predict. "
                "Avoid derived totals or columns that directly reveal the target."
            ),
        )

        st.markdown("### Evaluation settings")

        c1, c2, c3 = st.columns(3)

        with c1:
            test_size_pct = st.select_slider(
                "Test set size",
                options=[20, 25, 30],
                value=20,
                format_func=lambda x: f"{x}%",
            )

        with c2:
            trees = st.select_slider(
                "Number of trees",
                options=[100, 200, 300],
                value=200,
            )

        with c3:
            seed = st.number_input(
                "Random seed",
                min_value=1,
                max_value=99999,
                value=42,
                step=1,
            )

        # Avoid target leakage by removing likely derived fields and IDs.
        feature_candidates = [
            col for col in df.columns
            if col != target
            and not is_identifier(col)
            and not is_derived(col)
        ]

        # Explicitly remove columns that may directly reveal a score target.
        # For G3, G1 and G2 are legitimate prior grades but can make the
        # prediction task easier; user must decide whether they are available
        # at the intended prediction time.
        related_grade_cols = {
            "g1": {"g2", "g3"},
            "g2": {"g3"},
            "g3": set(),
        }
        target_lower = target.lower()
        remove_names = related_grade_cols.get(target_lower, set())

        feature_candidates = [
            col for col in feature_candidates
            if col.lower() not in remove_names
        ]

        if not feature_candidates:
            st.warning("No eligible feature columns remain for this target.")
        else:
            selected_features = st.multiselect(
                "Model features",
                feature_candidates,
                default=feature_candidates,
                help=(
                    "For a fair prediction task, choose only fields that would "
                    "be known at prediction time. Exclude future scores, "
                    "post-outcome fields, identifiers, and target-derived fields."
                ),
            )

            st.info(
                "The model is evaluated on a held-out test set. Missing feature "
                "values are imputed within the training pipeline. R² can be "
                "negative when predictions perform worse than a test-set mean baseline."
            )

            train_clicked = st.button(
                "Train and evaluate model",
                type="primary",
                use_container_width=True,
            )

            if train_clicked:
                if len(selected_features) == 0:
                    st.error("Select at least one model feature.")
                else:
                    model_data = filtered_df[
                        selected_features + [target]
                    ].copy()

                    # Remove rows without a valid target; impute features
                    # inside the pipeline to prevent train/test leakage.
                    model_data[target] = pd.to_numeric(
                        model_data[target], errors="coerce"
                    )
                    model_data = model_data.dropna(subset=[target])

                    if len(model_data) < 15:
                        st.error(
                            "Not enough valid target rows. At least 15 are "
                            "recommended for this simple evaluation."
                        )
                    elif model_data[target].nunique() < 2:
                        st.error(
                            "The selected target has fewer than two unique "
                            "values. Regression evaluation is not meaningful."
                        )
                    else:
                        X = model_data[selected_features]
                        y = model_data[target]

                        numeric_features = X.select_dtypes(
                            include=np.number
                        ).columns.tolist()
                        categorical_features = [
                            col for col in selected_features
                            if col not in numeric_features
                        ]

                        transformers = []

                        if numeric_features:
                            numeric_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                            ])
                            transformers.append(
                                ("numeric", numeric_pipeline, numeric_features)
                            )

                        if categorical_features:
                            categorical_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                )),
                            ])
                            transformers.append(
                                ("categorical", categorical_pipeline, categorical_features)
                            )

                        if not transformers:
                            st.error("No usable features were found.")
                        else:
                            preprocessing = ColumnTransformer(
                                transformers=transformers,
                                remainder="drop",
                            )

                            model = Pipeline([
                                ("preprocessing", preprocessing),
                                ("regressor", RandomForestRegressor(
                                    n_estimators=int(trees),
                                    random_state=int(seed),
                                    n_jobs=-1,
                                    min_samples_leaf=2,
                                )),
                            ])

                            try:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=test_size_pct / 100,
                                    random_state=int(seed),
                                )

                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                                mae = mean_absolute_error(y_test, predictions)
                                r2 = (
                                    r2_score(y_test, predictions)
                                    if len(y_test) >= 2
                                    else np.nan
                                )

                                results = pd.DataFrame({
                                    "Actual": y_test.to_numpy(),
                                    "Predicted": predictions,
                                    "Absolute Error": np.abs(
                                        y_test.to_numpy() - predictions
                                    ),
                                })

                                st.session_state["ml_result"] = {
                                    "target": target,
                                    "features": selected_features,
                                    "mae": float(mae),
                                    "r2": float(r2) if pd.notna(r2) else None,
                                    "model": model,
                                    "results": results,
                                    "feature_columns": selected_features,
                                    "train_rows": len(X_train),
                                    "test_rows": len(X_test),
                                    "target_min": float(y.min()),
                                    "target_max": float(y.max()),
                                }

                            except Exception as exc:
                                st.error(
                                    "Model training failed. Check the selected "
                                    f"columns and data types. Details: {exc}"
                                )

            result = st.session_state.get("ml_result")

            if result and result.get("target") == target:
                st.markdown("### Evaluation results")

                r1, r2_col, r3, r4 = st.columns(4)

                with r1:
                    show_metric(
                        "MAE",
                        f"{result['mae']:.3f}",
                        "Lower is generally better",
                    )

                with r2_col:
                    r2_value = result["r2"]
                    show_metric(
                        "R² SCORE",
                        f"{r2_value:.3f}" if r2_value is not None else "N/A",
                        "Closer to 1 can indicate better fit",
                    )

                with r3:
                    show_metric(
                        "TRAIN ROWS",
                        f"{result['train_rows']:,}",
                        "Training partition",
                    )

                with r4:
                    show_metric(
                        "TEST ROWS",
                        f"{result['test_rows']:,}",
                        "Unseen evaluation partition",
                    )

                results_df = result["results"]

                left, right = st.columns(2)

                with left:
                    fig = px.scatter(
                        results_df,
                        x="Actual",
                        y="Predicted",
                        title="Actual vs predicted",
                        color_discrete_sequence=["#67e8f9"],
                        hover_data=["Absolute Error"],
                    )

                    low = min(
                        results_df["Actual"].min(),
                        results_df["Predicted"].min(),
                    )
                    high = max(
                        results_df["Actual"].max(),
                        results_df["Predicted"].max(),
                    )

                    fig.add_trace(go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        name="Ideal prediction",
                        line=dict(color="#fb7185", dash="dash"),
                    ))

                    st.plotly_chart(
                        chart_layout(fig, 420),
                        use_container_width=True,
                    )

                with right:
                    fig = px.histogram(
                        results_df,
                        x="Absolute Error",
                        nbins=20,
                        title="Distribution of absolute errors",
                        color_discrete_sequence=["#60a5fa"],
                    )
                    st.plotly_chart(
                        chart_layout(fig, 420),
                        use_container_width=True,
                    )

                st.markdown("### Test-set predictions")
                st.dataframe(
                    results_df.round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇ Download test-set predictions",
                    data=csv_bytes(results_df),
                    file_name="student_model_predictions.csv",
                    mime="text/csv",
                )

                st.markdown("### Generate a new prediction")

                st.caption(
                    "Enter a new student's feature values. Numeric inputs default "
                    "to the training dataset median; categorical inputs default "
                    "to the most frequent category."
                )

                input_values = {}

                with st.form("prediction_form"):
                    cols = st.columns(2)

                    for idx, feature in enumerate(result["feature_columns"]):
                        series = df[feature]
                        with cols[idx % 2]:
                            if pd.api.types.is_numeric_dtype(series):
                                clean_series = pd.to_numeric(
                                    series, errors="coerce"
                                ).dropna()

                                default = (
                                    float(clean_series.median())
                                    if not clean_series.empty
                                    else 0.0
                                )

                                input_values[feature] = st.number_input(
                                    str(feature),
                                    value=default,
                                    key=f"new_{feature}",
                                )
                            else:
                                options = (
                                    series.dropna().astype(str)
                                    .value_counts()
                                    .index.tolist()
                                )

                                options = options[:200]

                                if options:
                                    input_values[feature] = st.selectbox(
                                        str(feature),
                                        options,
                                        key=f"new_{feature}",
                                    )
                                else:
                                    input_values[feature] = st.text_input(
                                        str(feature),
                                        key=f"new_{feature}",
                                    )

                    predict_clicked = st.form_submit_button(
                        "Predict outcome",
                        use_container_width=True,
                    )

                if predict_clicked:
                    new_row = pd.DataFrame(
                        [input_values],
                        columns=result["feature_columns"],
                    )

                    try:
                        predicted_value = result["model"].predict(new_row)[0]

                        st.success(
                            f"Estimated {result['target']}: "
                            f"{predicted_value:,.3f}"
                        )

                        st.caption(
                            "This is a model estimate, not a guaranteed outcome. "
                            "Its reliability depends on training data, feature "
                            "availability and evaluation performance."
                        )

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

                with st.expander("Feature importance"):
                    try:
                        fitted_preprocessor = result["model"].named_steps[
                            "preprocessing"
                        ]
                        fitted_model = result["model"].named_steps["regressor"]

                        names = fitted_preprocessor.get_feature_names_out()
                        importances = fitted_model.feature_importances_

                        importance_df = pd.DataFrame({
                            "Feature": names,
                            "Importance": importances,
                        }).sort_values("Importance", ascending=False).head(25)

                        fig = px.bar(
                            importance_df.sort_values("Importance"),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Random Forest feature importance",
                            color="Importance",
                            color_continuous_scale="Blues",
                        )

                        st.plotly_chart(
                            chart_layout(fig, 480),
                            use_container_width=True,
                        )

                        st.caption(
                            "Feature importance indicates model reliance, not "
                            "causality or the effect of changing a feature."
                        )

                    except Exception as exc:
                        st.info(f"Feature importance unavailable: {exc}")

# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":
    st.markdown('<div class="section-label">DATA GOVERNANCE</div>',
                unsafe_allow_html=True)
    st.title("Data quality center")
    st.caption("Inspect missing values, duplicates, data types and potential outliers.")

    missing_count = filtered_df.isna().sum()
    missing_pct = filtered_df.isna().mean() * 100
    duplicate_count = int(filtered_df.duplicated().sum())

    a, b, c = st.columns(3)
    a.metric("Missing cells", f"{int(missing_count.sum()):,}")
    b.metric("Duplicate rows", f"{duplicate_count:,}")
    c.metric(
        "Completeness",
        f"{(1 - filtered_df.isna().sum().sum() / max(filtered_df.size, 1)) * 100:.2f}%",
    )

    quality_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data type": [str(filtered_df[c].dtype) for c in filtered_df.columns],
        "Missing count": [int(missing_count[c]) for c in filtered_df.columns],
        "Missing (%)": [round(float(missing_pct[c]), 2) for c in filtered_df.columns],
        "Unique values": [
            int(filtered_df[c].nunique(dropna=True))
            for c in filtered_df.columns
        ],
    })

    st.markdown("### Column-level quality report")
    st.dataframe(
        quality_df.sort_values("Missing (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download quality report",
        data=csv_bytes(quality_df),
        file_name="student_data_quality_report.csv",
        mime="text/csv",
    )

    st.markdown("### Potential numeric outliers")

    if nums:
        outlier_rows = []

        for col in nums:
            series = pd.to_numeric(filtered_df[col], errors="coerce").dropna()

            if len(series) < 4:
                outlier_rows.append({
                    "Column": col,
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "Potential outliers": 0,
                    "Method": "Not enough values",
                })
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                outlier_count = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((series < lower) | (series > upper)).sum())

            outlier_rows.append({
                "Column": col,
                "Q1": round(float(q1), 3),
                "Q3": round(float(q3), 3),
                "IQR": round(float(iqr), 3),
                "Potential outliers": outlier_count,
                "Method": "1.5 × IQR",
            })

        outlier_df = pd.DataFrame(outlier_rows)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download outlier report",
            data=csv_bytes(outlier_df),
            file_name="student_outlier_report.csv",
            mime="text/csv",
        )

    st.markdown("### Create a cleaned copy")

    st.write(
        "Cleaning options below create a downloadable copy. They do not overwrite "
        "the original repository dataset."
    )

    drop_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    drop_empty_rows = st.checkbox("Remove rows that are entirely empty", value=True)

    if st.button("Prepare cleaned dataset", type="primary"):
        cleaned = filtered_df.copy()

        before = len(cleaned)

        if drop_empty_rows:
            cleaned = cleaned.dropna(how="all")

        if drop_duplicates:
            cleaned = cleaned.drop_duplicates()

        removed = before - len(cleaned)

        st.success(
            f"Cleaned copy prepared: {len(cleaned):,} rows remain; "
            f"{removed:,} rows removed."
        )

        st.download_button(
            "⬇ Download cleaned dataset",
            data=csv_bytes(cleaned),
            file_name="cleaned_student_performance.csv",
            mime="text/csv",
        )

        st.caption(
            "Missing values in individual cells are not automatically removed "
            "or imputed by this cleaning action."
        )

# ============================================================
# EXPORT CENTER
# ============================================================

elif page == "Export Center":
    st.markdown('<div class="section-label">REPORTING WORKSPACE</div>',
                unsafe_allow_html=True)
    st.title("Export center")
    st.caption("Download the current filtered data and reusable analysis reports.")

    st.markdown("### Current workspace")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(filtered_df):,}")
    c2.metric("Columns", f"{len(filtered_df.columns):,}")
    c3.metric(
        "Missing cells",
        f"{int(filtered_df.isna().sum().sum()):,}",
    )

    st.markdown("### Download reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>Filtered dataset</h4>
                <p class="muted">Download the currently selected records and columns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Export filtered CSV",
            data=csv_bytes(filtered_df),
            file_name="student_filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>Analysis summary</h4>
                <p class="muted">Export row counts, missing values and numeric statistics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Export analysis summary",
            data=csv_bytes(build_summary(filtered_df)),
            file_name="student_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### Column profile")
    profile = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data type": [str(filtered_df[c].dtype) for c in filtered_df.columns],
        "Non-null values": [
            int(filtered_df[c].notna().sum())
            for c in filtered_df.columns
        ],
        "Missing values": [
            int(filtered_df[c].isna().sum())
            for c in filtered_df.columns
        ],
        "Unique values": [
            int(filtered_df[c].nunique(dropna=True))
            for c in filtered_df.columns
        ],
    })

    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Export column profile",
        data=csv_bytes(profile),
        file_name="student_column_profile.csv",
        mime="text/csv",
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0 22px 0;">
        <div style="color:#93c5fd;font-size:12px;font-weight:800;letter-spacing:1.5px;">
            STUDENT PERFORMANCE ANALYTICS
        </div>
        <div style="color:#8da7c8;font-size:11px;margin-top:6px;">
            Built with Python · Streamlit · Pandas · Plotly · Scikit-learn
        </div>
        <div style="color:#7188a5;font-size:10px;margin-top:6px;">
            Predictions are estimates and should be interpreted alongside
            data quality, feature availability and model evaluation results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
