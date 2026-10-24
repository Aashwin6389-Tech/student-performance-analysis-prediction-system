
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081426 0%, #101f38 55%, #142943 100%);
    color: #F1F5F9;
}
[data-testid="stHeader"] {
    background: rgba(8,20,38,0.96);
}
[data-testid="stSidebar"] {
    background: #0B1729;
    border-right: 1px solid #263B55;
}
[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] span,
[data-testid="stMain"] li,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #F1F5F9;
}
h1 {
    font-size: 2.25rem !important;
    font-weight: 800 !important;
}
h2, h3 {
    font-weight: 700 !important;
}
[data-testid="stMetric"] {
    background: #172A43;
    border: 1px solid #344B67;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}
[data-testid="stMetricValue"] {
    color: #67E8F9 !important;
}
div[data-baseweb="select"] > div {
    background: #F8FAFC !important;
    border: 1px solid #94A3B8 !important;
}
div[data-baseweb="select"] * {
    color: #111827 !important;
}
div[data-baseweb="popover"] {
    background: #FFFFFF !important;
}
div[data-baseweb="menu"] {
    background: #FFFFFF !important;
}
div[data-baseweb="menu"] * {
    color: #111827 !important;
}
input, textarea {
    background: #FFFFFF !important;
    color: #111827 !important;
}
[data-testid="stFileUploader"] {
    background: #172A43;
    border: 1px dashed #67E8F9;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stFileUploader"] * {
    color: #F1F5F9 !important;
}
.stButton button, .stDownloadButton button {
    background: #0891B2;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #0E7490;
    color: #FFFFFF !important;
    border: 1px solid #67E8F9;
}
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border-radius: 10px;
}
hr {
    border-color: #344B67;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
st.sidebar.title("🎓 Student Analytics")
st.sidebar.caption("Academic Performance Dashboard")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload student CSV",
    type=["csv"]
)

possible_paths = [
    Path("data/Cleaned_Students_Performance.csv"),
    Path("Cleaned_Students_Performance.csv")
]

df = None
source_name = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Unable to read uploaded CSV: {e}")
        st.stop()
else:
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                source_name = str(path)
                break
            except Exception:
                pass

if df is None:
    st.error(
        "Student CSV not found. Upload your CSV using the sidebar, "
        "or check that it exists inside the data folder."
    )
    st.stop()

# ---------------- CLEAN DATA ----------------
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

# Detect score columns automatically
score_keywords = [
    "score", "marks", "grade", "g1", "g2", "g3",
    "math", "reading", "writing", "exam", "performance"
]

score_cols = []
for col in df.columns:
    name = col.lower().strip()
    if any(word in name for word in score_keywords):
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
            score_cols.append(col)

# Numeric columns useful for analysis
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Exclude identifiers from average score if possible
excluded = ["id", "student_id", "roll_no", "roll_number"]
analysis_numeric = [
    c for c in numeric_cols
    if c.lower() not in excluded
]

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.subheader("🔎 Filters")
filtered = df.copy()

categorical_cols = [
    c for c in df.columns
    if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 30
]

for col in categorical_cols[:4]:
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    if values:
        selected = st.sidebar.multiselect(
            col.replace("_", " ").title(),
            values,
            default=values,
            key=f"filter_{col}"
        )
        filtered = filtered[
            filtered[col].astype(str).isin(selected)
        ]

st.sidebar.divider()
st.sidebar.caption(f"Data source: {source_name}")
st.sidebar.caption("Built with Python, Pandas, Plotly & Streamlit")

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Analysis")
st.markdown(
    "### Academic insights • Interactive analytics • Data-driven decisions"
)
st.markdown(
    "Explore student records, compare academic indicators, "
    "identify data-quality issues and download filtered data."
)
st.divider()

# ---------------- NAVIGATION ----------------
page = st.radio(
    "Choose a section",
    [
        "Overview",
        "Data Explorer",
        "Performance Analysis",
        "Data Quality"
    ],
    horizontal=True
)

st.divider()

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("📊 Dashboard Overview")

    total_students = len(filtered)
    total_columns = len(filtered.columns)
    missing_cells = int(filtered.isna().sum().sum())
    duplicate_count = int(filtered.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Records", f"{total_students:,}")
    c2.metric("Data Features", total_columns)
    c3.metric("Missing Values", f"{missing_cells:,}")
    c4.metric("Duplicate Rows", f"{duplicate_count:,}")

    st.markdown("### 📈 Dataset Summary")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Dataset Preview")
        st.dataframe(
            filtered.head(10),
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.markdown("#### Missing Values by Column")
        missing = filtered.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            missing_df = missing.rename("Missing Values").reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                color="Missing Values",
                color_continuous_scale="Teal",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F1F5F9"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the filtered data.")

    if analysis_numeric:
        st.markdown("#### Numeric Feature Summary")
        st.dataframe(
            filtered[analysis_numeric].describe().round(2),
            use_container_width=True
        )

# ---------------- DATA EXPLORER ----------------
elif page == "Data Explorer":
    st.subheader("🗂️ Explore Student Data")

    st.write(
        f"Showing **{len(filtered):,}** records "
        f"and **{len(filtered.columns)}** columns."
    )

    search = st.text_input(
        "Search across the displayed data",
        placeholder="Type a name, school, gender or value..."
    )

    display_df = filtered.copy()

    if search:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )

# ---------------- PERFORMANCE ANALYSIS ----------------
elif page == "Performance Analysis":
    st.subheader("📈 Academic & Numeric Analysis")

    if not numeric_cols:
        st.warning(
            "No numeric columns found. Check the uploaded CSV format."
        )
    else:
        chosen_col = st.selectbox(
            "Select a numeric feature",
            numeric_cols,
            index=(
                numeric_cols.index(score_cols[0])
                if score_cols and score_cols[0] in numeric_cols
                else 0
            )
        )

        values = pd.to_numeric(
            filtered[chosen_col], errors="coerce"
        ).dropna()

        if values.empty:
            st.warning("No valid numeric values available for this feature.")
        else:
            a, b, c = st.columns(3)
            a.metric("Average", f"{values.mean():.2f}")
            b.metric("Median", f"{values.median():.2f}")
            c.metric("Standard Deviation", f"{values.std():.2f}" if len(values) > 1 else "N/A")

            left, right = st.columns(2)

            with left:
                st.markdown("#### Distribution")
                fig = px.histogram(
                    filtered,
                    x=chosen_col,
                    nbins=20,
                    title=f"Distribution of {chosen_col}",
                    color_discrete_sequence=["#22D3EE"],
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F1F5F9"
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("#### Box Plot")
                fig = px.box(
                    filtered,
                    y=chosen_col,
                    title=f"Spread of {chosen_col}",
                    color_discrete_sequence=["#A78BFA"],
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F1F5F9"
                )
                st.plotly_chart(fig, use_container_width=True)

        if len(score_cols) >= 2:
            st.markdown("#### Compare Academic Indicators")
            comparison = filtered[score_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            comparison = comparison.mean().reset_index()
            comparison.columns = ["Feature", "Average"]

            fig = px.bar(
                comparison,
                x="Feature",
                y="Average",
                color="Average",
                color_continuous_scale="Teal",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F1F5F9"
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown("#### Feature Correlation")
            corr = filtered[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Between Numeric Features",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F1F5F9"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA QUALITY ----------------
elif page == "Data Quality":
    st.subheader("🧹 Data Quality Report")

    quality = pd.DataFrame({
        "Column": filtered.columns,
        "Data Type": [str(filtered[c].dtype) for c in filtered.columns],
        "Missing Values": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "Missing (%)": [
            round(filtered[c].isna().mean() * 100, 2)
            for c in filtered.columns
        ],
        "Unique Values": [
            int(filtered[c].nunique(dropna=True))
            for c in filtered.columns
        ]
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Missing Values Chart")
    missing = filtered.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        missing_df = missing.rename("Count").reset_index()
        missing_df.columns = ["Column", "Count"]
        fig = px.bar(
            missing_df,
            x="Column",
            y="Count",
            color="Count",
            color_continuous_scale="Teal",
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F1F5F9"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("Great! No missing values found in the current data.")

    st.markdown("#### Duplicate Records")
    st.write(f"Duplicate rows in filtered data: **{filtered.duplicated().sum()}**")

    st.download_button(
        "⬇️ Download Data Quality Report",
        data=quality.to_csv(index=False).encode("utf-8"),
        file_name="student_data_quality_report.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Student Performance Analytics | Python • Pandas • Plotly • Streamlit"
)
