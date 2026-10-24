import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Student Analytics", page_icon="🎓", layout="wide")

BLUE = "#2563EB"
DARK = "#102A56"
BG = "#F4F7FC"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{font-family:Inter,sans-serif;}}
.stApp {{background:{BG};}}
section[data-testid="stSidebar"] {{background:white;border-right:1px solid #E6ECF5;}}
.hero {{background:linear-gradient(120deg,{DARK},#1D4ED8 65%,#60A5FA);border-radius:18px;padding:27px 30px;color:white;margin-bottom:18px;}}
.hero h1 {{font-size:30px;color:white;margin:0 0 8px 0;}}
.hero p {{color:#DBEAFE;margin:0;font-size:14px;}}
.metric-card {{background:white;border:1px solid #E6ECF5;border-radius:15px;padding:17px 18px;min-height:105px;}}
.metric-label {{color:#64748B;font-size:12px;font-weight:600;margin-bottom:8px;}}
.metric-value {{color:#0F172A;font-size:27px;font-weight:800;line-height:1.1;}}
.metric-foot {{color:#64748B;font-size:11px;margin-top:8px;}}
.section-title {{font-size:19px;font-weight:750;color:#0F172A;margin:8px 0 2px;}}
.section-subtitle {{font-size:12px;color:#64748B;margin-bottom:12px;}}
.pill {{display:inline-block;background:#DBEAFE;color:#1D4ED8;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:700;}}
.info-card {{background:white;border:1px solid #E6ECF5;border-radius:14px;padding:18px;}}
</style>
""", unsafe_allow_html=True)

def demo_data():
    return pd.DataFrame({
        "Student": ["Aman", "Priya", "Rahul", "Neha", "Rohit", "Sara", "Kabir", "Isha", "Dev", "Meera"],
        "Math": [78, 92, 65, 88, 55, 84, 73, 96, 61, 89],
        "Reading": [82, 95, 70, 90, 60, 87, 75, 94, 66, 91],
        "Writing": [80, 94, 68, 91, 58, 89, 72, 97, 63, 93],
    })

def metric_card(label, value, foot=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)

def section_title(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def load_csv(file):
    raw = file.getvalue()
    error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
        except Exception as exc:
            error = exc
    raise ValueError(f"Could not read CSV: {error}")

def numeric_cols(frame):
    return frame.select_dtypes(include="number").columns.tolist()

def score_cols(frame):
    nums = numeric_cols(frame)
    preferred = [c for c in nums if any(w in str(c).lower() for w in ("score", "marks", "math", "reading", "writing", "science", "english", "average", "grade"))]
    return preferred or nums

with st.sidebar:
    st.markdown("## 🎓 Student Analytics")
    st.caption("Performance Analysis System")
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Data Explorer", "Subject Analysis", "Data Quality", "Prediction (Coming Soon)"])
    st.divider()
    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    st.caption("Upload your own dataset to replace demo data.")
    st.divider()
    st.markdown("**How to use**")
    st.caption("1. Upload CSV\n2. Explore records and charts\n3. Check data quality\n4. Add ML prediction after validating the target column")

if uploaded:
    try:
        df = load_csv(uploaded)
        df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
        source = uploaded.name
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    df = demo_data()
    source = "Built-in demo data"

if df.empty:
    st.error("This CSV has no rows. Please upload a CSV with student records.")
    st.stop()

for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        count = df[col].notna().sum()
        if count and converted.notna().sum() / count >= 0.85:
            df[col] = converted

scores = score_cols(df)
if len(scores) >= 2 and "Average Score" not in df.columns:
    df["Average Score"] = df[scores].mean(axis=1).round(2)

with st.sidebar:
    st.markdown("### 🔎 Filters")
    filterable = [c for c in df.columns if c not in numeric_cols(df) and 1 < df[c].nunique(dropna=True) <= 20]
    filters = {}
    for col in filterable[:3]:
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        filters[col] = st.multiselect(col, opts, default=opts, key=f"filter_{col}")

view = df.copy()
for col, opts in filters.items():
    view = view[view[col].astype(str).isin(opts)]
if view.empty:
    st.warning("No rows match the selected filters. Clear or change the sidebar filters.")
    st.stop()

st.markdown(f"""
<div class="hero">
<span class="pill">STUDENT PERFORMANCE INTELLIGENCE</span>
<h1>Performance Dashboard</h1>
<p>Understand student outcomes with clear metrics, interactive charts, and data-quality checks.</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"Data source: **{source}** · Showing **{len(view):,}** of **{len(df):,}** records")

if page == "Overview":
    candidates = [c for c in score_cols(view) if c in view.columns]
    main_score = "Average Score" if "Average Score" in view.columns else (candidates[0] if candidates else None)
    if main_score:
        avg, high, low = view[main_score].mean(), view[main_score].max(), view[main_score].min()
    else:
        avg = high = low = np.nan
    a,b,c,d = st.columns(4)
    with a: metric_card("TOTAL RECORDS", f"{len(view):,}", "Rows in current view")
    with b: metric_card("AVERAGE SCORE", f"{avg:.1f}" if pd.notna(avg) else "N/A", main_score or "No numeric score column")
    with c: metric_card("HIGHEST SCORE", f"{high:.1f}" if pd.notna(high) else "N/A", "Highest selected score")
    with d: metric_card("LOWEST SCORE", f"{low:.1f}" if pd.notna(low) else "N/A", "Lowest selected score")
    st.write("")
    left,right = st.columns([1.3,1])
    with left:
        section_title("Score distribution", "How scores are spread across the selected records")
        if main_score:
            fig = px.histogram(view, x=main_score, nbins=15, color_discrete_sequence=[BLUE])
            fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=15,b=10))
            fig.update_yaxes(gridcolor="#E8EEF7")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Upload a dataset with numeric score columns.")
    with right:
        section_title("Average by subject", "Mean value for detected score columns")
        subjects = [c for c in candidates if c != "Average Score"]
        if subjects:
            means = view[subjects].mean().sort_values().reset_index()
            means.columns = ["Subject","Average"]
            fig = px.bar(means, x="Average", y="Subject", orientation="h", text_auto=".1f", color_discrete_sequence=[BLUE])
            fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=15,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No numeric score columns detected.")
    section_title("Student records", "Preview of the current data")
    st.dataframe(view.head(15), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download current view as CSV", view.to_csv(index=False).encode("utf-8"), "student_analysis_view.csv", "text/csv")

elif page == "Data Explorer":
    section_title("Data Explorer", "Inspect the dataset and review its columns.")
    a,b,c = st.columns(3)
    with a: metric_card("ROWS", f"{len(view):,}", "Filtered records")
    with b: metric_card("COLUMNS", f"{view.shape[1]:,}", "Available fields")
    with c: metric_card("NUMERIC COLUMNS", f"{len(numeric_cols(view)):,}", "Numeric fields")
    search = st.text_input("Search column names", placeholder="e.g. math, score, gender")
    cols = [c for c in view.columns if search.lower() in str(c).lower()] if search else list(view.columns)
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data", view.to_csv(index=False).encode("utf-8"), "student_data_filtered.csv", "text/csv")
    summary = pd.DataFrame({"Column": view.columns, "Data type": [str(view[c].dtype) for c in view.columns], "Non-empty": [int(view[c].notna().sum()) for c in view.columns], "Missing": [int(view[c].isna().sum()) for c in view.columns], "Unique values": [int(view[c].nunique(dropna=True)) for c in view.columns]})
    section_title("Column summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Subject Analysis":
    section_title("Subject Analysis", "Compare numeric columns and explore score relationships.")
    cols = score_cols(view)
    chosen = st.multiselect("Choose columns to compare", cols, default=cols[:min(4,len(cols))])
    if chosen:
        means = view[chosen].mean().sort_values(ascending=False).reset_index()
        means.columns = ["Column","Mean"]
        fig = px.bar(means, x="Column", y="Mean", text_auto=".2f", color="Mean", color_continuous_scale="Blues")
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        if len(chosen) >= 2:
            l,r = st.columns(2)
            with l:
                xcol = st.selectbox("X-axis", chosen, index=0)
                ycol = st.selectbox("Y-axis", chosen, index=1)
                fig = px.scatter(view, x=xcol, y=ycol, trendline="ols", opacity=.75, color_discrete_sequence=[BLUE])
                fig.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            with r:
                section_title("Correlation matrix")
                fig = px.imshow(view[chosen].corr(numeric_only=True), text_auto=".2f", color_continuous_scale="Blues", aspect="auto", zmin=-1, zmax=1)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        section_title("Score statistics")
        st.dataframe(view[chosen].describe().T.round(2), use_container_width=True)

elif page == "Data Quality":
    section_title("Data Quality", "Review missing values and duplicate records.")
    missing = view.isna().sum()
    a,b,c = st.columns(3)
    with a: metric_card("MISSING CELLS", f"{int(missing.sum()):,}", "Across all columns")
    with b: metric_card("DUPLICATE ROWS", f"{int(view.duplicated().sum()):,}", "Exact duplicate records")
    with c: metric_card("MISSINGNESS", f"{missing.sum()/max(view.size,1)*100:.1f}%", "Percent of cells missing")
    quality = pd.DataFrame({"Column":view.columns,"Missing values":[int(view[c].isna().sum()) for c in view.columns],"Missing %":[round(view[c].isna().mean()*100,2) for c in view.columns],"Unique values":[int(view[c].nunique(dropna=True)) for c in view.columns],"Data type":[str(view[c].dtype) for c in view.columns]}).sort_values("Missing %", ascending=False)
    st.dataframe(quality, use_container_width=True, hide_index=True)
    l,r = st.columns(2)
    with l:
        section_title("Missing values by column")
        miss = quality[quality["Missing values"] > 0]
        if len(miss):
            fig = px.bar(miss, x="Missing values", y="Column", orientation="h", color_discrete_sequence=["#F59E0B"])
            fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else: st.success("No missing values found in this view.")
    with r:
        section_title("Create a cleaned copy")
        st.caption("The original file is not changed. Numeric missing values use median; text values use mode.")
        if st.button("Prepare cleaned CSV", type="primary"):
            clean = view.drop_duplicates().copy()
            for col in numeric_cols(clean):
                if clean[col].isna().any(): clean[col] = clean[col].fillna(clean[col].median())
            for col in clean.columns:
                if clean[col].dtype == "object" and clean[col].isna().any():
                    mode = clean[col].mode(dropna=True)
                    if not mode.empty: clean[col] = clean[col].fillna(mode.iloc[0])
            st.session_state["clean_csv"] = clean.to_csv(index=False).encode("utf-8")
        if "clean_csv" in st.session_state:
            st.download_button("⬇️ Download cleaned CSV", st.session_state["clean_csv"], "student_data_cleaned.csv", "text/csv")

else:
    section_title("Prediction Module", "Machine learning will be added after confirming the real dataset and target column.")
    st.markdown("""
    <div class="info-card">
      <span class="pill">PHASE 2</span>
      <h3>Build a reliable prediction model</h3>
      <p style="color:#475569">We will first confirm which fields are valid features and what outcome should be predicted.</p>
      <ul style="color:#475569">
        <li>Choose a target column, such as a final score if available.</li>
        <li>Handle missing values and encode categorical fields.</li>
        <li>Split data into training and testing sets.</li>
        <li>Compare models and report evaluation metrics.</li>
      </ul>
      <p style="color:#64748B;font-size:12px">No predictions are generated in this version; this avoids showing unvalidated results.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload your real CSV and inspect its columns in Data Explorer. Then we can add an appropriate ML model.")

st.markdown('<p style="text-align:center;color:#94A3B8;font-size:11px;padding:20px 0">Student Analytics · Python · Streamlit · Pandas · Plotly</p>', unsafe_allow_html=True)
