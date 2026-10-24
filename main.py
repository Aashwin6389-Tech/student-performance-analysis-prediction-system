
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# ---------- THEME ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#081426,#10243D,#172D48);
    color: #F8FAFC;
}
[data-testid="stHeader"] {background: rgba(8,20,38,.95);}
[data-testid="stSidebar"] {
    background: #0B1729;
    border-right: 1px solid #334B68;
}
[data-testid="stSidebar"] * {color:#F8FAFC !important;}
.stApp h1,.stApp h2,.stApp h3,.stApp p,
.stApp label,.stApp li {color:#F1F5F9;}
[data-testid="stMetric"] {
    background:#172A43;
    border:1px solid #344B67;
    padding:16px;
    border-radius:12px;
}
[data-testid="stMetricLabel"] {color:#CBD5E1 !important;}
[data-testid="stMetricValue"] {color:#67E8F9 !important;}
div[data-baseweb="select"] > div {
    background:#FFFFFF !important;
    color:#111827 !important;
}
div[data-baseweb="select"] * {color:#111827 !important;}
div[data-baseweb="popover"],
div[data-baseweb="menu"] {background:#FFFFFF !important;}
div[data-baseweb="menu"] * {color:#111827 !important;}
input,textarea {
    background:#FFFFFF !important;
    color:#111827 !important;
}
[data-testid="stFileUploader"] {
    background:#172A43;
    border:1px dashed #67E8F9;
    border-radius:12px;
    padding:10px;
}
[data-testid="stFileUploader"] * {color:#F8FAFC !important;}
.stButton button,.stDownloadButton button {
    background:#0891B2;
    color:#FFFFFF !important;
    border:0;
    border-radius:8px;
    font-weight:700;
}
.stButton button:hover,.stDownloadButton button:hover {
    background:#0E7490;
    color:#FFFFFF !important;
}
.block-container {padding-top:2rem;}
hr {border-color:#334B68;}
</style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
st.sidebar.title("🎓 Student Analytics")
st.sidebar.caption("Academic Performance Intelligence")

uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])

paths = [
    Path("data/Cleaned_Students_Performance.csv"),
    Path("Cleaned_Students_Performance.csv")
]

df = None
source = ""

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = uploaded.name
    else:
        for path in paths:
            if path.exists():
                df = pd.read_csv(path)
                source = str(path)
                break
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

if df is None:
    st.warning("Upload a student CSV using the sidebar.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
df = df.drop_duplicates().copy()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

# Convert numeric-looking columns safely
for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(df[col].notna().sum() * 0.8)):
            df[col] = converted

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.sidebar.caption(f"Data source: {source}")
st.sidebar.caption(f"Original dataset: {len(df):,} rows")

# ---------- FILTERS ----------
st.sidebar.divider()
st.sidebar.subheader("🔎 Data Filters")
filtered = df.copy()

cat_cols = [
    c for c in df.select_dtypes(exclude=np.number).columns
    if 1 < df[c].nunique(dropna=True) <= 25
]

for col in cat_cols[:4]:
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect(
        col.replace("_", " ").title(),
        values,
        default=values,
        key=f"filter_{col}"
    )
    filtered = filtered[filtered[col].astype(str).isin(selected)]

# ---------- HEADER ----------
st.title("🎓 Student Performance Analysis & Prediction")
st.write(
    "Interactive academic analytics, data quality monitoring "
    "and machine learning based score estimation."
)

page = st.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Performance Analysis",
     "ML Prediction", "Data Quality"],
    horizontal=True
)
st.divider()

# ---------- OVERVIEW ----------
if page == "Overview":
    st.subheader("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Records", f"{len(filtered):,}")
    c2.metric("Features", len(filtered.columns))
    c3.metric("Missing Cells", f"{filtered.isna().sum().sum():,}")
    c4.metric("Numeric Features", len(numeric_cols))

    left, right = st.columns(2)

    with left:
        st.markdown("### Dataset Preview")
        st.dataframe(filtered.head(10), use_container_width=True)

    with right:
        st.markdown("### Missing Values")
        missing = filtered.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing):
            chart_data = missing.rename("Missing").reset_index()
            chart_data.columns = ["Column", "Missing"]
            fig = px.bar(
                chart_data, x="Column", y="Missing",
                color="Missing", template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in this filtered dataset.")

    if numeric_cols:
        st.markdown("### Statistical Summary")
        st.dataframe(
            filtered[numeric_cols].describe().round(2),
            use_container_width=True
        )

# ---------- DATA EXPLORER ----------
elif page == "Data Explorer":
    st.subheader("🗂️ Student Data Explorer")
    query = st.text_input("Search records", placeholder="Enter a value...")
    view = filtered.copy()

    if query:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)
        ).any(axis=1)
        view = view[mask]

    st.write(f"Records displayed: **{len(view):,}**")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="student_data.csv",
        mime="text/csv"
    )

# ---------- PERFORMANCE ANALYSIS ----------
elif page == "Performance Analysis":
    st.subheader("📈 Performance & Feature Analysis")

    if not numeric_cols:
        st.warning("No numeric columns are available.")
    else:
        feature = st.selectbox("Choose a numeric feature", numeric_cols)
        values = pd.to_numeric(filtered[feature], errors="coerce").dropna()

        if len(values):
            a, b, c = st.columns(3)
            a.metric("Mean", f"{values.mean():.2f}")
            b.metric("Median", f"{values.median():.2f}")
            c.metric("Std. Deviation", f"{values.std():.2f}")

            left, right = st.columns(2)

            with left:
                fig = px.histogram(
                    filtered, x=feature, nbins=20,
                    title=f"Distribution: {feature}",
                    template="plotly_dark",
                    color_discrete_sequence=["#22D3EE"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC"
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                fig = px.box(
                    filtered, y=feature,
                    title=f"Score Spread: {feature}",
                    template="plotly_dark",
                    color_discrete_sequence=["#A78BFA"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F8FAFC"
                )
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Between Features")
            corr = filtered[numeric_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- MACHINE LEARNING ----------
elif page == "ML Prediction":
    st.subheader("🤖 Student Score Prediction")
    st.write(
        "Train a Random Forest regression model using your dataset, "
        "evaluate it on a held-out test set, and estimate a score "
        "from selected numeric inputs."
    )

    if len(filtered) < 10:
        st.warning("At least 10 filtered records are recommended for training.")
    else:
        # Avoid derived totals/averages as prediction inputs
        derived_words = ["total", "average", "overall", "final_score"]
        candidates = [
            c for c in numeric_cols
            if c.lower() not in ["id", "student_id", "roll_no", "roll_number"]
        ]

        targets = [
            c for c in candidates
            if any(k in c.lower() for k in
                   ["score", "marks", "grade", "g1", "g2", "g3", "math",
                    "reading", "writing", "exam"])
            and not any(k in c.lower() for k in derived_words)
        ]

        if not targets:
            targets = candidates

        if len(targets) < 1:
            st.warning("No suitable numeric target column was found.")
        else:
            default_target = (
                "G3" if "G3" in targets else
                "writing_score" if "writing_score" in targets else
                targets[-1]
            )
            target = st.selectbox(
                "Choose the score to predict",
                targets,
                index=targets.index(default_target)
            )

            excluded = {
                target, "id", "student_id", "roll_no", "roll_number"
            }
            excluded.update(
                c for c in candidates
                if any(k in c.lower() for k in derived_words)
            )

            features = [c for c in candidates if c not in excluded]

            if not features:
                st.warning(
                    "No independent numeric input features remain. "
                    "Choose a different target or inspect the CSV."
                )
            else:
                st.caption("Model inputs: " + ", ".join(features))

                model_data = filtered[features + [target]].copy()
                for col in features + [target]:
                    model_data[col] = pd.to_numeric(
                        model_data[col], errors="coerce"
                    )
                model_data = model_data.replace([np.inf, -np.inf], np.nan)
                model_data = model_data.dropna()

                if len(model_data) < 10:
                    st.warning(
                        "Not enough complete rows for training. "
                        "Check missing values or choose another target."
                    )
                else:
                    X = model_data[features]
                    y = model_data[target]

                    if y.nunique() < 2:
                        st.warning("The selected target has too little variation.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model = RandomForestRegressor(
                            n_estimators=150,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Test Records", len(y_test))
                        m2.metric("Mean Absolute Error", f"{mae:.2f}")
                        m3.metric("Test R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

                        st.caption(
                            "MAE is the average absolute prediction error. "
                            "R² may be negative on small or difficult datasets. "
                            "These metrics do not guarantee future accuracy."
                        )

                        st.markdown("### Predict for a New Student")
                        input_values = {}

                        with st.form("prediction_form"):
                            cols = st.columns(2)
                            for i, col in enumerate(features):
                                series = X[col]
                                lo = float(series.min())
                                hi = float(series.max())
                                default = float(series.median())
                                step = max((hi - lo) / 100, 0.1)

                                with cols[i % 2]:
                                    input_values[col] = st.number_input(
                                        col.replace("_", " ").title(),
                                        min_value=lo,
                                        max_value=hi,
                                        value=default,
                                        step=step,
                                        key=f"input_{col}"
                                    )

                            submitted = st.form_submit_button(
                                "Predict Student Score"
                            )

                        if submitted:
                            new_student = pd.DataFrame(
                                [input_values], columns=features
                            )
                            estimate = float(model.predict(new_student)[0])
                            st.success(
                                f"Estimated {target}: {estimate:.2f}"
                            )
                            st.caption(
                                "This is a model-based estimate, not a "
                                "guaranteed academic outcome."
                            )

                        result_df = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": predictions
                        })
                        fig = px.scatter(
                            result_df, x="Actual", y="Predicted",
                            title="Actual vs Predicted Scores",
                            template="plotly_dark",
                            color_discrete_sequence=["#22D3EE"]
                        )
                        low = min(result_df.min().min(), result_df.max().min())
                        high = max(result_df.max().max(), result_df.max().max())
                        fig.add_shape(
                            type="line", x0=low, y0=low, x1=high, y1=high,
                            line=dict(color="#FBBF24", dash="dash")
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#F8FAFC"
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ---------- DATA QUALITY ----------
elif page == "Data Quality":
    st.subheader("🧹 Data Quality Report")

    quality = pd.DataFrame({
        "Column": filtered.columns,
        "Type": [str(filtered[c].dtype) for c in filtered.columns],
        "Missing": [int(filtered[c].isna().sum()) for c in filtered.columns],
        "Missing %": [
            round(float(filtered[c].isna().mean() * 100), 2)
            for c in filtered.columns
        ],
        "Unique Values": [
            int(filtered[c].nunique(dropna=True)) for c in filtered.columns
        ]
    })

    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.write(f"Duplicate rows: **{filtered.duplicated().sum()}**")

    st.download_button(
        "⬇️ Download Quality Report",
        quality.to_csv(index=False).encode("utf-8"),
        file_name="data_quality_report.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Student Performance Analytics | Python • Pandas • Plotly • "
    "Scikit-learn • Streamlit"
)
