# main.py
# Student Performance Analysis & Prediction System
# Professional Streamlit Data Science Project

import io
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StudentIQ | Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #08111f;
        color: #e8eef8;
    }

    [data-testid="stSidebar"] {
        background: #0d192a;
        border-right: 1px solid #1e3048;
    }

    [data-testid="stSidebar"] * {
        color: #dce7f7;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #f1f6ff !important;
        letter-spacing: -0.5px;
    }

    p, label, span, li {
        color: #d1dced;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #12233a, #101d30);
        border: 1px solid #233a56;
        padding: 19px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #9fb4d0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f8ff !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        color: #54d6a0 !important;
    }

    .hero {
        background: linear-gradient(120deg, #122a49, #15233d 60%, #102d43);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #294968;
        margin-bottom: 24px;
    }

    .hero-label {
        color: #65d7ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin: 8px 0;
    }

    .hero-subtitle {
        color: #b8c9df;
        font-size: 14px;
        line-height: 1.7;
        max-width: 850px;
    }

    .section-label {
        color: #67c8ff;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1.8px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .info-card {
        background: #101e31;
        border: 1px solid #233a56;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }

    .info-card h4 {
        margin: 0 0 9px 0;
        color: #f1f6ff;
        font-size: 15px;
    }

    .info-card p {
        color: #aebfd4;
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .muted {
        color: #96abc5;
        font-size: 13px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #315a80;
        min-height: 42px;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #087ec4, #246de0);
        color: white;
        border: none;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #101e31;
        border-color: #2b405c;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #253a55;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #22354d;
    }

    .footer {
        text-align: center;
        color: #8298b4;
        font-size: 12px;
        padding: 20px 0 0 0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_TEMPLATE = "plotly_dark"
CHART_COLORS = [
    "#38bdf8",
    "#818cf8",
    "#34d399",
    "#fbbf24",
    "#fb7185",
    "#c084fc",
    "#2dd4bf",
]


# ============================================================
# DATA LOADING
# ============================================================

DEFAULT_DATA_PATHS = [
    "data/Cleaned_Students_Performance.csv",
    "Cleaned_Students_Performance.csv",
    "data/student_performance.csv",
    "student_performance.csv",
]


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]
    return df


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes):
    last_error = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return normalize_columns(
                pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            )
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not read the CSV file: {last_error}")


@st.cache_data(show_spinner=False)
def load_default_dataset(path, modified_time):
    del modified_time
    return normalize_columns(pd.read_csv(path))


def load_data():
    uploaded_file = st.session_state.get("uploaded_dataset")

    if uploaded_file is not None:
        try:
            return read_csv_bytes(uploaded_file.getvalue()), uploaded_file.name
        except Exception as exc:
            st.error(f"Unable to read uploaded dataset: {exc}")
            return None, None

    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            try:
                return (
                    load_default_dataset(path, os.path.getmtime(path)),
                    os.path.basename(path),
                )
            except Exception as exc:
                st.error(f"Unable to load {path}: {exc}")

    return None, None


# ============================================================
# DATA HELPERS
# ============================================================

def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


def get_score_columns(df):
    numeric_cols = get_numeric_columns(df)
    preferred = [
        c for c in numeric_cols
        if any(
            token in c.lower()
            for token in [
                "score", "mark", "grade", "math", "reading",
                "writing", "exam", "g1", "g2", "g3", "result"
            ]
        )
        and not any(
            token in c.lower()
            for token in ["id", "roll", "index", "student_number"]
        )
    ]
    return preferred or numeric_cols


def identify_target_candidates(df):
    candidates = get_score_columns(df)

    derived_hints = [
        "total_score", "total_marks", "average_score",
        "average_marks", "overall_score", "overall_average",
        "final_score", "final_result", "percentage",
    ]

    candidates = [
        c for c in candidates
        if not any(hint in c.lower() for hint in derived_hints)
    ]

    return candidates


def numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    return numeric.describe().T.reset_index().rename(
        columns={"index": "Feature"}
    )


def make_download_button(label, data, filename, mime, key):
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def fig_style(fig, height=390):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d4e1f2", size=12),
        title_font=dict(size=16, color="#f1f6ff"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(
        gridcolor="#20334a",
        zerolinecolor="#20334a",
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="#20334a",
        zerolinecolor="#20334a",
        automargin=True,
    )
    return fig


def make_profile(df):
    rows = []

    for col in df.columns:
        series = df[col]
        rows.append({
            "Column": col,
            "Data Type": str(series.dtype),
            "Non-Null": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(series.isna().mean() * 100, 2),
            "Unique Values": int(series.nunique(dropna=True)),
        })

    return pd.DataFrame(rows)


def get_cleaned_copy(df, remove_duplicates=False, fill_missing=False):
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if fill_missing:
        for col in cleaned.columns:
            if cleaned[col].isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned[col]):
                median = cleaned[col].median()
                if pd.notna(median):
                    cleaned[col] = cleaned[col].fillna(median)
            else:
                mode = cleaned[col].mode(dropna=True)
                if not mode.empty:
                    cleaned[col] = cleaned[col].fillna(mode.iloc[0])

    return cleaned


def format_number(value, digits=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except (ValueError, TypeError):
        return str(value)


# ============================================================
# MODEL HELPERS
# ============================================================

def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_regression_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=1,
            n_jobs=-1,
        )

    if name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        )

    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42)

    if name == "Ridge Regression":
        return Ridge(alpha=1.0)

    return LinearRegression()


@st.cache_data(show_spinner=False)
def train_regression_models(df, target, features, test_size, model_names):
    model_data = df[features + [target]].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[target])

    if len(model_data) < 10:
        raise ValueError(
            "At least 10 rows with a valid target are recommended for model training."
        )

    if model_data[target].nunique() < 2:
        raise ValueError("The target must contain at least two distinct values.")

    X = model_data[features].copy()
    y = model_data[target].copy()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    usable_features = []

    for col in numeric_features:
        if X[col].notna().sum() > 0:
            usable_features.append(col)

    for col in categorical_features:
        if X[col].notna().sum() > 0:
            usable_features.append(col)

    if not usable_features:
        raise ValueError(
            "No usable feature columns remain after removing all-empty columns."
        )

    X = X[usable_features]

    if len(X) < 10:
        raise ValueError("Not enough valid rows for a train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    results = []
    fitted_models = {}
    predictions = {}

    for model_name in model_names:
        try:
            preprocessor = make_preprocessor(X_train)
            estimator = make_regression_model(model_name)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            try:
                r2 = r2_score(y_test, y_pred)
            except Exception:
                r2 = np.nan

            results.append({
                "Model": model_name,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R² Score": float(r2),
            })

            fitted_models[model_name] = pipeline
            predictions[model_name] = np.asarray(y_pred)

        except Exception as exc:
            results.append({
                "Model": model_name,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R² Score": np.nan,
                "Error": str(exc),
            })

    metrics = pd.DataFrame(results)

    if "MAE" in metrics.columns:
        valid = metrics.dropna(subset=["MAE"])
        if not valid.empty:
            best_model_name = valid.sort_values("MAE").iloc[0]["Model"]
        else:
            best_model_name = None
    else:
        best_model_name = None

    return {
        "metrics": metrics,
        "models": fitted_models,
        "predictions": predictions,
        "X_test": X_test,
        "y_test": y_test,
        "features": usable_features,
        "best_model": best_model_name,
        "row_count": len(model_data),
    }


def get_feature_importance(pipeline, feature_names):
    try:
        estimator = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        encoded_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance = np.abs(np.asarray(estimator.coef_)).ravel()
        else:
            return pd.DataFrame()

        if len(encoded_names) != len(importance):
            return pd.DataFrame()

        output = pd.DataFrame({
            "Feature": encoded_names,
            "Importance": importance,
        })

        output["Feature"] = (
            output["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
            .str.replace("onehot__", "", regex=False)
        )

        return output.sort_values(
            "Importance", ascending=False
        ).head(20)

    except Exception:
        return pd.DataFrame()


# ============================================================
# HEADER AND SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:8px 0 20px 0">
        <div style="font-size:12px;letter-spacing:2px;color:#5cc9ff;font-weight:800">
            STUDENTIQ
        </div>
        <div style="font-size:22px;font-weight:800;color:#ffffff;margin-top:5px">
            Performance<br>Intelligence
        </div>
        <div style="font-size:12px;color:#91a8c5;margin-top:8px">
            Academic Analytics Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Data Explorer",
        "Performance Analytics",
        "ML Prediction",
        "Student Risk",
        "Data Quality",
        "Export Center",
    ],
    label_visibility="visible",
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv"],
    key="uploaded_dataset",
    help="Upload a CSV file containing student data.",
)

st.sidebar.markdown("### Data filters")

df_original, dataset_name = load_data()

if df_original is None or df_original.empty:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">StudentIQ • Setup</div>
            <div class="hero-title">Welcome to StudentIQ</div>
            <div class="hero-subtitle">
                Your academic performance intelligence workspace.
                Upload a CSV dataset to begin exploring, analyzing and
                predicting student performance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "No default dataset was found. Upload your student CSV using "
        "the sidebar, or add your file to "
        "`data/Cleaned_Students_Performance.csv` in your repository."
    )

    st.markdown("### Recommended dataset columns")
    st.write(
        "The app works best with a mix of numeric and categorical columns, "
        "such as math_score, reading_score, writing_score, gender, "
        "parental_level_of_education, lunch, or test_preparation_course."
    )
    st.stop()

if len(df_original.columns) == 0:
    st.error("The dataset has no columns.")
    st.stop()

if len(df_original) == 0:
    st.error("The dataset has no rows.")
    st.stop()

df_original = normalize_columns(df_original)

if df_original.columns.duplicated().any():
    st.error(
        "Your dataset contains duplicate column names. Please rename "
        "duplicate columns and upload it again."
    )
    st.stop()

categorical_columns = get_categorical_columns(df_original)
numeric_columns = get_numeric_columns(df_original)

filters = {}

with st.sidebar.expander("Categorical filters", expanded=False):
    for col in categorical_columns[:15]:
        values = (
            df_original[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        values = sorted(values)

        if 0 < len(values) <= 100:
            selected = st.multiselect(
                col,
                options=values,
                default=[],
                key=f"filter_cat_{col}",
            )
            if selected:
                filters[col] = selected

with st.sidebar.expander("Numeric filters", expanded=False):
    for col in numeric_columns[:10]:
        series = df_original[col].dropna()

        if series.empty or series.nunique() < 2:
            continue

        low = float(series.min())
        high = float(series.max())

        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            continue

        selected_range = st.slider(
            col,
            min_value=low,
            max_value=high,
            value=(low, high),
            key=f"filter_num_{col}",
        )

        if selected_range != (low, high):
            filters[col] = selected_range

df = df_original.copy()

for col, selected in filters.items():
    if col in categorical_columns:
        df = df[df[col].astype(str).isin(selected)]
    elif col in numeric_columns:
        low, high = selected
        df = df[df[col].between(low, high) | df[col].isna()]

if df.empty:
    st.warning("No records match the current filters.")
    if st.sidebar.button("Clear all filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_cat_") or key.startswith("filter_num_"):
                del st.session_state[key]
        st.rerun()
    st.stop()

st.sidebar.divider()
st.sidebar.caption(f"Dataset: {dataset_name}")
st.sidebar.caption(f"Rows loaded: {len(df_original):,}")
st.sidebar.caption(f"Rows after filters: {len(df):,}")
st.sidebar.caption(f"Columns: {len(df_original.columns):,}")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-label">Academic Intelligence Platform</div>
        <div class="hero-title">Student Performance Analysis</div>
        <div class="hero-subtitle">
            Transform student data into meaningful insights with interactive
            analytics, machine learning predictions, academic risk indicators
            and exportable reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="muted" style="margin-top:-12px;margin-bottom:22px">
        Dataset: <b>{dataset_name}</b>
        &nbsp; • &nbsp; Last session: {datetime.now().strftime("%d %b %Y, %I:%M %p")}
        &nbsp; • &nbsp; Showing {len(df):,} of {len(df_original):,} records
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<div class="section-label">Executive overview</div>', unsafe_allow_html=True)
    st.header("Your academic data at a glance")
    st.write(
        "A high-level snapshot of the dataset, its completeness and the "
        "main numerical indicators."
    )

    total_records = len(df)
    total_features = len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    completeness = (
        100 * (1 - missing_cells / total_cells)
        if total_cells else 100
    )
    duplicate_count = int(df.duplicated().sum())

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Student Records", f"{total_records:,}")
    m2.metric("Data Features", f"{total_features:,}")
    m3.metric("Data Completeness", f"{completeness:.1f}%")
    m4.metric("Duplicate Rows", f"{duplicate_count:,}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Dataset composition")

        if numeric_columns:
            numeric_count = len(numeric_columns)
            categorical_count = len(categorical_columns)
            other_count = max(
                0, total_features - numeric_count - categorical_count
            )

            composition = pd.DataFrame({
                "Type": ["Numeric", "Categorical", "Other"],
                "Columns": [numeric_count, categorical_count, other_count],
            })

            fig = px.pie(
                composition,
                names="Type",
                values="Columns",
                hole=0.65,
                color_discrete_sequence=CHART_COLORS,
                title="Feature types",
            )
            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
            )
            st.plotly_chart(
                fig_style(fig, 360),
                use_container_width=True,
            )
        else:
            st.info("No numeric columns were detected in the current dataset.")

    with right:
        st.subheader("Data completeness by feature")

        profile = make_profile(df).sort_values("Missing %", ascending=False)

        fig = px.bar(
            profile.head(15).sort_values("Missing %"),
            x="Missing %",
            y="Column",
            orientation="h",
            color="Missing %",
            color_continuous_scale=["#34d399", "#38bdf8", "#fb7185"],
            title="Features with the most missing values",
        )
        st.plotly_chart(
            fig_style(fig, 360),
            use_container_width=True,
        )

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Numerical summary")

        summary = numeric_summary(df)

        if not summary.empty:
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No numerical columns available for summary.")

    with right:
        st.subheader("Quick data checks")

        checks = [
            ("Missing cells", f"{missing_cells:,}"),
            ("Duplicate rows", f"{duplicate_count:,}"),
            ("Numeric features", f"{len(numeric_columns):,}"),
            ("Categorical features", f"{len(categorical_columns):,}"),
        ]

        for label, value in checks:
            st.markdown(
                f"""
                <div class="info-card" style="margin-bottom:10px;padding:13px 16px">
                    <div style="color:#96abc5;font-size:12px">{label}</div>
                    <div style="color:#f1f6ff;font-size:22px;font-weight:800;margin-top:4px">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Preview records")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    make_download_button(
        "Download filtered dataset",
        dataframe_to_csv(df),
        "student_performance_filtered.csv",
        "text/csv",
        "overview_filtered_download",
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.markdown('<div class="section-label">Explore your dataset</div>', unsafe_allow_html=True)
    st.header("Data Explorer")
    st.write(
        "Search records, inspect columns and review descriptive statistics."
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        search_text = st.text_input(
            "Search across the dataset",
            placeholder="Enter a name, category, score or keyword...",
        )

    with c2:
        row_limit = st.selectbox(
            "Rows to display",
            [25, 50, 100, 250, 500, "All"],
            index=1,
        )

    view_df = df.copy()

    if search_text.strip():
        mask = view_df.astype(str).apply(
            lambda col: col.str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)

        view_df = view_df[mask]

    st.caption(f"{len(view_df):,} records match your search.")

    display_df = (
        view_df if row_limit == "All"
        else view_df.head(int(row_limit))
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    make_download_button(
        "Download search results",
        dataframe_to_csv(view_df),
        "student_search_results.csv",
        "text/csv",
        "explorer_download",
    )

    st.divider()

    st.subheader("Column profile")

    profile = make_profile(view_df)
    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Descriptive statistics")

    summary = numeric_summary(view_df)

    if not summary.empty:
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No numeric columns are available.")

    st.subheader("Inspect an individual feature")

    feature = st.selectbox(
        "Select a column",
        options=df.columns.tolist(),
        key="explorer_feature",
    )

    series = df[feature]

    left, right = st.columns(2)

    with left:
        st.markdown("**Column information**")
        st.write(f"**Data type:** {series.dtype}")
        st.write(f"**Non-null values:** {series.notna().sum():,}")
        st.write(f"**Missing values:** {series.isna().sum():,}")
        st.write(f"**Unique values:** {series.nunique(dropna=True):,}")

    with right:
        st.markdown("**Most frequent values**")
        frequency = (
            series.astype("string")
            .fillna("Missing")
            .value_counts()
            .head(15)
            .rename_axis("Value")
            .reset_index(name="Count")
        )
        st.dataframe(frequency, use_container_width=True, hide_index=True)


# ============================================================
# PERFORMANCE ANALYTICS
# ============================================================

elif page == "Performance Analytics":
    st.markdown('<div class="section-label">Discover patterns</div>', unsafe_allow_html=True)
    st.header("Performance Analytics")
    st.write(
        "Understand score distributions, feature relationships and "
        "differences between student groups."
    )

    if not numeric_columns:
        st.warning(
            "This page requires numeric columns. Please upload a dataset "
            "containing marks, scores or other numeric measurements."
        )
        st.stop()

    score_columns = get_score_columns(df)
    selected_score = st.selectbox(
        "Primary score feature",
        options=score_columns,
        key="analytics_score",
    )

    score_series = df[selected_score].dropna()

    if not score_series.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average", format_number(score_series.mean()))
        c2.metric("Median", format_number(score_series.median()))
        c3.metric("Minimum", format_number(score_series.min()))
        c4.metric("Maximum", format_number(score_series.max()))

    st.divider()

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x=selected_score,
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#38bdf8"],
            title=f"Distribution of {selected_score}",
        )
        st.plotly_chart(
            fig_style(fig, 420),
            use_container_width=True,
        )

    with right:
        fig = px.box(
            df,
            y=selected_score,
            points="outliers",
            color_discrete_sequence=["#818cf8"],
            title=f"Spread and outliers: {selected_score}",
        )
        st.plotly_chart(
            fig_style(fig, 420),
            use_container_width=True,
        )

    st.subheader("Relationship between two features")

    if len(numeric_columns) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_feature = st.selectbox(
                "X-axis",
                numeric_columns,
                index=0,
                key="analytics_x",
            )

        y_options = [c for c in numeric_columns if c != x_feature]

        with col2:
            y_feature = st.selectbox(
                "Y-axis",
                y_options,
                index=0,
                key="analytics_y",
            )

        color_options = ["None"] + categorical_columns
        color_feature = st.selectbox(
            "Group points by category (optional)",
            color_options,
            key="analytics_color",
        )

        color_arg = None if color_feature == "None" else color_feature

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=color_arg,
            opacity=0.8,
            hover_data=df.columns[: min(5, len(df.columns))].tolist(),
            title=f"{x_feature} vs {y_feature}",
        )

        st.plotly_chart(
            fig_style(fig, 470),
            use_container_width=True,
        )

        corr = df[[x_feature, y_feature]].corr().iloc[0, 1]

        if pd.notna(corr):
            st.info(
                f"Pearson correlation between {x_feature} and {y_feature}: "
                f"**{corr:.3f}**. Correlation indicates association, "
                "not necessarily causation."
            )
    else:
        st.info("At least two numeric features are needed for a scatter plot.")

    st.divider()
    st.subheader("Correlation matrix")

    if len(numeric_columns) >= 2:
        corr_df = df[numeric_columns].corr(numeric_only=True)

        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric feature correlations",
        )
        st.plotly_chart(
            fig_style(fig, max(420, 45 * len(numeric_columns))),
            use_container_width=True,
        )

        make_download_button(
            "Download correlation matrix",
            corr_df.to_csv().encode("utf-8-sig"),
            "correlation_matrix.csv",
            "text/csv",
            "correlation_download",
        )

    st.divider()
    st.subheader("Compare groups")

    if categorical_columns:
        group_col = st.selectbox(
            "Choose a group",
            categorical_columns,
            key="analytics_group",
        )

        group_summary = (
            df.groupby(group_col, dropna=False)[selected_score]
            .agg(["mean", "median", "count"])
            .reset_index()
            .rename(columns={
                "mean": "Average",
                "median": "Median",
                "count": "Student Count",
            })
            .sort_values("Average", ascending=False)
        )

        fig = px.bar(
            group_summary,
            x=group_col,
            y="Average",
            color="Average",
            color_continuous_scale="Blues",
            hover_data=["Median", "Student Count"],
            title=f"Average {selected_score} by {group_col}",
        )

        st.plotly_chart(
            fig_style(fig, 420),
            use_container_width=True,
        )

        st.dataframe(
            group_summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No categorical features are available for group comparison.")


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

elif page == "ML Prediction":
    st.markdown('<div class="section-label">Machine learning studio</div>', unsafe_allow_html=True)
    st.header("Performance Prediction")
    st.write(
        "Train regression models, compare evaluation metrics and estimate "
        "a numeric academic outcome from selected features."
    )

    target_candidates = identify_target_candidates(df)

    if len(numeric_columns) < 1:
        st.warning("A numeric target is required for regression.")
        st.stop()

    if not target_candidates:
        st.warning(
            "No likely academic score target was detected. Select a suitable "
            "numeric target manually below."
        )
        target_candidates = numeric_columns

    c1, c2 = st.columns([1, 1])

    with c1:
        target = st.selectbox(
            "Target to predict",
            options=target_candidates,
            key="ml_target",
            help="Choose the numeric outcome you want the model to predict.",
        )

    with c2:
        test_size = st.slider(
            "Test data percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        ) / 100

    # Prevent direct target leakage by default.
    feature_candidates = [
        c for c in df.columns
        if c != target
        and not any(
            token in c.lower()
            for token in [
                "student_id", "studentid", "roll_no", "roll_number",
                "registration_id", "record_id"
            ]
        )
    ]

    if not feature_candidates:
        st.error("No features are available after excluding the target.")
        st.stop()

    st.markdown("#### Select model features")

    default_features = [
        c for c in feature_candidates
        if not any(
            hint in c.lower()
            for hint in [
                "total_score", "total_marks", "average_score",
                "average_marks", "overall_score", "overall_average",
                "final_score", "final_result",
            ]
        )
    ]

    if not default_features:
        default_features = feature_candidates

    selected_features = st.multiselect(
        "Features used for prediction",
        options=feature_candidates,
        default=default_features,
        help=(
            "Avoid columns calculated directly from the target, since "
            "they can cause data leakage and misleading evaluation scores."
        ),
    )

    available_models = [
        "Random Forest",
        "Extra Trees",
        "Decision Tree",
        "Ridge Regression",
        "Linear Regression",
    ]

    selected_models = st.multiselect(
        "Models to compare",
        options=available_models,
        default=["Random Forest", "Linear Regression"],
    )

    if target in selected_features:
        st.error("The target cannot be included as an input feature.")
        st.stop()

    st.info(
        "Evaluation metrics are calculated on a held-out test split. "
        "Results depend on dataset size, quality, feature selection and "
        "the suitability of the target."
    )

    if st.button(
        "Train and evaluate models",
        type="primary",
        use_container_width=True,
    ):
        if not selected_features:
            st.error("Select at least one input feature.")
        elif not selected_models:
            st.error("Select at least one model.")
        else:
            with st.spinner("Training and evaluating models..."):
                try:
                    result = train_regression_models(
                        df=df,
                        target=target,
                        features=selected_features,
                        test_size=test_size,
                        model_names=selected_models,
                    )
                    st.session_state["ml_result"] = result
                    st.session_state["ml_target"] = target
                    st.session_state["ml_features"] = result["features"]
                    st.success("Model training completed.")
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

    result = st.session_state.get("ml_result")

    if result is not None and st.session_state.get("ml_target") == target:
        metrics = result["metrics"]
        fitted_models = result["models"]
        best_model_name = result["best_model"]

        st.divider()
        st.subheader("Model evaluation")

        if best_model_name:
            best_row = metrics[metrics["Model"] == best_model_name].iloc[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("Best model by MAE", best_model_name)
            m2.metric("Mean Absolute Error", format_number(best_row["MAE"]))
            m3.metric("R² Score", format_number(best_row["R² Score"]))

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        valid_metrics = metrics.dropna(subset=["MAE"])

        if not valid_metrics.empty:
            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    valid_metrics,
                    x="Model",
                    y="MAE",
                    color="Model",
                    title="Model comparison: lower MAE is better",
                    color_discrete_sequence=CHART_COLORS,
                )
                st.plotly_chart(
                    fig_style(fig, 380),
                    use_container_width=True,
                )

            with right:
                fig = px.bar(
                    valid_metrics,
                    x="Model",
                    y="R² Score",
                    color="Model",
                    title="R² score comparison",
                    color_discrete_sequence=CHART_COLORS,
                )
                st.plotly_chart(
                    fig_style(fig, 380),
                    use_container_width=True,
                )

        if best_model_name and best_model_name in fitted_models:
            y_test = result["y_test"]
            y_pred = result["predictions"][best_model_name]

            actual_vs_predicted = pd.DataFrame({
                "Actual": np.asarray(y_test),
                "Predicted": np.asarray(y_pred),
            })

            fig = px.scatter(
                actual_vs_predicted,
                x="Actual",
                y="Predicted",
                opacity=0.75,
                title=f"Actual vs predicted — {best_model_name}",
            )

            minimum = min(
                actual_vs_predicted["Actual"].min(),
                actual_vs_predicted["Predicted"].min(),
            )
            maximum = max(
                actual_vs_predicted["Actual"].max(),
                actual_vs_predicted["Predicted"].max(),
            )

            fig.add_trace(
                go.Scatter(
                    x=[minimum, maximum],
                    y=[minimum, maximum],
                    mode="lines",
                    name="Ideal prediction",
                    line=dict(color="#fb7185", dash="dash"),
                )
            )

            st.plotly_chart(
                fig_style(fig, 430),
                use_container_width=True,
            )

            residuals = (
                actual_vs_predicted["Actual"]
                - actual_vs_predicted["Predicted"]
            )

            residual_df = pd.DataFrame({"Prediction Error": residuals})

            fig = px.histogram(
                residual_df,
                x="Prediction Error",
                nbins=30,
                title="Prediction error distribution",
                color_discrete_sequence=["#818cf8"],
            )
            st.plotly_chart(
                fig_style(fig, 350),
                use_container_width=True,
            )

            predictions_df = result["X_test"].copy()
            predictions_df["Actual"] = np.asarray(y_test)
            predictions_df["Predicted"] = np.asarray(y_pred)
            predictions_df["Prediction Error"] = residuals.to_numpy()

            make_download_button(
                "Download test predictions",
                dataframe_to_csv(predictions_df),
                "student_model_predictions.csv",
                "text/csv",
                "prediction_download",
            )

            importance = get_feature_importance(
                fitted_models[best_model_name],
                result["features"],
            )

            if not importance.empty:
                st.subheader("Important model features")

                fig = px.bar(
                    importance.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Model feature importance",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(
                    fig_style(fig, max(350, len(importance) * 25)),
                    use_container_width=True,
                )

        st.divider()
        st.subheader("Predict a new record")

        best_model = (
            fitted_models.get(best_model_name)
            if best_model_name else None
        )

        if best_model is None:
            st.info("No model was successfully trained for new predictions.")
        else:
            model_features = result["features"]
            form_values = {}

            with st.form("new_prediction_form"):
                st.write(
                    "Enter values for the selected features. The trained "
                    "model will estimate the chosen target."
                )

                columns = st.columns(2)

                for index, feature in enumerate(model_features):
                    with columns[index % 2]:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            valid_values = df[feature].dropna()
                            default_value = (
                                float(valid_values.median())
                                if not valid_values.empty else 0.0
                            )

                            form_values[feature] = st.number_input(
                                feature,
                                value=default_value,
                                key=f"new_prediction_{feature}",
                            )
                        else:
                            options = (
                                df[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            options = sorted(options)

                            if options:
                                form_values[feature] = st.selectbox(
                                    feature,
                                    options=options,
                                    key=f"new_prediction_{feature}",
                                )
                            else:
                                form_values[feature] = ""

                submitted = st.form_submit_button(
                    "Generate prediction",
                    use_container_width=True,
                )

            if submitted:
                try:
                    new_row = pd.DataFrame(
                        [form_values],
                        columns=model_features,
                    )

                    prediction = float(best_model.predict(new_row)[0])

                    st.success(
                        f"Estimated {target}: {prediction:,.2f}"
                    )
                    st.caption(
                        "This is a model estimate, not a guaranteed "
                        "student outcome. Predictions may be unreliable "
                        "for records unlike the training data."
                    )

                except Exception as exc:
                    st.error(f"Prediction could not be generated: {exc}")


# ============================================================
# STUDENT RISK DETECTION
# ============================================================

elif page == "Student Risk":
    st.markdown('<div class="section-label">Academic support insights</div>', unsafe_allow_html=True)
    st.header("Student Risk Detection")
    st.write(
        "Use a selected academic score to identify records that may benefit "
        "from additional academic support."
    )

    score_candidates = get_score_columns(df)

    if not score_candidates:
        st.warning("A numeric score or marks column is required.")
        st.stop()

    score_col = st.selectbox(
        "Academic score feature",
        score_candidates,
        key="risk_score",
    )

    score_data = df[score_col].dropna()

    if score_data.empty or score_data.nunique() < 2:
        st.warning("The selected feature does not contain enough variation.")
        st.stop()

    threshold_method = st.radio(
        "Choose how to define a low-score threshold",
        ["Set threshold manually", "Use lower quartile (25th percentile)"],
        horizontal=True,
    )

    if threshold_method == "Set threshold manually":
        threshold = st.number_input(
            "Low-score threshold",
            value=float(score_data.quantile(0.25)),
        )
    else:
        threshold = float(score_data.quantile(0.25))
        st.info(f"Lower-quartile threshold: **{threshold:.2f}**")

    risk_df = df.copy()
    risk_df["Academic Risk Indicator"] = np.where(
        risk_df[score_col].isna(),
        "Score Missing",
        np.where(
            risk_df[score_col] < threshold,
            "Needs Attention",
            "No Low-Score Flag",
        ),
    )

    needs_attention = risk_df[
        risk_df["Academic Risk Indicator"] == "Needs Attention"
    ]
    missing_score = risk_df[
        risk_df["Academic Risk Indicator"] == "Score Missing"
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Records Reviewed", f"{len(risk_df):,}")
    c2.metric("Below Threshold", f"{len(needs_attention):,}")
    c3.metric("Missing Scores", f"{len(missing_score):,}")

    risk_percentage = (
        len(needs_attention) / max(1, len(risk_df)) * 100
    )

    st.progress(min(1.0, risk_percentage / 100))
    st.caption(
        f"{risk_percentage:.1f}% of displayed records are below the "
        "selected score threshold."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        risk_counts = (
            risk_df["Academic Risk Indicator"]
            .value_counts()
            .rename_axis("Indicator")
            .reset_index(name="Records")
        )

        fig = px.pie(
            risk_counts,
            names="Indicator",
            values="Records",
            hole=0.6,
            color_discrete_sequence=["#fb7185", "#34d399", "#fbbf24"],
            title="Academic score indicators",
        )
        st.plotly_chart(
            fig_style(fig, 380),
            use_container_width=True,
        )

    with right:
        fig = px.histogram(
            df,
            x=score_col,
            nbins=30,
            color_discrete_sequence=["#38bdf8"],
            title=f"{score_col} distribution and threshold",
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#fb7185",
            annotation_text="Selected threshold",
        )

        st.plotly_chart(
            fig_style(fig, 380),
            use_container_width=True,
        )

    st.subheader("Records flagged for review")

    if needs_attention.empty:
        st.success("No records fall below the current threshold.")
    else:
        st.dataframe(
            needs_attention,
            use_container_width=True,
            hide_index=True,
        )

        make_download_button(
            "Download flagged records",
            dataframe_to_csv(needs_attention),
            "student_records_for_review.csv",
            "text/csv",
            "risk_download",
        )

    st.warning(
        "These indicators are only screening aids based on the chosen "
        "score and threshold. They are not diagnoses, definitive risk "
        "classifications or substitutes for educator review. Consider "
        "context, data quality and student circumstances before taking action."
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":
    st.markdown('<div class="section-label">Data health and preparation</div>', unsafe_allow_html=True)
    st.header("Data Quality Center")
    st.write(
        "Inspect missing values, duplicate records and potential numerical "
        "outliers. Create a cleaned copy for download without modifying "
        "your source dataset."
    )

    missing_table = make_profile(df)
    missing_table = missing_table.sort_values(
        "Missing", ascending=False
    )

    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing Cells", f"{missing_total:,}")
    c2.metric("Duplicate Rows", f"{duplicate_total:,}")
    c3.metric(
        "Complete Columns",
        f"{int((df.isna().sum() == 0).sum()):,}",
    )

    st.subheader("Missing-value analysis")

    st.dataframe(
        missing_table[[
            "Column", "Data Type", "Non-Null",
            "Missing", "Missing %", "Unique Values"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    nonzero_missing = missing_table[missing_table["Missing"] > 0]

    if not nonzero_missing.empty:
        fig = px.bar(
            nonzero_missing.sort_values("Missing"),
            x="Missing",
            y="Column",
            orientation="h",
            color="Missing %",
            color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
            title="Missing values by feature",
        )
        st.plotly_chart(
            fig_style(fig, max(350, len(nonzero_missing) * 28)),
            use_container_width=True,
        )
    else:
        st.success("No missing values were found in the filtered dataset.")

    st.divider()
    st.subheader("Potential numerical outliers")

    outlier_rows = []

    for col in numeric_columns:
        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(
                ((series < lower) | (series > upper)).sum()
            )

        outlier_rows.append({
            "Feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Potential Outliers": outlier_count,
            "Outlier %": round(
                outlier_count / max(1, len(series)) * 100, 2
            ),
        })

    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows).sort_values(
            "Potential Outliers", ascending=False
        )
        st.dataframe(
            outlier_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Not enough numerical data to calculate outlier indicators.")

    st.caption(
        "The IQR method flags statistical extremes. An outlier is not "
        "necessarily an error and should be reviewed before removal."
    )

    st.divider()
    st.subheader("Create a cleaned copy")

    remove_duplicates = st.checkbox(
        "Remove exact duplicate rows",
        value=True,
    )
    fill_missing = st.checkbox(
        "Fill missing values using median/mode where possible",
        value=False,
    )

    cleaned_df = get_cleaned_copy(
        df,
        remove_duplicates=remove_duplicates,
        fill_missing=fill_missing,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Original filtered rows", f"{len(df):,}")
    c2.metric("Cleaned rows", f"{len(cleaned_df):,}")
    c3.metric(
        "Rows removed",
        f"{max(0, len(df) - len(cleaned_df)):,}",
    )

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True,
        hide_index=True,
    )

    make_download_button(
        "Download cleaned dataset",
        dataframe_to_csv(cleaned_df),
        "cleaned_student_performance.csv",
        "text/csv",
        "cleaned_dataset_download",
    )


# ============================================================
# EXPORT CENTER
# ============================================================

elif page == "Export Center":
    st.markdown('<div class="section-label">Reports and deliverables</div>', unsafe_allow_html=True)
    st.header("Export Center")
    st.write(
        "Download filtered records, data profiles, descriptive statistics "
        "and analysis summaries for reports and presentations."
    )

    st.subheader("Available exports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>Filtered dataset</h4>
                <p>All rows currently selected by your sidebar filters.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        make_download_button(
            "Export filtered CSV",
            dataframe_to_csv(df),
            "student_performance_export.csv",
            "text/csv",
            "export_dataset",
        )

    with col2:
        profile = make_profile(df)

        st.markdown(
            """
            <div class="info-card">
                <h4>Data profile</h4>
                <p>Column names, data types, missing values and unique counts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        make_download_button(
            "Export data profile",
            dataframe_to_csv(profile),
            "student_data_profile.csv",
            "text/csv",
            "export_profile",
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        summary = numeric_summary(df)

        st.markdown(
            """
            <div class="info-card">
                <h4>Descriptive statistics</h4>
                <p>Mean, standard deviation, quartiles and other numerical summaries.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not summary.empty:
            make_download_button(
                "Export statistics",
                dataframe_to_csv(summary),
                "student_descriptive_statistics.csv",
                "text/csv",
                "export_statistics",
            )
        else:
            st.info("No numeric statistics available.")

    with col2:
        quality_summary = pd.DataFrame({
            "Metric": [
                "Records",
                "Features",
                "Missing cells",
                "Duplicate rows",
                "Numeric features",
                "Categorical features",
            ],
            "Value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                len(get_numeric_columns(df)),
                len(get_categorical_columns(df)),
            ],
        })

        st.markdown(
            """
            <div class="info-card">
                <h4>Data quality summary</h4>
                <p>A concise overview of the current dataset's quality and structure.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        make_download_button(
            "Export quality summary",
            dataframe_to_csv(quality_summary),
            "student_data_quality_summary.csv",
            "text/csv",
            "export_quality",
        )

    st.divider()
    st.subheader("Preview report summary")

    numeric = df.select_dtypes(include=np.number)
    report_lines = [
        "STUDENTIQ — STUDENT PERFORMANCE ANALYSIS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {dataset_name}",
        "",
        "DATASET OVERVIEW",
        f"Records: {len(df)}",
        f"Features: {len(df.columns)}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "NUMERIC SUMMARY",
    ]

    if not numeric.empty:
        report_lines.append(numeric.describe().round(3).to_string())
    else:
        report_lines.append("No numerical columns found.")

    report_text = "\n".join(report_lines)

    st.text_area(
        "Report text",
        value=report_text,
        height=320,
    )

    make_download_button(
        "Download text report",
        report_text.encode("utf-8"),
        "student_performance_report.txt",
        "text/plain",
        "export_text_report",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>StudentIQ</b> · Student Performance Analysis & Prediction System<br>
        Built with Streamlit, Pandas, Plotly and Scikit-learn<br>
        For educational analytics and decision support only.
    </div>
    """,
    unsafe_allow_html=True,
)
