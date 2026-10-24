# Student Performance Analysis & Prediction System

An interactive data analytics and machine learning web application built with Python and Streamlit to explore student performance, analyze academic trends, check data quality, and predict student outcomes using a regression model.

## Overview

The Student Performance Analysis & Prediction System transforms student data into meaningful insights through interactive visualizations, data exploration tools, and machine learning. It is designed as an academic and portfolio project to demonstrate practical skills in data analytics and predictive modeling.

## Features

- **Interactive Dashboard:** View key dataset statistics and performance summaries.
- **Data Explorer:** Browse student records and explore dataset columns.
- **Performance Analysis:** Analyze numerical data and visualize performance trends.
- **Machine Learning Prediction:** Train a regression model and generate predictions when the dataset contains suitable numeric features and a target.
- **Data Quality Checks:** Identify missing values and inspect data quality.
- **CSV Dataset Support:** Load student data from the project dataset.
- **Interactive Charts:** Explore data through Plotly visualizations.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## Project Structure

```text
student-performance-analysis-prediction-system/
├── main.py
├── requirements.txt
├── README.md
└── data/
    └── Cleaned_Students_Performance.csv
```

## Dataset

The application uses a student performance CSV dataset stored in the `data/` directory.

The available analyses and prediction targets depend on the columns present in the dataset. Verify column names, missing values, and data types before interpreting results.

## Machine Learning

The application includes a regression-based prediction workflow using Scikit-learn.

Model evaluation may include:
- Mean Absolute Error (MAE)
- R² score

Performance depends on dataset quality, selected features, target variable, and the train-test split. Predictions are estimates and should not be interpreted as guaranteed academic outcomes.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Aashwin6389-Tech/student-performance-analysis-prediction-system.git
cd student-performance-analysis-prediction-system
```

### 2. Create and activate a virtual environment (optional)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run main.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Live Demo

[Open Student Performance Analysis & Prediction System](https://student-performance-analysis-prediction-systemgit-8u6xxods2ead.streamlit.app/https://student-performance-analysis-prediction-systemgit-8u6xxods2ead.streamlit.app/)

## Future Improvements

- Improve prediction accuracy through model comparison and validation.
- Add more meaningful academic performance indicators.
- Improve visual reporting and downloadable analysis reports.
- Add clear explanations of model limitations.

## Author

**Aashwin Tripathi**

B.Tech Computer Science Engineering — Data Science & Artificial Intelligence

Shri Ramswaroop Memorial University, Lucknow

GitHub: [Aashwin6389-Tech](https://github.com/Aashwin6389-Tech)

LinkedIn: [Aashwin Tripathi](https://linkedin.com/in/aashwin-tripathi-869021333)
