# Student Performance Analysis & Prediction System

Professional blue Streamlit dashboard with CSV upload, interactive charts, data explorer, data-quality checks, and downloadable cleaned data.

## Run on Windows using your existing `.venv311`

Open this folder in VS Code and run these commands in the terminal:

```powershell
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

If activation is blocked by PowerShell, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv311\Scripts\Activate.ps1
```

The app starts with demo data. Upload your actual CSV from the left sidebar to analyze it. Prediction is a placeholder until the actual dataset and target column are confirmed.

## Structure
```text
Student performance analysis and prediction system/
├── main.py
├── requirements.txt
├── README.md
├── data/
├── outputs/
└── .venv311/
```
