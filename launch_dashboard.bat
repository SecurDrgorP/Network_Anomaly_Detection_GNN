@echo off
REM Network Anomaly Detection Dashboard Launcher for Windows
REM This script sets up and launches the web dashboard for model comparison

echo 🔍 Network Anomaly Detection Dashboard Launcher
echo ================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade requirements
echo Installing/upgrading requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM Check if data files exist
echo Checking data files...
set "data_missing=0"

if not exist "data\processed\nodes.csv" (
    echo ❌ Required file not found: data\processed\nodes.csv
    set "data_missing=1"
)

if not exist "data\processed\results_gnn_predictions.csv" (
    echo ❌ Required file not found: data\processed\results_gnn_predictions.csv
    set "data_missing=1"
)

if not exist "data\processed\edges_train.csv" (
    echo ❌ Required file not found: data\processed\edges_train.csv
    set "data_missing=1"
)

if not exist "data\processed\edges_test.csv" (
    echo ❌ Required file not found: data\processed\edges_test.csv
    set "data_missing=1"
)

if "%data_missing%"=="1" (
    echo Please run the main training pipeline first to generate the required data files.
    pause
    exit /b 1
)

echo ✅ All required data files found!

REM Create output directory for reports
if not exist "output\dashboard_reports" mkdir "output\dashboard_reports"

REM Ask user which dashboard to launch
echo.
echo Choose dashboard option:
echo 1^) Streamlit Dashboard ^(Recommended - Better UI^)
echo 2^) Dash Dashboard ^(Alternative^)
echo 3^) Generate Static Report Only
echo 4^) Run Model Comparison Analysis

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo 🚀 Launching Streamlit Dashboard...
    echo Dashboard will be available at: http://localhost:8501
    echo Press Ctrl+C to stop the dashboard
    echo.
    streamlit run web_dashboard.py --server.port 8501 --server.address 0.0.0.0
) else if "%choice%"=="2" (
    echo 🚀 Launching Dash Dashboard...
    echo Dashboard will be available at: http://localhost:8050
    echo Press Ctrl+C to stop the dashboard
    echo.
    python dashboard_interactive.py
) else if "%choice%"=="3" (
    echo 📊 Generating static reports...
    python -c "from utils.model_comparison import load_and_compare_models; comparator = load_and_compare_models('data/processed/results_gnn_predictions.csv', 'output/dashboard_reports'); print('✅ Reports generated in output/dashboard_reports/')"
) else if "%choice%"=="4" (
    echo 🔬 Running model comparison analysis...
    python utils\model_comparison.py data\processed\results_gnn_predictions.csv output\dashboard_reports
) else (
    echo Invalid choice. Exiting...
    pause
    exit /b 1
)

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Dashboard session ended. Thank you for using the Network Anomaly Detection Dashboard!
pause