#!/bin/bash

# Network Anomaly Detection Dashboard Launcher
# This script sets up and launches the web dashboard for model comparison

echo "🔍 Network Anomaly Detection Dashboard Launcher"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "Installing/upgrading requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if data files exist
echo "Checking data files..."
REQUIRED_FILES=(
    "data/processed/nodes.csv"
    "data/processed/results_gnn_predictions.csv"
    "data/processed/edges_train.csv"
    "data/processed/edges_test.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        echo "Please run the main training pipeline first to generate the required data files."
        exit 1
    fi
done

echo "✅ All required data files found!"

# Create output directory for reports
mkdir -p output/dashboard_reports

# Ask user which dashboard to launch
echo ""
echo "Choose dashboard option:"
echo "1) Streamlit Dashboard (Recommended - Better UI)"
echo "2) Dash Dashboard (Alternative)"
echo "3) Generate Static Report Only"
echo "4) Run Model Comparison Analysis"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Launching Streamlit Dashboard..."
        echo "Dashboard will be available at: http://localhost:8501"
        echo "Press Ctrl+C to stop the dashboard"
        echo ""
        streamlit run web_dashboard.py --server.port 8501 --server.address 0.0.0.0
        ;;
    2)
        echo "🚀 Launching Dash Dashboard..."
        echo "Dashboard will be available at: http://localhost:8050"
        echo "Press Ctrl+C to stop the dashboard"
        echo ""
        python dashboard_interactive.py
        ;;
    3)
        echo "📊 Generating static reports..."
        python -c "
from utils.model_comparison import load_and_compare_models
comparator = load_and_compare_models('data/processed/results_gnn_predictions.csv', 'output/dashboard_reports')
print('✅ Reports generated in output/dashboard_reports/')
"
        ;;
    4)
        echo "🔬 Running model comparison analysis..."
        python utils/model_comparison.py data/processed/results_gnn_predictions.csv output/dashboard_reports
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

# Deactivate virtual environment
deactivate

echo ""
echo "Dashboard session ended. Thank you for using the Network Anomaly Detection Dashboard!"