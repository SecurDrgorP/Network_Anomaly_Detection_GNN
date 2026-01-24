#!/bin/bash

echo "Setting up web dashboard dependencies..."

# Install required packages
pip install streamlit plotly dash dash-bootstrap-components kaleido

echo "Web dashboard setup complete!"
echo ""
echo "To run the Streamlit dashboard:"
echo "  streamlit run dashboard.py"
echo ""
echo "To run the Dash dashboard:"
echo "  python dashboard_dash.py"
echo ""
echo "The dashboards will be available at:"
echo "  Streamlit: http://localhost:8501"
echo "  Dash: http://localhost:8050"