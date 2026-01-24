# Web Dashboard Quick Start Guide

## 🔍 Network Anomaly Detection Dashboard

This guide helps you launch and use the interactive web dashboard for comparing and visualizing different anomaly detection models.

### 📋 Prerequisites

1. **Data Files**: Ensure you have run the main training pipeline to generate:
   - `data/processed/nodes.csv`
   - `data/processed/results_gnn_predictions.csv`
   - `data/processed/edges_train.csv`
   - `data/processed/edges_test.csv`

2. **Python Environment**: Python 3.7+ with required packages

### 🚀 Quick Launch

#### Option 1: Use Launch Script (Recommended)

**Linux/Mac:**
```bash
./launch_dashboard.sh
```

**Windows:**
```cmd
launch_dashboard.bat
```

#### Option 2: Manual Launch

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit Dashboard:**
   ```bash
   streamlit run web_dashboard.py --server.port 8501
   ```
   
   **Or launch Dash Dashboard:**
   ```bash
   python dashboard_interactive.py
   ```

### 📊 Dashboard Features

#### 1. **Overview Section**
- Key metrics (total nodes, anomalies, normal nodes, anomaly rate)
- Data distribution visualizations
- CPU and Memory usage histograms

#### 2. **Data Analysis Section**
- Feature correlation matrix
- Scatter plots (CPU vs Memory, Degree vs CPU)
- Statistical summary tables

#### 3. **Model Comparison Section**
- Side-by-side performance metrics
- ROC curve comparisons
- Confusion matrices for each model
- Detailed classification reports

#### 4. **Network Visualization Section**
- Interactive network topology
- Node coloring based on anomaly status
- Network statistics and degree distribution

#### 5. **Performance Metrics Section**
- Performance vs threshold analysis
- Precision-Recall curves
- Score distributions
- Error analysis (false positives/negatives)

### 🔧 Available Dashboards

#### 1. **Streamlit Dashboard** (`web_dashboard.py`)
- **Best for**: Interactive exploration and analysis
- **Features**: 
  - Multi-tab interface
  - Real-time filtering and selection
  - Easy-to-use sidebar navigation
  - Export capabilities
- **Access**: http://localhost:8501

#### 2. **Dash Dashboard** (`dashboard_interactive.py`)
- **Best for**: Responsive web application
- **Features**:
  - Bootstrap-styled interface
  - Card-based layout
  - Interactive plotly charts
  - Professional appearance
- **Access**: http://localhost:8050

#### 3. **Static Reports** (`utils/model_comparison.py`)
- **Best for**: Generating reports and batch analysis
- **Features**:
  - PNG/HTML export
  - JSON reports
  - Comprehensive metrics
  - Batch processing

### 🎯 Model Comparison Features

#### Metrics Calculated:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among positive predictions
- **Recall (Sensitivity)**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

#### Available Models:
- **GNN Model**: Your trained Graph Neural Network
- **CPU Baseline**: High CPU usage threshold-based detection
- **Memory Baseline**: High memory usage threshold-based detection

#### Visualizations:
- **ROC Curves**: False positive rate vs True positive rate
- **Precision-Recall Curves**: Precision vs Recall trade-offs
- **Confusion Matrices**: Classification results breakdown
- **Metrics Comparison**: Bar charts of all metrics
- **Network Topology**: Interactive graph visualization

### 📈 Usage Tips

#### 1. **Comparing Models**
- Use the Model Comparison tab to see side-by-side performance
- Look at both ROC and PR curves - PR curves are better for imbalanced datasets
- Check confusion matrices to understand error types

#### 2. **Analyzing Network Structure**
- Red nodes in network visualization indicate anomalies
- Hover over nodes to see detailed information
- Use degree distribution to understand network connectivity

#### 3. **Performance Analysis**
- Use threshold analysis to find optimal operating points
- Check false positive/negative lists for error patterns
- Compare score distributions between normal and anomalous nodes

#### 4. **Exporting Results**
- Streamlit: Use the download buttons in each section
- Dash: Screenshots or browser save functionality
- Static reports: Automatic export to `output/dashboard_reports/`

### 🔍 Troubleshooting

#### Common Issues:

1. **"No data available" error**
   - Ensure data files exist in `data/processed/`
   - Run the main training pipeline first

2. **Import errors**
   - Install missing packages: `pip install -r requirements.txt`
   - Ensure virtual environment is activated

3. **Port already in use**
   - Change port in launch command
   - For Streamlit: `--server.port 8502`
   - For Dash: modify `port=8051` in `dashboard_interactive.py`

4. **Slow performance**
   - Large networks may take time to render
   - Consider sampling edges for visualization

5. **Browser compatibility**
   - Use modern browsers (Chrome, Firefox, Safari, Edge)
   - Ensure JavaScript is enabled

### 📁 Output Files

When using static report generation, files are saved to `output/dashboard_reports/`:

```
output/dashboard_reports/
├── metrics_comparison.csv          # Metrics table
├── roc_curves.png                 # ROC curves (static)
├── roc_curves_interactive.html    # ROC curves (interactive)
├── pr_curves.png                  # PR curves (static)
├── pr_curves_interactive.html     # PR curves (interactive)
├── metrics_comparison.png         # Metrics bar chart
├── confusion_matrices.png         # Confusion matrices
└── detailed_report.json          # Complete analysis report
```

### 🛠 Customization

#### Adding New Models:
1. **Edit** `utils/model_comparison.py`
2. **Add** new model data using `add_model()` method
3. **Update** dashboard files to include new model

#### Modifying Visualizations:
1. **Streamlit**: Edit functions in `web_dashboard.py`
2. **Dash**: Modify callback functions in `dashboard_interactive.py`
3. **Static**: Update plotting functions in `utils/model_comparison.py`

#### Changing Themes:
- **Streamlit**: Use `st.set_page_config()` theme options
- **Dash**: Modify Bootstrap theme in `external_stylesheets`

### 💡 Best Practices

1. **Regular Updates**: Re-run dashboard after model training
2. **Data Validation**: Check data quality before visualization
3. **Performance Monitoring**: Monitor model metrics over time
4. **Error Analysis**: Regularly check false positive/negative patterns
5. **Documentation**: Save important findings and insights

### 🆘 Getting Help

1. **Check logs**: Dashboard applications show detailed error messages
2. **Verify data**: Ensure all CSV files have expected columns
3. **Test environment**: Try launching with sample data first
4. **Memory usage**: Monitor system resources for large datasets

### 🎉 Next Steps

1. **Explore different views** in the dashboard
2. **Compare model performance** across different metrics
3. **Analyze error patterns** to improve models
4. **Generate reports** for documentation
5. **Customize visualizations** for specific needs

---

**Happy analyzing!** 🚀

For more detailed technical information, see the individual Python files and their docstrings.