"""
Model Analysis and Comparison Utilities
This module provides tools for comparing different anomaly detection models
and analyzing their performance differences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

class ModelComparator:
    """
    A comprehensive class for comparing multiple anomaly detection models
    """
    
    def __init__(self, results_data=None):
        """
        Initialize the model comparator
        
        Args:
            results_data (pd.DataFrame): DataFrame containing predictions and scores from different models
        """
        self.results_data = results_data
        self.models = {}
        self.metrics_cache = {}
        
        if results_data is not None:
            self._extract_models_from_data()
    
    def _extract_models_from_data(self):
        """Extract model predictions from the results dataframe"""
        if self.results_data is None:
            return
        
        # Extract ground truth
        if 'is_anomaly' in self.results_data.columns:
            self.y_true = self.results_data['is_anomaly'].values
        else:
            print("Warning: No ground truth 'is_anomaly' column found")
            return
        
        # Extract GNN model results
        if 'gnn_prediction' in self.results_data.columns:
            self.models['GNN'] = {
                'predictions': self.results_data['gnn_prediction'].values,
                'scores': self.results_data.get('gnn_anomaly_score', None)
            }
        
        # Create baseline model from CPU usage
        if 'cpu' in self.results_data.columns:
            cpu_threshold = self.results_data['cpu'].quantile(0.95)
            baseline_pred = (self.results_data['cpu'] > cpu_threshold).astype(int)
            baseline_scores = self.results_data['cpu'] / self.results_data['cpu'].max()
            
            self.models['CPU_Baseline'] = {
                'predictions': baseline_pred,
                'scores': baseline_scores
            }
        
        # Create memory-based baseline
        if 'mem' in self.results_data.columns:
            mem_threshold = self.results_data['mem'].quantile(0.95)
            mem_pred = (self.results_data['mem'] > mem_threshold).astype(int)
            mem_scores = self.results_data['mem'] / self.results_data['mem'].max()
            
            self.models['Memory_Baseline'] = {
                'predictions': mem_pred,
                'scores': mem_scores
            }
    
    def add_model(self, name, predictions, scores=None):
        """
        Add a new model for comparison
        
        Args:
            name (str): Name of the model
            predictions (array-like): Binary predictions (0/1)
            scores (array-like, optional): Prediction scores/probabilities
        """
        self.models[name] = {
            'predictions': np.array(predictions),
            'scores': np.array(scores) if scores is not None else None
        }
        
        # Clear cached metrics for this model
        if name in self.metrics_cache:
            del self.metrics_cache[name]
    
    def calculate_metrics(self, model_name):
        """
        Calculate comprehensive metrics for a specific model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        if model_name in self.metrics_cache:
            return self.metrics_cache[model_name]
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_data = self.models[model_name]
        y_pred = model_data['predictions']
        y_scores = model_data['scores']
        
        metrics = {
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1': f1_score(self.y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(self.y_true, y_pred)
        }
        
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(self.y_true, y_scores)
                metrics['auc_pr'] = average_precision_score(self.y_true, y_scores)
            except:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        else:
            metrics['auc_roc'] = None
            metrics['auc_pr'] = None
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        metrics.update({
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        self.metrics_cache[model_name] = metrics
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def get_all_metrics(self):
        """Get metrics for all models"""
        all_metrics = {}
        for model_name in self.models.keys():
            all_metrics[model_name] = self.calculate_metrics(model_name)
        return all_metrics
    
    def create_comparison_table(self):
        """Create a comparison table of all metrics"""
        all_metrics = self.get_all_metrics()
        
        # Convert to DataFrame for easy viewing
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df = metrics_df.round(4)
        
        return metrics_df
    
    def plot_roc_curves(self, save_path=None, interactive=True):
        """
        Plot ROC curves for all models with scores
        
        Args:
            save_path (str, optional): Path to save the plot
            interactive (bool): Whether to return interactive plotly figure
            
        Returns:
            plotly.graph_objects.Figure or matplotlib figure
        """
        if interactive:
            fig = go.Figure()
            
            for model_name in self.models.keys():
                if self.models[model_name]['scores'] is not None:
                    scores = self.models[model_name]['scores']
                    fpr, tpr, _ = roc_curve(self.y_true, scores)
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {roc_auc:.3f})',
                        line=dict(width=2)
                    ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random (AUC = 0.500)',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True,
                width=800,
                height=600
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            plt.figure(figsize=(10, 8))
            
            for model_name in self.models.keys():
                if self.models[model_name]['scores'] is not None:
                    scores = self.models[model_name]['scores']
                    fpr, tpr, _ = roc_curve(self.y_true, scores)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                return plt.gcf()
    
    def plot_precision_recall_curves(self, save_path=None, interactive=True):
        """Plot Precision-Recall curves for all models"""
        if interactive:
            fig = go.Figure()
            
            for model_name in self.models.keys():
                if self.models[model_name]['scores'] is not None:
                    scores = self.models[model_name]['scores']
                    precision, recall, _ = precision_recall_curve(self.y_true, scores)
                    avg_precision = average_precision_score(self.y_true, scores)
                    
                    fig.add_trace(go.Scatter(
                        x=recall, y=precision,
                        mode='lines',
                        name=f'{model_name} (AP = {avg_precision:.3f})',
                        line=dict(width=2)
                    ))
            
            # Add baseline
            no_skill = len(self.y_true[self.y_true == 1]) / len(self.y_true)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[no_skill, no_skill],
                mode='lines',
                name=f'No Skill (AP = {no_skill:.3f})',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title='Precision-Recall Curves Comparison',
                xaxis_title='Recall',
                yaxis_title='Precision',
                showlegend=True,
                width=800,
                height=600
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            plt.figure(figsize=(10, 8))
            
            for model_name in self.models.keys():
                if self.models[model_name]['scores'] is not None:
                    scores = self.models[model_name]['scores']
                    precision, recall, _ = precision_recall_curve(self.y_true, scores)
                    avg_precision = average_precision_score(self.y_true, scores)
                    
                    plt.plot(recall, precision, linewidth=2,
                            label=f'{model_name} (AP = {avg_precision:.3f})')
            
            # Baseline
            no_skill = len(self.y_true[self.y_true == 1]) / len(self.y_true)
            plt.axhline(y=no_skill, color='k', linestyle='--', 
                       label=f'No Skill (AP = {no_skill:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                return plt.gcf()
    
    def plot_metrics_comparison(self, metrics=None, save_path=None, interactive=True):
        """
        Create a bar chart comparing metrics across models
        
        Args:
            metrics (list): List of metrics to compare
            save_path (str): Path to save the plot
            interactive (bool): Whether to return interactive plotly figure
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        all_metrics = self.get_all_metrics()
        
        # Prepare data for plotting
        model_names = list(all_metrics.keys())
        metric_data = {metric: [] for metric in metrics}
        
        for model in model_names:
            for metric in metrics:
                value = all_metrics[model].get(metric, 0)
                # Handle None values
                if value is None:
                    value = 0
                metric_data[metric].append(value)
        
        if interactive:
            fig = go.Figure()
            
            x = np.arange(len(model_names))
            width = 0.15
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    x=[name + f'_offset_{i}' for name in model_names],
                    y=metric_data[metric],
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors[i % len(colors)],
                    offsetgroup=i
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                barmode='group',
                showlegend=True,
                width=800,
                height=600
            )
            
            # Update x-axis to show model names properly
            fig.update_xaxes(
                tickvals=[f'{name}_offset_2' for name in model_names],
                ticktext=model_names
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(model_names))
            width = 0.15
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, metric_data[metric], 
                      width, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                return fig
    
    def plot_confusion_matrices(self, save_path=None, interactive=True):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        
        if interactive:
            # Create subplot figure
            fig = make_subplots(
                rows=1, cols=n_models,
                subplot_titles=list(self.models.keys()),
                specs=[[{"type": "heatmap"} for _ in range(n_models)]]
            )
            
            for i, (model_name, model_data) in enumerate(self.models.items(), 1):
                cm = confusion_matrix(self.y_true, model_data['predictions'])
                
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16},
                        colorscale='Blues',
                        showscale=(i == n_models)  # Only show scale for last subplot
                    ),
                    row=1, col=i
                )
                
                # Update subplot axes
                fig.update_xaxes(title_text="Predicted", row=1, col=i)
                if i == 1:
                    fig.update_yaxes(title_text="Actual", row=1, col=i)
            
            fig.update_layout(
                title_text="Confusion Matrices Comparison",
                height=400,
                width=300 * n_models
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
                fig.write_image(save_path)
            
            return fig
        
        else:
            fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, model_data) in enumerate(self.models.items()):
                cm = confusion_matrix(self.y_true, model_data['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], cbar=(i == n_models - 1))
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Predicted')
                if i == 0:
                    axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                return fig
    
    def generate_detailed_report(self, save_path=None):
        """
        Generate a detailed comparison report
        
        Args:
            save_path (str): Path to save the report
            
        Returns:
            dict: Comprehensive report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'dataset_info': {
                'total_samples': len(self.y_true),
                'positive_samples': int(self.y_true.sum()),
                'negative_samples': int(len(self.y_true) - self.y_true.sum()),
                'class_balance': float(self.y_true.sum() / len(self.y_true))
            },
            'metrics': self.get_all_metrics()
        }
        
        # Add model rankings
        metrics_df = self.create_comparison_table()
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            if metric in metrics_df.columns:
                # Handle None values
                valid_data = metrics_df[metric].dropna()
                if not valid_data.empty:
                    ranking = valid_data.sort_values(ascending=False)
                    report[f'{metric}_ranking'] = ranking.to_dict()
        
        # Add best model for each metric
        report['best_models'] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            if metric in metrics_df.columns:
                valid_data = metrics_df[metric].dropna()
                if not valid_data.empty:
                    best_model = valid_data.idxmax()
                    best_score = valid_data.max()
                    report['best_models'][metric] = {
                        'model': best_model,
                        'score': float(best_score)
                    }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def export_results(self, output_dir):
        """
        Export all comparison results to a directory
        
        Args:
            output_dir (str): Directory to save all outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics table
        metrics_df = self.create_comparison_table()
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'))
        
        # Save plots
        self.plot_roc_curves(
            save_path=os.path.join(output_dir, 'roc_curves.png'),
            interactive=False
        )
        
        self.plot_precision_recall_curves(
            save_path=os.path.join(output_dir, 'pr_curves.png'),
            interactive=False
        )
        
        self.plot_metrics_comparison(
            save_path=os.path.join(output_dir, 'metrics_comparison.png'),
            interactive=False
        )
        
        self.plot_confusion_matrices(
            save_path=os.path.join(output_dir, 'confusion_matrices.png'),
            interactive=False
        )
        
        # Save detailed report
        self.generate_detailed_report(
            save_path=os.path.join(output_dir, 'detailed_report.json')
        )
        
        # Save interactive plots
        self.plot_roc_curves(
            save_path=os.path.join(output_dir, 'roc_curves_interactive.html'),
            interactive=True
        )
        
        self.plot_precision_recall_curves(
            save_path=os.path.join(output_dir, 'pr_curves_interactive.html'),
            interactive=True
        )
        
        print(f"All results exported to: {output_dir}")

def load_and_compare_models(results_file_path, output_dir=None):
    """
    Convenience function to load results and create comparisons
    
    Args:
        results_file_path (str): Path to results CSV file
        output_dir (str): Directory to save outputs (optional)
    
    Returns:
        ModelComparator: Configured comparator instance
    """
    # Load results
    results_df = pd.read_csv(results_file_path)
    
    # Create comparator
    comparator = ModelComparator(results_df)
    
    # Generate comparison table
    print("Model Comparison Results:")
    print("=" * 50)
    print(comparator.create_comparison_table())
    print("\n")
    
    # Generate detailed report
    report = comparator.generate_detailed_report()
    print("Best Models by Metric:")
    print("-" * 25)
    for metric, info in report.get('best_models', {}).items():
        print(f"{metric.upper()}: {info['model']} ({info['score']:.4f})")
    
    # Export results if output directory specified
    if output_dir:
        comparator.export_results(output_dir)
        print(f"\nDetailed results exported to: {output_dir}")
    
    return comparator

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        comparator = load_and_compare_models(results_path, output_path)
        
        # Display interactive plots
        try:
            roc_fig = comparator.plot_roc_curves(interactive=True)
            roc_fig.show()
            
            pr_fig = comparator.plot_precision_recall_curves(interactive=True)
            pr_fig.show()
            
            metrics_fig = comparator.plot_metrics_comparison(interactive=True)
            metrics_fig.show()
            
        except ImportError:
            print("Plotly not available for interactive plots")
    
    else:
        print("Usage: python model_comparison.py <results_csv_path> [output_directory]")
        print("Example: python model_comparison.py data/processed/results_gnn_predictions.csv output/comparison/")