import nbformat as nbf
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Network Anomaly Detection: GNN vs DBSCAN\n## Interactive Demo"),
    nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom utils import data_loader, feature_generator, dataset, baseline, models, train, visualization\nimport config\nimport torch\nimport numpy as np\nfrom sklearn.metrics import precision_recall_curve"),
    nbf.v4.new_code_cell("# 1. Load\nG_clean = data_loader.load_network()\nG_clean, G_dirty, df_nodes, df_edges_train, df_edges_test = feature_generator.generate_features_and_anomalies(G_clean)"),
    nbf.v4.new_code_cell("# 2. Baseline\nmetrics = baseline.run_dbscan(df_nodes)\nprint(metrics)"),
    nbf.v4.new_code_cell("# 3. Train\ntrain_data = dataset.create_pyg_data(df_nodes, df_edges_train)\nmodel = models.build_model(train_data.num_features)\nmodel, _ = train.train_model(model, train_data)"),
    nbf.v4.new_code_cell("# 4. Eval\ntest_data = dataset.create_pyg_data(df_nodes, df_edges_test)\nmodel.eval()\nwith torch.no_grad():\n    z = model.encode(test_data.x, test_data.edge_index)\n    adj_pred = model.decoder.forward_all(z)\n\nnode_scores = []\nfor i in range(len(df_nodes)):\n    neighbors = list(G_dirty.neighbors(i))\n    if not neighbors:\n        node_scores.append(0)\n        continue\n    link_probs = [adj_pred[i, n].item() for n in neighbors]\n    node_scores.append(1.0 - min(link_probs))"),
    nbf.v4.new_code_cell("# 5. Viz\nvisualization.plot_network(G_dirty, adj_pred, 0.5)")
]

nb['cells'] = cells
with open('project_notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated.")
