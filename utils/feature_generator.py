import numpy as np
import pandas as pd
import networkx as nx
import random
import config

def generate_features_and_anomalies(G):
    print("[Feature Generator] Creating Train (Clean) and Test (Dirty) sets...")
    random.seed(42)
    np.random.seed(42)
    
    node_data = []
    y_true_nodes = []
    
    for node in G.nodes():
        degree = G.degree[node]
        
        # NORMAL: CPU 0.1 - 1.0
        cpu = np.random.uniform(0.1, 1.0)
        mem = np.random.uniform(0.1, 1.0)
        label = 0
        
        # ATTRIBUTE ANOMALY: CPU 100.0 (Massive Spike for DBSCAN)
        if random.random() < config.PCT_ATTRIBUTE_ANOMALY:
            cpu = np.random.uniform(90.0, 100.0)
            label = 1
            
        node_data.append([node, cpu, mem, degree])
        y_true_nodes.append(label)

    df_nodes = pd.DataFrame(node_data, columns=['id', 'cpu', 'mem', 'degree'])
    df_nodes['is_anomaly'] = y_true_nodes
    
    # STRUCTURAL INJECTION (Test Set)
    G_test = G.copy()
    
    # Connect VLAN 0 to VLAN 3
    vlan_0 = [n for n in G.nodes() if n < config.NODES_PER_VLAN]
    vlan_3 = [n for n in G.nodes() if n >= config.NODES_PER_VLAN * (config.NUM_VLANS - 1)]
    
    bridges = []
    count = 0
    for _ in range(config.NUM_STRUCTURAL_ANOMALIES * 2):
        if count >= config.NUM_STRUCTURAL_ANOMALIES: break
        u = random.choice(vlan_0)
        v = random.choice(vlan_3)
        
        if not G_test.has_edge(u, v):
            G_test.add_edge(u, v)
            bridges.append((u, v))
            # Mark as anomaly
            df_nodes.at[u, 'is_anomaly'] = 1
            df_nodes.at[v, 'is_anomaly'] = 1
            count += 1

    df_edges_train = pd.DataFrame(list(G.edges()), columns=['src', 'dst'])
    df_edges_test = pd.DataFrame(list(G_test.edges()), columns=['src', 'dst'])
    
    print(f"[Feature Generator] Injected {count} bridges and {sum(y_true_nodes)-(count*2)} CPU spikes.")
    return G, G_test, df_nodes, df_edges_train, df_edges_test
