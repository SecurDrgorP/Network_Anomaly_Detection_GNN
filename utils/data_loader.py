import networkx as nx
import os
import config

def load_network():
    """Generates the CLEAN, secure VLAN topology."""
    print(f"[Data Loader] Generating secure VLANs (Train Set)...")
    sizes = [config.NODES_PER_VLAN] * config.NUM_VLANS
    # Strict Probability Matrix
    probs = [[config.PROB_INTRA if i == j else config.PROB_INTER 
              for j in range(config.NUM_VLANS)] 
             for i in range(config.NUM_VLANS)]
    
    # Clean Graph
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    
    # FIX: Remove set metadata that crashes GML writer
    if 'partition' in G.graph:
        del G.graph['partition']
    
    # Save Raw
    if not os.path.exists(config.RAW_DIR): os.makedirs(config.RAW_DIR)
    nx.write_gml(G, os.path.join(config.RAW_DIR, "secure_vlan.gml"))
    
    return G
