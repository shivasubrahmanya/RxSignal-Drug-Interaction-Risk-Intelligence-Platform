import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import joblib

def create_graph_dataset(ml_data_path, output_dir):
    print("Loading ML data for Graph Construction...")
    df = pd.read_parquet(ml_data_path)
    
    # We need a unique integer ID for every drug
    unique_drugs = sorted(list(set(df['drug_a'].unique()) | set(df['drug_b'].unique())))
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}
    idx_to_drug = {i: drug for i, drug in enumerate(unique_drugs)}
    
    print(f"Found {len(unique_drugs)} unique drugs for Graph Nodes.")
    
    # 1. Node Features (x)
    # We will use the frequency and mean risk as the initial node embeddings
    # We need to map these from the dataframe.
    print("Constructing Node Features...")
    node_features = np.zeros((len(unique_drugs), 2)) # [freq, mean_risk]
    
    # To get reliable node features, we'll aggregate them
    # But since they were target encoded, we can just grab them from the df
    # For Drug A
    a_feats = df[['drug_a', 'drug_a_freq', 'drug_a_mean_risk']].drop_duplicates('drug_a')
    for _, row in a_feats.iterrows():
        idx = drug_to_idx[row['drug_a']]
        node_features[idx, 0] = row['drug_a_freq']
        node_features[idx, 1] = row['drug_a_mean_risk']
        
    # For Drug B (fill in any missing ones)
    b_feats = df[['drug_b', 'drug_b_freq', 'drug_b_mean_risk']].drop_duplicates('drug_b')
    for _, row in b_feats.iterrows():
        idx = drug_to_idx[row['drug_b']]
        if node_features[idx, 0] == 0:
            node_features[idx, 0] = row['drug_b_freq']
            node_features[idx, 1] = row['drug_b_mean_risk']
            
    # Normalize node features
    x = torch.tensor(node_features, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    
    # 2. Edge Index and Edge Attributes
    print("Constructing Edge Indices and Attributes...")
    # An edge exists if drug_a and drug_b have been reported together.
    # The edge attribute is the risk_score (what we want to predict).
    
    source_nodes = df['drug_a'].map(drug_to_idx).values
    target_nodes = df['drug_b'].map(drug_to_idx).values
    
    # Undirected graph: add both directions
    edge_index_np = np.vstack([
        np.concatenate([source_nodes, target_nodes]),
        np.concatenate([target_nodes, source_nodes])
    ])
    
    edge_attr_np = np.concatenate([df['max_risk'].values, df['max_risk'].values])
    
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float).view(-1, 1)
    
    # 3. Create PyTorch Geometric Data Object
    print("Building PyG Data Object...")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Save the mapping and the graph
    print("Saving Graph Data...")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(data, os.path.join(output_dir, "faers_graph.pt"))
    joblib.dump(drug_to_idx, os.path.join(output_dir, "graph_drug_mapping.pkl"))
    
    print("Graph Construction Complete!")
    print(data)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_data_path = os.path.join(base_dir, "processed", "ml_training_data.parquet")
    output_dir = os.path.join(base_dir, "processed")
    
    create_graph_dataset(ml_data_path, output_dir)
