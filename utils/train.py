import torch
import config

def train_model(model, data):
    print(f"\n--- Training GNN on CLEAN Topology ({config.EPOCHS} epochs) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    model.train()
    losses = []
    
    for epoch in range(config.EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return model, losses
