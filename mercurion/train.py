from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from mercurion.model import MercurionMLP
from mercurion.utils import calculate_pos_weights
from mercurion.early_stopping import EarlyStopping


def load_data(X_path='data/processed/X.npy', y_path='data/processed/y.npy', batch_size=64, val_split=0.2):
    X = np.load(X_path)
    y = np.load(y_path)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader

def train_model(epochs=20, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Using device: {device}")
    train_loader, val_loader = load_data()

    model = MercurionMLP().to(device)
    pos_weight = calculate_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=15, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(y_batch.cpu())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        # Concatena i batch
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Sigmoid per convertire logits in probabilità
        all_probs = 1 / (1 + np.exp(-all_preds))

        # Binarizza per F1
        all_bin = (all_probs > 0.5).astype(int)

        # F1 score
        f1_micro = f1_score(all_targets, all_bin, average='micro', zero_division=0)
        f1_macro = f1_score(all_targets, all_bin, average='macro', zero_division=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
        except ValueError:
            roc_auc = float('nan')  # nel caso in cui non ci siano esempi positivi

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 micro: {f1_micro:.2%} | F1 macro: {f1_macro:.2%} | ROC-AUC: {roc_auc:.2%}")
    
        early_stopper(val_loss, model)

    
        if early_stopper.early_stop:
            print("🛑 Stopping early")
            break

if __name__ == "__main__":
    train_model(epochs=100)
