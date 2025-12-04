import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import UNet
from dataset import FoggyDataset

def train(model, foggy_dataset, gt_dataset, optimizer, criterion, batch_size=8, epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = FoggyDataset(foggy_dataset, gt_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}")
        
        prev_loss = 0.0
        for i, (foggy, gt) in pbar:
            foggy = foggy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            output = model(foggy)
            loss = criterion(output, gt)
            loss.backward() 
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 10 == 0:
                avg_ten_loss = (epoch_loss - prev_loss) / 10
                pbar.set_postfix({'loss': f'{avg_ten_loss:.4f}'})
                prev_loss = epoch_loss
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f"\n✓ Saved best model with loss: {best_loss:.4f}")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    