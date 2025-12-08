# unet/main.py (with argparse)
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import UNet
from train import train


def main():
    parser = argparse.ArgumentParser(description='Train UNet for dehazing')
    parser.add_argument('--foggy-dir', type=str, required=True,
                       help='Directory with foggy images')
    parser.add_argument('--clear-dir', type=str, required=True,
                       help='Directory with clear images')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='weights',
                       help='Directory to save model weights')
    
    args = parser.parse_args()
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    print(f"Training UNet:")
    print(f"  Foggy: {args.foggy_dir}")
    print(f"  Clear: {args.clear_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save dir: {args.save_dir}")
    
    train(
        model=model,
        foggy_dataset=args.foggy_dir,
        gt_dataset=args.clear_dir,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    print("\n✅ Done! Model saved to 'best.pth'")


if __name__ == "__main__":
    main()