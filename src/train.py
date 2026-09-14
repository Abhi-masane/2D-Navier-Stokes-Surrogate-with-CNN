import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SurrogateDataset
from model import MiniUNet


MANIFEST = "../dataset/manifest.json"
MODEL_PATH = "../results/best.pt"
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 1e-3


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions * mask, targets * mask)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = SurrogateDataset(MANIFEST, "train")
    val_set = SurrogateDataset(MANIFEST, "val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = MiniUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets, mask in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions * mask, targets * mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch + 1:3d}/{EPOCHS}  "
            f"train {train_loss:.5f}  val {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
