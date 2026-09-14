import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import SurrogateDataset
from model import MiniUNet


MANIFEST = "../dataset/manifest.json"
MODEL_PATH = "../results/best.pt"
FIGURE_PATH = "../results/test_case.png"


def relative_l2(prediction, target):
    return (torch.norm(prediction - target) / torch.norm(target)).item()


def save_example_plot(target, prediction, output_path):
    names = ["ux", "uy", "p"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))

    for row in range(3):
        true_field = target[row]
        predicted_field = prediction[row]
        error_field = torch.abs(predicted_field - true_field)

        fields = [true_field, predicted_field, error_field]
        labels = ["true", "prediction", "absolute error"]

        for col in range(3):
            image = axes[row, col].pcolormesh(
                fields[col].numpy(),
                cmap="coolwarm",
                shading="auto",
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].set_title(f"{names[row]} - {labels[col]}")
            fig.colorbar(image, ax=axes[row, col])

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    device = torch.device("cpu")
    test_set = SurrogateDataset(MANIFEST, "test")

    model = MiniUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    velocity_errors = []
    pressure_errors = []
    example = None

    with torch.no_grad():
        for index in range(len(test_set)):
            inputs, targets, mask = test_set[index]

            prediction = model(inputs.unsqueeze(0))[0]
            prediction = prediction * mask
            masked_target = targets * mask

            velocity_errors.append(
                relative_l2(prediction[:2], masked_target[:2])
            )
            pressure_errors.append(
                relative_l2(prediction[2], masked_target[2])
            )

            if example is None:
                example = (masked_target, prediction)

    mean_velocity_error = np.mean(velocity_errors) * 100.0
    mean_pressure_error = np.mean(pressure_errors) * 100.0

    print(f"velocity relative L2: {mean_velocity_error:.2f}%")
    print(f"pressure relative L2: {mean_pressure_error:.2f}%")

    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    target, prediction = example
    save_example_plot(target, prediction, FIGURE_PATH)
    print("saved", FIGURE_PATH)


if __name__ == "__main__":
    main()
