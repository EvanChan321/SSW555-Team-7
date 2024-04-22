import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

class EEGAutoencoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EEGAutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 795, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.05),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(0.05),
            nn.Linear(256, 192),
            nn.LeakyReLU(negative_slope=0.05)
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

def load_model(model_path, num_classes, device):
    model = EEGAutoencoderClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Error: Expected three arguments: paths to test_data.npy, test_label.npy, and model_weights.pth")

    test_data_path = sys.argv[1]
    test_labels_path = sys.argv[2]
    model_path = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)
    mean = np.mean(test_data)
    std = np.std(test_data)

    test_data_normalized = (test_data - mean) / std
    x_test_tensor = torch.Tensor(test_data_normalized).to(device)
    y_test_tensor = torch.LongTensor(test_labels).to(device)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True, shuffle=False)

    model = load_model(model_path, num_classes=5, device=device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
