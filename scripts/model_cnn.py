import torch
import torch.nn as nn
import yaml

# Load configuration
def load_config(path="config/config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# CNN with BatchNorm
class SimpleCNN(nn.Module):
    def __init__(self, filters, dense_units, dropout, num_classes):
        super().__init__()

        layers = []
        in_channels = 3
        
        #Feature extractor with BatchNorm
        for f in filters:
            layers += [
                nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                nn.BatchNorm2d(f),                 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_channels = f

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        classifier = []
        input_dim = in_channels

        for units in dense_units:
            classifier += [
                nn.Linear(input_dim, units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            input_dim = units

        classifier.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Build model from config
def build_model_from_config(cfg):
    cnn_cfg = cfg["model"]["cnn_from_scratch"]
    model = SimpleCNN(
        filters=cnn_cfg["filters"],
        dense_units=cnn_cfg["dense_units"],
        dropout=cnn_cfg["dropout_rate"],
        num_classes=cfg["model"]["num_classes"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    return model.to(device), device
