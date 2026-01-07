import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.utils.load_config import load_config
from src.training.utils.get_model import get_model
from src.data_pipeline.preprocessing.preprocessing import preprocessing

# Load config from experiment1
config_path = project_root / "src" / "training" / "experiments" / "experiment1" / "config.yml"
config = load_config(str(config_path))

# Setup device
device = torch.device(config['experiment']['device'])

# Create EfficientNet B0 NS model
print("Creating EfficientNet B0 NS model...")
model = get_model("tf_efficientnet_b0_ns", config).to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Load checkpoint
checkpoint_path = project_root / "src" / "training" / "experiments" / "experiment1" / "checkpoints" / "best_model_71.26.pth"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Checkpoint loaded successfully!")

# Create test data loader (uses test transforms automatically)
print("Loading test dataset...")
_, test_loader = preprocessing(config)
print(f"Test dataset loaded with {len(test_loader.dataset)} samples")

# Enable half precision if CUDA is available
enable_half = device.type == 'cuda'

@torch.inference_mode()
def inference():
    model.eval()
    
    labels = []
    
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)
    
    return labels

# Run inference
print("Running inference...")
predictions = inference()
print(f"Inference complete! Generated {len(predictions)} predictions")

# Create DataFrame and save
data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(predictions):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
output_path = project_root / "submission.csv"
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

