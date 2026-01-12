import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.utils.load_config import load_config
from src.training.utils.get_model import get_model
from src.data_pipeline.preprocessing.preprocessing import preprocessing


@torch.inference_mode()
def inference(model, test_loader, device, enable_half=True):
    """
    Run inference on test dataset.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test dataset
        device: Device to run inference on
        enable_half: Whether to use mixed precision (float16)
    
    Returns:
        list: Predicted labels
    """
    model.eval()
    labels = []
    
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--experiment', '-e', type=int, required=True,
                        help='Experiment number')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Checkpoint filename (e.g., best_model_71.26.pth) or full path')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                        help='Output CSV file path (default: submission.csv)')
    parser.add_argument('--enable_half', action='store_true', default=True,
                        help='Enable mixed precision inference (default: True)')
    parser.add_argument('--model_name', '-m', type=str, default=None,
                        help='Override model name from config (e.g., tf_efficientnet_b0_ns)')
    
    args = parser.parse_args()
    
    # Load config - resolve path relative to script location
    script_dir = Path(__file__).parent
    config_path = script_dir / f"../experiments/experiment{args.experiment}/config.yml"
    config = load_config(str(config_path.resolve()))
    
    # Setup device
    device = torch.device(config['experiment']['device'])
    
    # Create model (get_model handles case conversion)
    # Use override model name if provided, otherwise use config
    model_name = args.model_name if args.model_name else config['model']['name']
    print(f"Creating model: {model_name}")
    model = get_model(model_name, config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load checkpoint - try as full path first, then relative to experiment directory
    if os.path.isabs(args.checkpoint) or os.path.exists(args.checkpoint):
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = script_dir / f"../experiments/experiment{args.experiment}/checkpoints/{args.checkpoint}"
        checkpoint_path = checkpoint_path.resolve()
        # If not found, try experiment 1's checkpoints (for cross-experiment usage)
        if not checkpoint_path.exists():
            alt_path = script_dir / f"../experiments/experiment1/checkpoints/{args.checkpoint}"
            alt_path = alt_path.resolve()
            if alt_path.exists():
                checkpoint_path = alt_path
                print(f"Note: Using checkpoint from experiment1: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded successfully!")
    
    # Create test data loader (only test set, no training transforms needed)
    # We need to modify config to only load test set
    print("Loading test dataset...")
    _, test_loader = preprocessing(config)
    
    # Run inference
    print("Running inference...")
    predictions = inference(model, test_loader, device, enable_half=args.enable_half)
    
    # Create DataFrame and save
    data = {
        "ID": list(range(len(predictions))),
        "target": predictions
    }
    
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == '__main__':
    main()

