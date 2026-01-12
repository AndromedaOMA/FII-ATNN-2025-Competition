import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pprint
import wandb

wandb.login()

#from Project.src.training.utils.load_config import load_config
#from Project.src.training.utils.get_training_logic import get_training_logic
from src.training.utils.load_config import load_config
from src.training.utils.get_training_logic import get_training_logic
from src.data_pipeline.preprocessing.preprocessing import load_datasets


def create_sweep_train_function(config, train_dataset, test_dataset):
    """
    Create a sweep_train function that captures the config and datasets in a closure.
    
    Args:
        config: Configuration dictionary
        train_dataset: Pre-loaded train dataset
        test_dataset: Pre-loaded test dataset
    
    Returns:
        A function that can be used with wandb.agent
    """
    def sweep_train():
        """
        Wrapper function for wandb sweeps that uses the training logic from get_training_logic.
        """

        # Initialize wandb run first (this must be called before accessing wandb.config)
        wandb.init(project="ACNN-project")
        
        config_wb = wandb.config
        
        # Get training logic from sweep config if available, otherwise from main config
        if hasattr(config_wb, 'training_logic'):
            training_logic_name = config_wb.training_logic
        else:
            training_logic_name = config.get('training', {}).get('logic', 'baseline')
        
        training_logic = get_training_logic(training_logic_name)
        
        # Pass datasets to training logic
        training_logic(config, train_dataset=train_dataset, test_dataset=test_dataset)
    
    return sweep_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a wandb sweep')
    parser.add_argument('--experiment', '-e', type=int, default=None,
                        help='Experiment number (if not provided, will prompt)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--start_epoch', '-s', type=int, default=None,
                        help='Epoch index to start training from (required when using --checkpoint)')
    parser.add_argument('--best_acc', '-a', type=float, default=None,
                        help='Current best validation accuracy (required when using --checkpoint)')
    args = parser.parse_args()
    
    if args.experiment is None:
        print("Type the number of the experiment you want to run:")
        experiment_number = int(input())
    else:
        experiment_number = args.experiment
    
    config = load_config(f"../experiments/experiment{experiment_number}/config.yml")
    
    # Add resume training parameters to config if provided
    if args.checkpoint is not None:
        if args.start_epoch is None or args.best_acc is None:
            raise ValueError("When using --checkpoint, both --start_epoch and --best_acc must be provided")
        config['training']['resume'] = {
            'checkpoint_path': args.checkpoint,
            'start_epoch': args.start_epoch,
            'best_acc': args.best_acc
        }
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        print(f"Starting from epoch: {args.start_epoch}, Best accuracy: {args.best_acc}")

    pprint.pprint(f"Sweep configuration: {config['sweep']}")

    # Load datasets once before starting the sweep
    print("Loading datasets before sweep...")
    train_dataset, test_dataset = load_datasets(config)
    print("Datasets loaded successfully!")

    # Create sweep_train function with config and datasets captured in closure
    sweep_train = create_sweep_train_function(config, train_dataset, test_dataset)

    sweep_id = wandb.sweep(config['sweep'], project="ACNN-project")
    wandb.agent(sweep_id, function=sweep_train, count=7)
