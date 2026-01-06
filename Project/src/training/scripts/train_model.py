import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pprint
import wandb

wandb.login()

from src.training.utils.load_config import load_config
from src.training.utils.get_training_logic import get_training_logic


def create_sweep_train_function(config):
    """
    Create a sweep_train function that captures the config in a closure.
    
    Args:
        config: Configuration dictionary
    
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
        
        training_logic(config, config_wb)
    
    return sweep_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a wandb sweep')
    parser.add_argument('--experiment', '-e', type=int, default=None,
                        help='Experiment number (if not provided, will prompt)')
    args = parser.parse_args()
    
    if args.experiment is None:
        print("Type the number of the experiment you want to run:")
        experiment_number = int(input())
    else:
        experiment_number = args.experiment
    
    config = load_config(f"../experiments/experiment{experiment_number}/config.yml")

    pprint.pprint(f"Sweep configuration: {config['sweep']}")

    # Create sweep_train function with config captured in closure
    sweep_train = create_sweep_train_function(config)

    sweep_id = wandb.sweep(config['sweep'], project="ACNN-project")
    wandb.agent(sweep_id, function=sweep_train, count=7)
