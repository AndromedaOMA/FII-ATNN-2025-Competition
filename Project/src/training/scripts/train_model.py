import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pprint
import wandb

wandb.login()

from Project.src.training.utils.load_config import load_config
from Project.src.training.utils.get_training_logic import get_training_logic
# from src.training.utils.load_config import load_config
# from src.training.utils.get_training_logic import get_training_logic


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
        training_logic_name = config.get('training', {}).get('logic', 'baseline')
        training_logic = get_training_logic(training_logic_name)
        
        training_logic(config)
    
    return sweep_train


if __name__ == '__main__':
    print("Type the number of the experiment you want to run:")
    experiment_number = int(input())
    config = load_config(f"../experiments/experiment{experiment_number}/config.yml")

    pprint.pprint(f"Sweep configuration: {config['sweep']}")

    # Create sweep_train function with config captured in closure
    sweep_train = create_sweep_train_function(config)

    sweep_id = wandb.sweep(config['sweep'], project="ACNN-project")
    wandb.agent(sweep_id, function=sweep_train, count=7)
