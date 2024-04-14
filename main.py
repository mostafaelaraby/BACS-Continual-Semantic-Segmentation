import hydra
from training import Trainer
import os


@hydra.main(
    config_path="conf/joint_debug", config_name="config",
)
def train(config):
    import warnings
    warnings.filterwarnings("ignore")
    trainer = Trainer(config)
    result = trainer.fit()
    return result


def del_env(env_name):
    """Deletes environment variable if exists

    Args:
        env_name (str): env variable name to be removed
    """    
    if env_name in os.environ:
        del os.environ[env_name]


if __name__ == "__main__":
    # Remove slurm related env. variables to avoid pl bug of sigkill
    del_env("SLURM_NTASKS")
    del_env("SLURM_JOB_NAME")
    del_env("SLURM_JOB_ID")
    train()
