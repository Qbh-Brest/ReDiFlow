import warnings
warnings.filterwarnings("ignore")

import os
os.environ['RDKIT_LOGGER_LEVEL'] = 'ERROR'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
torch.set_float32_matmul_precision('high')

import lightning as L
from lightning import Trainer
from omegaconf import DictConfig

# Import the DataModule and Model
# You need to replace the import statements below with the actual file paths
from project_mine.src.datasets.pdbbind import PDBBindDataModule
from project_mine.src.module.FlowMatch import Base_FM_Model
print(">>> inference_base.py started", flush=True)


def simple_test():
    
    CKPT_PATH = "D:/PythonProject medicine/project_mine/workdir/base_fm/runs/2026-03-23_21-52-19/checkpoints/epoch_069.ckpt"

    # Data paths:Please replace them with the actual paths of data_dir and origin_data_dir respectively
    #DATA_DIR = "D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set_processed"  
    DATA_DIR = "D:/PythonProject medicine/project_mine/data/DOCKGEN_processed"  #posebusters_benchmark_set_processed
    ORIGIN_DATA_DIR = "D:/PythonProject medicine/project_mine/data/DOCKGEN"  
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    # ==============================================================================
    # Manually initialize DataModule(Hydra not used)
    # ==============================================================================
    print("Loading data...")

    class Args:
        def __init__(self):
            self.data_dir = DATA_DIR
            self.origin_data_dir = ORIGIN_DATA_DIR
            self.teacher_dir = teacher_dir  # Set to None if not available
            self.batch_size_per_device = BATCH_SIZE
            self.num_workers = NUM_WORKERS
            self.esm_path = None  # Set to None if not available

    datamodule = PDBBindDataModule(Args())
    datamodule.setup()  

    
    print(f"Loading model: {CKPT_PATH}")
    model = Base_FM_Model.load_from_checkpoint(CKPT_PATH,strict=False)


    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
        enable_progress_bar =True
    )

    print("Start sampling test···")
    trainer.test(model=model, datamodule=datamodule)
    print("Test completed！")


if __name__ == "__main__":
    simple_test()
