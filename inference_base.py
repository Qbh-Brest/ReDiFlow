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

# 导入 DataModule 和 Model
# 需要把下面的 import 改成实际的文件路径
from project_mine.src.datasets.pdbbind import PDBBindDataModule
from project_mine.src.module.FlowMatch import Base_FM_Model
print(">>> inference_base.py started", flush=True)


def simple_test():
    
    CKPT_PATH = "D:/PythonProject medicine/project_mine/workdir/base_fm/runs/2026-03-23_21-52-19/checkpoints/epoch_069.ckpt"

    # 数据路径！改成实际的 data_dir 和 origin_data_dir！
    #DATA_DIR = "D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set_processed"  
    DATA_DIR = "D:/PythonProject medicine/project_mine/data/DOCKGEN_processed"  #posebusters_benchmark_set_processed
    ORIGIN_DATA_DIR = "D:/PythonProject medicine/project_mine/data/DOCKGEN"  
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    # ==============================================================================
    # 手动初始化 DataModule（不用Hydra）
    # ==============================================================================
    print("正在加载数据...")

    class Args:
        def __init__(self):
            self.data_dir = DATA_DIR
            self.origin_data_dir = ORIGIN_DATA_DIR
            self.teacher_dir = teacher_dir  # 没有就填 None
            self.batch_size_per_device = BATCH_SIZE
            self.num_workers = NUM_WORKERS
            self.esm_path = None  # 没有就填 None

    datamodule = PDBBindDataModule(Args())
    datamodule.setup()  

    
    print(f"正在加载模型: {CKPT_PATH}")
    model = Base_FM_Model.load_from_checkpoint(CKPT_PATH,strict=False)


    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
        enable_progress_bar =True
    )

    print("开始采样测试！")
    trainer.test(model=model, datamodule=datamodule)
    print("测试完成！")


if __name__ == "__main__":
    simple_test()
