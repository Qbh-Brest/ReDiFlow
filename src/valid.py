import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.set_float32_matmul_precision('high')

import lightning as L
from lightning import Trainer
from omegaconf import DictConfig

# 1. 导入你自己的 DataModule 和 Model
# 你需要把下面的 import 改成你实际的文件路径！
from project_mine.src.datasets.pdbbind import PDBBindDataModule
from project_mine.src.module.FlowMatch import Base_FM_Model


def simple_validate():
    # ==============================================================================
    # 2. 简单设置一下参数（你可以根据实际情况改）
    # ==============================================================================
    # 你的 checkpoint 路径！把 best.ckpt 换成你实际的文件名！
    CKPT_PATH = "D:\PythonProject medicine/project_mine/workdir/base_fm/runs/2026-03-23_21-52-19/checkpoints/epoch_069.ckpt"

    # 你的数据路径！改成你实际的 data_dir 和 origin_data_dir！
    DATA_DIR = "D:\PythonProject medicine\project_mine\data\posebusters_benchmark_set_processed"  # 示例，改成你实际的
    ORIGIN_DATA_DIR = "D:\PythonProject medicine\project_mine\data\posebusters_benchmark_set"  # 示例，改成你实际的
    teacher_dir = "D:\PythonProject medicine\project_mine\data/teacher"
    # 其他简单参数
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    # ==============================================================================
    # 3. 手动初始化 DataModule（不用Hydra）
    # ==============================================================================
    print("正在加载数据...")

    # 这里需要你根据实际的 PDBBindDataModule 参数来填
    # 你可以看一下你原来的 configs/data/pdbbind.yaml 里的参数
    class Args:
        def __init__(self):
            self.data_dir = DATA_DIR
            self.origin_data_dir = ORIGIN_DATA_DIR
            self.teacher_dir = teacher_dir  # 没有就填 None
            self.batch_size_per_device = BATCH_SIZE
            self.num_workers = NUM_WORKERS
            self.esm_path = None  # 没有就填 None

    datamodule = PDBBindDataModule(Args())
    datamodule.setup()  # 必须调用 setup！

    # ==============================================================================
    # 4. 直接从 checkpoint 加载模型（最简单！）
    # ==============================================================================
    print(f"正在加载模型: {CKPT_PATH}")
    model = Base_FM_Model.load_from_checkpoint(CKPT_PATH,strict=False)


    # ==============================================================================
    # 5. 初始化 Trainer，只跑验证，绝对不训练！
    # ==============================================================================
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        limit_train_batches=0,  # 绝对不训练！
        logger=False,  # 不用打日志
        enable_checkpointing=False,  # 不用存checkpoint
    )

    # ==============================================================================
    # 6. 跑验证！
    # ==============================================================================
    print("开始验证！")
    trainer.validate(model=model, datamodule=datamodule)
    print("验证完成！")


if __name__ == "__main__":
    simple_validate()