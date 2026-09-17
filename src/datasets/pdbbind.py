import os

import pandas as pd
import torch
import numpy as np
import pickle
import networkx as nx
import lightning.pytorch as pl
from torch_geometric.data import Dataset, Batch
from torch_geometric.utils import to_networkx
from torch.utils.data import DataLoader
import random

from project_mine.src.module.FlowMatch import axis_angle_to_matrix, modify_conformer_torsion_angles_torch, \
    find_rigid_alignment, matrix_to_axis_angle, apply_transform
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 显存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ==============================================================================
# 辅助函数：计算旋转 Mask (如果预处理没做，这里补救一下)
# ==============================================================================
def get_transformation_mask(pyg_data):
    """
    只计算配体的旋转 Mask。
    """
    if isinstance(pyg_data, Batch): return torch.zeros(0).bool()

    # 提取配体边
    if 'ligand' in pyg_data.node_types:
        edge_index = pyg_data['ligand', 'ligand'].edge_index
        num_nodes = pyg_data['ligand'].num_nodes
    else:
        edge_index = pyg_data.edge_index
        num_nodes = pyg_data.num_nodes

    if edge_index.shape[1] == 0: return np.array([], dtype=object)

    # 构建仅包含配体的 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges_np = edge_index.T.cpu().numpy()
    G.add_edges_from(edges_np)

    to_rotate = []
    for i in range(edges_np.shape[0]):
        u, v = edges_np[i, 0], edges_np[i, 1]
        if u == v: continue

        G2 = G.copy()
        if G2.has_edge(u, v): G2.remove_edge(u, v)

        if not nx.is_connected(G2):
            comps = list(nx.connected_components(G2))
            smallest_comp = min(comps, key=len)
            if len(smallest_comp) > 0:
                to_rotate.append(list(smallest_comp) + [u, v])
                to_rotate.append(list(smallest_comp) + [v, u])

    return np.array(to_rotate, dtype=object) if len(to_rotate) > 0 else np.array([], dtype=object)

import math
import torch

def random_rotation_matrix_torch(dtype=torch.float32, device='cpu'):
    axis = torch.randn(3, dtype=dtype, device=device)
    axis = axis / (axis.norm() + 1e-8)

    angle = torch.empty(1, dtype=dtype, device=device).uniform_(0, 2 * math.pi)

    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=dtype, device=device)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    R = torch.eye(3, dtype=dtype, device=device) + sin_a * K + (1 - cos_a) * (K @ K)
    return R


def randomize_ligand_initial_pos(gt_pos, tr_sigma=3.0, use_small_rotation=False):
    """
    gt_pos: [N, 3]
    return: [N, 3]
    """
    dtype = gt_pos.dtype
    device = gt_pos.device

    center = gt_pos.mean(dim=0, keepdim=True)
    x = gt_pos - center

    if use_small_rotation:
        # 如果你想先做局部扰动，而不是完全随机方向，可以换成小角度
        axis = torch.randn(3, dtype=dtype, device=device)
        axis = axis / (axis.norm() + 1e-8)
        angle = torch.empty(1, dtype=dtype, device=device).uniform_(0, math.pi / 4)  # 0~45°
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=dtype, device=device)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R = torch.eye(3, dtype=dtype, device=device) + sin_a * K + (1 - cos_a) * (K @ K)
    else:
        R = random_rotation_matrix_torch(dtype=dtype, device=device)

    tr = torch.randn(1, 3, dtype=dtype, device=device) * tr_sigma
    init_pos = x @ R.T + center + tr
    return init_pos

import math
import torch

def random_rotation_matrix_torch(dtype=torch.float32, device='cpu', max_angle=None):
    axis = torch.randn(3, dtype=dtype, device=device)
    axis = axis / (axis.norm() + 1e-8)

    if max_angle is None:
        angle = torch.empty(1, dtype=dtype, device=device).uniform_(0, 2 * math.pi)
    else:
        angle = torch.empty(1, dtype=dtype, device=device).uniform_(-max_angle, max_angle)

    rot_vec = axis * angle
    R = axis_angle_to_matrix(rot_vec.unsqueeze(0)).squeeze(0)
    return R


def randomize_rigid_pose(pos, tr_sigma=3.0, max_rot_angle=None):
    """
    pos: [N, 3]
    return: [N, 3]
    """
    dtype = pos.dtype
    device = pos.device

    center = pos.mean(dim=0, keepdim=True)
    x = pos - center

    R = random_rotation_matrix_torch(dtype=dtype, device=device, max_angle=max_rot_angle)
    tr = torch.randn(1, 3, dtype=dtype, device=device) * tr_sigma

    new_pos = x @ R.T + center + tr
    return new_pos

def sample_random_torsion_updates(data, tor_sigma=1.0, tor_max_angle=math.pi):
    """
    返回 shape [M] 的随机扭转角，M = edge_mask.sum()
    单位：弧度
    """
    edge_mask = data['ligand'].edge_mask.bool()
    M = int(edge_mask.sum().item())

    if M == 0:
        return torch.zeros(0, dtype=torch.float32)

    device = data['ligand'].gt_pos.device
    dtype = data['ligand'].gt_pos.dtype

    tor_updates = torch.randn(M, device=device, dtype=dtype) * tor_sigma
    tor_updates = torch.clamp(tor_updates, -tor_max_angle, tor_max_angle)
    return tor_updates

def randomize_initial_pose_with_torsion(
    data,
    tr_sigma=3.0,
    max_rot_angle=None,
    tor_sigma=1.0,
    tor_max_angle=math.pi,
    torsion_prob=1.0,
    is_reverse_order=False,
):
    """
    基于 gt_pos 在线生成 initial_pos:
    1) 先刚体随机
    2) 再 torsion 随机

    返回:
        initial_pos: [N, 3]
        rand_tor_updates: [M]
    """
    gt_pos = data['ligand'].gt_pos.clone().float()
    device = gt_pos.device
    dtype = gt_pos.dtype

    # 1) 刚体随机化
    initial_pos = randomize_rigid_pose(
        gt_pos,
        tr_sigma=tr_sigma,
        max_rot_angle=max_rot_angle,
    )

    # 2) torsion 随机化
    edge_mask = data['ligand'].edge_mask.bool()
    M = int(edge_mask.sum().item())

    if M > 0 and torch.rand(1).item() < torsion_prob:
        rand_tor_updates = sample_random_torsion_updates(
            data,
            tor_sigma=tor_sigma,
            tor_max_angle=tor_max_angle,
        ).to(device=device, dtype=dtype)

        # edge_index: [2, E] -> 选可旋转边 -> [2, M] -> 转成 [M, 2]
        full_edge_index = data['ligand', 'ligand'].edge_index.long()
        rot_edge_index = full_edge_index[:, edge_mask].T.contiguous()

        mask_rotate = data['ligand'].mask_rotate

        # batch_idx
        if hasattr(data['ligand'], 'batch'):
            batch_idx = data['ligand'].batch
        else:
            batch_idx = torch.zeros(
                gt_pos.shape[0], dtype=torch.long, device=device
            )

        initial_pos = modify_conformer_torsion_angles_torch(
            pos=initial_pos,
            edge_index=rot_edge_index,
            mask_rotate=mask_rotate,
            torsion_updates=rand_tor_updates,
            batch_idx=batch_idx,
            is_reverse_order=is_reverse_order,
        )
    else:
        rand_tor_updates = torch.zeros(M, device=device, dtype=dtype)

    return initial_pos, rand_tor_updates

def add_pocket_with_p2rank(data,pocket_path,choice):
    if not choice:
        return data
    pocket_dict = None
    if pocket_path is not None:
        pocket_df = pd.read_csv(pocket_path)
        pocket_dict = {
            row["complex_id"]: [
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            ]
            for _, row in pocket_df.iterrows()
        }
    if not hasattr(data, "ligand_center"):
        data.ligand_center = None

    if pocket_dict is not None and data.complex_name in pocket_dict:
        data.ligand_center = torch.tensor(
            pocket_dict[data.complex_name],
            dtype=data["ligand"].pos.dtype,
            device=data["ligand"].pos.device
        )
    else:
        #data.pocket_center = data.ligand_center.clone()
        print(f"[WARN] no P2Rank pocket for {data.complex_name}, use ligand_center instead")

    return data
# ==============================================================================
# 纯净版 Dataset (不做任何图剪枝，完全信任 .pt 文件)
# ==============================================================================
class PDBBindDataset(Dataset):
    def __init__(
        self,
        data_dir,
        origin_data_dir,
        teacher_dir,
        index_list,
        esm_embeddings,
        transform=None,
        is_train=True,
        inference=False,
        fixed_val_t=0.5,
        fixed_seed_base=42,
    ):
        super().__init__(root=data_dir, transform=transform)
        self.data_dir = data_dir
        self.origin_data_dir = origin_data_dir
        self.teacher_dir = teacher_dir
        self.index_list = index_list
        self.esm_embeddings = esm_embeddings
        self.is_train = is_train
        self.inference = inference
        self.fixed_val_t = fixed_val_t
        self.fixed_seed_base = fixed_seed_base
        self.use_pocket_prior = False  # 完全原样
        self.pocket_prior_strategy = "top1"
        self.n_sample = 40
        self.test_seed = 42
        self.pocket_csv = r"D:\PythonProject medicine\project_mine\p2rank_top1_pockets.csv"

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):

        complex_name = self.index_list[idx]
        pt_path = os.path.join(self.data_dir, f"{complex_name}.pt")

        # 1. 加载 Graph (直接加载，不再动手动脚)
        data = torch.load(pt_path, map_location='cpu')
        device = data['ligand'].pos.device
        data.complex_name = complex_name

        # 2. 坐标校验
        if hasattr(data['ligand'], 'pos'):
            data['ligand'].gt_pos = data['ligand'].pos.clone().float()
        else:
            raise ValueError(f"Complex {complex_name} missing 'pos'")
        data = add_pocket_with_p2rank(data, self.pocket_csv, self.use_pocket_prior)
        # 4. Teacher Field

        # 5. ESM Embedding 注入
        if not hasattr(data, 'id'):
            data.id = complex_name

        if not hasattr(data['receptor'], 'x') or data['receptor'].x.shape[1] < 100:
            if self.esm_embeddings and complex_name in self.esm_embeddings:
                emb = self.esm_embeddings[complex_name]
                if isinstance(emb, torch.Tensor):
                    num_res = data['receptor'].num_nodes
                    if emb.shape[0] >= num_res:
                        if hasattr(data['receptor'], 'x'):
                            data['receptor'].x = torch.cat([data['receptor'].x, emb[:num_res]], dim=-1)
                        else:
                            data['receptor'].x = emb[:num_res]

        # 6. 处理旋转相关的 Mask
        if hasattr(data['ligand'], 'edge_mask'):
            data['ligand'].edge_mask = data['ligand'].edge_mask.bool()
        else:
            print(f"WARNING: {complex_name} has no edge_mask!")
            n_edges = data['ligand'].edge_index.shape[1] if hasattr(data['ligand'], 'edge_index') else 0
            data['ligand'].edge_mask = torch.zeros(n_edges, dtype=torch.bool)

        if not hasattr(data['ligand'], 'mask_rotate') or len(data['ligand'].mask_rotate) == 0:
            try:
                mask = get_transformation_mask(data)
                data['ligand'].mask_rotate = mask
            except:
                data['ligand'].mask_rotate = np.array([], dtype=object)

        # 3. 加载 Metadata (Grid & Box)
        #pkl_path = os.path.join(self.origin_data_dir, complex_name, "metadata.pkl")
        try:
            # ======================================================================
            # Initial Pose (用于 Flow Matching 的起点)
            # 核心修改：训练时优先在线随机生成 initial_pos，避免固定 start_coords 导致记忆
            # 如果你以后推理想强制使用 start_coords，可以加开关 self.use_saved_start_coords
            # ======================================================================
            use_saved_start_coords = getattr(self, 'use_saved_start_coords', False)
            if self.is_train:
                data['ligand'].initial_pos, data['ligand'].rand_tor_updates = randomize_initial_pose_with_torsion(
                    data,
                    tr_sigma=getattr(self, 'tr_sigma', 3.0),
                    max_rot_angle=getattr(self, 'max_rot_angle', None),
                    tor_sigma=getattr(self, 'tor_sigma', 1.0),
                    tor_max_angle=getattr(self, 'tor_max_angle', math.pi / 3),
                    torsion_prob=getattr(self, 'torsion_prob', 1.0),
                    is_reverse_order=getattr(self, 'is_reverse_torsion_order', False),
                )
            elif self.inference:
                data = self._build_inference_samples(
                    data,
                    n_sample=getattr(self, "n_sample", 40),
                    seed=getattr(self, "test_seed", 8888),
                )
            else:
                seed = self.fixed_seed_base + idx

                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                data['ligand'].initial_pos, data['ligand'].rand_tor_updates = randomize_initial_pose_with_torsion(
                    data,
                    tr_sigma=getattr(self, 'tr_sigma', 3.0),
                    max_rot_angle=getattr(self, 'max_rot_angle', None),
                    tor_sigma=getattr(self, 'tor_sigma', 1.0),
                    tor_max_angle=getattr(self, 'tor_max_angle', math.pi / 3),
                    torsion_prob=getattr(self, 'torsion_prob', 1.0),
                    is_reverse_order=getattr(self, 'is_reverse_torsion_order', False),
                )

        except:
            raise ValueError(f"ANYTHING PROBLEM WE GET IT\nSO YOU'D BETTER CHECK IT")

        # ==============================================================================
        # 核心步骤：复合物整体去中心化
        # ==============================================================================
        if hasattr(data['receptor'], 'pos'):
            complex_center = data['receptor'].pos.mean(dim=0, keepdim=True)

        data['receptor'].pos = data['receptor'].pos - complex_center
        data['atom'].pos = data['atom'].pos - complex_center
        if hasattr(data['ligand'], 'gt_pos'):
            data['ligand'].gt_pos = data['ligand'].gt_pos - complex_center

        if hasattr(data['ligand'], 'initial_pos'):
            data['ligand'].initial_pos = data['ligand'].initial_pos - complex_center

        data.complex_center = complex_center

        # ==============================================================================
        # 训练准备
        # ==============================================================================
        local_batch = torch.zeros(
            data['ligand'].pos.shape[0],
            device=device,
            dtype=torch.long
        )

        data.t_float = torch.rand((1,), device=device)
        dummy_t_scale = torch.ones(1, device=device)


        rot_mat, _ = find_rigid_alignment(data['ligand'].initial_pos, data['ligand'].gt_pos)
        #rot
        data['ligand'].u_rot = matrix_to_axis_angle(rot_mat.T).unsqueeze(0)
        # torsion
        data['ligand'].u_tor = build_tor_target_from_edge_mask(data)
        data['ligand'].u_tor = data['ligand'].u_tor.view(-1)
        data['ligand'].u_tor = (data['ligand'].u_tor + math.pi) % (2 * math.pi) - math.pi
        # translation
        data['ligand'].init_tr = data['ligand'].initial_pos.mean(0)
        data['ligand'].final_tr = data['ligand'].gt_pos.mean(0)
        data['ligand'].u_tr = data['ligand'].final_tr - data['ligand'].init_tr
        data['ligand'].u_tr = data['ligand'].u_tr.unsqueeze(0)
        #data.t_float = torch.ones_like(data.t_float)
        # 生成 xt
        xt = apply_transform(
            data['ligand'].initial_pos,
            data['ligand'].u_tr * data.t_float,
            data['ligand'].u_rot * data.t_float,
            data['ligand'].u_tor * data.t_float,
            local_batch,  # 用局部 batch
            dummy_t_scale,
            data
        )
        # from torch_scatter import scatter
        # rmsd = torch.sqrt(scatter(torch.sum((data['ligand'].initial_pos-data['ligand'].gt_pos) ** 2, -1), local_batch, reduce='mean') + 1e-8).mean()
        # print(f"rmsd: {rmsd}")
        data['ligand'].pos = xt.detach()

        data.complex_t = {'tr': data.t_float, 'rot': data.t_float, 'tor': data.t_float}
        data['ligand'].node_t = {
            'tr': data.t_float[local_batch],
            'rot': data.t_float[local_batch],
            'tor': data.t_float[local_batch]
        }

        if 'receptor' in data.node_types:
            receptor_local_batch = torch.zeros(
                data['receptor'].num_nodes, device=device, dtype=torch.long
            )
            data['receptor'].node_t = {
                'tr': data.t_float[receptor_local_batch],
                'rot': data.t_float[receptor_local_batch],
                'tor': data.t_float[receptor_local_batch]
            }

        return data

    def _set_inference_seed(self, seed: int = 8888):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_pocket_prior(self, data):
        """
        Return:
            prior: shape (1, 3) or None

        注意：
        - 这里只从 data.ligand_center 读 P2Rank prior。
        - 不做任何 GT ligand center 相关选择。
        - 默认取 top-1 pocket，避免 random pocket 带来的不可控方差。
        """
        if not getattr(self, "use_pocket_prior", False):
            return None

        if not hasattr(data, "ligand_center"):
            return None

        centers = data.ligand_center
        if centers is None:
            return None

        centers = centers.view(-1, 3)

        if centers.numel() == 0:
            return None

        # 如果你在 dataset 里无 prior 时设的是 zeros(1, 3)，这里可以过滤掉。
        # 极少数真实 pocket center 正好接近 0，但一般可以接受。
        if centers.abs().sum() < 1e-6:
            return None

        # 更严谨：如果你在 dataset 里存了 has_p2rank_prior，就优先用它判断
        if hasattr(data, "has_p2rank_prior"):
            has_prior = data.has_p2rank_prior
            if isinstance(has_prior, torch.Tensor):
                has_prior = bool(has_prior.item())
            if not has_prior:
                return None

        strategy = getattr(self, "pocket_prior_strategy", "top1")
        if strategy == "top1":
            prior = centers[0:1]

        elif strategy == "random":
            idx = torch.randint(
                0,
                centers.size(0),
                size=(),
                device=centers.device
            ).item()
            prior = centers[idx:idx + 1]

        else:
            raise ValueError(f"Unknown pocket_prior_strategy: {strategy}")

        return prior

    def _build_inference_samples(self, data, n_sample: int = 40, seed: int = 8888):
        """
        构建推理阶段的 data.sample_pos。
        如果 self.use_pocket_prior=True: 使用真实配体裁剪受体口袋，而不是 p2rank prior。
        否则保持原始全蛋白输入。
        """
        #print("Before crop:", data['receptor'].pos.shape[0])
        self._set_inference_seed(seed)
        device = data['ligand'].pos.device
        dtype = data['ligand'].pos.dtype

        data.n_sample = n_sample
        data.sample_pos = torch.empty(
            (data.n_sample, *data['ligand'].pos.shape),
            dtype=dtype,
            device=device
        )

        # 保存原始配体 pos，防止内部修改
        base_ligand_pos = data['ligand'].gt_pos.clone()

        use_pocket = getattr(self, 'use_pocket_prior', False)  # 复用标志位
        #print(f"use_pocket: {use_pocket}")
        if use_pocket:
            # 1. 裁剪受体（基于真实配体坐标）
            receptor_pos = data['receptor'].pos
            ligand_pos = base_ligand_pos
            diff = receptor_pos.unsqueeze(1) - ligand_pos.unsqueeze(0)
            dist = torch.linalg.norm(diff, dim=-1)
            min_dist = dist.min(dim=1).values
            cutoff = 20.0
            receptor_mask = min_dist <= cutoff
            kept_idx = receptor_mask.nonzero(as_tuple=False).view(-1)
            if kept_idx.numel() == 0:
                kept_idx = torch.argmin(min_dist).unsqueeze(0)
            data = data.subgraph({'receptor': kept_idx})

            # 2. 计算口袋“先验中心”——直接使用真实配体的几何中心
            #    （或者也可以使用裁剪后受体的中心，这里用真实配体更准确）
            true_ligand_center = base_ligand_pos.mean(dim=0, keepdim=True)  # (1, 3)
        else:
            true_ligand_center = None

        # 后续样本生成
        for sample_n in range(data.n_sample):
            data['ligand'].pos = base_ligand_pos.clone()

            sample_pos, _ = randomize_initial_pose_with_torsion(
                data,
                tr_sigma=getattr(self, 'tr_sigma', 3.0),
                max_rot_angle=getattr(self, 'max_rot_angle', None),
                tor_sigma=getattr(self, 'tor_sigma', 1.0),
                tor_max_angle=getattr(self, 'tor_max_angle', math.pi / 3),
                torsion_prob=getattr(self, 'torsion_prob', 1.0),
                is_reverse_order=getattr(self, 'is_reverse_torsion_order', False),
            )

            # 如果使用口袋信息，将随机旋转后的配体平移到真实配体中心附近
            if use_pocket and true_ligand_center is not None:
                # 计算当前随机化配体的中心
                current_center = sample_pos.mean(dim=0, keepdim=True)
                # 平移到真实配体中心
                sample_pos = sample_pos - current_center + true_ligand_center
                # 可选：添加小量平移噪声，模拟一定的不确定性
                tr_sigma_noise = getattr(self, 'tr_sigma_pocket', 1.0)  # 口袋内噪声可设小一些
                noise = torch.randn(1, 3, device=device, dtype=dtype) * tr_sigma_noise
                sample_pos = sample_pos + noise

            data.sample_pos[sample_n] = sample_pos

        data['ligand'].pos = base_ligand_pos
        data['ligand'].initial_pos = data.sample_pos[0]
        data.used_pocket_prior = use_pocket
        #print("After crop:", data['receptor'].pos.shape[0])
        return data

class PDBBindDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.origin_data_dir = args.origin_data_dir
        self.teacher_dir = getattr(args, 'teacher_dir', getattr(args, 'teacher_path', None))
        self.batch_size = args.batch_size_per_device
        self.num_workers = args.num_workers
        self.args = args
        self.train_idx_list = []
        self.val_idx_list = []
        self.test_idx_list = []
        self.esm_data = None  # <--- 定义的是 esm_data

    def setup(self, stage=None):
        print(f"DEBUG >>> Setup Start. Data Dir: {self.data_dir}")

        if hasattr(self.args, 'esm_path') and self.args.esm_path and os.path.exists(self.args.esm_path):
            print("Loading ESM Embeddings...")
            self.esm_data = torch.load(self.args.esm_path, map_location='cpu') # <--- 加载到 esm_data

        blacklist = ['1v97_1_MTE_1','2o5m_1_MNR_0','3uni_1_SAL_0','4tvd_1_BGC_4','4tvd_1_GLC_0','6nco_1_KQP_0','6wjy_2_U41_0'] #['6o0h']#["7D6O_MTE"]#['1v97_1_MTE_1','2o5m_1_MNR_0','3uni_1_SAL_0','4tvd_1_BGC_4','4tvd_1_GLC_0','6nco_1_KQP_0','6wjy_2_U41_0']   #["7D6O_MTE"]
        # with open("D:\PythonProject medicine\project_mine\data/unseen_receptors_index", "r", encoding="utf-8") as f:
        #     for line in f:
        #         # 清除换行、前后空格
        #         item = line.strip()
        #         # 跳过空行
        #         if item:
        #             blacklist.append(item)

        # print("黑名单列表加载完成，总数量：", len(blacklist))
        valid_complexes = []
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pt")]

        print(f"Scanning {len(all_files)} files...")

        for file_name in all_files:
            complex_name = file_name.split(".")[0]
            if complex_name  in blacklist:
                continue
            valid_complexes.append(complex_name)

        # 排序打乱
        valid_complexes.sort()
        np.random.seed(42)
        np.random.shuffle(valid_complexes)

        split_idx = int(len(valid_complexes) * 0.9)
        self.train_idx_list = valid_complexes[:split_idx]
        self.val_idx_list = valid_complexes[split_idx:]
        #self.test_idx_list = self.val_idx_list
        self.test_idx_list = valid_complexes   ##astex_set DOCKGEN_set PDBBind_set unseen_receptors
        # with open(f"D:/PythonProject medicine/project_mine/data/split/posebusters_test.txt", "w",encoding="utf-8") as f:
        #     f.write("\n".join(self.test_idx_list))
        print(f"Data Setup Complete. Total Valid: {len(valid_complexes)}")

    def train_dataloader(self):
        return DataLoader(
            PDBBindDataset(
                self.data_dir,
                self.origin_data_dir,
                self.teacher_dir,
                self.train_idx_list,
                self.esm_data,
                is_train=True,
                inference=False,
                fixed_val_t=0.5,
                fixed_seed_base=42,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=Batch.from_data_list,
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return DataLoader(
            PDBBindDataset(
                self.data_dir,
                self.origin_data_dir,
                self.teacher_dir,
                self.val_idx_list,
                self.esm_data,
                is_train=False,
                inference=False,
                fixed_val_t=0.5,
                fixed_seed_base=42,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=Batch.from_data_list,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self):
        return DataLoader(
            PDBBindDataset(
                self.data_dir,
                self.origin_data_dir,
                self.teacher_dir,
                self.test_idx_list,
                self.esm_data,
                is_train=False,
                inference=True,
                fixed_val_t=0.5,
                fixed_seed_base=42,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=Batch.from_data_list,
            persistent_workers=(self.num_workers > 0)
        )

    def __iter__(self):
        # 返回一个迭代器，比如：
        return iter(self.train_idx_list)

import math
import torch

def shortest_angle_diff_torch(a, b):
    # 返回 b-a 的最短角差到 (-pi, pi]
    d = b - a
    return (d + math.pi) % (2 * math.pi) - math.pi

def calc_dihedral(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = (p2 - p1)
    b2 = (p3 - p2)
    b1 = b1 / (torch.norm(b1) + 1e-12)

    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v, dim=0), w)
    return torch.atan2(y, x)

def build_neighbors_from_edge_index(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    E = edge_index.shape[1]
    for i in range(E):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        if v not in neighbors[u]:
            neighbors[u].append(v)
    return neighbors

def build_tor_target_from_edge_mask(data):
    # 用 initial_pos -> gt_pos 的角差，按 edge_mask 顺序生成 tor_target
    edge_index = data['ligand', 'ligand'].edge_index.long()
    edge_mask = data['ligand'].edge_mask.bool()
    x0 = data['ligand'].initial_pos.float()
    xT = data['ligand'].gt_pos.float()
    num_nodes = data['ligand'].num_nodes

    neighbors = build_neighbors_from_edge_index(edge_index, num_nodes)

    sel = edge_index[:, edge_mask]  # [2, M]


    M = sel.shape[1]
    tor = torch.zeros(M, dtype=torch.float32)

    for i in range(M):
        b = int(sel[0, i].item())
        c = int(sel[1, i].item())

        # 强制统一参考系统：始终从 小ID 指向 大ID
        flip = c < b
        left, right = (c, b) if flip else (b, c)

        # 找邻居必须基于 left 和 right，而不是 b 和 c，防止选出来的 a 和 d 在 flip 时乱跳
        a_cands = [n for n in neighbors[left] if n != right]
        d_cands = [n for n in neighbors[right] if n != left]

        if len(a_cands) == 0 or len(d_cands) == 0:
            tor[i] = 0.0  # 或者抛异常
            continue

        a = min(a_cands)
        d = min(d_cands)

        ang0 = calc_dihedral(x0[a], x0[left], x0[right], x0[d])
        angT = calc_dihedral(xT[a], xT[left], xT[right], xT[d])
        diff = shortest_angle_diff_torch(ang0, angT)

        # 如果当前边在 sel 里是反向的 (c < b)，要把角度符号反转回来，对齐执行器
        tor[i] = -diff #if flip else -diff

    # 硬校验：必须完全一致
    assert tor.numel() == int(edge_mask.sum().item()), "tor_target length mismatch"


    return tor
