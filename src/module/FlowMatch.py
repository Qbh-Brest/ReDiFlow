import csv
import math
import os
from typing import Any, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils.versions import torch_version
from pyasn1_modules.rfc7292 import x509CRL
from rdkit.Chem import rdMolAlign
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.cuda.amp import autocast


from matcha.utils.spyrmsd import symmrmsd
from matcha.utils.transforms import rotvec_to_rotmat
from project_mine.compute_energy_curve import save_simple_complex_all
from project_mine.merge_pose_sdfs import merge_pose_sdfs, write_complex_manifest
from project_mine.model import LearnableTimeBudget
from project_mine.pb_valid import PB_valid
# 假设这些是你的项目内部引用，请根据实际情况保留
from project_mine.src.utils.pylogger import RankedLogger
from project_mine.src.models.get_model import get_vector_field
torch.autograd.set_detect_anomaly(True)

log = RankedLogger(__name__, rank_zero_only=True)

# 显存优化配置：允许碎片内存重组
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# ===== [TOR-TARGET CHECK UTILS] BEGIN =====

def _dihedral(p0, p1, p2, p3):
	b0 = p1 - p0
	b1 = p2 - p1
	b2 = p3 - p2
	b1n = b1 / (torch.norm(b1) + 1e-12)
	v = b0 - (b0 @ b1n) * b1n
	w = b2 - (b2 @ b1n) * b1n
	x = v @ w
	y = torch.cross(b1n, v, dim=0) @ w
	return torch.atan2(y, x)

def _ang_diff(a, b):
	d = a - b
	return torch.atan2(torch.sin(d), torch.cos(d))

def _build_adj(num_nodes, edges_2col):
	# edges_2col: [E,2], 无向图按双向加入邻接
	adj = [[] for _ in range(num_nodes)]
	for i in range(edges_2col.shape[0]):
		u = int(edges_2col[i, 0].item())
		v = int(edges_2col[i, 1].item())
		adj[u].append(v)
		adj[v].append(u)
	return adj
# ===== [TOR-TARGET CHECK UTILS] END =====

# ==============================================================================
# 1. 物理计算核心: 极速版原子速度计算 (向量化 Torsion)
# ==============================================================================
from torch_scatter import scatter

from torch_geometric.utils import to_dense_batch
from scipy.spatial.transform import Rotation as R

import torch
from torch_scatter import scatter_mean


def modify_conformer_torsion_angles_torch(pos, edge_index, mask_rotate, torsion_updates, batch_idx,
										  is_reverse_order=False):
	"""严格按照作者逻辑修改的扭转执行函数（新增反转旋转顺序功能）"""
	new_pos = pos.clone()

	# 1. 【原作者逻辑】记录初始质心 (shift_center_back 准备)
	pos_mean = scatter_mean(pos, batch_idx, dim=0)[batch_idx]

	# ========== 仅新增：根据is_reverse_order确定旋转顺序 ==========
	# 正向：0→1→2→...→n-1（原逻辑）；反向：n-1→n-2→...→0
	rot_bond_range = range(edge_index.shape[0]) if not is_reverse_order else range(edge_index.shape[0] - 1, -1, -1)

	# 把原来的 range(edge_index.shape[0]) 换成 rot_bond_range
	for i in rot_bond_range:
		u, v = edge_index[i, 0], edge_index[i, 1]

		# 2. 【原作者逻辑】约定正向旋转：rot_vec = pos[u] - pos[v]
		rot_vec = new_pos[u] - new_pos[v]

		# 3. 【原作者逻辑】单位化并乘以角速度(转速) -> 得到真正的旋转向量
		rot_vec = rot_vec * torsion_updates[i] / (torch.linalg.norm(rot_vec)+1e-8)

		# 4. 利用现成的 axis_angle 转换函数生成旋转矩阵 (类似原作者的 rotvec_to_rotmat)
		# 注意：传入 shape 需要是 [1, 3] 以匹配批量转换函数，算完再 squeeze 变回 [3, 3]
		rot_mat = axis_angle_to_matrix(rot_vec.unsqueeze(0)).squeeze(0)

		# 处理 mask 格式
		m = mask_rotate[i]
		if isinstance(m, torch.Tensor):
			m = m.bool().view(-1)
		else:
			m = torch.tensor(m, device=pos.device, dtype=torch.bool).view(-1)

		if m.any():
			# 5. 【原作者逻辑】以 v 为枢轴旋转: (pos[mask] - pos[v]) @ rot_mat.T + pos[v]
			diff = new_pos[m] - new_pos[v]
			rotated_diff = torch.mm(diff, rot_mat.t())
			new_pos[m] = rotated_diff + new_pos[v]

	# 6. 【原作者逻辑核心】shift_center_back：强行抵消扭转带来的质心偏移！
	new_mean = scatter_mean(new_pos, batch_idx, dim=0)[batch_idx]
	new_pos = new_pos - new_mean + pos_mean

	return new_pos

# 辅助函数保持原样，完全没问题
def axis_angle_to_matrix(axis_angle, eps=1e-6):
	orig_dtype = axis_angle.dtype
	x = axis_angle.to(torch.float32)

	angle = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
	axis = x / angle

	K = torch.zeros((x.shape[0], 3, 3), device=x.device, dtype=x.dtype)
	K[:, 0, 1] = -axis[:, 2]
	K[:, 0, 2] =  axis[:, 1]
	K[:, 1, 0] =  axis[:, 2]
	K[:, 1, 2] = -axis[:, 0]
	K[:, 2, 0] = -axis[:, 1]
	K[:, 2, 1] =  axis[:, 0]

	I = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)

	sin_a = torch.sin(angle).unsqueeze(-1)
	cos_a = torch.cos(angle).unsqueeze(-1)

	R = I + sin_a * K + (1.0 - cos_a) * torch.bmm(K, K)

	# 对非常小角度，直接置为 I 更稳
	small = (torch.norm(x, dim=-1) < eps)
	if small.any():
		R[small] = torch.eye(3, device=x.device, dtype=x.dtype)

	return R.to(orig_dtype)


import torch
from torch.cuda.amp import autocast


# 强制在此函数内部禁用混合精度，全部使用 float32 进行安全计算
@autocast(enabled=False)
def find_rigid_alignment(pos_a, pos_b):
	"""计算从 pos_a 到 pos_b 的最佳刚体变换 (旋转矩阵和平移向量)"""
	orig_dtype = pos_a.dtype

	# 因为关闭了 autocast，这里转为 float32 后绝不会再被系统偷偷转成 Half
	pos_a_f32 = pos_a.to(torch.float32)
	pos_b_f32 = pos_b.to(torch.float32)

	a_mean = pos_a_f32.mean(0)
	b_mean = pos_b_f32.mean(0)
	a_centered = pos_a_f32 - a_mean
	b_centered = pos_b_f32 - b_mean

	cov_mat = a_centered.T @ b_centered
	if not torch.isfinite(cov_mat).all():
		raise RuntimeError("cov_mat has NaN/Inf")
	U, _, Vt = torch.linalg.svd(cov_mat)
	V = Vt.T
	det = torch.linalg.det(V @ U.T)

	if det < 0:
		V[:, -1] = -V[:, -1]

	rot = V @ U.T
	tr = b_mean

	# 计算完毕，安全转回原本的数据类型
	return rot.to(orig_dtype), tr.to(orig_dtype)


@autocast(enabled=False)
def matrix_to_axis_angle(rot_matrix,eps = 1e-6):
	orig_dtype = rot_matrix.dtype
	R = rot_matrix.to(torch.float32)
	if not torch.isfinite(R).all():
		raise RuntimeError("rot_matrix has NaN/Inf before matrix_to_axis_angle")
	# trace and theta
	trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
	cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
	theta = torch.acos(cos_theta)

	# skew-symmetric part
	wx = R[..., 2, 1] - R[..., 1, 2]
	wy = R[..., 0, 2] - R[..., 2, 0]
	wz = R[..., 1, 0] - R[..., 0, 1]
	w = torch.stack([wx, wy, wz], dim=-1)

	sin_theta = torch.sin(theta)
	axis_angle = torch.zeros_like(w)

	# case 1: normal angles
	normal_mask = sin_theta.abs() > eps
	if normal_mask.any():
		axis = w[normal_mask] / (2.0 * sin_theta[normal_mask].unsqueeze(-1))
		axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(eps)
		axis_angle[normal_mask] = axis * theta[normal_mask].unsqueeze(-1)

	# case 2: very small angles -> use first-order approximation
	small_mask = theta.abs() <= eps
	if small_mask.any():
		# for small theta, axis-angle ≈ vee(R - R^T)/2
		axis_angle[small_mask] = 0.5 * w[small_mask]

	# case 3: angles near pi
	pi_mask = (~small_mask) & (~normal_mask)
	if pi_mask.any():
		R_pi = R[pi_mask]
		theta_pi = theta[pi_mask]

		# from diagonal elements
		diag = torch.diagonal(R_pi, dim1=-2, dim2=-1)
		axis = torch.sqrt(torch.clamp((diag + 1.0) / 2.0, min=0.0))

		# fix signs using off-diagonal terms
		axis_x = axis[:, 0]
		axis_y = axis[:, 1]
		axis_z = axis[:, 2]

		axis_y = torch.copysign(axis_y, R_pi[:, 0, 1] + R_pi[:, 1, 0] + eps)
		axis_z = torch.copysign(axis_z, R_pi[:, 0, 2] + R_pi[:, 2, 0] + eps)

		axis = torch.stack([axis_x, axis_y, axis_z], dim=-1)
		axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(eps)

		axis_angle[pi_mask] = axis * theta_pi.unsqueeze(-1)

	return axis_angle.to(orig_dtype)

import numpy as np
import torch

def normalize_mask_rotate(mask_rotate, device):
	"""
	把各种奇怪格式的 mask_rotate 统一成 bool tensor: [num_rotatable_edges, num_atoms]
	"""

	def to_1d_bool_tensor(x):
		# torch tensor
		if isinstance(x, torch.Tensor):
			x = x.to(device).bool()
			if x.dim() == 0:
				x = x.view(1)
			return x

		# numpy array
		if isinstance(x, np.ndarray):
			if x.dtype == np.object_:
				# object array -> 先转 list，再递归
				x = x.tolist()
				return to_1d_bool_tensor(x)
			else:
				x = torch.from_numpy(x).to(device).bool()
				if x.dim() == 0:
					x = x.view(1)
				return x

		# list / tuple
		if isinstance(x, (list, tuple)):
			# 如果里面还是嵌套对象，先逐个展开
			elems = []
			for item in x:
				t = to_1d_bool_tensor(item)
				if t.dim() == 0:
					t = t.view(1)
				if t.dim() > 1:
					t = t.reshape(-1)
				elems.append(t)

			if len(elems) == 0:
				return torch.empty(0, dtype=torch.bool, device=device)

			# 如果这是一个“单条 mask 被包了一层”的情况
			# 例如 [array([True, False, ...])]
			if len(elems) == 1:
				return elems[0]

			# 否则把它们拼成一维
			return torch.cat([e.reshape(-1) for e in elems], dim=0).bool()

		# 标量
		return torch.as_tensor(x, device=device).bool().view(-1)

	# 顶层统一
	if isinstance(mask_rotate, torch.Tensor):
		mask_rotate = mask_rotate.to(device).bool()
		if mask_rotate.dim() == 1:
			mask_rotate = mask_rotate.unsqueeze(0)
		return mask_rotate

	if isinstance(mask_rotate, np.ndarray):
		if mask_rotate.dtype != np.object_:
			mask_rotate = torch.from_numpy(mask_rotate).to(device).bool()
			if mask_rotate.dim() == 1:
				mask_rotate = mask_rotate.unsqueeze(0)
			return mask_rotate
		else:
			mask_rotate = mask_rotate.tolist()

	# 顶层 list/tuple：每个元素对应一条 rotatable edge 的 mask
	if isinstance(mask_rotate, (list, tuple)):
		# ===== 关键补丁：DataLoader 常见情况，外面只包了一层 =====
		if len(mask_rotate) == 1:
			first = mask_rotate[0]

			if isinstance(first, torch.Tensor):
				first = first.to(device).bool()
				if first.dim() == 2:
					return first
				if first.dim() == 1:
					return first.unsqueeze(0)

			if isinstance(first, np.ndarray) and first.dtype != np.object_:
				first = torch.from_numpy(first).to(device).bool()
				if first.dim() == 2:
					return first
				if first.dim() == 1:
					return first.unsqueeze(0)
		# ===== 补丁结束 =====
		rows = []
		for m in mask_rotate:
			row = to_1d_bool_tensor(m).bool().reshape(-1)
			rows.append(row)

		if len(rows) == 0:
			return torch.empty((0, 0), dtype=torch.bool, device=device)

		# 检查每行长度一致
		n_atoms = rows[0].numel()
		for i, r in enumerate(rows):
			if r.numel() != n_atoms:
				raise ValueError(
					f"mask_rotate row {i} has len {r.numel()}, expected {n_atoms}"
				)

		return torch.stack(rows, dim=0).to(device)

	raise TypeError(f"Unsupported mask_rotate type: {type(mask_rotate)}")

def apply_transform(pos_0, tr_pred, rot_pred, tor_pred, batch_idx, t_scale, data):
	pos_working = pos_0.clone()
	device = pos_0.device

	#center = scatter_mean(pos_working, batch_idx, dim=0)
	rot_t = rot_pred * t_scale.view(-1, 1)
	R_mat = axis_angle_to_matrix(rot_t)
	pos_working = (pos_working-pos_working.mean(0))@R_mat + tr_pred + pos_working.mean(0)
	pos_working = pos_working.squeeze(0)
	# --- 第一步：应用扭转 (Torsion) ---
	ligand_edge_key = ('ligand', 'ligand')
	if ligand_edge_key not in data.edge_types:
		ligand_edge_key = ('ligand', 'lig_bond', 'ligand') if ('ligand', 'lig_bond',
															   'ligand') in data.edge_types else None

	if ligand_edge_key is not None:
		edge_mask = getattr(data['ligand'], 'edge_mask', None)
		mask_rotate = getattr(data['ligand'], 'mask_rotate', None)

		if tor_pred is not None and edge_mask is not None and mask_rotate is not None:
			edge_mask = edge_mask.bool()
			all_edges = data[ligand_edge_key].edge_index.t()
			rotatable_edges = all_edges[edge_mask]
			l_batch = batch_idx
			t_per_edge = t_scale[l_batch[rotatable_edges[:, 0]]].view(-1) if rotatable_edges.shape[0] > 0 else torch.empty(0,device=pos_working.device,dtype=pos_working.dtype)

			mask_rotate = normalize_mask_rotate(mask_rotate, pos_working.device)

			n_edge = int(rotatable_edges.shape[0])
			if n_edge > 0:
				tor_pred_aligned = tor_pred.reshape(-1)
				t_per_edge_aligned = t_per_edge.reshape(-1)
				torsion_updates = (tor_pred_aligned * t_per_edge_aligned)
				pos_final = modify_conformer_torsion_angles_torch(
					pos_working, rotatable_edges, mask_rotate, torsion_updates, batch_idx=l_batch
				)
	# --- 第二步：旋转 (Rotation) 与 第三步：平移 (Translation) ---
	# 【原作者逻辑对齐】: (pos - pos_mean) @ rot.T + tr (注意原作者说 tr 直接是新质心)
	# --- 这里我们进行一个判断，因为在其他作者的论文中他们的推理都是经过刚体对齐的，所以这里如果是验证我们就在扭转后加一个刚体对齐来抵消扭转误差
	# -----------------------------------------------------------------------------------------------------------------
	if hasattr(data['ligand'], 'edge_mask') and data['ligand'].edge_mask.sum() > 0:
	# 如果 edge_mask 中有任何一个 True，说明有可旋转键

		R, t = rigid_transform_Kabsch_3D_torch(pos_final.T, pos_working.T)

		pos_final = pos_final @ R.T + t.T
	else:
		pos_final = pos_working

	return pos_final


# ==============================================================================
# 3. 主模型定义 (Base_FM_Model)
# ==============================================================================

class Base_FM_Model(LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.model_macro = get_vector_field(args)
		self.loss_mse = nn.MSELoss(reduction='none')
		self.num_frames = 11  # 固定 11 帧
		# 权重超参数
		self.fm_weight = getattr(args, 'fm_weight', 1.0)
		self.density_weight = getattr(args, 'density_weight', 0.1)
		self.ema_macro = None
		# EMA Setup (如果需要的话，给微观模型加上 EMA，宏观往往不需要因为目标是线性的很简单)
		if self.args.use_ema:
			avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema_rate)
			self.ema_macro = torch.optim.swa_utils.AveragedModel(self.model_macro, multi_avg_fn=avg_fn)
			for param in self.ema_macro.parameters():
				param.requires_grad = False


		self.num_steps = self.num_frames - 1
		self.uniform_dt = 1.0 / self.num_steps

		# 三个自由度分别学自己的时间预算
		self.tr_time_budget = LearnableTimeBudget(self.num_steps, init_mode='middle_fast')
		self.rot_time_budget = LearnableTimeBudget(self.num_steps, init_mode='middle_fast')
		self.tor_time_budget = LearnableTimeBudget(self.num_steps, init_mode='middle_fast')

		self.test_results = []
		self.confidence_model = None

		self.save_hyperparameters(logger=True)
		# 开启手动优化，接管梯度回传，极致节省显存！
		self.automatic_optimization = False
		self.scaler = torch.cuda.amp.GradScaler(init_scale=1.0, growth_interval=1000)

	def forward(self, data, training=False):
		#print(self.ema_macro)
		"""Forward pass through the model."""
		if not training:
			model_macro = self.model_macro
		else:
			model_macro = self.model_macro
		return model_macro(data)

	def training_step(self, data, batch_idx):
		if self.global_step % 50 == 0:
			torch.cuda.empty_cache()

		batch_size = data.num_graphs
		device = data['ligand'].pos.device

		opt = self.optimizers()
		opt.zero_grad()

		with torch.amp.autocast('cuda',dtype=torch.bfloat16, enabled=True):
			# =========================
			# forward macro
			# =========================
			tr_macro, rot_macro, tor_macro = self.forward(data, training=True)
			tor_macro = tor_macro.view(-1)
			#print(f"tor_macro: {tor_macro},data['ligand'].u_tor:{data['ligand'].u_tor}")
			# =========================
			# losses
			# =========================
			loss_fm_tr = self.loss_mse(tr_macro, data['ligand'].u_tr).sum(1).mean(0)
			loss_fm_rot = self.loss_mse(rot_macro, data['ligand'].u_rot).sum(1).mean(0)
			loss_fm_tor = self.loss_mse(tor_macro, data['ligand'].u_tor).sum()
			#print(f"loss_fm_rot:{loss_fm_rot}")
			loss_fm = loss_fm_tr + loss_fm_rot + loss_fm_tor
			total_loss = self.fm_weight * loss_fm

		# =========================
		# backward
		# =========================
		self.scaler.scale(total_loss).backward()

		# detach / del
		tr_macro = tr_macro.detach()
		rot_macro = rot_macro.detach()
		tor_macro = tor_macro.detach()
		del tr_macro, rot_macro, tor_macro

		# 反缩放后再检查梯度
		self.scaler.unscale_(opt)


		# 可选：记录关键层 grad norm
		# self._log_named_grad_norms()

		self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
		self.scaler.step(opt)
		self.scaler.update()

		torch.cuda.empty_cache()

		total_loss_val = (self.fm_weight * loss_fm).detach()

		self.log('train/loss', total_loss_val, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
		self.log('train/loss_fm_tr', loss_fm_tr.detach(), on_step=True, sync_dist=True, batch_size=batch_size)
		self.log('train/loss_fm_rot', loss_fm_rot.detach(), on_step=True, sync_dist=True, batch_size=batch_size)
		self.log('train/loss_fm_tor', loss_fm_tor.detach(), on_step=True, sync_dist=True, batch_size=batch_size)

		return total_loss.detach()

	def validation_step(self, data, batch_idx):
		batch_size = data.num_graphs
		device = data['ligand'].pos.device
		l_batch = data['ligand'].batch

		x0 = data['ligand'].initial_pos.clone()
		gt_pos = data['ligand'].gt_pos


		docking_rmsd_init = torch.sqrt(scatter(torch.sum((x0-gt_pos) ** 2, -1), l_batch, reduce='mean') + 1e-8).mean()
		#print(f"rmsd_init:{docking_rmsd_init}")
		current_pos = x0.clone()

		total_steps = self.num_frames - 1
		total_steps = total_steps
		dt = 1.0 / total_steps  # 积分时间步长

		self.eval()
		with torch.no_grad():
			for i in range(0, total_steps):
				t_val = i / total_steps
				t_tensor = torch.full((batch_size,), t_val, device=device)

				data['ligand'].pos = current_pos

				data.complex_t = {'tr': t_tensor, 'rot': t_tensor, 'tor': t_tensor}
				data['ligand'].node_t = {'tr': t_tensor[l_batch], 'rot': t_tensor[l_batch], 'tor': t_tensor[l_batch]}

				if 'receptor' in data.node_types:
					r_batch = data['receptor'].batch if hasattr(data['receptor'], 'batch') else torch.zeros(
						data['receptor'].num_nodes, device=device).long()
					data['receptor'].node_t = {'tr': t_tensor[r_batch], 'rot': t_tensor[r_batch],
											   'tor': t_tensor[r_batch]}

				with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
					# 宏观推理
					tr_v, rot_v, tor_v = self.forward(data, training=False)
				#print(f"===========\nODE:{i},tr_v:{tr_v}")#, rot_v:{rot_v}, tor_v:{tor_v}")
				#print(f"ODE:{i},u_tr:{data['ligand'].u_tr}\n===========")#,u_rot:{data['ligand'].u_rot},u_tor:{data['ligand'].u_tor}")

				alpha = (tr_v*data['ligand'].u＿tr) / (data['ligand'].u＿tr * data['ligand'].u＿tr)
				#print(f"Ode:{i},alpha:{alpha}")
				has_torsion = False
				if hasattr(data['ligand'], 'edge_mask'):
					# 如果 edge_mask 中有任何一个 True，说明有可旋转键
					if data['ligand'].edge_mask.sum() > 0:
						has_torsion = True

				if not has_torsion:
					# 如果没有扭转键，将 tor_v 重置为 0，防止噪声干扰
					tor_v = torch.zeros_like(tor_v)

				# 【核心】：推理阶段更新位置，必须是 速度 * dt！
				dummy_t_scale = torch.ones(batch_size, device=device)

				#current_pos_mean = current_pos.mean(0)
				current_pos = apply_transform(
					current_pos,
					tr_v * dt,  # 平移速度 * 时间步长 = 实际位移
					rot_v * dt,  # 旋转速度 * 时间步长 = 实际旋转角
					tor_v * dt,  # 扭转速度 * 时间步长 = 实际扭转形变
					l_batch, dummy_t_scale, data
				)

				#print(f"关系2：ODE：{i}，{current_pos.mean(0) - current_pos_mean-tr_v*dt}")

			data['ligand'].pos = current_pos

			# 检查新坐标有没有异常
			if torch.isnan(current_pos).any() or torch.isinf(current_pos).any():
				print(f"⚠️ 推理坐标异常！epoch={self.current_epoch}, global_step={self.global_step}, val_batch_idx={batch_idx}, rollout_step={i}")
				print(f"current_pos abs max: {current_pos.abs().max().item():.6f}")
				print(f"tr_v abs max: {tr_v.abs().max().item():.6f}")
				print(f"rot_v abs max: {rot_v.abs().max().item():.6f}")
				print(f"tor_v abs max: {tor_v.abs().max().item():.6f}")
				raise RuntimeError("Validation current_pos has NaN/Inf")

			# 最终计算 RMSD

			docking_rmsd, mode = compute_symmetry_rmsd_from_complex_graph(data)
			#print(f"质心差：{data['ligand'].pos.mean(0)-gt_pos.mean(0)}")
			print("rmsd =", docking_rmsd)


			self.log('val/docking_rmsd', docking_rmsd, on_epoch=True, batch_size=batch_size, sync_dist=True)
			self.log('val/loss', docking_rmsd, on_epoch=True, batch_size=batch_size, sync_dist=True)

		return docking_rmsd


	def test_step(self, data, batch_idx):
		if data.complex_name[0] != "6a72_1_9UX_0":
			return None                                 #第274个有问题，第273个复合物名字叫2hnu_2_PHE-TYR_1
		batch_size = data.num_graphs
		device = data['ligand'].pos.device
		l_batch = data['ligand'].batch
		a_batch = data['atom'].batch if hasattr(data['atom'], 'batch') else torch.zeros(
			data['atom'].num_nodes, device=device).long()

		init_pos = data.sample_pos.clone()
		gt_pos = data['ligand'].gt_pos
		heavy_atoms = data['ligand'].gt_pos.shape[0]
		#print(f"gt_pos:{data['ligand'].gt_pos.shape},heavy_atoms:{heavy_atoms}")
		rmsds_list = []
		confidence_list = []
		centroid_distances = []
		valid_rates = []
		physical_valid_rates = []
		total_steps = self.num_frames - 1
		complex_energy_list =[]
		tr_dt_all = self.tr_time_budget.get_dt().detach().to(device)
		#print(f"tor_dt_all:{tr_dt_all}")
		#rot_dt_all = self.rot_time_budget.get_dt().detach().to(device)
		#tor_dt_all = self.tor_time_budget.get_dt().detach().to(device)

		tor_dt_all = tr_dt_all
		#tor_dt_all = tor_dt_all.repeat_interleave(2)/2
		#tor_dt_all = tor_dt_all.reshape(5, 2).sum(dim=1)
		rot_dt_all = tr_dt_all
		#rot_dt_all = rot_dt_all / rot_dt_all.sum()
		tor_dt_all = tor_dt_all / tor_dt_all.sum()
		tr_dt_all = torch.tensor(
			[0.14, 0.13, 0.12, 0.11, 0.10,
			 0.10, 0.09, 0.08, 0.07, 0.06],
			device="cuda:0"
		)
		#tr_dt_all = tr_dt_all.repeat_interleave(2) / 2

		#tr_dt_all = tr_dt_all.reshape(5, 2).sum(dim=1)
		tr_dt_all = tr_dt_all / tr_dt_all.sum()


		# rot_dt_all = torch.tensor(
		# 	[0.13, 0.13, 0.12, 0.12, 0.11,
		# 	 0.10, 0.09, 0.08, 0.07, 0.05],
		# 	device="cuda:0"
		# )
		#rot_dt_all = rot_dt_all.repeat_interleave(2) / 2

		#rot_dt_all = rot_dt_all.reshape(5, 2).sum(dim=1)
		rot_dt_all = rot_dt_all / rot_dt_all.sum()

		# tor_dt_all = torch.tensor(
		# 	[0.03, 0.04, 0.05, 0.07, 0.09,
		# 	 0.12, 0.15, 0.16, 0.16, 0.18],
		# 	device="cuda:0"
		# )
		# tor_dt_all = tor_dt_all / tor_dt_all.sum()

		#=====================================
		records = []
		dt = 0.1
		#print(f"complex_name: {data['complex_name'][0]}")
		self.eval()
		with torch.no_grad():
			for n in range(init_pos.shape[0]):
				current_pos = init_pos[n] - data.complex_center
				t_test = 1.0
				# if n!=29:
				# 	continue
				for i in range(total_steps):

					data['ligand'].pos = current_pos
					# t_val = [dt]*10
					# t_val =torch.tensor(t_val, device="cuda:0")
					#t_vall = i/total_steps

					with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
						# tr_v, rot_v, tor_v = self.split_time(data, t_val, i)

						tr_v,_,_ = self.split_time(data, tr_dt_all,i)
						_,rot_v,_ = self.split_time(data, rot_dt_all,i)
						_,_,tor_v = self.split_time(data, tor_dt_all,i)


					has_torsion = False
					if hasattr(data['ligand'], 'edge_mask') and data['ligand'].edge_mask.sum() > 0:
						has_torsion = True

					if not has_torsion:
						tor_v = torch.zeros_like(tor_v)

					dt_tr = tr_dt_all[i]
					dt_rot = rot_dt_all[i]
					dt_tor = tor_dt_all[i]
					dummy_t_scale = torch.ones(batch_size, device=device)
					# print(f"ODE:{i},tr_v:{tr_v},rot_v:{rot_v},tor_v:{tor_v}")
					# print(f"ODE:{i},real_tr:{data['ligand'].u_tr},real_rot:{data['ligand'].u_rot},real_tor:{data['ligand'].u_tor}")
					current_pos = apply_transform(
						current_pos,
						tr_v * dt_tr,
						rot_v * dt_rot,
						tor_v * dt_tor,
						l_batch,
						dummy_t_scale,
						data
					)
				# 重新赋值给 confidence model0
				t_tensor = torch.full((batch_size,), t_test, device=device)
				data.complex_t = {'tr': t_tensor, 'rot': t_tensor, 'tor': t_tensor}
				data['ligand'].node_t = {
					'tr': t_tensor[l_batch],
					'rot': t_tensor[l_batch],
					'tor': t_tensor[l_batch]
				}

				if 'receptor' in data.node_types:
					r_batch = data['receptor'].batch if hasattr(data['receptor'], 'batch') else torch.zeros(
						data['receptor'].num_nodes, device=device).long()
					data['receptor'].node_t = {
						'tr': t_tensor[r_batch],
						'rot': t_tensor[r_batch],
						'tor': t_tensor[r_batch]
					}

				data['atom'].node_t = {
					'tr': t_tensor[a_batch],
					'rot': t_tensor[a_batch],
					'tor': t_tensor[a_batch]
				}

				data['ligand'].pos = current_pos
				#pose_file = None#f"rescore_sample_{n}.sdf"      #rescore
				_,valid_rate,physical_valid_rate = PB_valid(data.complex_name[0], data['ligand'].pos+data.complex_center,pose_file=None)
				#valid_rate, physical_valid_rate = torch.tensor([0]), torch.tensor([0])
				docking_rmsd, mode = compute_symmetry_rmsd_from_complex_graph(data)
				conf = self.confidence_model(data)[:, 0]

				centroid_dist = torch.sqrt(((data['ligand'].pos.mean(0) - gt_pos.mean(0)) ** 2).sum())

				rmsds_list.append(float(docking_rmsd))
				confidence_list.append(float(conf.item()))
				centroid_distances.append(float(centroid_dist.item()))
				valid_rates.append(float(valid_rate.item()))
				physical_valid_rates.append(float(physical_valid_rate.item()))
				# 统计保存数据
		# 		records.append({
		# 			"complex_name": data.complex_name[0],
		# 			"pose_id": n,
		# 			"confidence": float(conf.item()),
		# 			"rmsd": float(docking_rmsd),
		# 			"pb_valid": float(valid_rate.item()),
		# 			"physical_valid": float(physical_valid_rate.item()),
		# 			"centroid_dist": float(centroid_dist.item()),
		# 		})
		# pd.DataFrame(records).to_csv(f"D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set/{data.complex_name[0]}/metadata.csv", index=False)
		# write_complex_manifest(complex_dir = f"D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set/{data.complex_name[0]}",
		# 					   complex_id= data.complex_name[0],receptor_name= f"{data.complex_name[0]}_protein",poses_name= f"{data.complex_name[0]}_ligands_process")
		# merge_pose_sdfs(pose_files = f"D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set/{data.complex_name[0]}/rescoring_poses",
		# 				output_sdf=f"D:/PythonProject medicine/project_mine/data/posebusters_benchmark_set/{data.complex_name[0]}/{data.complex_name[0]}_ligands_process.sdf")
		# 统计保存数据
		confidence_arr = np.array(confidence_list)
		rmsds_arr = np.array(rmsds_list)
		print(f"complex_name: {data['complex_name'][0]},key_numbers:{tor_v.shape if has_torsion else 0}")
		#print(f"rmsds:{rmsds_arr}")
		#print(f"score:{confidence_arr}")
		centroid_arr = np.array(centroid_distances)
		valid_rates_arr = np.array(valid_rates)
		physical_valid_rates_arr = np.array(physical_valid_rates)

		re_order = np.argsort(confidence_arr)[::-1]
		confidence_arr = confidence_arr[re_order]
		print(f"score:{confidence_arr}")
		#==================RESCORE_GNINA===================
		# import json
		# with open(r"D:\PythonProject medicine\project_mine\rescore_gnina_colab\content_5\posebusters_gnina_scores_8888\reorder_map.json","r", encoding="utf-8") as f:
		# 	reorder_map = json.load(f)
		# re_order = np.array(reorder_map[data.complex_name[0]], dtype=np.int64)
		# confidence_arr = confidence_arr[re_order]       ##如果上面是用的重打分那这里的分数并不代表真正的分数，出于某种原因我们没有修改它，但这不影响实验结果
		# ==================RESCORE_GNINA===================
		rmsds_arr = rmsds_arr[re_order]
		print(f"rmsds:{rmsds_arr}")
		centroid_arr = centroid_arr[re_order]
		#print(f"centroid_distance:{centroid_arr}")
		valid_rates_arr = valid_rates_arr[re_order]
		physical_valid_rates_arr = physical_valid_rates_arr[re_order]

		top1_rmsd = float(rmsds_arr[0])
		top5_best_rmsd = float(np.min(rmsds_arr[:5])) if len(rmsds_arr) >= 5 else float(np.min(rmsds_arr))
		top10_best_rmsd = float(np.min(rmsds_arr[:10])) if len(rmsds_arr) >= 10 else float(np.min(rmsds_arr))
		best_rmsd = float(np.min(rmsds_arr))
		rotatable_bonds = tor_v.shape if has_torsion else 0
		# # top1 固定第0号
		# top1_rmsd = float(rmsds_arr[0])
		# top1_conf = float(confidence_arr[0])
		#
		# # top5 最优
		# slice5 = rmsds_arr[:5] if len(rmsds_arr) >= 5 else rmsds_arr
		# idx5 = np.argmin(slice5)
		# top5_best_rmsd = float(slice5[idx5])
		# top5_best_conf = float(confidence_arr[idx5])
		#
		# # top10 最优
		# slice10 = rmsds_arr[:10] if len(rmsds_arr) >= 10 else rmsds_arr
		# idx10 = np.argmin(slice10)
		# top10_best_rmsd = float(slice10[idx10])
		# top10_best_conf = float(confidence_arr[idx10])
		#
		# # 全局最优
		# idx_best = np.argmin(rmsds_arr)
		# best_rmsd = float(rmsds_arr[idx_best])
		# best_conf = float(confidence_arr[idx_best])

		result = {
			"complex_name": data.complex_name[0] if isinstance(data.complex_name, (list, tuple)) else str(
				data.complex_name),
			"num_samples": int(len(rmsds_arr)),
			"top1_rmsd": top1_rmsd,
			"top5_best_rmsd": top5_best_rmsd,
			"top10_best_rmsd": top10_best_rmsd,
			"best_rmsd": best_rmsd,
			"top1_confidence": float(confidence_arr[0]),
			"success_top1_2A": int(top1_rmsd < 2.0),
			"success_top5_2A": int(top5_best_rmsd < 2.0),
			"success_top10_2A": int(top10_best_rmsd < 2.0),
			"success_best_2A": int(best_rmsd < 2.0),
			"success_top1_5A": int(top1_rmsd < 5.0),
			"success_top5_5A": int(top5_best_rmsd < 5.0),
			"success_top10_5A": int(top10_best_rmsd < 5.0),
			"success_best_5A": int(best_rmsd < 5.0),
			"top1_centroid_distance": float(centroid_arr[0]),
			"best_centroid_distance": float(np.min(centroid_arr)),

			# 同时满足 RMSD < 2A 和完整 PoseBusters 合理性
			"success_top1_2A_posebusters_valid": int((rmsds_arr[0] < 2.0) and bool(valid_rates_arr[0])),
			"success_top5_2A_posebusters_valid": int(
				np.any((rmsds_arr[:5] < 2.0) & (valid_rates_arr[:5].astype(bool)))
			),
			"success_best_2A_posebusters_valid": int(
				np.any((rmsds_arr < 2.0) & (valid_rates_arr.astype(bool)))
			),

			# 同时满足 RMSD < 2A 和物理合理性
			"success_top1_2A_physical_valid": int((rmsds_arr[0] < 2.0) and bool(physical_valid_rates_arr[0])),
			"success_top5_2A_physical_valid": int(
				np.any((rmsds_arr[:5] < 2.0) & (physical_valid_rates_arr[:5].astype(bool)))
			),
			"success_best_2A_physical_valid": int(
				np.any((rmsds_arr < 2.0) & (physical_valid_rates_arr.astype(bool)))
			),

			# "pb_valid_top1": int(valid_rates_arr[0].astype(bool)),
			# "pb_valid_count_40": int(np.sum(valid_rates_arr.astype(bool))),
			# "top1_conf":top1_conf,
			# "top5_best_conf":top5_best_conf,
			# "top10_best_conf":top10_best_conf,
			# "best_conf":best_conf,
			# "heavy_atoms":heavy_atoms,
			# "rotatable_bonds": rotatable_bonds,

		}

		#"sorted_rmsds": rmsds_arr.tolist(),"sorted_confidences": confidence_arr.tolist(),

		self.test_results.append(result)



		return result

	def split_time(self,data,t_val,i):

		batch_size = data.num_graphs
		device = data['ligand'].pos.device
		l_batch = data['ligand'].batch
		t_cat = torch.cat([torch.tensor([0]).to(device), t_val])
		t_val = t_cat[:i + 1].sum().item()
		self.eval()
		with torch.no_grad():
			t_tensor = torch.full((batch_size,), t_val, device=device)

			data.complex_t = {'tr': t_tensor, 'rot': t_tensor, 'tor': t_tensor}
			data['ligand'].node_t = {
				'tr': t_tensor[l_batch],
				'rot': t_tensor[l_batch],
				'tor': t_tensor[l_batch]
			}

			if 'receptor' in data.node_types:
				r_batch = data['receptor'].batch if hasattr(data['receptor'], 'batch') else torch.zeros(
					data['receptor'].num_nodes, device=device).long()
				data['receptor'].node_t = {
					'tr': t_tensor[r_batch],
					'rot': t_tensor[r_batch],
					'tor': t_tensor[r_batch]
				}

			with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
				tr_v, rot_v, tor_v = self.forward(data, training=False)
		return tr_v,rot_v,tor_v

	def on_test_start(self):
		self.test_results = []

		cfg = load_cfg(r'D:\PythonProject medicine\project_mine\workdir\diffdock_confidence_model\model_parameters.yml')
		self.confidence_model = get_diffdock_confidence_model(cfg)

		state_dict = torch.load(
			r'D:\PythonProject medicine\project_mine\workdir\diffdock_confidence_model\best_model_epoch75.pt',
			map_location='cpu'
		)
		self.confidence_model.load_state_dict(state_dict, strict=False)
		self.confidence_model = self.confidence_model.to(self.device)
		self.confidence_model.eval()

		print("confidence model loaded successfully.")


	def on_test_epoch_end(self):
		import os
		import pandas as pd

		environment = "base"#"base"#"Ours"
		sample = 40
		test_seed = 42
		p2rank = "p2rank"
		astex = "astex"
		if len(self.test_results) == 0:
			print("No test results found.")
			return

		df = pd.DataFrame(self.test_results)

		save_dir = r"D:\PythonProject medicine\project_mine\test_outputs"
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"test_results_{environment}_{sample}_seed{test_seed}_basetestnono.csv")
		df.to_csv(save_path, index=False, encoding="utf-8-sig")

		print(f"Test results saved to: {save_path}")
		#================一个实验做完可以注释掉================
		# 挑选你需要的8列，顺序固定
		# export_cols= [
		# 	"top1_rmsd",
		# # 	"heavy_atoms",
		# # 	"rotatable_bonds"
		# 	# "top1_conf",
		# 	"top5_best_rmsd",
		# 	# "top5_best_conf",
		# 	"top10_best_rmsd",
		# 	# "top10_best_conf",
		# 	"best_rmsd",
		# 	# "best_conf"
		# 	"success_top1_2A_posebusters_valid",
		# 	"success_top5_2A_posebusters_valid",
		# 	"success_best_2A_posebusters_valid"
		#
		# ]
		# df_export= df[export_cols].copy()
		# one_experiment_path= os.path.join(save_dir, f"CI_Ours_{test_seed}.csv")
		# df_export.to_csv(one_experiment_path, index=False)
		# ================一个实验做完可以注释掉================

		top1_2A = df["success_top1_2A"].sum()
		top5_2A = df["success_top5_2A"].sum()
		top10_2A = df["success_top10_2A"].sum()
		best_2A = df["success_best_2A"].sum()

		top1_5A = df["success_top1_5A"].sum()
		top5_5A = df["success_top5_5A"].sum()
		top10_5A = df["success_top10_5A"].sum()
		best_5A = df["success_best_5A"].sum()

		top1_2A_pb = df["success_top1_2A_posebusters_valid"].sum()
		top5_2A_pb = df["success_top5_2A_posebusters_valid"].sum()
		best_2A_pb = df["success_best_2A_posebusters_valid"].sum()

		top1_2A_phy = df["success_top1_2A_physical_valid"].sum()
		top5_2A_phy = df["success_top5_2A_physical_valid"].sum()
		best_2A_phy = df["success_best_2A_physical_valid"].sum()

		total = len(df)

		def _print_table(title, rows):
			print(f"\n{title}")
			col_widths = [max(len(str(x)) for x in col) for col in zip(*rows)]
			for i, row in enumerate(rows):
				line = " | ".join(str(val).ljust(col_widths[j]) for j, val in enumerate(row))
				print(line)
				if i == 0:
					print("-+-".join("-" * w for w in col_widths))

		print(f"\nTotal complexes: {total}")

		overall_rows = [
			["Metric", "Count", "Rate"],
			["Top1 < 2A", f"{top1_2A}/{total}", f"{top1_2A / total:.4f}"],
			["Top5 < 2A", f"{top5_2A}/{total}", f"{top5_2A / total:.4f}"],
			["Top10 < 2A", f"{top10_2A}/{total}", f"{top10_2A / total:.4f}"],
			["Best < 2A", f"{best_2A}/{total}", f"{best_2A / total:.4f}"],
			["Top1 < 5A", f"{top1_5A}/{total}", f"{top1_5A / total:.4f}"],
			["Top5 < 5A", f"{top5_5A}/{total}", f"{top5_5A / total:.4f}"],
			["Top10 < 5A", f"{top10_5A}/{total}", f"{top10_5A / total:.4f}"],
			["Best < 5A", f"{best_5A}/{total}", f"{best_5A / total:.4f}"],
			["Top1 < 2A + PB valid", f"{top1_2A_pb}/{total}", f"{top1_2A_pb / total:.4f}"],
			["Top5 < 2A + PB valid", f"{top5_2A_pb}/{total}", f"{top5_2A_pb / total:.4f}"],
			["Best < 2A + PB valid", f"{best_2A_pb}/{total}", f"{best_2A_pb / total:.4f}"],

			["Top1 < 2A + Physical valid", f"{top1_2A_phy}/{total}", f"{top1_2A_phy / total:.4f}"],
			["Top5 < 2A + Physical valid", f"{top5_2A_phy}/{total}", f"{top5_2A_phy / total:.4f}"],
			["Best < 2A + Physical valid", f"{best_2A_phy}/{total}", f"{best_2A_phy / total:.4f}"],

		]
		_print_table("===== Overall Metrics =====", overall_rows)

		rmsd_rows = [
			["Metric", "Q25", "Q50", "Q75", "Mean"],
			[
				"Top1 RMSD",
				f"{df['top1_rmsd'].quantile(0.25):.4f}",
				f"{df['top1_rmsd'].quantile(0.50):.4f}",
				f"{df['top1_rmsd'].quantile(0.75):.4f}",
				f"{df['top1_rmsd'].mean():.4f}",
			],
			[
				"Best RMSD",
				f"{df['best_rmsd'].quantile(0.25):.4f}",
				f"{df['best_rmsd'].quantile(0.50):.4f}",
				f"{df['best_rmsd'].quantile(0.75):.4f}",
				f"{df['best_rmsd'].mean():.4f}",
			],
		]

		if "top5_best_rmsd" in df.columns:
			rmsd_rows.append([
				"Top5 Best RMSD",
				f"{df['top5_best_rmsd'].quantile(0.25):.4f}",
				f"{df['top5_best_rmsd'].quantile(0.50):.4f}",
				f"{df['top5_best_rmsd'].quantile(0.75):.4f}",
				f"{df['top5_best_rmsd'].mean():.4f}",
			])

		if "top10_best_rmsd" in df.columns:
			rmsd_rows.append([
				"Top10 Best RMSD",
				f"{df['top10_best_rmsd'].quantile(0.25):.4f}",
				f"{df['top10_best_rmsd'].quantile(0.50):.4f}",
				f"{df['top10_best_rmsd'].quantile(0.75):.4f}",
				f"{df['top10_best_rmsd'].mean():.4f}",
			])

		_print_table("===== RMSD Statistics =====", rmsd_rows)

		centroid_rows = [
			["Metric", "Q25", "Q50", "Q75", "Mean"]
		]

		if "top1_centroid_distance" in df.columns:
			centroid_rows.append([
				"Top1 Centroid Dist",
				f"{df['top1_centroid_distance'].quantile(0.25):.4f}",
				f"{df['top1_centroid_distance'].quantile(0.50):.4f}",
				f"{df['top1_centroid_distance'].quantile(0.75):.4f}",
				f"{df['top1_centroid_distance'].mean():.4f}",
			])

		if "best_centroid_distance" in df.columns:
			centroid_rows.append([
				"Best Centroid Dist",
				f"{df['best_centroid_distance'].quantile(0.25):.4f}",
				f"{df['best_centroid_distance'].quantile(0.50):.4f}",
				f"{df['best_centroid_distance'].quantile(0.75):.4f}",
				f"{df['best_centroid_distance'].mean():.4f}",
			])

		if len(centroid_rows) > 1:
			_print_table("===== Centroid Distance Statistics =====", centroid_rows)

		summary = {
			"environment": environment,
			"sample": sample,
			"total": total,

			"top1_2A": top1_2A,
			"top5_2A": top5_2A,
			"top10_2A": top10_2A,
			"best_2A": best_2A,

			"top1_5A": top1_5A,
			"top5_5A": top5_5A,
			"top10_5A": top10_5A,
			"best_5A": best_5A,

			"top1_2A_pb": top1_2A_pb,
			"top5_2A_pb": top5_2A_pb,
			"best_2A_pb": best_2A_pb,

			"top1_2A_phy": top1_2A_phy,
			"top5_2A_phy": top5_2A_phy,
			"best_2A_phy": best_2A_phy,


			"top1_rmsd_q25": df["top1_rmsd"].quantile(0.25),
			"top1_rmsd_q50": df["top1_rmsd"].quantile(0.50),
			"top1_rmsd_q75": df["top1_rmsd"].quantile(0.75),
			"top1_rmsd_mean": df["top1_rmsd"].mean(),

			"best_rmsd_q25": df["best_rmsd"].quantile(0.25),
			"best_rmsd_q50": df["best_rmsd"].quantile(0.50),
			"best_rmsd_q75": df["best_rmsd"].quantile(0.75),
			"best_rmsd_mean": df["best_rmsd"].mean(),
		}

		summary_df = pd.DataFrame([summary])
		summary_path = os.path.join(save_dir, f"summary_{environment}_{sample}_seed{test_seed}_basetestnono.csv")
		summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
		print(f"Summary saved to: {summary_path}")

		# =========fill_experiment=========
		# ---------- 新增：每个复合物的 best_rmsd CSV + PB 汇总表 ----------
		required_cols = ["complex_name", "best_rmsd", "pb_valid_top1", "pb_valid_count_40"]
		if all(col in df.columns for col in required_cols):
			# 1. 导出 best_rmsd 明细 CSV（只有名字和 best_rmsd）
			detail_df = df[["complex_name", "best_rmsd"]].copy()
			detail_filename = f"fill_experiment_best_rmsd_detail_{environment}_{sample}_seed{test_seed}_basetestnono.csv"
			detail_save_path = os.path.join(save_dir, detail_filename)
			detail_df.to_csv(detail_save_path, index=False, encoding="utf-8-sig")
			print(f"Per-complex best RMSD saved to: {detail_save_path}")

			# 2. 计算两个 PB 汇总比例
			pb_top1_total = df["pb_valid_top1"].sum()  # 通过的复合物数
			pb_top1_rate = pb_top1_total / total  # 比例

			pb_count_total = df["pb_valid_count_40"].sum()  # 所有复合物 40 个采样中通过的次数总和
			pb_count_rate = pb_count_total / (total * sample)  # 总通过次数 / 总采样次数

			# 3. 打印汇总表格
			pb_summary_rows = [
				["Metric", "Count", "Rate"],
				[
					"PB Valid Top1 (any RMSD)",
					f"{pb_top1_total}/{total}",
					f"{pb_top1_rate:.4f}",
				],
				[
					"PB Valid Count/Total (40 samples)",
					f"{pb_count_total}/{total * sample}",
					f"{pb_count_rate:.4f}",
				],
			]
			_print_table("===== PoseBusters Validity (Independent of RMSD) =====", pb_summary_rows)
		else:
			missing = [c for c in required_cols if c not in df.columns]
			# assert not missing, f"⚠️ 以下列缺失，无法生成 PB 明细表：{missing}"
		# =================================


	def configure_optimizers(self):
		params = [p for p in self.parameters() if p.requires_grad]

		optimizer = torch.optim.AdamW(
			params,
			lr=self.args.lr,
			weight_decay=getattr(self.args, 'w_decay', 0.0)
		)

		t_max = getattr(self.args, "max_epochs", 1000)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=t_max,
			eta_min=1e-6
		)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "epoch",
				"frequency": 1,
				"name": "cosine_anneal"
			}
		}


	def on_train_epoch_end(self):
		opt = self.optimizers()
		current_lr = opt.param_groups[0]["lr"]
		self.log("train/lr", current_lr, prog_bar=True, sync_dist=True)

import yaml
def load_cfg(yml_path):
	class CFG:
		def __init__(self, d):
			self.__dict__.update(d)
	with open(yml_path, 'r',encoding='utf-8') as f:
		data = yaml.safe_load(f)
	return CFG(data)

@autocast(enabled=False)
def rigid_transform_Kabsch_3D_torch(A, B):
	# R = 3x3 rotation matrix, t = 3x1 column vector
	# This already takes residue identity into account.

	assert A.shape[1] == B.shape[1]
	num_rows, num_cols = A.shape
	if num_rows != 3:
		raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
	num_rows, num_cols = B.shape
	if num_rows != 3:
		raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


	# find mean column wise: 3 x 1
	centroid_A = torch.mean(A, axis=1, keepdims=True)
	centroid_B = torch.mean(B, axis=1, keepdims=True)

	# subtract mean
	Am = A - centroid_A
	Bm = B - centroid_B

	H = Am @ Bm.T

	# find rotation
	U, S, Vt = torch.linalg.svd(H)

	R = Vt.T @ U.T
	# special reflection case
	if torch.linalg.det(R) < 0:
		# print("det(R) < R, reflection detected!, correcting for it ...")
		SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
		R = (Vt.T @ SS) @ U.T
	assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

	t = -R @ centroid_A + centroid_B
	return R, t


import torch
from scipy.optimize import linear_sum_assignment

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# =========================
# 直接放一起，别依赖外部
# =========================
allowable_features = {
	'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
	'possible_chirality_list': [
		'CHI_UNSPECIFIED',
		'CHI_TETRAHEDRAL_CW',
		'CHI_TETRAHEDRAL_CCW',
		'CHI_OTHER'
	],
	'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
	'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
	'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
	'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
	'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
	'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
	'possible_hybridization_list': [
		'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
	],
	'possible_is_aromatic_list': [False, True],
	'possible_is_in_ring3_list': [False, True],
	'possible_is_in_ring4_list': [False, True],
	'possible_is_in_ring5_list': [False, True],
	'possible_is_in_ring6_list': [False, True],
	'possible_is_in_ring7_list': [False, True],
	'possible_is_in_ring8_list': [False, True],
	'possible_amino_acids': [
		'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
		'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
		'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP',
		'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'
	],
	'possible_atom_type_2': [
		'C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH',
		'NZ', 'O*', 'OD', 'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'
	],
	'possible_atom_type_3': [
		'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG',
		'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE',
		'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2',
		'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'
	],
}

def compute_symmetry_rmsd_from_complex_graph(data):
	"""
	对单个 complex_graph / dataset sample 计算 symmetry-corrected RMSD

	优先级:
	1. 优先用别人现成的 strict graph-isomorphism 方法: symmrmsd(...)
	2. strict 失败 -> type-aware Hungarian
	3. 再失败 -> coords-only Hungarian

	依赖:
	- 你前面已经有 symmrmsd(...) 这个函数
	- data['ligand'].pos
	- data['ligand'].gt_pos
	- 最好有 data['ligand'].x
	- 最好有 data['ligand', 'lig_bond', 'ligand'].edge_index

	返回:
		rmsd: float
		mode: str
	"""

	lig = data['ligand']

	if not hasattr(lig, 'pos') or not hasattr(lig, 'gt_pos'):
		raise ValueError("data['ligand'] 必须同时包含 pos 和 gt_pos")

	pred = lig.pos.detach().cpu().numpy()
	gt = lig.gt_pos.detach().cpu().numpy()

	if pred.shape != gt.shape:
		raise ValueError(f"pred/gt shape 不一致: pred={pred.shape}, gt={gt.shape}")
	if pred.ndim != 2 or pred.shape[1] != 3:
		raise ValueError(f"坐标 shape 应为 [N,3]，现在是 {pred.shape}")

	# =========================================================
	# 1) 优先用别人现成 strict graph-isomorphism 版: symmrmsd
	# =========================================================
	try:
		if not hasattr(lig, 'x'):
			raise ValueError("ligand.x 不存在，无法解码 atomic numbers")

		x = lig.x.detach().cpu().numpy()
		if x.ndim != 2 or x.shape[1] < 1:
			raise ValueError(f"ligand.x shape 非法: {x.shape}")

		# 你的预处理里 ligand.x[:,0] 是 atomic number 在 vocab 中的索引
		atomic_num_indices = x[:, 0].astype(np.int64)
		atomic_num_vocab = allowable_features['possible_atomic_num_list']

		atomicnums = []
		for idx in atomic_num_indices:
			if idx < 0 or idx >= len(atomic_num_vocab):
				raise ValueError(f"atomic number index 越界: {idx}")
			val = atomic_num_vocab[idx]
			if val == 'misc':
				raise ValueError("发现 misc atomic number，strict symmrmsd 无法安全使用")
			atomicnums.append(int(val))
		atomicnums = np.asarray(atomicnums, dtype=np.int64)

		# ligand bond edge_index
		edge_key = ('ligand', 'lig_bond', 'ligand')
		if hasattr(data, 'edge_types') and edge_key in data.edge_types:
			edge_index = data[edge_key].edge_index
		elif hasattr(lig, 'edge_index'):
			edge_index = lig.edge_index
		else:
			raise ValueError("找不到 ligand 的 edge_index")

		edge_index = edge_index.detach().cpu().numpy()
		if edge_index.shape[0] != 2:
			raise ValueError(f"edge_index shape 非法: {edge_index.shape}")

		num_nodes = gt.shape[0]
		adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int64)
		src, dst = edge_index
		adjacency[src, dst] = 1
		adjacency[dst, src] = 1

		# 用别人现成的方法
		rmsd = symmrmsd(
			coordsref=gt,
			coords=pred,
			apropsref=atomicnums,
			aprops=atomicnums,
			amref=adjacency,
			am=adjacency,
			center=False,      # docking pose 常见设定：不做额外对齐
			minimize=False,    # 不做旋转最小化
			cache=True,
		)
		return float(rmsd), "strict_graph"

	except Exception as e:
		strict_error = e

	# ==========================================
	# 2) strict 失败 -> type-aware Hungarian
	# ==========================================
	try:
		if not hasattr(lig, 'x'):
			raise ValueError("ligand.x 不存在，无法做 type-aware Hungarian")

		atom_types = lig.x[:, 0].detach().cpu().numpy().astype(np.int64)

		dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
		mismatch_penalty = 1e6
		dist = dist + mismatch_penalty * (atom_types[:, None] != atom_types[None, :])

		row_ind, col_ind = linear_sum_assignment(dist)

		gt_match = np.empty_like(gt)
		gt_match[row_ind] = gt[col_ind]

		rmsd = np.sqrt(np.mean(np.sum((pred - gt_match) ** 2, axis=-1)))
		return float(rmsd), f"hungarian_type (strict failed: {strict_error})"

	except Exception:
		pass

	# ==========================================
	# 3) 最后 -> coords-only Hungarian
	# ==========================================
	dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
	row_ind, col_ind = linear_sum_assignment(dist)

	gt_match = np.empty_like(gt)
	gt_match[row_ind] = gt[col_ind]

	rmsd = np.sqrt(np.mean(np.sum((pred - gt_match) ** 2, axis=-1)))
	return float(rmsd), f"hungarian_coords (strict failed: {strict_error})"
#===========以下将是采样打分部分，后续会进行迁移尽量不全部填进一个文件里============
def t_to_sigma_conf(t_tr, t_rot, t_tor, tr_sigma_min=0.1, tr_sigma_max=19, rot_sigma_min=0.03, rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14):
	tr_sigma = tr_sigma_min ** (1-t_tr) * tr_sigma_max ** t_tr
	rot_sigma = rot_sigma_min ** (1-t_rot) * rot_sigma_max ** t_rot
	tor_sigma = tor_sigma_min ** (1-t_tor) * tor_sigma_max ** t_tor
	return tr_sigma, rot_sigma, tor_sigma

def get_diffdock_confidence_model(args, t_to_sigma=t_to_sigma_conf, confidence_mode=True):
	from project_mine.src.models.time_step_embedding import get_timestep_embedding
	from project_mine.src.models.all_atom_score_model import TensorProductScoreModel as ConfidenceModel


	timestep_emb_func = get_timestep_embedding(
		embedding_type=args.embedding_type,
		embedding_dim=args.sigma_embed_dim,
		embedding_scale=args.embedding_scale)

	lm_embedding_type = None
	if args.esm_embeddings_path is not None:
		lm_embedding_type = 'esm'
	#print(args.rmsd_classification_cutoff)
	model = ConfidenceModel(t_to_sigma=t_to_sigma,
							no_torsion=args.no_torsion,
							timestep_emb_func=timestep_emb_func,
							num_conv_layers=args.num_conv_layers,
							lig_max_radius=args.max_radius,
							scale_by_sigma=args.scale_by_sigma,
							sigma_embed_dim=args.sigma_embed_dim,
							ns=args.ns, nv=args.nv,
							distance_embed_dim=args.distance_embed_dim,
							cross_distance_embed_dim=args.cross_distance_embed_dim,
							batch_norm=not args.no_batch_norm,
							dropout=args.dropout,
							use_second_order_repr=args.use_second_order_repr,
							cross_max_distance=args.cross_max_distance,
							dynamic_max_cross=args.dynamic_max_cross,
							lm_embedding_type=lm_embedding_type,  # type: ignore
							confidence_mode=confidence_mode,
							num_confidence_outputs=len(
								args.rmsd_classification_cutoff) + 1 if hasattr(args, 'rmsd_classification_cutoff') and isinstance(
								args.rmsd_classification_cutoff, list) else 1)
	return model