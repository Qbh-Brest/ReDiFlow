import torch
import torch.nn as nn
import torch.nn.functional as F
class LearnableTimeBudget(nn.Module):
    """
    学习长度为 num_frames-1 的非均匀时间步长:
        dt_k >= 0, sum_k dt_k = 1

    对应 ODE rollout 的每一步，而不是每一个离散帧点。
    """

    def __init__(self, num_steps, init_mode='middle_fast'):
        super().__init__()
        self.num_steps = num_steps

        if init_mode == 'middle_fast':
            init_dt = self._build_middle_fast_prior(num_steps)
        else:
            init_dt = torch.ones(num_steps, dtype=torch.float32) / num_steps

        self.logits = nn.Parameter(torch.log(init_dt + 1e-8))
        self.register_buffer("prior_dt", init_dt)

    def _build_middle_fast_prior(self, num_steps):
        x = torch.linspace(0, 1, num_steps)
        # 中间高，两头低
        values = 0.6 + 1.4 * torch.exp(-0.5 * ((x - 0.5) / 0.18) ** 2)
        values = values / values.sum()
        return values.float()

    def get_dt(self):
        return F.softmax(self.logits, dim=0)

    def forward(self, step_ids):
        dt = self.get_dt()
        step_ids = step_ids.clamp(min=0, max=self.num_steps - 1)
        return dt[step_ids]

    def regularization_loss(self, smooth_weight=1e-3, prior_weight=1e-3):
        dt = self.get_dt()
        smooth_loss = ((dt[1:] - dt[:-1]) ** 2).mean()
        prior_loss = F.kl_div((dt + 1e-8).log(), self.prior_dt, reduction='batchmean')
        return smooth_weight * smooth_loss + prior_weight * prior_loss


def to_step_ids(frame_indices, num_steps):
    """
    把 data.t_float / frame idx 映射到 rollout step id
    你现在 validation 是 total_steps = self.num_frames - 1
    所以这里统一映射到 [0, num_steps-1]
    """
    if frame_indices.dtype in [torch.int32, torch.int64]:
        step_ids = frame_indices.long()
    else:
        step_ids = torch.floor(frame_indices.float() * num_steps).long()

    step_ids = step_ids.clamp(0, num_steps - 1)
    return step_ids
