import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffurec import Diffu_xstart, _extract_into_tensor
from model import LayerNorm


class ConsistencyDiffuRec(nn.Module):
    def __init__(self, teacher_diffurec, args):
        super().__init__()

        self.betas = teacher_diffurec.betas
        self.alphas_cumprod = teacher_diffurec.alphas_cumprod
        self.alphas_cumprod_prev = teacher_diffurec.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = teacher_diffurec.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = teacher_diffurec.sqrt_one_minus_alphas_cumprod
        self.num_timesteps = teacher_diffurec.num_timesteps
        self.rescale_timesteps = teacher_diffurec.rescale_timesteps

        H = args.hidden_size
        self.hidden_size = H

        self.xstart_model = Diffu_xstart(H, args)
        self.xstart_model.load_state_dict(teacher_diffurec.xstart_model.state_dict())

        self.refine_head = nn.Sequential(
            nn.Linear(4 * H, H),
            nn.GELU(),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _backbone_forward(self, item_rep, x_t, t, mask_seq):
        st = self._scale_timesteps(t)
        coarse, rep_diffu = self.xstart_model(item_rep, x_t, st, mask_seq)

        h_last = rep_diffu[:, -1, :]

        mask = mask_seq.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        h_pool = (rep_diffu * mask).sum(dim=1) / denom

        tau = self.xstart_model.time_embed(
            self.xstart_model.timestep_embedding(st, self.hidden_size)
        )
        return coarse, h_last, h_pool, tau

    def predict_all(self, item_rep, x_t, t, mask_seq):
        coarse, h_last, h_pool, tau = self._backbone_forward(item_rep, x_t, t, mask_seq)
        r = self.refine_head(torch.cat([coarse, h_last, h_pool, tau], dim=-1))
        x0 = coarse + r
        return x0, coarse, r

    def predict_x0(self, item_rep, x_t, t, mask_seq):
        x0, _, _ = self.predict_all(item_rep, x_t, t, mask_seq)
        return x0

    @torch.no_grad()
    def teacher_ddim_step(self, teacher, item_rep, x_t_high, t_high, t_low, mask_seq):
        x_0_t, _ = teacher.xstart_model(
            item_rep, x_t_high, teacher._scale_timesteps(t_high), mask_seq
        )

        sa_h = _extract_into_tensor(teacher.sqrt_alphas_cumprod, t_high, x_t_high.shape)
        som_h = _extract_into_tensor(teacher.sqrt_one_minus_alphas_cumprod, t_high, x_t_high.shape)
        sa_l = _extract_into_tensor(teacher.sqrt_alphas_cumprod, t_low, x_t_high.shape)
        som_l = _extract_into_tensor(teacher.sqrt_one_minus_alphas_cumprod, t_low, x_t_high.shape)

        eps_pred = (x_t_high - sa_h * x_0_t) / som_h
        x_t_low = sa_l * x_0_t + som_l * eps_pred
        return x_t_low

    @torch.no_grad()
    def sample(self, item_rep, mask_seq, num_steps=1):
        device = next(self.parameters()).device
        bs = item_rep.shape[0]
        H = item_rep.shape[-1]
        T = self.num_timesteps

        x_t = torch.randn(bs, H, device=device)

        if num_steps == 1:
            t = torch.full((bs,), T - 1, device=device, dtype=torch.long)
            return self.predict_x0(item_rep, x_t, t, mask_seq)

        ts = np.linspace(T - 1, 1, num_steps).round().astype(int)
        x_0 = None
        for i, t_val in enumerate(ts):
            t = torch.full((bs,), int(t_val), device=device, dtype=torch.long)
            x_0 = self.predict_x0(item_rep, x_t, t, mask_seq)
            if i < len(ts) - 1:
                noise = torch.randn_like(x_0)
                t_next = torch.full((bs,), int(ts[i + 1]), device=device, dtype=torch.long)
                x_t = self.q_sample(x_0, t_next, noise)
        return x_0


class ConsistencyStudent(nn.Module):
    def __init__(self, teacher_model, args, ema_decay=0.95):
        super().__init__()
        self.args = args
        self.ema_decay = ema_decay

        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1

        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.item_embeddings.load_state_dict(teacher_model.item_embeddings.state_dict())
        for p in self.item_embeddings.parameters():
            p.requires_grad = False

        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm.load_state_dict(teacher_model.LayerNorm.state_dict())
        self.dropout = nn.Dropout(args.dropout)

        self.diffu_student = ConsistencyDiffuRec(teacher_model.diffu, args)
        self.diffu_student_ema = copy.deepcopy(self.diffu_student)
        for p in self.diffu_student_ema.parameters():
            p.requires_grad = False

        self.loss_ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def update_ema(self):
        for p_t, p_o in zip(self.diffu_student_ema.parameters(),
                            self.diffu_student.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_o.data, alpha=1 - self.ema_decay)

    def encode(self, sequence):
        e = self.item_embeddings(sequence)
        e = self.embed_dropout(e)
        e = self.LayerNorm(e)
        mask = (sequence > 0).float()
        return e, mask

    def consistency_loss(self, sequence, target, teacher_diffu,
                         contrast_temperature=0.1):
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.item_embeddings(target.squeeze(-1))

        bs = sequence.size(0)
        T = self.diffu_student.num_timesteps
        device = sequence.device

        n = torch.randint(1, T, (bs,), device=device)
        t_high = n
        t_low = n - 1

        noise = torch.randn_like(x_0)
        x_t_high = self.diffu_student.q_sample(x_0, t_high, noise)

        with torch.no_grad():
            x_t_low = self.diffu_student.teacher_ddim_step(
                teacher_diffu, item_rep, x_t_high, t_high, t_low, mask_seq
            )

        x0_high, coarse_high, res_high = self.diffu_student.predict_all(
            item_rep, x_t_high, t_high, mask_seq
        )

        with torch.no_grad():
            x0_low = self.diffu_student_ema.predict_x0(item_rep, x_t_low, t_low, mask_seq)

        cons_loss = F.mse_loss(x0_high, x0_low)

        scores = torch.matmul(x0_high, self.item_embeddings.weight.t())
        ce_loss = self.loss_ce(scores, target.squeeze(-1))

        anchor = F.normalize(x0_high, dim=-1)
        pos_neg = F.normalize(x_0, dim=-1)
        logits = (anchor @ pos_neg.t()) / contrast_temperature
        labels = torch.arange(bs, device=device)
        contrast_loss = F.cross_entropy(logits, labels)

        coarse_loss = F.mse_loss(coarse_high, x_0)

        res_loss = F.mse_loss(res_high, x_0 - coarse_high)

        return cons_loss, ce_loss, contrast_loss, coarse_loss, res_loss

    @torch.no_grad()
    def predict_scores(self, sequence, num_steps=1):
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.diffu_student.sample(item_rep, mask_seq, num_steps=num_steps)
        return torch.matmul(x_0, self.item_embeddings.weight.t())
