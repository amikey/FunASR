from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def standard_loss(ys, ts):
    losses = [F.binary_cross_entropy(torch.sigmoid(y), t) * len(y) for y, t in zip(ys, ts)]
    loss = torch.sum(torch.stack(losses))
    n_frames = torch.from_numpy(np.array(np.sum([t.shape[0] for t in ts]))).to(torch.float32).to(ys[0].device)
    loss = loss / n_frames
    return loss


def batch_pit_n_speaker_loss(ys, ts, n_speakers_list):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)  [(T1, C1),(T2, C2), ...,(TB, CB)]
      ts: B-length list of labels  [(T1, 2), (T2, 2), ..., (TB, 2)]
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = ts[0].shape[1]  # C
    # (B, T, C)
    olens = [y.shape[0] for y in ys]  # [T1, T2, ...]
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    ys_mask = [torch.ones(olen).to(ys.device) for olen in olens]
    ys_mask = torch.nn.utils.rnn.pad_sequence(ys_mask, batch_first=True, padding_value=0).unsqueeze(-1)  # (B, T, 1)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        # 通过滚动生成prediction和label之间, 各个speaker之间的对应情况, 然后计算各个speaker对的loss
        # 在上述基础之上, 通过排列生成各种permutation, 然后得到相应的总的loss
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, batch_first=True, padding_value=-1)  # padding, (B, T, C)
        # loss: (B, T, C)
        loss = F.binary_cross_entropy(torch.sigmoid(ys), ts_roll, reduction='none')
        if ys_mask is not None:
            loss = loss * ys_mask
        # sum over time: (B, C)
        # TODO: 感觉这里也需要考虑padding, 不应该直接算在loss中吧？？？？
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, dim=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    # 生成所有可能的排列, size: (Perm, n_speakers)
    perms = np.array(list(permutations(range(max_n_speakers)))).astype(np.float32)
    perms = torch.from_numpy(perms).to(losses.device)
    # y_ind: [0,1,2,3]
    y_ind = torch.arange(max_n_speakers, dtype=torch.float32, device=losses.device)
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    # 获取每种perm下, 每个y_ind对应的t_ind
    t_inds = torch.fmod(perms - y_ind, max_n_speakers).to(torch.long)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind.to(torch.long), t_ind], dim=1))  # 每种perm下, batch中每个样本的loss, size: (B,)
    # losses_perm: (B, Perm)  堆叠每种perm对应的loss
    losses_perm = torch.stack(losses_perm, dim=1)

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]

    # 有时候可能不同sample的speaker数不同, 因此需要去除多余的排列情况对应的loss
    masks = torch.full_like(losses_perm, device=losses.device, fill_value=float('inf'))
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])  # 计算batch中每个sample的最小loss的和
    n_frames = torch.from_numpy(np.array(np.sum([t.shape[0] for t in ts]))).to(losses.device)  # batch中每个sample的实际帧数之和
    min_loss = min_loss / n_frames  # 根据帧数归一化

    min_indices = torch.argmin(losses_perm, dim=1)  # 获取batch中每个sample的最小loss对应的perm索引
    # batch中每个sample的最小loss对应的label, list, [(T1, C1), (T2, C1), ..., (TB, CB)]
    labels_perm = [t[:, perms[idx].to(torch.long)] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    # min_loss: scale
    # labels_perm, list, [(T1, C1), (T2, C1), ..., (TB, CB)]
    return min_loss, labels_perm


def fast_batch_pit_n_speaker_loss(ys, ts):
    with torch.no_grad():
        bs = len(ys)
        indices = []
        for b in range(bs):
            y = ys[b].transpose(0, 1)
            t = ts[b].transpose(0, 1)
            C, _ = t.shape
            y = y[:, None, :].repeat(1, C, 1)
            t = t[None, :, :].repeat(C, 1, 1)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(y), t, reduction="none").mean(-1)
            C = bce_loss.cpu()
            indices.append(linear_sum_assignment(C))
    labels_perm = [t[:, idx[1]] for t, idx in zip(ts, indices)]

    return labels_perm


def cal_power_loss(logits, power_ts):
    losses = [F.cross_entropy(input=logit, target=power_t.to(torch.long)) * len(logit) for logit, power_t in
              zip(logits, power_ts)]
    loss = torch.sum(torch.stack(losses))
    n_frames = torch.from_numpy(np.array(np.sum([power_t.shape[0] for power_t in power_ts]))).to(torch.float32).to(
        power_ts[0].device)
    loss = loss / n_frames
    return loss
