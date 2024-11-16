import torch
import numpy as np



def cutmix(data, targets_A, targets_B, alpha):
    # 데이터 인덱스 셔플링
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets_A = targets_A[indices]
    shuffled_targets_B = targets_B[indices]

    # CutMix 비율
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    cut_ratio = 1. - lam
    cut_size = int(data.size(2) * cut_ratio)
    start = torch.randint(0, data.size(2) - cut_size + 1, (1,)).item()

    # 데이터 잘라 붙이기
    data[:, :, start:start+cut_size] = shuffled_data[:, :, start:start+cut_size]

    # 정확한 lam 비율 계산
    lam = 1 - cut_size / data.size(2)

    return data, (targets_A, shuffled_targets_A), (targets_B, shuffled_targets_B), lam