import torch
import numpy as np

def mixup(data, targets_A, targets_B, alpha):
    # 데이터 인덱스 셔플링
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets_A = targets_A[indices]
    shuffled_targets_B = targets_B[indices]

    # MixUp 비율
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # 데이터와 타겟의 선형 조합
    data = lam * data + (1 - lam) * shuffled_data

    return data, (targets_A, shuffled_targets_A), (targets_B, shuffled_targets_B), lam