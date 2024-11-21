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

    import torch



def make_low_freq_mask(decay_power, h, w, lam):
    """
    낮은 주파수의 마스크를 생성합니다.
    """
    freqs = np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2
    spectrum = np.random.randn(h, w) * np.exp(-decay_power * freqs)

    # 역 FFT를 통해 마스크 생성
    mask = np.fft.ifft2(spectrum).real
    mask = np.abs(mask)
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    # lam에 따라 마스크 이진화
    mask = (mask >= np.quantile(mask, 1 - lam)).astype(np.float32)

    return mask

def fmix(data, targets_A, targets_B, alpha, decay_power=3):
    # 데이터 인덱스 셔플링
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets_A = targets_A[indices]
    shuffled_targets_B = targets_B[indices]

    # Fmix 비율
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # 마스크 생성
    mask = make_low_freq_mask(decay_power, data.size(2), data.size(3), lam)
    mask = torch.tensor(mask).float().to(data.device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 배치 및 채널 차원 추가
    mask = mask.expand(data.size(0), data.size(1), data.size(2), data.size(3))

    # 데이터에 마스크 적용
    data = data * mask + shuffled_data * (1 - mask)

    # lam 계산
    lam = mask.mean().item()

    return data, (targets_A, shuffled_targets_A), (targets_B, shuffled_targets_B), lam


