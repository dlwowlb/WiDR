import copy
import zarr
from pathlib import Path
import numpy as np
import functools
import torch
from torch.utils.data import DataLoader, Dataset




def split_array_bychunk(array, chunksize, include_residual=True):
    len_ = len(array) // chunksize * chunksize
    main_part, array_residual = array[:len_], array[len_:]
    chunks = [
        main_part[i * chunksize: (i + 1) * chunksize]
        for i in range(len(main_part) // chunksize)
    ]
    if include_residual and len(array_residual) > 0:
        chunks.append(array_residual)
    return chunks


class CSIDA(Dataset):
    def __init__(self,
                 root,
                 roomid=None,
                 userid=None,
                 location=None,
                 orientation=None,
                 receiverid=None,
                 sampleid=None,
                 data_shape=None,
                 chunk_size=None,
                 mode=None,
                 trainmode=True,
                 trainsize=0.8):
        
        self.root = root
        self.data_shape = data_shape
        self.chunk_size = chunk_size
        self.mode = mode
        self.trainmode = trainmode
        self.trainsize = trainsize

        self.group = zarr.open_group(root.as_posix(), mode="r")

        self.gesture = self.group.csi_label_act[:]
        self.room_label = self.group.csi_label_env[:]
        self.location_label = self.group.csi_label_loc[:]
        self.userid_label = self.group.csi_label_user[:]

        self.total_samples = len(self.gesture)
        self.select = np.ones(self.total_samples, dtype=np.bool_)

        # 필터링
        if roomid is not None:
            room_select = functools.reduce(np.logical_or, [self.room_label == j for j in roomid])
            self.select = np.logical_and(self.select, room_select)
        if userid is not None:
            user_select = functools.reduce(np.logical_or, [self.userid_label == j for j in userid])
            self.select = np.logical_and(self.select, user_select)
        if location is not None:
            loc_select = functools.reduce(np.logical_or, [self.location_label == j for j in location])
            self.select = np.logical_and(self.select, loc_select)

        self.index = np.arange(self.total_samples)[self.select]
        np.random.shuffle(self.index)

        # train / test split
        split_point = int(len(self.index) * self.trainsize)
        if self.trainmode:
            self.index = self.index[:split_point]
        else:
            self.index = self.index[split_point:]

        self.samples = []
        for sample_index in self.index:
            chunk_tensor, ges_label, user_label = self.load_sample(sample_index)
            # data_shape='split' 가정
            for i in range(chunk_tensor.size(0)):
                # 각 chunk를 별도의 샘플로 처리
                self.samples.append((chunk_tensor[i], (ges_label, user_label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, labels = self.samples[idx]  
        # labels는 (ges_label, user_label) 형태
        
        data = data[0, :, :]  # 첫 번째 채널만 
        
        
        return data, labels

    def load_sample(self, sample_index):
        # 원본 전체 시계열 로드
        if self.mode == 'phase':
            sample_data = self.group.csi_data_pha[sample_index].astype(np.float32)
        elif self.mode == 'amplitude':
            sample_data = self.group.csi_data_amp[sample_index].astype(np.float32)
        else:
            amp_sample = self.group.csi_data_amp[sample_index].astype(np.float32)
            pha_sample = self.group.csi_data_pha[sample_index].astype(np.float32)
            sample_data = np.concatenate((amp_sample, pha_sample), axis=2)

        ges_label = torch.tensor(self.gesture[sample_index]).long()
        user_label = torch.tensor(self.userid_label[sample_index]).long()

        if self.data_shape == 'split':
            chunks = split_array_bychunk(sample_data, self.chunk_size, include_residual=False)
            # 마지막 chunk 보충
            chunks.append(sample_data[-self.chunk_size:])
            # chunks: list of [chunk_size, 3, freq_dim]

            chunk_tensor = torch.tensor(np.array(chunks), dtype=torch.float32)
            # permute [repeat, 3, chunk_size, freq_dim]
            chunk_tensor = chunk_tensor.permute(0, 2, 1, 3)
            return chunk_tensor, ges_label, user_label
        else:
            raise NotImplementedError("Only 'split' data_shape supported in this example.")

    def get_choose_label(self, id):
        if id == "user":
            return self.userid_label[self.index]
        if id == "room":
            return self.room_label[self.index]
        if id == "location":
            return self.location_label[self.index]
        if id == "gesture":
            return self.gesture[self.index]



        