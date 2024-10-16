import os
import numpy as np
import torch
import functools
import random
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import random
import h5py
from matplotlib import pyplot as plt
from DCMST.AB_DCMST import AB_DCMST, getTreeCost,get_nodes_chain_when_degree_is_2


class MatrixDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, predict_matrix_num=3, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.predict_matrix_num = predict_matrix_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num = (all_batch_num+self.input_matrix_num)*(1+self.predict_matrix_num)
        if sampling_rate2 is not None:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)


        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        # resort row and column
        # print('Row&Column rearranging...')
        # self.tms = self.rearrange_matrix_row_and_column(self.tms)

        self.tms = torch.from_numpy(self.tms)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)  # 打乱序号
            indices = indices[:int(self.sampling_rate2*len(self.tms))]  # clip tms
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')
        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms)
        if gpu_mode:
#             if np.prod(self.tms.shape) < 1e8:
                self.large_dataset_flag = False
                self.tms = self.tms.to(torch.device('cuda:0'))
                self.masks = self.masks.to(torch.device('cuda:0'))
#             else:
#                 # Cause hard to sending full dataset into GPU
#                 self.large_dataset_flag = True

    def get_masks(self, tms):
        masks = []
        for i in tms:
            masks.append(self.produce_a_mask(i))
        masks = torch.stack(masks)
        masks = masks.type(torch.FloatTensor)

        if self.gpu_mode and self.large_dataset_flag:
            masks = masks.to(torch.device('cuda:0'))
        return masks

    def produce_a_mask(self, matrix):
        amount = int(torch.prod(torch.from_numpy(np.array(matrix.shape))))
        one_num = int(amount * self.sampling_rate1)
        zero_num = amount - one_num
        mask = np.concatenate([np.ones(one_num), np.zeros(zero_num)])
        np.random.shuffle(mask)
        return torch.from_numpy(mask.reshape(matrix.shape))

    def rearrange_matrix_row_and_column(self, data: np.ndarray, time_len=None):
        def vector_distance_func(vector_i, vector_j):
            return np.sum(np.abs(vector_i - vector_j))  # compute SUM

            # denominator = np.linalg.norm(vector_i) * np.linalg.norm(vector_j)
            # return -vector_i.dot(vector_j) / denominator if denominator != 0 else 0  # compute COS similarity

            # return 0 if vector_i.std() == 0 or vector_j.std() == 0 else \
            #     -np.mean((vector_i-vector_i.mean())*(vector_j-vector_j.mean()))/(vector_i.std()*vector_j.std()) # correla

            # return -CalculateMI.calc_MI(vector_i, vector_j)  # compute MI

            # return -CalculateMI.calc_SU(vector_i, vector_j)  # compute SU


        """ rearrange by matrix serious """
        if time_len is None:
            time_len = 100 if data.shape[0] > 100 else int(data.shape[0])
        time_data = data[:time_len]

        # 254 stamp 424 loong 1284 toys
        selected_pos = 1284  # random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = time_data[:, j, :].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, i, :].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = time_data[:, :, j].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, :, i].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

        """ if rearrange by single matrix, use the next code """
        selected_pos = random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = data[selected_pos, j, :]
            for i in range(j):
                i_vector = data[selected_pos, i, :]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = data[selected_pos, :, j]
            for i in range(j):
                i_vector = data[selected_pos, :, i]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

    def __getitem__(self, index):
        # train data
        head_matrix_pos = index*(1+self.predict_matrix_num)
        tail_matrix_pos = (index+self.input_matrix_num)*(1+self.predict_matrix_num)

        indices = list(range(head_matrix_pos, tail_matrix_pos, self.predict_matrix_num+1))
        train = self.tms[indices]
        time_seq = self.time_seq[indices].to(torch.float32)

        if self.sampling_rate1 is not None:  # partial sampling
            train_mask = self.masks[head_matrix_pos: tail_matrix_pos: self.predict_matrix_num+1]
            train = train * train_mask

        # valid
        head_valid_pos = (index+self.input_matrix_num//2-1)*(1+self.predict_matrix_num)+1
        tail_valid_pos = (index+self.input_matrix_num//2-1)*(1+self.predict_matrix_num)+1+self.predict_matrix_num
        valid = self.tms[head_valid_pos: tail_valid_pos]

        train = torch.unsqueeze(train, dim=0).to(torch.float32)
        valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train = train.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))
            time_seq = time_seq.to(torch.device('cuda:0'))
        return [train, valid, time_seq]

    def __len__(self):
        return self.all_batch_num




class MatrixFillDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num =all_batch_num+self.input_matrix_num
        if sampling_rate2 is not None and sampling_rate2!=1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)


        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        # resort row and column
        # print('Row&Column rearranging...')
        # self.tms = self.rearrange_matrix_row_and_column(self.tms)

        self.tms = torch.from_numpy(self.tms)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)
            indices = indices[:int(self.sampling_rate2*len(self.tms))]
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')
        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms)
        if gpu_mode:
            # if np.prod(self.tms.shape) < 1e8:
                self.large_dataset_flag = False
                self.tms = self.tms.to(torch.device('cuda:0'))
                self.masks = self.masks.to(torch.device('cuda:0'))
            # else:
            #     # Cause hard to sending full dataset into GPU
            #     self.large_dataset_flag = True

    def get_masks(self, tms):
        masks = []
        for i in tms:
            masks.append(self.produce_a_mask(i))
        masks = torch.stack(masks)
        masks = masks.type(torch.FloatTensor)

        if self.gpu_mode and self.large_dataset_flag:
            masks = masks.to(torch.device('cuda:0'))
        return masks

    def produce_a_mask(self, matrix):
        amount = int(torch.prod(torch.from_numpy(np.array(matrix.shape))))
        one_num = int(amount * self.sampling_rate1)
        zero_num = amount - one_num
        mask = np.concatenate([np.ones(one_num), np.zeros(zero_num)])
        np.random.shuffle(mask)
        return torch.from_numpy(mask.reshape(matrix.shape))

    def rearrange_matrix_row_and_column(self, data: np.ndarray, time_len=None):
        def vector_distance_func(vector_i, vector_j):
            return np.sum(np.abs(vector_i - vector_j))  # compute SUM

            # denominator = np.linalg.norm(vector_i) * np.linalg.norm(vector_j)
            # return -vector_i.dot(vector_j) / denominator if denominator != 0 else 0  # compute COS similarity

            # return 0 if vector_i.std() == 0 or vector_j.std() == 0 else \
            #     -np.mean((vector_i-vector_i.mean())*(vector_j-vector_j.mean()))/(vector_i.std()*vector_j.std()) # correla

            # return -CalculateMI.calc_MI(vector_i, vector_j)  # compute MI

            # return -CalculateMI.calc_SU(vector_i, vector_j)  # compute SU


        """ rearrange by matrix serious """
        if time_len is None:
            time_len = 100 if data.shape[0] > 100 else int(data.shape[0])
        time_data = data[:time_len]

        # 254 stamp 424 loong 1284 toys
        selected_pos = 1284  # random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = time_data[:, j, :].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, i, :].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = time_data[:, :, j].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, :, i].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

        """ rearrange by single matrix """
        selected_pos = random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = data[selected_pos, j, :]
            for i in range(j):
                i_vector = data[selected_pos, i, :]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = data[selected_pos, :, j]
            for i in range(j):
                i_vector = data[selected_pos, :, i]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

    def __getitem__(self, index):
        # train data
        head_matrix_pos = index
        tail_matrix_pos = index+self.input_matrix_num

        indices = list(range(head_matrix_pos, tail_matrix_pos))
        valid = self.tms[indices]
        time_seq = self.time_seq[indices].to(torch.float32)

        train_mask = self.masks[head_matrix_pos: tail_matrix_pos]
        train = valid * train_mask

        train = torch.unsqueeze(train, dim=0).to(torch.float32)
        valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train = train.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))
            time_seq = time_seq.to(torch.device('cuda:0'))
        return [train, valid, time_seq]

    def __len__(self):
        return self.all_batch_num


class MatrixPredDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, predict_matrix_num=3, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.predict_matrix_num = predict_matrix_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num =all_batch_num+self.input_matrix_num+self.predict_matrix_num
        if sampling_rate2 is not None and sampling_rate2!=1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)


        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        # resort row and column
        # print('Row&Column rearranging...')
        # self.tms = self.rearrange_matrix_row_and_column(self.tms)

        self.tms = torch.from_numpy(self.tms)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)
            indices = indices[:int(self.sampling_rate2*len(self.tms))]
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')
        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms)
        if gpu_mode:
            # if np.prod(self.tms.shape) < 1e8:
                self.large_dataset_flag = False
                self.tms = self.tms.to(torch.device('cuda:0'))
                self.masks = self.masks.to(torch.device('cuda:0'))
            # else:
            #     # Cause hard to sending full dataset into GPU
            #     self.large_dataset_flag = True

    def get_masks(self, tms):
        masks = []
        for i in tms:
            masks.append(self.produce_a_mask(i))
        masks = torch.stack(masks)
        masks = masks.type(torch.FloatTensor)

        if self.gpu_mode and self.large_dataset_flag:
            masks = masks.to(torch.device('cuda:0'))
        return masks

    def produce_a_mask(self, matrix):
        amount = int(torch.prod(torch.from_numpy(np.array(matrix.shape))))
        one_num = int(amount * self.sampling_rate1)
        zero_num = amount - one_num
        mask = np.concatenate([np.ones(one_num), np.zeros(zero_num)])
        np.random.shuffle(mask)
        return torch.from_numpy(mask.reshape(matrix.shape))

    def rearrange_matrix_row_and_column(self, data: np.ndarray, time_len=None):
        """
            rearrange matrix row and column with five metrics mentioned in paper.
        """
        def vector_distance_func(vector_i, vector_j):
            return np.sum(np.abs(vector_i - vector_j))  # compute SUM

            # denominator = np.linalg.norm(vector_i) * np.linalg.norm(vector_j)
            # return -vector_i.dot(vector_j) / denominator if denominator != 0 else 0  # compute COS similarity

            # return 0 if vector_i.std() == 0 or vector_j.std() == 0 else \
            #     -np.mean((vector_i-vector_i.mean())*(vector_j-vector_j.mean()))/(vector_i.std()*vector_j.std()) # correla

            # return -CalculateMI.calc_MI(vector_i, vector_j)  # compute MI

            # return -CalculateMI.calc_SU(vector_i, vector_j)  # compute SU


        """ rearrange by matrix serious """
        if time_len is None:
            time_len = 100 if data.shape[0] > 100 else int(data.shape[0])
        time_data = data[:time_len]

        # 254 stamp 424 loong 1284 toys
        selected_pos = 1284  # random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = time_data[:, j, :].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, i, :].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = time_data[:, :, j].transpose((1, 0)).reshape(-1)
            for i in range(j):
                i_vector = time_data[:, :, i].transpose((1, 0)).reshape(-1)
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.tick_params(labelsize=17)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

        """ rearrange by single matrix """
        selected_pos = random.randint(0, self.tms.shape[-3]-1)  # random select a matrix to rearrange
        column_num = self.tms.shape[-1]
        row_num = self.tms.shape[-2]
        print(f"Select matrix index:\t{selected_pos}")

        plt.subplot(1, 2, 1)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')

        # rearrange row
        edges = []
        for j in range(row_num):
            j_vector = data[selected_pos, j, :]
            for i in range(j):
                i_vector = data[selected_pos, i, :]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(row_num, tree)
        # exchange row
        data = data[:, index_set, :]

        # rearrange column
        edges = []
        for j in range(column_num):
            j_vector = data[selected_pos, :, j]
            for i in range(j):
                i_vector = data[selected_pos, :, i]
                value = vector_distance_func(i_vector, j_vector)
                edges.append((value, j, i))
        tree = AB_DCMST(edges).getSolution(2)
        index_set = get_nodes_chain_when_degree_is_2(column_num, tree)
        # exchange column
        data = data[:, :, index_set]

        plt.subplot(1, 2, 2)
        plt.imshow(data[selected_pos], interpolation='nearest', cmap='Blues', origin='lower')
        plt.show()

        return data

    def __getitem__(self, index):
        # train data
        head_matrix_pos = index
        tail_matrix_pos = index+self.input_matrix_num

        indices = list(range(head_matrix_pos, tail_matrix_pos))
        train = self.tms[indices]
        time_seq = self.time_seq[indices].to(torch.float32)

        if self.sampling_rate1 is not None and self.sampling_rate1!=1:  # partial sampling
            train_mask = self.masks[head_matrix_pos: tail_matrix_pos]
            train = train * train_mask

        # valid
        head_valid_pos = index+self.input_matrix_num+1
        tail_valid_pos = head_valid_pos+self.predict_matrix_num
        valid = self.tms[head_valid_pos: tail_valid_pos]

        train = torch.unsqueeze(train, dim=0).to(torch.float32)
        valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train = train.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))
            time_seq = time_seq.to(torch.device('cuda:0'))
        return [train, valid, time_seq]

    def __len__(self):
        return self.all_batch_num


def get_traffic_matrix_geant(path: str = '.', all_batch_size=1000):
    """
        read in dataset GEANT from files.
    """
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('IntraTM') == -1:
            del (files[index])
        index -= 1
    # sort file
    files.sort()  # sort by timestamp

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print('Begin load GEANT')
    # 解析xml file
    tree = ET.ElementTree()
    for timestamp in files:
        tm = [[0.0 for j in range(24)] for i in range(24)]
        ele = tree.parse(path + '/' + timestamp)
        for row in ele[1]:
            row_id = int(row.get('id'))
            for node in row:
                col_id = int(node.get('id'))
                tm[row_id][col_id] = float(node.text)
        tms.append(tm)
    print('GEANT loaded')

    temp = np.array(tms)  # [all_batch, 24, 24]  : B,R,C
    x, y = np.split(temp, [1], 1)
    x, tms = np.split(y, [1], 2)


    return tms, torch.Tensor(list(range(len(tms))))


def get_traffic_matrix_taxibj(path: str = '.', all_batch_size=1000, InData=True):
    """
        read in dataset taxiBJ from files.
    """
    print('Begin load TaxiBj')
    f = h5py.File(path, 'r')
    print(f'Keys contains data&date, use InData:{InData}')
    tms = f['data'][:, 0, :, :] if InData else f['data'][:, 1, :, :]
    print('TaxiBj loaded')

    assert len(tms) >= all_batch_size
    tms = tms[:all_batch_size]
    return tms, torch.Tensor(list(range(len(tms))))


def get_traffic_matrix_abilene(path: str = '.', all_batch_size=1000):
    """
        read in dataset abilene from files.
    """
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('tm.2004') == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print('Begin load Abilene')
    for timestamp in files:
        tm = []
        with open(path + '/' + timestamp) as file:
            while True:
                line = file.readline()
                if line == '':
                    break
                if line[0] == '#':
                    continue
                line = line.strip().split(',')
                tm.append([float(num) for num in line])
        tms.append(tm)
    print('Data loaded')

    tms = np.array(tms)  # [all_batch, 12, 12] : B,R,C
    return tms, torch.Tensor(list(range(len(tms))))


def get_traffic_matrix_video(path, all_batch_size=1000, name: str = 'Video_matrix'):
    """
        read in video dataset from files, where the files obtained by function "trans_video_as_matrix(path, sequence_len)"
    """
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find(name) == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print(f'Begin load {name}')
    for timestamp in files:
        tm = np.loadtxt(path + '/' + timestamp, dtype=np.float)
        tms.append(tm)
    print('Data loaded')

    tms = np.stack(tms)  # [all_batch, w, h] : B,R,C

    # # shuffle video row like paper mentioned
    # aa = list(range(tms.shape[1]))
    # random.shuffle(aa)
    # temp = tms[:,aa,:]
    # tms = temp

    # # shuffle video column like paper mentioned
    # aa = list(range(tms.shape[2]))
    # random.shuffle(aa)
    # temp = tms[:,:,aa]
    # tms = temp
    # counter = 1
    # for i in range(len(tms)):
    #     if counter > 50 and counter%5==0:
    #         plt.subplot(121)
    #         plt.tick_params(labelsize=17)
    #         plt.imshow(tms[i], interpolation='nearest', cmap='Blues', origin='lower')
    #         plt.subplot(122)
    #         plt.tick_params(labelsize=17)
    #         plt.imshow(temp[i], interpolation='nearest', cmap='Blues', origin='lower')
    #         plt.show()
    #     counter += 1

    return tms, torch.Tensor(list(range(len(tms))))


def trans_video_as_matrix(path, length, name='Video_matrix'):
    def color2gray(frame):
        return 0.299*frame[:, :, 0]+0.587*frame[:, :, 1]+0.114*frame[:, :, 2]
    if not os.path.exists('./Video_matrix2/'):
        os.mkdir('./Video_matrix2/')
    vc = cv2.VideoCapture(path)
    for i in range(length):
        ret, frame = vc.read()
        if frame is None:
            break
        # write to file
        np.savetxt(f'./Video_matrix2/{name}_%05d.txt'%(i), color2gray(frame), fmt='%.2e')


def autocorrelation(x, lags):  # compute the auto correlation
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(), x[:n - i] - x[:n - i].mean())[0] \
              / (x[i:].std() * x[:n - i].std() * (n - i)) \
              for i in range(1, lags + 1)]
    return result


class CalculateMI:

    @staticmethod
    def calc_MI(X, Y, bins=None):
        """
            compute the mutual information (MI).
        """
        if bins is None:
            bins = X.shape[0]

        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = CalculateMI.shan_entropy(c_X)
        H_Y = CalculateMI.shan_entropy(c_Y)
        H_XY = CalculateMI.shan_entropy(c_XY)

        MI = H_X + H_Y - H_XY
        return MI

    @staticmethod
    def calc_SU(X, Y, bins=None):
        # compute Symmetrical Uncertainty
        # CAIDA. NLANR PMA. http://pma.nlanr.net. 2002-09-11/2005-04. Article (CrossRef Link).
        # Measure Correlation Analysis of Network Flow Based On Symmetric Uncertainty 2012

        if bins is None:
            bins = X.shape[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]
        H_X = CalculateMI.shan_entropy(c_X)
        H_Y = CalculateMI.shan_entropy(c_Y)

        MI = CalculateMI.calc_MI(X, Y, bins)

        return 2 * MI / (H_X + H_Y)

    @staticmethod
    def shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H


if __name__ == '__main__':
    # trans_video_as_matrix('./dataset/TAA/TAA.mp4', 4050)
    # tms = get_traffic_matrix_video('./dataset/TAA/', 40, 'Video_matrix')
    # print(tms.shape)
    # pass

    mat, time_seq = get_traffic_matrix_taxibj("TaxiBJ_dataset_path")
    print(mat.shape)

    # a = np.array([1,1,1,1])
    # b = np.array([2,2,2,2])
    # print(CalculateMI.calc_MI(a,b))

