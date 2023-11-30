import numpy as np
import torch
import torch.nn as nn
import pickle
# import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
import time
import random
import math
import copy
import os

# Seed
seed = 20  # 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

start = time.time()

cuda_unit = 0
cuda_us = "cuda:" + str(cuda_unit)
print("CUDA usage: " + str(torch.cuda.is_available()))
DEVICE = torch.device(cuda_us if torch.cuda.is_available() else 'cpu')

# the size of embeddings
num_embedding = 10

num_BS_antenna_x = 2
num_BS_antenna_y = 2
num_BS_antenna_z = 2
num_BS_antenna = num_BS_antenna_x * num_BS_antenna_y * num_BS_antenna_z

num_for_complex = 2
num_input = int(((num_BS_antenna ** 2 - num_BS_antenna) / 2) * num_for_complex) + 1  # upper triangular matrix without diagonal elements

# sampling region
num_each_row = 360
num_tot_row = 360

measurement_rate = 0.03  # measurement rate is 3 % -Rui, training / (training + validation + test)
ebd_sample_rate_sv1 = 1  # -Rui, # of embeddings in S + V1 patition / # of all training embeddings
ebd_sample_rate_s = 0.5  # -Rui, # of embeddings in S patition / # of embeddings in S + V1 patition
mini_batch_size = 32  # -Rui, draw locations from v1 and v2, no repeating

num_sample_training = int(num_each_row * num_tot_row * measurement_rate)
num_sample_validation = 1000
ebd_sample_val_v1 = int(num_sample_validation)
num_sample_ebd_sv1 = int(num_sample_training * ebd_sample_rate_sv1)  # -Rui, # of embeddings in S + V1 patition
num_sample_ebd_s = int(num_sample_ebd_sv1 * ebd_sample_rate_s)  # -Rui, # of embeddings in S patition
# -Rui, determine the iterations within each epoch
max_iter_wi_epoch = int((num_sample_ebd_sv1 - num_sample_ebd_s) / mini_batch_size)

ratio_v1 = 0.5  # -Rui, same mini-batch size of v1 and v2
ratio_v2 = 1 - ratio_v1  # -Rui, same mini-batch size of v1 and v2

# -Rui, performance test with multiple splits of trained embeddings
n_test_ebd_split = 10

num_BSs = 5
# WFC = 1  # weight factor for covariance e.g., 1:x:1 for loss of gain, normalized covariance, and AE
# WFG = 1  # weight factor for gain
# WFA = 1  # weight factor for AE

WF_gain = 1
WF_cov = 100
WF_AE = 1
WF_out_w = 0.0
WF_dcae_w = 0.0

# embedding normalization stability adding value
eps_ebd = 1e-10

# validation check
iter_es_check = (max_iter_wi_epoch - 1) * 10

max_iter = 60000  # - Rui, 5000
max_epoch_wo_AE = 10000  # -Rui,5000
min_iter_proposed = 2000  # - Rui, 7000
# min_epoch = 7000
# min_epoch_encdec = 7000  # - Rui, 7000

# -Rui, path
datapath = "./dataset/"
resultpath = "./result/P2O1_Rui/P2O1_trainsv_11001_v1_0p5_wd_0e0_0e0_5BS_3p0/"

if not os.path.exists(resultpath):
  os.mkdir(resultpath)

# data loading (anchor BS1 & BS2 & BS3 & BS4 & BS5)
with open(datapath + "dataset_BS(17)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset1 = pickle.load(f_dataset)
f_dataset.close()

with open(datapath + "dataset_BS(06)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset2 = pickle.load(f_dataset)
f_dataset.close()

with open(datapath + "dataset_BS(05)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset3 = pickle.load(f_dataset)
f_dataset.close()

with open(datapath + "dataset_BS(18)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset4 = pickle.load(f_dataset)
f_dataset.close()

with open(datapath + "dataset_BS(07)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset5 = pickle.load(f_dataset)
f_dataset.close()

# data loading (target BS)
with open(datapath + "dataset_BS(08)_O1_3p5(360x360)", "rb") as f_dataset:
    dataset_target = pickle.load(f_dataset)
f_dataset.close()

# location loading
with open(datapath + "dataset_loc_O1_3p5(360x360)", "rb") as f_dataset_loc:
    dataset_loc = pickle.load(f_dataset_loc)
f_dataset_loc.close()

# BS 8
max_gain_dB = -90.05227029799951
min_gain_dB = -101.21289058026761

##################################################

# sampling
rnd_list_training = []
rnd_list_validation = []

for i in range(num_sample_training):
    rnd = random.randint(0, num_each_row * num_tot_row - 1)
    while rnd in rnd_list_training:
        rnd = random.randint(0, num_each_row * num_tot_row - 1)
    rnd_list_training.append(rnd)

for i in range(num_sample_validation):
    rnd = random.randint(0, num_each_row * num_tot_row - 1)
    while (rnd in rnd_list_training) or (rnd in rnd_list_validation):
        rnd = random.randint(0, num_each_row * num_tot_row - 1)
    rnd_list_validation.append(rnd)

print('index file generation has been completed')


# with open(datapath + "list_inx_P2O1_training.pkl", "wb") as f:
    # pickle.dump(rnd_list_training, f)
# f.close()

# with open(datapath + "list_inx_P2O1_validation.pkl", "wb") as f:
    # pickle.dump(rnd_list_validation, f)
# f.close()

# print('index file saving has been completed')

# loading
# with open(datapath + "list_inx_P2O1_training.pkl", "rb") as f:
#     rnd_list_training = pickle.load(f)
# f.close()
#
# with open(datapath + "list_inx_P2O1_validation.pkl", "rb") as f:
#     rnd_list_validation = pickle.load(f)
# f.close()
#
# print('index file loading has been completed')

##################################################


class DatasetAnchor(Dataset):

    def __init__(self):
        # for n in range(num_sample_training):
        #
        #     inx_random = rnd_list_training[n]
        #     if n == 0:
        #         arr_cov1 = copy.deepcopy(dataset1[inx_random][:])
        #         arr_cov2 = copy.deepcopy(dataset2[inx_random][:])
        #         arr_cov3 = copy.deepcopy(dataset3[inx_random][:])
        #         arr_cov4 = copy.deepcopy(dataset4[inx_random][:])
        #         arr_cov5 = copy.deepcopy(dataset5[inx_random][:])
        #
        #         arr_cov = np.append(arr_cov1, arr_cov2)
        #         arr_cov = np.append(arr_cov, arr_cov3)
        #         arr_cov = np.append(arr_cov, arr_cov4)
        #         arr_cov = np.append(arr_cov, arr_cov5)
        #
        #         arr_loc = dataset_loc[inx_random][:]
        #     else:
        #         temp_arr_cov1 = copy.deepcopy(dataset1[inx_random][:])
        #         temp_arr_cov2 = copy.deepcopy(dataset2[inx_random][:])
        #         temp_arr_cov3 = copy.deepcopy(dataset3[inx_random][:])
        #         temp_arr_cov4 = copy.deepcopy(dataset4[inx_random][:])
        #         temp_arr_cov5 = copy.deepcopy(dataset5[inx_random][:])
        #
        #         temp_arr_cov = np.append(temp_arr_cov1, temp_arr_cov2)
        #         temp_arr_cov = np.append(temp_arr_cov, temp_arr_cov3)
        #         temp_arr_cov = np.append(temp_arr_cov, temp_arr_cov4)
        #         temp_arr_cov = np.append(temp_arr_cov, temp_arr_cov5)
        #
        #         arr_cov = np.vstack((arr_cov, temp_arr_cov))
        #         arr_loc = np.vstack((arr_loc, dataset_loc[inx_random][:]))

        arr_cov = np.append(copy.deepcopy(dataset1[rnd_list_training][:]),
                            copy.deepcopy(dataset2[rnd_list_training][:]), axis=1)
        arr_cov = np.append(arr_cov, copy.deepcopy(dataset3[rnd_list_training][:]), axis=1)
        arr_cov = np.append(arr_cov, copy.deepcopy(dataset4[rnd_list_training][:]), axis=1)
        arr_cov = np.append(arr_cov, copy.deepcopy(dataset5[rnd_list_training][:]), axis=1)

        arr_loc = copy.deepcopy(dataset_loc[rnd_list_training][:])

        self.x = torch.from_numpy(arr_cov)
        self.x = self.x.float()
        self.y = torch.from_numpy(arr_loc)
        self.n_samples = arr_cov.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class DatasetTarget(Dataset):

    def __init__(self):
        # for n in range(num_sample_training):
        #     inx_random = rnd_list_training[n]
        #     if n == 0:
        #         arr_cov = dataset_target[inx_random][:]
        #         arr_loc = dataset_loc[inx_random][:]
        #     else:
        #         temp_arr_cov = dataset_target[inx_random][:]
        #         arr_cov = np.vstack((arr_cov, temp_arr_cov))
        #         arr_loc = np.vstack((arr_loc, dataset_loc[inx_random][:]))

        self.x = torch.from_numpy(copy.deepcopy(dataset_target[rnd_list_training][:]))
        self.x = self.x.float()
        self.y = torch.from_numpy(copy.deepcopy(dataset_loc[rnd_list_training][:]))
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset_sampled_anchor = DatasetAnchor()
dataset_sampled_target = DatasetTarget()

data_loader_anchor = DataLoader(dataset=dataset_sampled_anchor, batch_size=num_sample_training, shuffle=False)
data_loader_target = DataLoader(dataset=dataset_sampled_target, batch_size=num_sample_training, shuffle=False)

##################################################
# - Rui,validation dataset
origin_valid_ts = torch.from_numpy(copy.deepcopy(dataset_target[rnd_list_validation, :])).float().to(DEVICE)
pre_input1 = copy.deepcopy(dataset1[rnd_list_validation, :])
pre_input2 = copy.deepcopy(dataset2[rnd_list_validation, :])
pre_input3 = copy.deepcopy(dataset3[rnd_list_validation, :])
pre_input4 = copy.deepcopy(dataset4[rnd_list_validation, :])
pre_input5 = copy.deepcopy(dataset5[rnd_list_validation, :])
pre_input = np.append(pre_input1, pre_input2, axis=1)
pre_input = np.append(pre_input, pre_input3, axis=1)
pre_input = np.append(pre_input, pre_input4, axis=1)
pre_input = np.append(pre_input, pre_input5, axis=1)
valid_input_ts = torch.from_numpy(pre_input).float().to(DEVICE)
dataset_loc_val = torch.from_numpy(copy.deepcopy(dataset_loc[rnd_list_validation][:])).to(DEVICE)
inx_ebd_val_v1 = torch.arange(ebd_sample_val_v1).to(DEVICE)
# inx_ebd_val_v2 = torch.arange(inx_ebd_val_v1.shape[0], num_sample_validation).to(DEVICE)
inx_ebd_val_v2 = torch.arange(ebd_sample_val_v1).to(DEVICE)
inx_ebd_val = torch.cat((inx_ebd_val_v1, inx_ebd_val_v2))

# origin_valid_ts = torch.from_numpy(dataset_target[rnd_list_validation[0:10], :]).float().to(DEVICE)
# pre_input1 = dataset1[rnd_list_validation[0:10], :]
# pre_input2 = dataset2[rnd_list_validation[0:10], :]
# pre_input3 = dataset3[rnd_list_validation[0:10], :]
# pre_input4 = dataset4[rnd_list_validation[0:10], :]
# pre_input5 = dataset5[rnd_list_validation[0:10], :]
# pre_input = np.append(pre_input1, pre_input2, axis=1)
# pre_input = np.append(pre_input, pre_input3, axis=1)
# pre_input = np.append(pre_input, pre_input4, axis=1)
# pre_input = np.append(pre_input, pre_input5, axis=1)
# valid_input_ts = torch.from_numpy(pre_input).float().to(DEVICE)
##################################################

##################################################
class Storage(nn.Module):
    def __init__(self):
        super().__init__()

        self.saved_interpolated_map = torch.zeros((num_embedding, num_tot_row, num_each_row))
        self.saved_input_map = torch.zeros((num_embedding + 1, num_tot_row, num_each_row))

        self.saved_converged_wo_encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.saved_converged_wo_decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, num_input),
            nn.Tanh()
        )

        self.saved_converged_encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.saved_converged_decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, num_input),
            nn.Tanh()
        )

        self.saved_converged_AE_encoder = nn.Sequential(
            nn.Conv2d(num_embedding + 1, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.PReLU()
        )

        self.saved_converged_AE_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, num_embedding, 3, stride=1, padding=1),
            nn.Tanh()
        )


class AutoencoderWoAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, num_input),
            nn.Tanh()
        )

    def forward(self, x, y):
        encoded = self.encoder(x)
        # normalized embeddings
        encoded = encoded / (encoded.norm(dim=1, keepdim=True) + eps_ebd)
        decoded = self.decoder(encoded)
        return decoded


class OuterEncDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, num_input),
            nn.Tanh()
        )

        self.interpolated_map = torch.zeros((num_embedding, num_tot_row, num_each_row))
        self.input_map = torch.zeros((num_embedding + 1, num_tot_row, num_each_row))
        self.var_init = 0

    def forward(self, x, y, inx_ebd_s, inx_ebd_v1_mini, inx_ebd_v2_mini, DEVICE):
        input_emb_img = torch.zeros((num_embedding + 1, num_tot_row, num_each_row)).to(DEVICE)
        decoded_in = torch.zeros(inx_ebd_v1_mini.shape[0] + inx_ebd_v2_mini.shape[0], num_embedding).to(DEVICE)

        encoded = self.encoder(x)
        # normalized embeddings
        encoded = encoded / (encoded.norm(dim=1, keepdim=True) + eps_ebd)
        # for i in range(x.size(dim=0)):
        #     input_emb_img[0:-1, y[i, 1], y[i, 0]] = encoded[i, :]
        #     input_emb_img[num_embedding, y[i, 1], y[i, 0]] = 1

        # -Rui, construct map using ONLY part of training embeddings
        input_emb_img[0:-1, y[inx_ebd_s, 1].long(), y[inx_ebd_s, 0].long()] = encoded[inx_ebd_s, :].transpose(1, 0)
        input_emb_img[num_embedding, y[inx_ebd_s, 1].long(), y[inx_ebd_s, 0].long()] = 1

        self.input_map = input_emb_img
        input_emb_img = input_emb_img[None, :]
        # -Rui, interpolate embeddings of all locations
        interpolated = model_AE(input_emb_img)
        interpolated = interpolated[0, :]
        self.interpolated_map = interpolated
        # decoded_in = torch.zeros((x.size(dim=0), num_embedding))
        # for i in range(x.size(dim=0)):
        #     decoded_in[i, :] = interpolated[:, y[i, 1], y[i, 0]]

        # -Rui, pick interpolated embeddings in v1 partition
        # -Rui, no loop
        decoded_in[0:inx_ebd_v1_mini.shape[0], :] = interpolated[:, y[inx_ebd_v1_mini, 1].long(), y[inx_ebd_v1_mini, 0].long()].transpose(1, 0)

        # -Rui, v2 partition for M case
        for i in range(inx_ebd_v2_mini.shape[0]):
            input_emb_img_v2 = input_emb_img[0, :].clone()
            input_emb_img_v2[0:-1, y[inx_ebd_v2_mini[i], 1].long(), y[inx_ebd_v2_mini[i], 0].long()] = encoded[inx_ebd_v2_mini[i], :]
            input_emb_img_v2[num_embedding, y[inx_ebd_v2_mini[i], 1].long(), y[inx_ebd_v2_mini[i], 0].long()] = 1
            input_emb_img_v2 = input_emb_img_v2[None, :]
            interpolated_v2 = model_AE(input_emb_img_v2)
            interpolated_v2 = interpolated_v2[0, :]
            decoded_in[inx_ebd_v1_mini.shape[0] + i, :] = interpolated_v2[:, y[inx_ebd_v2_mini[i], 1].long(), y[inx_ebd_v2_mini[i], 0].long()]

        # normalized embeddings
        decoded_in = decoded_in / (decoded_in.norm(dim=1, keepdim=True) + eps_ebd)
        decoded = self.decoder(decoded_in)

        return encoded, decoded_in, decoded, input_emb_img


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_embedding + 1, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, num_embedding, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


##################################################
# evaluation method, adapted from test evaluation codes 
def eval_cust(original, predicted, max_gain_dB, min_gain_dB, num_BS_antenna=8):
    # shape check
    if original.shape[0] != predicted.shape[0]:
        print('tensor shapes miss matched!')
        
    num_locs = original.shape[0]
    # convert gain values in -1 ~ +1 range to original values in dB scale
    original_gain = (max_gain_dB - min_gain_dB) * (original[:, 0] + 1) / 2 + min_gain_dB
    original_gain_linear = 10 ** (original_gain / 10)

    predicted_gain = (max_gain_dB - min_gain_dB) * (predicted[:, 0] + 1) / 2 + min_gain_dB
    predicted_gain_linear = 10 ** (predicted_gain / 10)

    SE_gain = (original_gain_linear - predicted_gain_linear) ** 2
    SO_gain = original_gain_linear ** 2

    sum_SE_gain = SE_gain.sum()
    sum_SO_gain = SO_gain.sum()

    # normalized covariance
    MSE_cov = 2 * ((predicted[:, 1:] - original[:, 1:]) ** 2).sum(axis=1)
    MSE_cov2 = MSE_cov / num_BS_antenna ** 2

    sum_MSE_cov = MSE_cov2.sum()

    # list_NMSE_gain = SE_gain / SO_gain
    # median_NMSE_gain = list_NMSE_gain.median()

    NMSE_gain_dB = 10 * (sum_SE_gain / sum_SO_gain).log10()
    # Median_NSE_gain_dB = 10 * median_NMSE_gain.log10()
    MSE_cov_dB = 20 * (sum_MSE_cov / num_locs).sqrt().log10()

    # return NMSE_gain_dB, Median_NSE_gain_dB, Avg_RMSE_cov_dB, \
    #       img_original_gain, img_predicted_gain, img_RMSE_cov2, img_RMSE_cov2_dB
    return NMSE_gain_dB, MSE_cov_dB       
##################################################

##################################################
# evaluation method, adapted from test evaluation codes, AE reconstruciton only 
def eval_AE_cust(encoded_in, decoded_in, num_embedding):
    num_AE_locs = decoded_in.shape[0]
    # normalized covariance
    MSE_AE = ((encoded_in - decoded_in) ** 2).sum(axis=1)
    MSE_AE2 = MSE_AE / num_embedding

    sum_MSE_AE = MSE_AE2.sum()
    mse_loss_AE = 20 * (sum_MSE_AE / num_AE_locs).sqrt().log10()
    return mse_loss_AE
##################################################


criterion = nn.MSELoss()

model_save = Storage().to(DEVICE)

model_wo_AE = AutoencoderWoAE().to(DEVICE)
model_outer = OuterEncDec().to(DEVICE)
model_AE = Autoencoder().to(DEVICE)

optimizer_wo_AE = torch.optim.Adam(list(model_wo_AE.encoder.parameters()) + list(model_wo_AE.decoder.parameters()),
                                   lr=1e-3)
optimizer_outer = torch.optim.Adam(list(model_outer.encoder.parameters()) + list(model_outer.decoder.parameters()),
                                   lr=1e-4)
optimizer_AE = torch.optim.Adam(list(model_AE.encoder.parameters()) + list(model_AE.decoder.parameters()), lr=1e-4)

##################################################

# without AE

minimum_loss = 10000
check_overfitting = 0
epoch_wo_AE_val_es = 0

arr_loss_training_wo_AE = []
arr_error_gain_training_wo_AE = []
arr_error_cov_training_wo_AE = []
arr_error_wd_training_wo_AE = []
arr_loss_validation_wo_AE = []
arr_error_gain_validation_wo_AE = []
arr_error_cov_validation_wo_AE = []
for epoch in range(max_epoch_wo_AE):
    for (inputs, labels) in data_loader_anchor:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_wo_AE.zero_grad()
        recon = model_wo_AE(inputs, labels)
    for (targets, temp_labels) in data_loader_target:
        targets = targets.to(DEVICE)
        loss_gain = criterion(recon[:, 0], targets[:, 0])
        loss_cov = criterion(recon[:, 1:num_input], targets[:, 1:num_input])

    # # for balancing between loss of gain and covariance
    # if epoch == 0:
    #     WF_gain = (loss_cov.item() / (loss_gain.item() + loss_cov.item())) * WFG
    #     WF_cov = (loss_gain.item() / (loss_gain.item() + loss_cov.item())) * WFC

    # -Rui, custom weight decay
    loss_out_wd = torch.tensor(0.0, requires_grad=True).to(DEVICE)
    for name, param in model_wo_AE.named_parameters():
        loss_out_wd += torch.norm(param)

    loss = WF_gain * loss_gain + WF_cov * loss_cov + WF_out_w * loss_out_wd
    loss.backward()
    optimizer_wo_AE.step()

    arr_loss_training_wo_AE.append(loss.item())
    arr_error_gain_training_wo_AE.append(loss_gain.item())
    arr_error_cov_training_wo_AE.append(loss_cov.item())
    arr_error_wd_training_wo_AE.append(loss_out_wd.item())

    model_wo_AE.eval()
    with torch.no_grad():
        encoded_val = model_wo_AE.encoder(valid_input_ts)
        # normalized embeddings
        encoded_val = encoded_val / (encoded_val.norm(dim=1, keepdim=True) + eps_ebd)
        predicted = model_wo_AE.decoder(encoded_val)

        loss_validation_gain = ((origin_valid_ts[:, 0] - predicted[:, 0]) ** 2).mean()
        loss_validation_cov = ((origin_valid_ts[:, 1:num_input] - predicted[:, 1:num_input]) ** 2).mean()

        loss_validation = WF_gain * loss_validation_gain + WF_cov * loss_validation_cov

        arr_loss_validation_wo_AE.append(loss_validation.item())
        arr_error_gain_validation_wo_AE.append(loss_validation_gain.item())
        arr_error_cov_validation_wo_AE.append(loss_validation_cov.item())

    model_wo_AE.train()

    print(
        '>>> No DCAE| epoch = %.0f/%.0f | loss_train: gain = %.4f cov = %.4f, wd = %.4f, comb = %.4f | loss_val: gain = %.4f, cov = %.4f, comb = %.4f ' %
        (epoch,
         max_epoch_wo_AE,
         arr_error_gain_training_wo_AE[epoch],
         arr_error_cov_training_wo_AE[epoch],
         arr_error_wd_training_wo_AE[epoch],
         arr_loss_training_wo_AE[epoch],
         arr_error_gain_validation_wo_AE[epoch],
         arr_error_cov_validation_wo_AE[epoch],
         arr_loss_validation_wo_AE[epoch]))

    # -Rui, save the model with minimum validation loss
    if minimum_loss >= loss_validation:
        minimum_loss = loss_validation
        model_save.saved_converged_wo_encoder.load_state_dict(copy.deepcopy(model_wo_AE.encoder.state_dict()))
        model_save.saved_converged_wo_decoder.load_state_dict(copy.deepcopy(model_wo_AE.decoder.state_dict()))
        epoch_wo_AE_val_es = epoch

    if epoch == max_epoch_wo_AE - 1:
        model_wo_AE.encoder.load_state_dict(copy.deepcopy(model_save.saved_converged_wo_encoder.state_dict()))
        model_wo_AE.decoder.load_state_dict(copy.deepcopy(model_save.saved_converged_wo_decoder.state_dict()))
        check_overfitting = 1
        print(f'convergence Epoch:{epoch + 1}')
        print(f'early stop Epoch:{epoch_wo_AE_val_es + 1}')
        print(f'ratio:{loss_validation.item() / minimum_loss.item()}')

    if check_overfitting == 1:
        max_epoch_wo_AE = epoch + 1
        break

##################################################

# proposed

minimum_loss = 10000
check_overfitting = 0

arr_loss_training = []
arr_error_gain_training_v2 = []
arr_error_gain_training_v1 = []
arr_error_cov_training_v2 = []
arr_error_cov_training_v1 = []
arr_error_gain_training_v2_nmse = []
arr_error_gain_training_v1_nmse = []
arr_error_cov_training_v2_mse = []
arr_error_cov_training_v1_mse = []
arr_error_AE_training = []
arr_error_AE_training_v1 = []
arr_error_AE_training_v2 = []
arr_error_AE_training_v1_mse = []
arr_error_AE_training_v2_mse = []

arr_loss_validation_v2 = []
arr_loss_validation_v1 = []
arr_error_gain_validation_v2 = []
arr_error_gain_validation_v1 = []
arr_error_cov_validation_v2 = []
arr_error_cov_validation_v1 = []
arr_error_gain_validation_v2_nmse = []
arr_error_gain_validation_v1_nmse = []
arr_error_cov_validation_v2_mse = []
arr_error_cov_validation_v1_mse = []
arr_error_AE_validation_v2 = []
arr_error_AE_validation_v1 = []
arr_error_AE_validation_v2_mse = []
arr_error_AE_validation_v1_mse = []

arr_error_out_wd_training = []
arr_error_dcae_wd_training = []

#  -Rui, indices of training splits
# inx_ebd_val = torch.randperm(inputs.shape[0]).to(DEVICE)[0:num_sample_ebd]

# - Rui, indices of s + v1 partition and v2 partition
inx_ebd_train = torch.arange(num_sample_training).to(DEVICE)
inx_ebd_sv1 = inx_ebd_train[0:num_sample_ebd_sv1].clone()
# inx_ebd_v2_fix = inx_ebd_train[num_sample_ebd_sv1:].clone()
iter_wi_epoch = 0
for iter in range(max_iter):

    # time_t = time.time()
    if model_outer.var_init == 0:
        model_outer.encoder.load_state_dict(copy.deepcopy(model_wo_AE.encoder.state_dict()))
        model_outer.decoder.load_state_dict(copy.deepcopy(model_wo_AE.decoder.state_dict()))
        model_outer.var_init = 1

    for (inputs, labels) in data_loader_anchor:
        # -Rui, sample ONLY part of training embeddings
        if iter_wi_epoch == max_iter_wi_epoch:
            iter_wi_epoch = 0
        # -Rui, shuffle v1 and v2 at first iteration in each epoch
        if iter_wi_epoch == 0:
            inx_ebd_sv1_perm = torch.randperm(num_sample_ebd_sv1).to(DEVICE)
            inx_ebd_s = inx_ebd_sv1[inx_ebd_sv1_perm[0:num_sample_ebd_s]].clone()
            inx_ebd_v1 = inx_ebd_sv1[inx_ebd_sv1_perm[num_sample_ebd_s:]].clone()
            # inx_ebd_v2 = inx_ebd_v2_fix[torch.randperm(inx_ebd_v2_fix.shape[0])].clone()
            inx_ebd_v2 = inx_ebd_sv1[inx_ebd_sv1_perm[num_sample_ebd_s:]].clone()
        
        inx_ebd_v1_mini = inx_ebd_v1[iter_wi_epoch * mini_batch_size: (iter_wi_epoch + 1) * mini_batch_size].clone()
        inx_ebd_v2_mini = inx_ebd_v2[iter_wi_epoch * mini_batch_size: (iter_wi_epoch + 1) * mini_batch_size].clone()
        iter_wi_epoch += 1
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_outer.zero_grad()
        optimizer_AE.zero_grad()
        encoded_in, decoded_in, recon, input_emb_img = model_outer(inputs, labels, inx_ebd_s, inx_ebd_v1_mini, inx_ebd_v2_mini, DEVICE)
    for (targets, temp_labels) in data_loader_target:
        targets = targets.to(DEVICE)
        # loss_gain = criterion(recon[:, 0], targets[:, 0])
        # loss_cov = criterion(recon[:, 1:num_input], targets[:, 1:num_input])
        loss_gain_v1 = criterion(recon[0:inx_ebd_v1_mini.shape[0], 0], targets[inx_ebd_v1_mini, 0])
        loss_cov_v1 = criterion(recon[0:inx_ebd_v1_mini.shape[0], 1:num_input], targets[inx_ebd_v1_mini, 1:num_input])
        loss_gain_v2 = criterion(recon[inx_ebd_v1_mini.shape[0]:, 0], targets[inx_ebd_v2_mini, 0])
        loss_cov_v2 = criterion(recon[inx_ebd_v1_mini.shape[0]:, 1:num_input], targets[inx_ebd_v2_mini, 1:num_input])
        nmse_gain_train_v1, mse_cov_train_v1 = eval_cust(targets[inx_ebd_v1_mini, :], recon[0:inx_ebd_v1_mini.shape[0], :], max_gain_dB, min_gain_dB, num_BS_antenna=8)
        nmse_gain_train_v2, mse_cov_train_v2 = eval_cust(targets[inx_ebd_v2_mini, :], recon[inx_ebd_v1_mini.shape[0]:, :], max_gain_dB, min_gain_dB, num_BS_antenna=8)

    # for cnt_AE in range(num_sample_training):
    #     if cnt_AE == 0:
    #         AE_input = model_outer.input_map[0:num_embedding, dataset_loc[rnd_list_training[cnt_AE], 1],
    #                    dataset_loc[rnd_list_training[cnt_AE], 0]]
    #         AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_training[cnt_AE], 1],
    #                           dataset_loc[rnd_list_training[cnt_AE], 0]]
    #     else:
    #         AE_input = torch.vstack(
    #             (AE_input, model_outer.input_map[0:num_embedding, dataset_loc[rnd_list_training[cnt_AE], 1],
    #                        dataset_loc[rnd_list_training[cnt_AE], 0]]))
    #         AE_interpolated = torch.vstack(
    #             (AE_interpolated, model_outer.interpolated_map[:, dataset_loc[rnd_list_training[cnt_AE], 1],
    #                               dataset_loc[rnd_list_training[cnt_AE], 0]]))

    # -Rui, no loop, and why don't return encoded and decoded_in? Only reconstruct embeddings with training locations?
    # AE_input = model_outer.input_map[0:num_embedding, dataset_loc[rnd_list_training, 1],
    #            dataset_loc[rnd_list_training, 0]].transpose(1, 0)
    # AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_training, 1],
    #                   dataset_loc[rnd_list_training, 0]].transpose(1, 0)

    loss_AE_v1 = criterion(encoded_in[inx_ebd_v1_mini, :], decoded_in[0:inx_ebd_v1_mini.shape[0],:])
    loss_AE_v1_mse = eval_AE_cust(encoded_in[inx_ebd_v1_mini, :], decoded_in[0:inx_ebd_v1_mini.shape[0],:], num_embedding)
    loss_AE_v2 = criterion(encoded_in[inx_ebd_v2_mini, :], decoded_in[inx_ebd_v1_mini.shape[0]:,:])
    loss_AE_v2_mse = eval_AE_cust(encoded_in[inx_ebd_v2_mini, :], decoded_in[inx_ebd_v1_mini.shape[0]:,:], num_embedding)
    
    # weight factors
    # if epoch == 0:
    #     temp_WF1 = loss_gain.item() * loss_cov.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF_gain = loss_cov.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF_cov = loss_gain.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF2 = loss_AE.item() / (temp_WF1 + loss_AE.item())
    #     WF_gain = temp_WF2 * temp_WF_gain * WFG
    #     WF_cov = temp_WF2 * temp_WF_cov * WFC
    #     WF_AE = temp_WF1 / (temp_WF1 + loss_AE.item()) * WFA

    # - Rui, custom weight decay
    loss_out_wd = torch.tensor(0.0, requires_grad=True).to(DEVICE)
    for name, param in model_outer.named_parameters():
        loss_out_wd += torch.norm(param)

    loss_dcae_wd = torch.tensor(0.0, requires_grad=True).to(DEVICE)
    for name, param in model_AE.named_parameters():
        loss_dcae_wd += torch.norm(param)

    loss_gain = loss_gain_v1 * ratio_v1 + loss_gain_v2 * ratio_v2
    loss_cov = loss_cov_v1 * ratio_v1 + loss_cov_v2 * ratio_v2
    loss_AE = loss_AE_v1 * ratio_v1 + loss_AE_v2 * ratio_v2
    loss = WF_gain * loss_gain + WF_cov * loss_cov + WF_AE * loss_AE + WF_out_w * loss_out_wd + WF_dcae_w * loss_dcae_wd

    loss.backward()
    optimizer_outer.step()
    optimizer_AE.step()

    arr_loss_training.append(loss.item())
    arr_error_gain_training_v2.append(loss_gain_v2.item())
    arr_error_gain_training_v1.append(loss_gain_v1.item())
    arr_error_cov_training_v2.append(loss_cov_v2.item())
    arr_error_cov_training_v1.append(loss_cov_v1.item())
    arr_error_gain_training_v2_nmse.append(nmse_gain_train_v2.item())
    arr_error_gain_training_v1_nmse.append(nmse_gain_train_v1.item())
    arr_error_cov_training_v2_mse.append(mse_cov_train_v2.item())
    arr_error_cov_training_v1_mse.append(mse_cov_train_v1.item())
    arr_error_AE_training.append(loss_AE.item())
    arr_error_AE_training_v1.append(loss_AE_v1.item())
    arr_error_AE_training_v2.append(loss_AE_v2.item())
    arr_error_AE_training_v1_mse.append(loss_AE_v1_mse.item())
    arr_error_AE_training_v2_mse.append(loss_AE_v2_mse.item())
    arr_error_out_wd_training.append(loss_out_wd.item())
    arr_error_dcae_wd_training.append(loss_dcae_wd.item())

    # print(f'Epoch:{epoch + 1}, Loss:{loss.item():.6f}')
    # print("time taken :", time.time() - start)

    # for m in range(num_sample_validation):
    #     test_inx = rnd_list_validation[m]
    #
    #     origin = dataset_target[test_inx, :]
    #     predicted = model_outer.decoder(model_outer.interpolated_map[:, dataset_loc[test_inx, 1], dataset_loc[test_inx, 0]])
    #     predicted = predicted.detach().numpy()
    #     temp_loss_validation_gain = (origin[0] - predicted[0]) ** 2
    #     temp_loss_validation_cov = 0
    #     for cnt in range(num_input - 1):
    #         temp_loss_validation_cov += (origin[cnt + 1] - predicted[cnt + 1]) ** 2
    #
    #     temp_loss_validation_cov = temp_loss_validation_cov / (num_input - 1)
    #     loss_validation_gain += temp_loss_validation_gain
    #     loss_validation_cov += temp_loss_validation_cov
    #
    #     if m == 0:
    #         pre_input1 = dataset1[test_inx, :]
    #         pre_input2 = dataset2[test_inx, :]
    #         pre_input3 = dataset3[test_inx, :]
    #         pre_input4 = dataset4[test_inx, :]
    #         pre_input5 = dataset5[test_inx, :]
    #         pre_input = np.append(pre_input1, pre_input2)
    #         pre_input = np.append(pre_input, pre_input3)
    #         pre_input = np.append(pre_input, pre_input4)
    #         pre_input = np.append(pre_input, pre_input5)
    #         validation_input = torch.from_numpy(pre_input)
    #         AE_input = model_outer.encoder(validation_input.float())
    #         AE_interpolated = model_outer.interpolated_map[:, dataset_loc[test_inx, 1], dataset_loc[test_inx, 0]]
    #     else:
    #         pre_input1 = dataset1[test_inx, :]
    #         pre_input2 = dataset2[test_inx, :]
    #         pre_input3 = dataset3[test_inx, :]
    #         pre_input4 = dataset4[test_inx, :]
    #         pre_input5 = dataset5[test_inx, :]
    #         pre_input = np.append(pre_input1, pre_input2)
    #         pre_input = np.append(pre_input, pre_input3)
    #         pre_input = np.append(pre_input, pre_input4)
    #         pre_input = np.append(pre_input, pre_input5)
    #         validation_input = torch.from_numpy(pre_input)
    #         AE_input = torch.vstack((AE_input, model_outer.encoder(validation_input.float())))
    #         AE_interpolated = torch.vstack(
    #             (AE_interpolated, model_outer.interpolated_map[:, dataset_loc[test_inx, 1],
    #                               dataset_loc[test_inx, 0]]))
    #
    #     predicted_encdec = model_outer.decoder(model_outer.encoder(validation_input.float()))
    #     predicted_encdec = predicted_encdec.detach().numpy()
    #     temp_loss_validation_encdec_gain = (origin[0] - predicted_encdec[0]) ** 2
    #     temp_loss_validation_encdec_cov = 0
    #     for cnt in range(num_input - 1):
    #         temp_loss_validation_encdec_cov += (origin[cnt + 1] - predicted_encdec[cnt + 1]) ** 2
    #
    #     temp_loss_validation_encdec_cov = temp_loss_validation_encdec_cov / (num_input - 1)
    #     loss_validation_encdec_gain += temp_loss_validation_encdec_gain
    #     loss_validation_encdec_cov += temp_loss_validation_encdec_cov
    #
    # loss_validation_gain = loss_validation_gain / num_sample_validation
    # loss_validation_cov = loss_validation_cov / num_sample_validation
    # loss_validation_AE = criterion(AE_input, AE_interpolated)
    #
    # loss_validation_encdec_gain = loss_validation_encdec_gain / num_sample_validation
    # loss_validation_encdec_cov = loss_validation_encdec_cov / num_sample_validation

    # -Rui, no double loop
    if iter % iter_es_check == 0:
        model_outer.eval()
        model_AE.eval()
        decoded_in_val_v2_all = []
        with torch.no_grad():
            AE_input = model_outer.encoder(valid_input_ts)
            # normalized embeddings
            AE_input = AE_input / (AE_input.norm(dim=1, keepdim=True) + eps_ebd)

            # M
            for i_ebd_val_v2 in range(inx_ebd_val_v2.shape[0]):
                input_emb_img_tmp = input_emb_img.clone()
                input_emb_img_tmp[:, 0:-1, dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 1].long(),
                    dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 0].long()] = AE_input[inx_ebd_val_v2[i_ebd_val_v2], :].clone()
                input_emb_img_tmp[:, num_embedding, dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 1].long(),
                    dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 0].long()] = 1

                interpolated_val_v2 = model_AE(input_emb_img_tmp)
                interpolated_val_v2 = interpolated_val_v2[0, :]
                decoded_in_val_v2 = interpolated_val_v2[:, dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 1].long(),
                                dataset_loc_val[inx_ebd_val_v2[i_ebd_val_v2], 0].long()].clone()
                decoded_in_val_v2_all.append(decoded_in_val_v2)

            # normalized embeddings
            decoded_in_val_v2_all = torch.vstack(decoded_in_val_v2_all)
            decoded_in_val_v2_all = decoded_in_val_v2_all / (decoded_in_val_v2_all.norm(dim=1, keepdim=True) + eps_ebd)
            decoded_val_v2_all = model_outer.decoder(decoded_in_val_v2_all)

            loss_validation_gain_v2 = ((origin_valid_ts[inx_ebd_val_v2, 0] - decoded_val_v2_all[:, 0]) ** 2).mean()
            loss_validation_cov_v2 = ((origin_valid_ts[inx_ebd_val_v2, 1:num_input] - decoded_val_v2_all[:, 1:num_input]) ** 2).mean()
            loss_validation_AE_v2 = criterion(AE_input[inx_ebd_val_v2, :], decoded_in_val_v2_all)
            loss_validation_AE_v2_mse = eval_AE_cust(AE_input[inx_ebd_val_v2, :], decoded_in_val_v2_all, num_embedding)
            
            nmse_gain_val_v2, mse_cov_val_v2 = eval_cust(origin_valid_ts[inx_ebd_val_v2, :], decoded_val_v2_all, max_gain_dB, min_gain_dB, num_BS_antenna=8)
            
            loss_validation_v2 = WF_gain * loss_validation_gain_v2 + WF_cov * loss_validation_cov_v2 + WF_AE * loss_validation_AE_v2

            # M1  
            decoded_in_val_v1 = model_outer.interpolated_map[:, dataset_loc_val[inx_ebd_val_v1, 1].long(),
                              dataset_loc_val[inx_ebd_val_v1, 0].long()].transpose(1, 0).clone()
            # AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_validation[0:10], 1],
            #                       dataset_loc[rnd_list_validation[0:10], 0]].transpose(1, 0)
            # normalized embeddings
            decoded_in_val_v1 = decoded_in_val_v1 / (decoded_in_val_v1.norm(dim=1, keepdim=True) + eps_ebd)
            decoded_val_v1 = model_outer.decoder(decoded_in_val_v1)
            loss_validation_gain_v1 = ((origin_valid_ts[inx_ebd_val_v1, 0] - decoded_val_v1[:, 0]) ** 2).mean()
            loss_validation_cov_v1 = ((origin_valid_ts[inx_ebd_val_v1, 1:num_input] - decoded_val_v1[:, 1:num_input]) ** 2).mean()

            # predicted_encdec = model_outer.decoder(AE_input)
            # loss_validation_encdec_gain = ((origin_valid_ts[:, 0] - predicted_encdec[:, 0]) ** 2).mean()
            # loss_validation_encdec_cov = ((origin_valid_ts[:, 1:num_input] - predicted_encdec[:, 1:num_input]) ** 2).mean()

            nmse_gain_val_v1, mse_cov_val_v1 = eval_cust(origin_valid_ts[inx_ebd_val_v1, :], decoded_val_v1, max_gain_dB, min_gain_dB, num_BS_antenna=8)

            loss_validation_AE_v1 = criterion(AE_input[inx_ebd_val_v1, :], decoded_in_val_v1)
            loss_validation_AE_v1_mse = eval_AE_cust(AE_input[inx_ebd_val_v1, :], decoded_in_val_v1, num_embedding)

            loss_validation_v1 = WF_gain * loss_validation_gain_v1 + WF_cov * loss_validation_cov_v1 + WF_AE * loss_validation_AE_v1

            loss_validation = loss_validation_v1 * ratio_v1 + loss_validation_v2 * ratio_v2

        model_outer.train()
        model_AE.train()

    arr_loss_validation_v2.append(loss_validation_v2.item())
    arr_error_gain_validation_v2.append(loss_validation_gain_v2.item())
    arr_error_cov_validation_v2.append(loss_validation_cov_v2.item())
    arr_error_gain_validation_v2_nmse.append(nmse_gain_val_v2.item())
    arr_error_cov_validation_v2_mse.append(mse_cov_val_v2.item())
    arr_error_AE_validation_v2.append(loss_validation_AE_v2.item())
    arr_error_AE_validation_v2_mse.append(loss_validation_AE_v2_mse.item())
    
    arr_loss_validation_v1.append(loss_validation_v1.item())
    arr_error_gain_validation_v1.append(loss_validation_gain_v1.item())
    arr_error_cov_validation_v1.append(loss_validation_cov_v1.item())
    arr_error_gain_validation_v1_nmse.append(nmse_gain_val_v1.item())
    arr_error_cov_validation_v1_mse.append(mse_cov_val_v1.item())
    arr_error_AE_validation_v1.append(loss_validation_AE_v1.item())
    arr_error_AE_validation_v1_mse.append(loss_validation_AE_v1_mse.item())
    
    # model_outer.eval()
    # model_AE.eval()
    # with torch.no_grad():
        # AE_input = model_outer.encoder(valid_input_ts)
        # # normalized embeddings
        # AE_input = AE_input / (AE_input.norm(dim=1, keepdim=True) + eps_ebd)
        # AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_validation, 1],
                          # dataset_loc[rnd_list_validation, 0]].transpose(1, 0).clone()
        # # AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_validation[0:10], 1],
        # #                       dataset_loc[rnd_list_validation[0:10], 0]].transpose(1, 0)
        # # normalized embeddings
        # AE_interpolated = AE_interpolated / (AE_interpolated.norm(dim=1, keepdim=True) + eps_ebd)
        # predicted = model_outer.decoder(AE_interpolated)
        # loss_validation_gain = ((origin_valid_ts[:, 0] - predicted[:, 0]) ** 2).mean()
        # loss_validation_cov = ((origin_valid_ts[:, 1:num_input] - predicted[:, 1:num_input]) ** 2).mean()

        # predicted_encdec = model_outer.decoder(AE_input)
        # loss_validation_encdec_gain = ((origin_valid_ts[:, 0] - predicted_encdec[:, 0]) ** 2).mean()
        # loss_validation_encdec_cov = ((origin_valid_ts[:, 1:num_input] - predicted_encdec[:, 1:num_input]) ** 2).mean()

        # loss_validation_AE = criterion(AE_input, AE_interpolated)

        # loss_validation = WF_gain * loss_validation_gain + WF_cov * loss_validation_cov + WF_AE * loss_validation_AE
        # loss_validation_encdec = WF_gain * loss_validation_encdec_gain + WF_cov * loss_validation_encdec_cov

        # arr_loss_validation.append(loss_validation.item())
        # arr_error_gain_validation.append(loss_validation_gain.item())
        # arr_error_cov_validation.append(loss_validation_cov.item())
        # arr_error_AE_validation.append(loss_validation_AE.item())

        # arr_loss_validation_encdec.append(loss_validation_encdec.item())
        # arr_error_gain_validation_encdec.append(loss_validation_encdec_gain.item())
        # arr_error_cov_validation_encdec.append(loss_validation_encdec_cov.item())

    # model_outer.train()
    # model_AE.train()

    print(
        '>>> iter: %.0f/%.0f|tr: gain = %.4f / %.4f, cov = %.4f / %.4f, ae = %.4f, wd = %.2f / %.2f, comb = %.4f'
        '|val: gain = %.4f / %.4f, cov = %.4f / %.4f, ae = %.4f / %.4f, comb = %.4f / %.4f ' %
        (iter,
         max_iter,
         arr_error_gain_training_v1[iter],
         arr_error_gain_training_v2[iter],
         arr_error_cov_training_v1[iter],
         arr_error_cov_training_v2[iter],
         arr_error_AE_training[iter],
         arr_error_out_wd_training[iter],
         arr_error_dcae_wd_training[iter],
         arr_loss_training[iter],
         arr_error_gain_validation_v1[iter],
         arr_error_gain_validation_v2[iter],
         arr_error_cov_validation_v1[iter],
         arr_error_cov_validation_v2[iter],
         arr_error_AE_validation_v1[iter],
         arr_error_AE_validation_v2[iter],
         arr_loss_validation_v1[iter],
         arr_loss_validation_v2[iter]))

    # if epoch > min_epoch:
    #     # for i in range(size_window_proposed - 1):
    #     #     arr_minimum_losses[i] = arr_minimum_losses[i + 1]
    #     # arr_minimum_losses[size_window_proposed - 1] = loss_validation
    #     # -Rui, no loop
    #     arr_minimum_losses[0:-1] = arr_minimum_losses[1:].clone()
    #     arr_minimum_losses[size_window_proposed - 1] = loss_validation

    # -Rui, save the model with minimum validation loss
    if iter > min_iter_proposed:
        if iter % iter_es_check == 0:
            if minimum_loss >= loss_validation:
                #     if minimum_loss >= loss_validation_gain_both:
                #     minimum_loss = loss_validation_gain_both
                minimum_loss = loss_validation
                model_save.saved_converged_encoder.load_state_dict(copy.deepcopy(model_outer.encoder.state_dict()))
                model_save.saved_converged_decoder.load_state_dict(copy.deepcopy(model_outer.decoder.state_dict()))
                model_save.saved_converged_AE_encoder.load_state_dict(copy.deepcopy(model_AE.encoder.state_dict()))
                model_save.saved_converged_AE_decoder.load_state_dict(copy.deepcopy(model_AE.decoder.state_dict()))
                model_save.saved_input_map = model_outer.input_map.clone()
                model_save.saved_interpolated_map = model_outer.interpolated_map.clone()
                iter_val_es = iter

        # if minimum_loss < arr_minimum_losses.min() or epoch == max_epoch - 1:
        if iter == max_iter - 1:
            check_overfitting = 1
            model_outer.encoder.load_state_dict(copy.deepcopy(model_save.saved_converged_encoder.state_dict()))
            model_outer.decoder.load_state_dict(copy.deepcopy(model_save.saved_converged_decoder.state_dict()))
            model_AE.encoder.load_state_dict(model_save.saved_converged_AE_encoder.state_dict())
            model_AE.decoder.load_state_dict(model_save.saved_converged_AE_decoder.state_dict())
            model_outer.input_map = model_save.saved_input_map.clone()
            model_outer.interpolated_map = model_save.saved_interpolated_map.clone()
            print(f'convergence iteration:{iter + 1}')
            print(f'early stop iteration:{iter_val_es + 1}')
            print(f'ratio:{loss_validation.item() / minimum_loss.item()}')

    # if check_overfitting_encdec == 0:
    #
    #     if epoch > min_epoch:
    #         # for i in range(size_window_encdec - 1):
    #         #     arr_minimum_losses_encdec[i] = arr_minimum_losses_encdec[i + 1]
    #         # arr_minimum_losses_encdec[size_window_encdec - 1] = loss_validation_encdec
    #         # -Rui, no loop
    #         arr_minimum_losses_encdec[0:-1] = arr_minimum_losses_encdec[1:].clone()
    #         arr_minimum_losses_encdec[size_window_encdec - 1] = loss_validation_encdec
    #
    #     if epoch > min_epoch_encdec:
    #         if minimum_loss_encdec >= loss_validation_encdec:
    #             minimum_loss_encdec = loss_validation_encdec
    #             model_save.saved_converged_encoder.load_state_dict(copy.deepcopy(model_outer.encoder.state_dict()))
    #             model_save.saved_converged_decoder.load_state_dict(copy.deepcopy(model_outer.decoder.state_dict()))
    #
    #         if minimum_loss_encdec < arr_minimum_losses_encdec.min() or epoch == max_epoch - 1:
    #             model_outer.encoder.load_state_dict(copy.deepcopy(model_save.saved_converged_encoder.state_dict()))
    #             model_outer.decoder.load_state_dict(copy.deepcopy(model_save.saved_converged_decoder.state_dict()))
    #             print("epoch for encdec convergence: ##############################", str(epoch + 1))
    #             print(str(loss_validation_encdec.item() / minimum_loss_encdec.item()))
    #             check_overfitting_encdec = 1

    if check_overfitting == 1:
        break

    # print(time.time() - time_t)
##################################################
# save training and validation log
with open(resultpath + 'arr_loss_training.pkl', 'wb') as f:
    pickle.dump(arr_loss_training, f)
    savemat(resultpath + 'arr_loss_training.mat',{'arr_loss_training':arr_loss_training})
f.close()
with open(resultpath + 'arr_error_gain_training_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_training_v2, f)
    savemat(resultpath + 'arr_error_gain_training_v2.mat',{'arr_error_gain_training_v2':arr_error_gain_training_v2})
f.close()
with open(resultpath + 'arr_error_gain_training_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_training_v1, f)
    savemat(resultpath + 'arr_error_gain_training_v1.mat',{'arr_error_gain_training_v1':arr_error_gain_training_v1})
f.close()
with open(resultpath + 'arr_error_cov_training_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_training_v2, f)
    savemat(resultpath + 'arr_error_cov_training_v2.mat',{'arr_error_cov_training_v2':arr_error_cov_training_v2})
f.close()
with open(resultpath + 'arr_error_cov_training_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_training_v1, f)
    savemat(resultpath + 'arr_error_cov_training_v1.mat',{'arr_error_cov_training_v1':arr_error_cov_training_v1})
f.close()
with open(resultpath + 'arr_error_gain_training_v2_nmse.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_training_v2_nmse, f)
    savemat(resultpath + 'arr_error_gain_training_v2_nmse.mat',{'arr_error_gain_training_v2_nmse':arr_error_gain_training_v2_nmse})
f.close()
with open(resultpath + 'arr_error_gain_training_v1_nmse.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_training_v1_nmse, f)
    savemat(resultpath + 'arr_error_gain_training_v1_nmse.mat',{'arr_error_gain_training_v1_nmse':arr_error_gain_training_v1_nmse})
f.close()
with open(resultpath + 'arr_error_cov_training_v2_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_training_v2_mse, f)
    savemat(resultpath + 'arr_error_cov_training_v2_mse.mat',{'arr_error_cov_training_v2_mse':arr_error_cov_training_v2_mse})
f.close()
with open(resultpath + 'arr_error_cov_training_v1_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_training_v1_mse, f)
    savemat(resultpath + 'arr_error_cov_training_v1_mse.mat',{'arr_error_cov_training_v1_mse':arr_error_cov_training_v1_mse})
f.close()
with open(resultpath + 'arr_error_AE_training.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_training, f)
    savemat(resultpath + 'arr_error_AE_training.mat',{'arr_error_AE_training':arr_error_AE_training})
f.close()
with open(resultpath + 'arr_error_AE_training_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_training_v1, f)
    savemat(resultpath + 'arr_error_AE_training_v1.mat',{'arr_error_AE_training_v1':arr_error_AE_training_v1})
f.close()
with open(resultpath + 'arr_error_AE_training_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_training_v2, f)
    savemat(resultpath + 'arr_error_AE_training_v2.mat',{'arr_error_AE_training_v2':arr_error_AE_training_v2})
f.close()
with open(resultpath + 'arr_error_AE_training_v1_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_training_v1_mse, f)
    savemat(resultpath + 'arr_error_AE_training_v1_mse.mat',{'arr_error_AE_training_v1_mse':arr_error_AE_training_v1_mse})
f.close()
with open(resultpath + 'arr_error_AE_training_v2_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_training_v2_mse, f)
    savemat(resultpath + 'arr_error_AE_training_v2_mse.mat',{'arr_error_AE_training_v2_mse':arr_error_AE_training_v2_mse})
f.close()
with open(resultpath + 'arr_loss_validation_v2.pkl', 'wb') as f:
    pickle.dump(arr_loss_validation_v2, f)
    savemat(resultpath + 'arr_loss_validation_v2.mat',{'arr_loss_validation_v2':arr_loss_validation_v2})    
f.close()
with open(resultpath + 'arr_loss_validation_v1.pkl', 'wb') as f:
    pickle.dump(arr_loss_validation_v1, f)
    savemat(resultpath + 'arr_loss_validation_v1.mat',{'arr_loss_validation_v1':arr_loss_validation_v1})    
f.close()
with open(resultpath + 'arr_error_gain_validation_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_validation_v2, f)
    savemat(resultpath + 'arr_error_gain_validation_v2.mat',{'arr_error_gain_validation_v2':arr_error_gain_validation_v2})    
f.close()
with open(resultpath + 'arr_error_gain_validation_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_validation_v1, f)
    savemat(resultpath + 'arr_error_gain_validation_v1.mat',{'arr_error_gain_validation_v1':arr_error_gain_validation_v1})    
f.close()
with open(resultpath + 'arr_error_cov_validation_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_validation_v2, f)
    savemat(resultpath + 'arr_error_cov_validation_v2.mat',{'arr_error_cov_validation_v2':arr_error_cov_validation_v2})    
f.close()
with open(resultpath + 'arr_error_cov_validation_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_validation_v1, f)
    savemat(resultpath + 'arr_error_cov_validation_v1.mat',{'arr_error_cov_validation_v1':arr_error_cov_validation_v1})    
f.close()
with open(resultpath + 'arr_error_gain_validation_v2_nmse.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_validation_v2_nmse, f)
    savemat(resultpath + 'arr_error_gain_validation_v2_nmse.mat',{'arr_error_gain_validation_v2_nmse':arr_error_gain_validation_v2_nmse})    
f.close()
with open(resultpath + 'arr_error_gain_validation_v1_nmse.pkl', 'wb') as f:
    pickle.dump(arr_error_gain_validation_v1_nmse, f)
    savemat(resultpath + 'arr_error_gain_validation_v1_nmse.mat',{'arr_error_gain_validation_v1_nmse':arr_error_gain_validation_v1_nmse})    
f.close()
with open(resultpath + 'arr_error_cov_validation_v2_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_validation_v2_mse, f)
    savemat(resultpath + 'arr_error_cov_validation_v2_mse.mat',{'arr_error_cov_validation_v2_mse':arr_error_cov_validation_v2_mse})    
f.close()
with open(resultpath + 'arr_error_cov_validation_v1_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_cov_validation_v1_mse, f)
    savemat(resultpath + 'arr_error_cov_validation_v1_mse.mat',{'arr_error_cov_validation_v1_mse':arr_error_cov_validation_v1_mse})    
f.close()
with open(resultpath + 'arr_error_AE_validation_v2.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_validation_v2, f)
    savemat(resultpath + 'arr_error_AE_validation_v2.mat',{'arr_error_AE_validation_v2':arr_error_AE_validation_v2})    
f.close()
with open(resultpath + 'arr_error_AE_validation_v1.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_validation_v1, f)
    savemat(resultpath + 'arr_error_AE_validation_v1.mat',{'arr_error_AE_validation_v1':arr_error_AE_validation_v1})    
f.close()
with open(resultpath + 'arr_error_AE_validation_v2_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_validation_v2_mse, f)
    savemat(resultpath + 'arr_error_AE_validation_v2_mse.mat',{'arr_error_AE_validation_v2_mse':arr_error_AE_validation_v2_mse})    
f.close()
with open(resultpath + 'arr_error_AE_validation_v1_mse.pkl', 'wb') as f:
    pickle.dump(arr_error_AE_validation_v1_mse, f)
    savemat(resultpath + 'arr_error_AE_validation_v1_mse.mat',{'arr_error_AE_validation_v1_mse':arr_error_AE_validation_v1_mse})    
f.close()
with open(resultpath + 'arr_error_out_wd_training.pkl', 'wb') as f:
    pickle.dump(arr_error_out_wd_training, f)
    savemat(resultpath + 'arr_error_out_wd_training.mat',{'arr_error_out_wd_training':arr_error_out_wd_training})    
f.close()
with open(resultpath + 'arr_error_dcae_wd_training.pkl', 'wb') as f:
    pickle.dump(arr_error_dcae_wd_training, f)
    savemat(resultpath + 'arr_error_dcae_wd_training.mat',{'arr_error_dcae_wd_training':arr_error_dcae_wd_training})    
f.close()
with open(resultpath + 'arr_error_dcae_wd_training.pkl', 'wb') as f:
    pickle.dump(arr_error_dcae_wd_training, f)
    savemat(resultpath + 'arr_error_dcae_wd_training.mat',{'arr_error_dcae_wd_training':arr_error_dcae_wd_training})    
f.close()    
##################################################

criterion_eval = nn.MSELoss(reduction='sum')

sum_SE_gain = 0
sum_SO_gain = 0  # SO : squared original
sum_MSE_cov = 0

sum_SE_CSI_gain = 0
sum_SO_CSI_gain = 0  # SO : squared original
sum_MSE_CSI_cov = 0

# -Rui, GM's version, one embedding map for one embedding
sum_SE_Both_gain_oo = 0
sum_SO_Both_gain_oo = 0  # SO : squared original
sum_MSE_Both_cov_oo = 0
# -Rui, one embedding map for all embeddings
# sum_SE_Both_gain_oa = 0
# sum_SO_Both_gain_oa = 0  # SO : squared original
# sum_MSE_Both_cov_oa = 0

sum_SE_wo_AE_gain = 0
sum_SO_wo_AE_gain = 0  # SO : squared original
sum_MSE_wo_AE_cov = 0

# cnt_index = 0
sample_interval = 1  # -Rui. testing, not applicable to other numbers

# list_NMSE_gain = torch.zeros(int(num_tot_row * num_each_row / sample_interval**2)).to(DEVICE)
# median_NMSE_gain = 0

# list_NMSE_CSI_gain = torch.zeros(int(num_tot_row * num_each_row / sample_interval ** 2)).to(DEVICE)
# median_NMSE_CSI_gain = 0

# -Rui, GM's version, one embedding map for one embedding
# list_NMSE_Both_gain_oo = torch.zeros(int(num_tot_row * num_each_row / sample_interval**2)).to(DEVICE)
# median_NMSE_Both_gain_oo = 0
# # -Rui, one embedding map for all embeddings
# list_NMSE_Both_gain_oa = torch.zeros(int(num_tot_row * num_each_row / sample_interval**2)).to(DEVICE)
# median_NMSE_Both_gain_oa = 0

# list_NMSE_wo_AE_gain = torch.zeros(int(num_tot_row * num_each_row / sample_interval ** 2)).to(DEVICE)
# median_NMSE_wo_AE_gain = 0

# for m in range(int(num_tot_row/sample_interval)):
#     for n in range(int(num_each_row/sample_interval)):
#         inx_cnt = (m*sample_interval) * num_each_row + (n*sample_interval)
# #
#         original = copy.deepcopy(dataset_target[inx_cnt][:])
#         temp_map = torch.tensor(model_save.saved_interpolated_map)
#         predicted = model_outer.decoder(temp_map[:, dataset_loc[inx_cnt, 1], dataset_loc[inx_cnt, 0]])
#
#         temp_input1 = copy.deepcopy(dataset1[inx_cnt][:])
#         temp_input2 = copy.deepcopy(dataset2[inx_cnt][:])
#         temp_input3 = copy.deepcopy(dataset3[inx_cnt][:])
#         temp_input4 = copy.deepcopy(dataset4[inx_cnt][:])
#         temp_input5 = copy.deepcopy(dataset5[inx_cnt][:])
#
#         temp_input = np.append(temp_input1, temp_input2)
#         temp_input = np.append(temp_input, temp_input3)
#         temp_input = np.append(temp_input, temp_input4)
#         temp_input = np.append(temp_input, temp_input5)
#         predicted_pre = torch.from_numpy(temp_input)
#         predicted_CSI = model_outer.decoder(model_outer.encoder(predicted_pre.float()))
#
#         embedding_value = model_outer.encoder(predicted_pre.float())
#         next_input_map = torch.tensor(model_save.saved_input_map)
#         next_input_map[0:-1, dataset_loc[inx_cnt, 1], dataset_loc[inx_cnt, 0]] = embedding_value
#         next_input_map[num_embedding, dataset_loc[inx_cnt, 1], dataset_loc[inx_cnt, 0]] = 1
#         next_input_map = next_input_map[None, :]
#         new_interpolated = model_AE.decoder(model_AE.encoder(next_input_map))
#         new_interpolated = new_interpolated[0, :]
#         selected = new_interpolated[:, dataset_loc[inx_cnt, 1], dataset_loc[inx_cnt, 0]]
#         predicted_Both = model_outer.decoder(selected)
#
#         predicted_wo_AE = model_wo_AE.decoder(model_wo_AE.encoder(predicted_pre.float()))
#
#         # convert gain values in -1 ~ +1 range to original values in dB scale
#         original_gain = (max_gain_dB - min_gain_dB) * (original[0] + 1) / 2 + min_gain_dB
#         original_gain_linear = 10 ** (original_gain / 10)
#
#         predicted_gain = (max_gain_dB - min_gain_dB) * (predicted[0].detach().numpy() + 1) / 2 + min_gain_dB
#         predicted_gain_linear = 10 ** (predicted_gain / 10)
#         predicted_CSI_gain = (max_gain_dB - min_gain_dB) * (predicted_CSI[0].detach().numpy() + 1) / 2 + min_gain_dB
#         predicted_CSI_gain_linear = 10 ** (predicted_CSI_gain / 10)
#         predicted_Both_gain = (max_gain_dB - min_gain_dB) * (predicted_Both[0].detach().numpy() + 1) / 2 + min_gain_dB
#         predicted_Both_gain_linear = 10 ** (predicted_Both_gain / 10)
#         predicted_wo_AE_gain = (max_gain_dB - min_gain_dB) * (predicted_wo_AE[0].detach().numpy() + 1) / 2 + min_gain_dB
#         predicted_wo_AE_gain_linear = 10 ** (predicted_wo_AE_gain / 10)
#
#         ####################
#
#         SE_gain = (original_gain_linear - predicted_gain_linear) ** 2
#         SO_gain = original_gain_linear ** 2
#
#         sum_SE_gain += SE_gain
#         sum_SO_gain += SO_gain
#
#         MSE_cov = (2 * criterion_eval(predicted[1:num_input], torch.tensor(original[1:num_input]))).detach().numpy()
#
#         sum_MSE_cov += MSE_cov / num_BS_antenna**2
#
#         list_NMSE_gain[cnt_index] = SE_gain / SO_gain
#
#         ####################
#
#         SE_CSI_gain = (original_gain_linear - predicted_CSI_gain_linear) ** 2
#
#         sum_SE_CSI_gain += SE_CSI_gain
#         sum_SO_CSI_gain += SO_gain
#
#         MSE_CSI_cov = (2 * criterion_eval(predicted_CSI[1:num_input],
#                                   torch.tensor(original[1:num_input]))).detach().numpy()
#
#         sum_MSE_CSI_cov += MSE_CSI_cov / num_BS_antenna**2
#
#         list_NMSE_CSI_gain[cnt_index] = SE_CSI_gain / SO_gain
#
#         ####################
#
#         SE_Both_gain = (original_gain_linear - predicted_Both_gain_linear) ** 2
#
#         sum_SE_Both_gain += SE_Both_gain
#         sum_SO_Both_gain += SO_gain
#
#         MSE_Both_cov = (2 * criterion_eval(predicted_Both[1:num_input],
#                                   torch.tensor(original[1:num_input]))).detach().numpy()
#
#         sum_MSE_Both_cov += MSE_Both_cov / num_BS_antenna**2
#
#         list_NMSE_Both_gain[cnt_index] = SE_Both_gain / SO_gain
#
#         ##########
#
#         SE_wo_AE_gain = (original_gain_linear - predicted_wo_AE_gain_linear) ** 2
#
#         sum_SE_wo_AE_gain += SE_wo_AE_gain
#         sum_SO_wo_AE_gain += SO_gain
#
#         MSE_wo_AE_cov = (2 * criterion_eval(predicted_wo_AE[1:num_input],
#                                   torch.tensor(original[1:num_input]))).detach().numpy()
#
#         sum_MSE_wo_AE_cov += MSE_wo_AE_cov / num_BS_antenna**2
#
#         list_NMSE_wo_AE_gain[cnt_index] = SE_wo_AE_gain / SO_gain
#
#         cnt_index += 1


#  -Rui, no double loops
inx_cnt_all = []
num_cnt_all = 0
num_tot_row_sub = int(num_tot_row / sample_interval)
num_each_row_sub = int(num_each_row / sample_interval)
for m in range(num_tot_row_sub):
    for n in range(num_each_row_sub):
        inx_cnt_all.append((m * sample_interval) * num_each_row + (n * sample_interval))
        num_cnt_all += 1

#  -Rui, indices of all training splits
inx_ebd_ats = []
for n_split in range(n_test_ebd_split):
    inx_ebd_sv1_test_perm = torch.randperm(num_sample_ebd_sv1).to(DEVICE)
    inx_ebd_ats.append(inx_ebd_sv1[inx_ebd_sv1_test_perm[0:num_sample_ebd_s]])

# -Rui, no double loop
model_AE.eval()
model_outer.eval()
model_save.eval()
model_wo_AE.eval()
with torch.no_grad():
    ############################################################
    # -Rui, refinement of input and interpolated map with all training data
    # Fill locs in embedding maps due to random sampling in the training
    # Can also be used in P1, but not necessary (as P1 alrdy performs well enough)
    # for (inputs, labels) in data_loader_anchor:
    #     # -Rui, sample all training embeddings
    #     inx_ebd = torch.randperm(inputs.shape[0]).to(DEVICE)
    #     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    #     model_outer(inputs, labels, inx_ebd, DEVICE)
    #
    # model_save.saved_input_map = model_outer.input_map
    # model_save.saved_interpolated_map = model_outer.interpolated_map
    # ############################################################

    temp_input1 = copy.deepcopy(dataset1[inx_cnt_all][:])
    temp_input2 = copy.deepcopy(dataset2[inx_cnt_all][:])
    temp_input3 = copy.deepcopy(dataset3[inx_cnt_all][:])
    temp_input4 = copy.deepcopy(dataset4[inx_cnt_all][:])
    temp_input5 = copy.deepcopy(dataset5[inx_cnt_all][:])

    temp_input = np.append(temp_input1, temp_input2, axis=1)
    temp_input = np.append(temp_input, temp_input3, axis=1)
    temp_input = np.append(temp_input, temp_input4, axis=1)
    temp_input = np.append(temp_input, temp_input5, axis=1)
    predicted_pre = torch.from_numpy(temp_input).float().to(DEVICE)
    encoded_CSI_test = model_outer.encoder(predicted_pre)
    # normalized embedding
    encoded_CSI_test = encoded_CSI_test / (encoded_CSI_test.norm(dim=1, keepdim=True) + eps_ebd)

    with open(resultpath + 'encoded_CSI_test.pkl', 'wb') as f:
        pickle.dump(encoded_CSI_test, f)
    f.close()

    predicted_CSI = model_outer.decoder(encoded_CSI_test)
    original = torch.from_numpy(copy.deepcopy(dataset_target[inx_cnt_all][:])).float().to(DEVICE)
    dataset_loc_test = torch.from_numpy(copy.deepcopy(dataset_loc[inx_cnt_all][:])).to(DEVICE)

    # -Rui, load multiple (random) training splits into batches
    input_map_mts = torch.zeros(n_test_ebd_split, num_embedding + 1, num_tot_row, num_each_row).to(DEVICE)
    for (inputs, labels) in data_loader_anchor:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        embedding_value_train = model_outer.encoder(inputs)
        # normalized embeddings
        embedding_value_train = embedding_value_train / (embedding_value_train.norm(dim=1, keepdim=True) + eps_ebd)
    for i_split in range(n_test_ebd_split):
        inx_ebd_split = inx_ebd_ats[i_split]
        input_map_mts[i_split, 0:-1, labels[inx_ebd_split, 1].long(), labels[inx_ebd_split, 0].long()] \
            = embedding_value_train[inx_ebd_split, :].transpose(1, 0)
        input_map_mts[i_split, num_embedding, labels[inx_ebd_split, 1].long(), labels[inx_ebd_split, 0].long()] = 1

    # -Rui, test with locations (interpolated maps)
    interpolated_map_mts = model_AE.decoder(model_AE.encoder(input_map_mts))
    selected_mts = interpolated_map_mts[:, :, dataset_loc_test[:, 1].long(), dataset_loc_test[:, 0].long()]
    selected_mts = selected_mts.transpose(2, 1).reshape(n_test_ebd_split * dataset_loc_test.shape[0], num_embedding)
    # normalized embeddings
    selected_mts = selected_mts / (selected_mts.norm(dim=1, keepdim=True) + eps_ebd)

    with open(resultpath + 'selected_mts.pkl', 'wb') as f:
        pickle.dump(selected_mts, f)
    f.close()

    predicted = model_outer.decoder(selected_mts)
    predicted = predicted.reshape(n_test_ebd_split, dataset_loc_test.shape[0], predicted.shape[1])

    predicted_gain = (max_gain_dB - min_gain_dB) * (predicted[:, :, 0] + 1) / 2 + min_gain_dB
    predicted_gain_linear = 10 ** (predicted_gain / 10)

    with open(resultpath + 'predicted_gain.pkl', 'wb') as f:
        pickle.dump(predicted_gain, f)
    f.close()

    with open(resultpath + 'predicted_gain_linear.pkl', 'wb') as f:
        pickle.dump(predicted_gain_linear, f)
    f.close()

    MSE_cov = (2 * (predicted[:, :, 1:num_input] - original[:, 1:num_input]) ** 2).sum(axis=2)

    with open(resultpath + 'predicted.pkl', 'wb') as f:
        pickle.dump(predicted, f)
    f.close()

    with open(resultpath + 'original.pkl', 'wb') as f:
        pickle.dump(original, f)
    f.close()

    with open(resultpath + 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    f.close()

    with open(resultpath + 'inx_ebd_ats.pkl', 'wb') as f:
        pickle.dump(inx_ebd_ats[0], f)
    f.close()

    with open(resultpath + 'dataset_loc_test.pkl', 'wb') as f:
        pickle.dump(dataset_loc_test, f)
    f.close()

    with open(resultpath + 'MSE_cov.pkl', 'wb') as f:
        pickle.dump(MSE_cov, f)
    f.close()

    # -Rui, GM's version, one embedding map for one embedding
    embedding_value_oo = model_outer.encoder(predicted_pre)
    # normalized embeddings
    embedding_value_oo = embedding_value_oo / (embedding_value_oo.norm(dim=1, keepdim=True) + eps_ebd)

    with open(resultpath + 'embedding_value_oo.pkl', 'wb') as f:
        pickle.dump(embedding_value_oo, f)
    f.close()

    predicted_Both_gain_oo = torch.zeros(n_test_ebd_split, embedding_value_oo.shape[0]).to(DEVICE)
    MSE_Both_cov_oo = torch.zeros(n_test_ebd_split, embedding_value_oo.shape[0]).to(DEVICE)
    predicted_Both_oo_all = torch.zeros(n_test_ebd_split, embedding_value_oo.shape[0], predicted.shape[2]).to(DEVICE)
    selected_oo_all = torch.zeros(n_test_ebd_split, embedding_value_oo.shape[0], num_embedding).to(DEVICE)
    for n_ebd in range(num_cnt_all):
        next_input_map_oo_test = input_map_mts.clone()
        next_input_map_oo_test[:, 0:-1, dataset_loc_test[n_ebd, 1], dataset_loc_test[n_ebd, 0]] = embedding_value_oo[
                                                                                                  n_ebd, :].clone()
        next_input_map_oo_test[:, num_embedding, dataset_loc_test[n_ebd, 1], dataset_loc_test[n_ebd, 0]] = 1
        new_interpolated_oo_test = model_AE.decoder(model_AE.encoder(next_input_map_oo_test))
        selected_oo = new_interpolated_oo_test[:, :, dataset_loc_test[n_ebd, 1], dataset_loc_test[n_ebd, 0]].clone()
        # normalized embeddings
        selected_oo = selected_oo / (selected_oo.norm(dim=1, keepdim=True) + eps_ebd)
        selected_oo_all[:, n_ebd, :] = selected_oo
        predicted_Both_oo = model_outer.decoder(selected_oo)
        predicted_Both_oo_all[:, n_ebd, :] = predicted_Both_oo
        predicted_Both_gain_oo[:, n_ebd] = (max_gain_dB - min_gain_dB) * (predicted_Both_oo[:, 0] + 1) / 2 + min_gain_dB
        # MSE_Both_cov_oo[i_iter] = (2 * criterion_eval(predicted_Both_oo[:, 1:num_input], original[n_ebd, 1:num_input]))
        MSE_Both_cov_oo[:, n_ebd] = (2 * (predicted_Both_oo[:, 1:num_input] - original[n_ebd, 1:num_input]) ** 2).sum(
            axis=1)
        if n_ebd % 10000 == 0:
            print("Testing oo case, iter: " + str(n_ebd))

    with open(resultpath + 'predicted_Both_oo_all.pkl', 'wb') as f:
        pickle.dump(predicted_Both_oo_all, f)
    f.close()

    with open(resultpath + 'predicted_Both_gain_oo.pkl', 'wb') as f:
        pickle.dump(predicted_Both_gain_oo, f)
    f.close()

    with open(resultpath + 'MSE_Both_cov_oo.pkl', 'wb') as f:
        pickle.dump(MSE_Both_cov_oo, f)
    f.close()

    with open(resultpath + 'selected_oo_all.pkl', 'wb') as f:
        pickle.dump(selected_oo_all, f)
    f.close()

    predicted_wo_AE_encoded = model_wo_AE.encoder(predicted_pre)
    # normalized embeddings
    predicted_wo_AE_encoded = predicted_wo_AE_encoded / (predicted_wo_AE_encoded.norm(dim=1, keepdim=True) + eps_ebd)
    predicted_wo_AE = model_wo_AE.decoder(predicted_wo_AE_encoded)

    # convert gain values in -1 ~ +1 range to original values in dB scale
    original_gain = (max_gain_dB - min_gain_dB) * (original[:, 0] + 1) / 2 + min_gain_dB

    with open(resultpath + 'original_gain.pkl', 'wb') as f:
        pickle.dump(original_gain, f)
    f.close()

    original_gain_linear = 10 ** (original_gain / 10)

    with open(resultpath + 'original_gain_linear.pkl', 'wb') as f:
        pickle.dump(original_gain_linear, f)
    f.close()

    predicted_CSI_gain = (max_gain_dB - min_gain_dB) * (predicted_CSI[:, 0] + 1) / 2 + min_gain_dB
    predicted_CSI_gain_linear = 10 ** (predicted_CSI_gain / 10)

    # -Rui, GM's version, one embedding map for one embedding
    # predicted_Both_gain = (max_gain_dB - min_gain_dB) * (predicted_Both[:, 0].detach().numpy() + 1) / 2 + min_gain_dB
    predicted_Both_gain_linear_oo = 10 ** (predicted_Both_gain_oo / 10)

    with open(resultpath + 'predicted_Both_gain_linear_oo.pkl', 'wb') as f:
        pickle.dump(predicted_Both_gain_linear_oo, f)
    f.close()

    # -Rui, one embedding map for all embeddings
    # predicted_Both_gain_oa = (max_gain_dB - min_gain_dB) * (predicted_Both_oa[:, 0] + 1) / 2 + min_gain_dB
    # predicted_Both_gain_linear_oa = 10 ** (predicted_Both_gain_oa / 10)

    predicted_wo_AE_gain = (max_gain_dB - min_gain_dB) * (predicted_wo_AE[:, 0] + 1) / 2 + min_gain_dB
    predicted_wo_AE_gain_linear = 10 ** (predicted_wo_AE_gain / 10)

    ####################
    # -Rui, channel gain maps
    img_original_gain = original_gain.reshape(int(num_tot_row / sample_interval),
                                              int(num_each_row / sample_interval))
    img_predicted_gain = predicted_gain.reshape(n_test_ebd_split, int(num_tot_row / sample_interval),
                                                int(num_each_row / sample_interval))
    img_predicted_CSI_gain = predicted_CSI_gain.reshape(int(num_tot_row / sample_interval),
                                                        int(num_each_row / sample_interval))
    img_predicted_Both_gain_oo = predicted_Both_gain_oo.reshape(n_test_ebd_split, int(num_tot_row / sample_interval),
                                                                int(num_each_row / sample_interval))
    img_predicted_wo_AE_gain = predicted_wo_AE_gain.reshape(int(num_tot_row / sample_interval),
                                                            int(num_each_row / sample_interval))
    ####################

    SE_gain = (original_gain_linear - predicted_gain_linear) ** 2
    SO_gain = original_gain_linear ** 2

    sum_SE_gain = SE_gain.sum(axis=1)
    sum_SO_gain = SO_gain.sum()

    # MSE_cov = (2 * criterion_eval(predicted[:, 1:num_input], original[:, 1:num_input]))
    MSE_cov2 = MSE_cov / num_BS_antenna ** 2

    # sum_MSE_cov = MSE_cov / num_BS_antenna**2
    sum_MSE_cov = MSE_cov2.sum(axis=1)

    list_NMSE_gain = SE_gain / SO_gain

    ####################

    SE_CSI_gain = (original_gain_linear - predicted_CSI_gain_linear) ** 2

    sum_SE_CSI_gain = SE_CSI_gain.sum()
    sum_SO_CSI_gain = SO_gain.sum()

    # MSE_CSI_cov = (2 * criterion_eval(predicted_CSI[:, 1:num_input], original[:, 1:num_input]))
    MSE_CSI_cov = 2 * ((predicted_CSI[:, 1:num_input] - original[:, 1:num_input]) ** 2).sum(axis=1)

    with open(resultpath + 'MSE_CSI_cov.pkl', 'wb') as f:
        pickle.dump(MSE_CSI_cov, f)
    f.close()

    MSE_CSI_cov2 = MSE_CSI_cov / num_BS_antenna ** 2

    # sum_MSE_CSI_cov = MSE_CSI_cov / num_BS_antenna**2
    sum_MSE_CSI_cov = MSE_CSI_cov2.sum()

    list_NMSE_CSI_gain = SE_CSI_gain / SO_gain

    ####################
    # -Rui, GM's version, one embedding map for one embedding
    SE_Both_gain_oo = (original_gain_linear - predicted_Both_gain_linear_oo) ** 2

    sum_SE_Both_gain_oo = SE_Both_gain_oo.sum(axis=1)
    sum_SO_Both_gain_oo = SO_gain.sum()

    # MSE_Both_cov = (2 * criterion_eval(predicted_Both[:, 1:num_input],
    #                          torch.tensor(original[:, 1:num_input]))).detach().numpy()

    # sum_MSE_Both_cov_oo = MSE_Both_cov_oo / num_BS_antenna ** 2
    MSE_Both_cov2_oo = MSE_Both_cov_oo / num_BS_antenna ** 2
    sum_MSE_Both_cov_oo = MSE_Both_cov2_oo.sum(axis=1)

    list_NMSE_Both_gain_oo = SE_Both_gain_oo / SO_gain

    # -Rui, one embedding map for all embedding
    # SE_Both_gain_oa = (original_gain_linear - predicted_Both_gain_linear_oa) ** 2
    #
    # sum_SE_Both_gain_oa = SE_Both_gain_oa.sum()
    # sum_SO_Both_gain_oa = SO_gain.sum()
    #
    # MSE_Both_cov_oa = (2 * criterion_eval(predicted_Both_oa[:, 1:num_input], original[:, 1:num_input]))
    #
    # sum_MSE_Both_cov_oa = MSE_Both_cov_oa / num_BS_antenna**2
    #
    # list_NMSE_Both_gain_oa = SE_Both_gain_oa / SO_gain

    ##########

    SE_wo_AE_gain = (original_gain_linear - predicted_wo_AE_gain_linear) ** 2

    sum_SE_wo_AE_gain = SE_wo_AE_gain.sum()
    sum_SO_wo_AE_gain = SO_gain.sum()

    # MSE_wo_AE_cov = (2 * criterion_eval(predicted_wo_AE[:, 1:num_input], original[:, 1:num_input]))
    MSE_wo_AE_cov = 2 * ((predicted_wo_AE[:, 1:num_input] - original[:, 1:num_input]) ** 2).sum(axis=1)
    MSE_wo_AE_cov2 = MSE_wo_AE_cov / num_BS_antenna ** 2

    # sum_MSE_wo_AE_cov = MSE_wo_AE_cov / num_BS_antenna**2
    sum_MSE_wo_AE_cov = MSE_wo_AE_cov2.sum()

    list_NMSE_wo_AE_gain = SE_wo_AE_gain / SO_gain

    ##############################################
    # -Rui, RMSE map of normalized covariance
    img_RMSE_cov2 = MSE_cov2.sqrt().reshape(n_test_ebd_split, int(num_tot_row / sample_interval),
                                            int(num_each_row / sample_interval))
    img_RMSE_cov2_dB = 20 * img_RMSE_cov2.log10()

    img_RMSE_CSI_cov2 = MSE_CSI_cov2.sqrt().reshape(int(num_tot_row / sample_interval),
                                                    int(num_each_row / sample_interval))
    img_RMSE_CSI_cov2_dB = 20 * img_RMSE_CSI_cov2.log10()

    img_RMSE_Both_cov2_oo = MSE_Both_cov2_oo.sqrt().reshape(n_test_ebd_split, int(num_tot_row / sample_interval),
                                                            int(num_each_row / sample_interval))
    img_RMSE_Both_cov2_dB_oo = 20 * img_RMSE_Both_cov2_oo.log10()

    img_RMSE_wo_AE_cov2 = MSE_wo_AE_cov2.sqrt().reshape(int(num_tot_row / sample_interval),
                                                        int(num_each_row / sample_interval))
    img_RMSE_wo_AE_cov2_dB = 20 * img_RMSE_wo_AE_cov2.log10()

    ##############################################

    median_NMSE_wo_AE_gain = list_NMSE_wo_AE_gain.median()

    # list_NMSE_gain = list_NMSE_gain.tolist()
    # list_NMSE_gain.sort()
    # median_NMSE_gain = list_NMSE_gain[int((num_tot_row * num_each_row / sample_interval**2) / 2)]
    median_NMSE_gain = list_NMSE_gain.median(dim=1)[0]

    median_NMSE_CSI_gain = list_NMSE_CSI_gain.median()
    # -Rui, GM's version, one embedding map for one embedding
    # list_NMSE_Both_gain_oo = list_NMSE_Both_gain_oo.tolist()
    # list_NMSE_Both_gain_oo.sort()
    # median_NMSE_Both_gain_oo = list_NMSE_Both_gain_oo[int((num_tot_row * num_each_row / sample_interval**2) / 2)]
    median_NMSE_Both_gain_oo = list_NMSE_Both_gain_oo.median(dim=1)[0]
    # # -Rui, one embedding map for all embedding
    # list_NMSE_Both_gain_oa = list_NMSE_Both_gain_oa.tolist()
    # list_NMSE_Both_gain_oa.sort()
    # median_NMSE_Both_gain_oa = list_NMSE_Both_gain_oa[int((num_tot_row * num_each_row / sample_interval**2) / 2)]

model_AE.train()
model_outer.train()
model_save.train()
model_wo_AE.train()

##################################################

# x_axis = list(np.arange(1, max_epoch_wo_AE + 1))
#
# plt.plot(x_axis, arr_loss_validation_wo_AE, 'b-', label='validation set')
# plt.plot(x_axis, arr_loss_validation_wo_AE, 'b.')
# plt.plot(x_axis, arr_loss_training_wo_AE, 'r-', label='training set')
# plt.plot(x_axis, arr_loss_training_wo_AE, 'r.')
# plt.legend()
# plt.title('loss graph (without AE)')
# plt.show()

#####

# x_axis = list(np.arange(1, max_epoch + 1))
#
# plt.plot(x_axis, arr_loss_validation, 'b-', label='validation set')
# plt.plot(x_axis, arr_loss_validation, 'b.')
# plt.plot(x_axis, arr_loss_training, 'r-', label='training set')
# plt.plot(x_axis, arr_loss_training, 'r.')
# plt.legend()
# plt.title('loss graph (proposed)')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_training, 'm-', label='SE of gain')
# plt.plot(x_axis, arr_error_gain_training, 'm.')
# plt.plot(x_axis, arr_error_cov_training, 'c-', label='MSE of cov')
# plt.plot(x_axis, arr_error_cov_training, 'c.')
# plt.plot(x_axis, arr_error_AE_training, 'k-', label='MSE of AE')
# plt.plot(x_axis, arr_error_AE_training, 'k.')
# plt.legend()
# plt.title('Error graph of training set')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_validation, 'm-', label='SE of gain')
# plt.plot(x_axis, arr_error_gain_validation, 'm.')
# plt.plot(x_axis, arr_error_cov_validation, 'c-', label='MSE of cov')
# plt.plot(x_axis, arr_error_cov_validation, 'c.')
# plt.plot(x_axis, arr_error_AE_validation, 'k-', label='MSE of AE')
# plt.plot(x_axis, arr_error_AE_validation, 'k.')
# plt.legend()
# plt.title('Error graph of validation set')
# plt.show()
#
# plt.plot(x_axis, arr_loss_validation_encdec, 'b-', label='validation set')
# plt.plot(x_axis, arr_loss_validation_encdec, 'b.')
# plt.legend()
# plt.title('loss graph (validation set, proposed_encdec)')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_validation_encdec, 'm-', label='SE of gain')
# plt.plot(x_axis, arr_error_gain_validation_encdec, 'm.')
# plt.plot(x_axis, arr_error_cov_validation_encdec, 'c-', label='MSE of cov')
# plt.plot(x_axis, arr_error_cov_validation_encdec, 'c.')
# plt.legend()
# plt.title('Error graph of validation set')
# plt.show()

##################################################
# -Rui, save channel gain maps and RMSE of normalized covariance maps
with open(resultpath + "img_original_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_original_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_predicted_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_predicted_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_predicted_CSI_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_predicted_CSI_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_predicted_Both_gain_oo_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_predicted_Both_gain_oo.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_predicted_wo_AE_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_predicted_wo_AE_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov2_dB_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov2_dB.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov2_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov2.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_CSI_cov2_dB_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_CSI_cov2_dB.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_Both_cov2_dB_oo_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_Both_cov2_dB_oo.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_Both_cov2_oo_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_Both_cov2_oo.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_wo_AE_cov2_dB_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(img_RMSE_wo_AE_cov2_dB.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "NMSE_predicted_Both_gain_oo_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump((10 * (sum_SE_Both_gain_oo / sum_SO_Both_gain_oo).log10()).detach().cpu().numpy(), f)
f.close()
with open(resultpath + "loc_NMSE_predicted_Both_gain_oo_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(list_NMSE_Both_gain_oo.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "NMSE_predicted_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump((10 * (sum_SE_gain / sum_SO_gain).log10()).detach().cpu().numpy(), f)
f.close()
with open(resultpath + "loc_NMSE_predicted_gain_BS(08)_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(list_NMSE_gain.detach().cpu().numpy(), f)
f.close()

save_interpolated_map = model_save.saved_interpolated_map.detach().cpu().numpy()
save_input_map = model_save.saved_input_map.detach().cpu().numpy()

with open(resultpath + "input_map_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(save_input_map, f)
f.close()
with open(resultpath + "interpolated_map_O1_3p5(360x360).pkl", "wb") as f:
    pickle.dump(save_interpolated_map, f)
f.close()

torch.save(model_wo_AE.state_dict(), resultpath + "model_wo_AE")
torch.save(model_outer.state_dict(), resultpath + "model_outer")
torch.save(model_AE.state_dict(), resultpath + "model_AE")

print('NMSE of gain in dB scale (No spatial interpolation): ' + str(
    10 * (sum_SE_wo_AE_gain / sum_SO_wo_AE_gain).log10().item()))
print('Median of NSE of gain in dB scale (No spatial interpolation): ' + str(10 * math.log10(median_NMSE_wo_AE_gain)))
print('Average RMSE of normalized covariance in dB scale (No spatial interpolation): ' + str(
    20 * (sum_MSE_wo_AE_cov / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10().item()))
print('\n')

print('NMSE of gain in dB scale (CSI-based M\'_2): ' + str(
    10 * (sum_SE_CSI_gain / sum_SO_CSI_gain).log10().item()))
print('Median of NSE of gain in dB scale (CSI-based M\'_2): ' + str(10 * math.log10(median_NMSE_CSI_gain)))
print('Average RMSE of normalized covariance in dB scale (CSI-based M\'_2): ' + str(
    20 * (sum_MSE_CSI_cov / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10().item()))
print('\n')

print('NMSE of gain in dB scale (Location-based M\'_1): avg = %.6f, min = %.6f, max = %.6f' %
      ((10 * (sum_SE_gain / sum_SO_gain).log10()).mean().item(),
       (10 * (sum_SE_gain / sum_SO_gain).log10()).min().item(),
       (10 * (sum_SE_gain / sum_SO_gain).log10()).max().item()
       ))
print('Median of NSE of gain in dB scale (Location-based M\'_1): avg = %.6f, min = %.6f, max = %.6f' %
      ((10 * median_NMSE_gain.log10()).mean().item(),
       (10 * median_NMSE_gain.log10()).min().item(),
       (10 * median_NMSE_gain.log10()).max().item()
       ))
print('Average RMSE of normalized covariance in dB scale (Location-based M\'_1): avg = %.6f, min = %.6f, max = %.6f' %
      ((20 * (sum_MSE_cov / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).mean().item(),
       (20 * (sum_MSE_cov / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).min().item(),
       (20 * (sum_MSE_cov / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).max().item()
       ))
print('\n')

# -Rui, GM's version, one embedding map for one embedding
print('oo case, multiple training splits')
print('NMSE of gain in dB scale (Both location & CSI M\'): avg = %.6f, min = %.6f, max = %.6f' %
      ((10 * (sum_SE_Both_gain_oo / sum_SO_Both_gain_oo).log10()).mean().item(),
       (10 * (sum_SE_Both_gain_oo / sum_SO_Both_gain_oo).log10()).min().item(),
       (10 * (sum_SE_Both_gain_oo / sum_SO_Both_gain_oo).log10()).max().item()
       ))
print('Median of NSE of gain in dB scale (Both location & CSI M\'): avg = %.6f, min = %.6f, max = %.6f' %
      ((10 * median_NMSE_Both_gain_oo.log10()).mean().item(),
       (10 * median_NMSE_Both_gain_oo.log10()).min().item(),
       (10 * median_NMSE_Both_gain_oo.log10()).max().item()
       ))
print(
    'Average RMSE of normalized covariance in dB scale (Both location & CSI M\'): avg = %.6f, min = %.6f, max = %.6f' %
    ((20 * (sum_MSE_Both_cov_oo / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).mean().item(),
     (20 * (sum_MSE_Both_cov_oo / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).min().item(),
     (20 * (sum_MSE_Both_cov_oo / (num_tot_row * num_each_row / sample_interval ** 2)).sqrt().log10()).max().item()
     ))
print('\n')
# # -Rui, one embedding map for all embedding
# print('oa case:')
# print('NMSE of gain in dB scale (Both location & CSI M\'): ' + str(
#     10 * (sum_SE_Both_gain_oa / sum_SO_Both_gain_oa).log10().item()))
# print('Median of NSE of gain in dB scale (Both location & CSI M\'): ' + str(10 * math.log10(median_NMSE_Both_gain_oa)))
# print('Average RMSE of normalized covariance in dB scale (Both location & CSI M\'): ' + str(
#     20 * (sum_MSE_Both_cov_oa / (num_tot_row * num_each_row / sample_interval**2)).sqrt().log10().item()))

print("time taken :", time.time() - start)
