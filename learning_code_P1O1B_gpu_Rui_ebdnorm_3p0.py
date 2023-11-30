import numpy as np
import torch
import torch.nn as nn
import pickle
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import random
import math
import copy

# Seed
seed = 4
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

start = time.time()

cuda_unit = 0
cuda_us = "cuda:" + str(cuda_unit)
print("CUDA usage: " + str(torch.cuda.is_available()))
DEVICE = torch.device(cuda_us if torch.cuda.is_available() else 'cpu')

num_embedding = 10

num_BS_antenna_x = 2
num_BS_antenna_y = 2
num_BS_antenna_z = 2
num_BS_antenna = num_BS_antenna_x * num_BS_antenna_y * num_BS_antenna_z

num_for_complex = 2
num_input = int(((num_BS_antenna ** 2 - num_BS_antenna) / 2) * num_for_complex) + 1  # upper triangular matrix without diagonal elements

num_each_row = 180
num_tot_row = 1620  # 1620

measurement_rate = 0.03  # measurement rate is 3 % -Rui, training / (training + validation + test)
ebd_sample_rate = 0.5  # -Rui, sampled training embeddings / training embeddings
n_ebd_map = 1  # -Rui, number of embedding maps (deprecated)

num_sample_training = int(num_each_row * num_tot_row * measurement_rate)
num_sample_validation = 1000
num_sample_ebd = int(num_sample_training * ebd_sample_rate)  # -Rui,number of sampled embeddings in each map

num_BSs = 1
# WFC = 1  # weight factor for covariance e.g., 1:x:1 for loss of gain, normalized covariance, and AE
# WFG = 1  # weight factor for gain
# WFA = 1  # weight factor for AE

WF_gain = 15
WF_cov = 15
WF_AE = 1

# embedding normalization stability adding value
eps_ebd = 1e-10

# drop out rate
dp_rate = 1e-20

max_epoch = 60000  # 7000
max_epoch_pretraining = 10000  # 7000
# min_epoch = 100  # 100
min_epoch_proposed = 2000  # 500
# min_epoch_encdec = 500  # 500

# size_window = 10000
# size_window_encdec = 10000

# -Rui, path
datapath = "./dataset/"
resultpath = "./result/P1O1B_Rui/P1O1B_seed20_3p0_15_15_1/"

# data loading (anchor BS)
with open(datapath + "dataset_BS(03)_O1_3p5B(180x1620)", "rb") as f_dataset:
    dataset = pickle.load(f_dataset)
f_dataset.close()

# location loading
with open(datapath + "dataset_loc_O1_3p5B(180x1620)", "rb") as f_dataset_loc:
    dataset_loc = pickle.load(f_dataset_loc)
f_dataset_loc.close()

# BS 3
max_gain_dB = -85.13724980910376
min_gain_dB = -175.50639749051302

##################################################

# sampling - generation
rnd_list_training = []
rnd_list_training_nonzero = []  # where gain is 0 for target BS
rnd_list_validation = []
num_nonzero_training = 0
num_nonzero_validation = 0

loop_true = 1
cnt_loop = 0
while loop_true == 1:
    rnd = random.randint(0, num_each_row * num_tot_row - 1)
    while rnd in rnd_list_training:
        rnd = random.randint(0, num_each_row * num_tot_row - 1)

    if dataset[rnd][0] != -2:
        rnd_list_training_nonzero.append(rnd)
        num_nonzero_training += 1

    rnd_list_training.append(rnd)
    cnt_loop += 1

    if cnt_loop == num_sample_training:
        loop_true = 0

for i in range(num_sample_validation):
    rnd = random.randint(0, num_each_row * num_tot_row - 1)
    while (rnd in rnd_list_training) or (rnd in rnd_list_validation):
        rnd = random.randint(0, num_each_row * num_tot_row - 1)
    rnd_list_validation.append(rnd)
    if dataset[rnd][0] != -2:
        num_nonzero_validation += 1

print('index file generation has been completed')

# with open(datapath + "list_inx_P1O1B_training.pkl", "wb") as f:
#     pickle.dump([rnd_list_training, rnd_list_training_nonzero, num_nonzero_training], f)
# f.close()
#
# with open(datapath + "list_inx_P1O1B_validation.pkl", "wb") as f:
#     pickle.dump([rnd_list_validation, num_nonzero_validation], f)
# f.close()
#
# print('index file saving has been completed')


# loading

# with open(datapath + "list_inx_P1O1B_training.pkl", "rb") as f:
#     list_training = pickle.load(f)
# f.close()
#
# rnd_list_training = list_training[0]
# rnd_list_training_nonzero = list_training[1]
# num_nonzero_training = list_training[2]
#
# with open(datapath + "list_inx_P1O1B_validation.pkl", "rb") as f:
#     list_validation = pickle.load(f)
# f.close()
#
# rnd_list_validation = list_validation[0]
# num_nonzero_validation = list_validation[1]

# for i in range(num_sample_training):
#     if dataset[rnd_list_training[i]][0] != -2:
#         rnd_list_training_nonzero.append(rnd_list_training[i])
#         num_nonzero_training += 1

# print('index file loading has been completed')


##################################################
#
# loc_x_min = 242.423  # User Grid 1
# loc_y_min = 297.171  # User Grid 1
#
# # Locations of one blockage and two reflectors
# loc_blockage_x = [244.197, 244.197]
# loc_blockage_y = [502.254, 502.254 - 24]  # the width of blockage is 24m
# loc_reflector_x_first = [251.262, 243.015]
# loc_reflector_y_first = [544.990, 544.990]
# loc_reflector_x_second = [250.697, 242.450]
# loc_reflector_y_second = [440.948, 440.948]
#
# for i in range(2):
#     loc_blockage_x[i] = int(np.around((loc_blockage_x[i] - loc_x_min) * 10) / 2)
#     loc_blockage_y[i] = int(np.around((loc_blockage_y[i] - loc_y_min) * 10) / 2)
#
#     loc_reflector_x_first[i] = int(np.around((loc_reflector_x_first[i] - loc_x_min) * 10) / 2)
#     loc_reflector_y_first[i] = int(np.around((loc_reflector_y_first[i] - loc_y_min) * 10) / 2)
#
#     loc_reflector_x_second[i] = int(np.around((loc_reflector_x_second[i] - loc_x_min) * 10) / 2)
#     loc_reflector_y_second[i] = int(np.around((loc_reflector_y_second[i] - loc_y_min) * 10) / 2)
#
# channel_obstacles = torch.zeros((1, num_tot_row, num_each_row))
# for i in range(loc_blockage_y[0] - loc_blockage_y[1] + 1):
#     channel_obstacles[0, loc_blockage_y[0] - i, loc_blockage_x[0]] = 1
# for i in range(loc_reflector_x_first[0] - loc_reflector_x_first[1] + 1):
#     channel_obstacles[0, loc_reflector_y_first[0], loc_reflector_x_first[0] - i] = 1
# for i in range(loc_reflector_x_second[0] - loc_reflector_x_second[1] + 1):
#     channel_obstacles[0, loc_reflector_y_second[0], loc_reflector_x_second[0] - i] = 1


##################################################


# class DatasetNonzero(Dataset):
#
#     def __init__(self):
#
#         for n in range(num_nonzero_training):
#             if n == 0:
#                 inx_random = rnd_list_training_nonzero[n]
#                 arr_cov = dataset[inx_random][:]
#                 arr_loc = dataset_loc[inx_random][:]
#             else:
#                 inx_random = rnd_list_training_nonzero[n]
#                 arr_cov = np.vstack((arr_cov, dataset[inx_random][:]))
#                 arr_loc = np.vstack((arr_loc, dataset_loc[inx_random][:]))
#
#         self.x = torch.from_numpy(arr_cov)
#         self.x = self.x.float()
#         self.y = torch.from_numpy(arr_loc)
#         self.n_samples = arr_cov.shape[0]
#
#     def __getitem__(self, index):
#         return self.x[index], self.y[index]
#
#     def __len__(self):
#         return self.n_samples


class DatasetAll(Dataset):

    def __init__(self):

        for n in range(num_sample_training):
            if n == 0:
                inx_random = rnd_list_training[n]
                arr_cov = copy.deepcopy(dataset[inx_random][:])
                if dataset[inx_random][0] == -2:
                    arr_cov[0] = -1
                arr_loc = dataset_loc[inx_random][:]
            else:
                inx_random = rnd_list_training[n]
                temp_arr_cov = copy.deepcopy(dataset[inx_random][:])
                if dataset[inx_random][0] == -2:
                    temp_arr_cov[0] = -1
                arr_cov = np.vstack((arr_cov, temp_arr_cov))
                arr_loc = np.vstack((arr_loc, dataset_loc[inx_random][:]))

        self.x = torch.from_numpy(arr_cov)
        self.x = self.x.float()
        self.y = torch.from_numpy(arr_loc)
        self.n_samples = arr_cov.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# dataset_sampled_nonzero = DatasetNonzero()
dataset_sampled_all = DatasetAll()

# data_loader_nonzero = DataLoader(dataset=dataset_sampled_nonzero, batch_size=num_nonzero_training, shuffle=False)
data_loader_all = DataLoader(dataset=dataset_sampled_all, batch_size=num_sample_training, shuffle=False)

##################################################
# - Rui,validation dataset
origin_valid_ts = torch.from_numpy(copy.deepcopy(dataset[rnd_list_validation, :])).float().to(DEVICE)
gain_check_inx_valid = origin_valid_ts[:, 0] == -2
origin_valid_ts[gain_check_inx_valid, 0] = -1


##################################################

##################################################


class Storage(nn.Module):
    def __init__(self):
        super().__init__()

        self.saved_interpolated_map = torch.zeros((num_embedding, num_tot_row, num_each_row))
        self.saved_input_map = torch.zeros((num_embedding + 1, num_tot_row, num_each_row))

        # self.saved_pretrained_encoder = nn.Sequential(
        #     nn.Linear(num_input * num_BSs, 128),
        #     nn.PReLU(),
        #     nn.Linear(128, 64),
        #     nn.PReLU(),
        #     nn.Linear(64, 32),
        #     nn.PReLU(),
        #     nn.Linear(32, 10),
        #     nn.Tanh()
        # )
        #
        # self.saved_pretrained_decoder = nn.Sequential(
        #     nn.Linear(10, 32),
        #     nn.PReLU(),
        #     nn.Linear(32, 64),
        #     nn.PReLU(),
        #     nn.Linear(64, 128),
        #     nn.PReLU(),
        #     nn.Linear(128, num_input),
        #     nn.Tanh()
        # )

        self.saved_converged_encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.saved_converged_decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, num_input),
            nn.Tanh()
        )

        self.saved_converged_AE_encoder = nn.Sequential(
            nn.Conv2d(num_embedding + 1, 64, 3, stride=1, padding=3),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.PReLU()
        )

        self.saved_converged_AE_decoder = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(64, num_embedding, 3, stride=1, padding=1),
            nn.Tanh()
        )


class PretrainedEncDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input * num_BSs, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, num_input),
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
            nn.Linear(num_input * num_BSs, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, num_embedding),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_embedding, 64),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, num_input),
            nn.Tanh()
        )

        self.interpolated_map = torch.zeros((num_embedding, num_tot_row, num_each_row))
        self.input_map = torch.zeros((num_embedding + 1, num_tot_row, num_each_row))
        self.var_init = 0

    def forward(self, x, y, inx_ebd, DEVICE):
        input_emb_img = torch.zeros((num_embedding + 1, num_tot_row, num_each_row)).to(DEVICE)

        encoded = self.encoder(x)
        # normalized embeddings
        encoded = encoded / (encoded.norm(dim=1, keepdim=True) + eps_ebd)
        # for i in range(x.size(dim=0)):
        #     input_emb_img[0:-1, y[i, 1], y[i, 0]] = encoded[i, :]
        #     input_emb_img[num_embedding, y[i, 1], y[i, 0]] = 1

        # -Rui, construct map using ONLY part of training embeddings
        input_emb_img[0:-1, y[inx_ebd, 1].long(), y[inx_ebd, 0].long()] = encoded[inx_ebd, :].transpose(1, 0)
        input_emb_img[num_embedding, y[inx_ebd, 1].long(), y[inx_ebd, 0].long()] = 1

        self.input_map = input_emb_img
        input_emb_img = input_emb_img[None, :]
        # -Rui, interpolate embeddings of all locations
        interpolated = model_AE(input_emb_img)
        interpolated = interpolated[0, :, 0:num_tot_row, 0:num_each_row]
        self.interpolated_map = interpolated
        # decoded_in = torch.zeros((x.size(dim=0), num_embedding))
        # for i in range(x.size(dim=0)):
        #     decoded_in[i, :] = interpolated[:, y[i, 1], y[i, 0]]
        # -Rui, pick all interpolated training embeddings
        decoded_in = interpolated[:, y[:, 1].long(), y[:, 0].long()].transpose(1, 0)
        # normalized embeddings
        decoded_in = decoded_in / (decoded_in.norm(dim=1, keepdim=True) + eps_ebd)

        decoded = self.decoder(decoded_in)

        return encoded, decoded_in, decoded


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_embedding + 1, 64, 3, stride=1, padding=3),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dp_rate),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(dp_rate),
            nn.ConvTranspose2d(64, num_embedding, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


##################################################

criterion = nn.MSELoss()

model_save = Storage().to(DEVICE)
model_pretrained = PretrainedEncDec().to(DEVICE)
model_outer = OuterEncDec().to(DEVICE)
model_AE = Autoencoder().to(DEVICE)

optimizer_pt = torch.optim.Adam(
    list(model_pretrained.encoder.parameters()) + list(model_pretrained.decoder.parameters()), lr=1e-3)
optimizer_AE = torch.optim.Adam(list(model_AE.encoder.parameters()) + list(model_AE.decoder.parameters()), lr=1e-4)
optimizer_outer = torch.optim.Adam(list(model_outer.encoder.parameters()) + list(model_outer.decoder.parameters()),
                                   lr=1e-4)

##################################################

# pretraining

minimum_loss = 10000
check_overfitting = 0

arr_loss_training_pt = []
arr_error_gain_training_pt = []
arr_error_cov_training_pt = []
arr_loss_validation_pt = []
arr_error_gain_validation_pt = []
arr_error_cov_validation_pt = []
for epoch in range(max_epoch_pretraining):

    for (inputs, labels) in data_loader_all:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_pt.zero_grad()
        recon = model_pretrained(inputs, labels)

    loss_gain = criterion(recon[:, 0], inputs[:, 0])
    loss_cov = criterion(recon[:, 1:num_input], inputs[:, 1:num_input])

    # -Rui, dk why there are two data loaders. Reflectors and blockage? Just use one here
    # for (inputs_nz, labels_nz) in data_loader_nonzero:
    #     recon_nz = model_pretrained(inputs_nz, labels_nz)
    # loss_cov = 2 * criterion(recon_nz[:, 1:num_input], inputs_nz[:, 1:num_input]) / num_sample_training

    # if epoch == 0:
    #     WF_gain = (loss_cov.item() / (loss_gain.item() + loss_cov.item()))
    #     WF_cov = (loss_gain.item() / (loss_gain.item() + loss_cov.item()))
    loss = WF_gain * loss_gain + WF_cov * loss_cov

    loss.backward()
    optimizer_pt.step()

    arr_loss_training_pt.append(loss.item())
    arr_error_gain_training_pt.append(loss_gain.item())
    arr_error_cov_training_pt.append(loss_cov.item())

    # for m in range(num_sample_validation):
    #     test_inx = rnd_list_validation[m]
    #
    #     origin = copy.deepcopy(dataset[test_inx, :])
    #     if dataset[test_inx][0] == -2:
    #         origin[0] = -1
    #     temp_input = torch.from_numpy(origin)
    #     predicted = model_pretrained.decoder(model_pretrained.encoder(temp_input.float()))
    #     predicted = predicted.detach().numpy()
    #     temp_loss_validation_gain = (origin[0] - predicted[0]) ** 2
    #     temp_loss_validation_cov = 0
    #     for cnt in range(num_input - 1):
    #         temp_loss_validation_cov += 2 * (origin[cnt + 1] - predicted[cnt + 1]) ** 2
    #     loss_validation_gain += temp_loss_validation_gain
    #     loss_validation_cov += temp_loss_validation_cov
    #
    # loss_validation_gain = loss_validation_gain / num_sample_validation
    # loss_validation_cov = loss_validation_cov / num_sample_validation

    # -Rui, no double loops
    model_pretrained.eval()
    with torch.no_grad():
        encoded_val = model_pretrained.encoder(origin_valid_ts)
        # normalized embeddings
        encoded_val = encoded_val / (encoded_val.norm(dim=1, keepdim=True) + eps_ebd)
        predicted = model_pretrained.decoder(encoded_val)

        loss_validation_gain = ((origin_valid_ts[:, 0] - predicted[:, 0]) ** 2).mean()
        loss_validation_cov = ((origin_valid_ts[:, 1:] - predicted[:, 1:]) ** 2).mean()

        loss_validation = WF_gain * loss_validation_gain + WF_cov * loss_validation_cov

        arr_loss_validation_pt.append(loss_validation.item())
        arr_error_gain_validation_pt.append(loss_validation_gain.item())
        arr_error_cov_validation_pt.append(loss_validation_cov.item())

    model_pretrained.train()

    print(
        '>>> st: ae_out | epoch = %.0f/%.0f | loss_train: gain = %.4f cov = %.4f, comb = %.4f | loss_val: gain = %.4f, cov = %.4f, comb = %.4f ' %
        (epoch,
         max_epoch_pretraining,
         arr_error_gain_training_pt[epoch],
         arr_error_cov_training_pt[epoch],
         arr_loss_training_pt[epoch],
         arr_error_gain_validation_pt[epoch],
         arr_error_cov_validation_pt[epoch],
         arr_loss_validation_pt[epoch]))

    # -Rui, save the model with minimum validation loss
    if minimum_loss >= loss_validation:
        minimum_loss = loss_validation
        model_save.saved_converged_encoder.load_state_dict(copy.deepcopy(model_pretrained.encoder.state_dict()))
        model_save.saved_converged_decoder.load_state_dict(copy.deepcopy(model_pretrained.decoder.state_dict()))
        epoch_pt_val_es = epoch

    if epoch == max_epoch_pretraining - 1:
        model_pretrained.encoder.load_state_dict(copy.deepcopy(model_save.saved_converged_encoder.state_dict()))
        model_pretrained.decoder.load_state_dict(copy.deepcopy(model_save.saved_converged_decoder.state_dict()))
        check_overfitting = 1
        print(f'convergence Epoch:{epoch + 1}')
        print(f'early stop Epoch:{epoch_pt_val_es + 1}')
        print(f'ratio:{loss_validation.item() / minimum_loss.item()}')

    if check_overfitting == 1:
        max_epoch_pretraining = epoch + 1
        break

##################################################

# proposed

minimum_loss = 10000
minimum_loss_encdec = 10000
check_overfitting = 0
check_overfitting_encdec = 0

arr_loss_validation = []
arr_loss_training = []
arr_error_gain_validation = []
arr_error_gain_training = []
arr_error_cov_validation = []
arr_error_cov_training = []
arr_error_AE_validation = []
arr_error_AE_training = []

arr_loss_validation_encdec = []
arr_error_gain_validation_encdec = []
arr_error_cov_validation_encdec = []

# arr_minimum_losses = torch.ones(size_window).to(DEVICE) * 10000
# arr_minimum_losses_encdec = torch.ones(size_window_encdec).to(DEVICE) * 10000

for epoch in range(max_epoch):

    if model_outer.var_init == 0:
        model_outer.encoder.load_state_dict(copy.deepcopy(model_pretrained.encoder.state_dict()))
        model_outer.decoder.load_state_dict(copy.deepcopy(model_pretrained.decoder.state_dict()))
        model_outer.var_init = 1

    for (inputs, labels) in data_loader_all:
        # -Rui, sample ONLY part of training embeddings
        inx_ebd = torch.randperm(inputs.shape[0]).to(DEVICE)[0:num_sample_ebd]
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_outer.zero_grad()
        optimizer_AE.zero_grad()
        encoded_in, decoded_in, recon = model_outer(inputs, labels, inx_ebd, DEVICE)

    loss_gain = criterion(recon[:, 0], inputs[:, 0])
    loss_cov = criterion(recon[:, 1:num_input], inputs[:, 1:num_input])
    # -Rui, dk why there are two data loaders. Reflectors and blockage? Just use one here
    # for (inputs_nz, labels_nz) in data_loader_nonzero:
    #     recon_nz = model_outer(inputs_nz, labels_nz)
    # loss_cov = 2 * criterion(recon_nz[:, 1:num_input], inputs_nz[:, 1:num_input]) / num_sample_training

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

    # -Rui, no loop
    # AE_input = model_outer.input_map[0:num_embedding, dataset_loc[rnd_list_training, 1],
    #            dataset_loc[rnd_list_training, 0]].transpose(1, 0)

    loss_AE = criterion(encoded_in, decoded_in)

    # weight factors
    # if epoch == 0:
    #     temp_WF1 = loss_gain.item() * loss_cov.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF_gain = loss_cov.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF_cov = loss_gain.item() / (loss_gain.item() + loss_cov.item())
    #     temp_WF2 = loss_AE.item() / (temp_WF1 + loss_AE.item())
    #     WF_gain = temp_WF2 * temp_WF_gain * WFG
    #     WF_cov = temp_WF2 * temp_WF_cov * WFC
    #     WF_AE = temp_WF1 / (temp_WF1 + loss_AE.item()) * WFA

    loss = WF_gain * loss_gain + WF_cov * loss_cov + WF_AE * loss_AE

    if check_overfitting_encdec == 0:
        loss.backward()
        optimizer_outer.step()
        optimizer_AE.step()
        training_stage = 'ae_all'
    else:
        loss.backward()
        optimizer_AE.step()
        training_stage = 'ae_in'

    arr_loss_training.append(loss.item())
    arr_error_gain_training.append(loss_gain.item())
    arr_error_cov_training.append(loss_cov.item())
    arr_error_AE_training.append(loss_AE.item())

    # print(f'Epoch:{epoch + 1}, Loss:{loss.item():.6f}')
    # print("time taken :", time.time() - start)

    loss_validation = 0
    loss_validation_gain = 0
    loss_validation_cov = 0
    loss_validation_AE = 0
    loss_validation_encdec = 0
    loss_validation_encdec_gain = 0
    loss_validation_encdec_cov = 0
    # for m in range(num_sample_validation):
    #     test_inx = rnd_list_validation[m]
    #
    #     origin = copy.deepcopy(dataset[test_inx, :])
    #     if dataset[test_inx][0] == -2:
    #         origin[0] = -1
    #     predicted = model_outer.decoder(
    #         model_outer.interpolated_map[:, dataset_loc[test_inx, 1], dataset_loc[test_inx, 0]])
    #     predicted = predicted.detach().numpy()
    #     temp_loss_validation_gain = (origin[0] - predicted[0]) ** 2
    #     temp_loss_validation_cov = 0
    #     for cnt in range(num_input - 1):
    #         temp_loss_validation_cov += 2 * (origin[cnt + 1] - predicted[cnt + 1]) ** 2
    #
    #     loss_validation_gain += temp_loss_validation_gain
    #     loss_validation_cov += temp_loss_validation_cov
    #
    #     if m == 0:
    #         pre_input = dataset[test_inx, :]
    #         validation_input = torch.from_numpy(pre_input)
    #         AE_input = model_outer.encoder(validation_input.float())
    #         AE_interpolated = model_outer.interpolated_map[:, dataset_loc[test_inx, 1], dataset_loc[test_inx, 0]]
    #     else:
    #         pre_input = dataset[test_inx, :]
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
    #         temp_loss_validation_encdec_cov += 2 * (origin[cnt + 1] - predicted_encdec[cnt + 1]) ** 2
    #
    #     loss_validation_encdec_gain += temp_loss_validation_encdec_gain
    #     loss_validation_encdec_cov += temp_loss_validation_encdec_cov
    #
    # loss_validation_gain = loss_validation_gain / num_sample_validation
    # loss_validation_cov = loss_validation_cov / num_sample_validation
    # loss_validation_AE = criterion(AE_input, AE_interpolated) / num_sample_validation
    #
    # loss_validation_encdec_gain = loss_validation_encdec_gain / num_sample_validation
    # loss_validation_encdec_cov = loss_validation_encdec_cov / num_sample_validation

    # -Rui, no double loop
    model_outer.eval()
    model_AE.eval()
    with torch.no_grad():
        AE_input = model_outer.encoder(origin_valid_ts)
        # normalized embeddings
        AE_input = AE_input / (AE_input.norm(dim=1, keepdim=True) + eps_ebd)
        AE_interpolated = model_outer.interpolated_map[:, dataset_loc[rnd_list_validation, 1],
                          dataset_loc[rnd_list_validation, 0]].transpose(1, 0).clone()
        # normalized embeddings
        AE_interpolated = AE_interpolated / (AE_interpolated.norm(dim=1, keepdim=True) + eps_ebd)
        predicted = model_outer.decoder(AE_interpolated)
        loss_validation_gain = ((origin_valid_ts[:, 0] - predicted[:, 0]) ** 2).mean()
        loss_validation_cov = ((origin_valid_ts[:, 1:] - predicted[:, 1:]) ** 2).mean()

        predicted_encdec = model_outer.decoder(AE_input)
        loss_validation_encdec_gain = ((origin_valid_ts[:, 0] - predicted_encdec[:, 0]) ** 2).mean()
        loss_validation_encdec_cov = ((origin_valid_ts[:, 1:] - predicted_encdec[:, 1:]) ** 2).mean()

        loss_validation_AE = criterion(AE_input, AE_interpolated)
        #
        loss_validation = WF_gain * loss_validation_gain + WF_cov * loss_validation_cov + WF_AE * loss_validation_AE
        loss_validation_encdec = WF_gain * loss_validation_encdec_gain + WF_cov * loss_validation_encdec_cov

        arr_loss_validation.append(loss_validation.item())
        arr_error_gain_validation.append(loss_validation_gain.item())
        arr_error_cov_validation.append(loss_validation_cov.item())
        arr_error_AE_validation.append(loss_validation_AE.item())

        arr_loss_validation_encdec.append(loss_validation_encdec.item())
        arr_error_gain_validation_encdec.append(loss_validation_encdec_gain.item())
        arr_error_cov_validation_encdec.append(loss_validation_encdec_cov.item())

    model_outer.train()
    model_AE.train()

    print(
        '>>> st: %s | epoch = %.0f/%.0f | loss_train: gain = %.4f, cov = %.4f, ae = %.4f, comb = %.4f '
        '| loss_val_aa: gain = %.4f, cov = %.4f, ae = %.4f, comb = %.4f '
        '| loss_val_ao: gain = %.4f, cov = %.4f, comb = %.4f' %
        (training_stage,
         epoch,
         max_epoch,
         arr_error_gain_training[epoch],
         arr_error_cov_training[epoch],
         arr_error_AE_training[epoch],
         arr_loss_training[epoch],
         arr_error_gain_validation[epoch],
         arr_error_cov_validation[epoch],
         arr_error_AE_validation[epoch],
         arr_loss_validation[epoch],
         arr_error_gain_validation_encdec[epoch],
         arr_error_cov_validation_encdec[epoch],
         arr_loss_validation_encdec[epoch]))

    # if epoch > min_epoch:
    #     # for i in range(size_window - 1):
    #     #     arr_minimum_losses[i] = arr_minimum_losses[i + 1]
    #     # arr_minimum_losses[size_window - 1] = loss_validation
    #     # -Rui, no loop
    #     arr_minimum_losses[0:-1] = arr_minimum_losses[1:].clone()
    #     arr_minimum_losses[size_window - 1] = loss_validation

    # -Rui, save the model with minimum validation loss
    if epoch > min_epoch_proposed:
        if minimum_loss >= loss_validation:
            minimum_loss = loss_validation
            model_save.saved_converged_encoder.load_state_dict(copy.deepcopy(model_outer.encoder.state_dict()))
            model_save.saved_converged_decoder.load_state_dict(copy.deepcopy(model_outer.decoder.state_dict()))
            model_save.saved_converged_AE_encoder.load_state_dict(copy.deepcopy(model_AE.encoder.state_dict()))
            model_save.saved_converged_AE_decoder.load_state_dict(copy.deepcopy(model_AE.decoder.state_dict()))
            model_save.saved_input_map = model_outer.input_map.clone()
            model_save.saved_interpolated_map = model_outer.interpolated_map.clone()
            epoch_val_es = epoch

        # if minimum_loss < arr_minimum_losses.min() or epoch == max_epoch - 1:
        if epoch == max_epoch - 1:
            check_overfitting = 1
            model_AE.encoder.load_state_dict(model_save.saved_converged_AE_encoder.state_dict())
            model_AE.decoder.load_state_dict(model_save.saved_converged_AE_decoder.state_dict())
            model_outer.encoder.load_state_dict(model_save.saved_converged_encoder.state_dict())
            model_outer.decoder.load_state_dict(model_save.saved_converged_decoder.state_dict())
            model_outer.input_map = model_save.saved_input_map.clone()
            model_outer.interpolated_map = model_save.saved_interpolated_map.clone()
            print(f'convergence Epoch:{epoch + 1}')
            print(f'early stop Epoch:{epoch_val_es + 1}')
            print(f'ratio:{loss_validation.item() / minimum_loss.item()}')

    # if check_overfitting_encdec == 0:
    #     if epoch > min_epoch:
    #         # for i in range(size_window - 1):
    #         #     arr_minimum_losses_encdec[i] = arr_minimum_losses_encdec[i + 1]
    #         # arr_minimum_losses_encdec[size_window - 1] = loss_validation_encdec
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
    #             model_outer.encoder.load_state_dict(model_save.saved_converged_encoder.state_dict())
    #             model_outer.decoder.load_state_dict(model_save.saved_converged_decoder.state_dict())
    #             print("epoch for encdec convergence: ##############################", str(epoch + 1))
    #             print(str(loss_validation_encdec.item() / minimum_loss_encdec.item()))
    #             check_overfitting_encdec = 1

    if check_overfitting == 1:
        max_epoch = epoch + 1
        break

##################################################

# img_original_gain = np.zeros((num_tot_row, num_each_row, 1))
# img_predicted_gain = np.zeros((num_tot_row, num_each_row, 1))
# img_RMSE_cov = np.zeros((num_tot_row, num_each_row, 1))
# img_RMSE_cov2 = np.zeros((num_tot_row, num_each_row, 1))
# img_RMSE_cov_dB = np.zeros((num_tot_row, num_each_row, 1))
# img_RMSE_cov2_dB = np.zeros((num_tot_row, num_each_row, 1))

sum_SE_gain = 0
sum_SO_gain = 0  # SO : squared original
sum_MSE_cov = 0
sum_MSE_cov2 = 0
sum_SO_cov = 0
median_NMSE_gain = 0

# num_nonzero = 0
#
# for m in range(int(num_tot_row)):
#     for n in range(int(num_each_row)):
#         inx_cnt = m * num_each_row + n
#
#         original = copy.deepcopy(dataset[inx_cnt, :])
#         if dataset[test_inx][0] == -2:
#             origin[0] = -1
#         temp_map = torch.tensor(model_save.saved_interpolated_map)
#         predicted = model_outer.decoder(temp_map[:, dataset_loc[inx_cnt, 1], dataset_loc[inx_cnt, 0]])
#
#         if inx_cnt == 0:
#             predicted_dataset = predicted.detach().numpy()
#         else:
#             predicted_dataset = np.vstack((predicted_dataset, predicted.detach().numpy()))
#
#         # convert gain values in -1 ~ +1 range to original values in dB scale
#         original_gain = (max_gain_dB - min_gain_dB) * (original[0] + 1) / 2 + min_gain_dB
#         original_gain_linear = 10 ** (original_gain / 10)
#
#         predicted_gain = (max_gain_dB - min_gain_dB) * (predicted[0].detach().numpy() + 1) / 2 + min_gain_dB
#         predicted_gain_linear = 10 ** (predicted_gain / 10)
#         ####################
#
#         img_original_gain[m, n, 0] = original_gain
#         img_predicted_gain[m, n, 0] = predicted_gain
#
#         SE_gain = (original_gain_linear - predicted_gain_linear) ** 2
#         SO_gain = original_gain_linear ** 2
#
#         sum_SE_gain += SE_gain
#         sum_SO_gain += SO_gain
#
#         if dataset[inx_cnt][0] != -2:
#
#             MSE_cov = (2 * criterion(predicted[1 + num_BS_antenna:num_input],
#                                      torch.tensor(original[1 + num_BS_antenna:num_input]))).detach().numpy()
#             MSE_cov2 = MSE_cov / num_BS_antenna ** 2
#
#             img_RMSE_cov[m, n, 0] = np.sqrt(MSE_cov)
#             img_RMSE_cov_dB[m, n, 0] = 20 * math.log10(img_RMSE_cov[m, n, 0])
#             img_RMSE_cov2[m, n, 0] = np.sqrt(MSE_cov2)
#             img_RMSE_cov2_dB[m, n, 0] = 20 * math.log10(img_RMSE_cov2[m, n, 0])
#
#             sum_MSE_cov += MSE_cov
#             sum_MSE_cov2 += MSE_cov2
#             SO_cov = np.sum(original[1:num_input] ** 2) + num_BS_antenna
#             sum_SO_cov += SO_cov
#
#             num_nonzero += 1
#         else:
#             img_RMSE_cov[m, n, 0] = np.nan
#             img_RMSE_cov_dB[m, n, 0] = np.nan
#             img_RMSE_cov2[m, n, 0] = np.nan
#             img_RMSE_cov2_dB[m, n, 0] = np.nan
#
#         list_NMSE_gain[inx_cnt] = SE_gain / SO_gain
#
# list_NMSE_gain = list_NMSE_gain.tolist()
# list_NMSE_gain.sort()
# median_NMSE_gain = list_NMSE_gain[int((num_tot_row * num_each_row) / 2)]

#  -Rui, no double loops
inx_cnt_all = []

for m in range(int(num_tot_row)):
    for n in range(int(num_each_row)):
        inx_cnt_all.append(m * num_each_row + n)

model_pretrained.eval()
model_AE.eval()
model_save.eval()
model_outer.eval()
with torch.no_grad():
    original = torch.from_numpy(copy.deepcopy(dataset[inx_cnt_all, :])).float().to(DEVICE)
    gain_check_inx_test = original[:, 0] == -2
    original[gain_check_inx_test, 0] = -1
    temp_map = model_save.saved_interpolated_map.clone()
    intpla_ebd = temp_map[:, dataset_loc[inx_cnt_all, 1], dataset_loc[inx_cnt_all, 0]].transpose(1, 0).clone()
    # normalized embedding
    intpla_ebd = intpla_ebd / (intpla_ebd.norm(dim=1, keepdim=True) + eps_ebd)
    predicted = model_outer.decoder(intpla_ebd)
    # convert gain values in -1 ~ +1 range to original values in dB scale
    original_gain = (max_gain_dB - min_gain_dB) * (original[:, 0] + 1) / 2 + min_gain_dB
    original_gain_linear = 10 ** (original_gain / 10)

    predicted_gain = (max_gain_dB - min_gain_dB) * (predicted[:, 0] + 1) / 2 + min_gain_dB
    predicted_gain_linear = 10 ** (predicted_gain / 10)
    ####################

    img_original_gain = original_gain.reshape(num_tot_row, num_each_row)
    img_predicted_gain = predicted_gain.reshape(num_tot_row, num_each_row)

    SE_gain = (original_gain_linear - predicted_gain_linear) ** 2
    SO_gain = original_gain_linear ** 2

    sum_SE_gain = SE_gain.sum()
    sum_SO_gain = SO_gain.sum()

    MSE_cov = 2 * ((predicted[:, 1:num_input] - original[:, 1:num_input]) ** 2).sum(axis=1)
    MSE_cov2 = MSE_cov / num_BS_antenna ** 2

    img_RMSE_cov = MSE_cov.sqrt().reshape(num_tot_row, num_each_row)
    img_RMSE_cov_dB = 20 * img_RMSE_cov.log10()
    img_RMSE_cov2 = MSE_cov2.sqrt().reshape(num_tot_row, num_each_row)
    img_RMSE_cov2_dB = 20 * img_RMSE_cov2.log10()

    # sum_MSE_cov = MSE_cov.sum()
    sum_MSE_cov2 = MSE_cov2.sum()
    # sum_SO_cov = (original[:, 1:num_input] ** 2).sum() + num_BS_antenna * original.shape[0]

    list_NMSE_gain = SE_gain / SO_gain

    median_NMSE_gain = list_NMSE_gain.median()

model_pretrained.train()
model_AE.train()
model_save.train()
model_outer.train()
##################################################
#
# x_axis = list(np.arange(1, max_epoch_pretraining + 1))
#
# plt.plot(x_axis, arr_loss_validation_pt, 'b.-', label='validation set')
# plt.plot(x_axis, arr_loss_training_pt, 'r.-', label='training set')
# plt.legend()
# plt.title('loss graph (pretraining)')
# plt.show()
#
# x_axis = list(np.arange(1, max_epoch + 1))
#
# plt.plot(x_axis, arr_loss_validation, 'b.-', label='validation set')
# plt.plot(x_axis, arr_loss_training, 'r.-', label='training set')
# plt.legend()
# plt.title('loss graph (proposed)')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_training, 'm.-', label='SE of gain')
# plt.plot(x_axis, arr_error_cov_training, 'c.-', label='MSE of cov')
# plt.plot(x_axis, arr_error_AE_training, 'k.-', label='MSE of AE')
# plt.legend()
# plt.title('Error graph of training set')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_validation, 'm.-', label='SE of gain')
# plt.plot(x_axis, arr_error_cov_validation, 'c.-', label='MSE of cov')
# plt.plot(x_axis, arr_error_AE_validation, 'k.-', label='MSE of AE')
# plt.legend()
# plt.title('Error graph of validation set')
# plt.show()
#
# plt.plot(x_axis, arr_loss_validation_encdec, 'b.-', label='validation set')
# plt.legend()
# plt.title('loss graph of enc/dec part')
# plt.show()
#
# plt.plot(x_axis, arr_error_gain_validation_encdec, 'm.-', label='SE of gain')
# plt.plot(x_axis, arr_error_cov_validation_encdec, 'c.-', label='MSE of cov')
# plt.legend()
# plt.title('Error graph of enc/dec part')
# plt.show()
#
# ##################################################
#
# val_width = 3
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_original_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_predicted_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov, aspect="auto", origin='lower')
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov_dB, aspect="auto", origin='lower')
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2, aspect="auto", origin='lower')
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2_dB, aspect="auto", origin='lower')
# plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
# plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# ##################################################

with open(resultpath + "predicted_dataset_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(predicted.detach().cpu().numpy(), f)
f.close()

with open(resultpath + "input_map_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(model_save.saved_input_map.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "interpolated_map_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(model_save.saved_interpolated_map.detach().cpu().numpy(), f)
f.close()

with open(resultpath + "img_original_gain_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_original_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_predicted_gain_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_predicted_gain.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov_dB_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov_dB.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov2_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov2.detach().cpu().numpy(), f)
f.close()
with open(resultpath + "img_RMSE_cov2_dB_O1_3p5B(180x1620).pkl", "wb") as f:
    pickle.dump(img_RMSE_cov2_dB.detach().cpu().numpy(), f)
f.close()

torch.save(model_outer.state_dict(), resultpath + "model_outer")
torch.save(model_AE.state_dict(), resultpath + "model_AE")

print('NMSE of gain in dB scale: ' + str((
    10 * (sum_SE_gain / sum_SO_gain).log10()).item()))
print('Median of NSE of gain in dB scale: ' + str((10 * median_NMSE_gain.log10()).item()))
print('Average RMSE of normalized covariance in dB scale: ' + str((
    20 * (sum_MSE_cov2 / (num_tot_row * num_each_row)).sqrt().log10()).item()))
# print('The number of nonzero points: ' + str(num_nonzero))

print("time taken :", time.time() - start)
