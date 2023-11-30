import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
# datapath = "./dataset/"
# resultpath = "./result/P1O1_Rui/P1O1_3p0_7e3_seed20_80135_ebdnorm_done/"
# resultpath_gp = "./result/P1_figures_GPresults/O1/"
# ######################################################
# # -Rui, O1
# # BS 17
# max_gain_dB = -93.58468858252584
# min_gain_dB = -101.33427620322566

# min_cov = 0
# max_cov = 1.5

# min_cov_dB = -55
# max_cov_dB = 5

# # GP results
# with open(resultpath_gp + "img_predicted_gain.pkl", "rb") as f:
    # img_predicted_gain_gp = pickle.load(f)
# f.close()

# with open(resultpath_gp + "img_RMSE_cov2.pkl", "rb") as f:
    # img_RMSE_cov2_gp = pickle.load(f)
# f.close()

# with open(resultpath_gp + "img_RMSE_cov2_dB.pkl", "rb") as f:
    # img_RMSE_cov2_dB_gp = pickle.load(f)
# f.close()

# # proposed results
# with open(resultpath + "input_map_O1_3p5(360x360).pkl", "rb") as f:
    # saved_input_map = pickle.load(f)
# f.close()
# with open(resultpath + "interpolated_map_O1_3p5(360x360).pkl", "rb") as f:
    # saved_interpolated_map = pickle.load(f)
# f.close()

# with open(resultpath + "img_original_gain_O1_3p5(360x360).pkl", "rb") as f:
    # img_original_gain = pickle.load(f)
# f.close()
# with open(resultpath + "img_predicted_gain_O1_3p5(360x360).pkl", "rb") as f:
    # img_predicted_gain = pickle.load(f)
# f.close()
# with open(resultpath + "img_RMSE_cov_O1_3p5(360x360).pkl", "rb") as f:
    # img_RMSE_cov = pickle.load(f)
# f.close()
# with open(resultpath + "img_RMSE_cov_dB_O1_3p5(360x360).pkl", "rb") as f:
    # img_RMSE_cov_dB = pickle.load(f)
# f.close()
# with open(resultpath + "img_RMSE_cov2_O1_3p5(360x360).pkl", "rb") as f:
    # img_RMSE_cov2 = pickle.load(f)
# f.close()
# with open(resultpath + "img_RMSE_cov2_dB_O1_3p5(360x360).pkl", "rb") as f:
    # img_RMSE_cov2_dB = pickle.load(f)
# f.close()

# # ##################################################
# # GP
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_predicted_gain_gp, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2_gp, aspect="auto", origin='lower', vmin=min_cov, vmax=max_cov)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2_dB_gp, aspect="auto", origin='lower', vmin=min_cov_dB, vmax=max_cov_dB)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# # ##################################################
# # Proposed
# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_original_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_predicted_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# # fig = plt.figure()
# # ax = fig.add_subplot()
# # im = plt.imshow(img_RMSE_cov, aspect="auto", origin='lower')
# # ax.set_xlabel('x', fontsize=14)
# # ax.set_ylabel('y', fontsize=14)
# # fig.colorbar(im, ax=ax)
# # plt.show()
# #
# # fig = plt.figure()
# # ax = fig.add_subplot()
# # im = plt.imshow(img_RMSE_cov_dB, aspect="auto", origin='lower')
# # ax.set_xlabel('x', fontsize=14)
# # ax.set_ylabel('y', fontsize=14)
# # fig.colorbar(im, ax=ax)
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2, aspect="auto", origin='lower', vmin=min_cov, vmax=max_cov)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# im = plt.imshow(img_RMSE_cov2_dB, aspect="auto", origin='lower', vmin=min_cov_dB, vmax=max_cov_dB)
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# fig.colorbar(im, ax=ax)
# plt.show()

###############################################
# -Rui, O1B
datapath = "./dataset/"
resultpath = "./result/P1O1B_Rui/P1O1B_3p0_7e3_dc_15151/"
resultpath_gp = "./result/P1_figures_GPresults/O1B/"
# BS 3
max_gain_dB = -85.13724980910376
min_gain_dB = -175.50639749051302
num_each_row = 180
num_tot_row = 1620  # 1620

min_cov = 0
max_cov = 1.4

min_cov_dB = -40
max_cov_dB = 5

# GP results
with open(resultpath_gp + "img_predicted_gain.pkl", "rb") as f:
    img_predicted_gain_gp = pickle.load(f)
f.close()

with open(resultpath_gp + "img_RMSE_cov2.pkl", "rb") as f:
    img_RMSE_cov2_gp = pickle.load(f)
f.close()

with open(resultpath_gp + "img_RMSE_cov2_dB.pkl", "rb") as f:
    img_RMSE_cov2_dB_gp = pickle.load(f)
f.close()

# proposed results
with open(resultpath + "input_map_O1_3p5B(180x1620).pkl", "rb") as f:
    saved_input_map = pickle.load(f)
f.close()
with open(resultpath + "interpolated_map_O1_3p5B(180x1620).pkl", "rb") as f:
    saved_interpolated_map = pickle.load(f)
f.close()

with open(resultpath + "img_original_gain_O1_3p5B(180x1620).pkl", "rb") as f:
    img_original_gain = pickle.load(f)
f.close()
with open(resultpath + "img_predicted_gain_O1_3p5B(180x1620).pkl", "rb") as f:
    img_predicted_gain = pickle.load(f)
f.close()
with open(resultpath + "img_RMSE_cov_O1_3p5B(180x1620).pkl", "rb") as f:
    img_RMSE_cov = pickle.load(f)
f.close()
with open(resultpath + "img_RMSE_cov_dB_O1_3p5B(180x1620).pkl", "rb") as f:
    img_RMSE_cov_dB = pickle.load(f)
f.close()
with open(resultpath + "img_RMSE_cov2_O1_3p5B(180x1620).pkl", "rb") as f:
    img_RMSE_cov2 = pickle.load(f)
f.close()
with open(resultpath + "img_RMSE_cov2_dB_O1_3p5B(180x1620).pkl", "rb") as f:
    img_RMSE_cov2_dB = pickle.load(f)
f.close()

##################################################

loc_x_min = 242.423  # User Grid 1
loc_y_min = 297.171  # User Grid 1
val_width = 3

# Locations of one blockage and two reflectors
loc_blockage_x = [244.197, 244.197]
loc_blockage_y = [502.254, 502.254 - 24]  # the width of blockage is 24m
loc_reflector_x_first = [251.262, 243.015]
loc_reflector_y_first = [544.990, 544.990]
loc_reflector_x_second = [250.697, 242.450]
loc_reflector_y_second = [440.948, 440.948]

for i in range(2):
    loc_blockage_x[i] = int(np.around((loc_blockage_x[i] - loc_x_min) * 10) / 2)
    loc_blockage_y[i] = int(np.around((loc_blockage_y[i] - loc_y_min) * 10) / 2)

    loc_reflector_x_first[i] = int(np.around((loc_reflector_x_first[i] - loc_x_min) * 10) / 2)
    loc_reflector_y_first[i] = int(np.around((loc_reflector_y_first[i] - loc_y_min) * 10) / 2)

    loc_reflector_x_second[i] = int(np.around((loc_reflector_x_second[i] - loc_x_min) * 10) / 2)
    loc_reflector_y_second[i] = int(np.around((loc_reflector_y_second[i] - loc_y_min) * 10) / 2)

channel_obstacles = torch.zeros((1, num_tot_row, num_each_row))
for i in range(loc_blockage_y[0] - loc_blockage_y[1] + 1):
    channel_obstacles[0, loc_blockage_y[0] - i, loc_blockage_x[0]] = 1
for i in range(loc_reflector_x_first[0] - loc_reflector_x_first[1] + 1):
    channel_obstacles[0, loc_reflector_y_first[0], loc_reflector_x_first[0] - i] = 1
for i in range(loc_reflector_x_second[0] - loc_reflector_x_second[1] + 1):
    channel_obstacles[0, loc_reflector_y_second[0], loc_reflector_x_second[0] - i] = 1


# ##################################################
# GP
fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_predicted_gain_gp, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_RMSE_cov2_gp, aspect="auto", origin='lower', vmin=min_cov, vmax=max_cov)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_RMSE_cov2_dB_gp, aspect="auto", origin='lower', vmin=min_cov_dB, vmax=max_cov_dB)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

##################################################
# proposed
fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_original_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_predicted_gain, aspect="auto", origin='lower', vmin=min_gain_dB, vmax=max_gain_dB)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

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

fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_RMSE_cov2, aspect="auto", origin='lower', vmin=min_cov, vmax=max_cov)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
im = plt.imshow(img_RMSE_cov2_dB, aspect="auto", origin='lower', vmin=min_cov_dB, vmax=max_cov_dB)
plt.plot(loc_blockage_x, loc_blockage_y, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_first, loc_reflector_y_first, 'r-', linewidth=val_width)
plt.plot(loc_reflector_x_second, loc_reflector_y_second, 'r-', linewidth=val_width)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
fig.colorbar(im, ax=ax)
plt.show()

################################################