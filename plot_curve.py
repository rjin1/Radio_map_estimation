import matplotlib.pyplot as plt

# ########################################
#
# # O1
#
# O1_MNSE_gain_PCA = [-14.2773, -14.3527, -14.4159, -14.4941, -14.6493]
# O1_RMSE_cov_PCA = [-8.2005, -8.5415, -8.8875, -9.0301, -9.0895]
#
# O1_MedNSE_gain_proposed = [-39.6907, -41.3394, -41.3146, -41.9915, -44.7288]
# O1_RMSE_cov_proposed = [-14.6845, -15.4870, -17.2561, -17.7260, -18.7030]
#
# ########################################
#
# # O1B
#
# O1B_MNSE_gain_PCA = [-4.3270, -4.4067, -4.4802, -4.5231, -4.5983]
# O1B_RMSE_cov_PCA = [-6.8322, -7.3921, -7.7431, -8.0226, -8.2745]
#
# O1B_MedNSE_gain_proposed = [-17.9799, -20.2990, -20.9773, -21.8278, -19.5665]
# O1B_RMSE_cov_proposed = [-10.2588, -11.5968, -12.4958, -13.1343, -13.2848]
#
# ########################################
#
# x_axis_A = [1, 1.5, 2, 2.5, 3]
#
# plt.plot(x_axis_A, O1_MedNSE_gain_proposed, 'r^-', label='Proposed O1')
# plt.plot(x_axis_A, O1_MNSE_gain_PCA, 'b^--', label='PCA O1')
# plt.plot(x_axis_A, O1B_MedNSE_gain_proposed, 'ro-', label='Proposed O1B')
# plt.plot(x_axis_A, O1B_MNSE_gain_PCA, 'bo--', label='PCA O1B')
# plt.legend(fontsize=10)
# plt.legend(bbox_to_anchor=(1.0, 0.45))
# plt.xlabel('Sampling fraction (%)', fontsize=14)
# plt.ylabel('Median normalized error (dB)', fontsize=14)
# plt.xticks([1, 1.5, 2, 2.5, 3])
# plt.show()
#
# plt.plot(x_axis_A, O1_RMSE_cov_proposed, 'r^-', label='Proposed O1')
# plt.plot(x_axis_A, O1_RMSE_cov_PCA, 'b^--', label='PCA O1')
# plt.plot(x_axis_A, O1B_RMSE_cov_proposed, 'ro-', label='Proposed O1B')
# plt.plot(x_axis_A, O1B_RMSE_cov_PCA, 'bo--', label='PCA O1B')
# plt.legend(fontsize=10)
# plt.legend(bbox_to_anchor=(0.33, 0.28))
# plt.xlabel('Sampling fraction (%)', fontsize=14)
# plt.ylabel('Mean square error (dB)', fontsize=14)
# plt.xticks([1, 1.5, 2, 2.5, 3])
# plt.show()

########################################

# O1

O1_NMSE_gain_PCA = []

O1_NMSE_gain_proposed = [-39.3771, -41.9002, -44.7214, -45.4085, -45.5321]

O1_NMSE_gain_gp = [-40.7848, -42.7378, -43.3371, -42.6664, -43.9798]

########################################

# O1B

O1B_NMSE_gain_PCA = []

O1B_NMSE_gain_proposed = [-5.9950, -7.2754, -8.1909, -8.8120, -9.5913]

O1B_NMSE_gain_gp = [0.2526, 25.7233, 8.2995, 19.5758, 25.8806]

########################################


x_axis_A = [1, 1.5, 2, 2.5, 3]

plt.plot(x_axis_A, O1_NMSE_gain_proposed, 'r^-', label='Proposed O1')
plt.plot(x_axis_A, O1_NMSE_gain_gp, 'b^--', label='GP O1')
plt.plot(x_axis_A, O1B_NMSE_gain_proposed, 'ro-', label='Proposed O1B')
plt.plot(x_axis_A, O1B_NMSE_gain_gp, 'bo--', label='GP O1B')
plt.legend(fontsize=10)
plt.legend(bbox_to_anchor=(1.0, 0.45))
plt.xlabel('Sampling fraction (%)', fontsize=14)
plt.ylabel('Normalized mean square error (dB)', fontsize=14)
plt.xticks([1, 1.5, 2, 2.5, 3])
plt.show()

############################

# O1

# O1_MNSE_gain_PCA = [-14.2773, -14.3527, -14.4159, -14.4941, -14.6493]
# O1_RMSE_cov_PCA = [-8.2005, -8.5415, -8.8875, -9.0301, -9.0895]

# O1_MedNSE_gain_proposed = [-39.6907, -41.3394, -41.3146, -41.9915, -44.7288]
O1_RMSE_cov_proposed = [-11.5742, -13.0774, -15.3971, -16.2453, -17.9883]

O1_RMSE_cov_gp = [-10.6784, -12.3044, -13.7105, -15.0513, -16.1860]

########################################

# O1B

# O1B_MNSE_gain_PCA = [-4.3270, -4.4067, -4.4802, -4.5231, -4.5983]
# O1B_RMSE_cov_PCA = [-6.8322, -7.3921, -7.7431, -8.0226, -8.2745]

# O1B_MedNSE_gain_proposed = [-17.9799, -20.2990, -20.9773, -21.8278, -19.5665]
O1B_RMSE_cov_proposed = [-10.2588, -11.5968, -12.4958, -13.1343, -13.2848]

O1B_RMSE_cov_gp = [-9.2747, -10.2883, -11.1302, -11.7753, -12.3675]

########################################

x_axis_A = [1, 1.5, 2, 2.5, 3]

plt.figure()
plt.plot(x_axis_A, O1_RMSE_cov_proposed, 'r^-', label='Proposed O1')
plt.plot(x_axis_A, O1_RMSE_cov_gp, 'b^--', label='GP O1')
plt.plot(x_axis_A, O1B_RMSE_cov_proposed, 'ro-', label='Proposed O1B')
plt.plot(x_axis_A, O1B_RMSE_cov_gp, 'bo--', label='GP O1B')
plt.legend(fontsize=10)
plt.legend(bbox_to_anchor=(0.33, 0.28))
plt.xlabel('Sampling fraction (%)', fontsize=14)
plt.ylabel('Mean square error (dB)', fontsize=14)
plt.xticks([1, 1.5, 2, 2.5, 3])
plt.show()

#######################################