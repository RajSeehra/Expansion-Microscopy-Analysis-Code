import numpy as np
import matplotlib.pyplot as plt


def plotting(bin_array, label, colour):
    # Account for the empty end columns and expansion factor of the sample.
    effective_bin_array = bin_array
    array_end = np.asarray(np.where(effective_bin_array[:, 0] == max(effective_bin_array[:, 0])))+1
    effective_bin_array = effective_bin_array[:int(array_end[0]), :]

    # Establish the Standard Error
    se_array = np.zeros((np.shape(effective_bin_array)[0], 1))
    for i in range(0, se_array.shape[0]):
        se_array[i, 0] = effective_bin_array[i, 5]# / np.sqrt(effective_bin_array[i, 6])

    # Cumulative summing
    # effective_bin_array[:, 4] = np.cumsum(effective_bin_array[:, 4])/np.max(np.cumsum(effective_bin_array[:, 4]))*100
    plt.plot(effective_bin_array[:, 1], effective_bin_array[:, 4], "k-", label=label, color=colour)
    plt.fill_between(effective_bin_array[:, 1], effective_bin_array[:, 4] + se_array[:, 0],
                     effective_bin_array[:, 4] - se_array[:, 0], facecolor=colour, alpha=0.25)

### INPUT ###
path = ("05_"
                      )
###

bin_arrayXYZ = np.load(path+"RMSE_dataXYZ.npy")
bin_arrayXY = np.load(path+"RMSE_dataXY.npy")
bin_arrayXZ = np.load(path+"RMSE_dataXZ.npy")
bin_arrayYZ = np.load(path+"RMSE_dataYZ.npy")

plotting(bin_arrayXYZ, 'XYZ', colour='b')
plotting(bin_arrayXY, 'XY', colour='m')
plotting(bin_arrayXZ[:70, :], 'XZ', colour='g')  # Adjusted to remove excess for the example.
plotting(bin_arrayYZ, 'YZ', colour='r')

# Conventional
plt.legend()
plt.xlabel('Measurement Length (microns)')
plt.ylabel('RMS Error (microns)')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on', left=True, right=True)
plt.show()