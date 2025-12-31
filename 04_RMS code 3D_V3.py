import os
import cv2
import tifffile
import numpy as np
from skimage import filters, morphology
from skimage.exposure import rescale_intensity


### HELPER FUNCTIONS ###
def calc_flow(img1, img2):
    """A function that takes a pre- and post-movement image and returns the x and y vector arrays for the difference.

    Parameters
    ----------
    img1 : ndarray
        the initial, pre-movement image.
    img2 : ndarray
        the post-movement image.

    Returns
    ----------
    u : ndarray
        an array of the same size as the input with the X movement vectors for each point set.

    v : ndarray
        an array of the same size as the input with the Y movement vectors for each point set.

    """

    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5,
                                        levels=3,
                                        winsize=winsize,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=poly_sigma,
                                        flags=0)
    u = flow[..., 0]
    v = flow[..., 1]

    return u, v


def axis_selector(RMSE_axis_, positioninformation):
    """ A function that takes the position data calculated earlier and the RMSE planes we wish to obtain RMSE for
    and removes none relevant axis. This allows the subsequent functions to process the data as though all the data
    is in the plane of the data.

    :param RMSE_axis_:string
    The axis to be processed. Valid options include: XY, YZ, XZ, XYZ, or 'all' to output all axis.
    :param positioninformation:ndarray
    The position_data array from the main program containing the original and distorted corrected coordinates.
    :return: positioninformation:ndarray
    The newly edited/unedited array with the appopriate axis zeroed.
    """
    if RMSE_axis_.lower() == 'xy':
        positioninformation[:, 0] = 0
        positioninformation[:, 3] = 0

    elif RMSE_axis_.lower() == 'xz':
        positioninformation[:, 1] = 0
        positioninformation[:, 4] = 0

    elif RMSE_axis_.lower() == 'yz':
        positioninformation[:, 2] = 0
        positioninformation[:, 5] = 0

    elif RMSE_axis_.lower() == 'xyz':
        positioninformation = positioninformation

    print(RMSE_axis_)
    return positioninformation


def RMSE_error_calculator(positioninformation):
    """ This function is the meat of the program. The calculator takes the position data that has been processed and
    for each coordinate calculates the distance between for every original coordinate and then every distorted
    coordinate. After which the distances for equivalent coordiante points pairs are subtracted from one another to
    calculate the difference between the original length and the distorted length. These values are then used to
    calculate the values for RMSE. The program is built to handle a large number of points wherein (up to a limit) the
    process can be broken down and complete with feedback into how far through the million+ data points being used.

    :param positioninformation:ndarray
    The position_data array from the main program containing the original and distorted corrected coordinates.
    :return: bin_array:ndarray
    The RMSE array with data binned to graph-able values.
    """
    # Loop variables
    counter = 0
    roaming_x = 0
    position_start = 0
    position_end = 0
    length = feature_coordinates.shape[0]
    print("There are", length, "feature coordinates.")

    # Bin Variables
    binsize = 1  # Arbitrary value
    maxY = 0  # Establish the max Y as a variable
    bin_array = np.zeros((int(np.sqrt((img_post.shape[0] * z_pixel_size) ** 2
                                      + (img_post.shape[1] * xy_pixel_size) ** 2
                                      + (img_post.shape[2] * xy_pixel_size) ** 2) / binsize), 7))

    for x in range(0, feature_coordinates.shape[0]):
        # Static arrays for each coordinate set to measure against
        pA = positioninformation[x, 0:3]
        sP = positioninformation[roaming_x:, 0:3]
        distances = np.linalg.norm(sP - pA, ord=2, axis=1)

        original_dist = np.abs(distances.transpose())

        # Deformed distances
        pA_2 = positioninformation[x, 3:6]
        sP_2 = positioninformation[roaming_x:, 3:6]
        distances_2 = np.linalg.norm(sP_2 - pA_2, ord=2, axis=1)

        # Original distance - deformed distance
        difference_dist = np.abs(distances.transpose() - distances_2.transpose())

        # Binning Data in this loop
        maxX = np.max(original_dist)
        maxY = max(maxY, np.max(difference_dist))
        bins = np.arange(0, maxX, binsize)

        digitize = np.digitize(original_dist, bins)  # Assign each of the dist values to a bin
        # Bin the values relative to a digitisation of their bin values
        binned_orig_array = [original_dist[digitize == i] for i in range(1, len(bins) - 1)]
        # Use the same digitization to bin the dif values relative to the original ones
        binned_diff_array = [difference_dist[digitize == i] for i in range(1, len(bins) - 1)]

        # Now need Mean and SD for each bin
        mean_sd_array = np.zeros((len(binned_orig_array), 7))
        for i in range(0, len(binned_orig_array)):
            mean_sd_array[i, 0] = np.mean((binned_orig_array[i]))  # Original coords MEAN per bin
            mean_sd_array[i, 1] = np.sum(np.square(binned_orig_array[i]))  # Original coords SUM of squares per bin
            mean_sd_array[i, 2] = np.std(binned_orig_array[i])  # Original coords SD per bin
            mean_sd_array[i, 3] = np.mean((binned_diff_array[i]))  # Difference coords MEAN per bin
            mean_sd_array[i, 4] = np.sum(np.square(binned_diff_array[i]))  # Difference coords SUM of squares per bin
            mean_sd_array[i, 5] = np.std(binned_diff_array[i])  # Difference coords SD per bin
            mean_sd_array[i, 6] = np.size(binned_orig_array[i])  # Original coords count per bin

            # Error will be generated by 0/0 so these can be safely adjusted to 0
            mean_sd_array = np.nan_to_num(mean_sd_array, copy=False)

            ### CALCULATIONS ###
            # We need to map the Means, Squared Sum and Standard Deviation's to the combined table.

            # Combining Standard deviations
            # Dataset 1:
            #   q1 = (n1 - 1) * var(x1) + n1 * mean(x1) ^ 2
            # Dataset 2:
            #   q2 = (n2 - 1) * var(x2) + n2 * mean(x2) ^ 2
            # Combine the q values
            #   qc = q1 + q2
            # Calculate the new standard deviation (where mean(x) is the combined mean)
            #   sc = sqrt((qc - (n1 + n2) * mean(x) ^ 2) / (n1 + n2 - 1))
            # Current Variance based value for bin_array and mean_sd_array need to be precomputed for qc's.
            oq1 = (bin_array[i, 6] - 1) * np.square(bin_array[i, 2]) + bin_array[i, 6] * np.square(bin_array[i, 0])
            oq2 = (mean_sd_array[i, 6] - 1) * np.square(mean_sd_array[i, 2]) + mean_sd_array[i, 6] * np.square(mean_sd_array[i, 0])

            dq1 = (bin_array[i, 6] - 1) * np.square(bin_array[i, 5]) + bin_array[i, 6] * np.square(bin_array[i, 3])
            dq2 = (mean_sd_array[i, 6] - 1) * np.square(mean_sd_array[i, 5]) + mean_sd_array[i, 6] * np.square(mean_sd_array[i, 3])

            # MEANS
            # Avoiding zeroes...
            if bin_array[i, 6] + mean_sd_array[i, 6] != 0:
                bin_array[i, 0] = ((bin_array[i, 0] * bin_array[i, 6]) + (mean_sd_array[i, 0] * mean_sd_array[i, 6])) / (
                                    bin_array[i, 6] + mean_sd_array[i, 6])
                bin_array[i, 3] = ((bin_array[i, 3] * bin_array[i, 6]) + (mean_sd_array[i, 3] * mean_sd_array[i, 6])) / (
                                    bin_array[i, 6] + mean_sd_array[i, 6])
            else:
                bin_array[i, 0] = 0
                bin_array[i, 3] = 0

            # SQUARED SUM
            bin_array[i, 1] = bin_array[i, 1] + mean_sd_array[i, 1]
            bin_array[i, 4] = bin_array[i, 4] + mean_sd_array[i, 4]

            # STANDARD DEVIATION
            # beware of zeroes...
            if bin_array[i, 6] + mean_sd_array[i, 6] - 1 > 0:
                bin_array[i, 5] = np.sqrt(((dq1 + dq2) - (bin_array[i, 6] + mean_sd_array[i, 6]) * np.square(bin_array[i, 3])) /
                                      (bin_array[i, 6] + mean_sd_array[i, 6] - 1))
                bin_array[i, 2] = np.sqrt(((oq1 + oq2) - (bin_array[i, 6] + mean_sd_array[i, 6]) * np.square(bin_array[i, 0])) /
                                      (bin_array[i, 6] + mean_sd_array[i, 6] - 1))
            else:
                bin_array[i, 2] = 0
                bin_array[i, 5] = 0

            # COUNT
            bin_array[i, 6] = bin_array[i, 6] + mean_sd_array[i, 6]

        # Next loop setup
        roaming_x = roaming_x + 1
        counter = counter + 1
        if counter % 1000 == 0:
            print(counter, "/", length, "    ", counter/length*100, "%")

    return bin_array


def data_process_output(bin_array, suffix):
    """ Simple save function with final RMSE calculations and error checking performed, with a final expansion factor
    adjustment.

    :param bin_array:string
    The array output from the RMSE_error_calculator.
    :param suffix:string
    The RMSE axis being calculated. It is appended to the end of the file name.
    :return: bin_array:ndarray
    The newly edited/unedited array.
    """
    ### Process the Data ###
    # First calculate the RMS from the summed square values.
    # We calculate the mean of the squares by dividing by the total count in each bin
    # Then square root the values element-wise to obtain the RMS.
    for i in range(0, bin_array.shape[0]):
        if bin_array[i, 6] != 0:
            bin_array[i, 1] = np.sqrt(bin_array[i, 1]/bin_array[i, 6])
            bin_array[i, 4] = np.sqrt(bin_array[i, 4]/bin_array[i, 6])
        else:
            bin_array[i, 1] = 0
            bin_array[i, 4] = 0

    # Then we need to account for expansion factor
    bin_array[:, :6] = bin_array[:, :6] / expansion_factor  # All values except count are adjusted.

    ### SAVE THE OUTPUT ###
    np.save('RMS_data' + suffix, bin_array)

    return bin_array


def main_code(RMSE_axis_, positioninformation):
    """ Function combining the main functions within this document. Runs the main RMSE code once through.

    :param RMSE_axis_:string
    The axis to be processed. Valid options include: XY, YZ, XZ, XYZ, or 'all' to output all axis.
    :param positioninformation:ndarray
    The position_data array from the main program containing the original and distorted corrected coordinates.
    :return: complete_bin_array:ndarray
    The final array for the axis processed.
    """
    position_info_data = axis_selector(RMSE_axis_, positioninformation)
    print(np.amax(position_info_data, axis=0))
    complete_bin_array = RMSE_error_calculator(position_info_data)
    complete_bin_array = data_process_output(complete_bin_array, RMSE_axis_)

    return complete_bin_array


def all_axis(RMSE_axis_, positioninfo):
    """ Allows multiple runs of the program in the instance that 'all' is used for RMSE_axis.

    :param RMSE_axis_:string
    The axis to be processed. Valid options include: XY, YZ, XZ, XYZ, or 'all' to output all axis.
    :param positioninfo:ndarray
    The position_data array from the main program containing the original and distorted corrected coordinates.
    """
    positioninformation = np.zeros((positioninfo.shape[0], positioninfo.shape[1]))
    positioninformation = positioninfo.copy()
    if RMSE_axis_.upper() == 'ALL':
        RMSE_axis_ = ('XYZ', 'XY', 'YZ', 'XZ')

        for element in RMSE_axis_:
            main_code(element, positioninformation)
            positioninformation = positioninfo.copy()

    elif RMSE_axis_.upper() == 'XY' or 'YZ' or 'XZ' or 'XYZ':
        main_code(RMSE_axis_, positioninformation)


    else:
        raise "Not a valid axis"


### INPUTS ###
# Imageset (.tif) with pre- and post-images aligned with the 3D post stack in the first channel and the 3d
# pre stack in the second channel. Post [0] and Pre [1]
image_pair = ("02_Aligned_PRE_POST_example.tif")

if not os.path.exists(image_pair):
    print(
        "ERROR: File '{}' not found: Please specify the file.".format(image_pair))

## Basic parameters
xy_pixel_size = 0.0488759  # in microns post expansion pixel size
z_pixel_size = 0.18        # in microns post expansion stack size.
expansion_factor = 4.032765865       # obtained from elastex/SIFT/fijiyama or other code.
winsize = 40               # in pixels for flow calculation
poly_sigma = 3             # for the flow calculation
post_image_sigma = 4       # for gaussian blurring the post image
min_brightpoints = 200000
RMSE_axis = "all"          # axis selection for error calculation. Options: XY, YZ, XZ, XYZ and ALL. (case-insensitive)

### Compute Local Distortion Field ###
img_pair_in = tifffile.imread(image_pair)

if np.ndim(img_pair_in) == 3:       # To allow 2D arrays to be parsed with minimal interference.
    img_pair_in = np.expand_dims(img_pair_in, 0)

img_pre = img_pair_in[:, 1, :, :]
img_post = img_pair_in[:, 0, :, :]

# Blur and rescale intensity to normalise images.
img_post_blur = filters.gaussian(img_post, post_image_sigma, preserve_range=True)
img_post_scaled = rescale_intensity(img_post_blur, (img_post_blur.min(), np.percentile(img_post, 99.9)), 'uint8')
img_pre_scaled = rescale_intensity(img_pre, (img_pre.min(), np.percentile(img_pre, 99.9)), 'uint8')


### COLLECTION OF DISTORTION VECTOR DATA ###
# The deformation vectors are calculated and stored in x, y and z components.
dist_vector_x = np.zeros(img_post.shape)    # Establish array for x data
dist_vector_y = np.zeros(img_post.shape)    # Establish array for y data

for i in range(0, img_post.shape[0]):
    dist_vector_x[i], dist_vector_y[i] = calc_flow(img_post[i], img_pre[i])
    if i//10 == 0:
        print(i, "xy vector data")

# PREPROCESS XY->XZ
img_post_reshaped = np.transpose(img_post, axes=(2, 1, 0))
img_pre_reshaped = np.transpose(img_pre, axes=(2, 1, 0))

dist_vector_z = np.zeros(img_post_reshaped.shape)   # Establish array for z data

for i in range(0, img_post_reshaped.shape[0]):
    u, dist_vector_z[i] = calc_flow(img_post_reshaped[i], img_pre_reshaped[i])
    if i//10 == 0:
        print(i, "xz vector data")


### Thresholding the Post Image ###
# We need to threshold the image to reduce the number of feature coordinates to a calculable range.
# To have the code complete within several hours a max of 200,000 feature coordinates is advised.
def threshold_skeleton_max(array, min_brightpoints=1000000, max_features=200000):
    """ A function that takes an image and thresholds based upon the brightness of the lowest brightness point
    selected for by the minimum number of bright points. This array is then skeletonised.
    If the array contains greater than an arbitrary max feature number (for computing time) the remaining points are
    further thresholded using the skeleton as a mask, and selecting for the remaining x number of the brightest points
    to return a number of feature coordinates contained in an array.

    :param array:ndarray
        The image array to be processed.
    :param min_brightpoints:int
        The integer minimum value of coordinates to threshold by. Higher values will include more of the image.
    :param max_features:int
        THe integer maximum value of the number of output coordinates, selected based upon the intensity values.
    :return: thresholded_array:ndarray
        The thresholded array containing all values above the threshold.
    """

    # Value check
    if min_brightpoints > np.size(array):
        min_brightpoints = np.size(array)*0.40
    # Initial Threshold: thresholds the image based upon the minimum number of bright points. Histogram based approach.
    flat = np.ravel(array)
    sorted_array = np.sort(flat, kind='mergesort')
    thresholded_value = sorted_array[-min_brightpoints]
    thresholded_array = array >= thresholded_value          # Array where all values above the threshold are contained.

    print(np.count_nonzero(thresholded_array))
    # Skeletonise: used to reduce structures to core areas. Better on larger structures, can overcompensate on smaller.
    if np.count_nonzero(thresholded_array) > max_features:
        skeleton = morphology.skeletonize(thresholded_array)
        print(np.count_nonzero(skeleton))

    else:
        skeleton = thresholded_array

    # Reduction: to max number of features. If features remain larger than comfortable computing time then we can reduce
    # further as appropriate.
    if np.count_nonzero(skeleton) > max_features:
        mask = array * skeleton
        sort = np.sort(np.ravel(mask), kind='mergesort')
        reduce = sort[-max_features]
        thresholded_array = mask > reduce

    else:
        thresholded_array = skeleton

    return thresholded_array


img_pti = threshold_skeleton_max(img_post, min_brightpoints, min_brightpoints*1.1)

# Find the coordinates of the skeleton
coordinate_tuple = np.nonzero(img_pti)
feature_coordinates = np.transpose(np.asarray(coordinate_tuple))  # Coordinates are in (z,y,x) order

### Create a Super Array ###
# Columns 1, 2 and 3 = Z, Y and X coordinates for the skeleton
# Columns 4, 5 and 6 = Z, Y and X coordinates after applying the deformation value
position_data = np.zeros((feature_coordinates.shape[0], 6))

# Map all the data to it
position_data[:, 0:3] = feature_coordinates
# Find all the distortion values for the below points
for x in range(0, position_data.shape[0]):
    position_data[x, 3] = (position_data[x, 0] +
                           dist_vector_z[int(position_data[x, 2]), int(position_data[x, 1]), int(position_data[x, 0])])  # z deformation
    position_data[x, 4] = (position_data[x, 1] +
                           dist_vector_y[int(position_data[x, 0]), int(position_data[x, 1]), int(position_data[x, 2])])  # y deformation
    position_data[x, 5] = (position_data[x, 2] +
                           dist_vector_x[int(position_data[x, 0]), int(position_data[x, 1]), int(position_data[x, 2])])  # x deformation

# Adjust for pixel size in each axis
position_data[:, 0] = position_data[:, 0]*z_pixel_size
position_data[:, 1:3] = position_data[:, 1:3]*xy_pixel_size
position_data[:, 3] = position_data[:, 3]*z_pixel_size
position_data[:, 4:] = position_data[:, 4:]*xy_pixel_size

### Measurement Lengths and Difference Calculations ###
# Column 1 = Skeleton Coordinate Magnitude
# Column 2 = Absolute value after Skeleton Coordinate Magnitude subtracted from Deformed Coordinate Magnitude
array_length = feature_coordinates.shape[0] + 1
measurement_array = np.zeros((int(array_length), 2))

# Now for some hefty maths...
# Column 0 is the length distance between the undistorted array points
# Column 1 is the difference in length between distorted and undistorted
dist_array = np.zeros((feature_coordinates.shape[0], 1))
original_coord = np.zeros((feature_coordinates.shape[0], 2))


### MAIN CODE ###
# Body of RMSE runs through embedded function in below line.
final_data = all_axis(RMSE_axis, position_data)


######## END ########
