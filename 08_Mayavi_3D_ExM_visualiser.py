import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile
import skimage
from skimage.exposure import rescale_intensity
from mayavi import mlab


def calc_flow(img1, img2):
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


def show_flow_field(u, v, ax=None, subsample=32, color='r', scale=1 / 2):
    if ax is None:
        ax = plt.gca()
    x_, y_ = np.meshgrid(range(0, u.shape[1]), range(0, u.shape[0]))
    ax.quiver(x_[subsample // 2::subsample, subsample // 2::subsample],
              y_[subsample // 2::subsample, subsample // 2::subsample],
              u[subsample // 2::subsample, subsample // 2::subsample],
              v[subsample // 2::subsample, subsample // 2::subsample],
              angles='xy',
              scale_units='xy',
              scale=scale,
              pivot='mid',
              headwidth=2,
              headlength=3,
              color=color,
              width=0.002)
    plt.savefig("dist_map.pdf", dpi=600)

# Inputs
### 3D stacks pre/post aligned in a single stack with two channels (one file). Order is post/pre for channels.
image_stack_location = ("02_Aligned_PRE_POST_example.tif")

if not os.path.exists(image_stack_location):
    print(
        "ERROR: File '{}' not found: Please check the location.".format(image_stack_location))

### Basic parameters
xy_pixel_size = 0.0488759  # in microns post expansion pixel size
z_pixel_size = 0.18  # in microns post expansion stack size.
subsample = 40
winsize = 35
poly_sigma = 3
post_image_sigma = 4    # akin to expansion factor. a blur to help the two images appear similar.
expansion_factor = 4.032765865

# Compute local distortion field
img_pair_in = tifffile.imread(image_stack_location)

if np.ndim(img_pair_in) == 3:       # To allow 2D arrays to be parsed with minimal interference.
    img_pair_in = np.expand_dims(img_pair_in, 0)

img_pre_original = img_pair_in[:, 1, :, :]
img_post_original = img_pair_in[:, 0, :, :]

img_post = skimage.filters.gaussian(img_post_original, post_image_sigma, preserve_range=True)
img_post = rescale_intensity(img_post, (img_post.min(), np.percentile(img_post, 99.9)), 'uint8')
img_pre = rescale_intensity(img_pre_original, (img_pre_original.min(), np.percentile(img_pre_original, 99.9)), 'uint8')

# XY
dist_vector_x = np.zeros(img_post.shape)
dist_vector_y = np.zeros(img_post.shape)

for i in range(0, img_post.shape[0]):
    dist_vector_x[i], dist_vector_y[i] = calc_flow(img_post[i], img_pre[i])
    if i//10 == 0:
        print(i, "xy")

dist_vector_x = (dist_vector_x * xy_pixel_size) / expansion_factor
dist_vector_y = (dist_vector_y * xy_pixel_size) / expansion_factor

# XZ
# PREPROCESS XY->XZ
img_post_reshaped = np.transpose(img_post, axes=(2, 1, 0))
img_pre_reshaped = np.transpose(img_pre, axes=(2, 1, 0))
dist_vector_z = np.zeros(img_post_reshaped.shape)


for i in range(0, img_post_reshaped.shape[0]):
    u, dist_vector_z[i] = calc_flow(img_post_reshaped[i], img_pre_reshaped[i])
    if i//10 == 0:
        print(i, "xz")

dist_vector_z = np.transpose(dist_vector_z, axes=(2, 1, 0))
dist_vector_z = (dist_vector_z * z_pixel_size) / expansion_factor


### MAYAVI VISUALISER CALCULATIONS ###

angle = np.zeros((np.size(dist_vector_x)))
flat_X = np.ravel(dist_vector_x)
flat_y = np.ravel(dist_vector_y)
flat_z = np.ravel(dist_vector_z)

for i in range(0, dist_vector_x.size):
    u = (flat_X[i], flat_y[i], flat_z[i])
    v = (flat_X[i], flat_y[i], 0)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    angle[i] = np.arccos(np.dot(u, v)/(norm_u * norm_v))
    if flat_z[i] < 0:
        angle[i] = angle[i] * -1
    if i % 100000 == 0:
        print(i, "matrix")

angle = np.nan_to_num(angle)
angle3d = angle.reshape(np.shape(dist_vector_x))

a = mlab.contour3d(img_post_original, color=(0.55, 0.7, 0.94), opacity=0.85)
b = mlab.contour3d(img_pre_original, color=(0.94, 0.6, 0.6), opacity=0.4)
c = mlab.quiver3d(dist_vector_z, dist_vector_y, dist_vector_x, mask_points=500, scale_factor=20.0,
                  scalars=angle3d, mode="arrow")

c.glyph.color_mode = "color_by_scalar"

a.actor.actor.scale = (z_pixel_size/xy_pixel_size, 1, 1)
b.actor.actor.scale = (z_pixel_size/xy_pixel_size, 1, 1)
c.actor.actor.scale = (z_pixel_size/xy_pixel_size, 1, 1)


mlab.show()

