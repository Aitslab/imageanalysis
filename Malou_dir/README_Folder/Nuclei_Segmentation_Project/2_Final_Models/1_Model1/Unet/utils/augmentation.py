import numpy as np
import skimage.transform
from skimage.util import img_as_ubyte
# Based on example code from:
# http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html

def deform(image1, image2, points=10, distort=5.0):
    
    # create deformation grid 
    rows, cols = image1.shape[0], image1.shape[1]
    src_cols = np.linspace(0, cols, points)
    src_rows = np.linspace(0, rows, points)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add distortion to coordinates
    s = src[:, 1].shape
    dst_rows = src[:, 1] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
    dst_cols = src[:, 0] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
    
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = skimage.transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = rows 
    out_cols = cols
    out1 = skimage.transform.warp(image1, tform, output_shape=(out_rows, out_cols), mode="symmetric")
    out2 = skimage.transform.warp(image2, tform, output_shape=(out_rows, out_cols), mode="symmetric")
    
    return img_as_ubyte(out1), img_as_ubyte(out2)


def resize(x, y):
    wf = 1 + np.random.uniform(-0.25, 0.25)
    hf = 1 + np.random.uniform(-0.25, 0.25)

    w,h = x.shape[0:2]

    wt, ht = int(wf*w), int(hf*h)

    new_x = skimage.transform.resize(x, (wt,ht))
    new_y = skimage.transform.resize(y, (wt,ht))

    return new_x, new_y

