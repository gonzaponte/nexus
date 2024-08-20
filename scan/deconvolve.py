import numpy  as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage     import rotate

from fire import bin_centers


IMG_WIDTH  = 100
BIN_SIZE   =   1
ITERATIONS = 75
ITER_THR   = 1e-2

bins     = np.arange(-IMG_WIDTH/2, IMG_WIDTH/2 + BIN_SIZE/2, BIN_SIZE)
centers  = bin_centers(bins)
img_size = len(centers)
img_x    = np.repeat(centers, len(centers))
img_y    = np.tile  (centers, len(centers))
img_xy   = np.stack([img_x, img_y], axis=0)

def interpolate_sipms(df):
    xy = df.loc[:, "cell_x cell_y".split()].values
    q  = df.sipm_hits.values
    img_q = griddata(xy0, q, img_xy, method="cubic", fill_value=0)
    return pd.DataFrame(dict(x=img_x, y=img_y, q=np.clip(img_q, 0)))

def deconvolve(df, psf):
    q   = df .q.reshape(img_size, img_size)
    psf = psf.q.reshape(img_size, img_size)
    out = richardson_lucy(q, psf, iterations=ITERATIONS, iter_thr=ITER_THR)
    diff = out[0] # relative difference after iterating
    last = out[1] # last iteration
    img  = out[2] # deconvolved image
    return diff, last, pd.DataFrame(dict(x=img_x, y=img_y, q=img.flatten()))

def rotate_img(df):
    theta  = rotation_angle(df, degrees=True)
    img    = df.q.reshape(img_size, img_size)
    newimg = rotate(img, theta, reshape=False, mode="nearest")
    return df.assign(q=newimg.flatten())

from fire import metadata
from fire import file_number
from fire import remove_subfile
from fire import barycenter

filename  = Path(sys.argv[1:])
output    = filename.parent.parent / "deconv" / filename.name
pitch     = metadata(onefile)["pitch"]

for f in filenames: assert f.exists()

print("output", output)
print("pitch ", pitch)

data = pd.read_hdf(filename, "/data")

print("Interpolating...")
data = data.groupby("event").apply()
