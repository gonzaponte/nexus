from pathlib import Path
import sys

import numpy  as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage     import rotate

from fire import bin_centers
from fire import metadata
from fire import check_output_directory

from invisible_cities.reco.deconv_functions import richardson_lucy


IMG_WIDTH  = 100
BIN_SIZE   =   1
ITERATIONS = 75
ITER_THR   = 1e-2
ITERATIONS = 500
ITER_THR   = 1e-4

bins     = np.arange(-IMG_WIDTH/2, IMG_WIDTH/2 + BIN_SIZE/2, BIN_SIZE)
centers  = bin_centers(bins)
img_size = len(centers)
img_x    = np.repeat(centers, len(centers))
img_y    = np.tile  (centers, len(centers))
img_xy   = np.stack([img_x, img_y], axis=1)

def interpolate_sipms(df):
    xy    = df.loc[:, "cell_x cell_y".split()].values
    q     = df.sipm_hits.values
    img_q = griddata(xy, q, img_xy, method="cubic", fill_value=0)
    return pd.DataFrame(dict(x=img_x, y=img_y, q=np.clip(img_q, 0, None)))

def deconvolve(df, psf):
    q   = df .  q.values.reshape(img_size, img_size)
    psf = psf.psf.values.reshape(psf_size, psf_size)
    img = richardson_lucy(q, psf, iterations=ITERATIONS, iter_thr=ITER_THR)
    return pd.DataFrame(dict(x=img_x, y=img_y, q=img.flatten()))

def rotation_angle(evt):
    dx = evt.x2 - evt.x1
    dy = evt.y2 - evt.y1
    return np.arctan2(dy, dx) * 180/np.pi

def rotate_img(df, evt):
    theta  = rotation_angle(evt)
    img    = df.q.values.reshape(img_size, img_size)
    newimg = rotate(img, -theta, reshape=False, mode="nearest")
    return df.assign(q=newimg.flatten())

assert len(sys.argv) == 2, "too many input files"

filename = Path(sys.argv[1])
psf_file = filename.parent.parent / "psfs"   / filename.name
output   = filename.parent.parent / "deconv" / filename.name
pitch    = metadata(filename)["pitch"]

assert filename.exists()
check_output_directory(output.parent)

print("output", output)
print("pitch ", pitch)

pairs  = pd.read_hdf(filename, "/pairs")
events = pd.read_hdf(filename, "/evtmap").set_index("event")
psf    = pd.read_hdf(psf_file, "/psf")

pairs.loc[:, "cell_x"] *= pitch
pairs.loc[:, "cell_y"] *= pitch
psf_size = int(len(psf)**0.5)

print("Applying deconvolution...")
imgs = []
for evt, df in pairs.groupby("event"):
    print(f"\r{evt:05}/{len(events):05}", end="", flush=True)
    input_img = interpolate_sipms(df)
    img       = deconvolve(input_img, psf)
    img       = rotate_img(img, events.loc[evt])
    img.loc[:, "event"] = evt
    imgs.append(img)

imgs = pd.concat(imgs, ignore_index=True)
imgs = imgs.loc[:, "event x y q".split()]
imgs  .to_hdf(output, "/imgs"  , complib="zlib", complevel=4, mode="w")
events.to_hdf(output, "/events", complib="zlib", complevel=4, mode="a")
