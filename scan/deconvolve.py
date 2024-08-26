from pathlib import Path
import sys

import numpy  as np
import pandas as pd
import tables as tb

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
    return np.clip(img_q, 0, None).reshape(img_size, img_size)

def rotation_angle(evt):
    dx = evt.x2 - evt.x1
    dy = evt.y2 - evt.y1
    return np.arctan2(dy, dx) * 180/np.pi

def rotate_img(img, evt):
    theta  = rotation_angle(evt)
    return rotate(img, -theta, reshape=False, mode="nearest")

assert len(sys.argv) == 2, "too many input files"

filename = Path(sys.argv[1])
psf_file = filename.parent.parent / "psfs"  / filename.name
output   = filename.parent.parent / "decos" / filename.name
pitch    = metadata(filename)["pitch"]

assert filename.exists()
check_output_directory(output.parent)

print("output", output)
print("pitch ", pitch)

pairs  = pd.read_hdf(filename, "/pairs")
events = pd.read_hdf(filename, "/evtinfo").set_index("event")
psf    = pd.read_hdf(psf_file, "/psf")

pairs.loc[:, "cell_x"] *= pitch
pairs.loc[:, "cell_y"] *= pitch
psf_size = int(len(psf)**0.5)
psf = psf.psf.values.reshape(psf_size, psf_size)

print("Applying deconvolution...")
with tb.open_file(output, "w", filters=tb.Filters(complib="zlib", complevel=4)) as file:
    store = file.create_earray(file.root, "imgs", atom=tb.Float32Atom(), shape=(0, img_size, img_size))
    file.create_array(file.root, "bins", bins)
    file.create_array(file.root, "xys", img_xy)

    for evt, df in pairs.groupby("event"):
        print(f"\r{evt:05}/{len(events):05}", end="", flush=True)
        img_input  = interpolate_sipms(df)
        img_output = richardson_lucy(img_input, psf, iterations=ITERATIONS, iter_thr=ITER_THR)
        img_output = rotate_img(img_output, events.loc[evt])
        store.append(img_output[np.newaxis])

events.reset_index().to_hdf(output, "/evtinfo", complib="zlib", complevel=4, mode="a")
