import os
import sys

from glob      import glob
from functools import reduce

from invisible_cities.io.dst_io import load_dst
from invisible_cities.io.dst_io import df_writer

import numpy  as np
import pandas as pd
import tables as tb


def load_psf(file):
    return load_dst(file, "PSF", "PSFs")


def combine_psfs(acc, new):
    columns  = ["xr", "yr", "zr", "x", "y", "z"]
    acc      = acc.assign(factor=acc.factor * acc.nevt)
    new      = new.assign(factor=new.factor * new.nevt)
    combined = pd.concat( [acc, new]
                        , ignore_index = True
                        , sort         = False
                        )
    combined = combined.groupby(columns, as_index=False).agg("sum")
    average  = combined.factor / combined.nevt
    acc      = combined.assign(factor = np.nan_to_num(average))
    return acc


input_folder = sys.argv[1]
output_file  = sys.argv[2] if len(sys.argv) > 2 else "merged.psf"

files_in     = glob(os.path.join(input_folder, "*.psf"))
combined_psf = reduce(combine_psfs, map(load_psf, files_in))

with tb.open_file(files_in[0]) as file:
    title = file.root.PSF.PSFs.title

with tb.open_file(output_file, 'w') as h5out:
    df_writer(h5out, combined_psf
             , "PSF", "PSFs"
             , compression = "ZLIB4"
             , descriptive_string = title
             )
