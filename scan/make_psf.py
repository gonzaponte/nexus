import sys
from pathlib import Path

import numpy  as np
import pandas as pd

from fire import metadata
from fire import file_number
from fire import remove_subfile
from fire import barycenter
from fire import bin_centers


def relativize(df, rel_thr=0.02):
    x0, y0 = barycenter(df, rel_thr).values
    df.loc[:, "cell_x"  ] -= x0
    df.loc[:, "cell_y"  ] -= y0
    df.loc[:, "tpb_hits"] /= df.tpb_hits.sum()
    return df


filenames = list(map(Path, sys.argv[1:]))
onefile   = filenames[0]
output    = onefile.parent.parent / "psfs" / remove_subfile(onefile)
pitch     = metadata(onefile)["pitch"]

for f in filenames: assert f.exists()

print("output", output)
print("pitch ", pitch)

print("Loading data...")
data = [ pd.read_hdf(filename, "/data").assign(file=file_number(filename))
         for filename in filenames ]
data = pd.concat(data, ignore_index=True)
data.loc[:, "cell_x"] *= pitch
data.loc[:, "cell_y"] *= pitch

print("Relativizing data...")
events = data.groupby("file event".split()).apply(relativize)

print("Computing PSF...")
bins = np.linspace(-100, 100, 201)
cs   = bin_centers(bins)
xs   = np.repeat(cs, len(cs))
ys   = np.tile  (cs, len(cs))
n    = np.histogram2d(events.cell_x, events.cell_y, (bins,)*2                       )[0].flatten()
psf  = np.histogram2d(events.cell_x, events.cell_y, (bins,)*2, weights=data.tpb_hits)[0].flatten()

psf = pd.DataFrame(dict(x=xs, y=ys, psf=psf/n, n=n))
psf.to_hdf(output, "/data", complib="zlib", complevel=4, mode="w")
