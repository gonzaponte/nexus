import sys

import pandas as pd

filename, pitch = sys.argv[1:]
pitch = float(pitch)

df = pd.read_csv(filename, sep=" ", header=None, names="event x y".split())
df.loc[:, "cell_x"] = (df.x - pitch/2) // pitch + 1
df.loc[:, "cell_y"] = (df.y - pitch/2) // pitch + 1

df = ( df.groupby("event cell_x cell_y".split(), as_index=False)
         .x
         .count()
         .rename(columns=dict(x="tpb_hits"))
         .assign( cell_x = lambda d: d.cell_x * pitch
                , cell_y = lambda d: d.cell_y * pitch)
     )

outputfile = filename.replace(".txt", ".h5")
df.to_hdf(outputfile, "/data", complib="zlib", complevel=4, mode="w")
