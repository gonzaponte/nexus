import os
import sys

import pandas as pd

filename, pitch = sys.argv[1:]
pitch = float(pitch)

df = pd.read_csv(filename, sep=" ", header=None, names="event x y".split())
df.loc[:, "cell_x"] = ((df.x + pitch/2) // pitch).astype(int)
df.loc[:, "cell_y"] = ((df.y + pitch/2) // pitch).astype(int)

df = ( df.groupby("event cell_x cell_y".split(), as_index=False)
         .x
         .count()
         .rename(columns=dict(x="tpb_hits"))
     )

outputfile = filename.replace(".txt", ".h5")
df.to_hdf(outputfile, "/data", complib="zlib", complevel=4, mode="w")

os.remove(filename)
