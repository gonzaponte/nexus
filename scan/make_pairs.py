import sys
from pathlib import Path

import numpy  as np
import pandas as pd

from fire import metadata
from fire import file_number
from fire import remove_subfile
from fire import barycenter

filenames = list(map(Path, sys.argv[1:]))
onefile   = filenames[0]
output    = onefile.parent.parent / "pairs" / remove_subfile(onefile)
pitch     = metadata(onefile)["pitch"]
DET_PROB  = 0.1170844

for f in filenames: assert f.exists()

print("output", output)
print("pitch ", pitch)

def augment(df):
    rotate90 = np.array([0, -1, 1, 0], dtype=int).reshape(2,2)
    xy0 = df.loc[:, "cell_x cell_y".split()].values.T
    xy1 = rotate90.dot(xy0)
    xy2 = rotate90.dot(xy1)
    xy3 = rotate90.dot(xy2)
    return pd.concat([ df.assign(rotation=0)
                     , df.assign(rotation=1, cell_x=xy1[0], cell_y=xy1[1])
                     , df.assign(rotation=2, cell_x=xy2[0], cell_y=xy2[1])
                     , df.assign(rotation=3, cell_x=xy3[0], cell_y=xy3[1])
                     ], ignore_index=True)

def propagate(df):
    df.loc[:, "sipm_hits"] = np.random.poisson(df.tpb_hits.values * DET_PROB)
    return df


def shuffle(df):
    lbls = "file event rotation".split()
    idxs = [idx for idx, _ in df.groupby(lbls)]
    idxs = np.random.permutation(idxs)
    idxs = [tuple(i) for i in idxs]
    df = df.set_index(lbls).loc[idxs].reset_index()
    return df


print("Loading data...")
data = [ pd.read_hdf(filename, "/data").assign(file=file_number(filename))
         for filename in filenames ]
print("list size = ", sum(map(sys.getsizeof, data))/1024**2)
print("events per file: ", list(map(lambda df: df.event.unique().size, data)))
data = pd.concat(data, ignore_index=True)
print("concat size = ", sys.getsizeof(data)/1024**2)

# For first run
#data.loc[:, "cell_x"] = np.round(data.cell_x / pitch).astype(int)
#data.loc[:, "cell_y"] = np.round(data.cell_y / pitch).astype(int)

print("Preparing data...")
data = augment(data)
print("augmented size = ", sys.getsizeof(data)/1024**2)

data = propagate(data)
print("propagated size = ", sys.getsizeof(data)/1024**2)

events = data.groupby("file event rotation".split(), as_index=False)
cogs   = events.apply(barycenter)

# data.to_hdf("cache.h5", "/events")
# cogs.to_hdf("cache.h5", "/cogs")
# data = pd.read_hdf("cache.h5", "/events")
# cogs = pd.read_hdf("cache.h5", "/cogs")

# bias the pair saampling so the distribution of distances is more
# uniform
bias = 0.4
bias = np.array([bias, 1, bias]) / (1 + 2*bias)

print("Pairing events...")
pairs = []
stuck = 0
while stuck<20:
    n    = len(cogs)
    print(f"\r{n:06} events left", end="", flush=True)
    cogs = shuffle(cogs)

    # Randomly add +-pitch or not, according to the bias probabilities
    # and group events in pairs. Each event in the first half of the
    # dataset is paired with its corresponding index in the second
    # half of the dataset. Since the events are randomly distributed,
    # there is no bias in pairing them like this.
    xp, yp = np.random.choice([-1, 0, 1], p=bias, size=(2, n))
    x1 = (cogs.xb.values + xp).reshape(2, n//2)
    y1 = (cogs.yb.values + yp).reshape(2, n//2)

    # Compute distances between pairs and keep only those in the
    # desired range (p/2 < d < 2p). If no valid pairs are found, try
    # again for up to `stuck` times
    dx = np.diff(x1, axis=0)[0]
    dy = np.diff(y1, axis=0)[0]
    dr = (dx**2 + dy**2)**0.5

    s   = (dr > 0.5) & (dr < 2)
    if not np.any(s):
        stuck += 1
        continue

    stuck   = 0
    indices = np.argwhere(s).flatten()
    cogs2   = cogs.assign(xp=xp, yp=yp)
    events1 = cogs2.iloc[indices       ]
    events2 = cogs2.iloc[indices + n//2]
    assert len(events1) == len(events2) == len(dr[s])

    for (_, event1), (_, event2), d in zip(events1.iterrows(), events2.iterrows(), dr[s]):
        data1 = events.get_group((event1.file, event1.event, event1.rotation)).copy()
        data2 = events.get_group((event2.file, event2.event, event2.rotation)).copy()

        data1.loc[:, "cell_x"] += event1.xp
        data1.loc[:, "cell_y"] += event1.yp
        data2.loc[:, "cell_x"] += event2.xp
        data2.loc[:, "cell_y"] += event2.yp

        pair = ( pd.concat([data1, data2], ignore_index=True)
                   .groupby("cell_x cell_y".split(), as_index=False)
                   .sipm_hits
                   .sum()
               )
        cog_x = (event1.xb + event1.xp + event2.xb + event2.xp)/2
        cog_y = (event1.xb + event1.xp + event2.xb + event2.xp)/2

        pair.loc[:, "file1"    ] = event1.file
        pair.loc[:, "event1"   ] = event1.event
        pair.loc[:, "rotation1"] = event1.rotation
        pair.loc[:, "file2"    ] = event2.file
        pair.loc[:, "event2"   ] = event2.event
        pair.loc[:, "rotation2"] = event2.rotation
        pair.loc[:, "distance" ] = d
        pair.loc[:, "cell_x"   ] = pair.cell_x - cog_x
        pair.loc[:, "cell_y"   ] = pair.cell_y - cog_y
        pairs.append(pair)

    s    = np.tile(s, 2)
    cogs = cogs.loc[~s]

print()
print(f"{len(cogs)}/{events.ngroups} events could not be paired")

pairs = pd.concat(pairs, ignore_index=True)
pairs = pairs.loc[:, "file1 event1 rotation1 file2 event2 rotation2 cell_x cell_y sipm_hits".split()]
pairs = pairs.astype(dict(file1=int, event1=int, rotation1=int, file2=int, event2=int, rotation2=int))
pairs.to_hdf(output, "/data", complib="zlib", complevel=4, mode="w")
