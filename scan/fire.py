import numpy  as np
import pandas as  pd


def metadata(filename):
    _, p, _, elgap, _, dfh, _, dah, _, fileno = filename.stem.split("_")
    return dict( pitch  = float(p)
               , elgap  = float(elgap)
               , dfh    = float(dfh)
               , dah    = float(dah)
               , fileno =   int(fileno))


def file_number(filename):
    return metadata(filename)["fileno"]


def remove_subfile(filename):
    return "_".join(filename.stem.split("_")[:-2]) + filename.suffix


def barycenter(df, rel_thr=0.02):
    df      = df.loc[df.tpb_hits > rel_thr*df.tpb_hits.max()]
    weights = df.tpb_hits.values / df.tpb_hits.sum()
    xb = np.sum(weights * df.cell_x.values)
    yb = np.sum(weights * df.cell_y.values)
    return pd.Series(dict(xb=xb, yb=yb))


def bin_centers(x):
    return x[:-1] + np.diff(x)/2
