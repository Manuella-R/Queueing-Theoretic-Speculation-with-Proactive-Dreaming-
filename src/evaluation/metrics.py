import numpy as np
import pandas as pd

def summarize(history):
    df = pd.DataFrame(history)
    cols = ["lat_p50","lat_p95","lat_p99","rho_tau_est","idle_budget","dream_cpu_secs","reward"]
    summary = df[cols].describe().loc[["mean","50%","max","min"]]
    return df, summary

def mttr_proxy(df):
    # naive: spikes in lat_p95 considered incidents; measure decay length back to median
    lat = df["lat_p95"].values
    med = np.nanmedian(lat)
    thr = med * 1.5
    mttrs = []
    i = 0
    n = len(lat)
    while i < n:
        if lat[i] > thr:
            start = i
            while i < n and lat[i] > thr:
                i += 1
            end = i
            mttrs.append(end - start)
        else:
            i += 1
    return float(np.mean(mttrs)) if mttrs else 0.0
