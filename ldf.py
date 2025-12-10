import numpy as np
from sklearn.neighbors import KernelDensity

def local_dependence(x, y, bandwidth=0.1, grid_points=100):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    # X için yoğunluk tahmini
    kde_x = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
    log_px = kde_x.score_samples(x)
    px = np.exp(log_px)

    # Birlikte yoğunluk tahmini (X,Y)
    xy = np.hstack([x, y])
    kde_xy = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy)
    log_pxy = kde_xy.score_samples(xy)
    pxy = np.exp(log_pxy)

    # LDF ≈ p(x,y) / p(x)*p(y)
    # >1 → pozitif bağımlılık güçleniyor
    # <1 → bağımlılık zayıf
    kde_y = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(y)
    log_py = kde_y.score_samples(y)
    py = np.exp(log_py)

    ldf_vals = pxy / (px * py)
    return ldf_vals
