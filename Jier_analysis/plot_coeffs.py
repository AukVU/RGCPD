import numpy as np 
import matplotlib.pyplot as plt 
import pywt as wv 

def plot_coeffs(data, w, mode, title, level, use_dwt=True):
    """Show dwt or swt coefficients for given data and wavelet."""
    w = wv.Wavelet(w)
    a = data
    ca = []
    cd = []
    if use_dwt:
        level_ = wv.dwt_max_level(len(data), w.dec_len)
    else:
        level_ = wv.swt_max_level(len(data))
    if level > level_:
        level = level_
        print("Appropriate level is changed to ", level)
    else:
        level_ = None 
    if use_dwt:
        for i in range(level):
            (a, d) = wv.dwt(a, w, mode) #  [(cAn, (cDn, ...,cDn-1, cD1)]
            ca.append(a)
            cd.append(d)
    else:
        coeffs = wv.swt(data, w, level)  # [(cA5, cD5), ..., (cA1, cD1)]
        for a, d in reversed(coeffs):
            ca.append(a)
            cd.append(d)

    fig = plt.figure(figsize=(16,9))
    ax_main = fig.add_subplot(len(ca) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, x in enumerate(ca):
        ax = fig.add_subplot(len(ca) + 1, 2, 3 + i * 2)
        ax.plot(x, 'r')
        ax.set_ylabel("A%d" % (i + 1))
        if use_dwt:
            ax.set_xlim(0, len(x) - 1)
        else:
            ax.set_xlim(w.dec_len * i, len(x) - 1 - w.dec_len * i)

    for i, x in enumerate(cd):
        ax = fig.add_subplot(len(cd) + 1, 2, 4 + i * 2)
        ax.plot(x, 'g')
        ax.set_ylabel("D%d" % (i + 1))
        # Scale axes
        ax.set_xlim(0, len(x) - 1)
        if use_dwt:
            ax.set_ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))
        else:
            vals = x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)]
            ax.set_ylim(min(0, 2 * min(vals)), max(0, 2 * max(vals)))
        plt.tight_layout()
