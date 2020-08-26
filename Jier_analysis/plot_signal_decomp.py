import numpy as np
import math
import pywt as wv 
import matplotlib.pyplot as plt 

def plot_signal_decomp(data, w, mode, title, level):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    assert isinstance(w, wv.Wavelet)
    # w = wv.Wavelet(w)
    a = data
    ca = []
    cd = []
    level_ = wv.dwt_max_level(len(data), w.dec_len)
    if level > level_:
        level = level_
        print("Appropriate level is changed to ", level)
    else:
        level_ = None 
    for i in range(level):
        (a, d) = wv.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(wv.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(wv.waverec(coeff_list, w))

    fig = plt.figure(figsize=(16,9), dpi=120)
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylim(min(0, 1.4 * min(y)), max(0, 1.4 * max(y)))
        ax.set_ylabel("D%d" % (i + 1))
    plt.tight_layout()

def plot_modwt_default(data, w, title):

    assert isinstance(w, wv.Wavelet)
    a = data
    level_ = wv.swt_max_level(len(a))
    coeffs =  wv.swt(a, w, level=level_, trim_approx=True, norm=True)

    ylim = [a.min(), a.max()]

    _, axes = plt.subplots(len(coeffs) + 1, figsize=(19, 8), dpi=120)
    axes[0].set_title(title)
    axes[0].plot(a, 'k', label='Original signal')
    axes[0].set_ylabel('deg C')
    axes[0].set_xlim(0, len(a) - 1)
    axes[0].set_ylim(ylim[0], ylim[1])
    axes[0].legend(loc=0)

    for i, _ in enumerate(coeffs):
        ax = axes[-i - 1]
        if i == 0:
            ax.set_ylabel("A%d" % (len(coeffs) - 1))
            ax.plot( coeffs[i], 'r')
        else:
            ax.set_ylabel("D%d" % (len(coeffs) - i))
            ax.plot( coeffs[i], 'g')
        # Scale axes
        ax.set_xlim(0, len(a) - 1)
        # ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    # plt.show()

def plot_modwt(data, w, title, level):

    assert isinstance(w, wv.Wavelet)
    a = get_pad_data(data=data)
    # level_ = wv.swt_max_level(len(a))
    # if level_ > level:
    #     level = level_
    # else:
    #     level_ = None

    coeffs =  wv.swt(a, w, level=level, trim_approx=True, norm=True) #[(cAn, (cDn, ...,cDn-1, cD1)]

    ylim = [a.min(), a.max()]

    _, axes = plt.subplots(len(coeffs) + 1, figsize=(19, 8), dpi=120)
    axes[0].set_title(title)
    axes[0].plot(a, 'k', label='Original signal')
    axes[0].set_ylabel('deg in Celsius')
    axes[0].set_xlim(0, len(a) - 1)
    axes[0].set_ylim(ylim[0], ylim[1])
    axes[0].legend(loc=0)


    for i, _ in enumerate(coeffs):
        ax = axes[-i - 1]
        if i == 0:
            ax.set_ylabel("A%d" % (len(coeffs) - 1))
            ax.plot( coeffs[i], 'r')
        else:
            ax.set_ylabel("D%d" % (len(coeffs) - i))
            ax.plot( coeffs[i], 'g')
        # Scale axes
        ax.set_xlim(0, len(a) - 1)
        # ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    # plt.show()

def compare_signal_vs_smooth(index, col,  original_ts, smoothed_ts, save_fig=False):
    fig, ax = plt.subplots(2, 1, figsize=(19, 8), dpi=120)
    ax[0].plot(index, original_ts, 'k', label='Original')
    ax[0].legend(loc=0)
    ax[1].plot(index, smoothed_ts, 'r-', label='Smoothed swt')
    ax[1].legend(loc=0)
    fig.suptitle('Decomposition of original ts '+col+' with its decomposed approximation')
    if save_fig == True:
        plt.savefig('Wavelet/orignal_ts_vs_smooth_decomp'+col+'.pdf', dpi=120)
        plt.savefig('Wavelet/orignal_ts_vs_smooth_decomp'+col+'.png', dpi=120)
    # plt.show()

def get_pad_width(data):
    x  = np.log2(len(data))
    x = math.ceil(x) 
    diff = 2**x - len(data)
    return diff//2

def get_pad_data(data):
    pad_width = get_pad_width(data)
    return np.pad(data, pad_width=pad_width, mode='symmetric')

def shift_coeffs(coeffs, size, filter_length):
    N = size 
    L = filter_length
    T = np.eye(N, N, k=-1)
    T[0][-1] = 1
    J_0 = len(coeffs) -1
    smooth = (2**J_0 - 1)*(L - 2)/(2*(L -1 ))
    power = [- 2** j/ 2 for j in range(1, len(coeffs[1:]) + 1) ]
    print(power, T.shape, N, coeffs[0].shape, sep='\n')
    T_ = [np.linalg.matrix_power(T, -1 * int(power[j]) ) for j,_ in enumerate(coeffs[1:])]
    print('[INFO] Passed Transform')
    details = [ [np.dot(T_[j], w_j) for j, w_j in enumerate(coeffs[1:])] for _ in range(len(coeffs[1:]))]
    print('[INFO] Finish details')
    T_a = np.linalg.matrix_power(T, -1 * int(smooth) )
    approx = [np.dot(T_a, vj) for vj in coeffs[0]]
    print('[INFO] Finish approx')
    print(approx, coeffs[0], sep='\n')
    new_coeffs = approx.append(details)
    return new_coeffs
