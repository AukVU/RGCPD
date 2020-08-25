#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import pywt
import pywt.data

ecg = pywt.data.ecg()
# set trim_approx to avoid keeping approximation coefficients for all levels

# set norm=True to rescale the wavelets so that the transform partitions the
# variance of the input signal among the various coefficient arrays.

coeffs = pywt.swt(ecg, wavelet='sym4', trim_approx=True, norm=True)
ca = coeffs[0]
details = coeffs[1:]

s = ecg
cA = []
cD = []
sym4 = pywt.Wavelet('sym4')
sym4_normalized = pywt.Wavelet(
    'sym4_normalized',
    filter_bank=[np.asarray(f)/np.sqrt(2) for f in sym4.filter_bank]
)
lvl_decomp = pywt.dwt_max_level(len(ecg), sym4_normalized.dec_len)
for i in range(lvl_decomp): # Using recursion to overwrite signal to go level deepeer
    s, d =  pywt.dwt(s, sym4_normalized, mode='periodic')
    cA.append(s)
    cD.append(d)
coeff = pywt.wavedec(ecg, sym4_normalized, mode=pywt.Modes.periodic)
wavelet = pywt.Wavelet('sym4')
# phi, psi, x = wavelet.wavefun(level=lvl_decomp)
# print(f'phi {len(phi)}  psi  {len(psi)}  x {len(x)}')
d_var = [np.var(c[1:-1], ddof=1) for c in coeff]


print("Variance of the ecg signal = {}".format(np.var(ecg, ddof=1)))
# print(len(coeffs), len(coeff), len(cA), len(cD))

var_cA = [np.var(c, ddof=1) for _, c in enumerate(cA)]
# var_test = [np.dot(c, c)/len(c) for i, c in enumerate(cA)]
var_cD = [np.var(c, ddof=1) for c in cD]
variances = [np.var(c, ddof=1) for c in coeffs]
detail_variances = variances[1:]
print(f'Sum of variance decomposition function WAVEDEC {np.sum(d_var)}')
print("Sum of variance across all SWT coefficients = {}".format(
    np.sum(variances)))
print(f'Variance SWT details {np.sum(detail_variances)} and APProx variancve {np.sum(variances[0])}')
print(f'Variance WAVEDEC Details {np.sum(d_var[1:])} and approx variance {np.sum(d_var[0])}')
print(f'Sum of variance DWT Approx {np.sum(var_cA)}, Details {np.sum(var_cD)}, Total {np.sum(var_cA) + np.sum(var_cD)}')
# print(f'Sum of variance with numpy {np.sum(var_test)/2}')
sys.exit()

# Create a plot using the same y axis limits for all coefficient arrays to
# illustrate the preservation of amplitude scale across levels when norm=True.
ylim = [ecg.min(), ecg.max()]

fig, axes = plt.subplots(len(coeffs) + 1)
axes[0].set_title("normalized SWT decomposition")
axes[0].plot(ecg)
axes[0].set_ylabel('ECG Signal')
axes[0].set_xlim(0, len(ecg) - 1)
axes[0].set_ylim(ylim[0], ylim[1])

for i, x in enumerate(coeffs):
    ax = axes[-i - 1]
    ax.plot(coeffs[i], 'g')
    if i == 0:
        ax.set_ylabel("A%d" % (len(coeffs) - 1))
    else:
        ax.set_ylabel("D%d" % (len(coeffs) - i))
    # Scale axes
    ax.set_xlim(0, len(ecg) - 1)
    ax.set_ylim(ylim[0], ylim[1])


# reorder from first to last level of coefficients
level = np.arange(1, len(detail_variances) + 1)

# create a plot of the variance as a function of level
plt.figure(figsize=(8, 6))
fontdict = dict(fontsize=16, fontweight='bold')
plt.plot(level, detail_variances[::-1], 'k.')
plt.xlabel("Decomposition level", fontdict=fontdict)
plt.ylabel("Variance", fontdict=fontdict)
plt.title("Variances of detail coefficients", fontdict=fontdict)
plt.show()