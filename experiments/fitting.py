import cmath

import numpy as np
import scipy as sp

# ====================================================== #

"""
Compare the fit between the check_measures (amps, avgi, and avgq by default) in data, using the compare_param_i-th parameter to do the comparison. Pick the best method of measurement out of the check_measures, and return the fit, fit_err, and any other get_best_data_params corresponding to that measurement.

If fitfunc is specified, uses R^2 to determine best fit.
"""


def get_best_fit(
    data,
    fitfunc=None,
    prefixes=["fit"],
    check_measures=("amps", "avgi", "avgq"),
    get_best_data_params=(),
    override=None,
):
    fit_errs = [data[f"{prefix}_err_{check}"] for check in check_measures for prefix in prefixes]
    all_check_measures = [f"{prefix}_err_{check}" for check in check_measures for prefix in prefixes]

    # fix the error matrix so "0" error is adjusted to inf
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0:
                fit_err_check[i][i] = np.inf

    fits = [data[f"{prefix}_{check}"] for check in check_measures for prefix in prefixes]

    if override is not None and override in all_check_measures:
        i_best = np.argwhere(np.array(all_check_measures) == override)[0][0]
    else:
        if fitfunc is not None:
            ydata = [
                data[check] for check in all_check_measures
            ]  # need to figure out how to make this support multiple qubits readout
            xdata = data["xpts"]

            # residual sum of squares
            ss_res_checks = np.array(
                [
                    np.sum((fitfunc(xdata, *fit_check) - ydata_check) ** 2)
                    for fit_check, ydata_check in zip(fits, ydata)
                ]
            )
            # total sum of squares
            ss_tot_checks = np.array([np.sum((np.mean(ydata_check) - ydata_check) ** 2) for ydata_check in ydata])
            # R^2 value
            r2 = 1 - ss_res_checks / ss_tot_checks

            # override r2 value if fit is bad
            for icheck, fit_err_check in enumerate(fit_errs):
                for i, fit_err in enumerate(np.diag(fit_err_check)):
                    if fit_err == np.inf:
                        r2[icheck] = np.inf
            i_best = np.argmin(r2)

        else:
            fit_errs = np.array(fit_errs)
            fits = np.array(fits)
            errs = []
            for fit, fit_err in zip(fits, fit_errs):
                if np.any(np.abs(fit_err) == np.inf):
                    errs.append(np.nan)
                else:
                    errs.append(np.average(np.sqrt(np.abs(np.diag(fit_err))) / np.abs(fit)))
            for i_err, err in enumerate(errs):
                if np.isnan(err):
                    errs[i_err] = np.inf
            i_best = np.argmin(errs)
            print("best", i_best, errs[i_best])

    best_data = [fits[i_best], fit_errs[i_best]]

    for param in get_best_data_params:
        assert len(fit_errs) == len(
            check_measures
        ), "this is a pathological use of this function anyway, so just restrict to these cases"
        # best_meas = all_check_measures[i_best]
        best_meas = check_measures[i_best]
        best_data.append(data[f"{param}_{best_meas}"])
    return best_data


# ====================================================== #


def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = ydata[-1]
    if fitparams[1] is None:
        fitparams[1] = ydata[0] - ydata[-1]
    if fitparams[2] is None:
        fitparams[2] = xdata[0]
    if fitparams[3] is None:
        fitparams[3] = (xdata[-1] - xdata[0]) / 5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


def logexpfunc(x, *p):
    decay = p
    return np.log(np.exp(-x / decay))


def fitlogexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (xdata[-1] - xdata[0]) / 5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(logexpfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #

# see https://www.science.org/doi/epdf/10.1126/science.aah5844
# assumes function has been scaled to log scale and goes from 0 to 1
def qp_expfunc(x, *p):
    nqp, t1qp, t1r = p
    return nqp * (np.exp(-x / t1qp) - 1) - x / t1r


def fitqpexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = 1.0
    if fitparams[1] is None:
        fitparams[1] = (xdata[-1] - xdata[0]) / 2
    if fitparams[2] is None:
        fitparams[2] = (xdata[-1] - xdata[0]) / 2
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = ([0.01, 0.01 * fitparams[1], 0.01 * fitparams[2]], [5, 2 * fitparams[1], 10 * fitparams[2]])
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        pOpt, pCov = sp.optimize.curve_fit(qp_expfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
def gaussian(x, a, x0, sigma, y0):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + y0


def fit_gauss(xdata, ydata, fitparams=None):
    # xmed, xstd should be gotten from the single shot data prior to fitting the histogram
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.max(ydata)
    if fitparams[1] is None:
        fitparams[1] = xdata[np.argmax(ydata)]
    if fitparams[2] is None:
        fitparams[2] = (np.max(xdata) - np.min(xdata)) / 8
    if fitparams[3] is None:
        fitparams[3] = np.median(ydata)
    pOpt = fitparams
    # print('fitparams guess:', fitparams)
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = (
        [fitparams[0] * 0.5, np.min(xdata), fitparams[2] * 0.1, 0],
        [fitparams[0] * 1.5, np.max(xdata), fitparams[2] * 10, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            gaussian, xdata, ydata, p0=np.array(fitparams, dtype="float64"), bounds=bounds
        )
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1**2)) + a2 * np.exp(-((x - b2) ** 2) / (2 * c2**2))


def fit_doublegauss(xdata, ydata, fitparams=None):
    # xmed, xstd should be gotten from the single shot data prior to fitting the histogram
    if fitparams is None:
        fitparams = [None] * 6
    else:
        fitparams = np.copy(fitparams)
    midpoint = np.average((np.max(xdata), np.min(xdata)))
    if fitparams[0] is None:
        fitparams[0] = np.max(ydata[: len(ydata) // 2])
    if fitparams[1] is None:
        fitparams[1] = xdata[np.argmax(ydata[: len(ydata) // 2])]
    if fitparams[2] is None:
        fitparams[2] = (midpoint - np.min(xdata)) / 4
    if fitparams[3] is None:
        fitparams[3] = np.max(ydata[len(ydata) // 2 :])
    if fitparams[4] is None:
        fitparams[4] = xdata[len(ydata) // 2 + np.argmax(ydata[len(ydata) // 2 :])]
    if fitparams[5] is None:
        fitparams[5] = (np.max(xdata) - midpoint) / 4
    pOpt = fitparams
    # print('fitparams guess:', fitparams)
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = (
        [fitparams[0] * 0.95, np.min(xdata), fitparams[2] * 0.2, fitparams[3] * 0.95, midpoint, fitparams[5] * 0.2],
        [fitparams[0] * 1.05, midpoint, fitparams[2] * 2, fitparams[3] * 1.05, np.max(xdata), fitparams[5] * 2],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            double_gaussian, xdata, ydata, p0=np.array(fitparams, dtype="float64"), bounds=bounds
        )
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


def nlorfunc(x, npeaks, *p):
    y0, yscale, x0, xscale, nbar, chi = p
    result = y0
    for i in range(2 * npeaks):
        yscale_i = yscale * poisson(i, nbar)
        result += yscale_i / (1 + (x - (x0 - chi * i)) ** 2 / xscale**2)
    return result


def fitnlor(xdata, ydata, npeaks, f0_guess, chi_guess, fitparams=None):
    # give a postive value for chi_guess to use a negative chi
    if fitparams is None:
        fitparams = [None] * 6
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = f0_guess
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 20
    if fitparams[4] is None:
        fitparams[4] = npeaks / 2
    if fitparams[5] is None:
        fitparams[5] = chi_guess
    pOpt = fitparams
    # print('fitparams guess:', fitparams)
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = (
        [
            min(ydata),
            0.1 * fitparams[1],
            f0_guess - chi_guess,
            0.1 * fitparams[3],
            0,
            0.25 * chi_guess,
        ],
        [max(ydata), 10 * fitparams[1], max(xdata), 10 * fitparams[3], npeaks, 4 * chi_guess],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        # print('npeaks', npeaks)
        fitnlorfunc = lambda x, *p: nlorfunc(x, npeaks, *p)
        pOpt, pCov = sp.optimize.curve_fit(fitnlorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0


def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    max_ind = np.argmax(np.abs(fourier[1:])) + 1
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None:
        fitparams[1] = max_freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = np.mean(ydata)
    bounds = (
        [0.5 * fitparams[0], 0.2 / (max(xdata) - min(xdata)), -360, np.min(ydata)],
        [2 * fitparams[0], 5 / (max(xdata) - min(xdata)), 360, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def quadraticfunc(x, *p):
    x0, y0, yscale = p
    return yscale * (x - x0) ** 2 + y0


def fitquadratic(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    # else:
    #     fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.mean(xdata)
    if fitparams[1] is None:
        fitparams[1] = np.mean(ydata)
    if fitparams[2] is None:
        fitparams[2] = (np.max(ydata) - np.min(ydata)) / (np.max(xdata) - np.min(xdata)) ** 2
    # bounds = (
    #     [0.5*fitparams[0], 0.2/(max(xdata)-min(xdata)), -360, np.min(ydata)],
    #     [2*fitparams[0], 5/(max(xdata)-min(xdata)), 360, np.max(ydata)]
    #     )
    # for i, param in enumerate(fitparams):
    #     if not (bounds[0][i] < param < bounds[1][i]):
    #         fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
    #         print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    bounds = None
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(quadraticfunc, xdata, ydata, p0=fitparams)  # , bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def rabifunc(x, *p):
    length, omega, f0, scale, z0 = p
    sqrt_part = 1 + (x - f0) ** 2 / omega**2
    factor = omega**2 / (omega**2 + (x - f0) ** 2)
    return z0 + scale * factor * np.sin(2 * np.pi * omega * np.sqrt(sqrt_part) * length / 2) ** 2


def fitrabi_gainslice(xdata, ydata, length, fitparams=None):
    """
    Fit a single slice through frequency at a single gain; length is given. xdata is frequency, ydata is measured signal
    fitparams will be length, omega, f0, scale, z0
    """
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = np.copy(fitparams)

    partitioned = np.partition(ydata, -2)
    largest = partitioned[-1]
    freq_largest = xdata[np.argwhere(ydata == largest)[0][0]]
    second_largest = partitioned[-2]
    freq_second_largest = xdata[np.argwhere(ydata == second_largest)[0][0]]
    guess_freq = np.mean([freq_largest, freq_second_largest])
    if fitparams[0] is None:
        fitparams[0] = length
    if fitparams[1] is None:
        fitparams[1] = 5
    if fitparams[2] is None:
        fitparams[2] = guess_freq
    if fitparams[3] is None:
        fitparams[3] = np.max(ydata) - np.min(ydata)
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)
    bounds = (
        [0.5 * fitparams[0], 0.1, min(xdata), 0.01 * fitparams[3], min(ydata)],
        [2 * fitparams[0], 100, max(xdata), 100 * fitparams[3], max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(rabifunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) * np.exp(-x / decay) + y0


def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = (max(ydata) - min(ydata)) / 2
    if fitparams[1] is None:
        fitparams[1] = max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)
    bounds = (
        [0.75 * fitparams[0], 0.1 / (max(xdata) - min(xdata)), -360, 0.3 * (max(xdata) - min(xdata)), np.min(ydata)],
        [1.25 * fitparams[0], 30 / (max(xdata) - min(xdata)), 360, np.inf, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #


def twofreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0 = p
    return y0 + np.exp(-x / decay0) * yscale0 * (
        (1 - yscale1) * np.sin(2 * np.pi * freq0 * x + phase_deg0 * np.pi / 180)
        + yscale1 * np.sin(2 * np.pi * freq1 * x + phase_deg1 * np.pi / 180)
    )


def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 10
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale0
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq0
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg0
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # exp decay
    if fitparams[4] is None:
        fitparams[4] = 0.1  # yscale1
    if fitparams[5] is None:
        fitparams[5] = 0.5  # MHz
    if fitparams[6] is None:
        fitparams[6] = 0  # phase_deg1
    if fitparams[7] is None:
        fitparams[7] = np.mean(ydata)  # y0
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.1 * (max(xdata) - min(xdata)),
            0.001,
            0.01,
            -360,
            np.min(ydata),
        ],
        [1.25 * fitparams[0], 30 / (max(xdata) - min(xdata)), 360, np.inf, 0.5, 10, 360, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(twofreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


def threefreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01, yscale2, freq2, phase_deg2, y02 = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    p2 = [yscale2, freq2, phase_deg2, y02]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1) * sinfunc(x, *p2)


def fitthreefreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 14
    else:
        fitparams = np.copy(fitparams)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None:
        fitparams[1] = max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  #
    if fitparams[5] is None:
        fitparams[5] = xdata[0]  # x0 (exp decay)
    if fitparams[6] is None:
        fitparams[6] = 1  # y scale
    if fitparams[7] is None:
        fitparams[7] = 1 / 10  # MHz
    if fitparams[8] is None:
        fitparams[8] = 0  # phase degrees
    if fitparams[9] is None:
        fitparams[9] = 0  # y0
    if fitparams[10] is None:
        fitparams[10] = 1  # y scale
    if fitparams[11] is None:
        fitparams[11] = 1 / 10  # MHz
    if fitparams[12] is None:
        fitparams[12] = 0  # phase degrees
    if fitparams[13] is None:
        fitparams[13] = 0  # y0
    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
            xdata[0] - (xdata[-1] - xdata[0]),
            0.5,
            0.01,
            -360,
            -0.1,
            0.5,
            0.01,
            -360,
            -0.1,
        ],
        [
            1.25 * fitparams[0],
            15 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.max(ydata),
            xdata[-1] + (xdata[-1] - xdata[0]),
            1.5,
            10,
            360,
            0.1,
            1.5,
            10,
            360,
            0.1,
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(threefreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #

# def hangerfunc(x, *p):
#     f0, Qi, Qe, phi, scale, a0 = p
#     Q0 = 1 / (1/Qi + np.real(1/Qe))
#     return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

# def hangerS21func(x, *p):
#     f0, Qi, Qe, phi, scale, a0 = p
#     Q0 = 1 / (1/Qi + np.real(1/Qe))
#     return a0 + np.abs(hangerfunc(x, *p)) - scale*(1-Q0/Qe)

# def hangerS21func_sloped(x, *p):
#     f0, Qi, Qe, phi, scale, a0, slope = p
#     return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope*(x-f0)


def hangerfunc(x, *p):
    """
    See https://arxiv.org/pdf/1909.01487
    """
    f0, Qi, Qe, phi = p
    Q0 = 1 / (1 / Qi + np.real(np.exp(1j * phi) / Qe))
    return 1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0)  # - (1-Q0/Qe)


def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, slope, a0 = p
    return np.abs(hangerfunc(x, f0, Qi, Qe, phi)) + slope * (x - f0) + a0


def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))


def fithanger(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 6
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(xdata)
    if fitparams[1] is None:
        fitparams[1] = np.average(xdata) / 0.2
    if fitparams[2] is None:
        fitparams[2] = np.average(xdata) / 0.2
    if fitparams[3] is None:
        fitparams[3] = 0
    if fitparams[4] is None:
        fitparams[4] = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    if fitparams[5] is None:
        fitparams[5] = np.mean(ydata)

    print("param guess", fitparams)

    # bounds = (
    #     [np.min(xdata), -1e9, -1e9, -2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))/10, -np.max(np.abs(ydata))],
    #     [np.max(xdata), 1e9, 1e9, 2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))*10, np.max(np.abs(ydata))]
    #     )
    bounds = (
        [np.min(xdata), 0, 0, -2 * np.pi, -1.5 * abs((ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])), -np.inf],
        [
            np.max(xdata),
            1e9,
            1e8,
            2 * np.pi,
            1.5 * abs((ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])),
            np.max(ydata),
        ],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    print("param bounds", bounds)

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds)
        print("popt", pOpt)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #

# Standard RB decay fit function
def rb_func(depth, p, a, offset):
    return a * p**depth + offset


# Gives the average error rate over all gates in sequence
def rb_error(p, d):  # d = dim of system = 2^(number of qubits)
    return 1 - (p + (1 - p) / d)


# return covariance of rb error
def error_fit_err(cov_p, d):
    return cov_p * (1 / d - 1) ** 2


# Run both regular RB and interleaved RB to calculate this
# Get error from here: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.080505
def rb_gate_fidelity(p_rb, p_irb, d, p_rb_err=None, p_irb_err=None):
    fid = 1 - (d - 1) * (1 - p_irb / p_rb) / d
    if p_rb_err is not None and p_irb_err is not None:
        # fid_err_sqrd = ((d-1)/d * (1/p_rb) * p_irb_err)**2 + ((d-1)/d * p_irb/(p_rb**2) * p_rb_err)**2
        # fid_err = np.sqrt(fid_err_sqrd)

        fid_err1 = (d - 1) * (np.abs(p_rb - p_irb / p_rb) + (1 - p_rb)) / d
        fid_err2 = 2 * (d**2 - 1) * (1 - p_rb) / (p_rb * d**2) + 4 * np.sqrt(1 - p_rb) * np.sqrt(d**2 - 1) / p_rb
        return fid, np.min((fid_err1, fid_err2))
    return fid


def fitrb(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = 0.9
    if fitparams[1] is None:
        fitparams[1] = np.max(ydata) - np.min(ydata)
    if fitparams[2] is None:
        fitparams[2] = np.min(ydata)
    bounds = ([0, 0, 0], [1, 10 * np.max(ydata) - np.min(ydata), np.max(ydata)])
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(rb_func, xdata, ydata, p0=fitparams, bounds=bounds)
        # print(pOpt)
        # print(pCov[0][0], pCov[1][1], pCov[2][2])
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# See Wood and Gambetta 2017 section III
def leakage_err(p, offset):
    """L1 error"""
    return (1 - offset) * (1 - p)


def seepage_err(p, offset):
    """L2 error"""
    return offset * (1 - p)


def rb_decay_l1_l2(depth, p1, a0, b0, c0, p2):
    """
    Fit decay of g state in subspace when accounting for leakage and seepage
    p1 should be fixed by fitting using the regular rb decay
    """
    return a0 + b0 * p1**depth + c0 * p2**depth


def rb_fidelity_l1_l2(d, p2, l1, p2_err=None, l1_err=None):  # d = dim of subspace
    fid = (1 / d) * ((d - 1) * p2 + 1 - l1)
    if p2_err is not None and l1_err is not None:
        fid_err_sqrd = ((1 / d) * (d - 1) * p2_err) ** 2 + ((1 / d) * l1_err) ** 2
        fid_err = np.sqrt(fid_err_sqrd)
        return fid, fid_err
    return fid


# offset should be passed in from fit using regular rb decay
def fitrb_l1_l2(xdata, ydata, p1, offset=0.5, fitparams=None, sigma=None):
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = 0.5 * offset  # a0
    if fitparams[1] is None:
        fitparams[1] = 0.5  # b0
    if fitparams[2] is None:
        fitparams[2] = 0.5  # c0
    if fitparams[3] is None:
        fitparams[3] = 0.9  # p2
    bounds = ([0, 0, 0, 0], [min((1.2 * offset, 1)), 1, 1, 1])
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            lambda depth, a0, b0, c0, p2: rb_decay_l1_l2(depth, p1, a0, b0, c0, p2),
            xdata,
            ydata,
            sigma=sigma,  # use the standard error std/sqrt(n)
            p0=fitparams,
            bounds=bounds,
        )
        # print(pOpt)
        # print(pCov[0][0], pCov[1][1], pCov[2][2])
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
# Adiabatic pi pulse functions
# beta ~ slope of the frequency sweep (also adjusts width)
# mu ~ width of frequency sweep (also adjusts slope)
# period: delta frequency sweeps through zero at period/2
# amp_max


def adiabatic_amp(t, amp_max, beta, period):
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t, mu, beta, period):
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(t, amp_max, mu, beta, period):
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j * np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j * np.cos(phase))
    return np.real(iamp), np.real(qamp)


# ====================================================== #
# Correcting for over/under rotation
# delta: angle error in degrees
# See reference: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.012301


def probg_Xhalf(n, *p):
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta))


def probg_X(n, *p):
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * np.cos(np.pi / 2 + 2 * n * delta))


def probg_Xhalf_decay(n, *p):
    a, delta, decay = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta)) * np.exp(-n / decay)


def fit_probg_Xhalf(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(probg_Xhalf, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


def fit_probg_X(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(probg_X, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


def fit_probg_Xhalf_decay(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    if fitparams[2] is None:
        fitparams[2] = 10
    bounds = (
        [min(ydata), -20.0, 1],
        [max(ydata), 20.0, 100],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(probg_Xhalf_decay, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
def poisson(n, *p):
    nbar = p[0]
    return np.exp(-nbar) * (nbar**n) / sp.special.factorial(n)


def fit_poisson(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = ydata[0]
    bounds = (
        [0],
        [10],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(poisson, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except (RuntimeError, ValueError) as e:
        print("Warning: fit failed with error:", e)
        # return 0, 0
    return pOpt, pCov
