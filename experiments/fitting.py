from audioop import avg
import numpy as np
import scipy as sp
import cmath

# ====================================================== #

"""
Compare the fit between the check_measures (amps, avgi, and avgq by default) in data, using the compare_param_i-th parameter to do the comparison. Pick the best method of measurement out of the check_measures, and return the fit, fit_err, and any other get_best_data_params corresponding to that measurement.
"""
def get_best_fit(data, compare_param_i, check_measures=('amps', 'avgi', 'avgq'), get_best_data_params=(), override=None):
    fit_errs = [data[f'fit_err_{check}'] for check in check_measures]
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0: fit_err_check[i][i] = np.inf
    fits = [data[f'fit_{check}'] for check in check_measures]
    if override is not None and override in check_measures:
        i_best = np.argwhere(np.array(check_measures) == override)[0][0]
        print(i_best)
    else:
        # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
        # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i] / fit[compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
        i_best = np.argmin([np.average(np.sqrt(np.abs(np.diag(fit_err) / fit))) for fit, fit_err in zip(fits, fit_errs)])
    best_meas = check_measures[i_best]

    best_data = [fits[i_best], fit_errs[i_best]]
    for param in get_best_data_params:
        best_data.append(data[f'{param}_{best_meas}'])
    return best_data

# ====================================================== #

def expfunc(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale*np.exp(-(x-x0)/decay)

def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[2] = xdata[0]
    if fitparams[3] is None: fitparams[3] = (xdata[-1]-xdata[0])/5
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale/(1+(x-x0)**2/xscale**2)

def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1])/2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(ydata)]
    if fitparams[3] is None: fitparams[3] = (max(xdata)-min(xdata))/10
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) + y0

def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    max_ind = np.argmax(np.abs(fourier[1:])) + 1
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=np.mean(ydata)
    bounds = (
        [0.5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, np.min(ydata)],
        [2*fitparams[0], 10/(max(xdata)-min(xdata)), 360, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #

def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0, x0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-(x-x0)/decay) + y0

def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*6
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    if fitparams[5] is None: fitparams[5]=xdata[0]
    bounds = (
        [0.5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.1, np.min(ydata), xdata[0]-(xdata[-1]-xdata[0])],
        [5*fitparams[0], 15/(max(xdata)-min(xdata)), 360, np.inf, np.max(ydata), xdata[-1]+(xdata[-1]-xdata[0])]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov

# ====================================================== #
    
def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale, a0 = p
    return a0 + np.abs(hangerfunc(x, *p))

def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))

def fithanger(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*6
    if fitparams[0] is None: fitparams[0]=np.average(xdata)
    if fitparams[1] is None: fitparams[1]=5000
    if fitparams[2] is None: fitparams[2]=1000
    if fitparams[3] is None: fitparams[3]=0
    if fitparams[4] is None: fitparams[4]=max(np.abs(ydata))-min(np.abs(ydata))
    if fitparams[5] is None: fitparams[5]=np.average(np.abs(ydata))
    print(fitparams)
    # bounds = (
    #     [np.min(xdata), -1e9, -1e9, -2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))/10, -np.max(np.abs(ydata))],
    #     [np.max(xdata), 1e9, 1e9, 2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))*10, np.max(np.abs(ydata))]
    #     )
    # for i, param in enumerate(fitparams):
    #     if not (bounds[0][i] < param < bounds[1][i]):
    #         fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
    #         print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func, xdata, ydata, p0=fitparams) #, bounds=bounds)
        print(pOpt)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        # return 0, 0
    return pOpt, pCov