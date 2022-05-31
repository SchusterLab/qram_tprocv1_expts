from audioop import avg
import numpy as np
import scipy as sp
import cmath

def expfunc(x, *p):
    return p[0]+p[1]*np.exp(-(x-p[2])/p[3])

def sinfunc(x, *p):
    return p[0]*np.sin(2*np.pi*p[1]*x + p[2]*np.pi/180) + p[3]

def decaysin(x, *p):
    return p[0]*np.sin(2*np.pi*p[1]*x + p[2]*np.pi/180) * np.exp(-(x-p[5])/p[3]) + p[4]

def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0]=ydata[-1]
    if fitparams[1] is None: fitparams[1]=ydata[0]-ydata[-1]
    if fitparams[2] is None: fitparams[2]=xdata[0]
    if fitparams[3] is None: fitparams[3]=(xdata[-1]-xdata[0])/5.
    pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
    return pOpt, pCov

def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    FFT = sp.fft.fft(ydata)
    fft_freqs = sp.fftpack.fftfreq(len(ydata), xdata[1] - xdata[0])
    max_ind = np.argmax(abs(FFT[4:int(len(ydata)/2)])) + 4
    fft_val = FFT[max_ind]
    if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
    if fitparams[1] is None: fitparams[1]=fft_freqs[max_ind]
    if fitparams[2] is None: fitparams[2]=(cmath.phase(fft_val)-np.pi/2.)*180./np.pi
    if fitparams[3] is None: fitparams[3]=np.mean(ydata)
    bounds = (
        [-5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, np.min(ydata)],
        [5*fitparams[0], 20/(max(xdata)-min(xdata)), 360, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]): fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
    pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
    return pOpt, pCov

def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*6
    FFT = sp.fft.fft(ydata)
    fft_freqs = sp.fftpack.fftfreq(len(ydata), xdata[1] - xdata[0])
    max_ind = np.argmax(abs(FFT[4:int(len(ydata)/2)])) + 4
    fft_val = FFT[max_ind]
    if fitparams[0] is None: fitparams[0]=(max(ydata)-min(ydata))/2
    if fitparams[1] is None: fitparams[1]=fft_freqs[max_ind]
    if fitparams[2] is None: fitparams[2]=(cmath.phase(fft_val)-np.pi/2.)*180./np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    if fitparams[5] is None: fitparams[5]=xdata[0]
    bounds = (
        [-5*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 1e-3, np.min(ydata), np.min(xdata)],
        [5*fitparams[0], 20*fitparams[1], 360, np.inf, np.max(ydata), 2*np.max(xdata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]): fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
    pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
    return pOpt, pCov
