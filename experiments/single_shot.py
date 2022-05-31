import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

def hist(data, plot=True, span=None, verbose=True):
    """
    span: histogram limit is the mean +/- span
    """
    Ig = data['Ig']
    Qg = data['Qg']
    Ie = data['Ie']
    Qe = data['Qe']
    plot_f = False
    if 'If' in data.keys():
        plot_f = True
        If = data['If']
        Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    if verbose: print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)}')

    xe, ye = np.median(Ie), np.median(Qe)
    if verbose: print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)}')

    if plot_f:
        xf, yf = np.median(If), np.median(Qf)
        if verbose: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)}')

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        fig.tight_layout()

        axs[0].scatter(Ig, Qg, label='g', color='b', marker='.')
        axs[0].scatter(Ie, Qe, label='e', color='r', marker='.')
        if plot_f: axs[0].scatter(If, Qf, label='f', color='g', marker='.')
        axs[0].scatter(xg, yg, color='k', marker='o')
        axs[0].scatter(xe, ye, color='k', marker='o')
        if plot_f: axs[0].scatter(xf, yf, color='k', marker='o')

        axs[0].set_xlabel('I [ADC levels]')
        axs[0].set_ylabel('Q [ADC levels]')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated')
        axs[0].axis('equal')
    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)

    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [xg-span, xg+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.')
        axs[1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.')
        if plot_f: axs[1].scatter(If_new, Qf_new, label='f', color='g', marker='.')
        axs[1].scatter(xg, yg, color='k', marker='o')
        axs[1].scatter(xe, ye, color='k', marker='o')    
        if plot_f: axs[1].scatter(xf, yf, color='k', marker='o')    

        axs[1].set_xlabel('I [ADC levels]')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Rotated')
        axs[1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[2].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[2].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        if plot_f: nf, binsf, pf = axs[2].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
        axs[2].set_xlabel('I [ADC levels]')       
        axs[2].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range = xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range = xlims)

    """Compute the fidelity using overlap of the histograms"""
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    threshold=binsg[tind]
    fid = contrast[tind]
    if plot: 
        axs[2].set_title(f"Readout Fidelity: {fid*100:.2f}%")
        axs[2].axvline(threshold, color='0.2', linestyle='--')

    return fid, threshold, theta*180/np.pi

"""
Measures the single shot readout fidelity of the system. We acquire single shot (I, Q) readout values by first preparing the qubit in its ground (blue dots) a certain number of times (in the below demo we take 5000 shots) and then preparing the qubit in its excited state (red dots) the same number of times. We then extract two parameters which are used to optimize the associated readout fidelity: the rotation angle of the IQ blobs and the threshold that classifies the two qubit states (ground and excited). We store these two parameters here <code>cfg.device.readouti.phase</code> and <code>cfg.device.readouti.threshold</code>.

Note that this experiment already assumes that you have found your qubit frequency and $\pi$ pulse amplitude. Every time you reset the QICK firmware the single shot angle and threshold changes. So, this experiment is used to calibrate any experiment below that uses single shot data (such as the Active Reset experiment).
"""
class HistogramProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        q_ind = self.cfg.expt.qubit
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q_ind]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q_ind]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q_ind])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q_ind])
        
        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res=self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch) # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        # add qubit and readout pulses to respective channels
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            # phase=0,
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)

        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all()
        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            self.sync_all()
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg=AttrDict(self.cfg)
        shots_i0 = self.di_buf[0] / self.readout_length
        shots_q0 = self.dq_buf[0] / self.readout_length
        shots_i1 = self.di_buf[1] / self.readout_length
        shots_q1 = self.dq_buf[1] / self.readout_length
        return shots_i0, shots_q0, shots_i1, shots_q1


class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_f: whether to also test the f state blob
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        for key, value in self.cfg.device.readout.items():
            if isinstance(value, list):
                self.cfg.device.readout.update({key: value[q_ind]})
        for key, value in self.cfg.device.qubit.items():
            if isinstance(value, list):
                self.cfg.device.qubit.update({key: value[q_ind]})
            elif isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if isinstance(value3, list):
                            value2.update({key3: value3[q_ind]})
            

        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind] 

        # rounds = 100
        # Ig = np.zeros(self.cfg.expt.reps)
        # Qg = np.zeros(self.cfg.expt.reps)
        # Ie = np.zeros(self.cfg.expt.reps)
        # Qe = np.zeros(self.cfg.expt.reps)
        # for r in tqdm(range(rounds)):
        #     x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False, debug=debug)
        #     i0, q0, i1, q1 = histpro.collect_shots()
        #     iq = ([i0, q0], [i1, q1])
        #     i, q = iq[adc_ch] # i/q[0]: ground state i/q, i/q[1]: excited state i/q
        #     Ig += i[0]
        #     Qg += q[0]
        #     Ie += i[1]
        #     Qe += q[1]

        data=dict()

        # Ground state shots
        cfg = AttrDict(self.cfg.copy())
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
        i0, q0, i1, q1 = histpro.collect_shots()
        iq = ([i0, q0], [i1, q1])
        i, q = iq[adc_ch]
        data['Ig'], data['Qg'] = iq[adc_ch]

        # Excited state shots
        cfg = AttrDict(self.cfg.copy())
        cfg.expt.pulse_e = True 
        cfg.expt.pulse_f = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
        i0, q0, i1, q1 = histpro.collect_shots()
        iq = ([i0, q0], [i1, q1])
        i, q = iq[adc_ch]
        data['Ie'], data['Qe'] = iq[adc_ch]

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            i0, q0, i1, q1 = histpro.collect_shots()
            iq = ([i0, q0], [i1, q1])
            i, q = iq[adc_ch]
            data['If'], data['Qf'] = iq[adc_ch]

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data=self.data
        
        fid, threshold, angle = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fid'] = fid
        data['angle'] = angle
        data['threshold'] = threshold
        
        return data

    
    def display(self, data=None, span=None, **kwargs):
        if data is None:
            data=self.data 
        
        fid, threshold, angle = hist(data=data, plot=True, span=span)
            
        print(f'fidelity: {fid}')
        print(f'rotation angle (deg): {angle}')
        print(f'set angle to (deg): {self.cfg.device.readout.phase - angle}')
        print(f'threshold: {threshold}')
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

class SingleShotOptExperiment(Experiment):
    """
    Single Shot optimization experiment over readout parameters
    expt = dict(
        reps: number of shots per expt
        start_f: start frequency (MHz)
        step_f: frequency step (MHz)
        expts_f: number of experiments in frequency

        start_gain: start gain (dac units)
        step_gain: gain step (dac units)
        expts_gain: number of experiments in gain sweep

        start_len: start readout len (dac units)
        step_len: length step (dac units)
        expts_len: number of experiments in length sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"]*np.arange(self.cfg.expt["expts_len"])
        
        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))

        for f_ind, f in enumerate(tqdm(fpts, disable=not progress)):
            for g_ind, gain in enumerate(gainpts):
                for l_ind, l in enumerate(lenpts):
                    shot = HistogramExperiment(soccfg=self.soccfg, config_file=self.config_file)
                    shot.cfg.device.readout.frequency = f
                    shot.cfg.device.readout.gain = gain
                    shot.cfg.device.readout.length = l 
                    shot.cfg.expt = dict(reps=self.cfg.expt.reps, check_f=False, qubit=self.cfg.expt.qubit)
                    shot.go(analyze=False, display=False, progress=False, save=False)
                    results = shot.analyze(verbose=False)
                    fid[f_ind, g_ind, l_ind] = results['fid']
                    threshold[f_ind, g_ind, l_ind] = results['threshold']
                    angle[f_ind, g_ind, l_ind] = results['angle']

        self.data = dict(fpts=fpts, gainpts=gainpts, lenpts=lenpts, fid=fid, threshold=threshold, angle=angle)
        return self.data

    def analyze(self, data=None, **kwargs):
        if data == None: data = self.data
        fid = data['fid']
        threshold = data['threshold']
        angle = data['angle']
        fpts = data['fpts']
        gainpts = data['gainpts']
        lenpts = data['lenpts']

        imax = np.unravel_index(np.argmax(fid), shape=fid.shape)
        print(imax)
        print(fpts)
        print(gainpts)
        print(lenpts)
        print(f'Max fidelity {fid[imax]}')
        print(f'Set params: \n angle (deg) {self.cfg.device.readout.phase[self.cfg.expt.qubit] - angle[imax]} \n threshold {threshold[imax]} \n freq [Mhz] {fpts[imax[0]]} \n gain [dac units] {gainpts[imax[1]]} \n readout length [us] {lenpts[imax[2]]}')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
