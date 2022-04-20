import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

"""
Measures the single shot readout fidelity of the system. We acquire single shot (I, Q) readout values by first preparing the qubit in its ground (blue dots) a certain number of times (in the below demo we take 5000 shots) and then preparing the qubit in its excited state (red dots) the same number of times. We then extract two parameters which are used to optimize the associated readout fidelity: the rotation angle of the IQ blobs and the threshold that classifies the two qubit states (ground and excited). We store these two parameters here <code>cfg.device.readouti.phase</code> and <code>cfg.device.readouti.threshold</code>.

Note that this experiment already assumes that you have found your qubit frequency and $\pi$ pulse amplitude. Every time you reset the QICK firmware the single shot angle and threshold changes. So, this experiment is used to calibrate any experiment below that uses single shot data (such as the Active Reset experiment).
"""
class HistogramProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        q_ind = self.cfg.expt.qubit
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q_ind]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q_ind]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q_ind])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q_ind])
        
        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_gain = self.sreg(self.qubit_ch, "gain")
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_res=self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch) # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        # print(self.sigma)
        
        # copy over parameters for inherited methods
        self.cfg.start = 0
        self.cfg.step = self.pi_gain
        self.cfg.expts = 2
        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.f_ge,
            phase=0,
            gain=0, # initialize gain to 0
            waveform="pi_qubit")
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.pi_gain)  # update frequency list index
        
    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg=AttrDict(self.cfg)
        shots_i0 = self.di_buf[0].reshape((cfg.expts, cfg.expt.reps)) / self.readout_length
        shots_q0 = self.dq_buf[0].reshape((cfg.expts, cfg.expt.reps)) / self.readout_length
        shots_i1 = self.di_buf[1].reshape((cfg.expts, cfg.expt.reps)) / self.readout_length
        shots_q1 = self.dq_buf[1].reshape((cfg.expts, cfg.expt.reps)) / self.readout_length
        return shots_i0, shots_q0, shots_i1, shots_q1
        
        
class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots
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
            

        histpro = HistogramProgram(soccfg=self.soccfg, cfg=self.cfg)
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind] 

        # rounds = 100
        # ig = np.zeros(self.cfg.expt.reps)
        # qg = np.zeros(self.cfg.expt.reps)
        # ie = np.zeros(self.cfg.expt.reps)
        # qe = np.zeros(self.cfg.expt.reps)
        # for r in tqdm(range(rounds)):
        #     x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False, debug=debug)
        #     i0, q0, i1, q1 = histpro.collect_shots()
        #     iq = ([i0, q0], [i1, q1])
        #     i, q = iq[adc_ch] # i/q[0]: ground state i/q, i/q[1]: excited state i/q
        #     ig += i[0]
        #     qg += q[0]
        #     ie += i[1]
        #     qe += q[1]

        x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)
        i0, q0, i1, q1 = histpro.collect_shots()
        iq = ([i0, q0], [i1, q1])
        i, q = iq[adc_ch] # i/q[0]: ground state i/q, i/q[1]: excited state i/q
        data=dict()
        # data['ig'] = ig / rounds
        # data['qg'] = qg / rounds
        # data['ie'] = ie / rounds
        # data['qe'] = qe / rounds
        data['ig'] = i[0]
        data['qg'] = q[0]
        data['ie'] = i[1]
        data['qe'] = q[1]
        
        self.data = data
        return data
        
    def analyze(self, data=None, span=40, **kwargs):
        if data is None:
            data=self.data
        
        fid, threshold, angle = self.hist(data=data, plot=False, span=span)
        data['fid'] = fid
        data['angle'] = angle
        data['threshold'] = threshold
        
        return data

    
    def display(self, data=None, span=None, **kwargs):
        if data is None:
            data=self.data 
        
        fid, threshold, angle = self.hist(data=data, plot=True, span=span)
            
        print(f'fidelity: {fid}')
        print(f'angle: {angle}')
        print(f'threshold: {threshold}')
        
        plt.tight_layout()
        plt.show()
    
    
    def hist(self, data=None, plot=True, span=1.0):
        """
        span: histogram limit is the mean +/- span
        """
        if data is None:
            data=self.data 
        ig = data['ig']
        qg = data['qg']
        ie = data['ie']
        qe = data['qe']

        numbins = 200

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        print('Ig', xg, 'Ig std dev', np.std(ig), 'Ie', xe, 'Qg', yg, 'Qe', ye)

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0].scatter(ig, qg, label='g', color='b', marker='.')
            axs[0].scatter(ie, qe, label='e', color='r', marker='.')
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xe, ye, color='k', marker='o')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((ye-yg),(xe-xg))
        """Rotate the IQ data"""
        ig_new = ig*np.cos(theta) - qg*np.sin(theta)
        qg_new = ig*np.sin(theta) + qg*np.cos(theta) 
        ie_new = ie*np.cos(theta) - qe*np.sin(theta)
        qe_new = ie*np.sin(theta) + qe*np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)
        print('Ig', xg, 'Ig std dev', np.std(ig), 'Ie', xe, 'Qg', yg, 'Qe', ye)

        xlims = [xg-span, xg+span]
        ylims = [yg-span, yg+span]

        if plot:
            axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='.')
            axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='.')
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xe, ye, color='k', marker='o')    
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title('Rotated')
            axs[1].axis('equal')

            """X and Y ranges for histogram"""

            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
            axs[2].set_xlabel('I(a.u.)')       

        else:        
            ng, binsg = np.histogram(ig_new, bins=numbins, range = xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range = xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
        tind=contrast.argmax()
        threshold=binsg[tind]
        fid = contrast[tind]
        if plot: axs[2].set_title(f"Readout Fidelity: {fid*100:.2f}%")

        return fid, threshold, theta
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)