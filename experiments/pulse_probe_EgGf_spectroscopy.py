import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

class PulseProbeEgGfSpectroscopyProgram(RAveragerProgram):
    """
    Qubit A: E<->G
    Qubit B: g<->f
    Drive applied on qubit B
    """
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits
        
        # all of these saved self.whatever instance variables should be indexed by the actual qubit number. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.readout.frequency, self.res_chs)]
        self.readout_length = [self.us2cycles(len) for len in self.cfg.device.readout.readout_length]

        for q in self.qubits:
            self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q])
            self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q])
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_length[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # drive is applied on qubit B
        self.r_freqB=self.sreg(self.qubit_chs[qB], "freq") # get frequency register for qubit_ch 
        self.r_freqB_update = 4 # register to hold the current sweep frequency
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB])

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qB])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qB])

        # send start frequency to r_freqB_update
        self.safe_regwi(self.q_rps[qB], self.r_freqB_update, self.f_start)
        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)

        for q in self.qubits:
            self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res[q], phase=self.deg2reg(cfg.device.readout.phase[q], gen_ch=self.res_chs[q]), gain=cfg.device.readout.gain[q], length=self.readout_length[q])
           
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits
        
        # initialize qubit A to E: expect to end in Eg
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_ge[qA], gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all()

        # apply Eg -> Gf pulse on B: expect to end in Gf
        self.set_pulse_registers(
            ch=self.qubit_chs[qB],
            style="const",
            freq=0, # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length))
        self.mathi(self.q_rps[qB], self.r_freqB, self.r_freqB_update, "+", 0)
        self.pulse(ch=self.qubit_chs[qB])
        self.sync_all()

        # take qubit A G->E and qubit B f->e: expect to end in Ee (or Gg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", freq=self.f_ge[qA], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")

        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns
        self.measure(
            pulse_ch=self.res_chs, 
            adcs=[0,1],
            adc_trig_offset=cfg.device.readout.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))
    
    def update(self):
        qA, qB = self.qubits
        self.mathi(self.q_rps[qB], self.r_freqB_update, self.r_freqB_update, '+', self.f_step) # update frequency list index
        

class PulseProbeEgGfSpectroscopyExperiment(Experiment):
    """
    Pulse Probe Eg-Gf Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEgGfSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits
        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]

        qspec_ef=PulseProbeEgGfSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
        
        data=dict(
            xpts=x_pts,
            avgi=(avgi[adcA_ch][0], avgi[adcB_ch][0]),
            avgq=(avgq[adcA_ch][0], avgq[adcB_ch][0]),
            amps=(np.abs(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                  np.abs(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
            phases=(np.angle(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                    np.angle(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
        )
        self.data=data
        return data

    def analyze(self, data=None, fit=True, sign=[[1, 1], [1, 1]], **kwargs):
        # sign of fit: [iA, qA], [iB, qB]
        if data is None: data=self.data
        self.sign = sign
        if fit:
            data['fitA_avgi']=dsfit.fitlor(data["xpts"], sign[0][0]*data['avgi'][0])
            data['fitA_avgq']=dsfit.fitlor(data["xpts"], sign[0][1]*data['avgq'][0])
            data['fitB_avgi']=dsfit.fitlor(data["xpts"], sign[1][0]*data['avgi'][1])
            data['fitB_avgq']=dsfit.fitlor(data["xpts"], sign[1][1]*data['avgq'][1])
        return data

    def display(self, data=None, fit=True, sign=None, **kwargs):
        # sign of fit: [iA, qA], [iB, qB]
        if data is None: data=self.data 
        if sign is None: sign = self.sign
        plt.figure(figsize=(14,8))
        plt.suptitle(f"Pulse Probe Eg-Gf Spectroscopy")

        plt.subplot(221, title='Qubit A', ylabel="I [adc level]")
        plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"], sign[0][0]*dsfit.lorfunc(data["fitA_avgi"], data["xpts"]))
            print(f'Found peak in avgi data (qubit A) at [MHz] {data["fitA_avgi"][2]}, HWHM {data["fitA_avgi"][3]}')
        plt.subplot(223, xlabel="Pulse Frequency [MHz]", ylabel="Q [adc levels]")
        plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"], sign[0][1]*dsfit.lorfunc(data["fitA_avgq"], data["xpts"]))
            print(f'Found peak in avgq data (qubit A) at [MHz] {data["fitA_avgq"][2]}, HWHM {data["fitA_avgq"][3]}')


        plt.subplot(222, title='Qubit B')
        plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"], sign[1][0]*dsfit.lorfunc(data["fitB_avgi"], data["xpts"]))
            print(f'Found peak in avgi data (qubit B) at [MHz] {data["fitB_avgi"][2]}, HWHM {data["fitB_avgi"][3]}')
        plt.subplot(224, xlabel="Pulse Frequency [MHz]")
        plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"], sign[1][1]*dsfit.lorfunc(data["fitB_avgq"], data["xpts"]))
            print(f'Found peak in avgq data (qubit B) at [MHz] {data["fitB_avgq"][2]}, HWHM {data["fitB_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)