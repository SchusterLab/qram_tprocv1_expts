import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo2QProgram, sort_counts, correct_readout_err, fix_neg_counts

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs: # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

    def set_gen_delays(self):
        for ch in self.gen_chs:
            delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]]
            delay_cycles = self.us2cycles(delay_ns*1e-3, gen_ch=ch)
            self.gen_delays[ch] = delay_cycles

    def sync_all(self, t=0):
        super().sync_all(t=t, gen_t0=self.gen_delays)


    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF
        if self.checkEF:
            if 'pulse_ge' not in self.cfg.expt: self.pulse_ge = True
            else: self.pulse_ge = self.cfg.expt.pulse_ge

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        
        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
        else: qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
            self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap.mixer_freq

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        if self.checkZZ:
            if qTest == 1: self.f_Q1_ZZ_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q1_ZZ]
            else: self.f_Q_ZZ1_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q_ZZ1]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.f_f0g1_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_f0g1, self.qubit_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # declare all res dacs
        self.measure_chs = []
        mask = [] # indices of mux_freqs, mux_gains list to play
        mux_mixer_freq = None
        mux_freqs = [0]*4 # MHz
        mux_gains = [0]*4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ['full', 'mux4']
            if self.res_ch_types[q] == 'full':
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q])
                    self.measure_chs.append(self.res_chs[q])
                
            elif self.res_ch_types[q] == 'mux4':
                assert self.res_chs[q] == 6
                mask.append(q)
                if mux_mixer_freq is None: mux_mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                else: assert mux_mixer_freq == cfg.hw.soc.dacs.readout.mixer_freq[q] # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = cfg.device.readout.frequency[q]
                mux_gains[q] = cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
        if 'mux4' in self.res_ch_types: # declare mux4 channel
            self.declare_gen(ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch)

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_ch_types[q] == 'int4':
                    mixer_freq = mixer_freqs[q]
                if self.swap_chs[q] not in self.gen_chs: 
                    self.declare_gen(ch=self.swap_chs[q], nqz=self.cfg.hw.soc.dacs.swap.nyquist[q], mixer_freq=mixer_freq)

        # define pi_test_sigma as the pulse that we are calibrating, update in outer loop over averager program
        self.pi_test_sigma = self.us2cycles(cfg.expt.length_placeholder, gen_ch=self.qubit_chs[qTest])
        if 'test_pi_half' in self.cfg.expt and self.cfg.expt.test_pi_half:
            self.pi_test_sigma = self.pi_test_sigma // 2
        self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate
        if 'gain' in self.cfg.expt: self.gain_pi_test = self.cfg.expt.gain 
        else: self.gain_pi_test = self.cfg.device.qubit.pulses.pi_ge.gain[qTest] # gain of the pulse we are trying to calibrate
        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qZZ], gen_ch=self.qubit_chs[qZZ])
            if qTest == 1:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q1_ZZ_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q1_ZZ_reg[qSort] # freq we are trying to calibrate
                if 'gain' not in self.cfg.expt: self.gain_pi_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qSort] # gain of the pulse we are trying to calibrate
            else:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q_ZZ1_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q_ZZ1_reg[qSort] # freq we are trying to calibrate
                if 'gain' not in self.cfg.expt: self.gain_pi_test = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[qSort] # gain of the pulse we are trying to calibrate
        if self.checkEF:
            self.f_pi_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate
            if 'gain' not in self.cfg.expt: self.gain_pi_test = self.cfg.device.qubit.pulses.pi_ef.gain[qTest] # gain of the pulse we are trying to calibrate


        # add qubit pulses to respective channels
        if cfg.expt.pulse_type.lower() == "gauss" and self.pi_test_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma*4)
        if self.checkZZ:
            self.add_gauss(ch=self.qubit_chs[qZZ], name="pi_qubitA", sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ*4)
        if self.checkEF:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            for q in self.cfg.expt.cool_qubits:
                self.pisigma_ef = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[q], gen_ch=self.qubit_chs[q]) # default pi_ef value
                self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=self.pisigma_ef, length=self.pisigma_ef*4)
                if self.cfg.device.qubit.pulses.pi_f0g1.type[q] == 'flat_top':
                    self.add_gauss(ch=self.swap_chs[q], name=f"pi_f0g1_{q}", sigma=3, length=3*4)
                else: assert False, 'not implemented'
        if 'n_pulses' in self.cfg.expt and self.cfg.expt.n_pulses is not None: # add pihalf initialization pulse for error amplification
            self.pi_test_half_sigma = self.us2cycles(cfg.expt.length_placeholder, gen_ch=self.qubit_chs[qTest]) // 2
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test_half", sigma=self.pi_test_half_sigma, length=self.pi_test_half_sigma*4)


        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))

        self.set_gen_delays()
        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)
        if self.checkZZ: qZZ, qTest = self.qubits
        else: qTest = self.qubits[0]

        self.reset_and_sync()

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            for q in self.cfg.expt.cool_qubits:
                self.setup_and_pulse(ch=self.qubit_chs[q], style="arb", phase=0, freq=self.f_ef_reg[q], gain=cfg.device.qubit.pulses.pi_ef.gain[q], waveform=f"pi_ef_qubit{q}")
                self.sync_all(5)

                pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[q]
                pisigma_f0g1 = self.us2cycles(self.cfg.device.qubit.pulses.pi_f0g1.sigma[q], gen_ch=self.swap_chs[q])
                if pulse_type == 'flat_top':
                    sigma_ramp_cycles = 3
                    flat_length_cycles = pisigma_f0g1 - sigma_ramp_cycles*4
                    self.setup_and_pulse(ch=self.swap_chs[q], style="flat_top", freq=self.f_f0g1_reg[q], phase=0, gain=self.cfg.device.qubit.pulses.pi_f0g1.gain[q], length=flat_length_cycles, waveform=f"pi_f0g1_{q}")
                else: assert False, 'not implemented'
                self.sync_all()
            self.sync_all(self.us2cycles(self.cfg.expt.cool_idle))

        # initializations as necessary
        if self.checkZZ:
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.f_ge_reg[qZZ], gain=cfg.device.qubit.pulses.pi_ge.gain[qZZ], waveform="pi_qubitA")
            self.sync_all(5)
        if self.checkEF and self.pulse_ge:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")
            self.sync_all(5)

        # play pi pulse that we want to calibrate
        if self.pi_test_sigma > 0:
            if 'n_pulses' in self.cfg.expt and self.cfg.expt.n_pulses is not None:
                n_pulses = self.cfg.expt.n_pulses
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_pi_test_reg, phase=0, gain=self.gain_pi_test, waveform="pi_test_half")
                self.sync_all()
            else: n_pulses = 0.5
            for i in range(int(2*n_pulses)):
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_pi_test_reg, phase=0, gain=self.gain_pi_test, waveform="pi_test")
                self.sync_all()

        if self.checkEF: # map excited back to qubit ground state for measurement
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")

        # align channels and measure
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in range(4)]))
        )

    """ Collect shots for all adcs, rotates by given angle (degrees), separate based on threshold (if not None), and averages over all shots (i.e. returns data[num_chs, 1] as opposed to data[num_chs, num_shots]) if requested.
    Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    """
    def get_shots(self, angle=None, threshold=None, avg_shots=False, verbose=False, return_err=False):
        buf_len = len(self.di_buf[0])

        if angle is None: angle = [0]*len(self.cfg.device.qubit.f_ge)
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufi = np.array([bufi[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        if threshold is not None: # categorize single shots
            bufi = np.array([np.heaviside(bufi[ch] - threshold[ch], 0) for ch in range(len(self.adc_chs))])
        avgi = np.average(bufi, axis=1) # [num_chs]
        bufi_err = np.std(bufi, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufi[i]) for i in range(4)])

        bufq = np.array([
            self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufq = np.array([bufq[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgq = np.average(bufq, axis=1) # [num_chs]
        bufq_err = np.std(bufq, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufq[i]) for i in range(4)])

        if avg_shots:
            idata = avgi
            qdata = avgq
        else:
            idata = bufi
            qdata = bufq

        if return_err: return idata, qdata, bufi_err, bufq_err
        else: return idata, qdata 

    """
    If post_process == 'threshold': uses angle + threshold to categorize shots into 0 or 1 and calculate the population
    If post_process == 'scale': uses angle + ge_avgs to scale the average of all shots on a scale of 0 to 1. ge_avgs should be of shape (num_total_qubits, 4) and should represent the pre-rotation Ig, Qg, Ie, Qe
    If post_process == None: uses angle to rotate the i and q and then returns the avg i and q
    """
    def acquire_rotated(self, soc, progress, angle=None, threshold=None, ge_avgs=None, post_process=None, verbose=False):
        avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress)
        if post_process == None: 
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)
            if angle is None: return avgi_rot, avgq_rot
            else: return avgi_rot, avgi_err
        elif post_process == 'threshold':
            assert threshold is not None
            popln, avgq_rot, popln_err, avgq_err = self.get_shots(angle=angle, threshold=threshold, avg_shots=True, verbose=verbose, return_err=True)
            return popln, popln_err
        elif post_process == 'scale':
            assert ge_avgs is not None
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)

            ge_avgs_rot = [None]*4
            for q, angle_q in enumerate(angle):
                if not isinstance(ge_avgs[q], (list, np.ndarray)): continue # this qubit was not calibrated
                Ig_q, Qg_q, Ie_q, Qe_q = ge_avgs[q]
                ge_avgs_rot[q] = [
                    Ig_q*np.cos(np.pi/180*angle_q) - Qg_q*np.sin(np.pi/180*angle_q),
                    Ie_q*np.cos(np.pi/180*angle_q) - Qe_q*np.sin(np.pi/180*angle_q)
                ]
            shape = None
            for q in range(4):
                if ge_avgs_rot[q] is not None:
                    shape = np.shape(ge_avgs_rot[q])
                    break
            for q in range(4):
                if ge_avgs_rot[q] is None: ge_avgs_rot[q] = np.zeros(shape=shape)
                
            ge_avgs_rot = np.asarray(ge_avgs_rot)
            avgi_rot -= ge_avgs_rot[:,0]
            avgi_rot /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            avgi_err /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            return avgi_rot, avgi_err
        else:
            assert False, 'Undefined post processing flag, options are None, threshold, scale'


# ====================================================== #

class LengthRabiExperiment(Experiment):
    """
    Length Rabi Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step, 
        expts: number of different length experiments, 
        reps: number of reps,
        gain: gain to use for the qubit pulse
        pulse_type: 'gauss' or 'const'
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabi', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, fit_func='decaysin'):
        if data is None:
            data=self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            xdata = data['xpts']
            fitparams = None
            if fit_func == 'sin': fitparams=[None]*4
            elif fit_func == 'decaysin': fitparams=[None]*5
            fitparams[1]=2.0/xdata[-1]
            if fit_func == 'decaysin': fit_fitfunc = fitter.fitdecaysin
            elif fit_func == 'sin': fit_fitfunc = fitter.fitsin
            p_avgi, pCov_avgi = fit_fitfunc(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fit_fitfunc(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fit_fitfunc(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=True, fit_func='decaysin'):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3
        if fit_func == 'decaysin': fit_func = fitter.decaysin
        elif fit_func == 'sin': fit_func = fitter.sinfunc

        plt.figure(figsize=(10, 5))
        plt.subplot(111, title=f"Length Rabi", xlabel="Length [ns]", ylabel="Amplitude [ADC units]")
        plt.plot(xpts_ns[:-1], data["amps"][:-1],'.-')
        if fit:
            p = data['fit_amps']
            plt.plot(xpts_ns[:-1], fit_func(data["xpts"][:-1], *p))

        plt.figure(figsize=(10,8))
        if 'gain' in self.cfg.expt: gain = self.cfg.expt.gain
        else: gain = self.cfg.device.qubit.pulses.pi_ge.gain[self.cfg.expt.qubits[-1]] # gain of the pulse we are trying to calibrate
        plt.subplot(211, title=f"Length Rabi (Qubit Gain {gain})", ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], data["avgi"][1:-1],'.-')
        if fit:
            p = data['fit_avgi']
            plt.plot(xpts_ns[0:-1], fit_func(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            if fit_func == 'decaysin': print('Decay from avgi [us]', p[3])
            print(f'Pi length from avgi data [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        
        print()
        plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], data["avgq"][1:-1],'.-')
        if fit:
            p = data['fit_avgq']
            plt.plot(xpts_ns[0:-1], fit_func(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            if fit_func == 'decaysin': print('Decay from avgq [us]', p[3])
            print(f'Pi length from avgq data [us]: {pi_length}')
            print(f'Pi/2 length from avgq data [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #

class NPulseExperiment(Experiment):
    """
    Play a pi/2 or pi pulse variable N times 
    Experimental Config
    expt = dict(
        start: start N [us],
        step
        expts 
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        test_pi_half: calibrate the pi/2 instead of pi pulse by dividing length cycles // 2
    )
    """

    def __init__(self, soccfg=None, path='', prefix='NPulseExpt', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        cycles = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], 'counts_calib':[]}

        self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF
        self.qubits = self.cfg.expt.qubits
        
        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
            self.qZZ = qZZ
        else: qTest = self.qubits[0]
        self.qTest = qTest

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'post_process' not in self.cfg.expt.keys(): # threshold or scale
            self.cfg.expt.post_process = None
            assert False, "you probably want to be doing this experiment with post processing or the fit will be weird"

        if self.cfg.expt.post_process is not None:
            if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and 'counts_calib' in self.cfg.expt and self.cfg.expt.angles is not None and self.cfg.expt.thresholds is not None and self.cfg.expt.ge_avgs is not None and self.cfg.expt.counts_calib is not None:
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data['counts_calib'] = self.cfg.expt.counts_calib
                print('Re-using provided angles, thresholds, ge_avgs')
            else:
                thresholds_q = [0]*4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0]*4
                fids_q = [0]*4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.tomo_qubits = [qTest, (qTest+1)%4]

                calib_prog_dict = dict()
                calib_order = ['gg', 'eg']
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state:err_tomo})

                g_prog = calib_prog_dict['gg']
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict['eg']
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f'Qubit  ({qTest})')
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [np.average(Ig[qTest]), np.average(Qg[qTest]), np.average(Ie[qTest]), np.average(Qe[qTest])]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}')


                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_calib'].append(counts)

                if debug:
                    print(f'thresholds={thresholds_q},')
                    print(f'angles={angles_q},')
                    print(f'ge_avgs={ge_avgs_q},')
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

            data['thresholds'] = thresholds_q
            data['angles'] = angles_q
            data['ge_avgs'] = ge_avgs_q
            data['counts_calib'] = np.array(data['counts_calib'])

        # ================= #
        # Begin actual experiment
        # ================= #
        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        length = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest] # length of pulse whose error we are trying to find
        if self.checkZZ:
            if qTest == 1: length = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort]
            else: length = self.cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort]
        if self.checkEF:
            length = self.cfg.device.qubit.pulses.pi_ef.sigma[qTest]

        self.cfg.expt.length_placeholder = float(length)

        if 'loops' not in self.cfg.expt: self.cfg.expt.loops = 1
        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for n_cycle in tqdm(cycles, disable=not progress or self.cfg.expt.loops > 1):

                self.cfg.expt.n_pulses = n_cycle
                lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = lengthrabi
                avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        
                avgi = avgi[qTest]
                avgq = avgq[qTest]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["amps"].append(amp)
                data["phases"].append(phase)
        data["xpts"] = cycles

        for k, a in data.items():
            data[k]=np.array(a)

        data['avgi'] = np.average(np.reshape(data['avgi'], (self.cfg.expt.loops, len(cycles))), axis=0)
        data['avgq'] = np.average(np.reshape(data['avgq'], (self.cfg.expt.loops, len(cycles))), axis=0)
        data['amps'] = np.average(np.reshape(data['amps'], (self.cfg.expt.loops, len(cycles))), axis=0)
        data['phases'] = np.average(np.reshape(data['phases'], (self.cfg.expt.loops, len(cycles))), axis=0)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment

        self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF
        self.qubits = self.cfg.expt.qubits
        
        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
            self.qZZ = qZZ
        else: qTest = self.qubits[0]
        self.qTest = qTest

        if data is None:
            data=self.data
        if fit:
            xdata = data['xpts']
            fitparams = None
            if self.cfg.expt.test_pi_half: fit_fitfunc = fitter.fit_probg_Xhalf
            else: fit_fitfunc = fitter.fit_probg_X
            if scale is not None:
                Ig, Qg, Ie, Qe = scale[self.qTest]
                reformatted_scale = [(Ig, Ie), (Qg, Qe), (np.abs(Ig+1j*Qg), np.abs(Ie+1j*Qe))]
            for fit_i, fit_axis in enumerate(['avgi', 'avgq', 'amps']):
                if scale is not None:
                    g_avg, e_avg = reformatted_scale[fit_i]
                    fit_data = (data[fit_axis] - g_avg) / (e_avg - g_avg)
                p, pCov = fit_fitfunc(xdata, fit_data, fitparams=fitparams)
                data[f'fit_{fit_axis}'] = p
                data[f'fit_err_{fit_axis}'] = pCov
        return data

    def display(self, data=None, fit=True, scale=None):
        if data is None:
            data=self.data 

        xdata = data['xpts']
        if self.cfg.expt.test_pi_half: fit_func = fitter.probg_Xhalf
        else: fit_func = fitter.probg_X

        title = f"Angle Error Q{self.qTest}" + (f' ZZ Q{self.qZZ}' if self.checkZZ else '') + (' EF' if self.checkEF else '') + (' $\pi/2$' if self.cfg.expt.test_pi_half else ' $\pi$')

        if scale is not None:
            Ig, Qg, Ie, Qe = scale[self.qTest]
            reformatted_scale = [(Ig, Ie), (Qg, Qe), (np.abs(Ig+1j*Qg), np.abs(Ie+1j*Qe))]

        plt.figure(figsize=(10, 5))
        label = '($X_{\pi/2}, X_{'+ ('\pi' if not self.cfg.expt.test_pi_half else '\pi/2') + '}^{2n}$)'
        plt.subplot(111, title=title, xlabel=f"Number repeated gates {label} [n]", ylabel="Amplitude (scaled)")
        plot_data = data['amps']
        if scale is not None:
            g_avg, e_avg = reformatted_scale[2]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data,'.-')
        if fit:
            p = data['fit_amps']
            pCov = data['fit_err_amps']
            captionStr = f'$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half: amp_ratio = (90 + p[1])/90
            else: amp_ratio = (180 - p[1])/180
            print(f'From amps: adjust amplitude to (current gain) / {amp_ratio}')
        plt.show()

        plt.figure(figsize=(10,8))
        if 'gain' in self.cfg.expt: gain = self.cfg.expt.gain
        else: gain = self.cfg.device.qubit.pulses.pi_ge.gain[self.cfg.expt.qubits[-1]] # gain of the pulse we are trying to calibrate
        plt.subplot(211, title=title, ylabel="I (scaled)")
        plot_data = data['avgi']
        if scale is not None:
            g_avg, e_avg = reformatted_scale[0]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data,'.-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half: amp_ratio = (90 + p[1])/90
            else: amp_ratio = (180 - p[1])/180
            print(f'From avgi: adjust amplitude to (current gain) / {amp_ratio}')
        print()
        
        label = '($X_{\pi/2}, X_{'+ ('\pi' if not self.cfg.expt.test_pi_half else '\pi/2') + '}^{2n}$)'
        plt.subplot(212, xlabel=f"Number repeated gates {label} [n]", ylabel="Q (scaled)")
        plot_data = data['avgq']
        if scale is not None:
            g_avg, e_avg = reformatted_scale[1]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data,'.-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half: amp_ratio = (90 + p[1])/90
            else: amp_ratio = (180 - p[1])/180
            print(f'From avgq: adjust amplitude to (current gain) / {amp_ratio}')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname