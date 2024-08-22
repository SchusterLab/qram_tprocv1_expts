import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import scipy as sp
import matplotlib.pyplot as plt

import experiments.fitting as fitter

# ====================================================== #

class AmplitudeRabiProgram(RAveragerProgram):
    # mu, beta are dimensionless
    def add_adiabatic(self, ch, name, mu, beta, period_us, length_us):
        period = self.us2cycles(period_us, gen_ch=ch)
        length = self.us2cycles(length_us, gen_ch=ch)

        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale']
        samps_per_clk = gencfg['samps_per_clk']
        length = np.round(length) * samps_per_clk
        period *= samps_per_clk
        t = np.arange(0, length)
        iamp, qamp = fitter.adiabatic_iqamp(t, amp_max=1, mu=mu, beta=beta, period=period)
        self.add_pulse(ch=ch, name=name, idata=maxv*iamp, qdata=maxv*qamp)

    # I_mhz_vs_us, Q_mhz_vs_us = functions of time in us, in units of MHz
    # times_us = times at which I_mhz_vs_us and Q_mhz_vs_us are defined
    def add_IQ(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us):
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale'] - 1
        samps_per_clk = gencfg['samps_per_clk']
        times_cycles = np.linspace(0, self.us2cycles(times_us[-1], gen_ch=ch), len(times_us))
        times_samps = samps_per_clk * times_cycles
        IQ_scale = max((np.max(np.abs(I_mhz_vs_us)), np.max(np.abs(Q_mhz_vs_us))))
        I_func = sp.interpolate.interp1d(times_samps, I_mhz_vs_us/IQ_scale, kind='linear', fill_value='extrapolate')
        Q_func = sp.interpolate.interp1d(times_samps, Q_mhz_vs_us/IQ_scale, kind='linear', fill_value='extrapolate')
        t = np.arange(0, np.round(times_samps[-1]))
        iamps = I_func(t)
        qamps = Q_func(t)
        plt.plot(iamps, '.-')
        # plt.plot(times_samps, I_func(times_samps), '.-')
        plt.plot(qamps, '.-')
        # plt.plot(times_samps, Q_func(times_samps), '.-')
        plt.show()
        self.add_pulse(ch=ch, name=name, idata=maxv*iamps, qdata=maxv*qamps)        

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
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

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF
        if self.checkEF:
            if 'pulse_ge' not in self.cfg.expt: self.pulse_ge = True
            else: self.pulse_ge = self.cfg.expt.pulse_ge

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.swap_f0g1_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
            self.swap_f0g1_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_f0g1.mixer_freq

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs

        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (4,4))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (4,4))
        self.pi_ge_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.gain, (4,4))
        self.pi_ge_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4,4))
        self.pi_ge_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain, (4,4))
        self.pi_ge_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain_pi_sigma, (4,4))
        self.pi_ef_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.gain, (4,4))
        self.pi_ef_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4,4))
        self.pi_ef_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain, (4,4))
        self.pi_ef_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain_pi_sigma, (4,4))

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.f_f0g1_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_f0g1, self.qubit_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # add swap pulse
        if 'apply_EgGf' in self.cfg.expt and self.cfg.expt.apply_EgGf:
            assert 'qubits_EgGf' in self.cfg.expt
            assert 'qDrive' in self.cfg.expt

            qA, qB = self.cfg.expt.qubits_EgGf

            qSort = qA
            if qA == 1: qSort = qB
            qDrive = 1
            if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
                qDrive = self.cfg.expt.qDrive
            qNotDrive = -1
            if qA == qDrive: qNotDrive = qB
            else: qNotDrive = qA
            self.qDrive = qDrive
            self.qNotDrive = qNotDrive
            self.qSort = qSort

            if qDrive == 1:
                self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
                self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
                mixer_freqs = self.cfg.hw.soc.dacs.swap.mixer_freq
                self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf[qSort], gen_ch=self.swap_chs[qSort])
                self.gain_EgGf = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort]
                self.type_EgGf = self.cfg.device.qubit.pulses.pi_EgGf.type[qSort]
                self.sigma_EgGf_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qSort], gen_ch=self.swap_chs[qSort])
            else:
                self.swap_chs = self.cfg.hw.soc.dacs.swap_Q.ch
                self.swap_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
                mixer_freqs = self.cfg.hw.soc.dacs.swap_Q.mixer_freq
                self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf_Q[qSort], gen_ch=self.swap_chs[qSort])
                self.gain_EgGf = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort]
                self.type_EgGf = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qSort]
                self.sigma_EgGf_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort], gen_ch=self.swap_chs[qSort])
            
            if self.type_EgGf.lower() == "gauss" and self.sigma_EgGf_cycles > 0:
                self.add_gauss(ch=self.swap_chs[qSort], name="pi_EgGf_swap", sigma=self.sigma_EgGf_cycles, length=self.sigma_EgGf_cycles*4)
            elif self.type_EgGf.lower() == "flat_top" and self.sigma_EgGf_cycles > 0:
                self.add_gauss(ch=self.swap_chs[qSort], name="pi_EgGf_swap", sigma=3, length=3*4)

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
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_f0g1_ch_types[q] == 'int4':
                    mixer_freq = mixer_freqs[q]
                if self.swap_f0g1_chs[q] not in self.gen_chs: 
                    self.declare_gen(ch=self.swap_f0g1_chs[q], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[q], mixer_freq=mixer_freq)


        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on (mostly for use for preparation if we need to calibrate ef)
        self.pisigma_ge = self.us2cycles(self.pi_ge_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.freq2reg(self.f_ges[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
        self.gain_ge_init = self.pi_ge_gains[qTest, qZZ] if self.pi_ge_gains[qTest, qZZ] > 0 else self.pi_ge_gains[qTest, qTest] # this contingency is possible if the ge pulse is not calibrated but we want to calibrate the EF pulse for a specific ZZ configuration
        if self.checkZZ: self.pisigma_ge_qZZ = self.us2cycles(self.pi_ge_sigmas[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ])

        # parameters for test pulse that we are trying to calibrate
        if 'sigma_test' not in self.cfg.expt or self.cfg.expt.sigma_test is None:
            self.cfg.expt.sigma_test = self.pi_ge_sigmas[qTest, qZZ]
            if self.checkEF: self.cfg.expt.sigma_test = self.pi_ef_sigmas[qTest, qZZ]
        self.f_pi_test_reg = self.freq2reg(self.f_ges[qTest, qZZ])
        if self.checkEF: self.f_pi_test_reg = self.freq2reg(self.f_efs[qTest, qZZ])
        self.test_pi_half = False # calibrate the pi/2 pulse instead of the pi pulse by taking half the sigma and calibrating the gain
        divide_len = True
        if 'divide_len' in self.cfg.expt: divide_len = self.cfg.expt.divide_len
        if 'test_pi_half' in self.cfg.expt and self.cfg.expt.test_pi_half:
            self.test_pi_half = self.cfg.expt.test_pi_half
            if divide_len:
                print(f'Calibrating half pi gain (divide length) when pi len is {self.cfg.expt.sigma_test}')
                self.cfg.expt.sigma_test /= 2
            else: print(f'Calibrating half pi gain (divide gain) when pi len is {self.cfg.expt.sigma_test}')
        self.pi_test_sigma = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        assert self.f_pi_test_reg > 0
        assert self.pi_test_sigma > 0


        # add qubit and readout pulses to respective channels
        if cfg.expt.pulse_type.lower() == "gauss" and self.pi_test_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma*4)
        elif cfg.expt.pulse_type.lower() == 'adiabatic' and self.pi_test_sigma > 0:
            assert 'beta' in self.cfg.expt and 'mu' in self.cfg.expt
            self.add_adiabatic(ch=self.qubit_chs[qTest], name='pi_test', mu=self.cfg.expt.mu, beta=self.cfg.expt.beta, period_us=self.cfg.expt.sigma_test, length_us=self.cfg.expt.sigma_test)
        elif cfg.expt.pulse_type.lower() == 'pulseiq':
            assert 'Icontrols' in self.cfg.expt and 'Qcontrols' in self.cfg.expt and 'times_us' in self.cfg.expt
            self.add_IQ(ch=self.qubit_chs[qTest], name='pi_test', I_mhz_vs_us=self.cfg.expt.Icontrols, Q_mhz_vs_us=self.cfg.expt.Qcontrols, times_us=self.cfg.expt.times_us)

        if self.checkZZ:
            self.add_gauss(ch=self.qubit_chs[qZZ], name="pi_qubitZZ", sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ*4)
        if self.checkEF:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            for q in self.cfg.expt.cool_qubits:
                self.pisigma_ef = self.us2cycles(self.pi_ef_sigmas[q, q], gen_ch=self.qubit_chs[q]) # default pi_ef value
                self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=self.pisigma_ef, length=self.pisigma_ef*4)
                if self.cfg.device.qubit.pulses.pi_f0g1.type[q] == 'flat_top':
                    self.add_gauss(ch=self.swap_f0g1_chs[q], name=f"pi_f0g1_{q}", sigma=3, length=3*4)
                else: assert False, 'not implemented'
            self.f_f0g1_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_f0g1, self.qubit_chs)]

        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))

        # initialize registers
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_gain = self.sreg(self.qubit_chs[qTest], "addr") # get gain register for qubit_ch    
        else: self.r_gain = self.sreg(self.qubit_chs[qTest], "gain") # get gain register for qubit_ch    
        self.r_gain2 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_gain2, self.cfg.expt.start)

        self.set_gen_delays()
        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None: qZZ = qTest

        self.reset_and_sync()

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            cool_qubits = self.cfg.expt.cool_qubits
            if 'cool_idle' in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            sorted_indices = np.argsort(cool_idle)[::-1] # sort cooling times longest first
            cool_qubits = np.array(cool_qubits)
            cool_idle = np.array(cool_idle)
            sorted_cool_qubits = cool_qubits[sorted_indices]
            sorted_cool_idle = cool_idle[sorted_indices]
            max_idle = sorted_cool_idle[0]
        
            last_pulse_len = 0
            remaining_idle = max_idle
            for q, idle in zip(sorted_cool_qubits, sorted_cool_idle):
                remaining_idle -= last_pulse_len

                last_pulse_len = 0
                self.setup_and_pulse(ch=self.qubit_chs[q], style="arb", phase=0, freq=self.freq2reg(self.f_efs[q, q], gen_ch=self.qubit_chs[q]), gain=self.pi_ef_gains[q, q], waveform=f"pi_ef_qubit{q}")
                self.sync_all()
                last_pulse_len += self.pi_ef_sigmas[q, q]*4

                pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[q]
                pisigma_f0g1 = self.us2cycles(self.cfg.device.qubit.pulses.pi_f0g1.sigma[q], gen_ch=self.swap_f0g1_chs[q])
                if pulse_type == 'flat_top':
                    sigma_ramp_cycles = 3
                    flat_length_cycles = pisigma_f0g1 - sigma_ramp_cycles*4
                    self.setup_and_pulse(ch=self.swap_f0g1_chs[q], style="flat_top", freq=self.f_f0g1_regs[q], phase=0, gain=self.cfg.device.qubit.pulses.pi_f0g1.gain[q], length=flat_length_cycles, waveform=f"pi_f0g1_{q}")
                else: assert False, 'not implemented'
                self.sync_all()
                last_pulse_len += self.cfg.device.qubit.pulses.pi_f0g1.sigma[q]

            remaining_idle -= last_pulse_len
            last_idle = max((remaining_idle, sorted_cool_idle[-1]))
            self.sync_all(self.us2cycles(last_idle))

        # initializations as necessary
        if self.checkZZ:                    
            assert self.pi_ge_gains[qZZ, qZZ] > 0
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.freq2reg(self.f_ges[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ]), gain=self.pi_ge_gains[qZZ, qZZ], waveform="pi_qubitZZ")
            # print('check zz qubit', qZZ)
            self.sync_all(0)
        if self.checkEF and self.pulse_ge:
            assert self.gain_ge_init > 0
            assert self.f_ge_init_reg > 0
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")
            # print('init pulse on q', qTest, 'freq', self.reg2freq(self.f_ge_init_reg, gen_ch=self.qubit_chs[qTest]), 'gain', self.gain_ge_init)
            self.sync_all(0)
        if 'apply_EgGf' in self.cfg.expt and self.cfg.expt.apply_EgGf:
            if self.type_EgGf == "gauss":
                self.setup_and_pulse(ch=self.swap_chs[self.qSort], style="arb", freq=self.f_EgGf_reg, phase=0, gain=self.gain_EgGf, waveform="pi_EgGf_swap") #, phrst=1)
            elif self.type_EgGf == 'flat_top':
                sigma_ramp_cycles = 3
                if 'sigma_ramp_cycles' in self.cfg.expt:
                    sigma_ramp_cycles = self.cfg.expt.sigma_ramp_cycles
                flat_length_cycles = self.sigma_EgGf_cycles - sigma_ramp_cycles*4
                if flat_length_cycles >= 3:
                    self.setup_and_pulse(
                        ch=self.swap_chs[self.qSort],
                        style="flat_top",
                        freq=self.f_EgGf_reg,
                        phase=0,
                        gain=self.gain_EgGf,
                        length=flat_length_cycles,
                        waveform="pi_EgGf_swap",
                    )
            else: # const
                self.setup_and_pulse(ch=self.swap_chs[self.qSort], style="const", freq=self.f_EgGf_reg, phase=0, gain=self.gain_EgGf, length=self.sigma_EgGf_cycles)
            self.sync_all(5)

        if self.pi_test_sigma > 0:
            # print(self.f_pi_test_reg)
            # print('test pulse on q', qTest, 'freq', self.reg2freq(self.f_pi_test_reg, gen_ch=self.qubit_chs[qTest]))
            if cfg.expt.pulse_type.lower() in ("gauss", "adiabatic", 'pulseiq'):
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="arb",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_test")
            else:
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest],
                    style="const",
                    freq=self.f_pi_test_reg,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest]))
        self.mathi(self.q_rps[qTest], self.r_gain, self.r_gain2, "+", 0)

        n_pulses = 1
        if 'n_pulses' in self.cfg.expt: n_pulses = self.cfg.expt.n_pulses
        if self.test_pi_half: n_pulses *= 2
        for i in range(n_pulses):
            self.pulse(ch=self.qubit_chs[qTest])
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
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )
 
    def update(self):
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None: qZZ = qTest

        step = self.cfg.expt.step
        if self.qubit_ch_types[qTest] == 'int4': step = step << 16
        self.mathi(self.q_rps[qTest], self.r_gain2, self.r_gain2, '+', step) # update test gain
        
# ====================================================== #
                      
class AmplitudeRabiExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        checkEF
        test_pi_half
        divide_len (for test_pi_half only)
        qTest: qubit on which to do the test pulse
        qZZ: None if not checkZZ, else specify other qubit to pi pulse
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabi', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None: qZZ = qTest

        print(f'Running amp rabi {"EF " if self.cfg.expt.checkEF else ""}on Q{qTest} {"with ZZ Q" + str(qZZ) if qZZ != qTest else ""}')

        amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
        # print(amprabi)
        # from qick.helpers import progs2json
        # print(progs2json([amprabi.dump_prog()]))
        
        xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        # print(amprabi)

        # shots_i = amprabi.di_buf[adc_ch].reshape((self.cfg.expt.expts, self.cfg.expt.reps)) / amprabi.readout_length_adc
        # shots_i = np.average(shots_i, axis=1)
        # print(len(shots_i), self.cfg.expt.expts)
        # shots_q = amprabi.dq_buf[adc_ch] / amprabi.readout_length_adc
        # print(np.std(shots_i), np.std(shots_q))
        
        avgi = avgi[qTest][0]
        avgq = avgq[qTest][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        # data={'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, fitparams=None):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # fitparams=[yscale, freq, phase_deg, y0]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data['xpts']
            if fitparams is None:
                fitparams=[None]*4
                n_pulses = 1
                if 'n_pulses' in self.cfg.expt: n_pulses = self.cfg.expt.n_pulses
                fitparams[1]=n_pulses/xdata[-1]
                # print(fitparams[1])

            ydata = data["amps"]
            # print(abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]))
            # fitparams=[max(ydata)-min(ydata), 1/2 / abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]), None, None, None]
            # fitparams=[max(ydata)-min(ydata), 1/2 / (max(xdata) - min(xdata)), 0, None, None]

            p_avgi, pCov_avgi = fitter.fitsin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitsin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitsin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=False, **kwargs):
        if data is None:
            data=self.data 

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest

        plt.figure(figsize=(10, 6))
        n_pulses = 1
        if 'n_pulses' in self.cfg.expt: n_pulses = self.cfg.expt.n_pulses
        title = f"Amplitude Rabi {'EF ' if self.cfg.expt.checkEF else ''}on Q{qTest} (Pulse Length {self.cfg.expt.sigma_test}{(', ZZ Q'+str(qZZ)) if self.checkZZ else ''}, {n_pulses} pulse)"
        plt.subplot(111, title=title, xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        plt.plot(data["xpts"], data["amps"],'.-')
        if fit:
            p = data['fit_amps']
            plt.plot(data["xpts"], fitter.sinfunc(data["xpts"], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain = (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from amps data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from amps data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(data["xpts"], data["avgi"],'.-')
        # plt.axhline(390)
        # plt.axhline(473)
        # plt.axvline(2114)
        # plt.axvline(3150)
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"], data["avgq"],'.-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #
                      
class AmplitudeRabiChevronExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz), 
        step_f: frequency step (MHz), 
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiChevron', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        if self.cfg.expt.checkZZ:
            assert len(self.cfg.expt.qubits) == 2
            qZZ, qTest = self.cfg.expt.qubits
            assert qZZ != 1
            assert qTest == 1
        else: qTest = self.cfg.expt.qubits[0]

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        data={"xpts":[], "freqpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain
        for freq in tqdm(freqpts):
            self.cfg.expt.f_pi_test = freq
            amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
        
            xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)
        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)
        
        data['xpts'] = xpts
        data['freqpts'] = freqpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['xpts']
        y_sweep = data['freqpts']
        avgi = data['avgi']
        avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I [ADC level]')
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()

        # plt.plot(y_sweep, data['amps'][:,-1])
        # plt.title(f'Gain {x_sweep[-1]}')
        # plt.show()

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname