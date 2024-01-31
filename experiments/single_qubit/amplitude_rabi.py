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

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

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

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        if self.checkZZ:
            if qTest == 1: self.f_Q1_ZZ_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q1_ZZ]
            else: self.f_Q_ZZ1_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q_ZZ1]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        elif self.res_ch_types[qTest] == 'mux4':
            assert self.res_chs[qTest] == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
            mux_freqs = cfg.device.readout.frequency
            mux_gains = cfg.device.readout.gain
            ro_ch=self.adc_chs[qTest]
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])
        
        # define pi_test_sigma as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.pi_test_sigma = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qZZ], gen_ch=self.qubit_chs[qZZ])
            if qTest == 1:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q1_ZZ_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q1_ZZ_reg[qZZ] # freq we are trying to calibrate
            else:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q_ZZ1_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q_ZZ1_reg[qSort] # freq we are trying to calibrate
        if self.checkEF:
            self.f_pi_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate
        if 'f_pi_test' in self.cfg.expt:
            self.f_pi_test_reg = self.freq2reg(self.cfg.expt.f_pi_test, gen_ch=self.qubit_chs[qTest])
        calibrate_half = False # calibrate the pi/2 pulse instead of the pi pulse by taking half the sigma and calibrating the gain
        
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

        # add readout pulses to respective channels
        if self.res_ch_types[qTest] == 'mux4':
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        else: self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=0, gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # initialize registers
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_gain = self.sreg(self.qubit_chs[qTest], "addr") # get gain register for qubit_ch    
        else: self.r_gain = self.sreg(self.qubit_chs[qTest], "gain") # get gain register for qubit_ch    
        self.r_gain2 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_gain2, self.cfg.expt.start)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        if self.checkZZ: qZZ, qTest = self.qubits
        else: qTest = self.qubits[0]

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            # self.sync_all()
        self.sync_all(10)

        # initializations as necessary
        if self.checkZZ:                    
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.f_ge_reg[qZZ], gain=cfg.device.qubit.pulses.pi_ge.gain[qZZ], waveform="pi_qubitZZ")
            self.sync_all(0)
        if self.checkEF and self.pulse_ge:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")
            self.sync_all(0)

        if self.pi_test_sigma > 0:
            # print(self.f_pi_test_reg)
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
                    length=self.us2cycles(self.cfg.expt.sigma_test))
        self.mathi(self.q_rps[qTest], self.r_gain, self.r_gain2, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])
        self.sync_all()
        if 'calibrate_half' in self.cfg.expt and self.cfg.expt.calibrate_half:
            self.pulse(ch=self.qubit_chs[qTest])
            self.sync_all()

        if self.checkEF: # map excited back to qubit ground state for measurement
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")

        # align channels and measure
        self.sync_all(5)
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )
 
    def update(self):
        if self.checkZZ: qZZ, qTest = self.qubits
        else: qTest = self.qubits[0]

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
        checkZZ
        checkEF
        calibrate_half
        qubits
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabi', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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

        # print('FUNKY WARNING')
        # qZZ, qTest = self.cfg.expt.qubits

        self.checkZZ = self.cfg.expt.checkZZ
        self.qubits = self.cfg.expt.qubits
        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
        else: qTest = self.qubits[0]

        if 'sigma_test' not in self.cfg.expt:
            if self.cfg.expt.checkZZ:
                if qTest == 1: self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort]
                else: self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort]
            elif self.cfg.expt.checkEF:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ef.sigma[qTest]
            else: 
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest]
        if 'calibrate_half' in self.cfg.expt and self.cfg.expt.calibrate_half:
            print(f'Calibrating half pi gain for pi len of {self.cfg.expt.sigma_test}')
            self.cfg.expt.sigma_test /= 2 
        
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
        
        # print('WARNING DOING SOMETHING FUNKY')
        # avgi = avgi[qTest][0]
        # avgq = avgq[qTest][0]
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        # data={'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data['xpts']
            fitparams=None

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

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        if self.cfg.expt.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
        else: qTest = self.cfg.expt.qubits[0]

        plt.figure(figsize=(10, 6))
        title = f"Amplitude Rabi on Q{qTest} (Pulse Length {self.cfg.expt.sigma_test}{(', ZZ Q'+str(qZZ)) if self.checkZZ else ''})"
        plt.subplot(111, title=title, xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        if fit:
            p = data['fit_amps']
            plt.plot(data["xpts"][1:-1], fitter.sinfunc(data["xpts"][1:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from amps data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from amps data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
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
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

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
        if 'sigma_test' not in self.cfg.expt:
            if self.cfg.expt.checkZZ:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qZZ]
            elif self.cfg.expt.checkEF:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ef.sigma[qTest]
            else:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest]
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

        plt.plot(y_sweep, data['amps'][:,-1])
        plt.title(f'Gain {x_sweep[-1]}')
        plt.show()

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname