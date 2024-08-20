import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.clifford_averager_program import QutritAveragerProgram
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo2QProgram, sort_counts, correct_readout_err, fix_neg_counts
import experiments.fitting as fitter

class CrosstalkEchoProgram(QutritAveragerProgram):
    def initialize(self):
        super().initialize()

        # measure crosstalk on Q1, drive the 2Q swaps between each qDrive and Q1
        self.qTest = self.cfg.expt.qTest
        self.qDrives = self.cfg.expt.qDrives

        self.swap_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap_Q.type

        self.f_EgGf_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_EgGf_Q, self.swap_chs)]
        self.pi_EgGf_Q_types = self.cfg.device.qubit.pulses.pi_EgGf_Q.type

        # declare swap dacs
        for qDrive in self.qDrives:
            mixer_freq = None
            if self.swap_ch_types[qDrive] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap_Q.mixer_freq[qDrive]
            if self.swap_chs[qDrive] not in self.gen_chs: 
                self.declare_gen(ch=self.swap_chs[qDrive], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qDrive], mixer_freq=mixer_freq)

        self.wait_cycles = self.us2cycles(self.cfg.expt.wait_us) // 2 # used both as wait time and as length of swap pulses, divide by 2 for the 2 segments

        # add swap pulses
        for qDrive in self.qDrives:
            assert self.pi_EgGf_Q_types[qDrive] == "flat_top"
            assert self.wait_cycles > 24
            self.add_gauss(ch=self.swap_chs[qDrive], name=f"pi_EgGf_swap_{qDrive}q1", sigma=3, length=3*4)
        
        # add correction pulse
        assert 'gain_x' in self.cfg.expt
        assert 'gain_y' in self.cfg.expt
        self.gain_x = self.cfg.expt.gain_x
        self.gain_y = self.cfg.expt.gain_y
        self.gain_amp = int(np.abs(self.gain_x + 1j*self.gain_y)) # gain units
        self.gain_phi = np.angle(self.gain_x + 1j*self.gain_y)
        if self.gain_amp > 0:
            assert 'pi_ge_crosstalk' in self.cfg.device.qubit.pulses
            self.f_cancel_regs = [self.freq2reg(fq + delta, gen_ch=ch) for delta, fq, ch in zip(self.cfg.device.qubit.pulses.pi_ge_crosstalk.delta, self.cfg.device.qubit.f_ge, self.qubit_chs)]
            assert self.cfg.device.qubit.pulses.pi_ge_crosstalk.type[self.qTest] == "flat_top"
            assert self.wait_cycles > 24
            self.add_gauss(ch=self.qubit_chs[self.qTest], name=f"crosstalk_{self.qTest}", sigma=3, length=3*4)


        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs: # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            # self.sync_all()
        self.sync_all(10)

        assert self.qTest == 1

        # ================= #
        # Initialization in g(g+e)ee
        # ================= #
        # num_qDrive = len(self.qDrives) # number of output qubits in e
        num_qDrive = 2 # number of output qubits in e
        prep_e = [2, 3]
        pi_gain = self.cfg.device.qubit.pulses.pi_ge.gain[self.qTest]
        pi_freq = self.f_ge_regs[self.qTest]
        pi_waveform = f'pi_ge_q{self.qTest}'
        pi_half_waveform = f'pi_gehalf_q{self.qTest}'
        pi_sigma = self.us2cycles(self.pi_sigmas_us[self.qTest], gen_ch=self.qubit_chs[self.qTest])

        if num_qDrive >= 1: # e
            first_e = prep_e[0]
            self.X_pulse(q=first_e, pihalf=False, play=True)

            # set params for Q1
            pi_gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[first_e]
            pi_freq = self.f_Q1_ZZ_regs[first_e]
            pi_waveform = f'qubit{self.qTest}_ZZ{first_e}'
            pi_half_waveform = f'qubit{self.qTest}_ZZ{first_e}_half'
            pi_sigma = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[first_e], gen_ch=self.qubit_chs[first_e])

        if num_qDrive == 2: # ee
            second_e = prep_e[1]
            ZZs = np.reshape(self.cfg.device.qubit.ZZs, (4,4))
            freq = self.freq2reg(self.cfg.device.qubit.f_ge[second_e] + ZZs[second_e, first_e], gen_ch=self.qubit_chs[second_e])
            waveform = f'qubit{second_e}_ZZ{first_e}'
            sigma_cycles = self.us2cycles(self.pi_sigmas_us[second_e], gen_ch=self.qubit_chs[second_e])
            self.add_gauss(ch=self.qubit_chs[second_e], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[second_e]
            self.setup_and_pulse(ch=self.qubit_chs[second_e], style='arb', freq=freq, phase=0, gain=gain, waveform=waveform)
            self.sync_all()

            # set params for Q1
            # use other params from pi_Q1_ZZ
            pi_freq = self.freq2reg(self.cfg.device.qubit.f_Q1_ZZ[first_e] + ZZs[self.qTest, second_e], gen_ch=self.qubit_chs[self.qTest])

        if pi_waveform not in self.envelopes:
            self.add_gauss(ch=self.qubit_chs[self.qTest], name=pi_waveform, sigma=pi_sigma, length=4*pi_sigma)
        if pi_half_waveform not in self.envelopes:
            self.add_gauss(ch=self.qubit_chs[self.qTest], name=pi_half_waveform, sigma=pi_sigma//2, length=4*pi_sigma//2)


        # ================= #
        # Begin ramsey sequence
        # ================= #
        # play pi/2 pulse with the freq that we want to calibrate
        self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=0, gain=pi_gain, waveform=pi_half_waveform)
        self.sync_all()


        # empty wait time
        self.sync_all()
        self.sync_all(self.wait_cycles)

        # echo pi pulse
        self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=90 if self.cfg.expt.cpmg else 0, gain=pi_gain, waveform=pi_waveform)
        self.sync_all()

        # do the 2q swap(s) with same length as wait time
        if len(self.qDrives) == 0 and self.gain_amp == 0:
            self.sync_all(self.wait_cycles)
        else:
            for qDrive in self.qDrives:
                sigma_ramp_cycles = 3
                flat_length_cycles = self.wait_cycles - sigma_ramp_cycles*4
                self.setup_and_pulse(
                    ch=self.swap_chs[qDrive],
                    style="flat_top",
                    freq=self.f_EgGf_regs[qDrive],
                    phase=0,
                    gain=self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive],
                    length=flat_length_cycles,
                    waveform=f"pi_EgGf_swap_{qDrive}q1",
                )
                # NO SYNC
            
            # correction pulse
            if self.gain_amp > 0:
                sigma_ramp_cycles = 3
                flat_length_cycles = self.wait_cycles - sigma_ramp_cycles*4
                self.setup_and_pulse(
                    ch=self.qubit_chs[self.qTest],
                    style="flat_top",
                    freq=self.f_cancel_regs[self.qTest],
                    phase=self.deg2reg(self.gain_phi * 180/np.pi),
                    gain=self.gain_amp,
                    length=flat_length_cycles,
                    waveform=f"crosstalk_{self.qTest}",
                )
        self.sync_all()

        # play pi/2 pulse with advanced phase
        phase_reg = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.wait_us, gen_ch=self.qubit_chs[self.qTest])
        self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=phase_reg, gain=pi_gain, waveform=pi_half_waveform)
        self.sync_all()

        # align channels and measure
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

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


class CrosstalkEchoExperiment(Experiment):
    """
    Modified echo experiment on Q1 with 2nd wait time replaced by the 2q swap(s) on Q2, Q3
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qTest: measure crosstalk (this should be Q1)
        qDrives: array of the 2Q swaps to drive between each qDrive and Q1; if array is empty then just plays an empty wait time
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CrosstalkEcho', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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

        wait_times = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        data={'xpts': wait_times, 'avgi':[], 'avgq':[], 'amps':[], 'phases':[], 'counts_calib':[]}

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
                if debug: print('Re-using provided angles, thresholds, ge_avgs')
            else:
                thresholds_q = [0]*4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0]*4
                fids_q = [0]*4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.tomo_qubits = [self.qTest, (self.qTest+1)%4]

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
                shot_data = dict(Ig=Ig[self.qTest], Qg=Qg[self.qTest], Ie=Ie[self.qTest], Qe=Qe[self.qTest])
                print(f'Qubit  ({self.qTest})')
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[self.qTest] = threshold[0]
                ge_avgs_q[self.qTest] = [np.average(Ig[self.qTest]), np.average(Qg[self.qTest]), np.average(Ie[self.qTest]), np.average(Qe[self.qTest])]
                angles_q[self.qTest] = angle
                fids_q[self.qTest] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[self.qTest]} \t threshold ge: {thresholds_q[self.qTest]}')


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

        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.expt.qTest]

        for t in tqdm(wait_times, disable=not progress):
            cfg = deepcopy(self.cfg)
            cfg.expt.wait_us = t
            ramsey = CrosstalkEchoProgram(soccfg=self.soccfg, cfg=cfg)
        
            avgi, avgq = ramsey.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        
 
            avgi = avgi[adc_ch]
            avgq = avgq[adc_ch]
            data['avgi'].append(avgi)
            data['avgq'].append(avgq)
            data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
            data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

        for k, a in data.items():
            data[k]=np.array(a)

        self.data=data
        return data

    def analyze(self, data=None, fit=True, fit_num_sin=1, fitparams=None):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # fitparams=[yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0] # two fit freqs
            # fitparams=[yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01, yscale2, freq2, phase_deg2, y02] # three fit freqs
            
            # Remove the first and last point from fit in case weird edge measurements
            if fit_num_sin == 2:
                fitfunc = fitter.fittwofreq_decaysin
                # fitparams = [None]*8
                # fitparams[1] = self.cfg.expt.ramsey_freq
                # fitparams[3] = 15 # decay
                # fitparams[4] = 0.05 # yscale1 (ratio relative to base oscillations)
                # fitparams[5] = 1/12.5 # freq1
                # # print('FITPARAMS', fitparams[7])
            elif fit_num_sin == 3:
                fitfunc = fitter.fitthreefreq_decaysin
                # fitparams = [None]*14
                # fitparams[1] = self.cfg.expt.ramsey_freq
                # fitparams[3] = 15 # decay
                # fitparams[6] = 1.1 # yscale1
                # fitparams[7] = 0.415 # freq1
                # fitparams[-4] = 1.1 # yscale2
                # fitparams[-3] = 0.494 # freq2
                # print('FITPARAMS', fitparams[7])
            else:
                fitfunc = fitter.fitdecaysin
                # fitparams=[None, self.cfg.expt.ramsey_freq, None, None, None]
            p_avgi, pCov_avgi = fitfunc(data['xpts'], data["avgi"], fitparams=fitparams)
            p_avgq, pCov_avgq = fitfunc(data['xpts'], data["avgq"], fitparams=fitparams)
            p_amps, pCov_amps = fitfunc(data['xpts'], data["amps"], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            print('p_amps', data['fit_amps'])
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            # print('p avgi', p_avgi)
            # print('p avgq', p_avgq)
            # print('p amps', p_amps)

            if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((self.cfg.expt.ramsey_freq - p_avgi[1], -self.cfg.expt.ramsey_freq - p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((self.cfg.expt.ramsey_freq - p_avgq[1], -self.cfg.expt.ramsey_freq - p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((self.cfg.expt.ramsey_freq - p_amps[1], -self.cfg.expt.ramsey_freq - p_amps[1]), key=abs)

            if fit_num_sin == 2:
                data['f_adjust_ramsey_avgi2'] = sorted((self.cfg.expt.ramsey_freq - p_avgi[5], -self.cfg.expt.ramsey_freq - p_avgi[5]), key=abs)
                data['f_adjust_ramsey_avgq2'] = sorted((self.cfg.expt.ramsey_freq - p_avgq[5], -self.cfg.expt.ramsey_freq - p_avgq[5]), key=abs)
                data['f_adjust_ramsey_amps2'] = sorted((self.cfg.expt.ramsey_freq - p_amps[5], -self.cfg.expt.ramsey_freq - p_amps[5]), key=abs)
        return data

    def display(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data=self.data
        
        qTest = self.cfg.expt.qTest
        qDrives = self.cfg.expt.qDrives

        f_pi_test = self.cfg.device.qubit.f_ge[qTest]
        title = f'Echo on Q{qTest}, drive {qDrives}, correction gains {[self.cfg.expt.gain_x, self.cfg.expt.gain_y]}'

        if fit_num_sin == 2: fitfunc = fitter.twofreq_decaysin
        elif fit_num_sin == 3: fitfunc = fitter.threefreq_decaysin
        else: fitfunc = fitter.decaysin

        plt.figure(figsize=(10, 6))
        plt.subplot(111,title=f"{title} (fRamsey: {self.cfg.expt.ramsey_freq} MHz)",
                    xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        if fit:
            p = data['fit_amps']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_amps']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], p[0], 0, p[3]), color='0.2', linestyle='--')
                print('ps amps', p[-1], p[0], p[3])
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], -p[0], 0, p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f"Fit frequency from amps [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}")
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print(f'Suggested new pi pulse frequencies from fit amps [MHz]:\n',
                      f'\t{data["f_adjust_ramsey_amps"][0]}\n',
                      f'\t{data["f_adjust_ramsey_amps"][1]}')
                if fit_num_sin == 2:
                    print('Beating frequencies from fit amps [MHz]:\n',
                          f'\tyscale base: {1-p[4]}',
                          f'\tfit freq {p[1]}\n',
                          f'\tyscale1: {p[4]}'
                          f'\tfit freq {p[5]}\n')
                print(f'T2 Ramsey from fit amps [us]: {p[3]}')

        plt.figure(figsize=(10,9))
        plt.subplot(211, 
            title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgi']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], p[0], 0, p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], -p[0], 0, p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                if fit_num_sin == 2:
                    print('Beating frequencies from fit avgi [MHz]:\n',
                          f'\tyscale base: {1-p[4]}',
                          f'\tfit freq {p[1]}\n',
                          f'\tyscale1: {p[4]}'
                          f'\tfit freq {p[5]}\n')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgq']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], p[0], 0, p[3]), color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[-1], -p[0], 0, p[3]), color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                      f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                if fit_num_sin == 2:
                    print('Beating frequencies from fit avgq [MHz]:\n',
                          f'\tyscale base: {1-p[4]}',
                          f'\tfit freq {p[1]}\n',
                          f'\tyscale1: {p[4]}'
                          f'\tfit freq {p[5]}\n')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


class CrosstalkEchoChevronExperiment(Experiment):
    """
    Modified echo experiment on Q1 with 2nd wait time replaced by the 2q swap(s) on Q2, Q3, apply a correction pulse on Q1 with gain=abs(gain_x + 1j*gain_y), phase=angle(gain_x + 1j*gain_y), with parameters swept (wait time is fixed).
    Experimental Config:
    expt = dict(
        start_x: sweep gain_x
        step_x:
        expts_x:
        start_y: sweep gain_y
        step_y:
        expts_y:
        tau: wait time, which is the same as the time for applying the 2 2q pulses
        ramsey_freq: frequency by which to advance phase [MHz] (typical use set to 0)
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qTest: measure crosstalk (this should be Q1)
        qDrives: array of the 2Q swaps to drive between each qDrive and Q1; if array is empty then just plays an empty wait time
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CrosstalkEcho', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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

        gain_x_sweep = self.cfg.expt.start_x + self.cfg.expt.step_x * np.arange(self.cfg.expt.expts_x)
        gain_y_sweep = self.cfg.expt.start_y + self.cfg.expt.step_y * np.arange(self.cfg.expt.expts_y)

        data={'x_sweep': gain_x_sweep, 'y_sweep': gain_y_sweep, 'avgi':[], 'avgq':[], 'amps':[], 'phases':[], 'counts_calib':[]}        

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
                sscfg.expt.tomo_qubits = [self.qTest, (self.qTest+1)%4]

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
                shot_data = dict(Ig=Ig[self.qTest], Qg=Qg[self.qTest], Ie=Ie[self.qTest], Qe=Qe[self.qTest])
                print(f'Qubit  ({self.qTest})')
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[self.qTest] = threshold[0]
                ge_avgs_q[self.qTest] = [np.average(Ig[self.qTest]), np.average(Qg[self.qTest]), np.average(Ie[self.qTest]), np.average(Qe[self.qTest])]
                angles_q[self.qTest] = angle
                fids_q[self.qTest] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[self.qTest]} \t threshold ge: {thresholds_q[self.qTest]}')


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

        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.expt.qTest]

        if 'wait_us' not in self.cfg.expt or self.cfg.expt.wait_us is None:
            self.cfg.expt.wait_us = (self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[2] + self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[3])/2
        print('using wait time', self.cfg.expt.wait_us)

        for gain_y in tqdm(gain_y_sweep):
            for gain_x in gain_x_sweep:
                cfg = deepcopy(self.cfg)
                cfg.expt.start = None
                cfg.expt.step = None
                cfg.expt.expts = None
                cfg.expt.gain_x = gain_x
                cfg.expt.gain_y = gain_y
                ramsey = CrosstalkEchoProgram(soccfg=self.soccfg, cfg=cfg)
        
                avgi, avgq = ramsey.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        
 
                avgi = avgi[adc_ch]
                avgq = avgq[adc_ch]
                data['avgi'].append(avgi)
                data['avgq'].append(avgq)
                data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
                data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

        for k, a in data.items():
            data[k]=np.array(a)
            if np.shape(data[k]) == (len(gain_y_sweep) * len(gain_x_sweep),):
                data[k] = np.reshape(data[k], (len(gain_y_sweep), len(gain_x_sweep)))

        self.data=data
        return data

    def analyze(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data=self.data
        
        if fit: pass
        return data

    def display(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data=self.data
        
        qTest = self.cfg.expt.qTest
        qDrives = self.cfg.expt.qDrives

        data = deepcopy(data)
        x_sweep = data['x_sweep']
        y_sweep = data['y_sweep']

        title = f'Crosstalk Calibration on Q{qTest} with drive on {qDrives}'

        plt.figure(figsize=(10, 8))
        this_data = data['amps']
        plt.pcolormesh(x_sweep, y_sweep, this_data, cmap='viridis', shading='auto')
        plt.colorbar()
        plt.xlabel('Gain_X')
        plt.ylabel('Gain_Y')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


class CrosstalkEchoTimeChevronExperiment(Experiment):
    """
    Modified echo experiment on Q1 with 2nd wait time replaced by the 2q swap(s) on Q2, Q3, apply a correction pulse on Q1 with gain=abs(gain_x + 1j*gain_y), phase=angle(gain_x + 1j*gain_y), with parameters swept, sweep wait time.
    Experimental Config:
    expt = dict(
        start_i: sweep gain_x or gain_y
        step_i:
        expts_i:
        start_t: sweep wait time
        step_t:
        expts_t:
        sweep_axis: 'x' or 'y'
        ramsey_freq: frequency by which to advance phase [MHz] (typical use set to 0)
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qTest: measure crosstalk (this should be Q1)
        qDrives: array of the 2Q swaps to drive between each qDrive and Q1; if array is empty then just plays an empty wait time
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CrosstalkEchoTime', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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
        
        self.qTest = self.cfg.expt.qTest
        self.qDrives = self.cfg.expt.qDrives

        gain_i_sweep = self.cfg.expt.start_i + self.cfg.expt.step_i * np.arange(self.cfg.expt.expts_i)
        wait_times = self.cfg.expt.start_t + self.cfg.expt.step_t * np.arange(self.cfg.expt.expts_t)
        print('gain sweep', gain_i_sweep)


        data={'gain_i_sweep': gain_i_sweep, 'wait_times': wait_times, 'avgi':[], 'avgq':[], 'amps':[], 'phases':[], 'counts_calib':[]} 

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
                sscfg.expt.tomo_qubits = [self.qTest, (self.qTest+1)%4]

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
                shot_data = dict(Ig=Ig[self.qTest], Qg=Qg[self.qTest], Ie=Ie[self.qTest], Qe=Qe[self.qTest])
                print(f'Qubit  ({self.qTest})')
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[self.qTest] = threshold[0]
                ge_avgs_q[self.qTest] = [np.average(Ig[self.qTest]), np.average(Qg[self.qTest]), np.average(Ie[self.qTest]), np.average(Qe[self.qTest])]
                angles_q[self.qTest] = angle
                fids_q[self.qTest] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[self.qTest]} \t threshold ge: {thresholds_q[self.qTest]}')


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

        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.expt.qTest]

        cfg = deepcopy(self.cfg)
        cfg.expt.start = self.cfg.expt.start_t
        cfg.expt.step = self.cfg.expt.step_t
        cfg.expt.expts = self.cfg.expt.expts_t
        cfg.expt.thresholds = thresholds_q
        cfg.expt.angles = angles_q
        cfg.expt.ge_avgs = ge_avgs_q
        cfg.expt.counts_calib = np.array(data['counts_calib'])
        expt_prog = CrosstalkEchoExperiment(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file) 
        expt_prog.cfg = cfg

        for gain_i in tqdm(gain_i_sweep):
            if expt_prog.cfg.expt.sweep_axis == 'x':
                expt_prog.cfg.expt.gain_x = gain_i
                expt_prog.cfg.expt.gain_y = 0
            elif expt_prog.cfg.expt.sweep_axis == 'y':
                expt_prog.cfg.expt.gain_x = 0
                expt_prog.cfg.expt.gain_y = gain_i
            expt_prog.acquire(debug=False)

            data['avgi'].append(expt_prog.data['avgi'])
            data['avgq'].append(expt_prog.data['avgq'])
            data['amps'].append(expt_prog.data['amps'])
            data['phases'].append(expt_prog.data['phases'])

        for k, a in data.items():
            data[k]=np.array(a)
            if np.shape(data[k]) == (len(gain_i_sweep) * len(wait_times),):
                data[k] = np.reshape(data[k], (len(gain_i_sweep), len(wait_times)))

        self.data=data
        return data

    def analyze(self, data=None, fit=True, fit_time=None):
        if data is None:
            data=self.data

        if not fit: return data

# t_ind = np.argmin(np.abs(data['wait_times']-3))
        
        return data

    def display(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data=self.data
        
        qTest = self.cfg.expt.qTest
        qDrives = self.cfg.expt.qDrives

        data = deepcopy(data)
        x_sweep = data['wait_times']
        y_sweep = data['gain_i_sweep']

        title = f'Crosstalk Calibration on Q{qTest} with drive on {qDrives}'

        plt.figure(figsize=(10, 8))
        this_data = data['amps']
        plt.pcolormesh(x_sweep, y_sweep, this_data, cmap='viridis', shading='auto')
        plt.axvline(0.410, color='r', linestyle='--')
        # plt.axhline(100)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Wait Times (us)')
        plt.ylabel(f'Gains ({self.cfg.expt.sweep_axis} axis) (DAC units)')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


# ===================================================================== #
# from qram_protocol_timestepped import QramProtocolProgram

# class SwapPhaseCalibrationProgram(QramProtocolProgram):
#     def initialize(self):
#         super().initialize()
#         self.qTest = self.cfg.expt.qTest
#         self.qDrives = self.cfg.expt.qDrives

#     def body(self):
#         cfg=AttrDict(self.cfg)

#         # Phase reset all channels
#         for ch in self.gen_chs.keys():
#             if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
#                 # print('resetting', ch)
#                 self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
#             # self.sync_all()
#         self.sync_all(10)


#         # if self.qTest == 1
#         # ================= #
#         # Begin ramsey sequence
#         # ================= #
#         # play pi/2 pulse with the freq that we want to calibrate
#         self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=0, gain=pi_gain, waveform=pi_half_waveform)
#         self.sync_all()


#         # empty wait time
#         self.sync_all()
#         self.sync_all(self.wait_cycles)

#         # echo pi pulse
#         self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=90 if self.cfg.expt.cpmg else 0, gain=pi_gain, waveform=pi_waveform)
#         self.sync_all()

#         # do the 2q swap(s) with same length as wait time
#         if len(self.qDrives) == 0 and self.gain_amp == 0:
#             self.sync_all(self.wait_cycles)
#         else:
#             for qDrive in self.qDrives:
#                 sigma_ramp_cycles = 3
#                 flat_length_cycles = self.wait_cycles - sigma_ramp_cycles*4
#                 self.setup_and_pulse(
#                     ch=self.swap_chs[qDrive],
#                     style="flat_top",
#                     freq=self.f_EgGf_regs[qDrive],
#                     phase=0,
#                     gain=self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive],
#                     length=flat_length_cycles,
#                     waveform=f"pi_EgGf_swap_{qDrive}q1",
#                 )
#                 # NO SYNC
            
#             # correction pulse
#             if self.gain_amp > 0:
#                 sigma_ramp_cycles = 3
#                 flat_length_cycles = self.wait_cycles - sigma_ramp_cycles*4
#                 self.setup_and_pulse(
#                     ch=self.qubit_chs[self.qTest],
#                     style="flat_top",
#                     freq=self.f_cancel_regs[self.qTest],
#                     phase=self.deg2reg(self.gain_phi * 180/np.pi),
#                     gain=self.gain_amp,
#                     length=flat_length_cycles,
#                     waveform=f"crosstalk_{self.qTest}",
#                 )
#         self.sync_all()

#         # play pi/2 pulse with advanced phase
#         phase_reg = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.wait_us, gen_ch=self.qubit_chs[self.qTest])
#         self.setup_and_pulse(ch=self.qubit_chs[self.qTest], style='arb', freq=pi_freq, phase=phase_reg, gain=pi_gain, waveform=pi_half_waveform)
#         self.sync_all()

#         # align channels and measure
#         self.sync_all()
#         measure_chs = self.res_chs
#         if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
#         self.measure(
#             pulse_ch=measure_chs, 
#             adcs=self.adc_chs,
#             adc_trig_offset=cfg.device.readout.trig_offset[0],
#             wait=True,
#             syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

#     """
#     If post_process == 'threshold': uses angle + threshold to categorize shots into 0 or 1 and calculate the population
#     If post_process == 'scale': uses angle + ge_avgs to scale the average of all shots on a scale of 0 to 1. ge_avgs should be of shape (num_total_qubits, 4) and should represent the pre-rotation Ig, Qg, Ie, Qe
#     If post_process == None: uses angle to rotate the i and q and then returns the avg i and q
#     """
#     def acquire_rotated(self, soc, progress, angle=None, threshold=None, ge_avgs=None, post_process=None, verbose=False):
#         avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress)
#         if post_process == None: 
#             avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)
#             if angle is None: return avgi_rot, avgq_rot
#             else: return avgi_rot, avgi_err
#         elif post_process == 'threshold':
#             assert threshold is not None
#             popln, avgq_rot, popln_err, avgq_err = self.get_shots(angle=angle, threshold=threshold, avg_shots=True, verbose=verbose, return_err=True)
#             return popln, popln_err
#         elif post_process == 'scale':
#             assert ge_avgs is not None
#             avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)

#             ge_avgs_rot = [None]*4
#             for q, angle_q in enumerate(angle):
#                 if not isinstance(ge_avgs[q], (list, np.ndarray)): continue # this qubit was not calibrated
#                 Ig_q, Qg_q, Ie_q, Qe_q = ge_avgs[q]
#                 ge_avgs_rot[q] = [
#                     Ig_q*np.cos(np.pi/180*angle_q) - Qg_q*np.sin(np.pi/180*angle_q),
#                     Ie_q*np.cos(np.pi/180*angle_q) - Qe_q*np.sin(np.pi/180*angle_q)
#                 ]
#             shape = None
#             for q in range(4):
#                 if ge_avgs_rot[q] is not None:
#                     shape = np.shape(ge_avgs_rot[q])
#                     break
#             for q in range(4):
#                 if ge_avgs_rot[q] is None: ge_avgs_rot[q] = np.zeros(shape=shape)
                
#             ge_avgs_rot = np.asarray(ge_avgs_rot)
#             avgi_rot -= ge_avgs_rot[:,0]
#             avgi_rot /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
#             avgi_err /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
#             return avgi_rot, avgi_err
#         else:
#             assert False, 'Undefined post processing flag, options are None, threshold, scale'
