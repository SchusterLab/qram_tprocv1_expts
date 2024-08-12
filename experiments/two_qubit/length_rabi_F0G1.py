import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter
from experiments.single_qubit.single_shot import hist
from experiments.clifford_averager_program import QutritAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram, ErrorMitigationStateTomo2QProgram, sort_counts, correct_readout_err, fix_neg_counts

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiF0G1Program(QutritAveragerProgram):
    def initialize(self):
        super().initialize()
        self.qubit = self.cfg.expt.qubit

        self.swap_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
        self.f_f0g1_reg = self.freq2reg(self.cfg.device.qubit.f_f0g1[self.qubit], gen_ch=self.swap_chs[self.qubit])

        if self.swap_chs[self.qubit] not in self.gen_chs: 
            self.declare_gen(ch=self.swap_chs[self.qubit], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[self.qubit])

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.swap_chs[self.qubit])

        # add swap pulse
        if self.cfg.expt.pulse_type.lower() == "gauss" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[self.qubit], name="pi_f0g1_swap", sigma=self.sigma_test, length=self.sigma_test*4)
        elif self.cfg.expt.pulse_type.lower() == "flat_top" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[self.qubit], name="pi_f0g1_swap", sigma=3, length=3*4)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        self.reset_and_sync()

        # ================= #
        # Initial states
        # ================= #

        # print('WARNING not actually initializing the state')
        # initialize qDrive to F: expect to end in F0
        self.X_pulse(q=self.qubit, play=True)
        self.Xef_pulse(q=self.qubit, play=True)
        self.sync_all()

        # ================= #
        # Do the pulse
        # ================= #

        # apply F0 -> G1 pulse on qDrive: expect to end in G1
        if self.sigma_test > 0:
            pulse_type = cfg.expt.pulse_type.lower()
            if pulse_type == "gauss":
                self.setup_and_pulse(ch=self.swap_chs[self.qubit], style="arb", freq=self.f_f0g1_reg, phase=0, gain=cfg.expt.gain, waveform="pi_f0g1_swap") #, phrst=1)
            elif pulse_type == 'flat_top':
                sigma_ramp_cycles = 3
                if 'sigma_ramp_cycles' in self.cfg.expt:
                    sigma_ramp_cycles = self.cfg.expt.sigma_ramp_cycles
                flat_length_cycles = self.sigma_test - sigma_ramp_cycles*4
                # print(cfg.expt.gain, flat_length, self.f_f0g1_reg)
                if flat_length_cycles >= 3:
                    self.setup_and_pulse(
                        ch=self.swap_chs[self.qubit],
                        style="flat_top",
                        freq=self.f_f0g1_reg,
                        phase=0,
                        gain=cfg.expt.gain,
                        length=flat_length_cycles,
                        waveform="pi_f0g1_swap",
                    )
            else: # const
                self.setup_and_pulse(ch=self.swap_chs[self.qubit], style="const", freq=self.f_f0g1_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test) #, phrst=1)
        self.sync_all()

        wait_time_us = 0
        if 'wait_time_us' in self.cfg.expt:
            wait_time_us = self.cfg.expt.wait_time_us

        self.sync_all(self.us2cycles(wait_time_us))

        setup_measure = None
        if 'setup_measure' in self.cfg.expt: setup_measure = self.cfg.expt.setup_measure

        # take qDrive g->e: measure the population of just the e state when e/f are not distinguishable by checking the g population
        if setup_measure == 'qDrive_ge':
            # print('playing ge pulse')
            self.X_pulse(q=self.qubit, play=True)
        
        if setup_measure == None: pass # measure the real g population only

        # # take qDrive f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        # # if setup_measure == 'qDrive_ef':
        # self.setup_and_pulse(ch=self.qubit_chs[qDrive], style="arb", freq=self.f_ef_regs[qDrive], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qDrive], waveform=f"pi_ef_qubit{qDrive}") #, phrst=1)

        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

# ===================================================================== #
        
class LengthRabiF0G1Experiment(Experiment):
    """
    Length Rabi f0g1 Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step, 
        expts: number of different length experiments, 
        reps: number of reps,
        gain: gain to use for the qubit pulse
        pulse_type: 'gauss' or 'const'
        qubit: qubit goes f->g, resonator of qubit goes 0->1
        singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiF0G1', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit

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
        
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], 'counts_calib':[], 'counts_raw':[]}

        # ================= #
        # Get single shot calibration for 2 qubits
        # ================= #
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'post_process' not in self.cfg.expt.keys(): # threshold or scale
            self.cfg.expt.post_process = None

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
                sscfg.expt.tomo_qubits = [self.cfg.expt.qubit]

                calib_prog_dict = dict()
                calib_order = ['gg', 'ge', 'eg', 'ee']
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state:err_tomo})

                g_prog = calib_prog_dict['gg']
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                for qi, q in enumerate(sscfg.expt.tomo_qubits):
                    calib_e_state = 'gg'
                    calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                    e_prog = calib_prog_dict[calib_e_state]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                    thresholds_q[q] = threshold[0]
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

                if debug:
                    print(f'thresholds={thresholds_q}')
                    print(f'angles={angles_q}')
                    print(f'ge_avgs={ge_avgs_q}')

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_calib'].append(counts)
                data['counts_calib'] = np.array(data['counts_calib'])
                # print(data['counts_calib'])

            data['thresholds'] = thresholds_q
            data['angles'] = angles_q
            data['ge_avgs'] = ge_avgs_q

        # ================= #
        # Begin actual experiment
        # ================= #

        if 'gain' not in self.cfg.expt:
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1.gain[self.qubit]
        if 'pulse_type' not in self.cfg.expt:
            self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[self.qubit]

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            assert not self.cfg.expt.measure_f, 'not implemented measure f'
            self.cfg.expt.setup_measure = 'qDrive_ef' # measure g vs. f (e)
            lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
            # print(lengthrabi)
            # from qick.helpers import progs2json
            # print(progs2json([lengthrabi.dump_prog()]))
            avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

            adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.qubit]
            data['avgi'].append(avgi[adc_ch])
            data['avgq'].append(avgq[adc_ch])
            data['amps'].append(np.abs(avgi[adc_ch]+1j*avgi[adc_ch]))
            data['phases'].append(np.angle(avgi[adc_ch]+1j*avgi[adc_ch]))

            # else:
            #     adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.qubit]
            #     self.cfg.expt.setup_measure = 'qDrive_ef' # measure g vs. f (e)
            #     lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
            #     popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

            #     gpop = 1 - popln[adc_ch]
            #     # in Eg (swap failed) or Gf (swap succeeded)
            #     if self.cfg.expt.post_process == 'threshold':
            #         shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
            #         # 00, 01, 10, 11
            #         counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
            #         data['counts_raw'][0].append(counts)
            #         counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
            #         counts = counts[0] # go back to just 1d array
            #         if qDrive == qB:
            #             epop_qNotDrive = (counts[2] + counts[3])/sum(counts)
            #             gpop_qDrive = (counts[0] + counts[2])/sum(counts)
            #         else: # qDrive = qA
            #             epop_qNotDrive = (counts[1] + counts[3])/sum(counts)
            #             gpop_qDrive = (counts[0] + counts[1])/sum(counts)


            #     self.cfg.expt.setup_measure = 'qDrive_ge' # measure e population
            #     lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
            #     popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

            #     epop_qDrive = 1 - popln[adcDrive_ch]
            #     if self.cfg.expt.post_process == 'threshold':
            #         shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
            #         # 00, 01, 10, 11
            #         counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
            #         data['counts_raw'][1].append(counts)
            #         # print('pre correct', counts)
            #         counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
            #         # print(counts)
            #         counts = counts[0] # go back to just 1d array
            #         if qDrive == qB:
            #             epop_qDrive = (counts[0] + counts[2])/sum(counts) # e population shows up as g population
            #         else: # qDrive = qA
            #             epop_qDrive = (counts[0] + counts[1])/sum(counts)
            #     fpop_qDrive = 1 - epop_qDrive - gpop_qDrive
            #     # print(gpop_qB, epop_qB, fpop_qB)

            #     if qDrive == qB:
            #         epop_qA = epop_qNotDrive
            #         fpop_qA = np.zeros_like(epop_qA)
            #         epop_qB = epop_qDrive
            #         fpop_qB = fpop_qDrive
            #     else:
            #         epop_qA = epop_qDrive
            #         fpop_qA = fpop_qDrive
            #         epop_qB = epop_qNotDrive
            #         fpop_qB = np.zeros_like(epop_qB)

            #     data['avgi'][0].append(epop_qA)
            #     data['avgq'][0].append(fpop_qA)

            #     data['avgi'][1].append(epop_qB)
            #     data['avgq'][1].append(fpop_qB) 
        
            data['xpts'].append(length)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True):
        if data is None:
            data=self.data
        if fit:
            # fitparams=[yscale, freq, phase_deg, decay, y0]
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None
            fitparams = [None, 2/data['xpts'][-1], None, None, None]

            try:
                p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'], data["avgi"], fitparams=fitparams)
                data[f'fit_avgi'] = p_avgi
            except Exception as e: print('Exception:', e)
            try:
                p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'], data["avgq"], fitparams=fitparams)
                data[f'fit_avgq'] = p_avgq
            except Exception as e: print('Exception:', e)
            # p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'], data["amps"][0], fitparams=None)

            if not self.cfg.expt.measure_f:
                # data[f'fit{q_name}_amps'] = p_amps
                data[f'fit_err_avgi'] = pCov_avgi   
                data[f'fit_err_avgq'] = pCov_avgq
                # data[f'fit{q_name}_err_amps'] = pCov_amps

        return data

    def display(self, data=None, fit=True):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3

        pi_lens = []

        rows = 2
        cols = 1
        index = rows*100 + cols*10
        plt.figure(figsize=(7*cols,8))

        plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        this_idx = index + 1
        plt.subplot(this_idx, ylabel='Population' if self.cfg.expt.post_process else "I [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, fit=fit)
        pi_lens.append(pi_len) 
        
        this_idx = index + cols + 1
        plt.subplot(this_idx, xlabel="Length [ns]", ylabel='Population' if self.cfg.expt.post_process else "Q [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, fit=fit)
        pi_lens.append(pi_len) 

        plt.tight_layout()
        plt.show()

        return pi_lens


    """
    q_index is the index in measure_qubits
    """
    def plot_rabi(self, data, data_name, fit_xpts, plot_xpts, fit=True):
        plt.plot(plot_xpts, data[data_name],'o-')
        pi_length=None
        if fit: 
            if f'fit_{data_name}' not in data: return None
            p = data[f'fit_{data_name}']
            plt.plot(plot_xpts, fitter.decaysin(fit_xpts, *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length = (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data [us]: {pi2_length}')
            print(f'\tDecay time [us]: {p[3]}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        if self.cfg.expt.post_process is not None:
            if np.max(data[data_name]) - np.min(data[data_name]) > 0.2:
                plt.ylim(-0.1, 1.1)
                # print(data_name, q_name)
        return pi_length



    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ===================================================================== #

class F0G1FreqLenChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping freq vs. len
    Experimental Config:
    expt = dict(
        start_len: start length [us],
        step_len: length step, 
        expts_len: number of different length experiments, 
        start_f: start freq [MHz],
        step_f: freq step, 
        expts_f: number of different freq experiments, 
        gain: gain to use for the qubit pulse
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """
    def __init__(self, soccfg=None, path='', prefix='RabiF0G1FreqLenChevron', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit

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

        freqpts = self.cfg.expt.start_f + self.cfg.expt.step_f * np.arange(self.cfg.expt.expts_f)
        lenpts = self.cfg.expt.start_len + self.cfg.expt.step_len * np.arange(self.cfg.expt.expts_len)
        
        data={"lenpts":lenpts, "freqpts":freqpts, "avgi":[], "avgq":[], "amps":[], "phases":[]}

        self.cfg.expt.start = self.cfg.expt.start_len
        self.cfg.expt.step = self.cfg.expt.step_len
        self.cfg.expt.expts = self.cfg.expt.expts_len

        if 'gain' not in self.cfg.expt:
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1.gain[self.qubit]
        if 'pulse_type' not in self.cfg.expt:
            self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[self.qubit]

        expt_prog = LengthRabiF0G1Experiment(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file) 
        expt_prog.cfg.expt = self.cfg.expt

        start_time = time.time()
        for freq in tqdm(freqpts, disable=not progress): 
            expt_prog.cfg.device.qubit.f_f0g1[self.qubit] = float(freq)
            expt_prog.go(analyze=False, display=False, progress=False, save=False)
            data['avgi'].append(expt_prog.data['avgi'])
            data['avgq'].append(expt_prog.data['avgq'])
            data['amps'].append(expt_prog.data['amps'])
            data['phases'].append(expt_prog.data['phases'])
            if time.time() - start_time < 600 and expt_prog.cfg.expt.post_process is not None: # redo the single shot calib every 10 minutes
                expt_prog.cfg.expt.thresholds = expt_prog.data['thresholds']
                expt_prog.cfg.expt.angles = expt_prog.data['angles']
                expt_prog.cfg.expt.ge_avgs = expt_prog.data['ge_avgs']
                expt_prog.cfg.expt.counts_calib = expt_prog.data['counts_calib']
            else:
                start_time = time.time()
                expt_prog.cfg.expt.thresholds = None
                expt_prog.cfg.expt.angles = None
                expt_prog.cfg.expt.ge_avgs = None
                expt_prog.cfg.expt.counts_calib = None

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fitparams=None, verbose=True):
        if data is None:
            data=self.data
        data = deepcopy(data)
        inner_sweep = data['lenpts']
        outer_sweep = data['freqpts']

        y_sweep = outer_sweep # index 0
        x_sweep = inner_sweep # index 1

        # fitparams = [yscale, freq, phase_deg, y0]
        # fitparams=[None, 2/x_sweep[-1], None, None]
        for data_name in ['avgi', 'avgq']:
            data.update({f'fit{data_name}':[None]})
            data.update({f'fit{data_name}_err':[None]})
            data.update({f'data_fit{data_name}':[None]})

            this_data = data[data_name]

            fit = [None]*len(y_sweep)
            fit_err = [None]*len(y_sweep)
            data_fit = [None]*len(y_sweep)

            for i_freq, freq in enumerate(y_sweep):
                try:
                    p, pCov = fitter.fitsin(x_sweep, this_data[i_freq, :], fitparams=fitparams)
                    fit[i_freq] = p
                    fit_err[i_freq] = pCov
                    data_fit[i_freq] = fitter.sinfunc(x_sweep, *p)
                except Exception as e: print('Exception:', e)

            data[f'fit{data_name}'] = fit
            data[f'fit{data_name}_err'] = fit_err
            data[f'data_fit{data_name}'] = data_fit


        # for k, a in data.items():
        #     data[k] = np.array(a)
        #     if np.shape(data[k]) == (2, len(y_sweep) * len(x_sweep)):
        #         data[k] = np.reshape(data[k], (2, len(y_sweep), len(x_sweep)))
        return data

    def display(self, data=None, fit=True, signs=[[1,1],[1,1]], plot_freq=None, plot_len=None, verbose=True, saveplot=False):
        if data is None:
            data=self.data 

        data = deepcopy(data)
        inner_sweep = data['lenpts']
        outer_sweep = data['freqpts']

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        if saveplot: plt.style.use('dark_background')

        plot_lens = []
        plot_freqs = []

        rows = 2
        cols = 1
        index = rows*100 + cols*10
        plt.figure(figsize=(8*cols,8))
        plt.suptitle(f"F0G1 Chevron Frequency vs. Length (Gain {self.cfg.expt.gain}, Q{self.cfg.expt.qubit})")

        # ------------------------------ #
        q_index = 0

        this_idx = index + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgi'
        fit_plot_freq, fit_plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, plot_rabi=(plot_freq or plot_len), verbose=verbose, plot_len=plot_len, plot_freq=plot_freq, sign=signs[0])

        this_idx = index + cols + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgq'
        fit_plot_freq, fit_plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, plot_rabi=(plot_freq or plot_len), verbose=verbose, plot_len=plot_len, plot_freq=plot_freq, sign=signs[1])

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_F0G1{self.cfg.expt.qubit}.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()

        # ------------------------------------------ #
        # ------------------------------------------ #
        """
        Plot fit chevron

        Calculate max/min
        display signs: [QA I, QB I, QA Q, QB Q]
        plot_freq, plot_len index: [QA I, QA Q, QB I, QB Q]
        """ 
        
        if saveplot: plt.style.use('dark_background')
        plt.figure(figsize=(7*cols,8))
        plt.suptitle(f"F0G1 Chevron Frequency vs. Length Fit (Gain {self.cfg.expt.gain}, Q{self.cfg.expt.qubit})")

        # ------------------------------ #
        this_idx = index + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgi'
        fit_plot_freq, fit_plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, sign=signs[0], plot_rabi=True, verbose=verbose, plot_len=plot_len, plot_freq=plot_freq)
        plot_freqs.append(plot_freq)
        plot_lens.append(fit_plot_len*1e-3)

        this_idx = index + cols + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgq'
        fit_plot_freq, fit_plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, sign=signs[1], plot_rabi=True, verbose=verbose, plot_len=plot_len, plot_freq=plot_freq)
        plot_freqs.append(plot_freq)
        plot_lens.append(fit_plot_len*1e-3)

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_F0G1{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}_fit.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()

        return plot_freqs, plot_lens

    """
    q_index is the index in measure_qubits
    """
    def plot_rabi_chevron(self, data, data_name, plot_xpts, plot_ypts, sign=None, plot_rabi=True, verbose=True, label=None, plot_freq=None, plot_len=None):
        this_data = data[data_name]
        plt.pcolormesh(plot_xpts, plot_ypts, this_data, cmap='viridis', shading='auto')
        qubit = self.cfg.expt.qubit
        if plot_rabi:
            assert sign is not None
            if sign == 1: func = np.max
            else: func = np.min
            good_pos = np.argwhere(this_data == func(this_data))
            if plot_freq is None: plot_freq = plot_ypts[good_pos[0,0]]
            if plot_len is None: plot_len = plot_xpts[good_pos[0,1]]
            print('plot freq', plot_freq, plot_len)
            if verbose:
                if sign == 1:
                    print(f'max q{qubit} {data_name}', np.max(this_data))
                else:
                    print(f'min q{qubit} {data_name}', np.min(this_data))
                print(good_pos)
                print(f'Q{qubit} {data_name} freq', plot_freq, 'len', plot_len)
            plt.axhline(plot_freq, color='r', linestyle='--')
            plt.axvline(plot_len, color='r', linestyle='--')
        if label is not None:
            if self.cfg.expt.post_process is not None:
                plt.colorbar(label=f'Population {data_name}')
            else: plt.colorbar(label='$S_{21}$'+ f' {data_name} [ADC level]')
        else: plt.colorbar().set_label(label=data_name, size=15) 
        if self.cfg.expt.post_process is not None: plt.clim(0, 1)
        return plot_freq, plot_len


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

