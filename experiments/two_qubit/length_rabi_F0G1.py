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
from experiments.clifford_averager_program import CliffordAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram, ErrorMitigationStateTomo2QProgram, sort_counts, correct_readout_err, fix_neg_counts

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiF0G1Program(CliffordAveragerProgram):
    def initialize(self):
        super().initialize()
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

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
            self.f_f0g1_reg = self.freq2reg(self.cfg.device.qubit.f_f0g1[qSort], gen_ch=self.swap_chs[qSort])
        else:
            self.swap_chs = self.cfg.hw.soc.dacs.swap_Q.ch
            self.swap_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_Q.mixer_freq
            self.f_f0g1_reg = self.freq2reg(self.cfg.device.qubit.f_f0g1_Q[qSort], gen_ch=self.swap_chs[qSort])

        mixer_freq = 0
        if self.swap_ch_types[qSort] == 'int4':
            mixer_freq = mixer_freqs[qSort]
        if self.swap_chs[qSort] not in self.gen_chs: 
            self.declare_gen(ch=self.swap_chs[qSort], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qSort], mixer_freq=mixer_freq)
        # else: print(self.gen_chs[self.swap_chs[qSort]]['nqz'])

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.swap_chs[qSort])

        # add swap pulse
        if self.cfg.expt.pulse_type.lower() == "gauss" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qSort], name="pi_f0g1_swap", sigma=self.sigma_test, length=self.sigma_test*4)
        elif self.cfg.expt.pulse_type.lower() == "flat_top" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qSort], name="pi_f0g1_swap", sigma=3, length=3*4)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qDrive = self.qDrive
        qNotDrive = self.qNotDrive
        qSort = self.qSort

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            # self.sync_all()
        self.sync_all(10)


        # ================= #
        # Initial states
        # ================= #

        # initialize qDrive to F: expect to end in F0
        self.X_pulse(q=qDrive, play=True)
        self.setup_and_pulse(ch=self.qubit_chs[qDrive], style="arb", phase=0, freq=self.f_ef_regs[qDrive], gain=cfg.device.qubit.pulses.pi_ef.gain[qDrive], waveform=f"pi_ef_qubit{qDrive}")
        self.sync_all(5)

        # ================= #
        # Do the pulse
        # ================= #

        # apply F0 -> G1 pulse on qDrive: expect to end in G1
        if self.sigma_test > 0:
            pulse_type = cfg.expt.pulse_type.lower()
            if pulse_type == "gauss":
                self.setup_and_pulse(ch=self.swap_chs[qSort], style="arb", freq=self.f_f0g1_reg, phase=0, gain=cfg.expt.gain, waveform="pi_f0g1_swap") #, phrst=1)
            elif pulse_type == 'flat_top':
                sigma_ramp_cycles = 3
                if 'sigma_ramp_cycles' in self.cfg.expt:
                    sigma_ramp_cycles = self.cfg.expt.sigma_ramp_cycles
                flat_length_cycles = self.sigma_test - sigma_ramp_cycles*4
                # print(cfg.expt.gain, flat_length, self.f_f0g1_reg)
                if flat_length_cycles >= 3:
                    self.setup_and_pulse(
                        ch=self.swap_chs[qSort],
                        style="flat_top",
                        freq=self.f_f0g1_reg,
                        phase=0,
                        gain=cfg.expt.gain,
                        length=flat_length_cycles,
                        waveform="pi_f0g1_swap",
                    )
            else: # const
                self.setup_and_pulse(ch=self.swap_chs[qSort], style="const", freq=self.f_f0g1_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test) #, phrst=1)
        self.sync_all(5)

        setup_measure = None
        if 'setup_measure' in self.cfg.expt: setup_measure = self.cfg.expt.setup_measure

        # take qDrive g->e: measure the population of just the e state when e/f are not distinguishable by checking the g population
        if setup_measure == 'qDrive_ge':
            # print('playing ge pulse')
            self.X_pulse(q=qDrive, play=True)
            self.sync_all(5)
        
        if setup_measure == None: pass # measure the real g population only

        # # take qDrive f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        # # if setup_measure == 'qDrive_ef':
        # self.setup_and_pulse(ch=self.qubit_chs[qDrive], style="arb", freq=self.f_ef_regs[qDrive], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qDrive], waveform=f"pi_ef_qubit{qDrive}") #, phrst=1)

        
        self.sync_all(5)
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
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
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
        singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiF0G1', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1: qSort = qB
        qDrive = 1
        if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
            qDrive = self.cfg.expt.qDrive
        qNotDrive = -1
        if qA == qDrive: qNotDrive = qB
        else: qNotDrive = qA

        if 'measure_qubits' not in self.cfg.expt: self.cfg.expt.measure_qubits = [qA, qB]

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
        for i_q in range(len(self.cfg.expt.measure_qubits)):
            data['avgi'].append([])
            data['avgq'].append([])
            data['amps'].append([])
            data['phases'].append([])
            data['counts_raw'].append([])

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
                sscfg.expt.tomo_qubits = self.cfg.expt.qubits

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
            if qDrive == 1: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1.gain[qSort]
            else: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1_Q.gain[qSort]
        if 'pulse_type' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[qSort]
            else: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1_Q.type[qSort]
        
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            if not self.cfg.expt.measure_f:
                if self.cfg.expt.post_process is not None and len(self.cfg.expt.measure_qubits) != 2:
                    assert False, 'more qubits not implemented for measure f'
                self.cfg.expt.setup_measure = 'qDrive_ef' # measure g vs. f (e)
                lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
                # print(lengthrabi)
                # from qick.helpers import progs2json
                # print(progs2json([lengthrabi.dump_prog()]))
                avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                for i_q, q in enumerate(self.cfg.expt.measure_qubits):
                    adc_ch = self.cfg.hw.soc.adcs.readout.ch[q]
                    data['avgi'][i_q].append(avgi[adc_ch])
                    data['avgq'][i_q].append(avgq[adc_ch])
                    data['amps'][i_q].append(np.abs(avgi[adc_ch]+1j*avgi[adc_ch]))
                    data['phases'][i_q].append(np.angle(avgi[adc_ch]+1j*avgi[adc_ch]))

            else:
                assert len(self.cfg.expt.measure_qubits) == 2, 'more qubits not implemented for measure f'
                adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
                adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
                self.cfg.expt.setup_measure = 'qDrive_ef' # measure g vs. f (e)
                lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
                popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                adcDrive_ch = self.cfg.hw.soc.adcs.readout.ch[qDrive]
                adcNotDrive_ch = self.cfg.hw.soc.adcs.readout.ch[qNotDrive]

                epop_qNotDrive = popln[adcNotDrive_ch]
                gpop_qDrive = 1 - popln[adcDrive_ch]
                # in Eg (swap failed) or Gf (swap succeeded)
                if self.cfg.expt.post_process == 'threshold':
                    shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
                    # 00, 01, 10, 11
                    counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
                    data['counts_raw'][0].append(counts)
                    counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
                    counts = counts[0] # go back to just 1d array
                    if qDrive == qB:
                        epop_qNotDrive = (counts[2] + counts[3])/sum(counts)
                        gpop_qDrive = (counts[0] + counts[2])/sum(counts)
                    else: # qDrive = qA
                        epop_qNotDrive = (counts[1] + counts[3])/sum(counts)
                        gpop_qDrive = (counts[0] + counts[1])/sum(counts)


                self.cfg.expt.setup_measure = 'qDrive_ge' # measure e population
                lengthrabi = LengthRabiF0G1Program(soccfg=self.soccfg, cfg=self.cfg)
                popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                epop_qDrive = 1 - popln[adcDrive_ch]
                if self.cfg.expt.post_process == 'threshold':
                    shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
                    # 00, 01, 10, 11
                    counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
                    data['counts_raw'][1].append(counts)
                    # print('pre correct', counts)
                    counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
                    # print(counts)
                    counts = counts[0] # go back to just 1d array
                    if qDrive == qB:
                        epop_qDrive = (counts[0] + counts[2])/sum(counts) # e population shows up as g population
                    else: # qDrive = qA
                        epop_qDrive = (counts[0] + counts[1])/sum(counts)
                fpop_qDrive = 1 - epop_qDrive - gpop_qDrive
                # print(gpop_qB, epop_qB, fpop_qB)

                if qDrive == qB:
                    epop_qA = epop_qNotDrive
                    fpop_qA = np.zeros_like(epop_qA)
                    epop_qB = epop_qDrive
                    fpop_qB = fpop_qDrive
                else:
                    epop_qA = epop_qDrive
                    fpop_qA = fpop_qDrive
                    epop_qB = epop_qNotDrive
                    fpop_qB = np.zeros_like(epop_qB)

                data['avgi'][0].append(epop_qA)
                data['avgq'][0].append(fpop_qA)

                data['avgi'][1].append(epop_qB)
                data['avgq'][1].append(fpop_qB) 
        
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

            q_names = ['A', 'B', 'C']

            for i_q, q in enumerate(self.cfg.expt.measure_qubits):
                q_name = q_names[i_q]
                try:
                    p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'], data["avgi"][i_q], fitparams=fitparams)
                    data[f'fit{q_name}_avgi'] = p_avgi
                except Exception as e: print('Exception:', e)
                try:
                    p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'], data["avgq"][i_q], fitparams=fitparams)
                    data[f'fit{q_name}_avgq'] = p_avgq
                except Exception as e: print('Exception:', e)
                # p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'], data["amps"][0], fitparams=None)

                if not self.cfg.expt.measure_f:
                    # data[f'fit{q_name}_amps'] = p_amps
                    data[f'fit{q_name}_err_avgi'] = pCov_avgi   
                    data[f'fit{q_name}_err_avgq'] = pCov_avgq
                    # data[f'fit{q_name}_err_amps'] = pCov_amps

        return data

    def display(self, data=None, fit=True):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3

        pi_lens = []

        rows = 2
        cols = len(self.cfg.expt.measure_qubits)
        index = rows*100 + cols*10
        plt.figure(figsize=(7*cols,8))

        plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})', ylabel='Population' if self.cfg.expt.post_process else "I [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=0, q_name='A', fit=fit)
        pi_lens.append(pi_len) 
        
        this_idx = index + cols + 1
        plt.subplot(this_idx, xlabel="Length [ns]", ylabel='Population' if self.cfg.expt.post_process else "Q [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=0, q_name='A', fit=fit)
        pi_lens.append(pi_len) 

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        pi_len = self.plot_rabi(data=data, data_name='avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=1, q_name='B', fit=fit)
        pi_lens.append(pi_len) 

        this_idx = index + cols + 2
        plt.subplot(this_idx, xlabel="Length [ns]")
        pi_len = self.plot_rabi(data=data, data_name='avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=1, q_name='B', fit=fit)
        pi_lens.append(pi_len) 

        if self.cfg.expt.measure_f:
            print('max QA f population:', np.max(data['avgq'][0]))
            print('min QB g population:', np.min(data['avgi'][1]))


        # ------------------------------ #
        if len(self.cfg.expt.measure_qubits) == 3:
            this_idx = index + 3
            plt.subplot(this_idx, title=f'Qubit C ({self.cfg.expt.measure_qubits[2]})', ylabel="I [adc level]")
            pi_len = self.plot_rabi(data=data, data_name='avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=2, q_name='C', fit=fit)
            pi_lens.append(pi_len) 

            this_idx = index + cols + 3
            plt.subplot(this_idx, xlabel="Length [ns]", ylabel="Q [adc levels]")
            pi_len = self.plot_rabi(data=data, data_name='avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=2, q_name='C', fit=fit)
            pi_lens.append(pi_len) 


        plt.tight_layout()
        plt.show()

        return pi_lens


    """
    q_index is the index in measure_qubits
    """
    def plot_rabi(self, data, data_name, fit_xpts, plot_xpts, q_index, q_name, fit=True):
        plt.plot(plot_xpts, data[data_name][q_index],'o-')
        pi_length=None
        if fit: 
            if f'fit{q_name}_{data_name}' not in data: return None
            p = data[f'fit{q_name}_{data_name}']
            plt.plot(plot_xpts, fitter.decaysin(fit_xpts, *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length = (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data (qubit {q_name}) [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data (qubit {q_name}) [us]: {pi2_length}')
            print(f'\tDecay time [us]: {p[3]}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        if self.cfg.expt.post_process is not None:
            if np.max(data[data_name][q_index]) - np.min(data[data_name][q_index]) > 0.2:
                plt.ylim(-0.1, 1.1)
                print(data_name, q_name)
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
        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1: # convention is to reorder the indices so qA is the differentiating index, qB is 1
            qSort = qB
        self.qDrive = 1
        if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
            self.qDrive = self.cfg.expt.qDrive
        qDrive = self.qDrive

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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        freqpts = self.cfg.expt.start_f + self.cfg.expt.step_f * np.arange(self.cfg.expt.expts_f)
        lenpts = self.cfg.expt.start_len + self.cfg.expt.step_len * np.arange(self.cfg.expt.expts_len)
        
        data={"lenpts":lenpts, "freqpts":freqpts, "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i_q in range(len(self.cfg.expt.measure_qubits)):
            data['avgi'].append([])
            data['avgq'].append([])
            data['amps'].append([])
            data['phases'].append([])

        self.cfg.expt.start = self.cfg.expt.start_len
        self.cfg.expt.step = self.cfg.expt.step_len
        self.cfg.expt.expts = self.cfg.expt.expts_len

        if 'gain' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1.gain[qSort]
            else: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_f0g1_Q.gain[qSort]
        if 'pulse_type' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[qSort]
            else: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_f0g1_Q.type[qSort]

        expt_prog = LengthRabiF0G1Experiment(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file) 
        expt_prog.cfg.expt = self.cfg.expt

        start_time = time.time()
        for freq in tqdm(freqpts, disable=not progress): 
            if qDrive == 1: expt_prog.cfg.device.qubit.f_f0g1[qSort] = float(freq)
            else: expt_prog.cfg.device.qubit.f_f0g1_Q[qSort] = float(freq)
            expt_prog.go(analyze=False, display=False, progress=False, save=False)
            for q_ind, q in enumerate(self.cfg.expt.measure_qubits):
                data['avgi'][q_ind].append(expt_prog.data['avgi'][q_ind])
                data['avgq'][q_ind].append(expt_prog.data['avgq'][q_ind])
                data['amps'][q_ind].append(expt_prog.data['amps'][q_ind])
                data['phases'][q_ind].append(expt_prog.data['phases'][q_ind])
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
            if np.shape(data[k]) == (2, len(freqpts) * len(lenpts)):
                data[k] = np.reshape(data[k], (2, len(freqpts), len(lenpts)))
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
            data.update({f'fit{data_name}':[None]*len(self.cfg.expt.measure_qubits)})
            data.update({f'fit{data_name}_err':[None]*len(self.cfg.expt.measure_qubits)})
            data.update({f'data_fit{data_name}':[None]*len(self.cfg.expt.measure_qubits)})
            for q_index in range(len(self.cfg.expt.measure_qubits)):
                this_data = data[data_name][q_index]

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

                data[f'fit{data_name}'][q_index] = fit
                data[f'fit{data_name}_err'][q_index] = fit_err
                data[f'data_fit{data_name}'][q_index] = data_fit


        # for k, a in data.items():
        #     data[k] = np.array(a)
        #     if np.shape(data[k]) == (2, len(y_sweep) * len(x_sweep)):
        #         data[k] = np.reshape(data[k], (2, len(y_sweep), len(x_sweep)))
        return data

    def display(self, data=None, fit=True, plot_rabi=True, signs=[[1,1],[1,1]], verbose=True, saveplot=False):
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
        cols = len(self.cfg.expt.measure_qubits)
        index = rows*100 + cols*10
        plt.figure(figsize=(7*cols,8))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length (Gain {self.cfg.expt.gain})")

        # ------------------------------ #
        q_index = 0

        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})')
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgi'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        this_idx = index + cols + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgq'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        # ------------------------------ #
        q_index = 1

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgi'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        this_idx = index + cols + 2
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'avgq'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)


        # ------------------------------ #
        if len(self.cfg.expt.measure_qubits) == 3:
            q_index = 2

            this_idx = index + 3
            plt.subplot(this_idx, title=f'Qubit C ({self.cfg.expt.measure_qubits[2]})')
            data_name = 'avgi'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

            this_idx = index + cols + 3
            plt.subplot(this_idx, xlabel="Length [ns]")
            data_name = 'avgq'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        # ------------------------------ #

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_F0G1{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}.png'
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
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length Fit (Gain {self.cfg.expt.gain})")

        # ------------------------------ #
        q_index = 0

        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})')
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgi'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[0], plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)

        this_idx = index + cols + 1
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgq'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[1], plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)

        # ------------------------------ #
        q_index = 1

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgi'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[2], plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)

        this_idx = index + cols + 2
        plt.subplot(this_idx)
        ax = plt.gca()
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'data_fitavgq'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[3], plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)


        # ------------------------------ #
        if len(self.cfg.expt.measure_qubits) == 3:
            q_index = 2

            this_idx = index + 3
            plt.subplot(this_idx, title=f'Qubit C ({self.cfg.expt.measure_qubits[2]})')
            data_name = 'data_fitavgi'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[4], plot_rabi=True, verbose=verbose)
            plot_freqs.append(plot_freq)
            plot_lens.append(plot_len*1e-3)

            this_idx = index + cols + 3
            plt.subplot(this_idx, xlabel="Length [ns]")
            data_name = 'data_fitavgq'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=signs[5], plot_rabi=True, verbose=verbose)
            plot_freqs.append(plot_freq)
            plot_lens.append(plot_len*1e-3)

        # ------------------------------ #

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
    def plot_rabi_chevron(self, data, data_name, plot_xpts, plot_ypts, q_index, sign=None, plot_rabi=True, verbose=True, label=None):
        this_data = data[data_name][q_index]
        plt.pcolormesh(plot_xpts, plot_ypts, this_data, cmap='viridis', shading='auto')
        qubit = self.cfg.expt.measure_qubits[q_index]
        plot_len = None
        plot_freq = None
        if plot_rabi:
            assert sign is not None
            if sign == 1: func = np.max
            else: func = np.min
            good_pos = np.argwhere(this_data == func(this_data))
            plot_freq = plot_ypts[good_pos[0,0]]
            plot_len = plot_xpts[good_pos[0,1]]
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