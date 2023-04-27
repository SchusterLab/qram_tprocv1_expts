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
class LengthRabiEgGfProgram(CliffordAveragerProgram):
    def initialize(self):
        super().initialize()
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits
 
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA])

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        mixer_freq = 0
        if self.swap_ch_types[qA] == 'int4':
            mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[qA]
        if self.swap_chs[qA] not in self.gen_chs: 
            self.declare_gen(ch=self.swap_chs[qA], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.swap_chs[qA])

        # add swap pulse
        if self.cfg.expt.pulse_type.lower() == "gauss" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=self.sigma_test, length=self.sigma_test*4)
        elif self.cfg.expt.pulse_type.lower() == "flat_top" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=3, length=3*4)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            self.sync_all()
        self.sync_all(10)

        # initialize qubit A to E: expect to end in Eg
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_ge_regs[qA], gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform=f"qubit{qA}") #, phrst=1)
        self.sync_all(5)

        # apply Eg -> Gf pulse on B: expect to end in Gf
        if self.sigma_test > 0:
            pulse_type = cfg.expt.pulse_type.lower()
            if pulse_type == "gauss":
                self.setup_and_pulse(ch=self.swap_chs[qA], style="arb", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, waveform="pi_EgGf_swap") #, phrst=1)
            elif pulse_type == 'flat_top':
                flat_length = self.sigma_test - 3*4
                if flat_length >= 3:
                    self.setup_and_pulse(
                        ch=self.swap_chs[qA],
                        style="flat_top",
                        freq=self.f_EgGf_reg,
                        phase=0,
                        gain=cfg.expt.gain,
                        length=flat_length,
                        waveform="pi_EgGf_swap",
                    )
                        #phrst=1)
            else: # const
                self.setup_and_pulse(ch=self.swap_chs[qA], style="const", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test) #, phrst=1)
        self.sync_all(5)

        setup_measure = None
        if 'setup_measure' in self.cfg.expt: setup_measure = self.cfg.expt.setup_measure

        # take qubit B g->e: measure the population of just the e state when e/f are not distinguishable by checking the g population
        if setup_measure == 'qB_ge':
            # print('playing ge pulse')
            self.X_pulse(q=qB, play=True)
            self.sync_all(5)
        
        if setup_measure == None: pass # measure the real g population only

        # take qubit B f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        # if setup_measure == 'qB_ef':
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef_regs[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform=f"pi_ef_qubit{qB}") #, phrst=1)

        
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
        
class LengthRabiEgGfExperiment(Experiment):
    """
    Length Rabi EgGf Experiment
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

    def __init__(self, soccfg=None, path='', prefix='LengthRabiEgGf', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

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

        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
        
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        data={"xpts":[], "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]], 'counts_calib':[]}

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
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
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
                    fid, threshold, angle = hist(data=shot_data, plot=False, verbose=False)
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
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]
        
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            # lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            if not self.cfg.expt.measure_f:
                self.cfg.expt.setup_measure = 'qB_ef' # measure g vs. f (e)
                lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                data['avgi'][0].append(avgi[adcA_ch])
                data['avgi'][1].append(avgi[adcB_ch])
                data['avgq'][0].append(avgq[adcA_ch])
                data['avgq'][1].append(avgq[adcB_ch])
                data['amps'][0].append(np.abs(avgi[adcA_ch]+1j*avgi[adcA_ch]))
                data['amps'][1].append(np.abs(avgi[adcB_ch]+1j*avgi[adcB_ch]))
                data['phases'][0].append(np.angle(avgi[adcA_ch]+1j*avgi[adcA_ch]))
                data['phases'][1].append(np.angle(avgi[adcB_ch]+1j*avgi[adcB_ch]))

            else:
                self.cfg.expt.setup_measure = None # measure g population
                lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                epop_qA = popln[adcA_ch]
                gpop_qB = 1 - popln[adcB_ch]
                if self.cfg.expt.post_process == 'threshold':
                    shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
                    # 00, 01, 10, 11
                    counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
                    counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
                    counts = counts[0] # go back to just 1d array
                    epop_qA = (counts[2] + counts[3])/sum(counts)
                    gpop_qB = (counts[0] + counts[2])/sum(counts)

                self.cfg.expt.setup_measure = 'qB_ge' # measure e population
                lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                epop_qB = 1 - popln[adcB_ch]
                if self.cfg.expt.post_process == 'threshold':
                    shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
                    # 00, 01, 10, 11
                    counts = np.array([sort_counts(shots[adcA_ch], shots[adcB_ch])])
                    # print('pre correct', counts)
                    counts = fix_neg_counts(correct_readout_err(counts, data['counts_calib']))
                    # print(counts)
                    counts = counts[0] # go back to just 1d array
                    epop_qB = (counts[0] + counts[2])/sum(counts) # e population shows up as g population
                fpop_qB = 1 - epop_qB - gpop_qB
                # print(gpop_qB, epop_qB, fpop_qB)

                data['avgi'][0].append(epop_qA) # let "avgi" be the e vs not e signal
                data['avgq'][0].append(0) # not measuring f state of qA, so just put 0

                data['avgi'][1].append(epop_qB) # let "avgi" be e vs. not e signal
                # data['avgi'][1].append(gpop_qB) # let "avgi" be g vs. not g signal
                data['avgq'][1].append(fpop_qB) # let "avgq" be f vs. not f signal
        
            data['xpts'].append(length)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None

            pA_avgi, pCovA_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][0][:-1], fitparams=None)
            pA_avgq, pCovA_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][0][:-1], fitparams=None)
            pA_amps, pCovA_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][0][:-1], fitparams=None)
            data['fitA_avgi'] = pA_avgi   
            data['fitA_avgq'] = pA_avgq

            if not self.cfg.expt.measure_f:
                data['fitA_amps'] = pA_amps
                data['fitA_err_avgi'] = pCovA_avgi   
                data['fitA_err_avgq'] = pCovA_avgq
                data['fitA_err_amps'] = pCovA_amps

            pB_avgi, pCovB_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][1][:-1], fitparams=None)
            # fitparams = [20, 1/0.6, None, None, None, None]
            pB_avgq, pCovB_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][1][:-1], fitparams=fitparams)
            pB_amps, pCovB_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][1][:-1], fitparams=None)
            data['fitB_avgi'] = pB_avgi   
            data['fitB_avgq'] = pB_avgq

            if not self.cfg.expt.measure_f:
                data['fitB_amps'] = pB_amps
                data['fitB_err_avgi'] = pCovB_avgi   
                data['fitB_err_avgq'] = pCovB_avgq
                data['fitB_err_amps'] = pCovB_amps

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3

        # plt.figure(figsize=(18,6))
        # plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        # plt.subplot(121, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Amplitude [adc level]", xlabel='Length [ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_length= (3/2 - p[2]/180)/2/p[1]
        #     pi2_length = pi_length/2
        #     print(f'Pi length from amps data (qubit A) [us]: {pi_length}')
        #     print(f'Pi/2 length from amps data (qubit A) [us]: {pi2_length}')
        #     plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
        #     plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        # plt.subplot(122, title=f'Qubit B ({self.cfg.expt.qubits[1]})', xlabel='Length[ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_length= (3/2 - p[2]/180)/2/p[1]
        #     pi2_length = pi_length/2
        #     print(f'Pi length from amps data (qubit B) [us]: {pi_length}')
        #     print(f'Pi/2 length from amps data (qubit B) [us]: {pi2_length}')
        #     plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
        #     plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="I [adc level]")
        plt.plot(xpts_ns[0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgi data (qubit A) [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data (qubit A) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.subplot(223, xlabel="Length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data (qubit A) [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data (qubit A) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.plot(xpts_ns[0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgi data (qubit B) [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data (qubit B) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.subplot(224, xlabel="Length [ns]")
        plt.plot(xpts_ns[0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data (qubit B) [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data (qubit B) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ===================================================================== #

class EgGfFreqLenChevronExperiment(Experiment):
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
    def __init__(self, soccfg=None, path='', prefix='RabiEgGfFreqLenChevron', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

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
        
        data={"lenpts":lenpts, "freqpts":freqpts, "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        self.cfg.expt.start = self.cfg.expt.start_len
        self.cfg.expt.step = self.cfg.expt.step_len
        self.cfg.expt.expts = self.cfg.expt.expts_len

        if 'gain' not in self.cfg.expt:
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]

        expt_prog = LengthRabiEgGfExperiment(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file) 
        expt_prog.cfg.expt = self.cfg.expt

        start_time = time.time()
        for freq in tqdm(freqpts, disable=not progress): 
            expt_prog.cfg.device.qubit.f_EgGf[qA] = float(freq)
            expt_prog.go(analyze=False, display=False, progress=False, save=False)
            for q_ind, q in enumerate(self.cfg.expt.qubits):
                data['avgi'][q_ind].append(expt_prog.data['avgi'][q_ind])
                data['avgq'][q_ind].append(expt_prog.data['avgq'][q_ind])
                data['amps'][q_ind].append(expt_prog.data['amps'][q_ind])
                data['phases'][q_ind].append(expt_prog.data['phases'][q_ind])
            if time.time() - start_time < 120 and expt_prog.cfg.expt.post_process is not None: # redo the single shot calib every 2 minutes
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

            # for length in tqdm(lenpts, disable=True):
            #     self.cfg.expt.sigma_test = float(length)
            #     lenrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            #     avgi, avgq = lenrabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=False, debug=debug)        

            #     for q_ind, q in enumerate(self.cfg.expt.qubits):
            #         data['avgi'][q_ind].append(avgi[adc_chs[q], 0])
            #         data['avgq'][q_ind].append(avgq[adc_chs[q], 0])
            #         data['amps'][q_ind].append(np.abs(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))
            #         data['phases'][q_ind].append(np.angle(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))

        for k, a in data.items():
            data[k] = np.array(a)
            if np.shape(data[k]) == (2, len(freqpts) * len(lenpts)):
                data[k] = np.reshape(data[k], (2, len(freqpts), len(lenpts)))
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, plot_freq=None, plot_len=None, saveplot=False, **kwargs):
        if data is None:
            data=self.data 

        inner_sweep = data['lenpts'][1:]
        outer_sweep = data['freqpts'][1:]
        data = deepcopy(data)
        data['avgi'] = (data['avgi'][0][1:, 1:], data['avgi'][1][1:, 1:])
        data['avgq'] = (data['avgq'][0][1:, 1:], data['avgq'][1][1:, 1:])

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        if saveplot: plt.style.use('dark_background')
        plt.figure(figsize=(14,10))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length")

        print('min qA', np.min(data['avgi'][0]))
        min_pos = np.argwhere(data['avgi'][0] == np.min(data['avgi'][0]))
        print(min_pos)
        plot_freq1 = y_sweep[min_pos[0,0]]
        plot_len1 = 1e3*x_sweep[min_pos[0,1]]
        print('freq', plot_freq1, 'len', plot_len1)
        if plot_freq is not None: plot_freq1 = plot_freq
        if plot_len is not None: plot_len1 = plot_len

        print('max qB', np.max(data['avgq'][1]))
        max_pos = np.argwhere(data['avgq'][1] == np.max(data['avgq'][1]))
        print(max_pos)
        plot_freq2 = y_sweep[max_pos[0,0]]
        plot_len2 = 1e3*x_sweep[max_pos[0,1]]
        print('freq', plot_freq2, 'len', plot_len2)
        if plot_freq is not None: plot_freq2 = plot_freq
        if plot_len is not None: plot_len2 = plot_len

        if saveplot:
            plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})')
            ax = plt.gca()
            ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
        else: plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Pulse Frequency [MHz]")
        # plt.pcolormesh(x_sweep, y_sweep, np.reshape(data['avgi'][0], (len(outer_sweep), len(inner_sweep))), cmap='viridis', shading='auto')
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgi'][0], cmap='viridis', shading='auto')
        if plot_freq1 is not None: plt.axhline(plot_freq1, color='r')
        if plot_len1 is not None: plt.axvline(plot_len1, color='r')
        if saveplot: plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15) 
        else:
            if self.cfg.expt.post_process is not None:
                if self.cfg.expt.measure_f: plt.colorbar(label='Population Not E -> E')
                else: plt.colorbar(label='Population G -> Not G')
            else: plt.colorbar(label='I [ADC level]')
        if self.cfg.expt.post_process is not None: plt.clim(0, 1)

        if saveplot:
            plt.subplot(223)
            ax = plt.gca()
            ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
            ax.set_xlabel("Length [ns]", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
        else: plt.subplot(223, xlabel="Length [ns]", ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgq'][0], cmap='viridis', shading='auto')
        if plot_freq1 is not None: plt.axhline(plot_freq1, color='r')
        if plot_len1 is not None: plt.axvline(plot_len1, color='r')
        if saveplot: plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15) 
        else:
            if self.cfg.expt.post_process is not None:
                if self.cfg.expt.measure_f:
                    plt.colorbar(label='Population Not F -> F')
                else: plt.colorbar(label='Population error')
            else: plt.colorbar(label='Q [ADC level]')
        if self.cfg.expt.post_process is not None: plt.clim(0, 1)


        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        if saveplot:
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=16)
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgi'][1], cmap='viridis', shading='auto')
        if plot_freq2 is not None: plt.axhline(plot_freq2, color='r')
        if plot_len2 is not None: plt.axvline(plot_len2, color='r')
        if saveplot: plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15) 
        else:
            if self.cfg.expt.post_process is not None:
                if self.cfg.expt.measure_f: plt.colorbar(label='Population Not E -> E')
                # if self.cfg.expt.measure_f: plt.colorbar(label='Population Not G -> G')
                else: plt.colorbar(label='Population G -> Not G')
            else: plt.colorbar(label='I [ADC level]')
        # if self.cfg.expt.post_process is not None: plt.clim(0, 1)
        # plt.clim(0.1, 0.4)

        if saveplot:
            plt.subplot(224)
            ax = plt.gca()
            ax.set_xlabel("Length [ns]", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
        else: plt.subplot(224, xlabel="Length [ns]")
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgq'][1], cmap='viridis', shading='auto')
        if plot_freq2 is not None: plt.axhline(plot_freq2, color='r')
        if plot_len2 is not None: plt.axvline(plot_len2, color='r')
        if saveplot: plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15) 
        else:
            if self.cfg.expt.post_process is not None:
                if self.cfg.expt.measure_f:
                    plt.colorbar(label='Population Not F -> F')
                else: plt.colorbar(label='Population error')
            else: plt.colorbar(label='Q [ADC level]')
        # if self.cfg.expt.post_process is not None: plt.clim(0, 1)
        # plt.clim(0.4, 0.8)

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_EgGf{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname