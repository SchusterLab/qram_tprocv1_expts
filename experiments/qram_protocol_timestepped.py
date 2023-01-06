import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy

from qick import *
from qick.helpers import gauss
from slab import Experiment, AttrDict

import experiments.fitting as fitter

from experiments.single_qubit.single_shot import hist
from experiments.clifford_averager_program import CliffordAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram, ErrorMitigationStateTomo2QProgram

class QramProtocolProgram(AbstractStateTomo2QProgram):
    def initialize(self):
        super().initialize()
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        self.f_EgGf_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_EgGf, self.swap_chs)]

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        for qA in self.qubits:
            if qA == 1: continue
            mixer_freq = 0
            if self.swap_ch_types[qA] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[qA]
            if self.swap_chs[qA] not in self.prog_gen_chs: 
                self.declare_gen(ch=self.swap_chs[qA], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)
            self.prog_gen_chs.append(self.swap_chs[qA])

        # get aliases for the sigmas we need in clock cycles
        self.pi_sigmas_us = cfg.device.qubit.pulses.pi_ge.sigma
        self.pi_ef_sigmas_us = cfg.device.qubit.pulses.pi_ef.sigma
        self.pi_Q1_ZZ_sigmas_us = cfg.device.qubit.pulses.pi_Q1_ZZ.sigma
        self.pi_EgGf_sigmas_us = cfg.device.qubit.pulses.pi_EgGf.sigma

        self.pi_ge_types = self.cfg.device.qubit.pulses.pi_ge.type
        self.pi_ef_types = self.cfg.device.qubit.pulses.pi_ef.type
        self.pi_Q1_ZZ_types = self.cfg.device.qubit.pulses.pi_Q1_ZZ.type
        self.pi_EgGf_types = self.cfg.device.qubit.pulses.pi_EgGf.type

        # update timestep in outer loop over averager program
        self.timestep_us = cfg.expt.timestep

        # add qubit pulses to respective channels
        for q in self.qubits:
            # assume ge and ef pulses are gauss
            pi_sigma_cycles = self.us2cycles(self.pi_sigmas_us[q], gen_ch=self.qubit_chs[q])
            self.add_gauss(ch=self.qubit_chs[q], name=f"qubit{q}", sigma=pi_sigma_cycles, length=pi_sigma_cycles*4)
            pi_ef_sigma_cycles = self.us2cycles(self.pi_ef_sigmas_us[q], gen_ch=self.qubit_chs[q])
            self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=pi_ef_sigma_cycles, length=pi_ef_sigma_cycles*4)
            if q != 1:
                pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[q], gen_ch=self.qubit_chs[1])
                self.add_gauss(ch=self.qubit_chs[1], name=f"qubit1_ZZ{q}", sigma=pi_Q1_ZZ_sigma_cycles, length=pi_Q1_ZZ_sigma_cycles*4)
                if self.pi_EgGf_types[q] == 'gauss':
                    self.add_gauss(ch=self.swap_chs[q], name=f"pi_EgGf_swap{q}", sigma=pi_Q1_ZZ_sigma_cycles, length=pi_Q1_ZZ_sigma_cycles*4)

        self.sync_all(200)

    def handle_next_pulse(self, count_us, ch, freq_reg, type, phase, gain, sigma_us, waveform):
        if type == 'gauss':
            new_count_us = count_us + 4 * sigma_us
        else:
            new_count_us = count_us + sigma_us

        if new_count_us <= self.timestep_us: # fit entire pulse
            # print('full pulse')
            if type == 'gauss':
                self.setup_and_pulse(ch=ch, style='arb', freq=freq_reg, phase=phase, gain=gain, waveform=waveform)
            elif type == 'const':
                self.setup_and_pulse(ch=ch, style='const', freq=freq_reg, phase=phase, gain=gain, length=self.us2cycles(sigma_us, gen_ch=ch))

        elif count_us < self.timestep_us: # fit part of pulse
            cut_length_us = self.timestep_us - count_us
            # print('cut length', cut_length_us)
            if type == 'gauss' :
                sigma_cycles = self.us2cycles(cut_length_us / 4, gen_ch=ch)
                if sigma_cycles > 0:
                    # print(1e3*self.timestep_us, 'sigma cycles', sigma_cycles)
                    self.add_gauss(ch=ch, name=f"{waveform}_cut", sigma=sigma_cycles, length=4*sigma_cycles)
                    self.setup_and_pulse(ch=ch, style='arb', freq=freq_reg, phase=phase, gain=gain, waveform=f"{waveform}_cut")
            elif type == 'const':
                cut_length_cycles = self.us2cycles(cut_length_us, gen_ch=ch)
                if cut_length_cycles > 1:
                    self.setup_and_pulse(ch=ch, style='const', freq=freq_reg, phase=phase, gain=gain, length=cut_length_cycles)

        # else: already done with protocol for this timestep
        return new_count_us

    def state_prep_pulse(self, qubits=None, **kwargs):
        cfg=AttrDict(self.cfg)

        # ================= #
        # Initial states
        # ================= #

        # initialize qubit 0 to E: expect to end in eggg
        self.X_pulse(q=0, play=True)
        self.sync_all()

        # initialize qubit 1 to g+e/2: apply pi_Q1_ZZ with qB=0: expect to end in eggg + eegg
        pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[0], gen_ch=self.qubit_chs[1])
        self.add_gauss(ch=self.qubit_chs[1], name='qubit1_ZZ0_half', sigma=pi_Q1_ZZ_sigma_cycles // 2, length=2*pi_Q1_ZZ_sigma_cycles)
        self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=0, gain=cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0_half')
        self.sync_all()

        # # initialize qubit 1 to e: apply pi_Q1_ZZ with qB=0: expect to end in eegg
        # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=0, gain=cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0')
        # self.sync_all()

        self.sync_all(5)

        # ================= #
        # Begin protocol
        # ================= #

        count_us = 0
        self.end_times_us = []

        # apply Eg-Gf with qA=0: 1. eggg -> gfgg [path 1]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply pi_Q1_ZZ with qB=0: 2. eegg -> eggg [divisional pi pulse between two paths of protocol]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=self.f_Q1_ZZ_regs[0], type=self.pi_Q1_ZZ_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], sigma_us=self.pi_Q1_ZZ_sigmas_us[0], waveform='qubit1_ZZ0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=2: 3. gfgg -> ggeg [path 1]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[2], freq_reg=self.f_EgGf_regs[2], type=self.pi_EgGf_types[2], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[2], sigma_us=self.pi_EgGf_sigmas_us[2], waveform='pi_EgGf_swap2')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=0: 4. eggg -> gfgg [path 2]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply pi_Q1_ZZ with qB=2: 5. ggeg -> geeg [path 1]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=self.f_Q1_ZZ_regs[2], type=self.pi_Q1_ZZ_types[2], phase=0, gain=cfg.device.qubit.pulses.pi_Q1_ZZ.gain[2], sigma_us=self.pi_Q1_ZZ_sigmas_us[2], waveform='qubit1_ZZ2')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=3: 6. gfgg -> ggge [path 2]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[3], freq_reg=self.f_EgGf_regs[3], type=self.pi_EgGf_types[3], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[3], sigma_us=self.pi_EgGf_sigmas_us[3], waveform='pi_EgGf_swap3')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply pi_Q1_ZZ with qB=3 or qB=2: 7. ggge -> gege [path 2, which should also affect path 1: geeg -> ggeg]
        f_Q1_ZZs = self.cfg.device.qubit.f_Q1_ZZ
        avg_freq_reg = self.freq2reg(np.average((f_Q1_ZZs[2], f_Q1_ZZs[3])), gen_ch=self.qubit_chs[1])
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=avg_freq_reg, type=self.pi_Q1_ZZ_types[3], phase=0, gain=cfg.device.qubit.pulses.pi_Q1_ZZ.gain[3], sigma_us=np.average((self.pi_Q1_ZZ_sigmas_us[2], self.pi_Q1_ZZ_sigmas_us[3])), waveform='qubit1_ZZ3')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # wait any remaining time
        # print('us left', self.timestep_us-count_us)
        if count_us < self.timestep_us:
            self.sync_all(self.us2cycles(self.timestep_us - count_us))
            self.sync_all()

        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))


class QramProtocolExperiment(Experiment):
    """
    Qram protocol over time sweep
    Experimental Config
    expt = dict(
       start: start protocol time [us],
       step: time step, 
       expts: number of different time experiments, 
       reps: number of reps per time step,
       tomo_2q: True/False whether to do 2q state tomography on state at last time step
       tomo_qubits: the qubits on which to do the 2q state tomo
       singleshot_reps: reps per state for singleshot calibration
       post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
       thresholds: (optional) don't rerun singleshot and instead use this
       ge_avgs: (optional) don't rerun singleshot and instead use this
       angles: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path='', prefix='qram_protocol', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
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

        timesteps = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        data={"xpts":[], "avgi":[[],[],[],[]], "avgq":[[],[],[],[]], "avgi_err":[[],[],[],[]], "avgq_err":[[],[],[],[]], "amps":[[],[],[],[]], "phases":[[],[],[],[]]}

        self.meas_order = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
        self.calib_order = ['gg', 'ge', 'eg', 'ee'] # should match with order of counts for each tomography measurement 
        self.tomo_qubits = self.cfg.expt.tomo_qubits
        data.update({'counts_tomo':[], 'counts_calib':[]})
        calib_prog_dict = dict()

        # ================= #
        # Get single shot calibration for all 4 qubits
        # ================= #

        post_process = self.cfg.expt.post_process
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and not self.cfg.expt.tomo_2q:
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
            print('Re-using provided angles, thresholds, ge_avgs')
        else:
            thresholds_q = [0]*4
            ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_q = [0]*4
            fids_q = [0]*4
            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps

            # g states for q0, q1
            sscfg.expt.qubits = [0, 1]
            sscfg.expt.state_prep_kwargs = dict(prep_state='gg')
            err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
            err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
            Ig, Qg = err_tomo.get_shots(verbose=False)
            calib_prog_dict.update({'gg':err_tomo})

            # e states for q0, q1
            for q, prep_state in enumerate(['eg', 'ge']):
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                Ie, Qe = err_tomo.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f'Qubit  ({q})')
                fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                thresholds_q[q] = threshold[0]
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                angles_q[q] = angle
                fids_q[q] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')
                if self.cfg.expt.tomo_2q and q in self.tomo_qubits:
                    calib_state = 'gg'
                    qi = np.where(np.array(self.tomo_qubits)==q)[0][0]
                    calib_state = calib_state[:qi] + 'e' + calib_state[qi+1:]
                    calib_prog_dict.update({calib_state:err_tomo})

            # g states for q2, q3
            sscfg.expt.qubits = [2, 3]
            sscfg.expt.state_prep_kwargs = dict(prep_state='gg')
            err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
            err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
            Ig, Qg = err_tomo.get_shots(verbose=False)

            # e states for q2, q3
            for q, prep_state in enumerate(['eg', 'ge'], start=2):
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                Ie, Qe = err_tomo.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f'Qubit  ({q})')
                fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                thresholds_q[q] = threshold[0]
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                angles_q[q] = angle
                fids_q[q] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')
                if self.cfg.expt.tomo_2q and q in self.tomo_qubits:
                    calib_state = 'gg'
                    qi = np.where(np.array(self.tomo_qubits)==q)[0][0]
                    calib_state = calib_state[:qi] + 'e' + calib_state[qi+1:]
                    calib_prog_dict.update({calib_state:err_tomo})
            
            if self.cfg.expt.tomo_2q:
                prep_state = 'ee'
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.qubits = self.tomo_qubits
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                calib_prog_dict.update({prep_state:err_tomo})

        print(f'thresholds={thresholds_q}')
        print(f'angles={angles_q}')
        print(f'ge_avgs={ge_avgs_q}')

        # Process the shots taken for the confusion matrix with the calibration angles (for tomography)
        for prep_state in self.calib_order:
            counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
            data['counts_calib'].append(counts)

        # ================= #
        # Begin protocol stepping
        # ================= #

        adc_chs = self.cfg.hw.soc.adcs.readout.ch

        for time_i, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            self.cfg.expt.timestep = float(timestep)

            # Perform 2q state tomo only on last timestep
            if self.cfg.expt.tomo_2q and time_i == len(timesteps) - 1:
                for basis in tqdm(self.meas_order):
                    # print(basis)
                    cfg = AttrDict(deepcopy(self.cfg))
                    cfg.expt.basis = basis
                    cfg.expt.qubits = self.tomo_qubits
                    tomo_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=self.cfg)
                    tomo_prog.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                    counts = tomo_prog.collect_counts(angle=angle, threshold=threshold)
                    data['counts_tomo'].append(counts)
                    self.pulse_dict.update({basis:tomo_prog.pulse_dict})

            else:
                protocol_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq, avgi_err, avgq_err = protocol_prog.acquire_rotated(soc=self.im[self.cfg.aliases.soc], progress=False, angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=post_process)

                for q in range(4):
                    data['avgi'][q].append(avgi[adc_chs[q]])
                    data['avgq'][q].append(avgq[adc_chs[q]])
                    data['avgi_err'][q].append(avgi_err[adc_chs[q]])
                    data['avgq_err'][q].append(avgq_err[adc_chs[q]])
                    data['amps'][q].append(np.abs(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))
                    data['phases'][q].append(np.angle(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))

                data['xpts'].append(float(timestep))

        data['end_times'] = protocol_prog.end_times_us
        print('end times', protocol_prog.end_times_us)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        return data

    def display(self, data=None, err=True, **kwargs):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3

        if self.cfg.expt.singleshot:
            plt.figure(figsize=(14,8))
            plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.title(f"Qram Protocol")

            if err:
                plt.errorbar(xpts_ns, data["avgi"][0], fmt='o-', yerr=data["avgi_err"][0], label='Q0')
                plt.errorbar(xpts_ns, data["avgi"][1], fmt='o-', yerr=data["avgi_err"][1], label='Q1')
                plt.errorbar(xpts_ns, data["avgi"][2], fmt='o-', yerr=data["avgi_err"][2], label='Q2')
                plt.errorbar(xpts_ns, data["avgi"][3], fmt='o-', yerr=data["avgi_err"][3], label='Q3')

            else:
                plt.plot(xpts_ns, data["avgi"][0],'.-', label='Q0')
                plt.plot(xpts_ns, data["avgi"][1],'.-', label='Q1')
                plt.plot(xpts_ns, data["avgi"][2],'.-', label='Q2')
                plt.plot(xpts_ns, data["avgi"][3],'.-', label='Q3')

            # plt.fill_between(xpts_ns, data["avgi"][0] - data["avgi_err"][0], data["avgi"][0] + data["avgi_err"][0], color=plt_colors[0], alpha=0.4, linestyle='-', edgecolor=plt_colors[0])
            # plt.fill_between(xpts_ns, data["avgi"][1] - data["avgi_err"][1], data["avgi"][1] + data["avgi_err"][1], color=plt_colors[1], alpha=0.4, linestyle='-', edgecolor=plt_colors[1])
            # plt.fill_between(xpts_ns, data["avgi"][2] - data["avgi_err"][2], data["avgi"][2] + data["avgi_err"][2], color=plt_colors[2], alpha=0.4, linestyle='-', edgecolor=plt_colors[2])
            # plt.fill_between(xpts_ns, data["avgi"][3] - data["avgi_err"][3], data["avgi"][3] + data["avgi_err"][3], color=plt_colors[3], alpha=0.4, linestyle='-', edgecolor=plt_colors[3])

            end_times = data['end_times']
            for end_time in end_times:
                plt.axvline(1e3*end_time, color='0.4', linestyle='--')

            plt.ylim(-0.02, 1.02)
            plt.legend()
            plt.xlabel('Time [ns]')
            plt.ylabel("G/E Population")
            plt.grid(linewidth=0.3)

        else:
            plt.figure(figsize=(14,20))
            plt.subplot(421, title=f'Qubit 0', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][0],'o-')
            plt.subplot(422, title=f'Qubit 0', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][0],'o-')

            plt.subplot(423, title=f'Qubit 1', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][1],'o-')
            plt.subplot(424, title=f'Qubit 1', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][1],'o-')

            plt.subplot(425, title=f'Qubit 2', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][2],'o-')
            plt.subplot(426, title=f'Qubit 2', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][2],'o-')

            plt.subplot(427, title=f'Qubit 3', xlabel='Time [ns]', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][3],'o-')
            plt.subplot(428, title=f'Qubit 3', xlabel='Time [ns]', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][3],'o-')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname