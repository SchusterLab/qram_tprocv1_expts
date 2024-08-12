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
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram, ErrorMitigationStateTomo2QProgram, sort_counts, correct_readout_err, fix_neg_counts, infer_gef_popln_2readout

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiEgGfProgram(QutritAveragerProgram):
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
            self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf[qSort], gen_ch=self.swap_chs[qSort])
        else:
            self.swap_chs = self.cfg.hw.soc.dacs.swap_Q.ch
            self.swap_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_Q.mixer_freq
            self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf_Q[qSort], gen_ch=self.swap_chs[qSort])

        mixer_freq = None
        if self.swap_ch_types[qSort] == 'int4':
            mixer_freq = mixer_freqs[qSort]
        if self.swap_chs[qSort] not in self.gen_chs: 
            self.declare_gen(ch=self.swap_chs[qSort], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qSort], mixer_freq=mixer_freq)
        # else: print(self.gen_chs[self.swap_chs[qSort]]['nqz'])

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.swap_chs[qSort])

        # add swap pulse
        if self.cfg.expt.pulse_type.lower() == "gauss" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qSort], name="pi_EgGf_swap", sigma=self.sigma_test, length=self.sigma_test*4)
        elif self.cfg.expt.pulse_type.lower() == "flat_top" and self.cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qSort], name="pi_EgGf_swap", sigma=3, length=3*4)

        # add second (calibrated) swap pulse
        if 'qubits_simul_swap' in self.cfg.expt and self.cfg.expt.qubits_simul_swap is not None:
            assert 'qDrive_simul' in self.cfg.expt

            qA, qB = self.cfg.expt.qubits_simul_swap

            qSort_simul = qA
            if qA == 1: qSort_simul = qB
            qDrive_simul = self.cfg.expt.qDrive_simul
            qNotDrive_simul = -1
            if qA == qDrive_simul: qNotDrive_simul = qB
            else: qNotDrive_simul = qA
            self.qDrive_simul = qDrive_simul
            self.qNotDrive_simul = qNotDrive_simul
            self.qSort_simul = qSort_simul

            if qDrive_simul == 1:
                self.swap_chs_simul = self.cfg.hw.soc.dacs.swap.ch
                self.swap_ch_types_simul = self.cfg.hw.soc.dacs.swap.type
                self.f_EgGf_reg_simul = self.freq2reg(self.cfg.device.qubit.f_EgGf[qSort_simul], gen_ch=self.swap_chs_simul[qSort_simul])
                self.gain_EgGf_simul = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort_simul]
                self.type_EgGf_simul = self.cfg.device.qubit.pulses.pi_EgGf.type[qSort_simul]
                self.sigma_EgGf_cycles_simul = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qSort_simul], gen_ch=self.swap_chs_simul[qSort_simul])
            else:
                self.swap_chs_simul = self.cfg.hw.soc.dacs.swap_Q.ch
                self.swap_ch_types_simul = self.cfg.hw.soc.dacs.swap_Q.type
                self.f_EgGf_reg_simul = self.freq2reg(self.cfg.device.qubit.f_EgGf_Q[qSort_simul], gen_ch=self.swap_chs_simul[qSort_simul])
                self.gain_EgGf_simul = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort_simul]
                self.type_EgGf_simul = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qSort_simul]
                self.sigma_EgGf_cycles_simul = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort_simul], gen_ch=self.swap_chs_simul[qSort_simul])
            
            if self.type_EgGf_simul.lower() == "gauss" and self.sigma_EgGf_cycles_simul > 0:
                self.add_gauss(ch=self.swap_chs_simul[qSort_simul], name="pi_EgGf_swap_simul", sigma=self.sigma_EgGf_cycles_simul, length=self.sigma_EgGf_cycles_simul*4)
            elif self.type_EgGf_simul.lower() == "flat_top" and self.sigma_EgGf_cycles_simul > 0:
                self.add_gauss(ch=self.swap_chs_simul[qSort_simul], name="pi_EgGf_swap_simul", sigma=3, length=3*4)

        if 'n_pulses' in self.cfg.expt and self.cfg.expt.n_pulses is not None: # add pihalf initialization pulse for error amplification
            if self.cfg.expt.pulse_type.lower() == "gauss" and self.cfg.expt.sigma_test > 0:
                self.pi_half_sigma_test = self.us2cycles(self.sigma_test, gen_ch=self.swap_chs[qSort]) // 2
                self.add_gauss(ch=self.swap_chs[qSort], name="pi_EgGf_swap_half", sigma=self.pi_half_sigma_test, length=self.pi_half_sigma_test*4)
            # for flat top, use the same ramp gauss pulse for the pihalf pulse

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qDrive = self.qDrive
        qNotDrive = self.qNotDrive
        qSort = self.qSort

        self.reset_and_sync()

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if 'cool_idle' in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        # ================= #
        # Initial states
        # ================= #

        init_state = self.cfg.expt.init_state

        if init_state == '|0>|0>':
            pass

        elif init_state == '|0>|1>':
            self.Y_pulse(q=1, play=True)
            self.sync_all()

        elif init_state == '|0>|0+1>':
            self.Y_pulse(q=1, play=True, pihalf=True)
            self.sync_all()

        elif init_state == '|0>|2>':
            self.Y_pulse(q=1, play=True)
            self.Yef_pulse(q=1, play=True)
            self.sync_all()

        elif init_state == '|1>|0>':
            self.Y_pulse(q=0, play=True, pihalf=False)
            self.sync_all()

        elif init_state == '|1>|0+1>':
            if not self.cfg.expt.use_IQ_pulse:
                self.Y_pulse(q=0, play=True)
                self.sync_all()

                phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
                # phase = self.deg2reg(0, gen_ch=self.qubit_chs[1])
                freq = self.f_Q1_ZZ_regs[0]

                waveform = 'qubit1_ZZ0_half'
                sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[0]/2, gen_ch=self.qubit_chs[1])
                gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0]
                self.add_gauss(ch=self.qubit_chs[1], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq, phase=phase, gain=gain, waveform=waveform)

                # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=phase, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0]//2, waveform='qubit1_ZZ0')
                self.sync_all()

            else:
                IQ_qubits = [0, 1]
                for q in IQ_qubits:
                    # play the I + Q component for each qubit in the IQ pulse
                    self.handle_IQ_pulse(name=f'pulse_1p_Q{q}', ch=self.qubit_chs[q], sync_after=False, play=True)
                self.sync_all()

        elif init_state == '|1>|1>':
            self.Y_pulse(q=0, play=True)
            self.sync_all(0)

            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=0, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0')
            self.sync_all()

        # elif init_state == '|0+1>|0+1>':
        #     if not self.cfg.expt.use_IQ_pulse:
        #         assert False, 'not implemented!'
        #     else:
        #         IQ_qubits = [0, 1]
        #         for q in IQ_qubits:
        #             # play the I + Q component for each qubit in the IQ pulse
        #             self.handle_IQ_pulse(name=f'pulse_pp_Q{q}', ch=self.qubit_chs[q], sync_after=False, play=True)
        #         self.sync_all()

        elif init_state == '|0+1>|0>':
            self.Y_pulse(q=0, play=True, pihalf=True) # -> 0+1
            self.sync_all()


        elif init_state == '|0+1>|1>':
            if not self.cfg.expt.use_IQ_pulse:
                self.Y_pulse(q=1, play=True) # -> 1
                self.sync_all()

                phase = self.deg2reg(-90, gen_ch=self.qubit_chs[0]) # +Y/2 -> 0+1
                freq = self.f_Q_ZZ1_regs[0]

                waveform = f'qubit0_ZZ1_half'
                gain = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[0]
                # gain = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[0] // 2
                sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[0]/2, gen_ch=self.qubit_chs[0])
                self.add_gauss(ch=self.qubit_chs[0], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                self.setup_and_pulse(ch=self.qubit_chs[0], style='arb', freq=freq, phase=phase, gain=gain, waveform=waveform)
                self.sync_all()
                # print('WARNING, LONGER SYNC ON THIS PULSE') self.sync_all(self.us2cycles(0.030))

            else:
                IQ_qubits = [0, 1]
                for q in IQ_qubits:
                    # play the I + Q component for each qubit in the IQ pulse
                    self.handle_IQ_pulse(name=f'pulse_p1_Q{q}', ch=self.qubit_chs[q], sync_after=False, play=True)
                self.sync_all()


        elif init_state == '|0+i1>|0+1>':
            self.X_pulse(q=0, play=True, pihalf=True, neg=True) # -> 0+i1
            self.sync_all()

            freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_ge_regs[1]]))
            gain = int(np.average([self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
            sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0]/2, self.pi_sigmas_us[1]/2])
            pi2_sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.qubit_chs[1])
            phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
            self.add_gauss(ch=self.qubit_chs[1], name='qubit1_semiZZ0_half', sigma=pi2_sigma_cycles, length=4*pi2_sigma_cycles)
            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq_reg, phase=phase, gain=gain, waveform='qubit1_semiZZ0_half')
            self.sync_all()
        
        elif 'Q1' in init_state: # specify other qubits to prepare. it will always be specified as QxQ1_regular_state_name, with regular state name specified as |Qx>|Q1>
            assert init_state[2:5] == 'Q1_'
            q_other = int(init_state[1])
            assert q_other == 2 or q_other == 3
            init_state_other = init_state[5:]

            if init_state_other == '|0>|0>':
                pass

            elif init_state_other == '|0>|1>':
                self.Y_pulse(q=1, play=True)
                self.sync_all()

            elif init_state_other == '|0>|2>':
                self.Y_pulse(q=1, play=True)
                self.Yef_pulse(q=1, play=True)
                self.sync_all()

            elif init_state_other == '|0>|0+1>':
                self.Y_pulse(q=1, play=True, pihalf=True)
                self.sync_all()

            elif init_state_other == '|1>|0>':
                self.Y_pulse(q=q_other, play=True, pihalf=False)
                self.sync_all()

            elif init_state_other == '|2>|0>':
                self.Y_pulse(q=q_other, play=True, pihalf=False)
                self.Yef_pulse(q=q_other, play=True)
                self.sync_all()

            elif init_state_other == '|1>|0+1>':
                self.Y_pulse(q=q_other, play=True)
                self.sync_all()

                phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
                # phase = self.deg2reg(0, gen_ch=self.qubit_chs[1])
                freq = self.f_Q1_ZZ_regs[q_other]

                waveform = f'qubit1_ZZ{q_other}_half'
                sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[q_other]/2, gen_ch=self.qubit_chs[1])
                gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[q_other]
                self.add_gauss(ch=self.qubit_chs[1], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq, phase=phase, gain=gain, waveform=waveform)

                # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[q_other], phase=phase, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[q_other]//2, waveform=f'qubit1_ZZ{q_other}')
                self.sync_all()

            elif init_state_other == '|1>|1>':
                self.Y_pulse(q=q_other, play=True)
                self.sync_all(0)

                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[q_other], phase=0, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[q_other], waveform=f'qubit1_ZZ{q_other}')
                self.sync_all()

            elif init_state_other == '|0+1>|0>':
                self.Y_pulse(q=q_other, play=True, pihalf=True) # -> 0+1
                self.sync_all()

            elif init_state_other == '|0+1>|1>':
                self.Y_pulse(q=1, play=True) # -> 1
                self.sync_all()

                phase = self.deg2reg(-90, gen_ch=self.qubit_chs[q_other]) # +Y/2 -> 0+1
                freq = self.f_Q_ZZ1_regs[q_other]

                waveform = f'qubit{q_other}_ZZ1_half'
                gain = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[q_other]
                # gain = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[q_other] // 2
                sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[q_other]/2, gen_ch=self.qubit_chs[q_other])
                self.add_gauss(ch=self.qubit_chs[q_other], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                self.setup_and_pulse(ch=self.qubit_chs[q_other], style='arb', freq=freq, phase=phase, gain=gain, waveform=waveform)
                self.sync_all()

            else:
                assert False, f'Init state {init_state} not valid'
        
        else:
            assert False, f'Init state {init_state} not valid'


        # ================= #
        # Do the pulse
        # ================= #

        if self.sigma_test > 0:
            if 'n_pulses' in self.cfg.expt and self.cfg.expt.n_pulses is not None:
                n_pulses = self.cfg.expt.n_pulses

                # play the pihalf initialization for the error amplification
                pulse_type = cfg.expt.pulse_type.lower()
                if pulse_type == "gauss":
                    self.setup_and_pulse(ch=self.swap_chs[qSort], style="arb", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, waveform="pi_EgGf_swap_half")
                elif pulse_type == 'flat_top':
                    sigma_ramp_cycles = 3
                    if 'sigma_ramp_cycles' in self.cfg.expt:
                        sigma_ramp_cycles = self.cfg.expt.sigma_ramp_cycles
                    flat_length_cycles = self.sigma_test//2 - sigma_ramp_cycles*4
                    if flat_length_cycles >= 3:
                        self.setup_and_pulse(
                            ch=self.swap_chs[qSort],
                            style="flat_top",
                            freq=self.f_EgGf_reg,
                            phase=0,
                            gain=cfg.expt.gain,
                            length=flat_length_cycles,
                            waveform="pi_EgGf_swap",
                        )
                else: # const
                    self.setup_and_pulse(ch=self.swap_chs[qSort], style="const", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test//2) #, phrst=1)
                self.sync_all()
            else: n_pulses = 0.5
            if 'test_pi_half' in self.cfg.expt and self.cfg.expt.test_pi_half:
                n_pulses *= 2

            # loop over error amplification (if no amplification we just loop 1x)
            for i in range(int(2*n_pulses)):

                # do the simultaneous 2q swap
                if 'qubits_simul_swap' in self.cfg.expt and self.cfg.expt.qubits_simul_swap is not None:
                    pulse_type = self.type_EgGf_simul.lower()
                    if pulse_type == "gauss":
                        self.setup_and_pulse(ch=self.swap_chs_simul[self.qSort_simul], style="arb", freq=self.f_EgGf_reg_simul, phase=0, gain=self.gain_EgGf_simul, waveform="pi_EgGf_swap_simul")
                    elif pulse_type == 'flat_top':
                        sigma_ramp_cycles = 3
                        flat_length_cycles = self.sigma_EgGf_cycles_simul - sigma_ramp_cycles*4
                        if flat_length_cycles >= 3:
                            self.setup_and_pulse(
                                ch=self.swap_chs_simul[self.qSort_simul],
                                style="flat_top",
                                freq=self.f_EgGf_reg_simul,
                                phase=0,
                                gain=self.gain_EgGf_simul,
                                length=flat_length_cycles,
                                waveform="pi_EgGf_swap_simul",
                            )
                    else: # const
                        self.setup_and_pulse(ch=self.swap_chs_simul[self.qSort_simul], style="const", freq=self.f_EgGf_reg_simul, phase=0, gain=self.gain_EgGf_simul, length=self.sigma_EgGf_cycles_simul)
                    # DO NOT SYNC FOR SIMULTANEOUS PULSE

                # apply Eg -> Gf pulse on qDrive: expect to end in Gf
                pulse_type = cfg.expt.pulse_type.lower()
                if pulse_type == "gauss":
                    self.setup_and_pulse(ch=self.swap_chs[qSort], style="arb", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, waveform="pi_EgGf_swap") #, phrst=1)
                elif pulse_type == 'flat_top':
                    sigma_ramp_cycles = 3
                    if 'sigma_ramp_cycles' in self.cfg.expt:
                        sigma_ramp_cycles = self.cfg.expt.sigma_ramp_cycles
                    flat_length_cycles = self.sigma_test - sigma_ramp_cycles*4
                    # print(cfg.expt.gain, flat_length, self.f_EgGf_reg)
                    if flat_length_cycles >= 3:
                        self.setup_and_pulse(
                            ch=self.swap_chs[qSort],
                            style="flat_top",
                            freq=self.f_EgGf_reg,
                            phase=0,
                            gain=cfg.expt.gain,
                            length=flat_length_cycles,
                            waveform="pi_EgGf_swap",
                        )
                            #phrst=1)
                else: # const
                    self.setup_and_pulse(ch=self.swap_chs[qSort], style="const", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test) #, phrst=1)
                self.sync_all()

        setup_measure = None
        if 'setup_measure' in self.cfg.expt: setup_measure = self.cfg.expt.setup_measure

        # take qDrive g->e: measure the population of just the e state when e/f are not distinguishable by checking the g population
        if setup_measure != None and 'qDrive_ge' in setup_measure:
            # print('playing ge pulse')
            # print('doing x pulse on qDrive')
            self.X_pulse(q=qDrive, play=True)

        if setup_measure == None or len(setup_measure) == 0: pass # measure the real g population only

        # take qDrive f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        if setup_measure != None and 'qDrive_ef' in setup_measure:
            # print('doing xef pulse on qDrive')
            self.Xef_pulse(q=qDrive, play=True)

        if setup_measure != None and 'qNotDrive_ge' in setup_measure:
            # print('doing x pulse on qNotDrive')
            self.X_pulse(q=qNotDrive, play=True)
        
        
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs, 
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
        pulse_type: 'gauss' 'flat_top 'const'
        qubits: qubits to swap between
        qDrive: drive qubit
        measure_qubits: qubits to save the readout
        singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiEgGf', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits
        self.measure_f = False
        if self.cfg.expt.measure_f is not None and len(self.cfg.expt.measure_f) >= 0:
            self.measure_f = True
            assert len(self.cfg.expt.measure_f) == 1
            q_measure_f = self.cfg.expt.measure_f[0]
            q_other = qA if q_measure_f == qB else qB
            # Need to make sure qubits are in the right order for all of the calibrations if we want to measure f! Let's just rename the cfg.expt.qubits so it's easy for the rest of this.
            self.cfg.expt.qubits = [q_other, q_measure_f]
        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1: qSort = qB
        qDrive = 1
        if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
            qDrive = self.cfg.expt.qDrive
        qNotDrive = -1
        if qA == qDrive: qNotDrive = qB
        else: qNotDrive = qA

        print('qA', qA, 'qB', qB, 'qDrive', qDrive)
        
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
        
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], 'counts_calib':[], 'counts_raw':[[]]}
        for i_q in range(len(self.cfg.expt.measure_qubits)):
            data['avgi'].append([])
            data['avgq'].append([])
            data['amps'].append([])
            data['phases'].append([])
        if self.cfg.expt.measure_f is not None:
            for i in range(len(self.cfg.expt.measure_f)): # measure g of everybody, second measurement of each measure_f qubit using the g/f readout
                data['counts_raw'].append([])

        # ================= #
        # Get single shot calibration for 2 qubits
        # ================= #
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'post_process' not in self.cfg.expt.keys(): # threshold or scale
            self.cfg.expt.post_process = None


        calib_order = ['gg', 'ge', 'eg', 'ee']
        if self.measure_f: calib_order += ['gf', 'ef']
        data['calib_order'] = calib_order
            
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
                    print(f'Qubit ({q}) ge')
                    fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                    thresholds_q[q] = threshold[0]
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_calib'].append(counts)
                # print(data['counts_calib'])

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

        if 'gain' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort]
            else: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort]
        if 'pulse_type' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf.type[qSort]
            else: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qSort]
            
            
        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
       
        
        if 'loops' not in self.cfg.expt: self.cfg.expt.loops = 1
        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for length in tqdm(lengths, disable=not progress or self.cfg.expt.loops > 1):
                self.cfg.expt.sigma_test = float(length)
                # lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                if self.cfg.expt.post_process is not None and len(self.cfg.expt.measure_qubits) != 2:
                    assert False, 'more qubits not implemented for measure f'
                if not self.measure_f: self.cfg.expt.setup_measure = 'qDrive_ef'
                lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                # print(lengthrabi)
                # from qick.helpers import progs2json
                # print(progs2json([lengthrabi.dump_prog()]))
                avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                # in Eg (swap failed) or Gf (swap succeeded)
                shots, _ = lengthrabi.get_shots(angle=angles_q, threshold=thresholds_q)
                # 00, 01, 10, 11
                counts = np.array(sort_counts(shots[adcA_ch], shots[adcB_ch]))
                data['counts_raw'][0].append(counts)

                for i_q, q in enumerate(self.cfg.expt.measure_qubits):
                    adc_ch = self.cfg.hw.soc.adcs.readout.ch[q]
                    data['avgi'][i_q].append(avgi[adc_ch])
                    data['avgq'][i_q].append(avgq[adc_ch])
                    data['amps'][i_q].append(np.abs(avgi[adc_ch]+1j*avgq[adc_ch]))
                    data['phases'][i_q].append(np.angle(avgi[adc_ch]+1j*avgq[adc_ch]))

        # ================= #
        # Measure the same thing with g/f distinguishing
        # ================= #

        if self.measure_f:
            data.update({'counts_calib_f':[]})

            # ================= #
            # Get f state single shot calibration (this must be re-run if you just ran measurement with the standard readout)
            # ================= #

            thresholds_f_q = [0]*4
            gf_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_f_q = [0]*4
            fids_f_q = [0]*4

            # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps
            sscfg.expt.tomo_qubits = self.cfg.expt.qubits # the order of this was set earlier in code so 2nd qubit is the measure f qubit
            sscfg.device.readout.frequency[q_measure_f] = sscfg.device.readout.frequency_ef[q_measure_f]
            sscfg.device.readout.readout_length[q_measure_f] = sscfg.device.readout.readout_length_ef[q_measure_f]

            calib_prog_dict = dict()
            for prep_state in tqdm(calib_order):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state:err_tomo})

            g_prog = calib_prog_dict['gg']
            Ig, Qg = g_prog.get_shots(verbose=False)

            # Get readout angle + threshold for qubits to distinguish g/f on one of the qubits
            for qi, q in enumerate(sscfg.expt.tomo_qubits):
                calib_f_state = 'gg'
                calib_f_state = calib_f_state[:qi] + f'{"f" if q == q_measure_f else "e"}' + calib_f_state[qi+1:]
                f_prog = calib_prog_dict[calib_f_state]
                If, Qf = f_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=If[q], Qe=Qf[q])
                print(f'Qubit ({q}){f" gf" if q == q_measure_f else " ge"}')
                fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                thresholds_f_q[q] = threshold[0]
                gf_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(If[q]), np.average(Qf[q])]
                angles_f_q[q] = angle
                fids_f_q[q] = fid[0]
                print(f'{"gf" if q == q_measure_f else "ge"} fidelity (%): {100*fid[0]}')

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_f_q, threshold=thresholds_f_q)
                data['counts_calib_f'].append(counts)

            print(f'thresholds_f={thresholds_f_q},')
            print(f'angles_f={angles_f_q},')
            print(f'gf_avgs={gf_avgs_q},')
            print(f"counts_calib_f={np.array(data['counts_calib_f']).tolist()}")

            data['thresholds_f'] = thresholds_f_q
            data['angles_f'] = angles_f_q
            data['gf_avgs'] = gf_avgs_q
            data['counts_calib_f'] = np.array(data['counts_calib_f'])

            if 'loops' not in self.cfg.expt: self.cfg.expt.loops = 1
            for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
                for length in tqdm(lengths, disable=not progress or self.cfg.expt.loops > 1):
                    self.cfg.expt.sigma_test = float(length)

                    assert len(self.cfg.expt.measure_qubits) == 2, 'more qubits not implemented for measure f'
                    assert self.cfg.expt.post_process is not None, 'post_process cannot be None if measuring f state'
                    assert self.cfg.expt.post_process == 'threshold', 'can only bin for f state with confusion matrix properly using threshold'

                    rabi_f_cfg = AttrDict(deepcopy(self.cfg))
                    rabi_f_cfg.device.readout.frequency[q_measure_f] = rabi_f_cfg.device.readout.frequency_ef[q_measure_f]
                    rabi_f_cfg.device.readout.readout_length[q_measure_f] = rabi_f_cfg.device.readout.readout_length_ef[q_measure_f]

                    lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=rabi_f_cfg)
                    popln, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_f_q, threshold=thresholds_f_q, ge_avgs=gf_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        

                    shots, _ = lengthrabi.get_shots(angle=angles_f_q, threshold=thresholds_f_q)
                    # 00, 02, 10, 12
                    counts = np.array(sort_counts(shots[adcA_ch], shots[adcB_ch]))
                    data['counts_raw'][1].append(counts)

        data['xpts'] = lengths

        for i_q, q in enumerate(self.cfg.expt.measure_qubits):
            data['avgi'][i_q] = np.average(np.reshape(data['avgi'][i_q], (self.cfg.expt.loops, len(lengths))), axis=0)
            data['avgq'][i_q] = np.average(np.reshape(data['avgq'][i_q], (self.cfg.expt.loops, len(lengths))), axis=0)
            data['amps'][i_q] = np.average(np.reshape(data['amps'][i_q], (self.cfg.expt.loops, len(lengths))), axis=0)
            data['phases'][i_q] = np.average(np.reshape(data['phases'][i_q], (self.cfg.expt.loops, len(lengths))), axis=0)

        for icounts in range(len(data['counts_raw'])):
            data['counts_raw'][icounts] = np.sum(np.reshape(data['counts_raw'][icounts], (self.cfg.expt.loops, len(lengths), 4)), axis=0) # --> counts_raw shape is (num_meas_f + 1, num_lengths, 4)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True):

        if data is None:
            data=self.data

        qA, qB = self.cfg.expt.qubits
        self.measure_f = False
        if self.cfg.expt.measure_f is not None and len(self.cfg.expt.measure_f) >= 0:
            self.measure_f = True
            assert len(self.cfg.expt.measure_f) == 1
            q_measure_f = self.cfg.expt.measure_f[0]
            q_other = qA if q_measure_f == qB else qB
            # Need to make sure qubits are in the right order for all of the calibrations if we want to measure f! Let's just rename the cfg.expt.qubits so it's easy for the rest of this.
            self.cfg.expt.qubits = [q_other, q_measure_f]
        qA, qB = self.cfg.expt.qubits

        if self.measure_f:
            data['counts_calib_total'] = np.concatenate((data['counts_calib'], data['counts_calib_f']), axis=1)
            # print('counts calib total', np.shape(data['counts_calib_total']))
            # print(data['counts_calib_total'])
            # print('counts raw', np.shape(data['counts_raw']))
            # print(data['counts_raw'])
            data['counts_raw_total'] = np.concatenate((data['counts_raw'][0], data['counts_raw'][1]), axis=1)
            # print('counts raw total', np.shape(data['counts_raw_total']))
            data['gpop'] = np.zeros(shape=(len(self.cfg.expt.measure_qubits), len(data['xpts'])))
            data['epop'] = np.zeros(shape=(len(self.cfg.expt.measure_qubits), len(data['xpts'])))
            data['fpop'] = np.zeros(shape=(len(self.cfg.expt.measure_qubits), len(data['xpts'])))
            for ilen, length in enumerate(data['xpts']):
                gpop_q, epop_q, fpop_q = infer_gef_popln_2readout(qubits=self.cfg.expt.qubits, counts_raw_total=data['counts_raw_total'][ilen], calib_order=data['calib_order'], counts_calib=data['counts_calib_total'], fix_neg_counts_flag=True)
                for iq, q in enumerate(self.cfg.expt.measure_qubits):
                    data['gpop'][iq, ilen] = gpop_q[q]
                    data['epop'][iq, ilen] = epop_q[q]
                    data['fpop'][iq, ilen] = fpop_q[q]

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

        rows = 3
        cols = len(self.cfg.expt.measure_qubits)
        index = rows*100 + cols*10
        plt.figure(figsize=(7*cols,11))

        plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})', ylabel='Population E' if self.cfg.expt.post_process else "I [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='epop' if self.measure_f else 'avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=0, q_name='A', fit=fit)
        pi_lens.append(pi_len) 
        # if self.cfg.expt.post_process: plt.ylim(-0.1, 1.1)
        
        this_idx = index + cols + 1
        plt.subplot(this_idx, ylabel='Population F' if self.cfg.expt.post_process else "Q [adc level]")
        pi_len = self.plot_rabi(data=data, data_name='fpop' if self.measure_f else 'avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=0, q_name='A', fit=fit)
        pi_lens.append(pi_len) 
        if self.cfg.expt.post_process:
            plt.axhline(1.0)
            plt.axhline(0.0)
            plt.ylim(-0.1, 1.1)

        this_idx = index + 2*cols + 1
        plt.subplot(this_idx, xlabel="Length [ns]", ylabel='Population G' if self.cfg.expt.post_process else "1 - I - Q [adc level]")
        plt.plot(1e3*data['xpts'], data['gpop'][0] if self.measure_f else (1 - (data['avgi'][0] + data['avgq'][0])), '.-')
        plt.ylabel('Population G')
        if self.cfg.expt.post_process:
            plt.axhline(1.0)
            plt.axhline(0.0)
            plt.ylim(-0.1, 1.1)

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        pi_len = self.plot_rabi(data=data, data_name='epop' if self.measure_f else 'avgi', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=1, q_name='B', fit=fit)
        pi_lens.append(pi_len) 
        if self.cfg.expt.post_process:
            plt.axhline(1.0)
            plt.axhline(0.0)
            plt.ylim(-0.1, 1.1)

        this_idx = index + cols + 2
        plt.subplot(this_idx)
        pi_len = self.plot_rabi(data=data, data_name='fpop' if self.measure_f else 'avgq', fit_xpts=data['xpts'], plot_xpts=xpts_ns, q_index=1, q_name='B', fit=fit)
        pi_lens.append(pi_len) 
        if self.cfg.expt.post_process:
            plt.axhline(1.0)
            plt.axhline(0.0)
            plt.ylim(-0.1, 1.1)

        this_idx = index + 2*cols + 2
        plt.subplot(this_idx, xlabel="Length [ns]")
        plt.plot(1e3*data['xpts'],  data['gpop'][1] if self.measure_f else (1 - (data['avgi'][1] + data['avgq'][1])), '.-')
        plt.ylabel('Population G')
        if self.cfg.expt.post_process:
            plt.axhline(1.0)
            plt.axhline(0.0)
            plt.ylim(-0.1, 1.1)

        if self.cfg.expt.measure_f:
            print('max QA f population:', np.max(data['fpop'][0]))
            print('min QA f population:', np.min(data['fpop'][0]))
            print('mean QA f population:', np.mean(data['fpop'][0]))
            print('max QA g population:', np.max(data['gpop'][0]))
            print('min QA g population:', np.min(data['gpop'][0]))
            print('mean QA g population:', np.mean(data['gpop'][0]))
            print('max QB e population:', np.max(data['epop'][1]))
            print('min QB e population:', np.min(data['epop'][1]))
            print('mean QB e population:', np.mean(data['epop'][1]))
        
        else:
            print('max QA amp:', np.max(data['amps'][0]))
            print('min QA amp:', np.min(data['amps'][0]))
            print('max QB amp:', np.max(data['amps'][1]))
            print('min QB amp:', np.min(data['amps'][1]))
        
       

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

        # plt.figure()
        # plt.plot(data['xpts'], 1 - data['avgq'][0] - data['avgi'][1])
        # plt.ylim(-0.1, 0.5)
        # plt.ylabel('1-F(A)-E(B) Population')
        # plt.xlabel('Length [us]')
        # plt.show()

        return pi_lens


    """
    q_index is the index in measure_qubits
    """
    def plot_rabi(self, data, data_name, fit_xpts, plot_xpts, q_index, q_name, fit=True):
        plt.plot(plot_xpts, data[data_name][q_index],'.-')
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
            print(f'Pi length from {data_name} data (qubit {q_name}) [us]: {pi_length}')
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
        loops: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """
    def __init__(self, soccfg=None, path='', prefix='RabiEgGfFreqLenChevron', config_file=None, progress=None):
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
            if qDrive == 1: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort]
            else: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort]
        if 'pulse_type' not in self.cfg.expt:
            if qDrive == 1: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf.type[qSort]
            else: self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qSort]

        expt_prog = LengthRabiEgGfExperiment(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file) 
        expt_prog.cfg.expt = self.cfg.expt

        start_time = time.time()
        for freq in tqdm(freqpts, disable=not progress): 
            if qDrive == 1: expt_prog.cfg.device.qubit.f_EgGf[qSort] = float(freq)
            else: expt_prog.cfg.device.qubit.f_EgGf_Q[qSort] = float(freq)
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

            # for length in tqdm(lenpts, disable=True):
            #     self.cfg.expt.sigma_test = float(length)
            #     lenrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            #     avgi, avgq = lenrabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=False)        

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

    def analyze(self, data=None, fit=True, fitparams=None, verbose=True):
        if data is None:
            data=self.data
        if not fit: return data

        data = deepcopy(data)
        inner_sweep = data['lenpts']
        outer_sweep = data['freqpts']

        y_sweep = outer_sweep # index 0
        x_sweep = inner_sweep # index 1

        # fitparams = [yscale, freq, phase_deg, y0]
        # fitparams=[None, 2/x_sweep[-1], None, None]
        for data_name in ['avgi', 'avgq', 'amps']:
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

        rows = 1
        cols = len(self.cfg.expt.measure_qubits)
        index = rows*100 + cols*10
        plt.figure(figsize=(7*cols,6))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length (Gain {self.cfg.expt.gain})")

        # ------------------------------ #
        q_index = 0

        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})')
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.set_xlabel("Length [ns]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        data_name = 'amps'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)
        # plt.axvline(296.184847, color='r', linestyle='--')
        # plt.axhline(5890.84708333 + 4.767395490444869, color='r', linestyle='--')

        # ------------------------------ #
        q_index = 1

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel("Length [ns]", fontsize=18)
        data_name = 'amps'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        # ------------------------------ #
        if len(self.cfg.expt.measure_qubits) == 3:
            q_index = 2

            this_idx = index + 3
            plt.subplot(this_idx, xlabel='Length [ns]', title=f'Qubit C ({self.cfg.expt.measure_qubits[2]})')
            data_name = 'amps'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, plot_rabi=False, verbose=verbose)

        # ------------------------------ #

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_EgGf{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()

        # ------------------------------------------ #
        # ------------------------------------------ #
        """
        Plot fit chevron

        Calculate max/min
        plot_freq, plot_len index: [QA amps, QB amps]
        """ 
        
        if not fit: return
        if saveplot: plt.style.use('dark_background')
        plt.figure(figsize=(7*cols,6))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length Fit (Gain {self.cfg.expt.gain})")

        # ------------------------------ #
        q_index = 0

        this_idx = index + 1
        plt.subplot(this_idx, title=f'Qubit A ({self.cfg.expt.measure_qubits[0]})')
        ax = plt.gca()
        ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel("Length [ns]", fontsize=18)
        data_name = 'data_fitamps'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=1, plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)

        # ------------------------------ #
        q_index = 1

        this_idx = index + 2
        plt.subplot(this_idx, title=f'Qubit B ({self.cfg.expt.measure_qubits[1]})')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel("Length [ns]", fontsize=18)
        data_name = 'data_fitamps'
        plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=-1, plot_rabi=True, verbose=verbose)
        plot_freqs.append(plot_freq)
        plot_lens.append(plot_len*1e-3)

        # ------------------------------ #
        if len(self.cfg.expt.measure_qubits) == 3:
            q_index = 2

            this_idx = index + 3
            plt.subplot(this_idx, xlabel='Length [ns]', title=f'Qubit C ({self.cfg.expt.measure_qubits[2]})')
            data_name = 'data_fitamps'
            plot_freq, plot_len = self.plot_rabi_chevron(data=data, data_name=data_name, plot_xpts=1e3*x_sweep, plot_ypts=y_sweep, q_index=q_index, sign=1, plot_rabi=True, verbose=verbose)
            plot_freqs.append(plot_freq)
            plot_lens.append(plot_len*1e-3)

        # ------------------------------ #

        plt.tight_layout()

        if saveplot:
            plot_filename = f'len_freq_chevron_EgGf{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}_fit.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()

        return plot_freqs, plot_lens

    """
    q_index is the index in measure_qubits
    """
    def plot_rabi_chevron(self, data, data_name, plot_xpts, plot_ypts, q_index, sign=None, plot_rabi=True, verbose=True, show_cbar=True, label=None, *cbar_params):
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
        if label is None:
            if self.cfg.expt.post_process is not None: label=f'Population {data_name}'
            else: label='$S_{21}$'+ f' {data_name} [ADC level]'
        if show_cbar:
            clb = plt.colorbar(label=label)
            # clb.ax.set_title(label)
        if self.cfg.expt.post_process is not None: plt.clim(0, 1)
        return plot_freq, plot_len


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #

class NPulseEgGfExperiment(Experiment):
    """
    Play a pi/2 or pi pulse (eg-gf) variable N times 
    Experimental Config
    expt = dict(
        start: start N [us],
        step
        expts 
        reps: number of reps,
        gain
        pulse_type: 'gauss' 'flat_top' 'const' (uses config value by default)
        qubits: qubits to swap between
        qDrive: drive qubit
        measure_qubits: qubits to save the readout
        singleshot: (optional) if true, uses threshold
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

        cycles = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

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

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_calib'].append(counts)
                # print(data['counts_calib'])

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
        if qDrive == 1:
            self.cfg.expt.length = self.cfg.device.qubit.pulses.pi_EgGf.sigma[qSort]
            if 'gain' not in self.cfg.expt:
                self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort]
            if 'pulse_type' not in self.cfg.expt:
                self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf.type[qSort]
        else:
            self.cfg.expt.length = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort]
            if 'gain' not in self.cfg.expt:
                self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort]
            if 'pulse_type' not in self.cfg.expt:
                self.cfg.expt.pulse_type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qSort]
        self.cfg.expt.sigma_test = float(self.cfg.expt.length)

        if 'loops' not in self.cfg.expt: self.cfg.expt.loops = 1
        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for n_cycle in tqdm(cycles, disable=not progress or self.cfg.expt.loops > 1):
                self.cfg.expt.n_pulses = n_cycle
                assert not self.cfg.expt.measure_f, 'measure f not implemented currently'
                if self.cfg.expt.post_process is not None and len(self.cfg.expt.measure_qubits) != 2:
                    assert False, 'more qubits not implemented for measure f'
                self.cfg.expt.se9tup_measure = 'qDrive_ef' # measure g vs. f (e)
                lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = lengthrabi
                avgi, avgq = lengthrabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        
                for i_q, q in enumerate(self.cfg.expt.measure_qubits):
                    adc_ch = self.cfg.hw.soc.adcs.readout.ch[q]
                    data['avgi'][i_q].append(avgi[adc_ch])
                    data['avgq'][i_q].append(avgq[adc_ch])
                    data['amps'][i_q].append(np.abs(avgi[adc_ch]+1j*avgi[adc_ch]))
                    data['phases'][i_q].append(np.angle(avgi[adc_ch]+1j*avgi[adc_ch]))
        data["xpts"] = cycles

        for k, a in data.items():
            data[k]=np.array(a)

        data['avgi'] = np.average(np.reshape(data['avgi'], (len(self.cfg.expt.measure_qubits), self.cfg.expt.loops, len(cycles))), axis=1)
        data['avgq'] = np.average(np.reshape(data['avgq'], (len(self.cfg.expt.measure_qubits), self.cfg.expt.loops, len(cycles))), axis=1)
        data['amps'] = np.average(np.reshape(data['amps'], (len(self.cfg.expt.measure_qubits), self.cfg.expt.loops, len(cycles))), axis=1)
        data['phases'] = np.average(np.reshape(data['phases'], (len(self.cfg.expt.measure_qubits), self.cfg.expt.loops, len(cycles))), axis=1)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment
        if data is None:
            data=self.data
        if fit:
            xdata = data['xpts']
            fitparams = None
            # if self.cfg.expt.test_pi_half: fit_fitfunc = fitter.fit_probg_Xhalf
            fit_fitfunc = fitter.fit_probg_X

            q_names = ['A', 'B', 'C']

            for i_q, q in enumerate(self.cfg.expt.measure_qubits):
                q_name = q_names[i_q]
                try:
                    p_avgi, pCov_avgi = fit_fitfunc(xdata, data["avgi"][i_q], fitparams=fitparams)
                    data[f'fit{q_name}_avgi'] = p_avgi
                    data[f'fit{q_name}_err_avgi'] = pCov_avgi
                except Exception as e: print('Exception:', e)
                try:
                    p_avgq, pCov_avgq = fit_fitfunc(xdata, data["avgq"][i_q], fitparams=fitparams)
                    data[f'fit{q_name}_avgq'] = p_avgq
                    data[f'fit{q_name}_err_avgq'] = pCov_avgq
                except Exception as e: print('Exception:', e)
                # p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'], data["amps"][0], fitparams=None)
        return data

    def display(self, data=None, fit=True, scale=None):
        if data is None:
            data=self.data 

        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1: qSort = qB
        qDrive = 1
        if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
            qDrive = self.cfg.expt.qDrive
        qNotDrive = -1
        if qA == qDrive: qNotDrive = qB
        else: qNotDrive = qA

        xdata = data['xpts']
        # if self.cfg.expt.test_pi_half: fit_func = fitter.probg_Xhalf
        fit_func = fitter.probg_X

        title = f"Angle Error Q{self.cfg.expt.measure_qubits[0]} Q{self.cfg.expt.measure_qubits[1]}"

        plt.figure(figsize=(10,8))
        plt.subplot(211, title=title, ylabel=f"QA ({self.cfg.expt.measure_qubits[0]}) (scaled)")
        plot_data = data['avgi'][0]
        plt.plot(xdata, plot_data,'.-')
        if fit:
            p = data['fitA_avgi']
            pCov = data['fitA_err_avgi']
            captionStr = f'$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            # if self.cfg.expt.test_pi_half: amp_ratio = (90 + p[1])/90
            if self.cfg.expt.measure_qubits[0] == qDrive: sign = 1
            else: sign = -1
            amp_ratio = (180 - sign*p[1])/180
            print(f'From QA: adjust length to {self.cfg.expt.length / amp_ratio}')
            print(f'\tadjust ratio {amp_ratio}')
        print()
        
        # label = '($X_{\pi/2}, X_{'+ ('\pi' if not self.cfg.expt.test_pi_half else '\pi/2') + '}^{2n}$)'
        label = '($X_{\pi/2}, X_{\pi}^{2n}$)'
        plt.subplot(212, xlabel=f"Number repeated gates {label} [n]", ylabel=f"QA ({self.cfg.expt.measure_qubits[1]}) (scaled)")
        plot_data = data['avgi'][1]
        plt.plot(xdata, plot_data,'.-')
        if fit:
            p = data['fitB_avgi']
            pCov = data['fitB_err_avgi']
            captionStr = f'$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            # if self.cfg.expt.test_pi_half: amp_ratio = (90 + p[1])/90
            if self.cfg.expt.measure_qubits[1] == qDrive: sign = 1
            else: sign = -1
            amp_ratio = (180 - sign*p[1])/180
            print(f'From QB: adjust length to {self.cfg.expt.length / amp_ratio}')
            print(f'\tadjust ratio {amp_ratio}')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname