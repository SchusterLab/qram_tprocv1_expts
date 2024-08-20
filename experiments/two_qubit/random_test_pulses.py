# Author: Connie 2022/02/17

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy

from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.single_qubit.single_shot import hist
from experiments.clifford_averager_program import QutritAveragerProgram, CliffordEgGfAveragerProgram

# ===================================================================== #

class MiscPulsesProgram(QutritAveragerProgram):

    def initialize(self):
        super().initialize()
        for q in range(self.num_qubits_sample):
            ch = self.cfg.hw.soc.dacs.qubit.ch[q]
            # if q == 1: continue
            print('test ch', ch)
            # self.handle_gauss_pulse(name=f'test_ch_{q}', ch=ch, sigma=self.us2cycles(0.020, gen_ch=ch), freq_MHz=self.cfg.device.qubit.f_ge[q], phase_deg=0, gain=2000, reload=True, play=False)
            self.handle_const_pulse(name=f'test_ch_{q}', ch=ch, length=self.us2cycles(0.020, gen_ch=ch), freq_MHz=self.cfg.device.qubit.f_ge[q], phase_deg=0, gain=2000, reload=True, play=False)
        self.sync_all(200)

    def body(self):
        self.reset_and_sync()
        print('begin')
        self.sync_all()

        print('test pulse')
        # Play test pulses
        for q in [0, 1, 2, 3]:
            if f'test_ch_{q}' not in self.pulse_dict.keys(): continue
            ch = self.cfg.hw.soc.dacs.qubit.ch[q]
            # self.handle_gauss_pulse(name=f'test_ch_{q}', ch=ch, reload=False, play=True, sync_after=False)
            self.handle_const_pulse(name=f'test_ch_{q}', ch=ch, reload=False, play=True, sync_after=False)
            # self.sync_all()
        # self.pulse(ch=[0, 1, 2, 3])
        self.sync_all()
        print('end test pulse')

        # print('hi')
        # self.wait_all(50)
        # print('end hi')
        # self.sync_all(50+70+70)
        self.sync_all(70)
        # self.sync_all(70)
        # self.sync_all(1000)
        for q in [0, 1, 2, 3]:
            if f'test_ch_{q}' not in self.pulse_dict.keys(): continue
            ch = self.cfg.hw.soc.dacs.qubit.ch[q]
            self.handle_const_pulse(name=f'test_ch_{q}', ch=ch, reload=False, play=True, sync_after=False)
            # self.sync_all()
        self.sync_all()


        # # Registers were cleared due to phase reset + empty pulse for measurement DACs
        # for i_ch, meas_ch in enumerate(self.measure_chs):
        #     if self.meas_ch_types[i_ch] == 'full':
        #         self.handle_const_pulse(name=f'measure{self.meas_ch_qs[i_ch]}', ch=meas_ch, play=False, set_reg=True)
        #     elif self.meas_ch_types[i_ch] == 'mux4':
        #         self.handle_mux4_pulse(name=f'measure', ch=meas_ch, play=False, set_reg=True)

        # print('delay', self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == 6)[0][0]]*1e-3)
        # self.sync_all(self.us2cycles(self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == 6)[0][0]]*1e-3))
        self.sync_all(0)
        # self.measure_chs = [6, 4]
        # self.pulse(ch=[6])

        print('measure chs', self.measure_chs)
        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=self.cfg.device.readout.trig_offset[0],
            wait=True,
            # syncdelay=self.us2cycles(max([self.cfg.device.readout.relax_delay[q] for q in self.qubits])))
            syncdelay=500)
            # syncdelay=None)
        # self.trigger(adcs=self.adc_chs, pins=None, adc_trig_offset=self.cfg.device.readout.trig_offset[0])
        # self.pulse(ch=[6], t='auto')
        # self.pulse(ch=[4], t='auto')
        # self.sync_all(500)

# ===================================================================== #

class MiscPulsesExperiment(Experiment):
    """
    Experimental Config:
    expt = dict(
        reps
        rounds
    )
    """

    def __init__(self, soccfg=None, path='', prefix='MiscPulses', config_file=None, progress=None):
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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        # ================= #
        self.cfg.expt.start = 0
        self.cfg.expt.step = 0
        self.cfg.expt.expts = 0
        prog = MiscPulsesProgram(self.soccfg, self.cfg)

        from qick.helpers import progs2json
        print(progs2json([prog.dump_prog()]))
        print()

        prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=True)

    def display(self, qubit, data=None, fit=True, **kwargs):
        pass
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
