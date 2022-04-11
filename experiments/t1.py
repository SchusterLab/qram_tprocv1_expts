import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class T1Program(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.hw.soc.dacs.readout.ch[cfg.device.readout.dac]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[cfg.device.qubit.dac]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[cfg.device.readout.dac])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[cfg.device.qubit.dac])
        
        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge)
        self.f_res=self.freq2reg(cfg.device.readout.frequency) # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        # print(self.sigma)
        
        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain,
                waveform="pi_qubit")
        else:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.f_ge,
                phase=0,
                gain=cfg.expt.start,
                length=self.pi_sigma)
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)
        
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.pulse(ch=self.qubit_ch)
        self.sync_all() # align channels
        self.sync(self.q_rp, self.r_wait) # wait for the time stored in the wait variable register
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time
                      
                      
class T1Experiment(Experiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        for key, value in self.cfg.device.readout.items():
            if isinstance(value, list):
                self.cfg.device.readout.update({key: value[q_ind]})
        for key, value in self.cfg.device.qubit.items():
            if isinstance(value, list):
                self.cfg.device.qubit.update({key: value[q_ind]})
            elif isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if isinstance(value3, list):
                            value2.update({key3: value3[q_ind]})                
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.device.readout.adc]
        
        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
        
        avgi = avgi[adc_ch][0]
        avgq = avgq[adc_ch][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the first and last point from fit in case weird edge measurements
        data['fit'] = dsfit.fitexp(data['xpts'][1:-1], data['amps'][1:-1], fitparams=None, showfit=False)
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        plt.figure(figsize=(10,6))
        plt.subplot(111,title="T1", xlabel="Wait Time (us)", ylabel="Amp [adc level]")
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        if "fit" in data:
            plt.plot(data["xpts"][1:-1], dsfit.expfunc(data["fit"], data["xpts"][1:-1]))
            print(f'Fit T1 [us]: {data["fit"][3]}')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)