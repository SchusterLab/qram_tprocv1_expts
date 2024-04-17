import experiments as meas

def make_tof(soc, expt_path, results_config_path, qubit_i):

    tof = meas.ToFCalibrationExperiment(soccfg=soc,
    path=expt_path,
    prefix=f"adc_trig_offset_calibration_qubit{qubit_i}",
    config_file=results_config_path)

    tof.cfg.expt = dict(pulse_length=0.5, # [us]
    readout_length=1.0, # [us]
    trig_offset=0, # [clock ticks]
    gain=30000, # blast the power just for the RFSoC calibration
    frequency=tof.cfg.device.readout.frequency[qubit_i], # [MHz]
    reps=1000, # Number of averages per point
    qubit=qubit_i) 

    tof.cfg.device.readout.relax_delay[qubit_i]=0.1 # wait time between experiments [us]
    return tof

def make_rspec_coarse(soc, expt_path, results_config_path, qubit_i, start=7000, span=250, reps=800, npts=5000):
    rspec = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_coarse",
    config_file=results_config_path,   
    )

    rspec.cfg.expt = dict(
        start = start, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]
    return rspec

def make_rspec_fine(soc, expt_path, results_config_path, qubit_i, j, center=7000, span=5, reps=500, smart=True):
    
    rspec = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_res{j}",
    config_file=results_config_path,   
    )
    npts = 1000 

    rspec.cfg.expt = dict(
        start = center-span/2, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        smart=smart,
        kappa=0.35,
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]
    return rspec

def make_rpowspec(soc, expt_path, cfg_file, qubit_i, res_freq, span_f=5, npts_f=250, span_gain=27000, start_gain=5000, npts_gain=10, reps=500, smart=False):

    rpowspec = meas.ResonatorPowerSweepSpectroscopyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ResonatorPowerSweepSpectroscopyExperiment_qubit{qubit_i}",
        config_file=cfg_file,
    )

    rpowspec.cfg.expt = dict(
        start_f = res_freq-span_f/2, # resonator frequency to be mixed up [MHz]
        step_f = span_f/npts_f, # min step ~1 Hz, 
        smart = smart, 
        expts_f=npts_f, # Number experiments stepping freq from start
        start_gain=start_gain,
        step_gain=int(span_gain/npts_gain), # Gain step size
        expts_gain=npts_gain+1, # Number experiments stepping gain from start
        reps= reps, # Number averages per point
        pulse_e=False, # add ge pi pulse before measurement
        pulse_f=False, # add ef pi pulse before measurement
        qubit=qubit_i,  
    ) 

    rpowspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]    
    rpowspec.cfg.device.readout.readout_length = 5
    return rpowspec

def make_qspec(soc, expt_path, cfg_path, qubit_i, span=None, npts=1500, reps=50, rounds=20, gain=None, coarse=True):

    
    if coarse is True and span is None:
        span=700 
        prefix = f"qubit_spectroscopy_coarse_qubit{qubit_i}"
    elif span is None:
        span=3
        prefix = f"qubit_spectroscopy_fine_qubit{qubit_i}"
    else:
        prefix = f"qubit_spectroscopy_qubit{qubit_i}"

    if coarse is True and gain is None: 
        gain=1000
    elif gain is None:
        gain=100

    qspec = meas.PulseProbeSpectroscopyExperiment(
    soccfg=soc,
    path = expt_path, 
    prefix = f"qubit_spectroscopy_coarse_qubit{qubit_i}",
    config_file=cfg_path,
    )

    qspec.cfg.expt = dict(
        start= qspec.cfg.device.qubit.f_ge[qubit_i]-span/2, # qubit frequency to be mixed up [MHz]
        step = span/npts, # min step ~1 Hz
        expts = npts, # Number experiments stepping from start
        reps = reps, # Number averages per point
        rounds = rounds, #Number of start to finish sweeps to average over 
        length = 50, # qubit probe constant pulse length [us]
        gain = gain, #qubit pulse gain  
        pulse_type = 'const', 
        #pulse_type = 'gauss',  
        qubit = qubit_i,
    ) 

    qspec.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
    return qspec

def make_lengthrabi(soc, expt_path, cfg_path, qubit_i, expts = 100, reps = 2000, gain = 1000, num_pulses = 1):

  lengthrabi = meas.LengthRabiExperiment(
      soccfg=soc,
      path=expt_path,
      prefix=f"length_rabi_qubit{qubit_i}",
      config_file=cfg_path,
  )

  lengthrabi.cfg.expt = dict(
      start =  0.0025, 
      step=  soc.cycles2us(1), # [us] this is the samllest possible step size (size of clock cycle)
      expts= expts, #51 
      reps= reps, #2000
      gain =  gain, #qubit gain [DAC units]
      #gain=lengthrabi.cfg.device.qubit.pulses.pi_ge.gain[qubit_i],
      pulse_type='gauss',
      # pulse_type='const',
      checkZZ=False,
      checkEF=False, 
      qubits=[qubit_i],
      num_pulses = 1, #number of pulses to play, must be an odd number 
  )

  return lengthrabi