import yaml
from slab import AttrDict
from functools import reduce


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def load(file_name):
    with open(file_name,'r') as file:
      auto_cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 
    return auto_cfg

def save(cfg, file_name):
    # dump it: 
    cfg= yaml.safe_dump(cfg.to_dict(), default_flow_style=  True)

    # write it: 
    with open(file_name, 'w') as modified_file:
        modified_file.write(cfg)

    # now, open the modified file again 
    with open(file_name,'r') as file:
      cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 
    
    return cfg


def recursive_get(d, keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

def update_qubit(file_name, field, value, qubit_i, verbose=True):
    cfg=load(file_name)
    if isinstance(field, tuple):
        v=recursive_get(cfg['device']['qubit'], field)
        old_value=v[qubit_i]
        v[qubit_i]=value
        nested_set(cfg['device']['qubit'], field, v)
        if verbose: 
            print(f'*Set cfg qubit {qubit_i} {field} to {value} from {old_value}*')
    else:
        old_value = cfg['device']['qubit'][field][qubit_i]
        cfg['device']['qubit'][field][qubit_i] = value
        if verbose: 
            print(f'*Set cfg qubit {qubit_i} {field} to {value} from {old_value}*')
    save(cfg, file_name)

    return cfg 

def update_readout(file_name, field, value, qubit_i, verbose=True):
    cfg=load(file_name)
    old_value = cfg['device']['readout'][field][qubit_i]
    cfg['device']['readout'][field][qubit_i] = value
    save(cfg, file_name)

    print(f'*Set cfg resonator {qubit_i} {field} to {value} from {old_value}*')
    return cfg 

def init_config(file_name, num_qubits):
        
    auto_cfg.device.qubit.T1 = []
    auto_cfg.device.qubit.f_ge = [] 
    auto_cfg.device.qubit.f_EgGf = []
    auto_cfg.device.qubit.f_ef = []
    auto_cfg.device.qubit.kappa = []
    auto_cfg.device.qubit.pulses.hpi_ge.gain = [] 
    auto_cfg.device.qubit.pulses.hpi_ge.sigma = [] 
    auto_cfg.device.qubit.pulses.pi_ge.gain = [] 
    auto_cfg.device.qubit.pulses.pi_ge.sigma = [] 
    auto_cfg.device.qubit.pulses.pi_EgGf.gain = [] 
    auto_cfg.device.qubit.pulses.pi_EgGf.sigma = []
    auto_cfg.device.qubit.pulses.pi_ef.gain = []
    auto_cfg.device.qubit.pulses.pi_ef.sigma = []
    auto_cfg.device.readout.Max_amp = [] 
    auto_cfg.device.readout.frequency = [] 
    auto_cfg.device.readout.gain = [] 
    auto_cfg.device.readout.phase = [] 
    auto_cfg.device.readout.readout_length = [] 
    auto_cfg.device.readout.threshold = [] 
    auto_cfg.device.readout.kappa = []  

    for i in range(num_qubits):      
        auto_cfg.device.qubit.T1.append(100)
        auto_cfg.device.qubit.f_ge.append(3300)
        auto_cfg.device.qubit.f_EgGf.append(2000)
        auto_cfg.device.qubit.f_ef.append(4000)
        auto_cfg.device.qubit.kappa.append(0)
        auto_cfg.device.qubit.pulses.hpi_ge.gain.append(500)
        auto_cfg.device.qubit.pulses.hpi_ge.sigma.append(0.20)
        auto_cfg.device.qubit.pulses.pi_ge.gain.append(1000)
        auto_cfg.device.qubit.pulses.pi_ge.sigma.append(0.01)
        auto_cfg.device.qubit.pulses.pi_EgGf.gain = int(10000)
        auto_cfg.device.qubit.pulses.pi_EgGf.sigma = 0.1
        auto_cfg.device.qubit.pulses.pi_ef.gain = int(10000)
        auto_cfg.device.qubit.pulses.pi_ef.sigma = 0.1
        auto_cfg.device.readout.Max_amp.append(int(1))
        auto_cfg.device.readout.frequency.append(7000)
        auto_cfg.device.readout.gain.append(int(15000))
        auto_cfg.device.readout.phase.append(0)
        auto_cfg.device.readout.readout_length.append(5)
        auto_cfg.device.readout.threshold.append(0)
        auto_cfg.device.readout.kappa.append(0)

    # # dump it: 
    auto_cfg= yaml.safe_dump(auto_cfg.to_dict(), default_flow_style=  True)

    # # write it: 
    with open(file_name, 'w') as modified_file:
        modified_file.write(auto_cfg)

    # now, open the modified file again 
    with open(file_name,'r') as file:
        auto_cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 