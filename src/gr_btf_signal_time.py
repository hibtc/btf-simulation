import numpy as np
from scipy.signal import chirp

def gr_btf_signal_time(tuneSamples, excAmp=1e-6,
                       nPerSample=1000, f_rev=1.,
                       sampling_freq=1.,saveFile=None):
    
    chirp_time = nPerSample*len(tuneSamples)/f_rev
    time = np.arange(0, chirp_time, 1/sampling_freq)
    samples = chirp(time, tuneSamples[0]*f_rev, chirp_time, tuneSamples[-1]*f_rev)

    if saveFile is not None:
        print(f"""
        \nSaving BTF excitation signal to {saveFile}
        """)
        np.save(saveFile, samples)

    return samples
