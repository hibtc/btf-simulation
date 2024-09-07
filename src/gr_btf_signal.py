import numpy as np

def gr_btf_signal(tuneSamples, excAmp=1e-6, nPerSample=1000, saveFile=None):

    turns = np.linspace(0, nPerSample, nPerSample)

    signal = []

    print(f"""
    \nGenerating BTF excitation signal\n
       Tune samples     : {int(len(tuneSamples))}
       Amplitude        : {excAmp} 
       Turns per sample : {int(nPerSample)}
    """)

    for qi in tuneSamples:

        si = excAmp*np.sin(2*np.pi*qi*turns)
        signal = np.hstack((signal, si))
        turns += nPerSample

    if saveFile is not None:
        print(f"""
        \nSaving BTF excitation signal to {saveFile}
        """)
        np.save(saveFile, signal)

    return signal
