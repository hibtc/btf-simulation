import numpy as np
import shutil

import os
import json

def create_parameters_dict(kPrimeL, excAmp, sg_samp_per_turn, 
                           beamEmittance, dpp, turnsPerExc, 
                           excSamples, nparticles, band, band_sgn, 
                           turns_init, correctChroma):
    parameters = {
        'kPrimeL': kPrimeL,
        'excAmp': excAmp,
        'sg_samp_per_turn': sg_samp_per_turn,
        'beamEmittance': beamEmittance,
        'dpp': dpp,
        'turnsPerExc': turnsPerExc,
        'excSamples': excSamples,
        'nparticles': nparticles,
        'band':band,
        'band_sgn':band_sgn, 
        'turns_init':turns_init,
        'correctChroma':correctChroma,
    }
    return parameters

def save_dict_to_json(parameters, filename):
    with open(filename, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

def load_parameters_from_json(filename, path='./'):
    with open(path+filename, 'r') as json_file:
        parameters = json.load(json_file)
    return parameters

def setSimDirInBash(folder_name):

    dst_path = os.path.join(folder_name, "submitBTFSim.sh")
    fileRead = open(dst_path, "r")
    runSimFile  = fileRead.read().format(dirName=folder_name)

    fileWrite = open(dst_path, "w")
    fileWrite.write(runSimFile)

    fileRead.close()
    fileWrite.close()


def create_folders_for_simulation(parameters, 
                                  base_folder='simulations', 
                                  templates_folder='Templates', 
                                  runSim=False):
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    base_parameters_filename = os.path.join(base_folder, 'base_parameters.json')
    #save_dict_to_json(parameters, base_parameters_filename)

    for key, value_list in parameters.items():
        
        if isinstance(value_list, list):
        
            for value in value_list:
                folder_name = os.path.join(base_folder, f"{key}_{value}")
                
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)

                # Create a folder for the maxwell output
                maxout_name = os.path.join(folder_name, "MAXOUT")
                os.mkdir(maxout_name)

                # Copy all files from the Templates folder to the simulation folder
                for file_name in os.listdir(templates_folder):
                    src_path = os.path.join(templates_folder, file_name)
                    dst_path = os.path.join(folder_name, file_name)
                    if os.path.isfile(src_path):#and 'gpu' in src_path:
                        shutil.copy(src_path, dst_path)

                # Set the working directory in the bash file
                setSimDirInBash(folder_name)

                # Save the scanned parameter value to a JSON file inside the folder
                scanned_parameter_filename = os.path.join(folder_name, 'parameters.json')
                #scanned_parameter = {key: value}
                parameters[key] = value
                save_dict_to_json(parameters, scanned_parameter_filename)

                simFile = os.path.join(folder_name, f"submitBTFSim.sh")
                if runSim:
                    print(' Queuing batch file')
                    print(f'  {simFile}\n')
                    os.system(f'sbatch {simFile}')
                    print()

#######################################################
#
# Example usage:
#     Creates a folder 'output_Test'
#     and within the folders 
#          kPrimeL0.0, kPrimeL0.5, kPrimeL0.9
#     with a file name parameters.json with
#     the base parameters defined here
#     The scanned parameter has to be in a list
#
#####################################################
#
# Values for the scans shown in
#    https://arxiv.org/abs/2404.02576
#
#    - Parameter to control the resonant driving term
#    kPrimeL =  [0.0, 0.6, 0.8, 1.0, 1.2]
#    - Excitation strength in [rad]
#    excAmp  = 58e-9
#    - Longitudinal samples per turn
#     (defines the Nyquist frequency)
#    sg_samp_per_turn = 20
#    - Momentum spread
#    dpp = 0.5e-3
#    - Beam emittance in both transverse planes in [mm mrad]
#    beamEmittance = 1.0
#    - Turns per excitation frequency step
#    turnsPerExc = 30000
#    - Number of frequency steps
#    excSamples = 701
#    - Number of particles
#    nparticles = int(1e5)
#    - Betatron side-band n at which the sweep takes place
#    band = 9
#    - Upper or lower side-band (1 -> upper, -1 -> lower)
#    band_sgn = 1
#    - Number of turns before the excitation is applied
#    turns_init = int(10000)
#    - If the chromaticity should be natural or corrected
#      (True -> chroma = 0, False -> Nat. chroma)
#    correctChroma = True
#
######################################################

kPrimeL = [0.0, 0.5, 0.9]
excAmp = 58e-9
sg_samp_per_turn = 20
beamEmittance = 1.0
#beamEmittance = [0.1, 0.5, 1.0, 1.5, 2.0]
dpp = 0.5e-3
turnsPerExc = 30000
#turnsPerExc = [1000, 5000, 10000, 15000, 20000, 30000]
excSamples = 701
nparticles = int(1e5)
band = 9
#band = [1,3,5,7,8,9]
band_sgn = 1
turns_init=int(10000)
correctChroma = True

parameters = create_parameters_dict(kPrimeL, excAmp, sg_samp_per_turn, 
                                    beamEmittance, dpp, turnsPerExc, 
                                    excSamples, nparticles, band, band_sgn, 
                                    turns_init, correctChroma)

##########################################
#   output folder
##########################################

simFolder = 'output_Test'

#########################################
#  This launches the simulation as
#    -  sbatch submitBTFsim.sh
#########################################

runSim = False

create_folders_for_simulation(parameters, base_folder=simFolder, runSim=runSim)
