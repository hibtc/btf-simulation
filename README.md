# btf-simulation

Here the skripts to run the BTF simulations are available

# Requirements

Please install following packages and their dependencies

- xsuite 0.5.0
- xtrack 0.39.0
- xdeps 0.3.0
- xfields 0.12.2
- xobjects 0.2.7
- xpart 0.15.1

All of them should be available via pip

# Script to create and launch the scans is provided

- run_param_scan_GPU.py

Edit this file to scan over different parameters

How to use it is documented in the file

The template 'submitBTFsim.sh' is an exemplary script
to submit the job via slurm, please make the proper
modifications to make it run in your cluster

# Before launching the simulation

Please modify the script

- Templates/run_btf.py

The path to your working directory has to be inserted in order to find the /src directory
