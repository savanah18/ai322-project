# Prerequisites
- Working Linux Environment
- Python 3.10


# Setup
## Setup Environment
1. Setup virtualenv
```
python3 -m pip install virtualenv
source venv/bin/activate
```
2. Install python dependencies
```
pip install -r requirements.txt
```

## Clone Simulators
> Clone on another directory
```
git clone https://github.com/wineslab/ns-o-ran-ns3-mmwave.git
git clone https://github.com/wineslab/o-ran-e2sim
```

# Create Configuration Files
For both standalone simulation and RL training. You should generate simulation configuration.
## Generating Network Sim Configurations
> Note that you
1. Go to network sim profile. Create sim profiles (based on existing sim templates). Save new file 
```
cd ./dev/sim_config/network_sim_profiles
```
2. Generating simulation configuration files
> specify created file from step 1 in --profile. Specify ouput directory for scenario configurations as well
```
cd ./dev/sim_config
python3 config_generator.py --config ./config_schema.yaml --profile ./network_sim_profiles/eval_106.yaml --output ./scenario_configuration_eval_106
```
> Note: There should be multiple configuration files generated in specified scenario configuration output dir


# Running ES RL Training
1. Verify existing source code for ES Training
```
 ls -l | grep es_
```
2. Configure. For every run,  change base directory to current git repository. Change config paths to the files generated in `Generating Network Sim Configuration`
3. Execute RL training
```
script=es_ppo_22.py
python3 $script
```

# Running Parallel Standalone Simulation (Optional)
1. Execute batch_simulator.sh
> Please see usage by running `./batch_simulator.sh`