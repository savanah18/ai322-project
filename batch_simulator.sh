#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 <config_dir> <batch_size> <actor> <es_mode>"
    exit 1
}

# Check for required arguments
if [ $# -ne 4 ]; then
    usage
fi

# Directory containing configuration files
#config_dir="/mnt/d/Workspace/Networks/RL-Project/ns-o-ran-gym/dev/sim_config/scenario_configurations_random"
config_dir="$1"

# Number of processes to run in parallel
#batch_size=8
batch_size="$2"

# Actor and ES mode
actor=$3
es_mode=$4

# Get a list of all configuration files
config_files=("$config_dir"/*)

# Total number of configuration files
total_configs=${#config_files[@]}

# Iterate through configuration files in batches
for ((i=0; i<total_configs; i+=batch_size)); do
    # Run up to batch_size processes in parallel
    for ((j=0; j<batch_size; j++)); do
        index=$((i + j))
        if ((index >= total_configs)); then
            break
        fi

        config_file="${config_files[$index]}"

        # Replace "process_command" with the actual command to run
        # echo "Processing $config_file"
        echo $config_file        
        log_file="output/eval/$(basename "$config_file").log"
        # Chek if log_file exists
        if [ -f "$log_file" ]; then
            echo "Log file $log_file already exists. Skipping execution."
        else
            echo Executing command: time python3 energy_saving.py --config $config_file  --ns3_path /home/gagluba/ns-3-mmwave-oran/ --optimized --actor $actor --es_mode $es_mode > $log_file 2>&1 &
            nohup time python3 energy_saving.py --config $config_file  --ns3_path /home/gagluba/ns-3-mmwave-oran/ --optimized --actor $actor --es_mode $es_mode > $log_file 2>&1 &
        fi
    done

    # Wait for all processes in the batch to finish
    wait
done

echo "All configurations processed!"
