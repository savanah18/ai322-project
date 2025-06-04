import yaml
import json
class ConfigGenerator:
    def __init__(self, config_path):
        try:
            with open(config_path, 'r') as file:
                self.config_schema = yaml.safe_load(file)
        except Exception as e:
            raise(e)
    
    def generate_sim_config(self, profile_path,output_path):
        try:
            with open(profile_path, 'r') as file:
                sim_profile = yaml.safe_load(file)
            print(sim_profile)

            yaml_base = {}
            for conf in sim_profile['base']:
                yaml_base[conf] = [sim_profile['base'][conf]]
            for conf in sim_profile['static']:
                yaml_base[conf] = [sim_profile['static'][conf]]

            print("Base", yaml_base)
            
            enumerative_conf = sim_profile['enumerative']
            for _rng in range(enumerative_conf["RngRun"][0], enumerative_conf["RngRun"][1]):
                for _trafficModel in range(enumerative_conf["trafficModel"][0], enumerative_conf["trafficModel"][1]):
                    for _configuration in range(enumerative_conf["configuration"][0], enumerative_conf["configuration"][1]):
                        for _dataRate in range(enumerative_conf["dataRate"][0], enumerative_conf["dataRate"][1]):
                            instance_yaml= yaml_base.copy()
                            instance_yaml["RngRun"] = [_rng]
                            instance_yaml["trafficModel"] = [_trafficModel]
                            instance_yaml["configuration"] = [_configuration]
                            instance_yaml["dataRate"] = [_dataRate]
                            config_save_path = f"{output_path}/{profile_path.split('/')[-1].split('.')[0]}_{_rng}_{_trafficModel}_{_configuration}_{_dataRate}.json"
                            print(f"Saving configuration to {config_save_path}")
                            with open(config_save_path, 'w') as outfile:
                                json.dump(instance_yaml, outfile, indent=4)



        except Exception as e:
            raise(e)



if __name__ == "__main__":
    config_path = "./config_schema.yaml"  # Replace with your config file path
    profile_path = "./network_sim_profiles/eval_106.yaml"  # Replace with your profile file path
    output_path = "./scenario_configurations_eval_106"
    # make sure the output path exists, create it if not
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generator = ConfigGenerator(config_path)
    generator.generate_sim_config(profile_path,output_path)
