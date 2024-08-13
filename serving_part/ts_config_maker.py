import yaml

def read_yaml_file(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

def convert_to_properties_format(yaml_content):
    properties_lines = []
    for key, value in yaml_content.items():
        property_line = f'      "{key}":{value}'
        properties_lines.append(property_line)
    return properties_lines

def append_to_properties_file(properties_file_path, properties_lines):
    starting_line = "\n\n# from hparams.yaml\n"
    model_starting_line = \
r"""
models={\
  "spr_seg": {\
    "1.0": {\
"""
    model_ending_line = \
r"""
    }\
  }\
}
"""
    with open(properties_file_path, 'a') as file:
        file.write(starting_line)
        file.write(model_starting_line)
        
        last_idx = len(properties_lines) - 1
        for idx, line in enumerate(properties_lines):
            file.write(line + ",\\\n" if idx != last_idx else line + "\\")
        
        file.write(model_ending_line)

if __name__ == "__main__":    
    # Define paths to your YAML file and properties file
    yaml_file_path = './hparams.yaml'
    properties_file_path = 'config.properties'

    # Read YAML file
    yaml_content = read_yaml_file(yaml_file_path)

    # Convert YAML content to properties format
    properties_lines = convert_to_properties_format(yaml_content)

    # Append to existing properties file
    append_to_properties_file(properties_file_path, properties_lines)

    print(f"YAML content successfully added to {properties_file_path}")