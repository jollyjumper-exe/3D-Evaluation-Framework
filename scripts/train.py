import subprocess

def process_mass(scenes, indices):
    commands = ['conda activate nerfstudio']
    
    for scene in scenes:
        for index in indices:
            commands.append(f'ns-process-data polycam --data input/{scene}/working/{index}/polycam.zip --output-dir data/{scene}/{scene + str(index)}')

    execute(commands)

def train_mass(scenes, indices, model):
    commands = ['conda activate nerfstudio']
    
    for scene in scenes:
        for index in indices:
            commands.append(f'ns-train {model} --data data/{scene}/{scene + str(index)}')

    execute(commands)

def execute(commands):
    for cmd in commands:
            try:
                # Execute the command
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                # Handle any errors that occur during command execution
                print(f"Error executing command: {cmd}")
                print(e)
    
if __name__ == "__main__":
    process_mass(['Bunny'], [0])
    train_mass(['Bunny'], [0], 'splatfacto')
    