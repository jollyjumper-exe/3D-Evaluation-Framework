import subprocess
import os

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
            commands.append(f'ns-train {model} --data data/{scene}/{scene + str(index)} --viewer.quit-on-train-completion True')

    execute(commands)

def render_mass(scenes, indices, model):
    commands = ['conda activate nerfstudio']

    for scene in scenes:
        for index in indices:
            folder_path = f'outputs/{scene + str(index)}/{model}'
            items = os.listdir(folder_path)
            folder = [item for item in items if os.path.isdir(os.path.join(folder_path, item))][0]
            commands.append(f'ns-render camera-path --load-config outputs/{scene + str(index)}/{model}/{folder}/config.yml --camera-path-filename input/{scene}/working/{index}/camerapath.json --output-path images/{scene}/generated/{index}.mp4')
    
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
    