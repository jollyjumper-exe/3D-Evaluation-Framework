import subprocess

def train(scene, model, source):
    cmd = f'ns-train {model} --data {source}'

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Output of the command:")
    print(result.stdout)

if __name__ == "__main__":
    train("", "splatfacto", "input/bunny/working/1/train")