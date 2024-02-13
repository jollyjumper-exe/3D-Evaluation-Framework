import sys
import os
import argparse
import random

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(module_path)
from split_data import split_data
import train as train
import video_to_img as vti

parser = argparse.ArgumentParser(description='A script with command-line arguments.')
parser.add_argument('-scene', type=str, help='Specify the scale value.')
parser.add_argument('-size', type=int, help='Specify the size of the dataset.')
parser.add_argument('-model', type=str, help='Specify the models that should be trained.')
parser.add_argument('-testsize', type=int, default=10, help='Specify the models that should be trained.')

args = parser.parse_args()
scene = args.scene
size = args.size
model = args.model
testsize = args.testsize

random_indices = [int(random.uniform(0, size-1)) for _ in range(testsize)]

#split all data
for index in random_indices:
    split_data(scene, index)

#process data
train.process_mass([scene], random_indices)

#train models
train.train_mass([scene], random_indices, model)
    
#render results
train.render_mass([scene], random_indices, model)
vti.extract_frame_mass(scene)