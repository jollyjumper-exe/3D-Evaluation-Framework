import sys
import os
import argparse
import random

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(module_path)
from split_data import split_data
import train as train
import video_to_img as vti
import cv_calcs 
import align_and_crop

parser = argparse.ArgumentParser(description='A script with command-line arguments.')
parser.add_argument('-scene', type=str, help='Specify the scale value.')
parser.add_argument('-size', type=int, help='Specify the size of the dataset.')
parser.add_argument('-model', type=str, help='Specify the models that should be trained.')
parser.add_argument('-testsize', type=int, default=10, help='Specify the models that should be trained.')
parser.add_argument('-indices', nargs="+", default=None, help='List of indices.')

args = parser.parse_args()
scene = args.scene
size = args.size
model = args.model
testsize = args.testsize
if args.indices == None: indices = [int(random.uniform(0, size-1)) for _ in range(testsize)]
else: indices = [int(i) for i in args.indices]

if not os.path.exists(f'metric.csv'):
    with open(f'metric.csv', 'w') as file:
        file.write('scene,model,index,psnr,ssim\n')

#split all data
#for index in indices:
#    split_data(scene, index)

#process data
#train.process_mass([scene], indices)

#train models
#train.train_mass([scene], indices, model)
    
#render results
#train.render_mass([scene], indices, model)
#vti.extract_frame_mass(scene)

#for index in indices:
#    align_and_crop.align(scene, index)

cv_calcs.calc_and_output_metrics(f'images/{scene}', scene, model)