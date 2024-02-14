# Comparing NeRF & 3DGS
![BHT](https://www.bht-berlin.de/configuration/Resources/Public/assets/images/BHT_Logo_print.png)

## Overview
This project aims to develop evaluation matrices for comparing images generated by different 3D reconstruction methods. The evaluation matrices will provide quantitative measures to assess the accuracy, completeness, and fidelity of the generated image representations.

## Getting Started
1. Clone the repository: 

```git clone https://gitlab.bht-berlin.de/s87298/cv-wise2023-project//```

2. Install nerfstudio using this [guide](https://docs.nerf.studio/quickstart/installation.html).

4. Using the anaconda command prompt, activate your nerfstudio environment.

```conda activate nerfstudio```

3. Install 3DGS within your nerfstudio environment.

```pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.3```

The actual version of gsplat might change.

4. Place your data inside input. The directory structure should look like this: 


``` 
├── input
│   ├── scene_name
│   │   ├── raw
|   |   |   ├── info.json
|   |   |   ├── export_polycam.zip
```

- scene_name: Choose a fitting name for your scene. Like 'Bunny', if you see a bunny on your images.
- info.json: A file containing a scene name and your camera matrices.
- export_polycam.zip: The exported Zip file from polycam.


4. Run the app from root:

```python scripts/app.py -scene <scene_name> -size <dataset_size> -model <model_name> -indices <indices> -testsize <testsize>```

- scene_name: Name of the scene inside your input folder.
- dataset_size: Amount of images inside your scene.
- model_name: Name of the model, you want nerfstudio to use. E.g. nerfacto, splatfacto.
- indices: A list of fix indices of images you want to investigate. This is neccessary if you want to compare your results to different models.
- testsize: An optional parameter, if you provide no list of indices that is used to extract a random test set.

## Methodology
Our method aims to get comparative results between different models. For those purposes, our process is structured like this:

![](https://i.imgur.com/bpxEfMF.png)

1. Data Pool Creation: We start by compiling a comprehensive data pool for our scene. This includes an export from Polycam, which contains detailed scene information, and an accompanying info.json file. The info.json file provides essential camera matrices for each frame within the export, along with corresponding frame titles.

2. Data Extraction: Given a specific index, we retrieve the corresponding frame and camera matrix from our data pool.

3. Dataset Preparation: We create distinct datasets tailored to the provided indices. These datasets are meticulously curated, ensuring that all data pertaining to the extracted frame is accurately isolated.

4. Model Training: Each dataset undergoes rigorous model training, harnessing the power of machine learning algorithms to capture intricate patterns within the scene.

5. Rendering: Employing the extracted camera matrix, we render models with precision, striving to faithfully represent the intricacies of the scene.

6. Alignment and Warping: Utilizing SIFT-Features, we meticulously align the extracted frame with the rendered model. By identifying matched features, we calculate the necessary transformation to warp the perspective, ensuring seamless integration.

7. Evaluation: Finally, we evaluate the alignment's quality using sophisticated metrics such as the Structural Similarity Index Measure (SSIM) and Peak Signal-to-Noise Ratio (PSNR), providing valuable insights into the fidelity of our rendered scene.

![](https://i.imgur.com/SgIpQsD.png)

## Results

![](https://i.imgur.com/VhbN0VK.png)
3DGS with PSNR: 28.71 (left), NeRF with PSNR: 28.68 (right) 

## Related Work