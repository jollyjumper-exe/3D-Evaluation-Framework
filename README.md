# Comparing NeRF & 3DGS
![BHT](https://www.bht-berlin.de/configuration/Resources/Public/assets/images/BHT_Logo_print.png)

## Overview
This project aims to develop an evaluation framework for comparing images generated by different 3D reconstruction methods. Our novel approach seeks to provide a more realistic portrayal of performance. Typically, around ten percent of our data needs to be cut to create an evaluation set used for model evaluation. However, working with relatively small datasets could have a devastating effect on performance. We aim to address this challenge and compare our solution to conventional methods.

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
We tested our methodology with three scenes in total. We used a scan of a 3D printed stanford bunny, a common packaging box and a breakfast bowl.
Each model was tested with 20 different frames.

![](https://i.imgur.com/HkaSz9Q.png)

3DGS with PSNR: 28.71 & SSIM: 0.62 (left), NeRF with PSNR: 28.68 & SSIM: 0.60(right)

![](https://i.imgur.com/W3esACk.png)

3DGS with PSNR: 30.92 & SSIM: 0.78 (left), NeRF with PSNR: 29.61 & SSIM: 0.79(right)

![](https://i.imgur.com/19PnGf2.png)

3DGS with PSNR: 29.72 & SSIM: 0.56 (left), NeRF with PSNR: 29.92 & SSIM: 0.78(right) 

### NeRF

| Scene | Input    | PSNR     | PSNR-NS  | SSMI     | SSMI-NS  |
|-------|----------|----------|----------|----------|----------|
| Bunny | 143      | 26.048   | 20.104   | 0.708    | 0.714    |
| Box   | 92       | 26.411   | 20.086   | 0.745    | 0.649    |
| Bowl  | 96       | 25.928   | 17.491   | 0.754    | 0.597    |

### 3DGS

| Scene | Input    | PSNR     | PSNR-NS  | SSMI     | SSMI-NS  |
|-------|----------|----------|----------|----------|----------|
| Bunny |143       | 28.429   | 26.279   | 0.690    | 0.607    |
| Box   |92        | 27.721   | 25.683   | 0.737    | 0.735    |
| Bowl  |96        | 28.504   | 26.706   | 0.667    | 0.595    |

For comparison purposes, we put the results of the nerfstudio evaluation next to our results.

### Notes on alignment
Our automatic alignment method was tested by aligning an image to itself, expecting high PSNR and SSIM scores close to 1. However, the obtained results were as follows:

![](https://i.imgur.com/NLRYxLE.png)

PSNR: 32.716, SSIM: 0.96

Although these scores are relatively high, they are not perfect. The difference image reveals a vulnerability to noise, indicating room for improvement in our evaluation method.


### Sanity Check

## Conclusion
Our evaluation method offers a clearer portrayal of a model's recreational ability compared to the usual evaluation method, as it leverages almost the entire dataset for training. We highlighted the significant impact on performance observed with the usual evaluation method. However, our method requires significantly more processing time. Nevertheless, it yields similar relative results when comparing multiple models to each other. Thus, while our method demonstrates better results for individual models, it aligns with conclusions from other studies, making it valuable for assessing a model's real performance but less essential for comparison purposes.

## Related Work

### [Gaussian Splatting with NeRF-based Color and Opacity](https://arxiv.org/html/2312.13729v2)
This study compares 3DGS and NeRF with the usual method. The general comparisons in this study show similar results to ours.

### [Structural Accuracy vs. Efficiency: A Comparative Study of LiDAR, NeRF, and 3D Gaussian Splatting for Model Retrieva](https://gitlab.bht-berlin.de/s87298/masterprojekt) 
Another project created by us, featuring similar research on structural recreation.

### [nerfstudio](https://docs.nerf.studio)
For any questions on nerfstudio, refer to the official nerfstudio documentation.