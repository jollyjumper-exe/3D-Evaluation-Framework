import csv
import numpy as np
import cv2
import math
import matplotlib as mpl
import glob
from torchvision.models import inception_v3
import torch
from torchvision import transforms
from scipy import linalg

# Loading the pre-trained Inception network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_fid(img1, img2, model, num_images=50):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # Get the inception output for the real images
    with torch.no_grad():
        pred_img1 = model(img1_tensor)
        pred_img2 = model(img2_tensor)

    # Calculate mean and covariance
    mu1, sigma1 = pred_img1.mean(dim=0), torch.tensor(np.cov(pred_img1.cpu().numpy(), rowvar=False))
    mu2, sigma2 = pred_img2.mean(dim=0), torch.tensor(np.cov(pred_img2.cpu().numpy(), rowvar=False))

    # Calculate FID score
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score.item()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def load_images(folder):
    images = []
    for filename in glob.glob(folder + '/*.jpg'):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


def compare_images(modelname, methodname, real_images, rendered_images, inception_model):
    psnr_values = []
    ssim_values = []
    fid_values = []

    for real_image, rendered_image in zip(real_images, rendered_images):
        # Convert images to numpy arrays
        real_np = np.array(real_image)
        rendered_np = np.array(rendered_image)

        # Calculate PSNR
        psnr_value = calculate_psnr(real_np, rendered_np)
        psnr_values.append(psnr_value)

        # Calculate SSIM
        ssim_value = calculate_ssim(real_np, rendered_np)
        ssim_values.append(ssim_value)

        # Calculate FID
        fid_value = calculate_fid(real_np, rendered_np, inception_model)
        fid_values.append(fid_value)

    # Calculate average values
    psnr_avg = np.mean(psnr_values)
    ssim_avg = np.mean(ssim_values)
    fid_avg = np.mean(fid_values)

    # Write results to CSV file
    with open('metric.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([modelname, methodname, psnr_avg, ssim_avg, fid_avg])

def calc_and_output_metrics(path, scene, model):
    real_images = load_images(f'{path}/original/')  # List of real images
    rendered_images = load_images(f'{path}/generated/')  # List of rendered images
    compare_images(scene, model, real_images, rendered_images, inception_model)