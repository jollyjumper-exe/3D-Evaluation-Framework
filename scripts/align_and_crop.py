import cv2
import numpy as np

def align_and_crop_images(image1_path, image2_path):
    image1_color = cv2.imread(image1_path)
    image2_color = cv2.imread(image2_path)

    image1 = cv2.cvtColor(image1_color, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        aligned_image_color = cv2.warpPerspective(image1_color, H, (image2_color.shape[1], image2_color.shape[0]))

        h, w, _ = image2_color.shape
        aligned_and_cropped_image_color = aligned_image_color[0:h, 0:w]

        return aligned_and_cropped_image_color

    else:
        print("Not enough matches found.")

def align(scene, index):
    image2_path = f'images\\{scene}\\generated\\{index}.jpg'
    image1_path = f'images\\{scene}\\original\\{index}.jpg'

    aligned_and_cropped_image_color = align_and_crop_images(image1_path, image2_path)

    cv2.imwrite(f'images\\{scene}\\original\\{index}.jpg', aligned_and_cropped_image_color)