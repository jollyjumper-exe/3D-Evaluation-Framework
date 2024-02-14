import cv2
import sys
import os

def extract_frame(video_path, output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Check if the frame is read successfully
    if not ret:
        print("Error: Unable to read frame")
        return
    
    # Save the frame as an image
    cv2.imwrite(output_image_path, frame)
    
    # Release the video capture object
    cap.release()
    
    print("Frame extracted and saved as", output_image_path)

def extract_frame_mass(scene):
    folder_path = f'images/{scene}/generated'
    files = os.listdir(folder_path)

    # Iterate over the files and call the process_file method for each one
    for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".mp4") and os.path.isfile(file_path):
                extract_frame(file_path, os.path.join(folder_path, (file_name.split(".")[0] + '.jpg')))
                os.remove(file_path)