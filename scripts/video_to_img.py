import cv2
import sys

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

# Specify the path to the input video file
video_path = "input_video.mp4"

# Specify the path where you want to save the output image
output_image_path = "output_image.jpg"

# Call the function to extract the frame
video_path, output_image_path = sys.argv[1], sys.argv[2]
extract_frame(video_path, output_image_path)