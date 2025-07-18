import cv2
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
import threading

# Initialize tkinter window
root = tk.Tk()
root.title('Image Processing Demo')
root.state('zoomed')  # Maximize window on startup

# Global variables to hold image data
original_image = None
processed_image = None

# Global variables
cap = None
video_playing = False
label_processed = None

# Object detection parameters
prototxt = 'Model/MobileNetSSD_deploy.prototxt'
model = 'Model/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def perform_object_detection(image, prototxt_path, model_path, min_confidence=0.2):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img.shape[0], img.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            label = f"{classes[class_id]}: {confidence:.2f}%"
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), colors[class_id], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(img, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# Function to start video object detection
def start_video_object_detection():
    global cap, video_playing, label_processed, label_original
    
    # Clear any existing widgets or images on the screen
    clear_screen()
    
    # Reset global variables
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    video_playing = True
    label_processed = Label(root)
    label_processed.pack(padx=10, pady=10)
    
    # Start a separate thread for video processing
    thread = threading.Thread(target=process_video)
    thread.start()

# Function to clear the screen of any existing widgets
def clear_screen():
    global label_original, label_processed
    
    # Remove label_original widget if it exists
    if label_original:
        label_original.pack_forget()
        label_original.destroy()
        label_original = None
    
    # Remove label_processed widget if it exists
    if label_processed:
        label_processed.pack_forget()
        label_processed.destroy()
        label_processed = None

# Function to process video frames for object detection
def process_video():
    global cap, video_playing, label_processed

    while video_playing:
        ret, img = cap.read()
        if ret:
            height, width = img.shape[0], img.shape[1]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007, (300, 300), 130)

            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > min_confidence:
                    class_id = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    label = f"{classes[class_id]}: {confidence:.2f}%"
                    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), colors[class_id], 2)
                    y = start_y - 15 if start_y - 15 > 15 else start_y + 15
                    cv2.putText(img, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update label with the processed frame
            label_processed.img = img_tk  # Keep a reference to avoid garbage collection
            label_processed.config(image=img_tk)
            label_processed.pack()

    cap.release()

# Function to stop video object detection
def stop_video_object_detection():
    global video_playing
    video_playing = False
    cap.release()  # Release the video capture
    clear_screen()

# Function to perform histogram equalization on the currently loaded image
def perform_histogram_equalization():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        equalized_image = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
        equalized_image_pil = Image.fromarray(equalized_image)
        
        display_image(equalized_image_pil, position=2)


# Function to perform image segmentation using thresholding
def perform_segmentation():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, segmented_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        segmented_image_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB))
        
        display_image(segmented_image_pil, position=2)

# Function to perform Gaussian blur on the currently loaded image
def perform_gaussian_blur():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        blurred_image = cv2.GaussianBlur(img_cv, (5, 5), 0)
        blurred_image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
        
        display_image(blurred_image_pil, position=2)

# Function to perform Median blur on the currently loaded image
def perform_median_blur():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        median_blurred_image = cv2.medianBlur(img_cv, 5)
        median_blurred_image_pil = Image.fromarray(cv2.cvtColor(median_blurred_image, cv2.COLOR_BGR2RGB))
        
        display_image(median_blurred_image_pil, position=2)

# Function to perform Sobel filter on the currently loaded image
def perform_sobel_filter():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(img_cv_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_cv_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_image_pil = Image.fromarray(np.uint8(sobel_combined))
        
        display_image(sobel_image_pil, position=2)

# Function to perform Arithmetic Mean filter on the currently loaded image
def perform_arithmetic_mean_filter():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        mean_filtered_image = cv2.blur(img_cv, (5, 5))  # Example kernel size (5x5)
        mean_filtered_image_pil = Image.fromarray(cv2.cvtColor(mean_filtered_image, cv2.COLOR_BGR2RGB))
        
        display_image(mean_filtered_image_pil, position=2)

# Function to perform Laplacian filter on the currently loaded image
def perform_laplacian_filter():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(img_cv_gray, cv2.CV_64F)
        laplacian_image_pil = Image.fromarray(np.uint8(np.abs(laplacian)))
        
        display_image(laplacian_image_pil, position=2)

# Function to perform Erosion on the currently loaded image
def perform_erosion():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        eroded_image = cv2.erode(img_cv_gray, kernel, iterations=1)
        eroded_image_pil = Image.fromarray(eroded_image)
        
        display_image(eroded_image_pil, position=2)

# Function to perform Dilation on the currently loaded image
def perform_dilation():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(img_cv_gray, kernel, iterations=1)
        dilated_image_pil = Image.fromarray(dilated_image)
        
        display_image(dilated_image_pil, position=2)

# Function to perform Fourier Transform on the currently loaded image
def perform_fourier_transform():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(img_cv_gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
        
        plt.subplot(121), plt.imshow(img_cv_gray, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

# Function to perform Opening on the currently loaded image
def perform_opening():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        opened_image = cv2.morphologyEx(img_cv_gray, cv2.MORPH_OPEN, kernel)
        opened_image_pil = Image.fromarray(opened_image)
        
        display_image(opened_image_pil, position=2)

# Function to perform Closing on the currently loaded image
def perform_closing():
    global original_image
    
    if original_image:
        img_cv_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        closed_image = cv2.morphologyEx(img_cv_gray, cv2.MORPH_CLOSE, kernel)
        closed_image_pil = Image.fromarray(closed_image)
        
        display_image(closed_image_pil, position=2)

# Function to open an image file
def open_image():
    global original_image
    
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = Image.open(file_path)
        display_image(original_image, position=1)
        enable_menu_options()

# Function to perform Canny Edge Detection on the currently loaded image
def perform_canny_edge_detection():
    global original_image
    
    if original_image:
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)  # Adjust parameters as needed
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_pil = Image.fromarray(edges_rgb)
        
        display_image(edges_pil, position=2)


# Function to enable menu options after loading an image
def enable_menu_options():
    edit_menu.entryconfig("Object Detection", state="normal")
    edit_menu.entryconfig("Canny Edge Detection", state="normal")
    filter_menu.entryconfig("Gaussian Blur", state="normal")
    filter_menu.entryconfig("Median Blur", state="normal")
    filter_menu.entryconfig("Mean Filter", state="normal")
    filter_menu.entryconfig("Laplacian Filter", state="normal")
    filter_menu.entryconfig("Sobel Filter", state="normal")
    enhancement_menu.entryconfig("Segmentation", state="normal")
    enhancement_menu.entryconfig("Histogram Equalization", state="normal")
    transformation_menu.entryconfig("Erosion", state="normal")
    transformation_menu.entryconfig("Dilation", state="normal")
    transformation_menu.entryconfig("Opening", state="normal")
    transformation_menu.entryconfig("Closing", state="normal")
    transformation_menu.entryconfig("Fourier Transform", state="normal")

# Function to perform object detection on currently loaded image
def perform_object_detection_on_current():
    global original_image, processed_image
    
    if original_image:
        # Perform object detection
        prototxt_path = 'Model/MobileNetSSD_deploy.prototxt'
        model_path = 'Model/MobileNetSSD_deploy.caffemodel'
        processed_image = perform_object_detection(original_image, prototxt_path, model_path)
        display_image(processed_image, position=2)

# Function to display image in tkinter window
def display_image(image, position):
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    image_width = window_width // 2
    image_height = window_height
    
    resized_image = image.resize((image_width, image_height))
    photo = ImageTk.PhotoImage(resized_image)
    
    if position == 1:
        label_original.config(image=photo)
        label_original.image = photo
        label_original.pack(side=LEFT, padx=10, pady=10)
    elif position == 2:
        label_processed.config(image=photo)
        label_processed.image = photo
        label_processed.pack(side=RIGHT, padx=10, pady=10)

# GUI setup
menubar = Menu(root)
root.config(menu=menubar)

# File menu
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label='Open...', command=open_image)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=root.destroy)
menubar.add_cascade(label='File', menu=file_menu)

# Enhancement menu (Gaussian Blur, Median Blur, Sobel Filter, Segmentation, Histogram Equalization)
enhancement_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Enhancement', menu=enhancement_menu)

# Filter submenu under Enhancement
filter_menu = Menu(enhancement_menu, tearoff=0)
enhancement_menu.add_cascade(label='Filter', menu=filter_menu)
filter_menu.add_command(label='Gaussian Blur', command=perform_gaussian_blur, state="disabled")
filter_menu.add_command(label='Median Blur', command=perform_median_blur, state="disabled")
filter_menu.add_command(label='Mean Filter', command=perform_arithmetic_mean_filter, state="disabled")
filter_menu.add_command(label='Laplacian Filter', command=perform_laplacian_filter, state="disabled")
filter_menu.add_command(label='Sobel Filter', command=perform_sobel_filter, state="disabled")

# Transformation submenu under Enhancement
transformation_menu = Menu(enhancement_menu, tearoff=0)
enhancement_menu.add_cascade(label='Transformation', menu=transformation_menu)
transformation_menu.add_command(label='Erosion', command=perform_erosion, state="disabled")
transformation_menu.add_command(label='Dilation', command=perform_dilation, state="disabled")
transformation_menu.add_command(label='Opening', command=perform_opening, state="disabled")
transformation_menu.add_command(label='Closing', command=perform_closing, state="disabled")
transformation_menu.add_command(label='Fourier Transform', command=perform_fourier_transform, state="disabled")


# Histogram Equalization under Enhancement
enhancement_menu.add_command(label='Histogram Equalization', command=perform_histogram_equalization, state="disabled")

# Segmentation under Enhancement
enhancement_menu.add_command(label='Segmentation', command=perform_segmentation, state="disabled")

# Detection menu (Object Detection)
edit_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Detection', menu=edit_menu)
edit_menu.add_command(label='Object Detection', command=perform_object_detection_on_current, state="disabled")
edit_menu.add_command(label='Canny Edge Detection', command=perform_canny_edge_detection, state="disabled")
edit_menu.add_command(label='Start Object Detection', command=start_video_object_detection)
edit_menu.add_command(label='Stop Object Detection', command=stop_video_object_detection)

net = cv2.dnn.readNetFromCaffe(prototxt, model)
# Create Labels to display images
label_original = Label(root)
label_processed = Label(root)

root.mainloop()
