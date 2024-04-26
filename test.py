import os
import cv2
import numpy as np
from collections import deque

def save_last_20_images(image_queue, folder_path):
    last_20_images = deque(maxlen=20)
    print("image_queue")

    while True: 
        print(image_queue)
        img = image_queue.get()
        image_files = [f for f in os.listdir('./images') if f.endswith('.jpg')]

        # Save the image with sequential file names
        file_name = f"{folder_path}/image_{len(image_files) + 1}.jpg"
        cv2.imwrite(file_name, img)

        
        # Sort the image files based on their names
        image_files.sort()

        print(len(image_files)>20)

        # Remove excess images if more than 20 images exist
        image_files = [f for f in os.listdir('./images') if f.endswith('.jpg')]

        while(len(image_files)) > 20:
            excess_files = image_files[:len(image_files) - 20]
            for file in excess_files:
                os.remove(os.path.join(folder_path, file))


from queue import Queue
import time
import threading

# Initialize your image queue or source
image_queue = Queue()

# Start the function in a separate thread
folder_path = "images"
thread = threading.Thread(target=save_last_20_images, args=(image_queue, folder_path))
thread.start()

# Continuously provide images to the queue (replace this with your actual image source)
for i in range(100):
    # Generate a dummy image (replace this with your image source)
    dummy_image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    cv2.rectangle(dummy_image, (0,0), (340, 40), (234, 234, 77), 1)
    cv2.putText(dummy_image,f"label{i}", (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    image_queue.put(dummy_image)

    time.sleep(1)  # Simulate some processing