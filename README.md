# Body Filter

## Abstract

In this machine learning computer vision project, we present an innovative approach for real-time human body detection and interactive body colorization using a combination of **Gretchen, Python3, OpenCV, and FastSAM**. The main objective of this research is to develop a robust and efficient system capable of accurately identifying human bodies in images and live video streams, and subsequently applying dynamic color filters to enhance the visual experience.


## Introduction to the project and its goal

This project’s mission is to detect human bodies in real time and apply filters of different kinds using **FastSAM(Fast Segment Anything Model)**.

## Description of technologies used (within reason)

* To achieve this, we leverage the powerful capabilities of the **You Only Look Once [(YOLO v8)](https://github.com/ultralytics/ultralytics)** deep learning object detection model, which enables us to detect human bodies with high precision and speed. YOLO v8's ability to process images in real-time proves to be instrumental in making our system responsive and suitable for various applications.    
* The **Fast Segment Anything Model(FastSAM)** is a CNN Segment Anything Model trained using only 2% of the SA-1B dataset published by SAM authors. FastSAM achieves comparable performance with the SAM method at 50× higher run-time speed. This FastSAM model uses YOLO v8’s object segmentation model for implementation.    
* The project employs **Python3** as the primary programming language for its flexibility and extensive libraries support. Python3 enables us to integrate various modules seamlessly, making use of the popular OpenCV library for image and video processing tasks. OpenCV plays a crucial role in pre-processing the input data and post-processing the detected human bodies.    
* **Gretchen**, a custom-built framework, is developed to streamline the application of dynamic color filters to the identified human bodies. This framework allows users to interactively select and apply different color filters, offering a captivating and immersive visual experience. The interactive nature of Gretchen enhances the system's appeal for creative and entertainment-oriented applications.   



## Description of your implementation

Our implementation of the code relies on the base [FastSAM library](https://github.com/CASIA-IVA-Lab/FastSAM) and 
[live FastSAM implementation](https://github.com/niconielsen32/FastSAM-Live) 
based on a Github repository by niconielsen32. The weights used for object segmentation is the provided FastSAM-s.pt by the FastSAM library 
which is based on yolov8-seg weights. We have allowed it to accept arguments whether to allow bounding boxes on all segmentation targets the 
model can detect as well as to allow contour outlines on the designated target(“person”). The current default confidence of our model is 0.7 
with a segmentation image size of 256.    
As for the actual “filter”, we have modified the code to only segment based on the text prompt we provide 
which the default is “person”. The text prompt can be modified to segment other things based on the text. The default filter on the segmented mask 
detected by the model has been modified to a default black and an optional Gaussian blur both of which can be altered by arguments.

## Results   

The results of our implementation is a partial success. The filter works as expected which is to apply a colored mask and Gaussian blur on whatever the model detects with the given “text prompt”. However, the Gaussian filter of the current implementation can only be on the colored mask itself not the pixel values of the area of the mask.    
Also, the FastSAM library albeit being much faster than its predecessors still requires a proper gpu for smooth implementation, adaptations were required to reduce the image size and smaller training weight model which reduces the accuracy of our program. With `imgsz=256` and `weight=FastSAM-s.pt`, the time to process each frame has been reduced from about 5 seconds to 80.7ms. Still, the actual user experience is about 1 frame per 5~10 seconds which requires additional hardware upgrades or other adaptations to reduce the lag of the program.

## Discussion (what was particularly challenging, surprising, interesting?)

The particularly challenging part was to understand the FastSAM library and try to make it work on Gretchen. Especially, the default Camera library provided creates images of the same shape and dtype as what OpenCV’s cv2.videoCapture function would do, but the images were not processable by YOLO v8 despite being in the list of acceptable image formats. We eventually set on OpenCV which unfortunately led us to not use ROS environment for compatibility. Also, the hardware limitations were severe as the code of our implementation would not run on the computers of teammates with better capabilities and gpus.     
The surprising part was that although it is quite slow it still works generally as expected and with the default text prompt “person”, rather than the entire outline of the person itself it would focus on the person’s hair, face, or shirt depending on the distance to the camera. This seems to be a residue of the model being trained to “segment everything”.


## Conclusion

In conclusion, our team has worked hard to learn Machine Learning concepts and apply them to our project with partial success. It was a challenging experience with many technical difficulties and setbacks but it also gave us a chance to learn the integration of Gretchen, Python3, OpenCV, and FastSAM(YOLO v8) into a body detection and filtering application. Future possible improvements would be to apply filters directly in the segmented areas and not on the masks as well as possibly using services such as AWS to improve accuracy and speed to process individual frames.    
Overall, We believe that this project was a meaningful experience in learning the basics of Artificial Intelligence as well as robotics.
