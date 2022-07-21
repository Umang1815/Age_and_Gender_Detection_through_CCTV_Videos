# Age_and_Gender_Detection_through_CCTV_Videos
The task of this project is to estimate people's age and gender form a surveillance video.

# Objectives
The task appears simple but there are number of issues that must be taken into consideration inorder to develop an efficient system. Through this system we aim to achieve the following objectives:

- The system must use maximum information. It must use both face and body as well as information from all the frames inorder to estimate the age and gender.
- The system must have a good FPS rate so that it could be used in real time.
- The results must be usable and interpretable.

# Methodology
Following approach was used to build the desired system.

- **Object Detection**: The first task was to detect humans in the indiviual frame. YOLO was used for this purpose. It is the SOTA algorithm used for multi-object detection task which gives high accuracy even when applied in real-time on videos
- **Tracker**: After detecting the person in a frame it is necessary to track that person throughout the frame. Byte Tracker was used for this purpose. It is the SOTA algorithm for multi-object tracking.
- **Upscaling**: Surveillance video usually have a low quality making it difficult to identify the person's face. To overcome this issue upscaling of face and body images extracted from the video was needed. SwinIR was used for this purpose. This task can be skipped to increase the fps of the system.
- **Models**: Details of the models are given below

![Model_Tabel]()

From the table it is visible that model with SWIN Transformer as a backbone performed best.
- **Dataset**: Details of the dataset that were used to train the model are given below

![Dataset_Tabel]()
## Pipeline Solution

### Image Pipeline
 This is the model solution
 
![Image Pipeline](https://github.com/Umang1815/Age_and_Gender_Detection_through_CCTV_Videos/blob/main/image_pipeline.JPG)
