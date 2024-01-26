# Facial Expression recognition, and computing Valence and Arousal

AFFECT is a psychological term used to describe the outward expression of emotion and feelings. Affective computing seeks to develop systems and devices that can recognize, interpret, and simulate human affects through various channels such as face, voice, and biological signals. Face and facial expressions are undoubtedly one of the most important nonverbal channels used by the human being to convey internal emotion. In this project, we have compared state of the art CNN architectures i.e ResNet and Xception on the given dataset and mapped the arousal and valence values and the corresponding expression labels to the image.

## Dataset Description
The dataset was provided by the instructor. Another way to access the dataset is through the following google drive link
```python
#To download dataset
!pip install --upgrade --no-cache-dir gdown

#Training and Val set
!gdown https://drive.google.com/uc?id=1zay8ZTsBKj3ftVBphGWp-4WsAGIxJzAS

#Test Set
!gdown https://drive.google.com/uc?id=1Nc5G2azx3iTzG1Wm_9PqRGP5Wst0OmT8
```

In the given dataset of face Images, following is provided
1. Location of the faces in the images
2. Location of the 68 facial landmarks
3. Eight emotion and non-emotion categorical labels (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt)
4. Valence and arousal values of the facial expressions in continuous domain, and are provided as floating point numbers in the interval [-1,+1].

The images are cropped and resized to 224 x 224 pixels (RGB color). The 8
expression labels and the Arosal/Valence values as well as the facial landmark
points of the training and test set are in the database. You can split the training
dataset in to train and validation.

## Transfer Learning
In transfer learning, the pre-trained model has already learned important
features and patterns from a large dataset in a previous task, and these learned
features can be used as a starting point for a new related task. Instead of
starting the training process from random weights, transfer learning allows the
model to start with some prior knowledge, which can reduce the amount of
data needed for training and improve the overall performance of the model.
In this project, we have used the timms library to use pre-trained models on
Image Net and then we are fine tuning them on our dataset.
To install timms library run the following command

```bash
pip install timm
```

## CNN Architectures
We have used two CNN architectures here: Resnet and Xception. The details are given in their corresponding .ipynb files

## Resources
Training the model takes a long time usually hours in cpu. Google Collab has an inbuilt gpu that lessens the amount of time taken. To access gpu, you can follow these steps:
1. Click on "Runtime" in the menu bar and select "Change runtime type".
2. In the "Runtime type" dropdown menu, select "GPU" as the hardware accelerator.
3. Click "Save" to apply the changes.

Once you have selected the GPU as your hardware accelerator, you can run your code on the GPU just like you would on a CPU. However, keep in mind that not all operations are optimized for GPU usage, so you may need to modify your code to take advantage of the GPU's parallel processing capabilities. To check that the GPU is available and working properly, you can run the following code snippet in a code cell:

```python
import tensorflow as tf
tf.test.gpu_device_name()
```
If the GPU is available, this will return the name of the GPU device. If it is not available, this will return an empty string.

Note that the GPU available on Colab may vary based on availability and usage. If a GPU is not available when you first start your Colab session, you can try restarting the session or checking back later.
