# Q1. Summary

1. Application of Convolutional Neural Network to classify 7 different emotions on FER-2013 dataset from Kaggle.

# Q2. Dataset Description

## Dataset Description

I have used [FER-2013](https://www.kaggle.com/msambare/fer2013/download) dataset from Kaggle. The dataset comprises of facial images, with emphasis on the importance of emotions. All the images have single face in the frame and each image is **48x48** pixel grayscale color scheme. The face is more or less centered and occupies about the same amount of space in each image. There are **7** different emotions category shown in the facial expression of the person. The 7 categories are numbered from **0 to 6** both included and can be mapped to the expression as follows: **(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)**

The dataset occupies about 60Mb of diskspace. Its divided into 2 seperate folders, **train and test**. The train folder corresponds to the training set with **28,709** images and test folder corresponds to the test set with **3,589** images. Each of this folders are sub divided into the 7 different sub directories for 7 different emotions. 

The following table shows the number of images in each category for training and testing:

|       | angry | disgust | fear | happy | neutral |  sad | surprise |
| :---- | ----: | ------: | ---: | ----: | ------: | ---: | -------: |
| train |  3995 |     436 | 4097 |  7215 |    4965 | 4830 |     3171 |
| test  |   958 |     111 | 1024 |  1774 |    1233 | 1247 |      831 |

We notice that the images are not equally distributed, this is an imbalanced dataset. This can be futher emphasised used the number of images in train and test bar plot shown below:

![Fig 1.](figs/dataset_counts.png)

We notice that the **disgust** emotion category has least number of images for training and testing and **happy** category has the most number of images. This figure shows that, the number of images in each category are not equal, however, the distribution of number of images in category across the train and test dataset are almost similar.

The following figure gives an idea of how the images look in 7 different categories:

![Fig 2](figs/sample_images.png)

Clearly from this sample set of images displayed, we can tell that:

1. the face of a person is roughly always centered
2. the images are grayscale
3. the facial expression of a person matches the emotion description on that image

# Q3. Details
...