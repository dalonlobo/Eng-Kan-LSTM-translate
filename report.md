# Q1. Summary

1. Describe the project titled: "Human facial expression detector using Deep Learning Network"
2. Application of Convolutional Neural Network to classify 7 different emotions on FER-2013 dataset, performance analysis.
3. Adding batch normalization after each layer.
4. Added dropout so that the model generalizes better.
5. Callback for earlystopping is beneficial, since it allows us to experiment faster.
6. Use data augmentation, to make the model more robust for changes in images
7. Final best model found after hyper parameter tuning, that performed best.
8. Used the trained model for prediction on my facial expressions and converted those expressions to emojis.
9. System challenges faced during the model training and mitigated it with different approaches.

# Q2. Dataset Description

## Dataset Description

I have used [FER-2013](https://www.kaggle.com/msambare/fer2013/download) dataset from Kaggle. The dataset comprises of facial images, with emphasis on the importance of emotions. All the images have single face in the frame and each image is **48x48** pixel grayscale color scheme. The face is more or less centered and occupies about the same amount of space in each image. There are **7** different emotions category shown in the facial expression of the person. The 7 categories are numbered from **0 to 6** both included and can be mapped to the expression as follows: **(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)**

The dataset occupies about 60Mb of diskspace. It's divided into 2 seperate folders, **train and test**. The train folder corresponds to the training set with **28,709** images and test folder corresponds to the test set with **3,589** images. Each of these folders are sub divided into the 7 different sub directories for 7 different emotions. 

The following table shows the number of images in each category for training and testing:

|       | angry | disgust | fear | happy | neutral |  sad | surprise |
| :---- | ----: | ------: | ---: | ----: | ------: | ---: | -------: |
| train |  3995 |     436 | 4097 |  7215 |    4965 | 4830 |     3171 |
| test  |   958 |     111 | 1024 |  1774 |    1233 | 1247 |      831 |

We notice that the images are not equally distributed, this is an imbalanced dataset. This can be further emphasized by the number of images in train and test bar plot shown below:

![Fig 1.](figs/dataset_counts.png)

We notice that the **disgust** emotion category has least number of images for training and testing and **happy** category has the most number of images. This figure shows that, the number of images in each category are not equal, however, the distribution of number of images in category across the train and test dataset are almost similar.

The following figure gives an idea of how the images look in 7 different categories:

![Fig 2](figs/sample_images.png)

Clearly from this sample set of images displayed, we can tell that:

1. the face of a person is roughly always centered
2. the frame should only contain the face of the person
3. the images are grayscale
4. the facial expression of a person matches the emotion description on that image

# Q3. Details

1. Facial expressions(emotions) are an important factor how humans communicate with each other. Humans can interpret facial expressions naturally, however, computers struggle to do the same. This project focuses on classification i.e., detection of human emotions from the images with facial expressions features using deep learning technique called Convolutional Neural Networks.
   
2. In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of artificial neural network, most commonly applied to analyze images. It is a powerful technique, as demonstrated in earlier assignments. CNN preserves the spatial features, which gives it an advantage over other methods like KNN, SVM etc. For this reason I used simple CNN with 2 convolutional layers with 3 by 3 kernel size, followed by max pooling, then 2 dense layers with final dense layer has softmax as activation function so that we can predict the classes. The training/validation accuracy/loss are shown below. I get a peak train accuracy of 0.63 and peak test accuracy of 0.59. Similarly noticed the loss decreasing as we train for more epochs. This shows that the model learns in every epoch i.e., it updates its weights during backpropagation.
   
   ![Fig 3](figs/Model_1.png)

    Here I've noticed the choice of loss function also plays an important role in model training. I've used categorical_crossentropy, which is ideal for multi class classification. Correct loss function has to be chosen based on the dataset and usecase. Used Adam(short for Adaptive Moment Estimation) optimizer which is an update to RMSProp optimizer[4].

3. Added batch normalization after each layer. The accuracy did not change much as shown in figure below. There is a sudden drop in accuracy at epoch 4, which could be caused by random selection of images for validation, and most of the images might have belonged to "disgust" category which has very few images in training set.

    ![Fig 4](figs/Model_2.png)

    Learned during assignment 4, that batch normalization plays an important to reduce internal covariate shift (https://arxiv.org/abs/1502.03167). When the output of the previous layer was given as input to the next layer, it gets normalized before propagated through to the next layer and hence the numbers will not grow uncontrollably in the network, thereby giving improvement in our performance. Another advantage of batch norm is it gives our network a resistance to vanishing gradient during training [5]. I get a peak train accuracy of 0.62 and peak test accuracy of 0.57. Here the accuracy did not improve much, however, batch norm is required for a good model training. Future experiments show increase in performance.

4. Added dropout layers after each layers. Since it is a hyperparameter, the rate of dropout is adjusted by tuning. Dropout is used to prevent large neural network models from overfitting, it has been widely used as an efficient regularization technique in practice[6]. This regularization is important, so that the model generalizes well, i.e., it does not over-fit the training data.
   
   ![Fig 5](figs/Model_3.png)

    Here we notice that the accuracy has a huge drop, this is because I dropped many nodes in the network. In future experiments, I've tuned this parameter to get better results.

5. Used an interesting feature of Keras, a callback EarlyStopping. This is passed to the model.fit, it monitors the loss, and stops the training process if the model loss does not improve. This is specially important during experimentation, so that we don't waste our time waiting for a model to complete all the epochs that we mentioned before stopping. In my case, this stopped the training many times when loss did not improve. I set the patience=3, which waits for 3 epochs for loss to improve.

    I also used few other callback functions like, ModelCheckpoint, ReduceLROnPlateau, CSVLogger. ModelCheckpoint was used to save the model checkpoints after epochs so that, if training crashes we can load from the saved weights. I've used the save model weights from here to do the predictions later. CSVLogger is only used to log the various statistics after each metric in an csv file.

    ReduceLROnPlateau is another helpful function which reduces the learning rate, in our case the learning rate of Adam optimizer when the model has stopped improving. When model trains, with a permanent large learning rate, the model's performance will bounce around the minimum most likely always overshooting it. So reducing it can help the model reach better minimum, however this is also a hyperparameter, since the plateau parameters  had to be tuned. 

6. [Data augmentation](https://en.wikipedia.org/wiki/Data_augmentation#:~:text=Data%20augmentation%20in%20data%20analysis,training%20a%20machine%20learning%20model.) is used to increase the amount of data by adding slightly modified copies of already existing data. We came to know while learning UNET that data augmentation can be used to increase the number of training data, and also these augmentations can make the trained model more robust to changes in real world data. So I used 2 augmentations i.e., zoom and horizontal flip of images using ImageDataGenerator from Keras.

7. After many experimentations and hyper parameter tuning, found a model with 4 convolutional layers perform the best. It has all the features mentioned above and the model layers are shown in the figure below.

    ![Fig 6](figs/lobo_net.png)

    The training/validation accuracy/loss are shown below.

    ![Fig 7](figs/Model_4.png)

    The model train accuracy was 0.73 and test accuracy was 0.63 when ran the model on respective datasets after training phase using model.evaluate. This is not as good as an accuracy I got in assignments, its lower than that, thats because this is a difficult multiclass classification problem with images. Recognizing the facial expressions is hard task. The best model on FER2013 dataset that I used was 0.76 by a model Ensemble ResMaskingNet with 6 other CNNs as stated by paperswithcode.com[87]

    My model performed well after many modification because, I found the best number of convolutional layers, for generalizing I've used dropout layers and batch normalization as well and used hyper parameter tuning to find the best parameters for each. 

8. Model is not useful if it doesn't perform on real world tasks. So I took images of my face using webcam with different expressions and ran model prediction on those images. The model weights where saved in the previous steps, so I did not train my model again, whereas used the same weights and predicted my facial expression. Mapped this expression to an emoji, and displayed the results. The results are shown below.

    ![Fig 8](figs/fer-pred.png)

    Initally the model performed very poorly since the images where not cropped to include only the face, cropping is necessary since the model is trained with images that have only face in them. After cropping images, got these results, among the 6 images shown, there was 1 incorrect prediction. The incorrect prediction was on expression disgust, this category has least number of images in our training, that could be the reason it did not classify correctly. However, knowing how to use the model for a real world application is highly beneficial.


9. Model training was performed on GPU, Nvidia GeForce GTX 1050 Ti with 4 GB of memory. Training was the most intensive task, which when performed, it utilized the full capacity of the system. The main parameter that I used so that the system does not run out of memory was the training batch size. Successfully performed training on various settings with batch size = 32, which was ideal on my system. When we increase the batch size, the number of training images that get loaded into memory for processing increases. Some studies show that large batch sizes don't generalize well [3]. The lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function [3]. When batch size = 32, the total training images = 28709 / 32 = 897 minibatches were created. One epoch is complete when training is completed on all these 897 mini batches.

    A suggestion from Dr. Levmann, that was used in this is, when I was struggling with compute resources to train such a huge model, training was performed on a reduced set of images. This greatly help me to experiment with different architectures.



## References

1. Course project dataset: https://www.kaggle.com/msambare/fer2013/download
2. Real-time Convolutional Neural Networks for Emotion and Gender Classification https://arxiv.org/pdf/1710.07557.pdf
3. Trade-off between batch size and number of iterations to train a neural network: https://stats.stackexchange.com/a/236393/184082
4. Wiki-Adam: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
5. Batch Norm: https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
6. Dropout: https://arxiv.org/pdf/1904.03392.pdf
7. Benchmark on FER2013: https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013

## Run instructions

1. Install all required packages.
2. Run `python main.py`
3. Use `config.toml` to configure different run modes. Use `mode="FER-train"` for training the FER model, likewise other modes as defined in `config.toml`.
4. Successfully compiled with `mypy`, `isort` and `flake8`