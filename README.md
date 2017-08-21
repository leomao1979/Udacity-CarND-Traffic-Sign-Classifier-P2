## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[training_distribution]: ./output/training_data_distribution.png "Training Data Distribution"
[validation_distribution]: ./output/validation_data_distribution.png "Validation Data Distribution"
[test_distribution]: ./output/test_data_distribution.png "Test Data Distribution"
[random_picked_signs]: ./output/random_picked_signs.png "Randomly Picked Signs"
[rgb_gray_images]: ./output/rgb_gray_images.png "RGB Gray Images"

[augmented_images]: ./output/augmented_images.png "Augmented Images"
[extended_training_distribution]: ./output/extended_training_data_distribution.png "Extended Training Data Distribution"
[merged_training_distribution]: ./output/merged_training_data_distribution.png "Merged Training Data Distribution"

[downloaded_bumpy_road]: ./downloaded/bumpy_road.jpg "Bumpy Road Image"
[downloaded_pedestrians]: ./downloaded/pedestrians.jpg "Pedestrian Image"
[downloaded_speed_limit_30]: ./downloaded/speed_limit_30.jpg "Speed Limit 30 Image"
[downloaded_stop_sign]: ./downloaded/stop_sign.jpg "Stop Sign Image"
[downloaded_turn_left]: ./downloaded/turn_left.jpg "Turn Left Image"
[downloaded_turn_right]: ./downloaded/turn_right.jpg "Turn Right Image"

[figure_conv_layer1]: ./output/featuremap_pltnum_1.png "Figure of Convolutional Layer 1"
[figure_maxpooling_layer1]: ./output/featuremap_pltnum_2.png "Figure of Max Pooling Layer 1"
[figure_conv_layer2]: ./output/featuremap_pltnum_3.png "Figure of Convolutional Layer 2"
[figure_maxpooling_layer2]: ./output/featuremap_pltnum_4.png "Figure of Max Pooling Layer 2"

## Rubric Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one

You're reading it! and here is a link to my [project code](https://github.com/leomao1979/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are three bar charts showing the distribution of traffic signs in the training, validation and test set.

![Training Data Distribution][training_distribution]

![Validation Data Distribution][validation_distribution]

![Test Data Distribution][test_distribution]

Besides the bar charts, I also randomly picked 10 signs for each type to help understand the training set better.

![Randomly Picked Signs][random_picked_signs]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did the followings to preprocess image data.
1. Converted to grayscale using opencv
2. Augmented images with skimage to add more data to train set
3. Normalized to (-1, 1)

##### Convert to grayscale
Here is an example of a traffic sign image before and after grayscaling.

![RGB Gray Images][rgb_gray_images]

Grayscaling improves the training efficiency. My test on AWS shows that it takes around 25% less time to train the same data set with grayscale images than RGB images.

|                       | RGB images | Grayscale Images |
|:---------------------:|:----------------:|:---------------:|
| Original Training Set <br>（size: 34,799)   | 105s | 77s |
| Augmented Training Set <br> (size: 195,292) | 540s | 400s |

##### Augment images
Some sign classes have very few (less than 250) samples in the training set, so I decided to generate additional data to improve accuracy.

Here is an example of an original image and augmented images:

![Augmented Images][augmented_images]

To add more data to the the training set, I used numpy to flip images and skimage.transform to rotate and scale images.

Here is the distribution of generated data:

![Extended Training Data Distribution][extended_training_distribution]

Below is the data distribution combined of extended and original training set. The size is 191,662, around 5 times of the original one.

![Merged Training Data Distribution][merged_training_distribution]

##### Image normalization

I approximately normalized the image following suggestion of the instruction using '(pixel - 128) / 128', so the data ranges from -1.0 to 1.0.

#### 2. Describe what your final model architecture looks like

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400x120 weights, outputs 120        									|
| RELU					|						dropout	   	|
| Fully connected		| 120x80 weights, dropout, outputs 80       									|
| RELU					|			  dropout									|
| Fully connected		| 80x43 weights, outputs 43       									|
| Softmax				|  cross-entropy cost function       									|

#### 3. Describe how you trained your model

To train the model, I used an AdamOptimizer with the following hyperparameters:

rate = 0.001    # learning rate
EPOCHS = 20
BATCH_SIZE = 128
beta = 0.01

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.968
* test set accuracy of 0.946

I chose LeNet-5 for traffic sign classifier and did the followings:
* Implemented LeNet-5 network and trained with normalized data set, the validation accuracy I got was around 0.92. From logs of training loops, training set accuracy jumped high very quickly. It reached 0.99+ in just several loops then validation accuracy kept flat after that
* Then I added dropout to the two fully connected layers and trained again. From logs the training accuracy went up slower and we got several more loops to improve validation accuracy. The result of validation accuracy we got was around 0.94
* Then I generated additional data by flipping, rotating and scaling training images. The size of augmented training set is 195,292, around 5 times of the original set. The final result of validation accuracy is 0.968

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report

Here are six German traffic signs that I found on the web:

![Pedestrian][downloaded_pedestrians] ![Bumpy Road][downloaded_bumpy_road] ![Speed Limit 30][downloaded_speed_limit_30]
![Stop Sign][downloaded_stop_sign] ![Turn Left][downloaded_turn_left] ![Turn Right][downloaded_turn_right]

The third image might be difficult to classify because the speed limit sign takes a smaller portion of entire image, compare to randomly picked training images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

Here are the results of the prediction:

| Image			          |     Prediction	        				|
|:-------------------:|:-------------------------------:|
| Pedestrians      		| Pedestrians   									|
| Bumpy Road     			| Bumpy Road 										  |
| Speed limit (30km/h)| Speed limit (30km/h)				    |
| Stop	      		    | Stop					 				          |      
| Turn Left			      | Turn Left      							    |
| Turn Right			    | Turn Right      							  |

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.6%. I guess the reason is, these signs I downloaded from Internet have better quality than those in test set. Need to download and try more later.

It failed to classify the first and third signs before additional images were generated. After image augmentation, it got the 100% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

The model is pretty sure about all the images I downloaded. Should find some low quality images and test again.

The top five soft max probabilities of first image (Pedestrians) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 0.995         			  | Pedestrians   									  |
| 0.005     				    | General Caution										|
| 4.3e-08				        | Road narrows on the right					|
| 2.8e-10	      			  | Traffic signals  					 				|
| 7.2e-12   				    | Dangerous curve to the right      |

The top five soft max probabilities of second image (Bumpy Road) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 1.0         			    | Bumpy Road   									    |
| 6.2e-11     				  | Bicycles crossing									|
| 2.1e-14				        | Road narrows on the right					|
| 6.0e-20	      			  | Road work  					 				      |
| 5.42e-24   				    | Traffic signals                   |

The top five soft max probabilities of third image (Speed limit) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 0.9998         			  | Speed limit (30km/h)  				    |
| 0.0002      				  | Speed limit (80km/h)							|
| 2.35e-6				        | Keep right					              |
| 3.97e-08	      			| Speed limit (50km/h)	 				    |
| 1.4e-08   				    | End of speed limit (80km/h)       |

The top five soft max probabilities of fourth image (Stop) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 1.0         			    | Stop 				                      |
| 3.3e-11      				  | No vehicles							          |
| 9.05e-13				      | Priority road					            |
| 2.2e-13	      			  | Keep right	 				              |
| 7.8e-14   				    | Yield                             |

The top five soft max probabilities of fifth image (Turn left ahead) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 1.0         			    | Turn left ahead 	                |
| 1.2e-10      				  | Keep right							          |
| 9.7e-11				        | Ahead only					              |
| 7.8e-14   				    | Yield                             |
| 1.8e-14	      			  | Stop	 				                    |

The top five soft max probabilities of sixth image (Turn right ahead) were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| 1.0         			    | Turn right ahead 				          |
| 1.86e-16      				| Ahead only						            |
| 4.56e-26				      | Keep left					                |
| 1.67e-30	      			| Yield	 				                    |
| 9.65e-34   				    | No entry                          |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Below is the feature maps visualization for the third image, Speed Limit (30km/h).

Convolutional Layer 1, 28x28x6

![Figure of Convolutional Layer 1][figure_conv_layer1]

Max Pooling Layer 1, 14x14x6

![Figure of Max Pooling Layer 1][figure_maxpooling_layer1]

Convolutional Layer 2, 10x10x16

![Figure of Convolutional Layer 2][figure_conv_layer2]

Max Pooling Layer 2, 5x5x16

![Figure of Max Pooling Layer 2][figure_maxpooling_layer2]
