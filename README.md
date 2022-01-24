# AWS-MLE-Capstone-project
Udacity AWS Machine Learning Engineer Capstone Project
# Plants Disease Detection Using Deep Learning


## Code Files used throughout the project

* **capstone_project.ipynb** -- This jupyter notebook contains all the code and steps that we performed in this project and their outputs.
* **hpo_tuning.py** - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
* **code/train_model.py** - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning.This script contains code to calculate Accuray as well as other Metrics while the job is running.
* **code/requirements.txt** - Requirement files that contains external library requirements that are required by the training script
* **train_model_no_metrics.py** - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning but without any metrics calculation expect Accuray metrics.
* **endpoint_inference.py** - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences  and post-processing using the saved model from the training job.


## Non-code Files and Folders Description.

* **project_report.pdf** -- Final project report
* **project_report.docx** -- Docx version of Final project report
* **proposal.pdf** -- Accepted Project proposal
* **proposal.docx** -- Docx version of Final project proposal
* **testImages** -- This folder contains random test images that were used for evaluation of the deployed final model.
* **visualizations** -- This folder contains images of all relevant important visualizations like confusion matri, required for evaluation
* **snapshots** -- This folder contains images/snapshots of all relevant important evidence required for evaluation
* **benchmark_papers** -- Contains pdf copy of reference papers
* **profiler_report.zip** -- This contains a zipped file of the profiler outputs that were received while training the final ml model


## Third-party Libraries used in the project.
1. [split-folders](https://pypi.org/project/split-folders/) - Split folders with files (e.g. images) into train, validation and test (dataset) folders.
2. [tqdm](https://github.com/tqdm/tqdm) - Used to get visual updates/progress for copying files.
3. [torchmetrics] (https://torchmetrics.readthedocs.io/en/latest/) - For calculating metrics of the training jobs.


## Project Overview
### Domain Background

Plant diseases are one of the major factors responsible for substantial losses in yield of
plants, leading to huge economic losses. According to a study by the Associated Chambers
of Commerce and Industry of India, annual crop losses due to diseases and pest’s amount to
Rs.50,000 crore ($500 billion) in India alone, which is significant in a country where the
farmers are responsible for feeding a population of close to 1.3 billion people. The value of
plant science is therefore huge.
</br><p>Accurate identification and diagnosis of plant diseases are very important in the era of
climate change and globalization for food security. Accurate and early identification of plant
diseases could help in the prevention of spread of invasive pests/pathogens. In addition, for an
efficient and economical management of plant diseases accurate, sensitive and specific
diagnosis is necessary.</p>
The growth of GPU’s ( Graphical Processing Units ) has helped academics and business
in the advancement of Deep Learning methods, allowing them to explore deeper and more
sophisticated Neural Networks. Using concepts of Image Classification and Transfer
Learning we could train a Deep Learning model to categorize Plant leaf’s images to predict
whether the plant is healthy or has any diseases. This could help in the early detection of any
diseases in plants and could help take preventive measures to prevent huge crop losses.


This is an Image classification project that uses and fine-tunes a pretrained ResNet50 model with AWS Sagemaker for Plant disease classification from plant images.


## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. <br>
Open the jupter notebook "capstone_project.ipynb" and start by installing all the dependencies. <br>
For ease of use you may want to use "Python_Pytorch_36" Kernel so that you do not need to install most of the pytorch libraries <br>

## Dataset
We will be using the plant disease detection dataset[4] that is based on the original “PlantVillage Dataset”[5]. The dataset consists of 39 different classes, where 38 classes are of plant leaf categorised according to whether they are diseased or healthy and remaining one class consists of images with Background without plant leaves. The provided modified dataset[4] that we will be using has  been created using six different augmentation techniques on the original dataset[5] for increasing the data-set size. The techniques are image flipping, Gamma correction, noise injection, PCA colour augmentation, rotation, and scaling. The dataset that we will be using contains 61,486 images.
</br>
Given the dataset is huge and has 39 categories, for the purpose of this project we will be using only a subset of the plant classes for our project.
We will only be using the classes:</br>
Sr No.	Plant Image Class Name	Class Image dataset size
1.	Cherry_powdery_mildew	1053 images
2.	Cherry_healthy	1001 images
3.	Pepper_bacterial_spot	1000 images
4.	Pepper_healthy	1478 images
5.	Potato_early_blight	1000 images
6.	Potato_healthy	1001 images
7.	Potato_late_blight	1000 images
8.	Strawberry_healthy	1001 images
9.	Strawberry_leaf_scorch	1110 images


![S3 Upload Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/s3_dataset_zip_and_folder_snapshot.PNG)
Dataset Split into Train, Val, Test sets
![S3 Split dataset Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/s3_plant_datset_split_snapshot.PNG)

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 


* We will be using a pretrained Resnet50  model from pytorch vision library (https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html)
* We will be adding in two Fully connected Neural network Layers on top of the above Resnet50 model.
* Note: We will be using concepts of Transfer learning and so we will be freezing all the exisiting Convolutional layers in the pretrained resnet50 model and only changing gradients for the tow fully connected layers that we have added.
* Then we will perform Hyperparameter tuning, to help figure out the best hyperparameters to be used for our model.
* Next we will be using the best hyperparameters and fine-tuning our Resent50 model.
* We will also be adding in configuration for Profiling and Debugging our training mode by adding in relevant hooks in the Training and Testing( Evaluation) phases.
* Next we will be deploying our model. While deploying we will create our custom inference script. The custom inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.
* Finally we will be testing out our model with some test images of plants, to verfiy if the model is working as per our expectations.

## Hyperparameter Tuning

* The ResNet50 model with a two Fully connected Linear NN layer's is used for this image classification problem. ResNet-50 is 50 layers deep and is trained on a million images of 1000 categories from the ImageNet database. Furthermore the model has a lot of trainable parameters, which indicates a deep architecture that makes it better for image recognition
* The optimizer that we will be using for this model is AdamW ( For more info refer : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )
### Hyperparameter Tuning Sagemaker snapshot
![HPO Tuning Job Sagemaker](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/best_hyperparam_job_trial_1_summary.PNG)
### Multiple training jobs triggered by the HyperParameter Tuning Job
![HyperParameter Training Job Execution Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/hpo_tuning_training_jobs_trial_1.PNG)
### Best hyperparameter Training Job Accuracy
![Best Hyperparameters Training Job Log Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/best_hpo_training_job_trial_1_accuracy.PNG)

## Debugging and Profiling

We had set the Debugger hook to record and keep track of the Loss Criterion metrics of the process in training and validation/testing phases. The Plot of the Cross entropy loss is shown below:
![Cross Entropy Loss Tensor Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/training_crossentrophy_loss_graph.PNG)

## Results
Results look pretty good, as we had utilized the GPU while hyperparameter tuning and training of the fine-tuned ResNet50 model. We used the ml.g4dn.xlarge instance type for the runing the traiing purposes.
However while deploying the model to an endpoint we used the "ml.t2.medium" instance type to save cost and resources.

### Final Model's Testing Accuracy 
![Test_accuracy](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/Final_training_testing_metrics.PNG)


### Final Model's Validation Accuracy 
![Val_accuracy](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/Final_training_validation_metrics.PNG)

## Model Deployment
* Model was deployed to a "ml.t2.medium" instance type and we used the "endpoint_inference.py" script to setup and deploy our working endpoint.
* For testing purposes , we will be using some test images that we have stored in the "testImages" folder. 
* We will be reading in some test images from the folder and try to send those images as input and invoke our deployed endpoint
* We will be doing this via two approaches
  * Firstly using the Prdictor class object
  * Secondly using the boto3 client
### Deployed Active Endpoint Snapshot
![Deployed Endpoint Snapshot](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/deployed_model_endpoint.PNG)
### Deployed Endpoint Logs Snapshot, showing that the request was recieved and processed successfully by the endpoint


### Free-Form Testing Sample output returned from endpoint Snapshot
![Test Image_1](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/test_sample_1.PNG)
![Test Image_2](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/test_sample_2.PNG)
![Test Image_3](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/test_sample_3.PNG)
![Test Image_4](https://github.com/Prafull-parmar/AWS-MLE-Capstone-project/blob/main/snapshots/test_sample_4.PNG)


