**Images are uploaded to github image folder in case they are too small, attmept to zoom in on rpubs document if possible**

## Setup software on VM
+ Setup virtual machine on GCP compute engine
    + I chose a windows VM with desktop
+ Install Google chrome
+ First step was to initialize VM, Google chrome has a RDP add on which allows you to access VM easily
    + I didn't choose to load the VM with a container.  
        + There is potential that this could cause problems in workflow between team if other members  were to use Linux or some other version/platform.
    + Issues aside, loading up VM and installing Anaconda is extremely easy.  
        +  Anaconda comes with python 3, pip, Jupyter, and some base packages
        + This allows for pipenv setup as well as testing within jupyter for quick editing
        
## Accessing data
+ My initial csv files were loaded into my script files from github
    + Being that this project had us dealing with cloud, I decided all files from would be loaded from Google cloud storage buckets
+ The initial setup of the buckets was easy, but messing around with authorizations was not so easy.
    + That is why i decided to just make my bucket public
        + This can be accomplished through gsutil within Google command prompt and through the web browser like in the image below


![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_1_roles.PNG)

### Initial problems

+  Writing to buckets or accessing other Google API, from compute engine, requires permissions which need to be set on compute engine on installation.  
    + Luckily Google has updated the engine to allow for you to change these permissions, but the engine needs to be stopped in order to accomplish this 
+ Writing from python to buckets within virtual machine, required use of a custom function within the Google cloud documentation


![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_2_compute_engine_credentials.PNG)

## End Intial Setup
+ We now have a virtual machine, with anaconda loaded and access to read and write to Google storage buckets

## Create testing environment
+ Through experimentation within jupyter, I was able to edit my original scripts to begin testing in a clean environment.
+ This highlights a problem with the data flow process.
    + Building requirements.txt file is more of an iterative process.  
    + I wanted to make my project flow purely pythonic 
        + My requirements.txt file, needs to exist outside of my python code.  
            + Perhaps this is semantic, as several other aspects of the project were done through web browser, but it stuck with me that I couldn't build a requirements text from within an empty python environment
            + Therefore, I chose to load a preconstructed requirements.txt file into my virtual environment, which i document below

### Workflow

+ All steps below occur on new virtual environment compute engine

#### Step1- create pipenv

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_3_create%20pipenv.PNG)


+ Easy with anaconda

#### Step2- load _requirements

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_4_load%20_requirements.PNG)

+ Load in txt file
+ below I show packages in clean environment

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_5_proof%20of%20clean%20environment.PNG)


#### Step3- load csv files from cloud bucket

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_6_load%20pull_from_cloud.PNG)


+ Creates local df of train/test csv from original project

#### Step4- Begin loading scripts

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_7_run%20train_model.PNG)


+ Train_model.py does the following
+ Writes a classification_report_csv and pipeline locally to cwd
+ Writes a classification_report_csv and pipeline to Google cloud bucket
+ Images of new bucket displayed below

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_8_updated_bucket.PNG)


#### Step4- update old score script

+ load pipeline off cloud
    + new updated code posted below

![](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-hw3-justinherman42/blob/master/screenshots/step_9_score_model_differs.PNG)


+ Incorporating above code into score.py
+ Incorporating write to cloud for results

![](C:\Users\justin\Documents\GitHub\DATA 605\New_folder\step_10_score_model_upload_predicted_survival_to_cloud)


