# Critical Thinking
## Analyzing my workflow

The approach I designed for migration onto AWS and GCP was essentially the same thing.  For the applied process of this project, you can see the steps taken to migrate our HW2 assignment onto GCP
[here](https://rpubs.com/justin_herman_42/428765).  Below I will briefly summarize some the steps.  I will then explain the failures in the approach I took, in terms of issues that can arise in terms of an Agile data science workflow.  After having addressed the issues with my approach, I will talk about some of the strengths and weaknesses of both products and cloud services. 

## Description of Workflow
+ Spin up a windows virtual machine with compute engine or EC2 engine
+ Grant access permissions to my VM
+ Install anaconda environment
    + Load up Jupyter and virtual environments
+ Create GCP storage buckets(or S3 storage) to write and read from
    + Can be done via command line,through web browser, or directly through python/script
+ Alter original scripts to read and write data to the buckets.
+ write results to cloud bucket

## Stengths of Workflow
+ Makes use of increased computing power
+ Makes use of increased storage
+ Easy setup, granting permissions is the most difficult part of process
+ Permissions can be given to anyone who needs to make use of the VM
+ Permissions can be given to anyone on team that needs to make use of storage buckets
    + Bucket can act as central portal for team members coordination.

## Weaknesses of Workflow
+ The Setup itself
    + Having not used a docker or a container, in order to load up a new instance on the compute engine I would have to go through the tedious steps of granting permissions and installing required programs onto my blank virtual machine
        + Team flow could be disrupted if others use different VM environment's which may have problems dealing with certain packages.
        + Even if other team members use the same environment, they may take different setup approaches which can cause issues
+ Lack of automation
    + Scripts must be run manually to produce results
    + Scripts are not built with enough error handling
    + Scripts are built more to get to algorithm, than to interact and provide insight with team members
+ Workflow doesn't make use of increased computing power
    + I didn't set up a cluster
    + Workflow does not provide a way to scale my projects onto multiple compute engines.

### Caveats to Weaknesses
+ Entire project is based on static data
    + There is no need to make use of increased computing power or to scrape data and provide insight  daily.  

## General description of differences of AWS and GCP
+ Taken from 
[link](https://hackernoon.com/aws-vs-google-cloud-platform-which-cloud-service-provider-to-choose-94a65e4ef0c5)
[link2](https://medium.com/@robaboukhalil/a-tale-of-two-clouds-amazon-vs-google-4f2520516a38)
+ It appears the differences between AWS and GCP are rather minor, specifically in the context of start up projects.  
    + AWS has been around the longest, therefore its tool data set is larger and more tested
    + I thought the setup tutorials for AWS were much better organized than GCP
    + GCP seems to have larger discounts and overall cheaper computing
    + GCP seems to have stronger data encryption 

## Big picture strengths/weaknesses of Cloud products
+ Strengths
    + Increasing computing power and storage capability with pay as you go options
+ Weakness
    + Depending on size of project, cost can be too high
    + Possible Network latency issues


# Applied section
+ In case the link was missed that was provided earlier, documentation of applied GCP compute engine can be found 
[here](https://rpubs.com/justin_herman_42/428765) and will be provided as an inline notebook file as well, titled "applied notebook"
