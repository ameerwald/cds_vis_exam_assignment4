
# Assignment 4 - Using pretrained CNNs for image classification again

## Github repo link 

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_vis_exam_assignment4).

## The data

For this assignment, the data is images of fruit from this [Kaggle dataset](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class). It is a collection of scraped images divided into 10 types of fruit. 

## Assignment description  

For this chosen assigment I chose to apply the VGG16 model to a different dataset 

- Write code which trains a classifier on this dataset using the pretrained model VGG16
- Save the training and validation history plots
- Save the classification report
- Discuss the results 



## Repository 

| Folder         | Description          
| ------------- |:-------------:
| Data      | I have hidden this folder because the dataset in too large to push to Github  
| Notes  | Notes to help me figure out how to run the code in a py script 
| Out  | Classification Report and history plot  
| Src  | Py script  
| Utils  | Preprocessing script



## To run the scripts 
As the dataset is too large to store in my repo, use the link above to access the data. Download and unzip the data. Then create a folder called  ```data``` within the assignment 4 folder, along with the other folders in the repo. Then the code will run without making any changes. If the data is placed elsewhere, then the path should be updated in the code. 

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment4/ folder level, run the following chunk of code. This will create a virtual environment, install the correct requirements, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```

This has been run on an ubuntu system on ucloud and therefore could have issues when run another way.

## Discussion of results 
The plot history seems to indicate that the model could be trained over more epochs based on the continuing downward curve. The accuracy is also quite poor overall at 0.11, especially for apples at 0. This could be due to the continued need for training as I mentioned but it more likly due to an imbalance in the data, the quality of the images or an issue with the model itself. I thought this would be better because while it is a small dataset I did use data augmentation but in comparison to the much larger datasets we have worked with previously, perhaps this is to be expected. 