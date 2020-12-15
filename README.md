# Self Path: Self Supervision for Classification of Histology Images with Limited Budget of Annotation
This is implementation of our paper [link]

## Sorting Dataset
For our experiment, we save all datasets in one folder, and then we created pickle file based on the image names and the budget of annotations: For example
Pickle file is a dictionary similar to {"Train1":{"WSI":[wsi names],"patches":{"Normal":[patch names], "Tumour":[patch_ names]}},
    "Train0.5":....}- "Train1", "Train0.5", "Train0.2" and etc. are related to budeget of annotation for 100, 50%, 20% and etc. "WSI" is for WSI names and "patches" are for the name of patches that are extracted from WSIs. Ofcourse, you can change the dataloader according to your preferences.
## Requirements    
`albumentations==0.3.1
numpy==1.17.0
tqdm==4.40.0
matplotlib==3.1.1
scikit_image==0.15.0
openslide_python==1.1.1
opencv_python_headless==4.1.0.25
pandas==0.25.1
config==0.5.0.post0
imutils==0.5.3
optimizers==v1.9
Pillow==8.0.1
scikit_learn==0.23.2
skimage==0.0
spams==2.6.2.5
tensorboardX==2.1
torch==1.7.1
torchvision==0.8.2`

## Usage
- All directories for the input data and also hyperparameters can be set in "config.py"
    - 'budget' in cinfig file indicates the budget of annotation according to the dictionary mentioned above.
    - tasks in config are set by defining following list/dictionaries:    
    
    ```task_names = ['main_task', 'magnification']#['main_task', 'magnification', 'jigsaw', 'domain_classifier', hematoxylin, 'rot']'
    
     tasks = {'magnification': {'type': 'classification_self', 'n_classes': 4},
             'main_task': {'type': 'classification_main', 'n_classes': 2},
             'jigsaw': {'type': 'classification_self', 'n_classes': 24},
             'domain_classifier': {'type': 'classification_adapt', 'n_classes': 2},
             'hematoxylin': {'type': 'pixel_self', 'n_classes': 1},
             'flip': {'type': 'classification_self', 'n_classes': 2},
             'rot': {'type': 'classification_self', 'n_classes': 4},
             'auto': {'type': 'pixel_self', 'n_classes': 3}
             }
             
    loss_weight = {'magnification': 1, 'domain_classifier': 1,
                   'main_task': 1, 'jigsaw': 1, 'hematoxylin': 1,
                   'flip': 1, 'rot':1, 'auto': 1}
                   
- Run 'main.py' to start training for designated budget of annotations.
    - You can use your [wandb](https://www.wandb.com) account for tracking experiments.
    
    
