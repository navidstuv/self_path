# Self Path
This is implementation of our paper [link]

## Sorting Dataset
For our experiment, we save all datasets in on folder, and then we created pickle file based on the image names and the budget of annotations: For example
Pickle file is a dictionary similar to {"Train1":{"WSI":[wsi names],"patches":{"Normal":[patch names], "Tumour":[patch_ names]}},
    "Train0.5":....}- "Train1", "Train0.5", "Train0.2" and etc. are related to budeget of annotation for 100, 50%, 20% and etc. "WSI" is for WSI names and "patches" are for the name of patches that are extracted from WSIs. Ofcourse, you can change the dataloader according to your preferences.
