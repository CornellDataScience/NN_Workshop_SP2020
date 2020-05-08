# NN_Workshop_SP2020
Code + Description for NN Workshop in Spring 2020

## Inital Setup
Install anaconda by following these instructions: https://www.cs.cornell.edu/courses/cs1110/2020sp/materials/python.html

After doing this, in terminal (Mac or Linux) or Anaconda Prompt (Windows) type in the following command:

```bash
conda env create -f environment/environment.yml --prefix ./environment
```

Then you will have everything installed and the virtual environment created!

Before starting to work with these files, make sure to run the command below:

```bash
conda activate ./environment
```

**Updating** if you previously ran the above commands:

```bash
conda env update --prefix ./environment --file environment/environment.yml  --prune
```

## Data Set up
Download the data zip folder, annotations, and img ids from [here](https://drive.google.com/).

1. Unzip the subset data zipfile. You should now have a coco folder.
2. Inside that folder you should see a folder labeled images. Create a folder named annotations
3. Put `instances_train2017.json` within the annotations folder
4. Make sure `image_ids.pkl` is in the coco folder

Now you are good to go!


## Model Weights

