import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pycocotools.coco import COCO


class ZoomDataset(Dataset):
    """ZoomDataset [summary]
    
    [extended_summary]
    
    :param path_to_pkl: Path to PKL file with Images
    :type path_to_pkl: str
    :param path_to_labels: path to file with labels
    :type path_to_labels: str
    """
    def __init__(self, path_to_image_ids, path_to_images, path_to_labels):
        self.coco = coco=COCO(path_to_labels)
        with open(path_to_image_ids, "rb") as fp:
            image_ids = pickle.load(fp)
        self.image_ids = image_ids
        self.image_dir = path_to_images
        self.num_images = len(image_ids)


    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return self.num_images

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        img_id = self.image_ids[index]
        img_dict = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_dict["file_name"])
        img = Image.open(img_path).resize((224,224), Image.BILINEAR)

        catIds = self.coco.getCatIds(catNms=['person'])
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        
        maskArr = np.zeros((img_dict['height'], img_dict['width']))
        for i in range(len(anns)):
            maskArr = np.maximum(self.coco.annToMask(anns[i]), maskArr)
        
        mask = Image.fromarray(maskArr).resize((224,224), Image.BILINEAR)

        return np.array(img), np.array(mask)


def get_data_loaders(path_to_image_ids,
                     path_to_images,
                     path_to_labels,
                     train_val_test=[0.8, 0.2, 0.2], 
                     batch_size=32):
    """get_data_loaders [summary]
    
    [extended_summary]
    
    :param path_to_csv: [description]
    :type path_to_csv: [type]
    :param train_val_test: [description], defaults to [0.8, 0.2, 0.2]
    :type train_val_test: list, optional
    :param batch_size: [description], defaults to 32
    :type batch_size: int, optional
    :return: [description]
    :rtype: [type]
    """
    # First we create the dataset given the path to the .csv file
    dataset = ZoomDataset(path_to_image_ids, path_to_images, path_to_labels)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed. You can take your code from last time

    ## BEGIN: YOUR CODE
    train_size = int(train_val_test[0]*dataset_size)
    val_size = int(train_val_test[1]*dataset_size)
    test_size = int(train_val_test[2]*dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:val_size+train_size]
    test_indices = indices[val_size+train_size:]
    ## END: YOUR CODE

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
