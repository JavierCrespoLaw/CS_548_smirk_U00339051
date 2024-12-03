import os
import numpy as np
from datasets.base_dataset import BaseDataset
import cv2
import numpy as np



class ExpressionDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'Expression'
        

    def __getitem_aux__(self, index):
        image = cv2.imread(self.data_list[index][0])

        if not os.path.exists(self.data_list[index][1]):
            print('Mediapipe landmarks not found for %s'%(self.data_list[index]))
            return None
        
        landmarks_mediapipe = np.load(self.data_list[index][1], allow_pickle=True)

        data_dict = self.prepare_data(image=image, landmarks_mediapipe=landmarks_mediapipe)
        
        return data_dict


    
def get_datasets_Expressions(config):
    train_list = []
    image_list = []
    mediapipe_list = []

    for dirpath, dirnames, images in os.walk(config.dataset.Expression_Recognition_path):
        for image in images:
            
    #for folder in os.listdir(config.dataset.Expression_Recognition_val_path):
    #    for image in os.listdir(folder):

            if image.endswith(".jpg"):
                image_path = os.path.join(dirpath, image)

                mediapipe_landmarks_path = os.path.join(config.dataset.mediapipe, dirpath, image.split(".")[0] + ".npy")

                train_list.append([image_path, mediapipe_landmarks_path])

    dataset = ExpressionDataset(train_list, config, test=False)
    return dataset
    
def get_datasets_Expressions_val(config):
    train_list = []
    image_list = []
    mediapipe_list = []

    for dirpath, dirnames, images in os.walk(config.dataset.Expression_Recognition_val_path):
        for image in images:
            
    #for folder in os.listdir(config.dataset.Expression_Recognition_val_path):
    #    for image in os.listdir(folder):

            if image.endswith(".jpg"):
                image_path = os.path.join(dirpath, image)

                mediapipe_landmarks_path = os.path.join(config.dataset.mediapipe, dirpath, image.split(".")[0] + ".npy")

                train_list.append([image_path, mediapipe_landmarks_path])
                
    #for dirpath, dirnames, images in os.walk(config.dataset.Expression_Recognition_val_mediapipe_landmarks_path):
    #    for image in images:
            
    #for folder in os.listdir(config.dataset.Expression_Recognition_val_path):
    #    for image in os.listdir(folder):

    #        if image.endswith(".npy"):
    #            image_path = os.path.join(dirpath, image)
    #            mediapipe_list.append(image_path)
                #mediapipe_landmarks_path = os.path.join(dirpath, image.split(".")[0] + ".npy")
                
    #for i in range(len(mediapipe_list)):
    #    train_list.append([image_list[i], mediapipe_list[i]])

    dataset = ExpressionDataset(train_list, config, test=False)
    return dataset