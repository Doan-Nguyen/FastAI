from imports import * 
from configs import * 
from datasets import * 
from resnet import * 

if __name__ == '__main__':
    data = data_preprocess(configs.images_path)
    train(data)