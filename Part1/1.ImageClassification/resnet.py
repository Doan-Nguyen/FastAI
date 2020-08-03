from imports import * 
from datasets import *

def train(data):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    # learn.model
    ### train with 10 epochs 
    learn.fit_one_cycle(10)
    learn.save('stage-1')
    return learn

def val():
    learn = train(data)
    learn.unfreeze()
    learn.fit_one_cycle(1)
    learn.load('stage-1');
    learn.lr_find()
    learn.recorder.plot()
    learn.unfreeze()
    learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

