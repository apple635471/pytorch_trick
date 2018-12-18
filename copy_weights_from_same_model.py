import torch 
import torchvision
import torchvision.models as models
import numpy as np

def copy_weights_from_same_model(model):
    new_model = models.resnet18()
    new_model.load_state_dict(model.state_dict())
    return new_model

if __name__ == "__main__":

    model = models.resnet18(pretrained=True)
    new_model = copy_weights_from_same_model(model)
    
    weight = model.state_dict()["conv1.weight"].numpy()
    new_weight = new_model.state_dict()["conv1.weight"].numpy()

    print(np.array_equal(weight, new_weight))
