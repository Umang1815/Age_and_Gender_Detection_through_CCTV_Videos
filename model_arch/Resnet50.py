from torchvision import models 
from torch import nn

modelresnet50 = models.resnet50(pretrained = True)
modelresnet50.fc = nn.Sequential(
  nn.Linear(in_features=2048, out_features=1024) ,
  nn.ReLU(),
  nn.Linear(in_features=1024, out_features=512) ,
  nn.ReLU(),
  nn.Dropout(p=0.6), 
  nn.Linear(in_features=512 , out_features=5)
)