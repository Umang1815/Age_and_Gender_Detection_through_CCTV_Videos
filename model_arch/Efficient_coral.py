import torch
import timm
import torch.nn as nn
from coral_pytorch.layers import CoralLayer


class Eff_Coral(nn.Module):
    def __init__(self, n_age=80):
        super().__init__()
        self.eff = timm.create_model('efficientnet_b4', pretrained=False)
        

        self.eff.classifier = nn.Identity()


        self.fc_age = CoralLayer(size_in=1792, num_classes=n_age)

        self.gender = nn.Linear(in_features = 1792, out_features  = 1)
        
       
    def forward(self, x):
        x = self.eff(x)
        x_gender = self.gender(x)
        logits =  self.fc_age(x)
        probas = torch.sigmoid(logits)
        

        return (logits, probas), x_gender