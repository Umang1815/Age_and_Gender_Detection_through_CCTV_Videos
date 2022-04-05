import timm
import torch
import torch.nn as nn
from coral_pytorch.layers import CoralLayer

class Swin_Coral(nn.Module):
    def __init__(self, n_age=80):
        super().__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
        

        self.swin.head = nn.Identity()


        self.fc_age = CoralLayer(size_in=1024, num_classes=n_age)

        self.gender = nn.Linear(in_features = 1024, out_features  = 1)
        
       
    def forward(self, x):
        x = self.swin(x)
        x_gender = self.gender(x)
        logits =  self.fc_age(x)
        probas = torch.sigmoid(logits)
        

        return (logits, probas), x_gender