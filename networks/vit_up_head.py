import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .weight_init import trunc_normal_

from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VisionTransformerUpHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, num_frames = 6, num_patchs=720, img_size=(192,640), embed_dim=1024):
        super(VisionTransformerUpHead, self).__init__()

        self.MLP_q1 = Mlp(in_features=720, hidden_features=512, out_features=num_frames-1)
        self.MLP_q2 = Mlp(in_features=1024, hidden_features=512, out_features=4)

        self.MLP_p1 = Mlp(in_features=720, hidden_features=512, out_features=num_frames-1)
        self.MLP_p2 = Mlp(in_features=1024, hidden_features=512, out_features=3)


        

    def forward(self, x):
        """ TO DO: How to utilize the features to predict poses ?
        """

        outputs = {}
        features = x[-1]
        #print(features.shape)
        
        features = features.transpose(1,2)

        q = self.MLP_q1(features)
        q = q.transpose(1,2)
        q = self.MLP_q2(q)

        p = self.MLP_p1(features)
        p = p.transpose(1,2)
        p = self.MLP_p2(p)

        #print(q.shape)
        

        #axisangle
        #translation

        outputs["q"] = q
        outputs["p"] = p
        

        return outputs