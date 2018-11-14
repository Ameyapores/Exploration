from model import resnet18
import torch.nn as nn

class exploration(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(exploration, self).__init__()
        resnet = resnet18(num_inputs, pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTMCell(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        features = self.resnet(inputs)
        #print (features.size())
        features = features.view(-1, 512)
        hx, cx = self.lstm(features, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)