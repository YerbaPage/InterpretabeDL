from torch import nn

class Lwf_model(nn.Module):
    def __init__(self, model, anchor_model):
        super(Lwf_model, self).__init__()
        self.model=model
        self.anchor_model=anchor_model
        for param in self.anchor_model.parameters():
            param.requires_grad = False

    def forward(self,**inputs):
        ret = self.model(**inputs)
        ret_anchor = self.anchor_model(**inputs)
        return ret