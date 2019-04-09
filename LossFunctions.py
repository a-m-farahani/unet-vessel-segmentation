from torch import nn
from torch.autograd import Function, Variable

class JaccardLoss(nn.Module):
    def __init__(self,eps=1e-7):
        super(JaccardLoss,self).__init__()
        self.epsilon = eps

    def forward(self,output,target):
        ious = 0
        counter = 0
        for i in range(output.size()[0]):
            for j in range(output.size()[1]):
                intersection = ((output[i,j,:,:]==1) & (target[i,j,:,:]==1)).sum()
                union = ((output[i,j,:,:]==1) | (target[i,j,:,:]==1)).sum()
                iou = intersection / (union + self.epsilon)
                ious += iou
                counter += 1
        return (ious/counter)*100
  
class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()
    def forward(self, output, mask):
        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss
  
def confusion(output, target):
    output[output>0]=1
    p = torch.sum(target==1).item()
    n = torch.sum(target==0).item()
    
    tp = (output*target).sum().item()
    tn = ((1-output)*(1-target)).sum().item()
    fp = ((1-target)*output).sum().item()
    fn = ((1-output)*target).sum().item()
              
    res = {"P":p,"N":n,"TP":tp,"TN":tn,"FP":fp,"FN":fn,"TPR":(tp/p),"TNR":(tn/n),"FPR":(fp/n),"FNR":(fn/p),"Accuracy":(tp+tn)/(p+n)}
    return res
