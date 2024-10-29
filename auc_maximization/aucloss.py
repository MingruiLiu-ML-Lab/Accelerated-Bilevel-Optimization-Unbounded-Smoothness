import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score as  roc_auc

def roc_auc_score(label_id, logits):
    try:
        auc = roc_auc(label_id, logits)
    except ValueError as e:
        auc = 0.5
    return auc

class AUCMLoss(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function: a novel loss function to directly optimize AUROC

    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
    outputs:
        loss value

    Reference:
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T.,
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification.
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """

    def __init__(self, a, b, alpha, margin=1.0, imratio=None, device=None):
        super(AUCMLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.p = imratio
        self.a = a #torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = b# torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = alpha #torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    def forward(self, y_pred, y_true):
        # y_pred = torch.softmax(y_pred, dim=1)[:, -1] # [B, 2]
        y_pred = y_pred[:, -1] # [0.1 (cls=0), 0.9(cls=1)] => [0.9] only positive score
        y_pred = torch.sigmoid(y_pred)
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]

        y_pred = y_pred.reshape(-1, 1)  # be carefull about these shapes
        y_true = y_true.reshape(-1, 1)
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * (self.alpha+1) * (self.p * (1 - self.p) * self.margin + \
                                 torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (
                                             1 == y_true).float()))) - \
               self.p * (1 - self.p) * self.alpha ** 2
        return loss