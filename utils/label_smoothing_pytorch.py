import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=0.1, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = 1000

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        self.class_num = pred.size()[-1]

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()

if __name__ == '__main__':
    torch.manual_seed(15)
    criterion = LabelSmoothingCrossEntropy()
    out = torch.randn(20, 10)
    lbs = torch.randint(10, (20,))
    print('out:', out, out.size())
    print('lbs:', lbs, lbs.size())

    import torch.nn.functional as F
    
    loss = criterion(out, lbs)
    print('loss:', loss)
