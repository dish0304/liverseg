import torch
import torch.nn as nn
from torch.autograd import Function

def compute_dice(inputs,targets,pos_index=0):
    smooth=0.00001
    # print(inputs.shape)
    inputs=torch.sigmoid(inputs)
    # print('inputs',inputs.shape)
    foreground_probs=inputs[:,pos_index,...]
    targets=targets.float()
    # print(foreground_probs.shape)
    # print(targets.shape)
    intersection=(foreground_probs*targets).sum()
    denom=(foreground_probs**2+targets**2).sum()
    return (2*intersection+smooth)/(denom+smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,inputs,targets):
        loss=1.0-compute_dice(inputs,targets)

        return  loss


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size()[1]

        if targets.max()>class_num:
            print('the values in target is wrong')
            raise SystemExit()

        dice_sum = 0
        for i in range(0,class_num):
            inter = torch.sum(logits[:, i] * (targets==i))
            union = torch.sum(logits[:, i]) + torch.sum(targets==i)
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / (class_num)



class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)



class Direction_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred_flux, gt_flux, weight_matrix):
        b,c,h,w=pred_flux.shape
        device_id = pred_flux.device
        weight_matrix = weight_matrix.cuda(device_id)
        gt_flux = gt_flux.cuda(device_id)
        if b==1:
            gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)
            # norm loss
            norm_loss = weight_matrix * (pred_flux - gt_flux) ** 2
            norm_loss = norm_loss.sum()

            # angle loss
            pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1) + 1e-9)
            temp=torch.sum(pred_flux * gt_flux, dim=1)
            angle_loss = weight_matrix * (torch.acos(temp)) ** 2
            angle_loss = angle_loss.sum()

            loss = norm_loss + angle_loss
        else:
            loss=weight_matrix.sum()*0
            for i in range(b):
                pred=torch.zeros((1,c,h,w)).cuda(device_id)
                gt=torch.zeros((1,c,h,w)).cuda(device_id)
                weight=torch.zeros((1,h,w)).cuda(device_id)
                pred[0,:,:,:]=pred_flux[i,:,:,:]
                gt[0,:,:,:]=gt_flux[i,:,:,:]
                weight[0,:,:]=weight_matrix[i,:,:]
                gt = 0.999999 * gt / (gt.norm(p=2, dim=1) + 1e-9)

                # norm loss
                norm_loss = weight * (pred - gt) ** 2
                norm_loss = norm_loss.sum()

                # angle loss
                pred = 0.999999 * pred / (pred.norm(p=2, dim=1) + 1e-9)
                temp=torch.sum(pred * gt, dim=1)
                angle_loss = weight * (torch.acos(temp)) ** 2
                angle_loss = angle_loss.sum()

                loss=loss+norm_loss+angle_loss
        loss=loss/b
        return loss

