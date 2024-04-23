# it is KERNEL PRUNNING -> kernels also known as filters in CNN's are getting pruned and removed
from time import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def prune_model(model,percent = 0.8):
    """function for kernel/filter prunning for CNN based models

    Args:
        model (nn.Module): any pytorch model
        percent (float, optional): percent of prunning needed. Defaults to 0.8.

    Returns:
        model: returns the prunned model
    """
    total = 0
    total_kernel=0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
            oc,ic,h,w=m.weight.size()
            total_kernel+=m.weight.data.numel()/(w*h)


    conv_weights = torch.zeros(total)
    conv_max_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            oc,ic,h,w=m.weight.size()
            weight_max=torch.max(m.weight.data.abs().view(oc,ic,w*h),-1)[0].view(oc,ic,1,1).expand(oc,ic,h,w)
            conv_max_weights[index:(index+size)] = weight_max.contiguous().view(-1).clone()
            index += size

    y, i = torch.sort(conv_max_weights)
    thre_index = int(total * percent)
    ################ prune ###################
    thre = y[thre_index]
    zero_flag=False
    pruned = 0
    print('Percent {} ,Pruning threshold: {}'.format(percent,thre))
    index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            oc,ic,h,w=m.weight.size()
            mask = conv_max_weights[index:(index+size)].gt(thre).float().detach().view(oc,ic,h,w)

            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            index += size
            if int(torch.sum(mask)) == 0:
                zero_flag = True
    return model
