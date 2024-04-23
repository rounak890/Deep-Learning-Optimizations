import torch
import torch.nn as nn

# prunning or netowkr sliming
def network_sliming(model,percent : float = 0.7):
    """It slims the network on basis of BatchNorm layers

    Args:
        model (nn.Module): The model to be pruned
        percent (float): prunning percentage. Defaults to 0.7.

    Returns:
        model : returns the pruned model
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            total += m.weight.data.shape[0]  # it gives the scaling factor of batch norm layer

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    thre_index = int(total * percent)


    y, i = torch.sort(bn)
    
    if total != 0:
        thre = y[thre_index]

        pruned = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre)
                mask = mask.float()
                pruned = pruned + mask.shape[0] - torch.sum(mask).item()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)

    return model
