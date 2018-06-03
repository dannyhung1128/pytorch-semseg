import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    log_p[:, 1] = log_p[:, 1] * 15 
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None: # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, size_average=size_average)

    return loss

def gamma_fast(gt, permutation):
    p = len(permutation)
    gt = gt.gather(0, Variable(permutation).cuda())
    gts = gt.sum()

    intersection = gts.float() - gt.float().cumsum(0)
    union = gts.float() + (1 - gt).float().cumsum(0)
    jaccard = 1. - intersection / union

    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def find_proximal(x0, gam, lam, eps=1e-6, max_steps=20, debug={}):
    # x0: sorted margins data
    # gam: initial gamma_fast(target, perm)
    # regularisation parameter lam
    x = x0.clone()
    act = (x >= eps).nonzero()
    finished = False
    if not act.size():
        finished = True
    else:
        active = act[-1, 0]
        members = {i: {i} for i in range(active + 1)}
        if active > 0:
            equal = (x[:active] - x[1:active+1]) < eps
            for i, e in enumerate(equal):
                if e:
                    members[i].update(members[i + 1])
                    members[i + 1] = members[i]
            project(gam, active, members)
    step = 0
    while not finished and step < max_steps and active > -1:
        step += 1
        res = compute_step_length(x, gam, active, eps)
        delta, ind = res
        
        if ind == -1:
            active = active - len(members[active])
        
        stop = torch.dot(x - x0, gam) / torch.dot(gam, gam) + 1. / lam
        if 0 <= stop < delta:
            delta = stop
            finished = True
        
        x = x - delta * gam
        if not finished:
            if ind >= 0:
                repr = min(members[ind])
                members[repr].update(members[ind + 1])
                for m in members[ind]:
                    if m != repr:
                        members[m] = members[repr]
            project(gam, active, members)
        if "path" in debug:
            debug["path"].append(x.numpy())

    if "step" in debug:
        debug["step"] = step
    if "finished" in debug:
        debug["finished"] = finished
    return x, gam


def lovasz_binary(margins, label, prox=False, max_steps=20, debug={}):
    # 1d vector inputs
    # Workaround: can't sort Variable bug
    # prox: False or lambda regularization value
    _, perm = torch.sort(margins.data, dim=0, descending=True)
    margins_sorted = margins[perm]
    grad = gamma_fast(label, perm)
    loss = torch.dot(F.relu(margins_sorted), grad)
    if prox is not False:
        xp, gam = find_proximal(margins_sorted.data, grad, prox, max_steps=max_steps, eps=1e-6, debug=debug)
        hook = margins_sorted.register_hook(lambda grad: Variable(margins_sorted.data - xp))
        return loss, hook, gam
    else:
        return loss

def lovasz_single(pred, label, prox=False, max_steps=20, debug={}):
    loss = 0
    for c in range(pred.size(1)):
        logit = pred[:, c, :, :]
        # single images
        mask_torch = (label.view(-1) == c)
        #num_preds = mask.long().sum()
        #print(num_preds)
        #if num_preds == 0:
        #    # only void pixels, the gradients should be 0
        #    return logits.sum() * 0.
        n, h, w = logit.size()
        nt, ht, wt = label.size()

        # Handle inconsistent size between input and target
        if h > ht and w > wt: # upsample labels
            label = label.unsqueeze(1)
            logit = logit.unsqueeze(1)
            label = F.upsample(label, size=(h, w), mode='nearest')
            label = label.squeeze(1)
            logit = logit.unsqueeze(1)
        elif h < ht and w < wt: # upsample images
            label = label.unsqueeze(1)
            logit = logit.unsqueeze(1)
            logit = F.upsample(logit, size=(ht, wt), mode='nearest')
            label = label.squeeze(1)
            logit = logit.unsqueeze(1)
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        mask = (label.view(-1) == c).data.cpu().numpy()
        #label[mask] = Variable(torch.LongTensor(np.ones(mask.size()))).cuda()
        target = np.zeros(label.size())
        target[mask] = 1
        target = Variable(torch.LongTensor(target).cuda())
        #target = Variable(target).cuda()
        target = target.contiguous().view(-1)[mask_torch]
        signs = 2. * target.float() - 1.
        logit = logit.contiguous().view(-1)[mask_torch]
        margins = (1. - logit * signs)
        loss += lovasz_binary(margins, target, prox, max_steps, debug=debug)
    return loss


def lovaszloss(logits, labels, prox=False, max_steps=20, debug={}):
    # image-level Lovasz hinge
    if len(logits) == 1:
        # single image case
        loss = lovasz_single(logits.squeeze(0), labels.squeeze(0), prox, max_steps, debug)
    else:
        losses = []
        for logit in logits:
            loss = lovasz_single(logit, labels, prox, max_steps, debug)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss
