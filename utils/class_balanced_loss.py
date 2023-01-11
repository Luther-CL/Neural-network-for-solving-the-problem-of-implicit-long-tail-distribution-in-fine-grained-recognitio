import numpy as np
import torch
import torch.nn.functional as F


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma=None):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to('cuda:0')
    weights = weights.unsqueeze(0)
    weights = (weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot)
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)


    if loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


# if __name__ == '__main__':
#     no_of_classes = 5
#     logits = torch.rand(10, no_of_classes).float()
#     labels = torch.randint(0, no_of_classes, size=(10,))
#     beta = 0.9999
#     gamma = 2.0
#     samples_per_cls = [2, 3, 1, 2, 2]
#     loss_type = "focal"
#     cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
#     print(cb_loss)
