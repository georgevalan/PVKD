import torch
from torch import nn, optim


import torch
import torch.nn as nn


def get_soft_target_loss_using_knowledge_distillation(teacher_logits, student_logits, T=2, soft_target_loss_weight=0.25):

    #Soften the student logits by applying softmax first and log() second
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by
    # the authors of the paper "Distilling the knowledge in a neural network"
    soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

    # Weighted sum of the two losses
    return soft_target_loss_weight * soft_targets_loss

# # Compare the student test accuracy with and without the teacher, after distillation
# print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
# print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
# print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")