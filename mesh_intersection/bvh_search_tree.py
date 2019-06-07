# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_cuda


class BVHFunction(autograd.Function):

    max_collisions = 8

    @staticmethod
    @torch.no_grad()
    def forward(ctx, triangles):
        outputs = bvh_cuda.forward(triangles,
                                   max_collisions=BVHFunction.max_collisions)
        ctx.save_for_backward(outputs, triangles)
        return outputs

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class BVH(nn.Module):

    def __init__(self, max_collisions=8):
        super(BVH, self).__init__()
        self.max_collisions = max_collisions
        BVHFunction.max_collisions = self.max_collisions

    def forward(self, triangles):
        return BVHFunction.apply(triangles)
