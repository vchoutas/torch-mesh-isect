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

import numpy as np

import torch
import torch.nn as nn


class FilterFaces(nn.Module):

    def __init__(self, faces_parents=None, faces_segm=None,
                 ign_part_pairs=None):
        super(FilterFaces, self).__init__()

        if faces_parents is not None:
            faces_parents_tensor = torch.tensor(faces_parents,
                                                dtype=torch.long)
            self.register_buffer('faces_parents', faces_parents_tensor)
        else:
            self.faces_parents = None

        if faces_segm is not None:
            faces_segm_tensor = torch.tensor(faces_segm,
                                             dtype=torch.long)
            self.register_buffer('faces_segm', faces_segm_tensor)
        else:
            self.faces_segm = None

        self.extra_ign_pairs = False
        if ign_part_pairs is not None:
            pairs = map(lambda x: list(map(int, x.split(','))),
                        ign_part_pairs)
            pairs = np.asarray(list(pairs), dtype=np.int64)

            self.extra_ign_pairs = True
            self.register_buffer('extra_pairs',
                                 torch.tensor(pairs, dtype=torch.long))

    def forward(self, collision_idxs):
        # Get the part index for every collision pair
        if self.faces_segm is not None:
            recv_segm = self.faces_segm[collision_idxs[:, :, 0]]
            intr_segm = self.faces_segm[collision_idxs[:, :, 1]]

            # Find the collision pairs that are on the same part
            same_part_mask = recv_segm.eq(intr_segm).ge(1)\
                .to(collision_idxs.dtype)
        else:
            same_part_mask = torch.zeros_like(collision_idxs[:, :, 0])

        if self.faces_parents is not None:
            # Find the index of the parent part for every collision pair
            recv_parents = self.faces_parents[collision_idxs[:, :, 0]]
            intr_parents = self.faces_parents[collision_idxs[:, :, 1]]

            # If one member of the pair is on a part that is the parent of the
            # other, then ignore the collision
            kintree_mask = (recv_segm.eq(intr_parents) +
                            intr_segm.eq(recv_parents)).ge(1)\
                .to(collision_idxs.dtype)
        else:
            kintree_mask = torch.zeros_like(collision_idxs[:, :, 0])

        # If either of the above conditions is true, then the collision will be
        # ignored
        mask = ((kintree_mask + same_part_mask).ge(1) *
                collision_idxs[:, :, 0].ge(0)).to(collision_idxs.dtype)
        mask.unsqueeze_(dim=-1)

        if self.extra_ign_pairs and self.faces_segm is not None:
            mask1 = (
                recv_segm.unsqueeze(dim=-1).eq(self.extra_pairs[:, 0]) *
                intr_segm.unsqueeze(dim=-1).eq(self.extra_pairs[:, 1])).ge(1)\
                .sum(dim=-1, keepdim=True)
            mask2 = (
                recv_segm.unsqueeze(dim=-1).eq(self.extra_pairs[:, 1]) *
                intr_segm.unsqueeze(dim=-1).eq(self.extra_pairs[:, 0])).ge(1)\
                .sum(dim=-1, keepdim=True)

            mask += (mask1 + mask2).ge(1).to(dtype=collision_idxs.dtype)
            mask = mask.ge(1).to(dtype=collision_idxs.dtype)

        return mask * (-1) + (1 - mask) * collision_idxs
