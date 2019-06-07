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
import os
import time

import pickle

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from mesh_intersection.filter_faces import FilterFaces
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

from smplx import create


def main():
    description = 'Example script for untangling SMPL self intersections'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch SMPL-Untangle')
    parser.add_argument('--param_fn', type=str,
                        nargs='*',
                        required=True,
                        help='The pickle file with the model parameters')
    parser.add_argument('--interactive', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Display the mesh during the optimization' +
                        ' process')
    parser.add_argument('--delay', type=int, default=50,
                        help='The delay for the animation callback in ms')
    parser.add_argument('--model_folder', type=str,
                        default='models',
                        help='The path to the LBS model')
    parser.add_argument('--model_type', type=str,
                        default='smpl', choices=['smpl', 'smplx', 'smplh'],
                        help='The type of model to create')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to distance')
    parser.add_argument('--optimize_pose', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the joint pose')
    parser.add_argument('--optimize_shape', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the shape of the model')
    parser.add_argument('--sigma', default=0.5, type=float,
                        help='The height of the cone used to calculate the' +
                        ' distance field loss')
    parser.add_argument('--lr', default=1, type=float,
                        help='The learning rate for SGD')
    parser.add_argument('--coll_loss_weight', default=1e-4, type=float,
                        help='The weight for the collision loss')
    parser.add_argument('--pose_reg_weight', default=0, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--shape_reg_weight', default=0, type=float,
                        help='The weight for the shape regularizer')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--print_timings', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Print timings for all the operations')

    args = parser.parse_args()

    model_folder = args.model_folder
    model_type = args.model_type
    param_fn = args.param_fn
    interactive = args.interactive
    delay = args.delay
    point2plane = args.point2plane
    #  optimize_shape = args.optimize_shape
    #  optimize_pose = args.optimize_pose
    lr = args.lr
    coll_loss_weight = args.coll_loss_weight
    pose_reg_weight = args.pose_reg_weight
    shape_reg_weight = args.shape_reg_weight
    max_collisions = args.max_collisions
    sigma = args.sigma
    part_segm_fn = args.part_segm_fn
    print_timings = args.print_timings

    if interactive:
        import trimesh
        import pyrender

    device = torch.device('cuda')
    batch_size = len(param_fn)

    params_dict = defaultdict(lambda: [])
    for idx, fn in enumerate(param_fn):
        with open(fn, 'rb') as param_file:
            data = pickle.load(param_file, encoding='latin1')

        assert 'betas' in data, \
            'No key for shape parameter in provided pickle file'
        assert 'global_pose' in data, \
            'No key for the global pose in the given pickle file'
        assert 'pose' in data, \
            'No key for the pose of the joints in the given pickle file'

        for key, val in data.items():
            params_dict[key].append(val)

    params = {}
    for key in params_dict:
        params[key] = np.stack(params_dict[key], axis=0).astype(np.float32)
        if len(params[key].shape) < 2:
            params[key] = params[key][np.newaxis]
    if 'global_pose' in params:
        params['global_orient'] = params['global_pose']
    if 'pose' in params:
        params['body_pose'] = params['pose']

    if part_segm_fn:
        # Read the part segmentation
        with open(part_segm_fn, 'rb') as faces_parents_file:
            data = pickle.load(faces_parents_file, encoding='latin1')
        faces_segm = data['segm']
        faces_parents = data['parents']
        # Create the module used to filter invalid collision pairs
        filter_faces = FilterFaces(faces_segm, faces_parents).to(device=device)

    # Create the body model
    body = create(model_folder, batch_size=batch_size,
                  model_type=model_type).to(device=device)
    body.reset_params(**params)

    # Clone the given pose to use it as a target for regularization
    init_pose = body.body_pose.clone().detach()

    # Create the search tree
    search_tree = BVH(max_collisions=max_collisions)

    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)

    mse_loss = nn.MSELoss(reduction='sum').to(device=device)

    face_tensor = torch.tensor(body.faces.astype(np.int64), dtype=torch.long,
                               device=device).unsqueeze_(0).repeat([batch_size,
                                                                    1, 1])
    with torch.no_grad():
        output = body(get_skin=True)
        verts = output.vertices

    bs, nv = verts.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + \
        (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]

    optimizer = torch.optim.SGD([body.body_pose], lr=lr)

    if interactive:
        # Plot the initial mesh
        with torch.no_grad():
            output = body(get_skin=True)
            verts = output.vertices

            np_verts = verts.detach().cpu().numpy()

        def create_mesh(vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                        wireframe=False):

            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                          [1, 0, 0])
            tri_mesh.apply_transform(rot)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color)
            return pyrender.Mesh.from_trimesh(
                tri_mesh,
                material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                               ambient_light=(1.0, 1.0, 1.0))
        for bidx in range(np_verts.shape[0]):
            curr_verts = np_verts[bidx].copy()
            body_mesh = create_mesh(curr_verts, body.faces,
                                    color=(0.3, 0.3, 0.3, 0.99),
                                    wireframe=True)

            pose = np.eye(4)
            pose[0, 3] = bidx * 2
            scene.add(body_mesh,
                      name='body_mesh_{:03d}'.format(bidx),
                      pose=pose)

        viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                 viewport_size=(1200, 800),
                                 cull_faces=False,
                                 run_in_thread=True)

    query_names = ['recv_mesh', 'intr_mesh', 'body_mesh']

    step = 0
    while True:
        optimizer.zero_grad()

        if print_timings:
            start = time.time()

        if print_timings:
            torch.cuda.synchronize()
        output = body(get_skin=True)
        verts = output.vertices

        if print_timings:
            torch.cuda.synchronize()
            print('Body model forward: {:5f}'.format(time.time() - start))

        if print_timings:
            torch.cuda.synchronize()
            start = time.time()
        triangles = verts.view([-1, 3])[faces_idx]
        if print_timings:
            torch.cuda.synchronize()
            print('Triangle indexing: {:5f}'.format(time.time() - start))

        with torch.no_grad():
            if print_timings:
                start = time.time()
            collision_idxs = search_tree(triangles)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision Detection: {:5f}'.format(time.time() -
                                                          start))
            if part_segm_fn:
                if print_timings:
                    start = time.time()
                collision_idxs = filter_faces(collision_idxs)
                if print_timings:
                    torch.cuda.synchronize()
                    print('Collision filtering: {:5f}'.format(time.time() -
                                                              start))

        if print_timings:
            start = time.time()
        pen_loss = coll_loss_weight * \
            pen_distance(triangles, collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f}'.format(time.time() - start))

        shape_reg_loss = torch.tensor(0, device=device,
                                      dtype=torch.float32)
        if shape_reg_weight > 0:
            shape_reg_loss = shape_reg_weight * torch.sum(body.betas ** 2)
        pose_reg_loss = torch.tensor(0, device=device,
                                     dtype=torch.float32)
        if pose_reg_weight > 0:
            pose_reg_loss = pose_reg_weight * \
                mse_loss(body.pose, init_pose)

        loss = pen_loss + pose_reg_loss + shape_reg_loss

        np_loss = loss.detach().cpu().squeeze().tolist()
        if type(np_loss) != list:
            np_loss = [np_loss]
        msg = '{:.5f} ' * len(np_loss)
        print('Loss per model:', msg.format(*np_loss))

        if print_timings:
            start = time.time()
        loss.backward(torch.ones_like(loss))
        if print_timings:
            torch.cuda.synchronize()
            print('Backward pass: {:5f}'.format(time.time() - start))

        if interactive:
            with torch.no_grad():
                output = body(get_skin=True)
                verts = output.vertices

                np_verts = verts.detach().cpu().numpy()

            np_collision_idxs = collision_idxs.detach().cpu().numpy()
            np_receivers = np_collision_idxs[:, :, 0]
            np_intruders = np_collision_idxs[:, :, 1]

            viewer.render_lock.acquire()

            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if any([query in node.name for query in query_names]):
                    scene.remove_node(node)

            for bidx in range(batch_size):
                recv_faces_idxs = np_receivers[bidx][np_receivers[bidx] >= 0]
                intr_faces_idxs = np_intruders[bidx][np_intruders[bidx] >= 0]
                recv_faces = body.faces[recv_faces_idxs]
                intr_faces = body.faces[intr_faces_idxs]

                curr_verts = np_verts[bidx].copy()
                body_mesh = create_mesh(curr_verts, body.faces,
                                        color=(0.3, 0.3, 0.3, 0.99),
                                        wireframe=True)

                pose = np.eye(4)
                pose[0, 3] = bidx * 2
                scene.add(body_mesh,
                          name='body_mesh_{:03d}'.format(bidx),
                          pose=pose)

                if len(intr_faces) > 0:
                    intr_mesh = create_mesh(curr_verts, intr_faces,
                                            color=(0.9, 0.0, 0.0, 1.0))
                    scene.add(intr_mesh,
                              name='intr_mesh_{:03d}'.format(bidx),
                              pose=pose)

                if len(recv_faces) > 0:
                    recv_mesh = create_mesh(curr_verts, recv_faces,
                                            color=(0.0, 0.9, 0.0, 1.0))
                    scene.add(recv_mesh, name='recv_mesh_{:03d}'.format(bidx),
                              pose=pose)
            viewer.render_lock.release()

            if not viewer.is_active:
                break

            time.sleep(delay / 1000)
        optimizer.step()

        step += 1


if __name__ == '__main__':
    main()
