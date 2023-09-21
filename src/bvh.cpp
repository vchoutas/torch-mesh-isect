/*
   Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
   holder of all proprietary rights on this computer program.
   You can only use this computer program if you have closed
   a license agreement with MPG or you get the right to use the computer
   program from someone who is authorized to grant you that right.
   Any use of the computer program without a valid license is prohibited and
   liable to prosecution.

   Copyright©2019 Max-Planck-Gesellschaft zur Förderung
   der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
   for Intelligent Systems and the Max Planck Institute for Biological
   Cybernetics. All rights reserved.

   Contact: ps-license@tuebingen.mpg.de
*/

#include <torch/extension.h>

#include <vector>

void bvh_cuda_forward(at::Tensor triangles, at::Tensor* collision_tensor_ptr,
        int max_collisions = 16);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor bvh_forward(at::Tensor triangles, int max_collisions = 16) {
    CHECK_INPUT(triangles);
    at::Tensor collisionTensor = -1 * at::ones({triangles.size(0),
            triangles.size(1) * max_collisions, 2},
            at::device(triangles.device()).dtype(at::kLong));

    bvh_cuda_forward(triangles,
            &collisionTensor, max_collisions);

    return torch::autograd::make_variable(collisionTensor, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bvh_forward, "BVH collision forward (CUDA)",
        py::arg("triangles"), py::arg("max_collisions") = 16);
}
