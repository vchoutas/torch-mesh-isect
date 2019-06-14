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

#include <ATen/ATen.h>

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <vector>
#include <iostream>
#include <string>
#include <type_traits>

#include "double_vec_ops.h"
#include "helper_math.h"

// Size of the stack used to traverse the Bounding Volume Hierarchy tree
#ifndef STACK_SIZE
#define STACK_SIZE 64
#endif /* ifndef STACK_SIZE */

// Upper bound for the number of possible collisions
#ifndef MAX_COLLISIONS
#define MAX_COLLISIONS 16
#endif

#ifndef EPSILON
#define EPSILON 1e-16
#endif /* ifndef EPSILON */

// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#ifndef COLLISION_ORDERING
#define COLLISION_ORDERING 1
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

typedef unsigned int MortonCode;

template <typename T>
using vec3 = typename std::conditional<std::is_same<T, float>::value, float3,
                                       double3>::type;

template <typename T>
using vec2 = typename std::conditional<std::is_same<T, float>::value, float2,
                                       double2>::type;

template <typename T>
std::ostream &operator<<(std::ostream &os, const vec3<T> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}


std::ostream &operator<<(std::ostream &os, const vec3<float> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<double> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, vec3<T> x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

__host__ __device__ inline double3 fmin(const double3 &a, const double3 &b) {
  return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3 &a, const double3 &b) {
  return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
  __host__ __device__ int operator()(long2 vec) const {
    return vec.x >= 0 && vec.y >= 0;
  }
};

template <typename T>
__host__ __device__ __forceinline__ float vec_abs_diff(const vec3<T> &vec1,
                                                       const vec3<T> &vec2) {
  return fabs(vec1.x - vec2.x) + fabs(vec1.y - vec2.y) + fabs(vec1.z - vec2.z);
}

template <typename T>
__host__ __device__ __forceinline__ float vec_sq_diff(const vec3<T> &vec1,
                                                      const vec3<T> &vec2) {
  return dot(vec1 - vec2, vec1 - vec2);
}

template <typename T> struct AABB {
public:
  __host__ __device__ AABB() {
    min_t.x = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
    min_t.y = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
    min_t.z = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

    max_t.x = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
    max_t.y = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
    max_t.z = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
  };

  __host__ __device__ AABB(const vec3<T> &min_t, const vec3<T> &max_t)
      : min_t(min_t), max_t(max_t){};
  __host__ __device__ ~AABB(){};

  __host__ __device__ AABB(T min_t_x, T min_t_y, T min_t_z, T max_t_x,
                           T max_t_y, T max_t_z) {
    min_t.x = min_t_x;
    min_t.y = min_t_y;
    min_t.z = min_t_z;
    max_t.x = max_t_x;
    max_t.y = max_t_y;
    max_t.z = max_t_z;
  }

  __host__ __device__ AABB<T> operator+(const AABB<T> &bbox2) const {
    return AABB<T>(
        min(this->min_t.x, bbox2.min_t.x), min(this->min_t.y, bbox2.min_t.y),
        min(this->min_t.z, bbox2.min_t.z), max(this->max_t.x, bbox2.max_t.x),
        max(this->max_t.y, bbox2.max_t.y), max(this->max_t.z, bbox2.max_t.z));
  };

  __host__ __device__ T operator*(const AABB<T> &bbox2) const {
    return (min(this->max_t.x, bbox2.max_t.x) -
            max(this->min_t.x, bbox2.min_t.x)) *
           (min(this->max_t.y, bbox2.max_t.y) -
            max(this->min_t.y, bbox2.min_t.y)) *
           (min(this->max_t.z, bbox2.max_t.z) -
            max(this->min_t.z, bbox2.min_t.z));
  };

  vec3<T> min_t;
  vec3<T> max_t;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const AABB<T> &x) {
  os << x.min_t << std::endl;
  os << x.max_t << std::endl;
  return os;
}

template <typename T> struct MergeAABB {

public:
  __host__ __device__ MergeAABB(){};

  // Create an operator Struct that will be used by thrust::reduce
  // to calculate the bounding box of the scene.
  __host__ __device__ AABB<T> operator()(const AABB<T> &bbox1,
                                         const AABB<T> &bbox2) {
    return bbox1 + bbox2;
  };
};

template <typename T> struct Triangle {
public:
  vec3<T> v0;
  vec3<T> v1;
  vec3<T> v2;

  __host__ __device__ Triangle(const vec3<T> &vertex0, const vec3<T> &vertex1,
                               const vec3<T> &vertex2)
      : v0(vertex0), v1(vertex1), v2(vertex2){};

  __host__ __device__ AABB<T> ComputeBBox() {
    return AABB<T>(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)),
                   min(v0.z, min(v1.z, v2.z)), max(v0.x, max(v1.x, v2.x)),
                   max(v0.y, max(v1.y, v2.y)), max(v0.z, max(v1.z, v2.z)));
  }
};

template <typename T> using TrianglePtr = Triangle<T> *;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Triangle<T> &x) {
  os << x.v0 << std::endl;
  os << x.v1 << std::endl;
  os << x.v2 << std::endl;
  return os;
}

template <typename T>
__global__ void ComputeTriBoundingBoxes(Triangle<T> *triangles,
                                        int num_triangles, AABB<T> *bboxes) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_triangles) {
    bboxes[idx] = triangles[idx].ComputeBBox();
  }
}

template <typename T>
__device__ inline vec2<T> isect_interval(const vec3<T> &sep_axis,
                                         const Triangle<T> &tri) {
  // Check the separating sep_axis versus the first point of the triangle
  T proj_distance = dot(sep_axis, tri.v0);

  vec2<T> interval;
  interval.x = proj_distance;
  interval.y = proj_distance;

  proj_distance = dot(sep_axis, tri.v1);
  interval.x = min(interval.x, proj_distance);
  interval.y = max(interval.y, proj_distance);

  proj_distance = dot(sep_axis, tri.v2);
  interval.x = min(interval.x, proj_distance);
  interval.y = max(interval.y, proj_distance);

  return interval;
}

template <typename T>
__device__ inline bool TriangleTriangleOverlap(const Triangle<T> &tri1,
                                               const Triangle<T> &tri2,
                                               const vec3<T> &sep_axis) {
  // Calculate the projected segment of each triangle on the separating
  // axis.
  vec2<T> tri1_interval = isect_interval(sep_axis, tri1);
  vec2<T> tri2_interval = isect_interval(sep_axis, tri2);

  // In order for the triangles to overlap then there must exist an
  // intersection of the two intervals
  return (tri1_interval.x <= tri2_interval.y) &&
         (tri1_interval.y >= tri2_interval.x);
}

template <typename T>
__device__ bool TriangleTriangleIsectSepAxis(const Triangle<T> &tri1,
                                             const Triangle<T> &tri2) {
  // Calculate the edges and the normal for the first triangle
  vec3<T> tri1_edge0 = tri1.v1 - tri1.v0;
  vec3<T> tri1_edge1 = tri1.v2 - tri1.v0;
  vec3<T> tri1_edge2 = tri1.v2 - tri1.v1;
  vec3<T> tri1_normal = cross(tri1_edge1, tri1_edge2);

  // Calculate the edges and the normal for the second triangle
  vec3<T> tri2_edge0 = tri2.v1 - tri2.v0;
  vec3<T> tri2_edge1 = tri2.v2 - tri2.v0;
  vec3<T> tri2_edge2 = tri2.v2 - tri2.v1;
  vec3<T> tri2_normal = cross(tri2_edge1, tri2_edge2);

  // If the triangles are coplanar then the first 11 cases are all the same,
  // since the cross product will just give us the normal vector
  vec3<T> axes[17] = {
      tri1_normal,
      tri2_normal,
      cross(tri1_edge0, tri2_edge0),
      cross(tri1_edge0, tri2_edge1),
      cross(tri1_edge0, tri2_edge2),
      cross(tri1_edge1, tri2_edge0),
      cross(tri1_edge1, tri2_edge1),
      cross(tri1_edge1, tri2_edge2),
      cross(tri1_edge2, tri2_edge0),
      cross(tri1_edge2, tri2_edge1),
      cross(tri1_edge2, tri2_edge2),
      // Triangles are coplanar
      // Check the axis created by the normal of the triangle and the edges of
      // both triangles.
      cross(tri1_normal, tri1_edge0),
      cross(tri1_normal, tri1_edge1),
      cross(tri1_normal, tri1_edge2),
      cross(tri1_normal, tri2_edge0),
      cross(tri1_normal, tri2_edge1),
      cross(tri1_normal, tri2_edge2),
  };

  bool isect_flag = true;
#pragma unroll
  for (int i = 0; i < 17; ++i) {
    isect_flag = isect_flag && (TriangleTriangleOverlap(tri1, tri2, axes[i]));
  }

  return isect_flag;
}

// Returns true if the triangles share one or multiple vertices
template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
bool
shareVertex(const Triangle<T> &tri1, const Triangle<T> &tri2) {

    return (tri1.v0.x == tri2.v0.x && tri1.v0.y == tri2.v0.y && tri1.v0.z == tri2.v0.z) ||
        (tri1.v0.x == tri2.v1.x && tri1.v0.y == tri2.v1.y && tri1.v0.z == tri2.v1.z) ||
        (tri1.v0.x == tri2.v2.x && tri1.v0.y == tri2.v2.y && tri1.v0.z == tri2.v2.z) ||
        (tri1.v1.x == tri2.v0.x && tri1.v1.y == tri2.v0.y && tri1.v1.z == tri2.v0.z) ||
        (tri1.v1.x == tri2.v1.x && tri1.v1.y == tri2.v1.y && tri1.v1.z == tri2.v1.z) ||
        (tri1.v1.x == tri2.v2.x && tri1.v1.y == tri2.v2.y && tri1.v1.z == tri2.v2.z) ||
        (tri1.v2.x == tri2.v0.x && tri1.v2.y == tri2.v0.y && tri1.v2.z == tri2.v0.z) ||
        (tri1.v2.x == tri2.v1.x && tri1.v2.y == tri2.v1.y && tri1.v2.z == tri2.v1.z) ||
        (tri1.v2.x == tri2.v2.x && tri1.v2.y == tri2.v2.y && tri1.v2.z == tri2.v2.z);
}

template <typename T>
__global__ void checkTriangleIntersections(long2 *collisions,
                                           Triangle<T> *triangles,
                                           int num_cand_collisions,
                                           int num_triangles) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_cand_collisions) {
    int first_tri_idx = collisions[idx].x;
    int second_tri_idx = collisions[idx].y;

    Triangle<T> tri1 = triangles[first_tri_idx];
    Triangle<T> tri2 = triangles[second_tri_idx];
    bool do_collide = TriangleTriangleIsectSepAxis<T>(tri1, tri2) &&
                      !shareVertex<T>(tri1, tri2);
    if (do_collide) {
      collisions[idx] = make_long2(first_tri_idx, second_tri_idx);
    } else {
      collisions[idx] = make_long2(-1, -1);
    }
  }
  return;
}

template <typename T> struct BVHNode {
public:
  AABB<T> bbox;

  BVHNode<T> *left;
  BVHNode<T> *right;
  BVHNode<T> *parent;
  // Stores the rightmost leaf node that can be reached from the current
  // node.
  BVHNode<T> *rightmost;

  __host__ __device__ inline bool isLeaf() { return !left && !right; };

  // The index of the object contained in the node
  int idx;
};

template <typename T> using BVHNodePtr = BVHNode<T> *;

template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    bool
    checkOverlap(const AABB<T> &bbox1, const AABB<T> &bbox2) {
  return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) &&
         (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) &&
         (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

template <typename T>
__device__ int traverseBVH(long2 *collisionIndices, BVHNodePtr<T> root,
                           const AABB<T> &queryAABB, int queryObjectIdx,
                           BVHNodePtr<T> leaf, int max_collisions,
                           int *counter) {
  int num_collisions = 0;
  // Allocate traversal stack from thread-local memory,
  // and push NULL to indicate that there are no postponed nodes.
  BVHNodePtr<T> stack[STACK_SIZE];
  BVHNodePtr<T> *stackPtr = stack;
  *stackPtr++ = nullptr; // push

  // Traverse nodes starting from the root.
  BVHNodePtr<T> node = root;
  do {
    // Check each child node for overlap.
    BVHNodePtr<T> childL = node->left;
    BVHNodePtr<T> childR = node->right;
    bool overlapL = checkOverlap<T>(queryAABB, childL->bbox);
    bool overlapR = checkOverlap<T>(queryAABB, childR->bbox);

#if COLLISION_ORDERING == 1
    /*
       If we do not impose any order, then all potential collisions will be
       reported twice (i.e. the query object with the i-th colliding object
       and the i-th colliding object with the query). In order to avoid
       this, we impose an ordering, saying that an object can collide with
       another only if it comes before it in the tree. For example, if we
       are checking for the object 10, there is no need to check the subtree
       that has the objects that are before it, since they will already have
       been checked.
    */
    if (leaf >= childL->rightmost) {
      overlapL = false;
    }
    if (leaf >= childR->rightmost) {
      overlapR = false;
    }
#endif

    // Query overlaps a leaf node => report collision.
    if (overlapL && childL->isLeaf()) {
      // Append the collision to the main array
      // Increase the number of detection collisions
      // num_collisions++;
      int coll_idx = atomicAdd(counter, 1);
      collisionIndices[coll_idx] =
          // collisionIndices[num_collisions % max_collisions] =
          // *collisionIndices++ =
          make_long2(min(queryObjectIdx, childL->idx),
                     max(queryObjectIdx, childL->idx));
      num_collisions++;
    }

    if (overlapR && childR->isLeaf()) {
      int coll_idx = atomicAdd(counter, 1);
      collisionIndices[coll_idx] = make_long2(
          // min(queryObjectIdx, childR->idx),
          // max(queryObjectIdx, childR->idx));
          // collisionIndices[num_collisions % max_collisions] = make_long2(
          min(queryObjectIdx, childR->idx), max(queryObjectIdx, childR->idx));
      num_collisions++;
    }

    // Query overlaps an internal node => traverse.
    bool traverseL = (overlapL && !childL->isLeaf());
    bool traverseR = (overlapR && !childR->isLeaf());

    if (!traverseL && !traverseR) {
      node = *--stackPtr; // pop
    }
    else {
        node = (traverseL) ? childL : childR;
        if (traverseL && traverseR) {
            *stackPtr++ = childR; // push
        }
    }
  } while (node != nullptr);

  return num_collisions;
}

template <typename T>
__global__ void findPotentialCollisions(long2 *collisionIndices,
                                        BVHNodePtr<T> root,
                                        BVHNodePtr<T> leaves, int *triangle_ids,
                                        int num_primitives,
                                        int max_collisions, int *counter) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_primitives) {

    BVHNodePtr<T> leaf = leaves + idx;
    int triangle_id = triangle_ids[idx];
    int num_collisions =
        traverseBVH<T>(collisionIndices, root, leaf->bbox, triangle_id,
                       leaf, max_collisions, counter);
  }
  return;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        expandBits(MortonCode v) {
  // Shift 16
  v = (v * 0x00010001u) & 0xFF0000FFu;
  // Shift 8
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  // Shift 4
  v = (v * 0x00000011u) & 0xC30C30C3u;
  // Shift 2
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        morton3D(T x, T y, T z) {
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  MortonCode xx = expandBits((MortonCode)x);
  MortonCode yy = expandBits((MortonCode)y);
  MortonCode zz = expandBits((MortonCode)z);
  return xx * 4 + yy * 2 + zz;
}

template <typename T>
__global__ void ComputeMortonCodes(Triangle<T> *triangles, int num_triangles,
                                   AABB<T> *scene_bb,
                                   MortonCode *morton_codes) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_triangles) {
    // Fetch the current triangle
    Triangle<T> tri = triangles[idx];
    vec3<T> centroid = (tri.v0 + tri.v1 + tri.v2) / (T)3.0;

    T x = (centroid.x - scene_bb->min_t.x) /
          (scene_bb->max_t.x - scene_bb->min_t.x);
    T y = (centroid.y - scene_bb->min_t.y) /
          (scene_bb->max_t.y - scene_bb->min_t.y);
    T z = (centroid.z - scene_bb->min_t.z) /
          (scene_bb->max_t.z - scene_bb->min_t.z);

    morton_codes[idx] = morton3D<T>(x, y, z);
  }
  return;
}

__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    int
    LongestCommonPrefix(int i, int j, MortonCode *morton_codes,
                        int num_triangles, int *triangle_ids) {
  // This function will be called for i - 1, i, i + 1, so we might go beyond
  // the array limits
  if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
    return -1;

  MortonCode key1 = morton_codes[i];
  MortonCode key2 = morton_codes[j];

  if (key1 == key2) {
    // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
    // bits in key[1, 2]. Add the number of leading zeros between the
    // indices
    return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
  } else {
    // Keys are different
    return __clz(key1 ^ key2);
  }
}

template <typename T>
__global__ void BuildRadixTree(MortonCode *morton_codes, int num_triangles,
                               int *triangle_ids, BVHNodePtr<T> internal_nodes,
                               BVHNodePtr<T> leaf_nodes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_triangles - 1)
    return;

  int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
                                       num_triangles, triangle_ids);
  int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
                                       num_triangles, triangle_ids);
  // Find the direction of the range
  int direction = delta_next - delta_last >= 0 ? 1 : -1;

  int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
                                      num_triangles, triangle_ids);

  // Do binary search to compute the upper bound for the length of the range
  int lmax = 2;
  while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
                             num_triangles, triangle_ids) > delta_min) {
    lmax *= 2;
  }

  // Use binary search to find the other end.
  int l = 0;
  int divider = 2;
  for (int t = lmax / divider; t >= 1; divider *= 2) {
    if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
                            num_triangles, triangle_ids) > delta_min) {
      l = l + t;
    }
    t = lmax / divider;
  }
  int j = idx + l * direction;

  // Find the length of the longest common prefix for the current node
  int node_delta =
      LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
  int s = 0;
  divider = 2;
  // Search for the split position using binary search.
  for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
    if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
                            num_triangles, triangle_ids) > node_delta) {
      s = s + t;
    }
    t = (l + (divider - 1)) / divider;
  }
  // gamma in the Karras paper
  int split = idx + s * direction + min(direction, 0);

  // Assign the parent and the left, right children for the current node
  BVHNodePtr<T> curr_node = internal_nodes + idx;
  if (min(idx, j) == split) {
    curr_node->left = leaf_nodes + split;
    (leaf_nodes + split)->parent = curr_node;
  } else {
    curr_node->left = internal_nodes + split;
    (internal_nodes + split)->parent = curr_node;
  }
  if (max(idx, j) == split + 1) {
    curr_node->right = leaf_nodes + split + 1;
    (leaf_nodes + split + 1)->parent = curr_node;
  } else {
    curr_node->right = internal_nodes + split + 1;
    (internal_nodes + split + 1)->parent = curr_node;
  }
}

template <typename T>
__global__ void CreateHierarchy(BVHNodePtr<T> internal_nodes,
                                BVHNodePtr<T> leaf_nodes, int num_triangles,
                                Triangle<T> *triangles, int *triangle_ids,
                                int *atomic_counters) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_triangles)
    return;

  BVHNodePtr<T> leaf = leaf_nodes + idx;
  // Assign the index to the primitive
  leaf->idx = triangle_ids[idx];

  Triangle<T> tri = triangles[triangle_ids[idx]];
  // Assign the bounding box of the triangle to the leaves
  leaf->bbox = tri.ComputeBBox();
  leaf->rightmost = leaf;

  BVHNodePtr<T> curr_node = leaf->parent;
  int current_idx = curr_node - internal_nodes;

  // Increment the atomic counter
  int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
  while (true) {
    // atomicAdd returns the old value at the specified address. Thus the
    // first thread to reach this point will immediately return
    if (curr_counter == 0)
      break;

    // Calculate the bounding box of the current node as the union of the
    // bounding boxes of its children.
    AABB<T> left_bb = curr_node->left->bbox;
    AABB<T> right_bb = curr_node->right->bbox;
    curr_node->bbox = left_bb + right_bb;
    // Store a pointer to the right most node that can be reached from this
    // internal node.
    curr_node->rightmost =
        curr_node->left->rightmost > curr_node->right->rightmost
            ? curr_node->left->rightmost
            : curr_node->right->rightmost;

    // If we have reached the root break
    if (curr_node == internal_nodes)
      break;

    // Proceed to the parent of the node
    curr_node = curr_node->parent;
    // Calculate its position in the flat array
    current_idx = curr_node - internal_nodes;
    // Update the visitation counter
    curr_counter = atomicAdd(atomic_counters + current_idx, 1);
  }

  return;
}

template <typename T>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T>* __restrict__ triangles,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size) {

#if PRINT_TIMINGS == 1
  // Create the CUDA events used to estimate the execution time of each
  // kernel.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  thrust::device_vector<AABB<T>> bounding_boxes(num_triangles);

  int blockSize = NUM_THREADS;
  int gridSize = (num_triangles + blockSize - 1) / blockSize;
#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the bounding box for all the triangles
#if DEBUG_PRINT == 1
  std::cout << "Start computing triangle bounding boxes" << std::endl;
#endif
  ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, bounding_boxes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished computing triangle bounding_boxes" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Compute Triangle Bounding boxes = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the union of all the bounding boxes
  AABB<T> host_scene_bb = thrust::reduce(
      bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished Calculating scene Bounding Box" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Scene bounding box reduction = " << milliseconds << " (ms)"
            << std::endl;
#endif

  // TODO: Custom reduction ?
  // Copy the bounding box back to the GPU
  AABB<T> *scene_bb_ptr;
  cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
  cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
             cudaMemcpyHostToDevice);

  thrust::device_vector<MortonCode> morton_codes(num_triangles);
#if DEBUG_PRINT == 1
  std::cout << "Start Morton Code calculation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the morton codes for the centroids of all the primitives
  ComputeMortonCodes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, scene_bb_ptr,
      morton_codes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished calculating Morton Codes ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Morton code calculation = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if DEBUG_PRINT == 1
  std::cout << "Creating triangle ID sequence" << std::endl;
#endif
  // Construct an array of triangle ids.
  thrust::sequence(triangle_ids->begin(), triangle_ids->end());
#if DEBUG_PRINT == 1
  std::cout << "Finished creating triangle ID sequence ..." << std::endl;
#endif

  // Sort the triangles according to the morton code
#if DEBUG_PRINT == 1
  std::cout << "Starting Morton Code sorting!" << std::endl;
#endif

  try {
#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                        triangle_ids->begin());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
    std::cout << "Finished morton code sorting!" << std::endl;
#endif
#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Morton code sorting = " << milliseconds << " (ms)"
              << std::endl;
#endif
  } catch (thrust::system_error e) {
    std::cout << "Error inside sort: " << e.what() << std::endl;
  }

#if DEBUG_PRINT == 1
  std::cout << "Start building radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Construct the radix tree using the sorted morton code sequence
  BuildRadixTree<T><<<gridSize, blockSize>>>(
      morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
      internal_nodes, leaf_nodes);
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Building radix tree = " << milliseconds << " (ms)" << std::endl;
#endif
  // Create an array that contains the atomic counters for each node in the
  // tree
  thrust::device_vector<int> counters(num_triangles);

#if DEBUG_PRINT == 1
  std::cout << "Start Linear BVH generation" << std::endl;
#endif
  // Build the Bounding Volume Hierarchy in parallel from the leaves to the
  // root
  CreateHierarchy<T><<<gridSize, blockSize>>>(
      internal_nodes, leaf_nodes, num_triangles, triangles,
      triangle_ids->data().get(), counters.data().get());

  cudaCheckError();

#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
  std::cout << "Finished with LBVH generation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Hierarchy generation = " << milliseconds << " (ms)"
            << std::endl;
#endif

  cudaFree(scene_bb_ptr);
  return;
}

void bvh_cuda_forward(at::Tensor triangles, at::Tensor *collision_tensor_ptr,
                      int max_collisions = 16) {
  const auto batch_size = triangles.size(0);
  const auto num_triangles = triangles.size(1);

  thrust::device_vector<int> triangle_ids(num_triangles);

  int blockSize = NUM_THREADS;
  int gridSize = (num_triangles + blockSize - 1) / blockSize;

  thrust::device_vector<long2> collisionIndices(num_triangles * max_collisions);

#if PRINT_TIMINGS == 1
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  // int *counter;
  thrust::device_vector<int> collision_idx_cnt(batch_size);
  thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);

  // Construct the bvh tree
  AT_DISPATCH_FLOATING_TYPES(
      triangles.type(), "bvh_tree_building", ([&] {
        thrust::device_vector<BVHNode<scalar_t>> leaf_nodes(num_triangles);
        thrust::device_vector<BVHNode<scalar_t>> internal_nodes(num_triangles -
                                                                1);
        auto triangle_float_ptr = triangles.data<scalar_t>();

        for (int bidx = 0; bidx < batch_size; ++bidx) {

          Triangle<scalar_t> *triangles_ptr =
              (TrianglePtr<scalar_t>)triangle_float_ptr +
              num_triangles * bidx;

          thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                       make_long2(-1, -1));

#if DEBUG_PRINT == 1
          std::cout << "Start building BVH" << std::endl;
#endif
          buildBVH<scalar_t>(internal_nodes.data().get(),
                             leaf_nodes.data().get(), triangles_ptr,
                             &triangle_ids, num_triangles, batch_size);
#if DEBUG_PRINT == 1
          std::cout << "Successfully built BVH" << std::endl;
#endif

#if DEBUG_PRINT == 1
          std::cout << "Launching collision detection ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
          cudaEventRecord(start);
#endif
          // std::cout << tmp[0].right->bbox << std::endl;

          findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
              collisionIndices.data().get(),
              internal_nodes.data().get(),
              leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
              max_collisions, &collision_idx_cnt.data().get()[bidx]);
          cudaDeviceSynchronize();

#if PRINT_TIMINGS == 1
          cudaEventRecord(stop);
#endif
          cudaCheckError();
#if DEBUG_PRINT == 1
          std::cout << "AABB Collision detection finished ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
          cudaEventSynchronize(stop);
          float milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          std::cout << "FindPotentialCollisions = " << milliseconds << " (ms)"
                    << std::endl;
#endif

      // Calculate the number of potential collisions
#if DEBUG_PRINT == 1
          std::cout << "Starting stream compaction to keep only valid"
                    << " potential collisions" << std::endl;
#endif

#if PRINT_TIMINGS == 1
          cudaEventRecord(start);
#endif
          int num_cand_collisions =
              thrust::reduce(thrust::make_transform_iterator(
                                 collisionIndices.begin(), is_valid_cnt()),
                             thrust::make_transform_iterator(
                                 collisionIndices.end(), is_valid_cnt()));
#if PRINT_TIMINGS == 1
          cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
          std::cout << "Bounding box collisions detected = "
                    << num_cand_collisions << std::endl;
#endif

#if PRINT_TIMINGS == 1
          cudaEventSynchronize(stop);
          milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          std::cout << "Count AABB collisions elapsed time = " << milliseconds
                    << " (ms)" << std::endl;
#endif
          if (num_cand_collisions > 0) {

#if PRINT_TIMINGS == 1
            cudaEventRecord(start);
#endif
            // Keep only the pairs of ids where a bounding box to bounding box
            // collision was detected.
            thrust::device_vector<long2> collisions(num_cand_collisions,
                                                    make_long2(-1, -1));
            thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                            collisions.begin(), is_valid_cnt());

            cudaCheckError();
#if PRINT_TIMINGS == 1
            cudaEventRecord(stop);
#endif
#if PRINT_TIMINGS == 1
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Stream compaction for AABB collisions copy elapsed"
                      << " time = " << milliseconds << " (ms)" << std::endl;
#endif

#if DEBUG_PRINT == 1
            std::cout << "Finished with stream compaction ..." << std::endl;
#endif

#if DEBUG_PRINT == 1
            std::cout << "Check for triangle to triangle intersection ..."
                      << std::endl;
#endif

#if PRINT_TIMINGS == 1
            cudaEventRecord(start);
#endif
            int tri_grid_size =
                (collisions.size() + blockSize - 1) / blockSize;
            checkTriangleIntersections<scalar_t><<<tri_grid_size, blockSize>>>(
                collisions.data().get(), triangles_ptr, collisions.size(),
                num_triangles);
#if PRINT_TIMINGS == 1
            cudaEventRecord(stop);
#endif
            cudaCheckError();

#if DEBUG_PRINT == 1
            std::cout << "Finished triangle to triangle intersection ..."
                      << std::endl;
#endif

#if PRINT_TIMINGS == 1
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Triangle-to-Triangle intersection tests elapsed"
                      << " time = " << milliseconds << " (ms)" << std::endl;
#endif

#if PRINT_TIMINGS == 1
            cudaEventRecord(start);
#endif
            long *dev_ptr = collision_tensor_ptr->data<long>();
            cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 2,
                       (long *)collisions.data().get(),
                       2 * collisions.size() * sizeof(long),
                       cudaMemcpyDeviceToDevice);
            cudaCheckError();

#if PRINT_TIMINGS == 1
            cudaEventRecord(stop);
#endif

#if PRINT_TIMINGS == 1
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Copy CUDA array to tensor " << milliseconds << " (ms)"
                      << std::endl;
#endif
          }
        }
      }));

}
