// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

#ifndef __CUDA_KDTREE_H__
#define __CUDA_KDTREE_H__
#include "KDtree.h"
#include <vector>

struct CUDA_KDNode
{
    int level;
    int parent, left, right;
    float split_value;
    int num_indexes;
    int indexes;
};

using namespace std;

class CUDA_KDTree
{
public:
    ~CUDA_KDTree();
    void CreateKDTree(KDNode *root, int num_nodes, const vector <Point> &data);
    int GetNumPoints(){ return m_num_points;}

public:
    CUDA_KDNode *m_gpu_nodes;
    int *m_gpu_indexes;
    Point *m_gpu_points;

    int m_num_points;
};

void CheckCUDAError(const char *msg);

#endif
