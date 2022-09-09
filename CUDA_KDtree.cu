// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

#include "CUDA_KDtree.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

CUDA_KDTree::~CUDA_KDTree()
{
    cudaFree(m_gpu_nodes);
    cudaFree(m_gpu_indexes);
    cudaFree(m_gpu_points);
}

void CUDA_KDTree::CreateKDTree(KDNode *root, int num_nodes, const vector <Point> &data)
{
    // Create the nodes again on the CPU, laid out nicely for the GPU transfer
    // Not exactly memory efficient, since we're creating the entire tree again
    m_num_points = data.size();

    for (int i = 0; i < 2; i++)
    {
        cudaMalloc((void**)&m_gpu_nodes, sizeof(CUDA_KDNode) * num_nodes);
        cudaMalloc((void**)&m_gpu_indexes, sizeof(int) * m_num_points);
        cudaMalloc((void**)&m_gpu_points, sizeof(Point) * m_num_points);
    }
    CheckCUDAError("CreateKDTree");

    vector <CUDA_KDNode> cpu_nodes(num_nodes);
    vector <int> indexes(m_num_points);
    vector <KDNode*> to_visit;

    int cur_pos = 0;

    to_visit.push_back(root);
    int counter = 0;
    while(to_visit.size()) {
        
        

        vector <KDNode*> next_search;
        
        int counter2 = 0;

        while(to_visit.size()) {

            //printf("***still building KDTree (GPU) - %d -> %d\n", counter, counter2);

            KDNode *cur = to_visit.back();
            to_visit.pop_back();

            int id = cur->id;

            cpu_nodes[id].level = cur->level;
            cpu_nodes[id].parent = cur->_parent;
            cpu_nodes[id].left = cur->_left;
            cpu_nodes[id].right = cur->_right;
            cpu_nodes[id].split_value = cur->split_value;
            cpu_nodes[id].num_indexes = cur->indexes.size();

            if(cur->indexes.size()) {
                for(unsigned int i=0; i < cur->indexes.size(); i++)
                    indexes[cur_pos+i] = cur->indexes[i];

                cpu_nodes[id].indexes = cur_pos;
                cur_pos += cur->indexes.size();
            }
            else {
                cpu_nodes[id].indexes = -1;
            }

            if(cur->left)
                next_search.push_back(cur->left);

            if(cur->right)
                next_search.push_back(cur->right);
            
            counter2 += 1;
        }

        to_visit = next_search;

        counter += 1;
    }

    cudaMemcpy(m_gpu_nodes, &cpu_nodes[0], sizeof(CUDA_KDNode)*cpu_nodes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_indexes, &indexes[0], sizeof(int)*indexes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_points, &data[0], sizeof(Point)*data.size(), cudaMemcpyHostToDevice);

    CheckCUDAError("CreateKDTree");
}

