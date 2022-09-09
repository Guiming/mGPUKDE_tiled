// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

// Revised based on http://nghiaho.com/uploads/code/CUDA_KDtree.zip

#ifndef __KDTREE_H__
#define __KDTREE_H__
#include <cstddef>
#include <vector>

using namespace std;

#define KDTREE_DIM 2 // data dimensions

struct Point
{
    float coords[KDTREE_DIM];
};

class KDNode
{
public:
    KDNode()
    {
        parent = NULL;
        left = NULL;
        right = NULL;
        split_value = -1;
        _parent = -1;
        _left = -1;
        _right = -1;
    }

    int id; // for GPU
    int level;
    KDNode *parent, *left, *right;
    int _parent, _left, _right; // for GPU
    float split_value;
    vector <int> indexes; // index to points
};

class KDtree
{
public:
    KDtree();
    ~KDtree();
    void Create(vector <Point> &pts, int max_levels = 99 /* You can limit the search depth if you want */);
    void Search(const Point &query, int *ret_index, double *ret_sq_dist); // nearest neighbor search
    
    void SearchKNN(const Point& query, int k, vector<int> &ret_index, vector<double> &ret_sq_dist); // nearest neighbor search

    void SearchRange(const Point &query, double range, vector<int> &ret_index, vector<double> &ret_sq_dist); // range search
    //void SearchRangeBruteForce(const Point &query, float range, vector<int> &ret_index, vector<float> &ret_sq_dist); // brute force range search
    int GetNumNodes() const { return m_id; }
    KDNode* GetRoot() const { return m_root; }

    static bool SortPoints(const int a, const int b);

private:
    vector <Point> *m_pts;
    KDNode *m_root;
    int m_current_axis;
    int m_levels;
    int m_cmps; // count how many comparisons were made in the tree for a query
    int m_id; // current node ID

    void Split(KDNode *cur, KDNode *left, KDNode *right);
    void SearchAtNode(KDNode *cur, const Point &query, int *ret_index, double *ret_dist, KDNode **ret_node);
    void SearchAtNodeRange(KDNode* cur, const Point& query, double range, int* ret_index, double* ret_dist);

    void SearchAtNodeKNN(KDNode* cur, const Point& query, int k, vector<int>& ret_index, vector<double>& ret_dist, KDNode** ret_node);
    void SearchAtNodeRangeKNN(KDNode* cur, const Point& query, int k, double& range, vector<int>& ret_index, vector<double>& ret_dist);
    
    inline double Distance(const Point &a, const Point &b) const;
};

double KDtree::Distance(const Point &a, const Point &b) const
{
    double deltaX = a.coords[0] - b.coords[0];
    double deltaY = a.coords[1] - b.coords[1];
    return deltaX * deltaX + deltaY * deltaY;
}

#endif
