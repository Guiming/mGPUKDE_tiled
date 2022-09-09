// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

// Revised based on http://nghiaho.com/uploads/code/CUDA_KDtree.zip

#include "KDtree.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <float.h>
#include <cmath>


// Eww global. Need this for the sort function
// A pointer to the class itself, allows the sort function to determine the splitting axis to sort by
static KDtree *myself = NULL;

KDtree::KDtree()
{
    myself = this;
    m_id = 0;
}

KDtree::~KDtree()
{
    // Delete all the ndoes
    vector <KDNode*> to_visit;

    to_visit.push_back(m_root);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            KDNode *cur = to_visit.back();
            to_visit.pop_back();

            if(cur->left)
                next_search.push_back(cur->left);

            if(cur->right)
                next_search.push_back(cur->right);

            delete cur;
        }

        to_visit = next_search;
    }

    m_root = NULL;
}

void KDtree::Create(vector <Point> &pts, int max_levels)
{
    m_pts = &pts;
    m_levels = max_levels;

    m_root = new KDNode();
    m_root->id = m_id++;
    m_root->level = 0;
    m_root->indexes.resize(pts.size());

    for(unsigned int i=0; i < pts.size(); i++) {
        m_root->indexes[i] = i;
    }

    vector <KDNode*> to_visit;
    to_visit.push_back(m_root);
    int counter = 0;
    while(to_visit.size()) {
        
        vector <KDNode*> next_search;
        int counter2 = 0;
        while(to_visit.size()) {
            
            //printf("***still building KDTree (CPU) - %d -> %d\n", counter, counter2);

            KDNode *node = to_visit.back();
            to_visit.pop_back();

            if(node->level < max_levels) {
                if(node->indexes.size() > 1) {
                    KDNode *left = new KDNode();
                    KDNode *right = new KDNode();

                    left->id = m_id++;
                    right->id = m_id++;

                    Split(node, left, right);

                    // Clear current indexes
                    {
                        vector <int> dummy;
                        node->indexes.swap(dummy);
                    }

                    node->left = left;
                    node->right = right;

                    node->_left = left->id;
                    node->_right = right->id;

                    if(left->indexes.size())
                        next_search.push_back(left);

                    if(right->indexes.size())
                        next_search.push_back(right);
                }
            }
            counter2 += 1;
        }

        to_visit = next_search;
        counter += 1;
    }
}

bool KDtree::SortPoints(const int a, const int b)
{
    vector <Point> &pts = *myself->m_pts;

    return pts[a].coords[myself->m_current_axis] < pts[b].coords[myself->m_current_axis];
}

void KDtree::Split(KDNode *cur, KDNode *left, KDNode *right)
{
    // Assume left/right nodes are created already

    vector <Point> &pts = *m_pts;
    m_current_axis = cur->level % KDTREE_DIM;;

    sort(cur->indexes.begin(), cur->indexes.end(), KDtree::SortPoints);

    int mid = cur->indexes[cur->indexes.size() / 2];
    cur->split_value = pts[mid].coords[m_current_axis];

    left->parent = cur;
    right->parent = cur;

    left->level = cur->level+1;
    right->level = cur->level+1;

    left->_parent = cur->id;
    right->_parent = cur->id;

    for(unsigned int i=0; i < cur->indexes.size(); i++) {
        int idx = cur->indexes[i];

        if(pts[idx].coords[m_current_axis] < cur->split_value)
            left->indexes.push_back(idx);
        else
            right->indexes.push_back(idx);
    }
}
//*
void KDtree::SearchAtNode(KDNode *cur, const Point &query, int *ret_index, double *ret_dist, KDNode **ret_node)
{
    int best_idx = 0;
    double best_dist = DBL_MAX;
    vector <Point> &pts = *m_pts;

   // First pass
    while(true) {
        int split_axis = cur->level % KDTREE_DIM;

        m_cmps++;

        if(cur->left == NULL) {
            *ret_node = cur;

            for(unsigned int i=0; i < cur->indexes.size(); i++) {
                m_cmps++;

                int idx = cur->indexes[i];

                double dist = Distance(query, pts[idx]);

                if(dist < best_dist && dist > 0.0) {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            break;
        }
        else if(query.coords[split_axis] < cur->split_value) {
            cur = cur->left;
        }
        else {
            cur = cur->right;
        }
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}

void KDtree::SearchAtNodeKNN(KDNode* cur, const Point& query, int k, vector<int> &ret_index, vector<double> &ret_dist, KDNode** ret_node)
{
    vector <Point>& pts = *m_pts;

    // First pass
    while (true) {
        int split_axis = cur->level % KDTREE_DIM;

        m_cmps++;

        if (cur->left == NULL) {
            *ret_node = cur;

            for (unsigned int i = 0; i < cur->indexes.size(); i++) {
                m_cmps++;

                int idx = cur->indexes[i];

                double dist = Distance(query, pts[idx]);
                //printf("distance computed...\n");
                if (dist == 0.0) continue; // skip self

                if (ret_index.size() < k) {
                    ret_dist.push_back(dist);
                    ret_index.push_back(idx);
                    //printf("\nret_index.size() in SearchAtNodeKNN = %d\n", ret_index.size());
                }
                else {
                    int maxIdx = max_element(ret_dist.begin(), ret_dist.end()) - ret_dist.begin();
                    if (dist < ret_dist[maxIdx]) {
                        ret_dist[maxIdx] = dist;
                        ret_index[maxIdx] = idx;
                    }
                    //printf("ret_index.size() >= k in SearchAtNodeKNN\n");
                }
            }

            break;
        }
        else if (query.coords[split_axis] < cur->split_value) {
            cur = cur->left;
        }
        else {
            cur = cur->right;
        }
    }

}

void KDtree::SearchAtNodeRangeKNN(KDNode* cur, const Point& query, int k, double& range, vector<int> &ret_index, vector<double> &ret_dist)
{

    vector <Point>& pts = *m_pts;
    vector <KDNode*> to_visit;

    to_visit.push_back(cur);

    while (to_visit.size()) {
        vector <KDNode*> next_search;

        while (to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if (cur->left == NULL) {
                for (unsigned int i = 0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    int idx = cur->indexes[i];
                    double dist = Distance(query, pts[idx]);
                    //printf("distance computed...\n");
                    
                    if (dist == 0.0) continue; // skip self

                    if (ret_index.size() < k) {
                        ret_dist.push_back(dist);
                        ret_index.push_back(idx);
                    }
                    else {
                        
                        int maxIdx = max_element(ret_dist.begin(), ret_dist.end()) - ret_dist.begin();
                        
                        if (dist < ret_dist[maxIdx]) {
                            ret_dist[maxIdx] = dist;
                            ret_index[maxIdx] = idx;
                        }

                        range = sqrt(*max_element(ret_dist.begin(), ret_dist.end()));
                    }
                }
            }
            else {
                double d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                m_cmps++;

                if (fabs(d) > range) {
                    if (d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
        }

        to_visit = next_search;
    }
}

// search for K Nearest Neighbors - return index and distances
void KDtree::SearchKNN(const Point& query, int k, vector<int> &ret_index, vector<double> &ret_dist)
{
    // Find the first closest node, this will be the upper bound for the next searches
    vector <Point>& pts = *m_pts;
    KDNode* best_node = NULL;
    double radius = DBL_MAX;
    m_cmps = 0;

    SearchAtNodeKNN(m_root, query, k, ret_index, ret_dist, &best_node);

    if (ret_index.size() == k) {
        radius = sqrt(*max_element(ret_dist.begin(), ret_dist.end()));
    }

    //printf("radius %f\n", radius);

    // Now find other possible candidates
    KDNode* cur = best_node;

    while (cur->parent != NULL) {
        // Go up
        KDNode* parent = cur->parent;
        int split_axis = (parent->level) % KDTREE_DIM;

        // Search the other node
        //KDNode* tmp_node;
        //KDNode* search_node = NULL;

        if (fabs(parent->split_value - query.coords[split_axis]) <= radius) {
            // Search opposite node
            if (parent->left != cur)
                SearchAtNodeRangeKNN(parent->left, query, k, radius, ret_index, ret_dist);
            else
                SearchAtNodeRangeKNN(parent->right, query, k, radius, ret_index, ret_dist);
        }
        cur = parent;
    }
}

void KDtree::SearchAtNodeRange(KDNode *cur, const Point &query, double range, int *ret_index, double *ret_dist)
{
    int best_idx = 0;
    double best_dist = DBL_MAX;
    vector <Point> &pts = *m_pts;
    vector <KDNode*> to_visit;

    to_visit.push_back(cur);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if(cur->left == NULL) {
                for(unsigned int i=0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    int idx = cur->indexes[i];
                    double d = Distance(query, pts[idx]);

                    if(d < best_dist && d > 0.0) {
                        best_dist = d;
                        best_idx = idx;
                    }
                }
            }
            else {
                double d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                m_cmps++;

                if(fabs(d) > range) {
                    if(d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
        }

        to_visit = next_search;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}

void KDtree::Search(const Point &query, int *ret_index, double *ret_dist)
{
    // Find the first closest node, this will be the upper bound for the next searches
    vector <Point> &pts = *m_pts;
    KDNode *best_node = NULL;
    int best_idx = 0;
    double best_dist = DBL_MAX;
    double radius = 0;
    m_cmps = 0;

    SearchAtNode(m_root, query, &best_idx, &best_dist, &best_node);

    radius = sqrt(best_dist);

    //printf("radius %f\n", radius);

    // Now find other possible candidates
    KDNode *cur = best_node;

    while(cur->parent != NULL) {
        // Go up
        KDNode *parent = cur->parent;
        int split_axis = (parent->level) % KDTREE_DIM;

        // Search the other node
        int tmp_idx;
        double tmp_dist = DBL_MAX;
        //KDNode *tmp_node;
        //KDNode *search_node = NULL;

        if(fabs(parent->split_value - query.coords[split_axis]) <= radius) {
            // Search opposite node
            if(parent->left != cur)
                SearchAtNodeRange(parent->left, query, radius, &tmp_idx, &tmp_dist);
            else
                SearchAtNodeRange(parent->right, query, radius, &tmp_idx, &tmp_dist);
        }

        if(tmp_dist < best_dist) {
            best_dist = tmp_dist;
            best_idx = tmp_idx;
        }

        cur = parent;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}
//*/
// Added by Guiming
/*Search for points within distance (squared) range*/
void KDtree::SearchRange(const Point &query, double range, vector<int> &ret_index, vector<double> &ret_sq_dist)
{

    //printf("CPU: x=%f y=%f z=%f\n", query.coords[0], query.coords[1], query.coords[2]);
    //printf("== %f\n", sqrt(range));
    //SearchAtNodeRange(m_root, query, range, ret_index, ret_sq_dist);
    vector <Point> &pts = *m_pts;
    vector <KDNode*> to_visit;
    KDNode *cur = m_root;
    m_cmps = 0;
    to_visit.push_back(cur);

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            cur = to_visit.back();
            to_visit.pop_back();

            int split_axis = cur->level % KDTREE_DIM;

            if(cur->left == NULL) {
                //printf("cur->indexes.size(): %d\n", cur->indexes.size());
                  //int *indexes = &(cur->indexes[0]);
                for(unsigned int i=0; i < cur->indexes.size(); i++) {
                    m_cmps++;

                    //int idx = indexes[i];
                    int idx = cur->indexes[i];

                    double d = Distance(query, pts[idx]);

                    if(d < range) {
                        ret_sq_dist.push_back(d);
                        ret_index.push_back(idx);
                        //printf("CPU: %d %f\n", idx, d);
                    }
                }
            }
            else {
                double d = query.coords[split_axis] - cur->split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                //m_cmps++;

                if(fabs(d*d) > range) {
                    if(d < 0)
                        next_search.push_back(cur->left);
                    else
                        next_search.push_back(cur->right);
                }
                else {
                    next_search.push_back(cur->left);
                    next_search.push_back(cur->right);
                }
            }
            //to_visit = next_search;
        }

        to_visit = next_search;
    }
    //printf("%d comparisons\n", m_cmps)  ;
}


