// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

/*
* Kernel function
*/

#ifndef _KDE_KERNEL_H_
#define _KDE_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>
#include <device_functions.h>

#include "SamplePoints.h"
#include "AsciiRaster.h"
#include "Utilities.h"

#include "KDtree.h"
#include "CUDA_KDtree.h"

__device__ int STACK_DEPTH_MAX = 0;

__device__ float dReductionSum = 0.0f; // sum of log of densities /// this gets updated only on DEVICE 0 who does the reduction
//__device__ float dDen0_0 = 1.0f; // sum of log of densities

__device__ double Distance(const Point &a, const Point &b)
{
    double deltaX = a.coords[0] - b.coords[0];
    double deltaY = a.coords[1] - b.coords[1];
    return deltaX * deltaX + deltaY * deltaY;
}

__device__ void dSearchRange(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query, const double range, int &ret_num_nbrs, int *ret_indexes, float *ret_dists)
{
    //printf("begin dSearchRange!\n");
    // Goes through all the nodes that are within "range"
    int cur = 0; // root
    int num_nbrs = 0;

    // Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
    // We'll use a fixed length stack, increase this as required
    int to_visit[CUDA_STACK];
    int to_visit_pos = 0;

    to_visit[to_visit_pos++] = cur;

    while(to_visit_pos) {
        int next_search[CUDA_STACK];
        int next_search_pos = 0;

        while(to_visit_pos) {
            cur = to_visit[to_visit_pos-1];
            to_visit_pos--;

            int split_axis = nodes[cur].level % KDTREE_DIM;

            if(nodes[cur].left == -1) {
                for(int i=0; i < nodes[cur].num_indexes; i++) {
                    int idx = indexes[nodes[cur].indexes + i];
                    double d = Distance(query, pts[idx]);

                    if(d < range) {
                        ret_indexes[num_nbrs] = idx;
                        ret_dists[num_nbrs] = d;
                        num_nbrs++;
                        //printf("find nbr in dSearchRange!\n");
                    }
                }
            }
            else {
                double d = query.coords[split_axis] - nodes[cur].split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                if(fabs(d*d) > range) {
                    if(d < 0)
                        next_search[next_search_pos++] = nodes[cur].left;
                    else
                        next_search[next_search_pos++] = nodes[cur].right;
                }
                else {
                    next_search[next_search_pos++] = nodes[cur].left;
                    next_search[next_search_pos++] = nodes[cur].right;
                }
            }
        }

        // No memcpy available??
        for(int i=0; i  < next_search_pos; i++)
            to_visit[i] = next_search[i];

        to_visit_pos = next_search_pos;
    }
    //printf("A:ret_num_nbrs=%d\n", num_nbrs);
    ret_num_nbrs = num_nbrs;
    //printf("end dSearchRange! %d nbrs found!\n", ret_num_nbrs);
}

// squared distance btw two points
__device__  double dDistance2(float x0, float y0, float x1, float y1){
	double dx = x1 - x0;
	double dy = y1 - y0;
	return dx*dx + dy*dy;
}

// Gaussian kernel
__device__ float dGaussianKernel(double h2, double d2){
	return expf(d2 / (-2.0f * h2)) / (h2 * TWO_PI);
}

// Edge correction with fixed bandwidth h2 (squared)
__global__ void CalcEdgeCorrectionWeights(double h2, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	//Add offset for multiple GPUs
	
	tid += dPoints.start;
	
	// directly return if ID goes out of range
	if(tid >= dPoints.end){
		return;
	}

	// By Guiming @ 2016-09-01
	if(dPoints.distances[tid] >= CUT_OFF_FACTOR * h2){
		dWeights[tid] = 1.0f;
		return;
	}

	// otherwise calculate edge effect correction weight point ID = tid
	float cellSize = dAscii.cellSize;
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;

	//printf("%d %d\n", nCols, nRows);

	float cellArea = cellSize * cellSize;

	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float ew = 0.0f;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);

	float cell_x, cell_y, val;
	double d2;//, g;
	double h = sqrt(h2);
	//int row, col;

	// added by Guiming @2016-09-11
	// narrow down the row/col range
	int row_lower = 0;
	int row_upper = nRows - 1;
	int col_lower = 0;
	int col_upper = nCols - 1;
	if(NARROW){
		int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * h, nRows, yLLCorner, cellSize);
		row_lower = MAX(0, r);
		row_upper = MIN(nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * h, nRows, yLLCorner, cellSize));
		col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * h, xLLCorner, cellSize));
		col_upper = MIN(nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * h, xLLCorner, cellSize));
	}

	//printf("[%d %d], [%d %d]", row_lower, row_upper, col_lower, col_upper);

	for (int row = row_lower; row <= row_upper; row++){
		for (int col = col_lower; col <= col_upper; col++){
			val = dAscii.elements[row*nCols+col];
			if (val != noDataValue){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				d2 = dDistance2(p_x, p_y, cell_x, cell_y);
				if(d2 < CUT_OFF_FACTOR * h2){
					ew += dGaussianKernel(h2, d2) * cellArea;
				}
			}
		}
	}
	dWeights[tid] = 1.0f / ew;
}

// Edge correction with adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void CalcEdgeCorrectionWeights(float* dHs, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if(tid >= dPoints.end){
		return;
	}

	float h = dHs[tid];
	double h2 = (double)h * h;

	// By Guiming @ 2016-09-01
	if(dPoints.distances[tid] >= CUT_OFF_FACTOR * h2){
		dWeights[tid] = 1.0f;
		return;
	}

	// otherwise calculate edge effect correction weight point ID = tid
	float cellSize = dAscii.cellSize;
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue; 

	//printf("%d %d\n", nCols, nRows);

	float cellArea = cellSize * cellSize;

	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float ew = 0.0f;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);

	float cell_x, cell_y, val;
	double d2;//, g;
	//int row, col;

	// added by Guiming @2016-09-11
	// narrow down the row/col range
	int row_lower = 0;
	int row_upper = nRows - 1;
	int col_lower = 0;
	int col_upper = nCols - 1;
	if(NARROW){
		int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * h, nRows, yLLCorner, cellSize);
		row_lower = MAX(0, r);
		row_upper = MIN(nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * h, nRows, yLLCorner, cellSize));
		col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * h, xLLCorner, cellSize));
		col_upper = MIN(nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * h, xLLCorner, cellSize));
	}

	for (int row = row_lower; row <= row_upper; row++){
		for (int col = col_lower; col <= col_upper; col++){
			val = dAscii.elements[row*nCols+col];
			if (val != noDataValue){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				if(d2 < CUT_OFF_FACTOR * h2){
					ew += dGaussianKernel(h2, d2) * cellArea;
				}
			}
		}
	}
	dWeights[tid] = 1.0f / ew;
}

// Guiming 2021-08-30 //never used
__global__ void InitCellDensities(const AsciiRaster dAscii)
{	
	size_t n;
	if (dAscii.compute_serialized) {
		n = dAscii.nVals;
	}
	else {
		n = dAscii.nRows * dAscii.nCols;
	}
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	if (dAscii.compute_serialized) {
		dAscii.elementsVals[tid] = 0.0f;
	}
	else {
		float val = dAscii.elements[tid];
		if (val == dAscii.noDataValue) {
			return;
		}
		dAscii.elements[tid] = 0.0f;
	}
}

// Kernel density estimation with fixed bandwidth h2 (squared)
__global__ void KernelDesityEstimation_pPoints(double h2, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

	// # of rows and cols
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;
	
	float p_x, p_y, p_w;    // x, y coord, weight of point
	float e_w = 1.0;    // edge effect correction weight
	
	double d2;
	int p_col, p_row;
	float cell_x, cell_y; // x,y coord of cell
	float den = 0.0f;

	p_x = dPoints.xCoordinates[tid];
	p_y = dPoints.yCoordinates[tid];
	p_w = dPoints.weights[tid];
	e_w = dWeights[tid];

	p_row = YCOORD_TO_ROW(p_y, nRows, yLLCorner, cellSize);
	p_col = XCOORD_TO_COL(p_x, xLLCorner, cellSize);
	int deltarc = ceil(sqrt(CUT_OFF_FACTOR * h2) / cellSize) + 1;

	float g, tmp;

	if (dAscii.compute_serialized) {
		for (size_t i = 0; i < dAscii.nVals; i++) {

			int row = dAscii.rowcolIdx[i] / nCols;
			int col = dAscii.rowcolIdx[i] % nCols;

			if ((row >= p_row - deltarc && row < p_row + deltarc) && (col >= p_col - deltarc && col < p_col + deltarc)) {

				//printf("(%d, %d), (%d, %d), %d\n", p_row, p_col, row, col, deltarc);

				// x, y coord of this cell
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				if (d2 < CUT_OFF_FACTOR * h2) {
					g = dGaussianKernel(h2, d2) * p_w * e_w;
					tmp = atomicAdd(&dAscii.elementsVals[i], g);
				}
			}
		}
	}
	else {
		for (int row = p_row - deltarc; row < p_row + deltarc; row++) {
			for (int col = p_col - deltarc; col < p_col + deltarc; col++) {
				//printf("row * nCols + col = %llu\n", row * nCols + col);
				// should do KDE on this cell?
				float val = dAscii.elements[row * nCols + col];
				if (val == noDataValue) {
					continue;
				}

				// x, y coord of this cell
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				if (d2 < CUT_OFF_FACTOR * h2) {
					g = dGaussianKernel(h2, d2) * p_w * e_w;
					tmp = atomicAdd(&dAscii.elements[row * nCols + col], g);

				}
			}
		}
	}
}

// Kernel density estimation with fixed bandwidth h2 (squared)
__global__ void KernelDesityEstimation_pRaster(double h2, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dAscii.start;
	// directly return if ID goes out of range
	if (tid >= dAscii.end) {
		return;
	}

	// # of rows and cols
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;
	float cell_x, cell_y; // x,y coord of cell
	float p_x, p_y, p_w;    // x, y coord, weight of point
	int numPoints = dPoints.numberOfPoints;
	double d2;
	float e_w = 1.0;    // edge effect correction weight
	float den;
	size_t col, row;

	// which row, col?
	if (dAscii.compute_serialized) {
		row = dAscii.rowcolIdx[tid] / nCols;
		col = dAscii.rowcolIdx[tid] % nCols;
	}
	else {
		
		// should do KDE on this cell?
		float val = dAscii.elements[tid];

		if (val == noDataValue) {
			return;
		}

		row = tid / nCols;
		col = tid - row * nCols;
	}

	// x, y coord of this cell
	cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
	cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
	
	den = 0.0f;
	for (int p = 0; p < numPoints; p++) {
		p_x = dPoints.xCoordinates[p];
		p_y = dPoints.yCoordinates[p];
		p_w = dPoints.weights[p];
		e_w = dWeights[p];
		d2 = dDistance2(p_x, p_y, cell_x, cell_y);

		if (d2 < CUT_OFF_FACTOR * h2) {
			den += dGaussianKernel(h2, d2) * p_w * e_w;
		}
	}

	if (dAscii.compute_serialized) {
		dAscii.elementsVals[tid] = den; // intensity, not probability
	}
	else{
		dAscii.elements[tid] = den; // intensity, not probability
	}
}

// Kernel density estimation with adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void KernelDesityEstimation_pPoints(float* dHs, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}
	
	//printf("***KDE start for Point %d\n", tid);

	// # of rows and cols
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;

	float p_x, p_y, p_w; 
	double h2;    // x, y coord, weight of point
	float e_w = 1.0;    // edge effect correction weight

	double d2;
	int p_col, p_row;
	float cell_x, cell_y; // x,y coord of cell

	p_x = dPoints.xCoordinates[tid];
	p_y = dPoints.yCoordinates[tid];
	p_w = dPoints.weights[tid];
	e_w = dWeights[tid];
	float h = dHs[tid];
	h2 = (double)h * h;

	p_row = YCOORD_TO_ROW(p_y, nRows, yLLCorner, cellSize);
	p_col = XCOORD_TO_COL(p_x, xLLCorner, cellSize);

	//printf("point %d, p_row %d, p_col %d \n", tid, p_row, p_col);

	int deltarc = ceil(sqrt(CUT_OFF_FACTOR * h2) / cellSize) + 1;

	if (max(0, p_row - deltarc) >= min(p_row + deltarc, (int)nRows)) {
		return;
	}
	if (max(0, p_col - deltarc) >= min(p_col + deltarc, (int)nCols)) {
		return;
	}

	//if (p_row < 0 || p_col < 0 || p_row >= nRows || p_col >= nCols) {

	//	printf("p_row = %d, p_col = %d, deltarc = %d\n", p_row, p_col, deltarc);
	//	printf("%d, %d, %d, %d\n", max(0, p_row - deltarc), min(p_row + deltarc, (int)nRows), max(0, p_col - deltarc), min(p_col + deltarc, (int)nCols));
	//}
	float g, tmp;

	if (dAscii.compute_serialized) {
		for (size_t i = 0; i < dAscii.nVals; i++) {

			int row = dAscii.rowcolIdx[i] / nCols;
			int col = dAscii.rowcolIdx[i] % nCols;

			if ((row >= p_row - deltarc && row < p_row + deltarc) && (col >= p_col - deltarc && col < p_col + deltarc)) {
				//printf("(%d, %d), (%d, %d), %d\n", p_row, p_col, row, col, deltarc);
				// x, y coord of this cell
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				if (d2 < CUT_OFF_FACTOR * h2) {
					g = dGaussianKernel(h2, d2) * p_w * e_w;
					tmp = atomicAdd(&dAscii.elementsVals[i], g);

				}
			}
		}
	}
	else {
		
		for (int row = max(0, p_row - deltarc); row < min(p_row + deltarc, (int)nRows); row++) {
			for (int col = max(0, p_col - deltarc); col < min(p_col + deltarc, (int)nCols); col++) {
				//for (int row = 0; row < nRows; row++) {
				//	for (int col = 0; col < nCols; col++) {
						// should do KDE on this cell?
				//printf("row = %d, col = %d, row * nCols + col = %ll\n", row, col, row * nCols + col);

				if (row * nCols + col >= nRows * nCols) printf("***ERROR***\n");

				float val = dAscii.elements[row * nCols + col];
				if (val == noDataValue) {
					continue;
				}

				// x, y coord of this cell
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				if (d2 < CUT_OFF_FACTOR * h2) {
					g = dGaussianKernel(h2, d2) * p_w * e_w;
					tmp = atomicAdd(&dAscii.elements[row * nCols + col], g);
				}
			}
		}
	}

	//printf("***KDE done for Point %d\n", tid);
}

// Kernel density estimation with adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void KernelDesityEstimation_pRaster(float* dHs, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dAscii.start;
	// directly return if ID goes out of range
	if (tid >= dAscii.end) {
		return;
	}

	// # of rows and cols
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;
	float cell_x, cell_y; // x,y coord of cell
	float p_x, p_y, p_w;    // x, y coord, weight of point
	int numPoints = dPoints.numberOfPoints;
	float h;
	double d2;
	float e_w = 1.0f;    // edge effect correction weight
	float den;
	size_t col, row;

	// which row, col?
	if (dAscii.compute_serialized) {
		row = dAscii.rowcolIdx[tid] / nCols;
		col = dAscii.rowcolIdx[tid] % nCols;
	}
	else {
		// should do KDE on this cell?
		float val = dAscii.elements[tid];

		if (val == noDataValue) {
			return;
		}

		row = tid / nCols;
		col = tid - row * nCols;
	}

	// x, y coord of this cell
	cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
	cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

	//printf("***KDE start for cell %d\n", tid);
	den = 0.0f;
	for (int p = 0; p < numPoints; p++){
		p_x = dPoints.xCoordinates[p];
		p_y = dPoints.yCoordinates[p];
		p_w = dPoints.weights[p];
		e_w = dWeights[p];
		h = dHs[p];
		d2 = dDistance2(p_x, p_y, cell_x, cell_y);

		if(d2 < CUT_OFF_FACTOR * h * h){
			den += dGaussianKernel(h * h, d2) * p_w *e_w;
		}

		//den += dGaussianKernel(h * h, d2) * p_w *e_w;
	}
	if (dAscii.compute_serialized) {
		dAscii.elementsVals[tid] = den;
	}
	else {
		dAscii.elements[tid] = den; // intensity, not probability
	}
	//printf("***KDE done for cell %d\n", tid);
}

// Guiming 2021-08-15
__global__ void InitGPUDen(float* gpuDen, const int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}
	gpuDen[tid] = 0.0f;
}


// Guiming 2021-08-30 //never used
__global__ void InitdWeights(float* dWeights, const int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}
	dWeights[tid] = 1.0f;
}

// KD tree approach
// Density at each point under fixed bandwidth h2 (squared)
///*
__global__ void DensityAtPointsKdtr(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, double h2, const SamplePoints dPoints, float *dWeights, float *gpuDen){

	//printf("gpu_kdtree->m_num_points %d\n", gpu_kdtree->m_num_points);
	//DEBUGGING THIS VERSION
  // serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}
	/*if (tid % 1000000 == 0) {
		printf("...starting DensityAtPointsKdtr on Point %d\n", tid);
		//printf("tid=%d blockIdx.y=%d gridDim.y=%d blockIdx.x=%d blockDim.x=%d threadIdx.x=%d\n", tid, blockIdx.y, gridDim.y, blockIdx.x, blockDim.x, threadIdx.x);
	}*/
	// now calculate density
	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];

  Point query;
  query.coords[0] = p_x;
  query.coords[1] = p_y;

  /*
  int n_NBRS;
  int gpu_ret_indexes[N_NBRS];
  float gpu_ret_dist[N_NBRS];

  float range = CUT_OFF_FACTOR * h2;
  dSearchRange(nodes, indexes, pts, query, range, n_NBRS, gpu_ret_indexes, gpu_ret_dist);

  int idx;
  float d2, g;
  float p_w = dPoints.weights[tid];
  float e_w = dWeights[tid];
  for(int i = 0; i < n_NBRS; i++){
      idx = gpu_ret_indexes[i];
      d2 = gpu_ret_dist[i];
      g = dGaussianKernel(h2, d2) * p_w *e_w;
      float tmp = atomicAdd(&gpuDen[idx], g);
  }
  */

  // call to dSearchRange requires too much memory for gpu_ret_indexes and gpu_ret_dist.
  // Embeding the code directly instead
  ////////////////////////////////////////////////////////////////////////////
  float g, tmp;
  float p_w = dPoints.weights[tid];
  float e_w = dWeights[tid];
  double range = CUT_OFF_FACTOR * h2;

  // Goes through all the nodes that are within "range"
  int cur = 0; // root
  //int num_nbrs = 0;

  // Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
  // We'll use a fixed length stack, increase this as required
  int to_visit[CUDA_STACK];
  int to_visit_pos = 0;

  to_visit[to_visit_pos++] = cur;

   while(to_visit_pos) {
      int next_search[CUDA_STACK];
      int next_search_pos = 0;

      while(to_visit_pos) {	  

          cur = to_visit[to_visit_pos-1];
          to_visit_pos--;

          int split_axis = nodes[cur].level % KDTREE_DIM;

          if(nodes[cur].left == -1) {
              for(int i=0; i < nodes[cur].num_indexes; i++) {
                  int idx = indexes[nodes[cur].indexes + i];
                  double d = Distance(query, pts[idx]);

                  if(d < range) {
                      //ret_indexes[num_nbrs] = idx;
                      //ret_dists[num_nbrs] = d;
                      //num_nbrs++;

                      g = dGaussianKernel(h2, d) * p_w *e_w;
                      tmp = atomicAdd(&gpuDen[idx], g);

                  }
              }
          }
          else {
              double d = query.coords[split_axis] - nodes[cur].split_value;

              // There are 3 possible scenarios
              // The hypercircle only intersects the left region
              // The hypercircle only intersects the right region
              // The hypercricle intersects both

              if(fabs(d*d) > range) {
                  if(d < 0)
                      next_search[next_search_pos++] = nodes[cur].left;
                  else
                      next_search[next_search_pos++] = nodes[cur].right;
              }
              else {
                  next_search[next_search_pos++] = nodes[cur].left;
                  next_search[next_search_pos++] = nodes[cur].right;
              }
          }
      }

	  STACK_DEPTH_MAX = max(STACK_DEPTH_MAX, next_search_pos);
	  
      // No memcpy available??
      for(int i=0; i  < next_search_pos; i++)
          to_visit[i] = next_search[i];

      to_visit_pos = next_search_pos;
  }
  ////////////////////////////////////////////////////////////////////////
   //if (tid % 1000000 == 0) printf("...ending DensityAtPointsKdtr on Point %d\n", tid);

}

__global__ void dCopyDensityValues(const SamplePoints dPoints, float *dWeights, const double h2, float *gpuDen, float* dDen0, float* dDen1){
    // serial point ID
    int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	
	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

    float p_w = dPoints.weights[tid];
    float e_w = dWeights[tid];
    if(dDen1 != NULL){
      float g = dGaussianKernel(h2, 0.0f) * p_w *e_w;
	  float _den1 = max(gpuDen[tid] - g, EPSILONDENSITY);
      dDen1[tid] = logf(_den1);
    }
    if(dDen0 != NULL){
      dDen0[tid] = logf(gpuDen[tid]);
    }

    // reset gpuDen to 0.0
    gpuDen[tid] = 0.0f;
}

// KD tree approach
// Density at each point under adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void DensityAtPointsKdtr(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, float* dHs, const SamplePoints dPoints, float *dWeights, float *gpuDen){
  // serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}
	/*
	if (tid % 1000000 == 0) {
		printf("...starting DensityAtPointsKdtr on Point %d\n", tid);
		//printf("tid=%d blockIdx.y=%d gridDim.y=%d blockIdx.x=%d blockDim.x=%d threadIdx.x=%d\n", tid, blockIdx.y, gridDim.y, blockIdx.x, blockDim.x, threadIdx.x);
	}
	*/
	// now calculate density
	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];

	  Point query;
	  query.coords[0] = p_x;
	  query.coords[1] = p_y;

	  /*
	  int n_NBRS;
	  int gpu_ret_indexes[N_NBRS];
	  float gpu_ret_dist[N_NBRS];

	  float h = dHs[tid];
	  float h2 = h * h;
	  float range = CUT_OFF_FACTOR * h2;
	  dSearchRange(nodes, indexes, pts, query, range, n_NBRS, gpu_ret_indexes, gpu_ret_dist);

	  int idx;
	  float d2, g;
	  float p_w = dPoints.weights[tid];
	  float e_w = dWeights[tid];
	  for(int i = 0; i < n_NBRS; i++){
		  idx = gpu_ret_indexes[i];
		  d2 = gpu_ret_dist[i];
		  g = dGaussianKernel(h2, d2) * p_w *e_w;
		  float tmp = atomicAdd(&gpuDen[idx], g);
	  }
	  */

	  // call to dSearchRange requires too much memory for gpu_ret_indexes and gpu_ret_dist.
	  // Embeding the code directly instead
	  ////////////////////////////////////////////////////////////////////////////
	  float h = dHs[tid];
	  double h2 = h * h;
	  double range = CUT_OFF_FACTOR * h2;

	  float g, tmp;
	  float p_w = dPoints.weights[tid];
	  float e_w = dWeights[tid];

	  // Goes through all the nodes that are within "range"
	  int cur = 0; // root

	  // Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
	  // We'll use a fixed length stack, increase this as required
	  int to_visit[CUDA_STACK];
	  int to_visit_pos = 0;

	  to_visit[to_visit_pos++] = cur;

	  while(to_visit_pos) {
		  int next_search[CUDA_STACK];
		  int next_search_pos = 0;

		  while(to_visit_pos) {
			  cur = to_visit[to_visit_pos-1];
			  to_visit_pos--;

			  int split_axis = nodes[cur].level % KDTREE_DIM;

			  if(nodes[cur].left == -1) {
				  for(int i=0; i < nodes[cur].num_indexes; i++) {
					  int idx = indexes[nodes[cur].indexes + i];
					  float d = Distance(query, pts[idx]);

					  if(d < range) {
						  g = dGaussianKernel(h2, d) * p_w *e_w;
						  tmp = atomicAdd(&gpuDen[idx], g);
					  }
				  }
			  }
			  else {
				  float d = query.coords[split_axis] - nodes[cur].split_value;

				  // There are 3 possible scenarios
				  // The hypercircle only intersects the left region
				  // The hypercircle only intersects the right region
				  // The hypercricle intersects both

				  if(fabs(d*d) > range) {
					  if(d < 0)
						  next_search[next_search_pos++] = nodes[cur].left;
					  else
						  next_search[next_search_pos++] = nodes[cur].right;
				  }
				  else {
					  next_search[next_search_pos++] = nodes[cur].left;
					  next_search[next_search_pos++] = nodes[cur].right;
				  }
			  }
		  }

		  STACK_DEPTH_MAX = max(STACK_DEPTH_MAX, next_search_pos);

		  // No memcpy available??
		  for(int i=0; i  < next_search_pos; i++)
			  to_visit[i] = next_search[i];

		  to_visit_pos = next_search_pos;
	  }
	  ////////////////////////////////////////////////////////////////////////
	  //if (tid % 1000000 == 0) printf("...ending DensityAtPointsKdtr on Point %d\n", tid);
}

__global__ void dCopyDensityValues(const SamplePoints dPoints, float *dWeights, float* dHs, float *gpuDen, float* dDen0, float* dDen1){
    // serial point ID
    int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	
	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

    float p_w = dPoints.weights[tid];
    float e_w = dWeights[tid];
    float h = dHs[tid];
    if(dDen1 != NULL){
      float g = dGaussianKernel(h*h, 0.0f) * p_w *e_w;
      //dDen1[tid] = logf(gpuDen[tid] - g);
	  float _den1 = max(gpuDen[tid] - g, EPSILONDENSITY);
	  dDen1[tid] = logf(_den1);
    }
    if(dDen0 != NULL){
      dDen0[tid] = logf(gpuDen[tid]);
    }

    // reset gpuDen to 0.0
    gpuDen[tid] = 0.0f;
}

// compute spatially varying bandwidths
__global__ void CalcVaryingBandwidths(const SamplePoints dPoints, float h, float * dHs)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	//Offset tid for multiple GPUs
	//Add offset for multiple GPUs
	int tid0 = tid;
	tid += dPoints.start;
	//if (tid % 1000000 == 0) {
	//	printf("tid0=%d dPoints.start=%d tid=%d blockIdx.y=%d gridDim.y=%d blockIdx.x=%d blockDim.x=%d threadIdx.x=%d\n", tid0, dPoints.start, tid, blockIdx.y, gridDim.y, blockIdx.x, blockDim.x, threadIdx.x);
	//}
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

	// otherwise calculate varying bandwidth for point ID = tid
	dHs[tid] = h;
	//printf("dHs[%d]=%f\n", tid, dHs[tid]);
}

// compute spatially varying bandwidths
__global__ void CalcVaryingBandwidths(const SamplePoints dPoints, float* dDen0, float h, float alpha, float * dHs, float dReductionSum)
{

	// serial point ID
	 int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	 //printf("alpha tid=%d blockIdx.y=%d gridDim.y=%d blockIdx.x=%d blockDim.x=%d threadIdx.x=%d\n", tid, blockIdx.y, gridDim.y, blockIdx.x, blockDim.x, threadIdx.x);
	 int tid0 = tid;
	//Add offset for multiple GPUs
	tid += dPoints.start;
	//if (tid % 1000000 == 0) {
	//	printf("alpha tid=%d dPoints.start=%d tid=%d blockIdx.y=%d gridDim.y=%d blockIdx.x=%d blockDim.x=%d threadIdx.x=%d\n", tid0, dPoints.start, tid, blockIdx.y, gridDim.y, blockIdx.x, blockDim.x, threadIdx.x);
	//}
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

	// otherwise calculate varying bandwidth for point ID = tid
	int n = dPoints.numberOfPoints;
	float g = expf(dReductionSum / n);
	float den = dDen0[tid];
	//printf("g=%f dReductionSum=%f den=%f\n", g, dReductionSum, den);
	//if(tid == 0){
	//	den = dDen0_0;
	//}
	//float tmph = (h * (powf(expf(den) / g, alpha))); // this outmost () is NECESSARY!
	dHs[tid] = (h * (powf(expf(den) / g, alpha)));
	//printf("dReductionSum: %f, n: %d, g: %f, dDen[%d]: %f, dHs[%d]: %4.5f \n", dReductionSum, n, g, tid, expf(den), tid, dHs[tid]);
	//dHs[tid] = h;
	//printf("g=%f dReductionSum=%f den=%f dHs[%d]=%f\n", g, dReductionSum, den, tid, dHs[tid]);
}

// **===----------------- Parallel reduction (sum) ---------------------===**
//! @param g_data           input array in global memory
//                          result is expected in index 0 of g_idata
//! @param N                input number of elements to scan from input data
//! @param iteration        current iteration in reduction
// **===------------------------------------------------------------------===**
__global__ void ReductionSum(float *g_data, int idxg, int div, int N, int iteration, int num_active_items)
{
	// use shared memory
	__shared__ float s_data[BLOCK_SIZE];

	int thread_id = threadIdx.x;

	int serial_thread_id = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	
	int tmp = 1;
	for (int i = 0; i < iteration; i++) {
		tmp *= BLOCK_SIZE;
	}
	int arrIdx = idxg + serial_thread_id * tmp; //pow((float)BLOCK_SIZE, (float)iteration);

	//if (iteration > 1) printf("%d, %f\n", tmp, pow((float)BLOCK_SIZE, (float)iteration));

	//if(arrIdx < N && serial_thread_id < num_active_items){
	if (arrIdx < (idxg + div) && serial_thread_id < num_active_items) {
		s_data[thread_id] = g_data[arrIdx];
		//if (iteration >= 2) printf("pow = %f %d %d\n", pow((float)BLOCK_SIZE, (float)iteration), (int)pow((float)BLOCK_SIZE, (float)iteration), tmp);
	}
	else{
		s_data[thread_id] = 0.0f;
	}

	// sync threads to ensure all data are loaded into shared memory
	__syncthreads();

	// # of elements in the array to reduce
	int n_ele = BLOCK_SIZE; // initial # of elements = 1024

	// recursively reduce the array
	while(n_ele > 1){
		int m = n_ele / 2;
		if(thread_id < m){
			s_data[thread_id] += s_data[thread_id + m];
		}
		__syncthreads();
		n_ele /= 2;
	}

	// write result back to global memory
	if(thread_id == 0){
		// avoid using pow() or powf() due to precision issues
		int tmp = 1;
		for (int i = 0; i < iteration; i++) {
			tmp *= blockDim.x;
		}

		int idx = idxg + (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x * tmp; //pow((float)blockDim.x, (float)iteration);
		//if (iteration > 1) printf("%d, %f\n", tmp, pow((float)BLOCK_SIZE, (float)iteration));

		//if(idx < N){
		if (idx < idxg + div) {
			g_data[idx] = s_data[0];
		}

		//if(num_active_items <= BLOCK_SIZE){
		if (num_active_items <= BLOCK_SIZE) {
			//dReductionSum = g_data[0];
			dReductionSum = g_data[idxg];
			//printf("g_data[%d] = %f\n", idxg, g_data[idxg]);
			
		}
	}
}


// **===----------------- Parallel reduction (sum) ---------------------===**
//! @param g_data           input array in global memory
//                          result is expected in index 0 of g_idata
//! @param N                input number of elements to scan from input data
//! @param iteration        current iteration in reduction
// **===------------------------------------------------------------------===**
__global__ void ReductionSum_V0(float* g_data, int N, int iteration, int num_active_items)
{
	// use shared memory
	__shared__ float s_data[BLOCK_SIZE];

	int thread_id = threadIdx.x;
	int serial_thread_id = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//unsigned int thread_id = threadIdx.x;
	//unsigned int serial_thread_id = blockDim.x* blockIdx.x + threadIdx.x;

	//if (iteration == 0) printf("%d %d\n", iteration, serial_thread_id);

	// each thread loads one element from global to shared memory
	//unsigned int arrIdx =  serial_thread_id * powf(BLOCK_SIZE, iteration);

	//unsigned int arrIdx = serial_thread_id * pow((float)BLOCK_SIZE, (float)iteration);
	// avoid using pow() or powf() due to precision issues
	int tmp = 1;
	for (int i = 0; i < iteration; i++) {
		tmp *= BLOCK_SIZE;
	}
	int arrIdx = serial_thread_id * tmp; //pow((float)BLOCK_SIZE, (float)iteration);

	if (arrIdx < N && serial_thread_id < num_active_items) {
		s_data[thread_id] = g_data[arrIdx];
		//if (iteration >= 2) printf("pow = %f %d %d\n", pow((float)BLOCK_SIZE, (float)iteration), (int)pow((float)BLOCK_SIZE, (float)iteration), tmp);
	}
	else {
		s_data[thread_id] = 0.0f;
	}

	//if (iteration == 2) printf("", thread_id, s_data[thread_id]);

	// sync threads to ensure all data are loaded into shared memory
	__syncthreads();

	// # of elements in the array to reduce
	int n_ele = BLOCK_SIZE; // initial # of elements = 1024

	// recursively reduce the array
	while (n_ele > 1) {
		int m = n_ele / 2;
		if (thread_id < m) {
			s_data[thread_id] += s_data[thread_id + m];
		}
		__syncthreads();
		n_ele /= 2;
	}

	// write result back to global memory
	if (thread_id == 0) {
		//unsigned int idx = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x * powf(blockDim.x, iteration);
		// avoid using pow() or powf() due to precision issues
		int tmp = 1;
		for (int i = 0; i < iteration; i++) {
			tmp *= blockDim.x;
		}

		int idx = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x * tmp; //pow((float)blockDim.x, (float)iteration);
		//unsigned long idx = blockDim.x * blockIdx.x * (unsigned long)pow((float)blockDim.x, (float)iteration);

		//if (iteration >= 2) printf("pow = %f %d %d\n", pow((float)blockDim.x, (float)iteration), (int)pow((float)blockDim.x, (float)iteration), tmp);

		if (idx < N) {
			/*if(idx == 0){
				dDen0_0 = g_data[idx];
			}*/
			g_data[idx] = s_data[0];
			//if (iteration > 1) printf("iteration = %d, g_data[%d] = %f\n", iteration, idx, g_data[idx]);
		}


		//if(num_active_items <= BLOCK_SIZE){
		if (num_active_items <= BLOCK_SIZE) {
			dReductionSum = g_data[0];

		}

		//printf("%d %d %f %f %d\n", thread_id, iteration, dReductionSum, g_data[0], num_active_items);
	}

	//if(thread_id = BLOCK_SIZE - 1) printf("%d %d %f %f %d\n", thread_id, iteration, dReductionSum, g_data[0], num_active_items);


}

// Mark the cells on boundary with 1 on a raster representation of the study area
// By Guiming @ 2016-09-02
__global__ void dMarkBoundary(AsciiRaster dAscii)
{
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dAscii.start;
	// directly return if ID goes out of range
	if (tid >= dAscii.end) {
		return;
	}

	// # of rows and cols
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;

	//printf("tid %d \n", tid);
	// otherwise, do KDE
	float noDataValue = dAscii.noDataValue;
	// which row, col?
	size_t row = tid / nCols;
	size_t col = tid - row * nCols;

	// should do KDE on this cell?
	float val = dAscii.elements[tid];

	if(val == noDataValue) {
		return;
	}

	if(row == 0 || (row == nRows - 1) || col == 0 || (col == nCols - 1)){ //cells on the outmost rows and cols
		dAscii.elements[row * dAscii.nCols + col] = 1.0f;
		return;
	}
	else{ // cells in interior
		if(dAscii.elements[(row - 1) * nCols + col - 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}
		if(dAscii.elements[row * nCols + col - 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}
		if(dAscii.elements[(row + 1) * nCols + col - 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}

		if(dAscii.elements[(row - 1) * nCols + col] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}
		if(dAscii.elements[(row + 1) * nCols + col] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}

		if(dAscii.elements[(row - 1) * nCols + col + 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}
		if(dAscii.elements[row * nCols + col + 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}
		if(dAscii.elements[(row + 1) * nCols + col + 1] == noDataValue){
			dAscii.elements[row * nCols + col] = 1.0f;
			return;
		}

		dAscii.elements[row * nCols + col] = 0.0f;
	}
}

// Compute the nearest distance to boundary (squared) at each point
// By Guiming @ 2016-09-02
__global__ void dCalcDist2Boundary(SamplePoints dPoints, const AsciiRaster dAscii)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	//Add offset for multiple GPUs
	tid += dPoints.start;
	// directly return if ID goes out of range
	if (tid >= dPoints.end) {
		return;
	}

	

	// otherwise calculate edge effect correction weight point ID = tid
	float cellSize = dAscii.cellSize;
	size_t nCols = dAscii.nCols;
	size_t nRows = dAscii.nRows;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	//float noDataValue = dAscii.noDataValue;

	//printf("%d %d\n", nCols, nRows);

	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float minDist = float_MAX;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);

	float cell_x, cell_y, val;
	double d2;
	size_t row, col;

	if (dAscii.compute_serialized) {
		for (size_t i = 0; i < dAscii.nVals; i++) {
			if (dAscii.elementsVals[i] == 1.0f) {

				row = dAscii.rowcolIdx[i] / nCols;
				col = dAscii.rowcolIdx[i] % nCols;

				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				d2 = dDistance2(p_x, p_y, cell_x, cell_y);
				if (d2 < minDist) {
					minDist = d2;
				}
			}
		}
	}
	else {
		for (row = 0; row < nRows; row++) {
			for (col = 0; col < nCols; col++) {
				val = dAscii.elements[row * nCols + col];
				if (val == 1.0f) {
					cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
					cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
					d2 = dDistance2(p_x, p_y, cell_x, cell_y);
					if (d2 < minDist) {
						minDist = d2;
					} // END IF
				} // END IF
			} // END FOR
		} // ENF FOR
	}

	dPoints.distances[tid] = minDist;
	//printf("---done computing distance for point %d\n", tid);
}

//Timothy @ 02/11/2021
//The following kernels all serve the purpose of printing off values of variables utilized by the GPU
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void PrintPoints(SamplePoints points, int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	//Print each value contained in struct for Device Points
	printf("%d: Point - x: %f | y: %f | weight: %f | distance: %f\n", 
		tid, points.xCoordinates[tid], points.yCoordinates[tid], points.weights[tid], points.distances[tid]);
}

__global__ void PrintAscii(AsciiRaster ascii, int n)
{
	// serial point ID
	size_t tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	printf("Ascii | Element: %f\n", ascii.elements[tid]);
}

__global__ void PrintWeights(float* weights, int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	

	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	printf("%d Weights: %f\n", tid, weights[tid]);
}

__global__ void PrintBand(float* dBandwidths, int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	printf("Bandwidth: %f\n", dBandwidths[tid]);
}

__global__ void PrintDen(float* den, int n)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if (tid >= n) {
		return;
	}

	printf("Density %d: %f\n", tid, den[tid]);
}

__global__ void DoNothing()
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	return;

}

#endif // #ifndef _KDE_KERNEL_H_
