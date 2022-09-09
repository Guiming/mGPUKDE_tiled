// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license


/*
NOTES: how to enable openmp in compiling CUDA code:
In Project Properties -> Configuration Properties -> CUDA C/C++ -> Command Line -> Additional Options: -Xcompiler "/openmp"
https://stackoverflow.com/questions/3211614
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <array>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <omp.h>

#include "SamplePoints.h"
#include "AsciiRaster.h"
#include "Utilities.h"

#include "KDtree.h"
#include "CUDA_KDtree.h"

#include "kde_kernel_kdtr.cu"
//#include "CUDA_KDtree.cu" // it seems this is not needed since we already have CUDA_KDtree.h included
//#include "Utilities.h"

#include "GeoTIFF.cpp"

using namespace std;
// for timing
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

// distance squared between two points
inline  float Distance2(float x0, float y0, float x1, float y1){
	float dx = x1 - x0;
	float dy = y1 - y0;
	return dx*dx + dy*dy;
}

// mean center of points
void MeanCenter(SamplePoints Points, float &mean_x, float &mean_y);

// (squared) standard distance of points
void StandardDistance2(SamplePoints Points, float &d2);

// bandwidth squared
inline float BandWidth2(SamplePoints Points){
	float d2;
	StandardDistance2(Points, d2);
	return sqrtf(2.0f / (3 * Points.numberOfPoints)) * d2;
}

// Gaussian kernel
inline float GaussianKernel(float h2, float d2){
	if(d2 >= CUT_OFF_FACTOR * h2){
		return 0.0f;
	}
	return expf(d2 / (-2.0f * h2)) / (h2*TWO_PI);
}

//Timothy @ 01/21/2020
//EDIT: Changed AllocateDeviceSamplePoints to return void, and instead utilize pointers to an array
//Changed all functions to utilize the array of pointers
void AllocateDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints Points);
void CopyToDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints hPoints);
void CopyFromDeviceSamplePoints(SamplePoints hPoints, const SamplePoints* dPoints);
SamplePoints AllocateSamplePoints(int n); // random points

// if bandwidths is True, bandwidths are provided in the file(Hoption = -1)
// By Guiming @ 2021-09-10
SamplePoints ReadSamplePoints(const char *csvFile, bool bandwidths); // points read from a .csv file: x, y, [h,] w

// bandwidths are provided in the file (Hoption = -1)
// By Guiming @ 2021-09-10
//SamplePoints ReadSamplePoints(const char* csvFile, float* hs); // points read from a .csv file, with bandwidth: x, y, h, w

// By Guiming @ 2016-09-04
SamplePoints CopySamplePoints(const SamplePoints Points);
void FreeDeviceSamplePoints(SamplePoints* dPoints);
void FreeSamplePoints(SamplePoints* Points);
void WriteSamplePoints(SamplePoints* Points, const char * csvFile);
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, const char * csvFile);
//void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, float* KNNdist, const char* csvFile);
void ReformPoints(SamplePoints* dPoints, const SamplePoints hPoints); //Timothy @ 08/13/2021
void ReformPoints(SamplePoints* dPoints); //Timothy @ 08/13/2021
void DividePoints(SamplePoints* dPoints); //Timothy @ 08/13/2021

void AllocateDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster Ascii);
void CopyToDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii);
void CopyFromDeviceAsciiRaster(AsciiRaster hAscii, const AsciiRaster dAscii);
AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue);
AsciiRaster ReadAsciiRaster(char * asciiFile); // ascii raster read from a .asc file
AsciiRaster ReadGeoTIFFRaster(char* geotiffFile); // geotiff raster read from a .tif file
//Guiming  2022-01-18 - Construct AsciiRaster from a tile read from a GeoTIFF
AsciiRaster AsciiRasterFromGeoTIFFTile(double* geotransform, const char* projection, int nrows, int ncols, double nodata, float** data);
AsciiRaster CopyAsciiRaster(const AsciiRaster Ascii);
void FreeDeviceAsciiRaster(AsciiRaster* Ascii);
void FreeAsciiRaster(AsciiRaster* Ascii);
void WriteAsciiRaster(AsciiRaster* Ascii, const char * asciiFile);
void WriteGeoTIFFRaster(AsciiRaster* Ascii, const char* geotiffFile);

void ReformAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii); //Guiming 2021-08-18 Combine rasters amongst GPUs into a single raster
void ReformGPUAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii); //Guiming 2021-08-30 Add up cell densities from all devices (gpuDen) into one single array

float* AllocateEdgeCorrectionWeights(SamplePoints Points);
void CopyToDeviceWeights(float** dWeights, const float* hWeights, const int n);
void FreeEdgeCorrectionWeights(float* weights);
void ReformECWeights(float** dWeights, float* hWeights); //Timothy @ 08/13/2021

void AllocateDeviceEdgeCorrectionWeights(float** dWeights, SamplePoints Points);
void FreeDeviceEdgeCorrectionWeights(float** weights);

///////// Guiming on 2016-03-16 ///////////////
// the array holding bandwidth at each point
float* AllocateBandwidths(int n); // n is number of points
//Allocation on device now done with pointers instead of return
void AllocateDeviceBandwidths(float** dBandwidths, int n); // n is number of points
void CopyToDeviceBandwidths(float** dBandwidth, const float* hBandwidths, const int n);
void CopyFromDeviceBandwidths(float* hBandwidth, const float* dBandwidths, const int n);
void FreeDeviceBandwidths(float** bandwidths);
void FreeBandwidths(float* bandwidths);
void ReformBandwidths(float** dBand, float* hBand); //Timothy @ 08/13/2021 - Reform bandwidth arrays on host and copy back accross devices

// the array holding inclusive/exclusive density at each point
float* AllocateDen(int n); // n is number of points
void AllocateDeviceDen(float** dDen, int n); // n is number of points
void CopyToDeviceDen(float** dDen, const float* hDen, const int n);
void CopyFromDeviceDen(float* hDen, const float* dDen, const int n);
void CopyDeviceDen(float* dDenTo, const float* dDenFrom, const int n);
void FreeDeviceDen(float** den);
void FreeDen(float* den);
void ReformDensities(float** dDen, float* den); //Timothy @ 12/29/21 - Reforms densities from all devices back into one single array
void ReformGPUDensities(float** gpuDen, float* den); //Guiming @ 08/15/21 - Add up densities from all devices (gpuDen) into one single array

// compute the optimal Maximum Likelihood Estimation fixed bandwidth
// By Guiming @ 2016-02-26
float MLE_FixedBandWidth(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dDen0 = NULL, float** dDen1 = NULL);

// compute fixed bandwidth density at sample points
// By Guiming @ 2016-05-21
void ComputeFixedDensityAtPoints(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, float* dDen0 = NULL, float* dDen1 = NULL);

// compute the log likelihood given single bandwidth h
// By Guiming @ 2016-02-26
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dDen0 = NULL, float** dDen1 = NULL);

// compute the log likelihood given bandwidths hs
// By Guiming @ 2016-02-26
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float* hs, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float h = 1.0f, float alpha = -0.5f, float** dDen0cpy = NULL);

// compute the log likelihood given a center (h0, alpha0) and step (stepH, stepA)
// By Guiming @ 2016-03-06
void hj_likelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, int lastdmax, float* logLs, float* hs = NULL, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float** dDen0cpy = NULL);

// compute the optimal h and alpha (parameters for calculating the optimal adaptive bandwith)
// By Guiming @ 2016-03-06
void hooke_jeeves(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, float* hs = NULL, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float** dDen0cpy = NULL);

float compGML(float* den0, int n);
///////// Guiming on 2016-03-16 ///////////////


// exact edge effects correction (Diggle 1985)
void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights);
void EdgeCorrectionWeightsExact(SamplePoints Points, float *hs, AsciiRaster Ascii, float *weights);

// check whether the result from sequential computation and that from parallel computation agree
void CheckResults(AsciiRaster AsciiSEQ, AsciiRaster AsciiPARA);

// reduction an array on GPU
void ReductionSumGPU_V0(float* dArray, int numberOfElements);
float ReductionSumGPU(float** dArray, int numberOfElements);

// extract study area boundary from a raster
// By Guiming @ 2016-09-02
void MarkBoundary(AsciiRaster* Ascii, bool useGPU = false);

// compute the closest distances from sample points to study area boundary
// By Guiming @ 2016-09-02
void CalcDist2Boundary(SamplePoints* Points, AsciiRaster* Ascii, bool useGPU = false);

// sort the sample points on their distances to study area boundary
// By Guiming @ 2016-09-04
void SortSamplePoints(SamplePoints Points);

// sort the sample points on their distances to study area boundary
// if bandwidths are provided in file (Hoption = -1), need to adjust the order of bandwidths
// By Guiming @ 2021-09-10
void SortSamplePoints(SamplePoints Points, float* hs);

// comparison function for sort
// By Guiming @ 2016-09-04
int compare ( const void *pa, const void *pb );

void BuildCPUKDtree (SamplePoints Points);
void BuildGPUKDtree ();

void EnableP2P(); //Timothy @ 08/13/2020 - Enable P2P Access Across Devices
void nextDev(int numDev, int& curDev); //Timothy @ 08/14/2020 - Determine next Device to be used
void DevProp(); //Timothy @ 08/24/2020 - Check device properties, primarily for troubleshooting purposes

// By Timothy @ 02/26/2020
//This performs the same tasks as ComputeFixedDensityAtPoints function, however it is designed specifically to run in accross multiple
//GPUs asynchronously
void ComputeFixedDensityDevice(cudaStream_t* streams, AsciiRaster* Ascii, SamplePoints* Points, float** edgeWeights, float h, float* den0, float** dDen0);
void printDdensities(float** gpuDen);
void cudaStreamStatus();
void cudaSynchronizeStreams();
void printHascii(AsciiRaster ascii);
void printHpoints();
void printHdensities(float* hden, int n);

int test_ReductionSumGPU(int n);

void ComputeNearestNeighborDistances(float& meanNNDist, float& minNNDist, float& maxNNDist);
void ComputeKNearestNeighborDistances(int k, float* knn_dist);


/* Run in 2 modes
 *
 * Mode 0: Do not read points and mask from files.
 *         User specify # of points and cell size of the estimated intensity surface.
 *         Random points with x, y coordinates in the range [0,100] will be generated.
 *         The cell size (must be less than 100) determines how many cells in the intensity surface raster.
 *
 *         ./kde_cuda [mode] [#points] [cellsize] [bwoption] [skipSEQ] [skipPARA] [num_gpu] [denFN_seq] [denFN_cuda]
 *         e.g., ./kde_cuda 0 100 1.0 2 0 0
 *
 * Mode 1: Read points and mask from files.
 *
 *         ./kde_cuda [mode] [points_file] [mask_file] [bwoption] [skipSEQ] [skipPARA] [num_gpu] [denFN_seq] [denFN_cuda]
 *         e.g., ./kde_cuda 1 ../Points.csv ../Mask.asc 2 0 0
 *
*/

/* be very careful with these global variables
 * they are declared in this way to avoid passing additional parameters in functions
*/

int GPU_N = 2;
int GPU_START = 0;

KDtree tree; // pointer to the kd tree, can be accessed in any function
//CUDA_KDTree GPU_tree[GPU_N]; //pointer to the GPU kd tree, can be accessed in any function. EDIT: A copy of the tree 
CUDA_KDTree* GPU_tree;
//is now kept on each GPU with each of these pointers corresponding to a GPU.

vector <Point> dataP; // pointer to the vector to hold data points in kd tree, initilized when building kd tree
//float* gpuDen[GPU_N]; // this is a global array allocated on gpu to store density values. Used in DensityAtPointsKdtr
float** gpuDen;
//int* gpu_ret_indexes;
//float* gpu_ret_dists;
//float* zeroDen;
int MAX_N_NBRS = 0;

//Timothy @ 08/13/2020
//int GPU_N = 1; //Holds number of GPUs on machine
//int GPU_C = 0; //Keeps track of our current GPU

//cudaStream_t streams[GPU_N]; //Streams to be used for parallelism
cudaStream_t* streams;

SamplePoints sPoints; // sample of point events

float* gedgeWeights;  // edge effect correct weights (for each point in the sample)
float* knndistance;  // the K's nearest neighbor distance (for each point in the sample)

float* hs; // bandwidth for points on host
float* den0;
float* den1;
float reductionSum;
float meanNNDist = 0.0f;
float minNNDist = 0.0f;
float maxNNDist = 0.0f;

bool UPDATEWEIGHTS = 1;
bool SAMPLEWEIGHTS = 1;

void OMP_TEST();

int main(int argc, char *argv[]){
	try {

		//OMP_TEST();
		//exit(0);
		auto T1 = high_resolution_clock::now();

		int NPNTS = 100;                // default # of points
		float CELLSIZE = 1.0f;          // default cellsize
		char* pntFn = "data/Points.csv";  // default points file
		char* maskFn = "data/Mask.asc";   // default mask file
		bool fromFiles = true;          // by default, read Points and Mask from files

		int NGPU = 1;

		int SKIPSEQ = 0;                // by default, do not skip sequential execution
		int SKIPPARA = 0;               // by default, do not skip parallel execution

		//Guiming May 1, 2016
		int Hoption = 0; // 0 for rule of thumb
						 // 1 for h optimal
						 // 2 for h adaptive
						 // -1 use whatever bandwidths provided in the input file
		
		int KNN = 0; // use knn distance as bandwidth for adaptive KDE?
		int KNN_k = 20; // K's nearest neighbor distance
		float KNN_coef = 1.0; // knn distance multiplied by a coefficient

		char* denSEQfn = "data/den_SEQ.asc";
		char* denCUDAfn = "data/den_CUDA.asc";

		// parse commandline arguments
		if (argc != 15) {
			printf("Incorrect arguments provided. Exiting...\n");
			printf("Run in mode 0:\n ./kde_cuda 0 #points cellsize h_option enable_edge_corection enable_sample_weight skip_sequential skip_parallel num_gpu enable_knn knn_k knn_coef denfn_seq denfn_cuda\n");
			printf("Run in mode 1:\n ./kde_cuda 1 points_file mask_file h_option enable_edge_corection enable_sample_weight skip_sequential skip_parallel num_gpu enable_knn knn_k knn_coef denfn_seq denfn_cuda\n");
			return 1;
		}
		else {
			int mode = atoi(argv[1]);
			if (mode == 0) {
				fromFiles = false;
				NPNTS = atoi(argv[2]);
				CELLSIZE = (float)atof(argv[3]);
				Hoption = atoi(argv[4]);

				if (Hoption == -1) {
					printf("***Error - should never use bandwidth option -1 in mode 0 (i.e., randomly generately points). Exiting...");
					return 1;
				}

				UPDATEWEIGHTS = atoi(argv[5]);
				SAMPLEWEIGHTS = atoi(argv[6]);

				SKIPSEQ = atoi(argv[7]);
				SKIPPARA = atoi(argv[8]);

				NGPU = atoi(argv[9]);

				KNN = atoi(argv[10]);
				KNN_k = atoi(argv[11]);
				KNN_coef = (float)atof(argv[12]); // a coefficient to multiply KNN distance, or to multiply bandwidths provided in file

				denSEQfn = argv[13];
				denCUDAfn = argv[14];
			}
			else if (mode == 1) {
				pntFn = argv[2];
				maskFn = argv[3];
				Hoption = atoi(argv[4]);
				
				UPDATEWEIGHTS = atoi(argv[5]);
				SAMPLEWEIGHTS = atoi(argv[6]);

				SKIPSEQ = atoi(argv[7]);
				SKIPPARA = atoi(argv[8]);
				
				NGPU = atoi(argv[9]);

				KNN = atoi(argv[10]);
				KNN_k = atoi(argv[11]);
				KNN_coef = (float)atof(argv[12]); // a coefficient to multiply KNN distance, or to multiply bandwidths provided in file

				denSEQfn = argv[13];
				denCUDAfn = argv[14];
			}
			else {
				printf("Incorrect arguments provided. Exiting...\n");
				printf("Run in mode 0:\n ./kde_cuda 0 #points cellsize h_option enable_edge_corection enable_sample_weight skip_sequential skip_parallel num_gpu enable_knn knn_k knn_coef denfn_seq, denfn_cuda\n");
				printf("Run in mode 1:\n ./kde_cuda 1 points_file mask_file h_option enable_edge_corection enable_sample_weight skip_sequential skip_parallel num_gpu enable_knn knn_k knn_coef denfn_seq, denfn_cuda\n");
				return 1;
			}

		}

		//Timothy @ 08/13/2020
		//Assign and print number of Compute Capable Devices
		int nGPU;
		cudaGetDeviceCount(&nGPU);
		printf("Number of Capable Devices: %d\n", nGPU);
		if (nGPU < NGPU) {
			printf("Number of requested devices (%d) EXCEEDS number of available devices (%d)\n", NGPU, nGPU);
		}
		GPU_N = min(nGPU, NGPU);
		printf("%d device(s) out of %d devices available are used...\n\n", GPU_N, nGPU);

		// skip the first GPU where possible as it's may be used by the OS for other purposes
		if (GPU_N < nGPU) {
			GPU_START = 0;
		}

		//GPU_tree = new CUDA_KDTree[GPU_N];
		//gpuDen = new float* [GPU_N];
		streams = new cudaStream_t[GPU_N];

		//exit(0);
		//printf("Current GPU: %d\n", GPU_C);

		/*for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			DevProp();
		}
		cudaSetDevice(GPU_START);*/

		//Timothy @ 08/24/2020
		//Enable P2P Access across devices
		//EnableP2P();

		cudaError_t error;

		///*
		//Timothy @ 12/29/2020
		//Create streams for each available device

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i+GPU_START);
			error = cudaStreamCreate(&streams[i]);
		}
		if (error != cudaSuccess)
		{
			printf("Failed to create streams (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		cudaSetDevice(GPU_START); //Reset device to first GPU

		
		//SamplePoints sPoints; // sample of point events
		AsciiRaster Mask;    // a mask indicating the extent of study area
		AsciiRaster DenSurf, DenSurf_CUDA; // the estimated intensity surface
		//float *edgeWeights;  // edge effect correct weights (for each point in the sample)
		bool correction = true; // enable edge effect correction
		srand(100); // If not read from files, generate random points

		auto t1 = high_resolution_clock::now();
		//Read or generate points
		if (fromFiles) {
			//Mask = ReadAsciiRaster(maskFn);
			Mask = ReadGeoTIFFRaster(maskFn);
			/*
			AsciiRaster rst = CopyAsciiRaster(Mask);
			WriteGeoTIFFRaster(&rst, (const char*)"maskout.tif");
			printf("write success\n");
			FreeAsciiRaster(&Mask);
			FreeAsciiRaster(&rst);
			exit(0);
			*/
			if (Hoption == -1) { 
				sPoints = ReadSamplePoints(pntFn, true);
				hs = AllocateBandwidths(sPoints.numberOfPoints);
				for (int i = 0; i < sPoints.numberOfPoints; i++) {
					hs[i] = KNN_coef * sPoints.distances[i];
					sPoints.distances[i] = 0.0f;
					//printf("hs[%d] = %f, %f\n", i, sPoints.distances[i], hs[i]);
				}
			}
			else {
				sPoints = ReadSamplePoints(pntFn, false);
			}
		}
		else {

			sPoints = AllocateSamplePoints(NPNTS);
			Mask = AllocateAsciiRaster(int(100 / CELLSIZE), int(100 / CELLSIZE), 0.0f, 0.0f, CELLSIZE, -9999.0f);

		}		
		auto t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a double. */
		duration<double, std::milli> ms_double = t2 - t1;
		printf("...reading in data took %f ms\n", ms_double.count());

		//printHpoints();
		//exit(0);
		/*
		// testing ReductionSumGPU
		int FLAG = test_ReductionSumGPU(sPoints.numberOfPoints);
		if (FLAG == 1) {
			printf("***test ReductionSumGPU on %d 1's failed. exiting...\n", sPoints.numberOfPoints);

			FreeAsciiRaster(&Mask);
			FreeSamplePoints(&sPoints);

			exit(0);
		}
		exit(0);
		*/

		if (Hoption > -1) {
			hs = AllocateBandwidths(sPoints.numberOfPoints);
		}
		gedgeWeights = AllocateEdgeCorrectionWeights(sPoints);
		den0 = AllocateDen(sPoints.numberOfPoints);
		den1 = AllocateDen(sPoints.numberOfPoints);
		knndistance = AllocateBandwidths(sPoints.numberOfPoints);

		DenSurf = CopyAsciiRaster(Mask);

		// parameters
		int numPoints = sPoints.numberOfPoints;
		int nCols = Mask.nCols;
		int nRows = Mask.nRows;
		float xLLCorner = Mask.xLLCorner;
		float yLLCorner = Mask.yLLCorner;
		float noDataValue = Mask.noDataValue;
		float cellSize = Mask.cellSize;

		printf("number of points: %d\n", numPoints);
		printf("cell size: %f\n", cellSize);
		printf("number of cells: %llu\n", (size_t)nCols * nRows);

		printf("bandwidth option: %d\n", Hoption);
		printf("enable edge correction: %d\n", UPDATEWEIGHTS);
		printf("enable sample weight: %d\n", SAMPLEWEIGHTS);

		printf("skip executing SEQUENTIAL program? %d\n", SKIPSEQ);
		printf("skip executing PARALLEL program? %d\n", SKIPPARA);

		printf("use KNN distance for adatpive bandwidth? %d\n", KNN);
		printf("KNN K: %d\n", KNN_k);
		printf("multipication coefficient for KNN distance: %f\n", KNN_coef);
	
		printf("number of threads per block: %d\n", BLOCK_SIZE);



		// do the work
		float cell_x; // x coord of cell
		float cell_y; // y coord of cell
		float p_x;    // x coord of point
		float p_y;    // x coord of point
		float p_w;    // weight of point
		float e_w = 1.0;    // edge effect correction weight

		float h = sqrtf(BandWidth2(sPoints));
		printf("rule of thumb bandwidth h0: %.5f\n", h);

		// timing
		//double start, stop;
		float elaps_seq, elaps_exc, elaps_inc;

		if (SKIPSEQ == 0) {

			//gedgeWeights = NULL;
			//gedgeWeights = AllocateEdgeCorrectionWeights(sPoints);

		///////////////////////// SEQUENTIAL /////////////////////////////////
			///////////////////////// START CPU TIMING /////////////////////////////
			cudaEvent_t startCPU;
			error = cudaEventCreate(&startCPU);

			if (error != cudaSuccess)
			{
				printf("Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			cudaEvent_t stopCPU;
			error = cudaEventCreate(&stopCPU);

			if (error != cudaSuccess)
			{
				printf("Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Record the start event
			error = cudaEventRecord(startCPU, NULL);
			if (error != cudaSuccess)
			{
				printf("Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//cudaStreamSynchronize(streams[0]);
			///////////////////////// END OF START CPU TIMING /////////////////////////////

			// By Guiming @ 2016-09-11
			
			size_t pNum = Mask.nCols * Mask.nRows;
			size_t NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
			size_t GRID_SIZE_W = (size_t)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			DoNothing << <dimGrid_W, BLOCK_SIZE, 0, streams[0] >> > ();
			cudaStreamSynchronize(streams[0]);

			if(UPDATEWEIGHTS){

				MarkBoundary(&Mask); // either on GPU or CPU
				//printHascii(Mask);
				//WriteAsciiRaster(&Mask, "rw/Maskcpu.asc");

				CalcDist2Boundary(&sPoints, &Mask);
				//printHpoints();
				//WriteAsciiRaster(&Mask, "output/boundary.asc");
				if (Hoption == -1) {
					//printf("***\n");
					SortSamplePoints(sPoints, hs);
					//printf("***\n");
				}
				else {
					SortSamplePoints(sPoints);
				}
				//printf("***after sorting 0 \n");
				//printHpoints();			
				//exit(0);
			}
			// By Guiming @ 2016-11-03
			BuildCPUKDtree(sPoints);

			//Guiming 8/28/2021
			//Compute average nearest neighbor distance
			if (false && Hoption == 1) {
				ComputeNearestNeighborDistances(minNNDist, maxNNDist, meanNNDist);
				//printf("nearest neighbor distances: mean = %f, min = %f, max = %f\n", meanNNDist, minNNDist, maxNNDist);
			}

			if (Hoption == 2 && KNN == 1) {
				ComputeKNearestNeighborDistances(KNN_k, knndistance);
			}

			//exit(0);

			if (Hoption == -1) {
				if (UPDATEWEIGHTS) {
					// compute edge effect correction weights
					EdgeCorrectionWeightsExact(sPoints, hs, Mask, gedgeWeights);
				}
			}
			else {

				//float* hs = AllocateBandwidths(numPoints);
				for (int i = 0; i < numPoints; i++)
				{
					hs[i] = h;
				}

				if (UPDATEWEIGHTS) {
					// compute edge effect correction weights
					EdgeCorrectionWeightsExact(sPoints, h, Mask, gedgeWeights);
				}

				if (Hoption == 1) {
					float hopt = MLE_FixedBandWidth(&Mask, &sPoints, &gedgeWeights, h, NULL, NULL, false);
					printf("cross validated optimal fixed bandwidth hopt: %.5f\n", hopt);

					for (int i = 0; i < numPoints; i++) {
						hs[i] = hopt;
					}

					// update edge correction weights
					if (UPDATEWEIGHTS) {
						EdgeCorrectionWeightsExact(sPoints, hs, Mask, gedgeWeights);
					}
				}

				if (Hoption == 2) {
					//float* den0 = AllocateDen(numPoints);
					//float* den1 = AllocateDen(numPoints);

					if (KNN == 0) {
						float h0 = h;
						float alpha0 = -0.5;
						float stepH = h0 / 5;
						float stepA = 0.1;
						float* optParas = (float*)malloc(3 * sizeof(float));

						hooke_jeeves(&Mask, &sPoints, NULL, &gedgeWeights, h0, alpha0, stepH, stepA, optParas, hs, den0, den1, false);
						h0 = optParas[0];
						alpha0 = optParas[1];
						float logL = optParas[2];

						if (DEBUG) printf("h0: %.5f alpha0: %.5f Lmax: %.5f\n", h0, alpha0, logL);

						free(optParas);
						optParas = NULL;

						ComputeFixedDensityAtPoints(Mask, sPoints, gedgeWeights, h0, den0, NULL, false);
						float gml = compGML(den0, numPoints);
						for (int i = 0; i < numPoints; i++) {
							hs[i] = h0 * powf(den0[i] / gml, alpha0);
						}

					}
					else { // use knn distance as adaptive bandwidth
						for (int i = 0; i < numPoints; i++) {
							hs[i] = KNN_coef * knndistance[i];
						}
					}

					// update edge correction weights
					if (UPDATEWEIGHTS) {
						EdgeCorrectionWeightsExact(sPoints, hs, Mask, gedgeWeights);
					}
				}
			}
			// KDE
			for (int row = 0; row < nRows; row++) {
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				for (int col = 0; col < nCols; col++) {
					cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
					//int idx = row * nCols + col;
					size_t idx = row * nCols + col;
					if (DenSurf.elements[idx] != noDataValue) {

						float den = 0.0;
						float hp;
						for (int p = 0; p < numPoints; p++) {
							p_x = sPoints.xCoordinates[p];
							p_y = sPoints.yCoordinates[p];
							p_w = sPoints.weights[p];
							hp = hs[p];
							if (correction) {
								e_w = gedgeWeights[p];
							}
							float d2 = Distance2(p_x, p_y, cell_x, cell_y);
							den += GaussianKernel(hp * hp, d2) * p_w * e_w;
						}
						DenSurf.elements[idx] = den; // intensity, not probability
					}
				}
			}

			///////////////////////// STOP CPU TIMING /////////////////////////////
			//cudaStreamSynchronize(streams[0]);
			// Record the stop event
			error = cudaEventRecord(stopCPU, NULL);

			if (error != cudaSuccess)
			{
				printf("Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Wait for the stop event to complete
			error = cudaEventSynchronize(stopCPU);
			if (error != cudaSuccess)
			{
				printf("Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//printf("startCPU = %f, stopCPU = %f\n", startCPU, stopCPU);
			elaps_seq = 0.0f;
			error = cudaEventElapsedTime(&elaps_seq, startCPU, stopCPU);

			if (error != cudaSuccess)
			{
				printf("Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			
			///////////////////////// END OF STOP CPU TIMING /////////////////////////////
			//printf("MAX_N_NBRS=%d\n", MAX_N_NBRS);
			printf("Computation on CPU took %.4f ms\n\n", elaps_seq);

			// write results to file
			//WriteAsciiRaster(&DenSurf, denSEQfn);
			WriteGeoTIFFRaster(&DenSurf, denSEQfn);
			WriteSamplePoints(&sPoints, hs, gedgeWeights, "pntsSEQ.csv");

			// clean up (only those not needed any more)
			//FreeEdgeCorrectionWeights(gedgeWeights);
			////FreeAsciiRaster(&DenSurf);
			//FreeBandwidths(hs);
		}
		////////////////////////// END OF SEQUENTIAL //////////////////////////////

		//////////////////////////  CUDA  /////////////////////////////////////////
		if (SKIPPARA == 0) {

			GPU_tree = new CUDA_KDTree[GPU_N];
			gpuDen = new float* [GPU_N];

			DenSurf_CUDA = CopyAsciiRaster(Mask);
			//SamplePoints dPoints[GPU_N];
			//float* dWeights[GPU_N];
			//AsciiRaster dAscii[GPU_N];

			SamplePoints* dPoints = new SamplePoints[GPU_N];
			float** dWeights = new float* [GPU_N];
			AsciiRaster* dAscii = new AsciiRaster[GPU_N];

			AllocateDeviceSamplePoints(dPoints, sPoints);
			AllocateDeviceEdgeCorrectionWeights(dWeights, sPoints);
			// initialize edge correction weights on device
			for (int i = 0; i < sPoints.numberOfPoints; i++) {
				gedgeWeights[i] = 1.0f;
			}
			CopyToDeviceWeights(dWeights, gedgeWeights, sPoints.numberOfPoints);

			AllocateDeviceAsciiRaster(dAscii, Mask);

			// Guiming @ 2016-03-17
			//float* hs = AllocateBandwidths(sPoints.numberOfPoints);
			float* zeroDen = AllocateDen(sPoints.numberOfPoints);

			// Guimig 2021-08-15
			//gedgeWeights = NULL;
			//gedgeWeights = AllocateEdgeCorrectionWeights(sPoints);

			for (int i = 0; i < numPoints; i++) {
				if (Hoption > -1) hs[i] = h;
				zeroDen[i] = 0.0f;
			}

			//float* dHs[GPU_N];
			float** dHs = new float* [GPU_N];
			AllocateDeviceBandwidths(dHs, sPoints.numberOfPoints);

			//float* den0 = AllocateDen(sPoints.numberOfPoints);
			//float* den1 = AllocateDen(sPoints.numberOfPoints);
			//float* dDen0[GPU_N];
			float** dDen0 = new float* [GPU_N];
			AllocateDeviceDen(dDen0, sPoints.numberOfPoints);
			//float* dDen0cpy[GPU_N];
			float** dDen0cpy = new float* [GPU_N];
			AllocateDeviceDen(dDen0cpy, sPoints.numberOfPoints);

			//float* den1 = AllocateDen(sPoints.numberOfPoints);
			//float* dDen1[GPU_N];
			float** dDen1 = new float* [GPU_N];
			AllocateDeviceDen(dDen1, sPoints.numberOfPoints);

			AllocateDeviceDen(gpuDen, sPoints.numberOfPoints);
			//printDdensities(gpuDen);

			//exit(0);
			//gpu_ret_indexes =
			//gpu_ret_dists =

			//printf("Allocate DONE...\n"); //DEBUGGING

			///////////////////////// START GPU INCLUSIVE TIMING /////////////////////////////
			cudaEvent_t startInc;
			error = cudaEventCreate(&startInc);

			if (error != cudaSuccess)
			{
				printf("Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			cudaEvent_t stopInc;
			error = cudaEventCreate(&stopInc);

			if (error != cudaSuccess)
			{
				printf("Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Record the start event
			error = cudaEventRecord(startInc, NULL);
			if (error != cudaSuccess)
			{
				printf("Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			///////////////////////// END OF START GPU INCLUSIVE TIMING /////////////////////////////
			CopyToDeviceBandwidths(dHs, hs, sPoints.numberOfPoints);

			/*int pNum = sPoints.numberOfPoints;

			int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
			*/
			// reset distnaces before copying to GPU
			/*
			for (int i = 0; i < pNum; i++) {
				sPoints.distances[i] = 0.0;
			}
			printHpoints();*/

			CopyToDeviceSamplePoints(dPoints, sPoints);
			///*
			for (int i = 0; i < GPU_N; i++) {
				printf("Device %d start %d end %d\n", i, dPoints[i].start, dPoints[i].end);

			}
			//*/
			//exit(0);


			// !!!!
			CopyToDeviceAsciiRaster(dAscii, Mask);

			size_t cells = dAscii[0].nCols * dAscii[0].nRows;
			CopyToDeviceDen(gpuDen, zeroDen, sPoints.numberOfPoints);

			//printf("Copied...\n"); //DEBUGGING

			///////////////////////// START GPU EXCLUSIVE TIMING /////////////////////////////
			cudaEvent_t startExc;
			error = cudaEventCreate(&startExc);

			if (error != cudaSuccess)
			{
				printf("Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			cudaEvent_t stopExc;
			error = cudaEventCreate(&stopExc);

			if (error != cudaSuccess)
			{
				printf("Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Record the start event
			error = cudaEventRecord(startExc, NULL);
			if (error != cudaSuccess)
			{
				printf("Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF START GPU EXLUSIVE TIMING /////////////////////////////


			///////////////////////// START SORTING TIMING /////////////////////////////
			cudaEvent_t startSort;
			error = cudaEventCreate(&startSort);

			if (error != cudaSuccess)
			{
				printf("Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			cudaEvent_t stopSort;
			error = cudaEventCreate(&stopSort);

			if (error != cudaSuccess)
			{
				printf("Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Record the start event
			error = cudaEventRecord(startSort, NULL);
			if (error != cudaSuccess)
			{
				printf("Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF START SORTING TIMING /////////////////////////////
			///*
			if (UPDATEWEIGHTS) {
				// By Guiming @ 2016-09-11
				MarkBoundary(dAscii, true); // either on GPU or CPU
				//cudaStreamSynchronize(streams[0]);
				ReformAsciiRaster(dAscii, Mask);

				//WriteAsciiRaster(&Mask, "rw/Maskcpu.asc");
				//CopyFromDeviceAsciiRaster(Mask, dAscii[0]);
				//WriteAsciiRaster(&Mask, "rw/Maskgpu.asc");
				//exit(0);


				CalcDist2Boundary(dPoints, dAscii, true);
				//cudaStreamSynchronize(streams[0]);

				//ReformPoints(dPoints, sPoints); // done in CalcDist2Boundary

				//CopyFromDeviceSamplePoints(sPoints, dPoints);

				//printHpoints();
				//printf("***here***\n");
				if (Hoption == -1) {
					SortSamplePoints(sPoints, hs);
					CopyToDeviceBandwidths(dHs, hs, sPoints.numberOfPoints);
				}
				else {
					SortSamplePoints(sPoints);
				}
				//printf("***after sorting 1 \n");
				//printHpoints();
				//printf("***here***\n");
				//WriteSamplePoints(&sPoints, (const char*)"test.csv");

				//EDIT:Timothy @ 12/10/2020
				//When adding back sorted points, divide points as they are copied accross GPUs
				CopyToDeviceSamplePoints(dPoints, sPoints);
				//printf("***here***\n");
				//cudaStreamStatus();
			}
			//cudaSetDevice(GPU_START);
			///////////////////////// STOP SORTING TIMING /////////////////////////////
			// Record the stop event
			error = cudaEventRecord(stopSort, NULL);

			if (error != cudaSuccess)
			{
				printf("Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Wait for the stop event to complete
			error = cudaEventSynchronize(stopSort);
			if (error != cudaSuccess)
			{
				printf("Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			float elaps_sort = 0.0f;
			error = cudaEventElapsedTime(&elaps_sort, startSort, stopSort);

			if (error != cudaSuccess)
			{
				printf("Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF STOP SORTING TIMING /////////////////////////////
			printf("#Sorting took %.3f ms\n", elaps_sort);

			//printf("Sorted...\n"); //DEBUGGING

			///////////////////////// START KDTREE TIMING /////////////////////////////
			cudaEvent_t startKd;
			error = cudaEventCreate(&startKd);

			if (error != cudaSuccess)
			{
				printf("Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			cudaEvent_t stopKd;
			error = cudaEventCreate(&stopKd);

			if (error != cudaSuccess)
			{
				printf("Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Record the start event
			error = cudaEventRecord(startKd, NULL);
			if (error != cudaSuccess)
			{
				printf("Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF START KDTREE TIMING /////////////////////////////

			// By Guiming @ 2016-11-03
			if (SKIPSEQ == 1) {
				//printf("bulding kdtree on CPU since it has not been built yet\n");
				BuildCPUKDtree(sPoints);

				if (Hoption == 1) {
					ComputeNearestNeighborDistances(minNNDist, maxNNDist, meanNNDist);
				}

				if (Hoption == 2 && KNN == 1) {
					ComputeKNearestNeighborDistances(KNN_k, knndistance);
				}

			}
			BuildGPUKDtree(); // needs to build the CPUKDtree first

			///////////////////////// STOP KDTREE TIMING /////////////////////////////
			// Record the stop event
			error = cudaEventRecord(stopKd, NULL);

			if (error != cudaSuccess)
			{
				printf("Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Wait for the stop event to complete
			error = cudaEventSynchronize(stopKd);
			if (error != cudaSuccess)
			{
				printf("Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			float elaps_kd = 0.0f;
			error = cudaEventElapsedTime(&elaps_kd, startKd, stopKd);

			if (error != cudaSuccess)
			{
				printf("Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF STOP KDTREE TIMING /////////////////////////////
			printf("#Building kd tree took %.3f ms\n", elaps_kd);

			for (int i = 0; i < GPU_N; i++) {
				printf("Device %d start %d end %d\n", i, dPoints[i].start, dPoints[i].end);

			}

			if (Hoption == -1) { // use bandwidth provided in file
				if (UPDATEWEIGHTS)
				{	
					auto t1 = high_resolution_clock::now();
					//EDIT: Timothy @ 12/29/2020
					//Run Kernel Asynchronously accross GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						//printf("Current Device: %d\n", i);
						//Alg Step: 1
						CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
						//cudaStreamSynchronize(streams[i]);
					}
					ReformECWeights(dWeights, gedgeWeights);

					auto t2 = high_resolution_clock::now();
					/* Getting number of milliseconds as a double. */
					duration<double, std::milli> ms_double = t2 - t1;
					printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
				}
			}
			else{
				if (UPDATEWEIGHTS)
				{	
					auto t1 = high_resolution_clock::now();
					//EDIT: Timothy @ 12/29/2020
					//Run Kernel Asynchronously accross GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						//printf("Current Device: %d\n", i);
						//Alg Step: 1
						CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (h * h, dPoints[i], dAscii[i], dWeights[i]);
						//cudaStreamSynchronize(streams[i]);
					}
					ReformECWeights(dWeights, gedgeWeights);

					auto t2 = high_resolution_clock::now();
					/* Getting number of milliseconds as a double. */
					duration<double, std::milli> ms_double = t2 - t1;
					printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
				}
				//cudaSetDevice(GPU_START); //Reset device to first GPU
				//printf("mGPU Kernel...\n"); //DEBUGGING
				//cudaSetDevice(GPU_START); //Reset device to first GPU
				// Guiming @ 2016-03-17
				/////////////////////////////////////////////////////////////////////////////////////////
				int numPoints = sPoints.numberOfPoints;

				//for (int i = 0; i < 3; i++) {
				//	Hoption = i;

				//	if (false && Hoption == 1) {
				//		ComputeNearestNeighborDistances(minNNDist, maxNNDist, meanNNDist);
						//printf("nearest neighbor distances: mean = %f, min = %f, max = %f\n", meanNNDist, minNNDist, maxNNDist);
				//	}

					//////////////////////////////////////////////////
				if (Hoption == 1) {
					float hopt = MLE_FixedBandWidth(dAscii, dPoints, dWeights, h, NULL, den1, true, NULL, dDen1);

					// inat 2019 hopt = 3086.36523
					//float hopt = 3086.36523;

					// inat 2020 hopt = 3487.82935
					//float hopt = 3487.82935;

					printf("cross validated optimal fixed bandwidth hopt: %.5f\n", hopt);

					// kind of combusome
					//Timothy @ 02/19/2020
					//Running following kernels accross all GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						CalcVaryingBandwidths << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dPoints[i], hopt, dHs[i]);
					}
					/*
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);
						cudaStreamSynchronize(streams[i]);
					}*/
					//cudaStreamStatus();
					ReformBandwidths(dHs, hs);


					if (UPDATEWEIGHTS)
					{	
						auto t1 = high_resolution_clock::now();
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);

							int pNum = dPoints[i].end - dPoints[i].start;
							int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
							dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

							CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
						}
						/*
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							cudaStreamSynchronize(streams[i]);
						}*/
						//cudaStreamStatus();
						ReformECWeights(dWeights, gedgeWeights);

						auto t2 = high_resolution_clock::now();
						/* Getting number of milliseconds as a double. */
						duration<double, std::milli> ms_double = t2 - t1;
						printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
					}


					//cudaSetDevice(GPU_START); //Reset device to first GPU
				}

				if (Hoption == 2) {

					if (KNN == 0) {

						float h0 = h;
						float alpha0 = -0.5;
						float stepH = h0 / 5;
						float stepA = 0.1;
						float* optParas = (float*)malloc(3 * sizeof(float));

						///*
						hooke_jeeves(dAscii, dPoints, dPoints, dWeights, h0, alpha0, stepH, stepA, optParas, hs, den0, den1, true, dHs, dDen0, dDen1, dDen0cpy);
						h0 = optParas[0];
						alpha0 = optParas[1];
						float logL = optParas[2];
						//*/

						// inat 2019 h0: 13482.11816 alpha0: -2.55000 Lmax: -160438752.00000
						//h0 = 13482.11816;
						//alpha0 = -2.55000;
						//float logL = -160438752.000;

						// inat h0: 2020 3808.95190 alpha0: -0.01250 Lmax: -241185664.00000
						//h0 = 3808.95190;
						//alpha0 = -0.01250;
						//float logL = -241185664.00000;

						if (DEBUG) printf("h0: %.5f alpha0: %.5f Lmax: %.5f\n", h0, alpha0, logL);
						free(optParas);
						optParas = NULL;
						//exit(1);

						//cudaStreamStatus();

						ComputeFixedDensityDevice(streams, dAscii, dPoints, dWeights, h0, den0, dDen0);

						//cudaStreamStatus();

						//printf("***after - reforming den0 Ab\n");
						//ReformDensities(gpuDen, den0);

						//cudaSynchronizeStreams();
						//printf("***after\n");
						//printDdensities(gpuDen);

						//ReformDensities(dDen0, den0);


						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							CopyDeviceDen(dDen0cpy[i], dDen0[i], numPoints);
						}

						//cudaSetDevice(GPU_START);		
						//ReductionSumGPU_V0(dDen0cpy[0], numPoints);				
						//cudaMemcpyFromSymbol(&reductionSum, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);
						reductionSum = ReductionSumGPU(dDen0cpy, numPoints);
						printf("reductionSum=%f\n", reductionSum);
						//cudaStreamSynchronize(streams[0]);

						// update bandwidth on GPU
						//Timothy @ 02/19/2020
						//Running following kernels accross all GPUs
						for (int i = 0; i < GPU_N; i++)
						{
							printf("...CalcVaryingBandwidths on device %d\n", i + GPU_START);
							cudaSetDevice(i + GPU_START);

							int pNum = dPoints[i].end - dPoints[i].start;
							int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
							dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
							//printf("dPoints[i].end=%d dPoints[i].start=%d pNum=%d, NBLOCK_W=%d, GRID_SIZE_W=%d\n", dPoints[i].end, dPoints[i].start, pNum, NBLOCK_W, GRID_SIZE_W);
							//exit(0);

							CalcVaryingBandwidths << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dPoints[i], dDen0[i], h0, alpha0, dHs[i], reductionSum);
						}
						/*
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							cudaStreamSynchronize(streams[i]);
						}*/

						ReformBandwidths(dHs, hs);
					}
					else { // use knn distance as the adaptive bandwidth
						for (int p = 0; p < numPoints; p++) {
							hs[p] = KNN_coef * knndistance[p];
						}
						CopyToDeviceBandwidths(dHs, hs, numPoints);
					}
					// update weights
					//CopyToDeviceBandwidths(dHs, hs, numPoints);
					if (UPDATEWEIGHTS) {
						auto t1 = high_resolution_clock::now();
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);

							int pNum = dPoints[i].end - dPoints[i].start;
							int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
							dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

							CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
						}
						/*
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							cudaStreamSynchronize(streams[i]);
						}*/
						ReformECWeights(dWeights, gedgeWeights);

						auto t2 = high_resolution_clock::now();
						/* Getting number of milliseconds as a double. */
						duration<double, std::milli> ms_double = t2 - t1;
						printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
					}

					//cudaSetDevice(GPU_START); //Reset device to first GPU
				}
			}
			//Reform data
			//ReformPoints(dPoints); // Guiming 2021-08-14 ReformPoints() is unnecessary
			//ReformBandwidths(dHs, hs);
			//ReformECWeights(dWeights, gedgeWeights);
			//printf("dHs\n");
			//printDdensities(dWeights);

			//cudaStreamSynchronize(streams[0]);
			//cudaSetDevice(GPU_START); //Reset device to first GPU

			//printf("Done...\n\n");

/////////////////////////////////////////////START::::KERNEL DENSITY ESTIMATION IN TILE FASHION/////////////////////////////////////////////////////////////

			// set cell densities in dAscii to 0.0f
			auto t1 = high_resolution_clock::now();
			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				// invoke kernel to do density estimation
				size_t n_cells = dAscii[i].nRows * dAscii[i].nCols;
				int NBLOCK_K = (n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
				int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
				dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);
				InitCellDensities << <dimGrid_K, BLOCK_SIZE, 0, streams[i] >> > (dAscii[i]);
			}

			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				cudaStreamSynchronize(streams[i]);
			}


			/////////////////////////////////////////////////////////////////////////////////
			for (int i = 0; i < GPU_N; i++)
			{
				printf("...KernelDesityEstimation on device %d\n", i + GPU_START);
				cudaSetDevice(i + GPU_START);
				
				if(sPoints.numberOfPoints > dAscii[i].nCols * dAscii[i].nRows){
					// invoke kernel to do density estimation
					int NBLOCK_K = (dAscii[i].end - dAscii[0].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
					int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
					dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);

					KernelDesityEstimation_pRaster <<<dimGrid_K, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
				}
				
				else {
					// invoke kernel to do density estimation
					int NBLOCK_K = (dPoints[i].end - dPoints[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
					int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
					dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);

					KernelDesityEstimation_pPoints <<<dimGrid_K, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
				}

			}

			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				cudaStreamSynchronize(streams[i]);
			}

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a double. */
			duration<double, std::milli> ms_double = t2 - t1;
			printf("...KernelDesityEstimation took %f ms\n", ms_double.count());

			//cudaStreamSynchronize(streams[0]);
			cudaSetDevice(GPU_START);
			///////////////////////// STOP GPU EXCLUSIVE TIMING /////////////////////////////
			// Record the stop event
			error = cudaEventRecord(stopExc, NULL);

			if (error != cudaSuccess)
			{
				printf("Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Wait for the stop event to complete
			error = cudaEventSynchronize(stopExc);
			if (error != cudaSuccess)
			{
				printf("Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			elaps_exc = 0.0f;
			error = cudaEventElapsedTime(&elaps_exc, startExc, stopExc);

			if (error != cudaSuccess)
			{
				printf("Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF STOP GPU EXCLUSIVE TIMING /////////////////////////////

			//ReformAsciiRaster(dAscii, DenSurf_CUDA);
			ReformGPUAsciiRaster(dAscii, DenSurf_CUDA);

			// copy results back to host
			//CopyFromDeviceAsciiRaster(DenSurf_CUDA, dAscii[0]);
			//cudaStreamSynchronize(streams[0]);
			///////////////////////// STOP GPU INCLUSIVE TIMING /////////////////////////////
			// Record the stop event
			error = cudaEventRecord(stopInc, NULL);

			if (error != cudaSuccess)
			{
				printf("Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			// Wait for the stop event to complete
			error = cudaEventSynchronize(stopInc);
			if (error != cudaSuccess)
			{
				printf("Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			elaps_inc = 0.0f;
			error = cudaEventElapsedTime(&elaps_inc, startInc, stopInc);

			if (error != cudaSuccess)
			{
				printf("Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			///////////////////////// END OF STOP GPU INCLUSIVE TIMING /////////////////////////////
			printf("Computation on GPU took %.3f ms (EXCLUSIVE)\n", elaps_exc);
			printf("Computation on GPU took %.3f ms (INCLUSIVE)\n", elaps_inc);

			if (SKIPSEQ == 0) {
				printf("SPEEDUP: %.3fx (EXCLUSIVE) %.3fx (INCLUSIVE)\n", elaps_seq / elaps_exc, elaps_seq / elaps_inc);
				// check resutls
				CheckResults(DenSurf, DenSurf_CUDA);
			}
			// write results to file
			//WriteAsciiRaster(&DenSurf_CUDA, denCUDAfn);
			
			char num_char[5 + sizeof(char)];
			sprintf(num_char, "%d", Hoption);
			
			char fntif[100];
			char fncsv[100];

			sprintf(fntif, "%s", "");
			sprintf(fncsv, "%s", "");

			const char* exttif = ".tif";
			const char* extcsv = ".csv";			
			
			strcat(fntif, (const char*)denCUDAfn);
			strcat(fntif, (const char*)num_char);
			strcat(fntif, (const char*)exttif);

			strcat(fncsv, (const char*)denCUDAfn);
			strcat(fncsv, (const char*)num_char);
			strcat(fncsv, (const char*)extcsv);

			printf("denCUDAfn = %s\n", fntif);
			WriteGeoTIFFRaster(&DenSurf_CUDA, (const char*)fntif);
			//WriteGeoTIFFRaster(&DenSurf_CUDA, (const char*)denCUDAfn);

/////////////////////////////////////////////END::::KERNEL DENSITY ESTIMATION IN TILE FASHION/////////////////////////////////////////////////////////////
			
			WriteSamplePoints(&sPoints, hs, gedgeWeights, (const char*)fncsv);
			//WriteSamplePoints(&sPoints, hs, gedgeWeights, "pntsCUDA.csv");
		//}
/////////////////////////////////////////////////////////////////////////////
		
			t1 = high_resolution_clock::now();
			// clean up
			FreeDeviceSamplePoints(dPoints);
			FreeDeviceEdgeCorrectionWeights(dWeights);
			FreeDeviceAsciiRaster(dAscii);
			FreeAsciiRaster(&DenSurf_CUDA);			
			FreeDeviceBandwidths(dHs);
			FreeDeviceDen(dDen0);
			FreeDeviceDen(dDen0cpy);
			FreeDeviceDen(dDen1);
			FreeDen(zeroDen);
			FreeDeviceDen(gpuDen);
			t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a double. */
			ms_double = t2 - t1;
			printf("...cleaning up took %f ms\n", ms_double.count());
			
		}
		t1 = high_resolution_clock::now();
		//delete streams;
		//delete GPU_tree;

		FreeDen(den0);
		FreeDen(den1);

		FreeAsciiRaster(&DenSurf);
		FreeAsciiRaster(&Mask);
		FreeSamplePoints(&sPoints);
		// By Guiming @ 2016-09-02
		//free(sPoints.distances);
		//sPoints.distances = NULL;

		FreeBandwidths(hs);
		FreeEdgeCorrectionWeights(gedgeWeights);
		FreeDen(knndistance);

		t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a double. */
		ms_double = t2 - t1;
		printf("...cleaning up took %f ms\n", ms_double.count());

		//printf("MAX_N_NBRS=%d\n", MAX_N_NBRS);
		printf("Done...\n\n");

		auto T2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a double. */
		duration<double, std::milli> MS_DOUBLE = T2 - T1;
		printf("...In total it took %f ms\n", MS_DOUBLE.count());
	}
	catch (const std::exception& ex)
	{
		return 1;
	}
	return 0;
}

// mean center of points
void MeanCenter(SamplePoints Points, float &mean_x, float& mean_y){
	float sum_x = 0.0;
	float sum_y = 0.0;
	float sum_w = 0.0;

	for (int p = 0; p < Points.numberOfPoints; p++){
		float w_p = Points.weights[p];
		sum_w += w_p;
		sum_x += w_p * Points.xCoordinates[p];
		sum_y += w_p * Points.yCoordinates[p];
	}

	mean_x = sum_x / sum_w;
	mean_y = sum_y / sum_w;

}

// standard distance squared
void StandardDistance2(SamplePoints Points, float &d2){

	float mean_x, mean_y;
	MeanCenter(Points, mean_x, mean_y);

	float sum2 = 0.0;
	float sum_w = 0.0;
	for (int p = 0; p < Points.numberOfPoints; p++){
		float w_p = Points.weights[p];
		sum_w += w_p;
		sum2 += w_p * Distance2(mean_x, mean_y, Points.xCoordinates[p], Points.yCoordinates[p]);
	}

	d2 = sum2 / sum_w;
}

// generate random sample points
SamplePoints AllocateSamplePoints(int n){
	SamplePoints Points;

	Points.numberOfPoints = n;
	Points.start = 0;
	Points.end = n;
	size_t size = n*sizeof(float);

	Points.xCoordinates = (float*)malloc(size);
	Points.yCoordinates = (float*)malloc(size);
	Points.weights = (float*)malloc(size);
	Points.distances = (float*)malloc(size); // By Guiming @ 2016-09-02

	for (int i = 0; i < n; i++)
	{
		Points.xCoordinates[i] = rand() * 100.0f / RAND_MAX;
		Points.yCoordinates[i] = rand() * 100.0f / RAND_MAX;
		Points.weights[i] = 1.0f;
		Points.distances[i] = 0.0f; // By Guiming @ 2016-09-02
		//printf("x:%.2f y:%.2f w:%.2f\n", Points.xCoordinates[i], Points.yCoordinates[i], Points.weights[i]);
	}
	return Points;
}

// points read from a .csv file
SamplePoints ReadSamplePoints(const char *csvFile, bool bandwidths){
	FILE *f = fopen(csvFile, "rt");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	const int CSV_LINE_LENGTH = 256;
	SamplePoints Points;
	int n = 0;
	char line[CSV_LINE_LENGTH];
	char ch;

	while (!feof(f))
	{
		ch = fgetc(f);
		if (ch == '\n')
		{
			n++;
		}
	}

	if (n == 1){
		printf("No point in file!\n");
		exit(1);
	}

	n = n - 1; // do not count the header line
	Points.numberOfPoints = n;
	Points.xCoordinates = (float*)malloc(n*sizeof(float));
	Points.yCoordinates = (float*)malloc(n*sizeof(float));
	Points.weights = (float*)malloc(n*sizeof(float));
	Points.distances = (float*)malloc(n*sizeof(float)); // By Guiming @ 2016-09-02

	int counter = 0;
	char * pch;
	float x, y, h, w;
	rewind(f); // go back to the beginning of file
	fgets(line, CSV_LINE_LENGTH, f); //skip the header line
	while (fgets(line, CSV_LINE_LENGTH, f) != NULL){
		pch = strtok(line, ",\n");
		x = atof(pch);
		while (pch != NULL)
		{
			pch = strtok(NULL, ",\n");
			y = atof(pch);

			if (bandwidths) {
				
				pch = strtok(NULL, ",\n");
				h = atof(pch);

				pch = strtok(NULL, ",\n");
				if (pch == NULL) {
					printf("***Error - No bandwidth provided in the file\n. Exiting...");
					exit(1);
				}

				w = atof(pch);
			}
			else {
				pch = strtok(NULL, ",\n");
				w = atof(pch);
			}

			break;
		}
		Points.xCoordinates[counter] = x;
		Points.yCoordinates[counter] = y;
		if (SAMPLEWEIGHTS) 
			Points.weights[counter] = w;
		else Points.weights[counter] = 1.0f;
		
		if (bandwidths) { // By Guiming @ 2021-09-10
			Points.distances[counter] = h;
		}
		else{
			Points.distances[counter] = 0.0f; // By Guiming @ 2016-09-02
		}

		counter++;
	}

	fclose(f);

	return Points;
}

/*
// points read from a .csv file: x, y, h, w
SamplePoints ReadSamplePoints(const char* csvFile, float* hs) {
	FILE* f = fopen(csvFile, "rt");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	const int CSV_LINE_LENGTH = 256;
	SamplePoints Points;
	int n = 0;
	char line[CSV_LINE_LENGTH];
	char ch;

	while (!feof(f))
	{
		ch = fgetc(f);
		if (ch == '\n')
		{
			n++;
		}
	}

	if (n == 1) {
		printf("No point in file!\n");
		exit(1);
	}

	n = n - 1; // do not count the header line
	Points.numberOfPoints = n;
	Points.xCoordinates = (float*)malloc(n * sizeof(float));
	Points.yCoordinates = (float*)malloc(n * sizeof(float));
	Points.weights = (float*)malloc(n * sizeof(float));
	Points.distances = (float*)malloc(n * sizeof(float)); // By Guiming @ 2016-09-02

	// it seems this memory alloc is invisible to the caller
	// a workaround is to borrow Points.distances as a temporary container
	//hs = (float*)malloc(n * sizeof(float));

	int counter = 0;
	char* pch;
	float x, y, h, w;
	rewind(f); // go back to the beginning of file
	fgets(line, CSV_LINE_LENGTH, f); //skip the header line
	while (fgets(line, CSV_LINE_LENGTH, f) != NULL) {
		pch = strtok(line, ",\n");
		x = atof(pch);
		while (pch != NULL)
		{
			pch = strtok(NULL, ",\n");
			y = atof(pch);
			
			pch = strtok(NULL, ",\n");
			h = atof(pch);

			pch = strtok(NULL, ",\n");
			if (pch == NULL) {
				printf("***Error - No bandwidth provided in the file\n. Exiting...");
				exit(1);
			}
			
			w = atof(pch);
			break;
		}
		Points.xCoordinates[counter] = x;
		Points.yCoordinates[counter] = y;
		if (SAMPLEWEIGHTS)
			Points.weights[counter] = w;
		else Points.weights[counter] = 1.0f;
		Points.distances[counter] = h; // By Guiming @ 2021-09-10
		
		//hs[counter] = h;
		//printf("hs[%d] = %f\n", counter, hs[counter]);

		counter++;
	}

	fclose(f);

	return Points;
}
*/

void AllocateDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints Points){
	//Timothy @ 01/15/2021
	//EDIT: Changing dPoints to be a array of pointers to each set of points on each device.
	for (int i = 0; i < GPU_N; i++)
	{
		dPoints[i] = Points;
	
		dPoints[i].numberOfPoints = Points.numberOfPoints;
		size_t size = Points.numberOfPoints * sizeof(float);
		cudaError_t error;

		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dPoints[i].xCoordinates, size);
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc((void**)&dPoints[i].yCoordinates, size);
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc((void**)&dPoints[i].weights, size);
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		// By Guiming @ 2016-09-02
		error = cudaMalloc((void**)&dPoints[i].distances, size);
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

//original
//void CopyToDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints hPoints) {
//	size_t size = hPoints.numberOfPoints * sizeof(float);
//
//	//for(int i = 0; i < hPoints.numberOfPoints; i++)
//	//	printf("x:%.2f y:%.2f w:%.2f\n", hPoints.xCoordinates[i], hPoints.yCoordinates[i], hPoints.weights[i]);
//
//	//printf("copy %d points to device\n", size);
//	cudaError_t error;
//
//	error = cudaMemcpy(dPoints[0].xCoordinates, hPoints.xCoordinates, size, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("ERROR in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
//	error = cudaMemcpy(dPoints[0].yCoordinates, hPoints.yCoordinates, size, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("ERROR in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
//	error = cudaMemcpy(dPoints[0].weights, hPoints.weights, size, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("ERROR in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
//
//	// By Guiming @ 2016-09-02
//	error = cudaMemcpy(dPoints[0].distances, hPoints.distances, size, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess)
//	{
//		printf("ERROR in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
//		exit(EXIT_FAILURE);
//	}
//}


//EDIT: Timothy @ 03/26/2021
//Added additional variable to track division of points across multiple GPUs
void CopyToDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints hPoints) {
	size_t size = hPoints.numberOfPoints * sizeof(float);
	int n = hPoints.numberOfPoints; //Number of points on GPU
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	int divNum = 0; //Tracks our place in the original n number of points
	int index = 0; //Tracks indexing for our multiple GPUs
	cudaError_t error;
	dPoints[0].end = div;

	//Timothy @ 01/15/2020
	//Copying the points to each GPU so the data is present across all devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START); //Set device (GPU) being actively copied to
		dPoints[i].start = index; //Begin tracking division of points
		dPoints[i].end = index + div; //Tracking end of division

		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((i == GPU_N - 1) && (rem != 0)) 
		{
			div += rem;
		}
		dPoints[i].numberOfPoints = n; //# of points is assigned here to compensate for remainders.
		error = cudaMemcpy(dPoints[i].xCoordinates, hPoints.xCoordinates, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 1 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(dPoints[i].yCoordinates, hPoints.yCoordinates, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(dPoints[i].weights, hPoints.weights, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 3 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(dPoints[i].distances, hPoints.distances, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 4 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		index = div; //Set starting index of next group of sample points to the end of previous group.
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyFromDeviceSamplePoints(SamplePoints hPoints, const SamplePoints* dPoints){
	size_t size = dPoints[0].numberOfPoints * sizeof(float);
	cudaError_t error;

	cudaSetDevice(GPU_START);

	error = cudaMemcpy(hPoints.xCoordinates, dPoints[0].xCoordinates, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("ERROR 1 in CopyFromDeviceSamplePoints: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(hPoints.yCoordinates, dPoints[0].yCoordinates, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
    {
        printf("ERROR 2 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(hPoints.weights, dPoints[0].weights, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("ERROR 3 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	// By Guiming @ 2016-09-02
	error = cudaMemcpy(hPoints.distances, dPoints[0].distances, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
	    printf("ERROR 4 in CopyToDeviceSamplePoints: %s\n", cudaGetErrorString(error));
	    exit(EXIT_FAILURE);
	}
}

// write to .csv file
void WriteSamplePoints(SamplePoints* Points, const char * csvFile){
	FILE *f = fopen(csvFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "x, y\n");
	for (int p = 0; p < Points->numberOfPoints; p++){
		fprintf(f, "%f, %f\n", Points->xCoordinates[p], Points->yCoordinates[p]);
	}
	fclose(f);
}

// write to .csv file
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, const char * csvFile){
	FILE *f = fopen(csvFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "x, y, h, w\n");
	for (int p = 0; p < Points->numberOfPoints; p++){
		fprintf(f, "%f, %f, %f, %f\n", Points->xCoordinates[p], Points->yCoordinates[p], Hs[p], Ws[p]);
	}
	fclose(f);
}

/*
// write to .csv file
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, float* KNNdist, const char* csvFile) {
	FILE* f = fopen(csvFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "x, y, h, w knndist\n");
	for (int p = 0; p < Points->numberOfPoints; p++) {
		fprintf(f, "%f, %f, %f, %f, %f\n", Points->xCoordinates[p], Points->yCoordinates[p], Hs[p], Ws[p], KNNdist[p]);
	}
	fclose(f);
}
*/

void FreeSamplePoints(SamplePoints* Points) {
	free(Points->xCoordinates);
	Points->xCoordinates = NULL;

	free(Points->yCoordinates);
	Points->yCoordinates = NULL;

	free(Points->weights);
	Points->weights = NULL;
	
	// By Guiming @ 2016-09-02
	free(Points->distances);
	Points->distances = NULL;
}

void FreeDeviceSamplePoints(SamplePoints* dPoints){
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(dPoints[i].xCoordinates);
		if (error != cudaSuccess)
		{
			printf("ERROR 1 in FreeDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		dPoints->xCoordinates = NULL;

		error = cudaFree(dPoints[i].yCoordinates);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in FreeDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		dPoints->yCoordinates = NULL;

		error = cudaFree(dPoints[i].weights);
		if (error != cudaSuccess)
		{
			printf("ERROR 3 in FreeDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		dPoints->weights = NULL;

		// By Guiming @ 2016-09-02
		error = cudaFree(dPoints[i].distances);
		if (error != cudaSuccess)
		{
			printf("ERROR in FreeDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		dPoints->distances = NULL;
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// this is a mask
AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue){
	AsciiRaster Ascii;

	Ascii.nCols = nCols;
	Ascii.nRows = nRows;
	Ascii.xLLCorner = xLLCorner;
	Ascii.yLLCorner = yLLCorner;
	Ascii.cellSize = cellSize;
	Ascii.noDataValue = noDataValue;
	Ascii.start = 0;
	Ascii.end = nRows * nCols;

	Ascii.geotransform = new double[6];
	Ascii.geotransform[0] = (double)xLLCorner;
	Ascii.geotransform[1] = (double)cellSize;
	Ascii.geotransform[2] = 0;

	Ascii.geotransform[3] = (double)(yLLCorner + cellSize * nRows);
	Ascii.geotransform[4] = 0;
	Ascii.geotransform[5] = (double)cellSize*(-1.0);

	Ascii.projection = NULL;

	size_t size = Ascii.nCols * Ascii.nRows;
	Ascii.elements = (float*)malloc(size * sizeof(float));
	size_t idx;
	for (int row = 0; row < Ascii.nRows; row++){
		for (int col = 0; col < Ascii.nCols; col++){
			//if (row < 2 || col < 2)
			//	Ascii.elements[row * nCols + col] = Ascii.noDataValue;
			//else
			idx = row * nCols + col;
			Ascii.elements[idx] = 0.0f;
		}
	}
	return Ascii;
	
}

// copy a ascii raster
AsciiRaster CopyAsciiRaster(const AsciiRaster anotherAscii){
	AsciiRaster Ascii;

	Ascii.nCols = anotherAscii.nCols;
	Ascii.nRows = anotherAscii.nRows;
	Ascii.xLLCorner = anotherAscii.xLLCorner;
	Ascii.yLLCorner = anotherAscii.yLLCorner;
	Ascii.cellSize = anotherAscii.cellSize;
	Ascii.noDataValue = anotherAscii.noDataValue;
	Ascii.start = anotherAscii.start;
	Ascii.end = anotherAscii.end;

	Ascii.geotransform = NULL;
	Ascii.projection = NULL;

	if (anotherAscii.geotransform != NULL) {
		Ascii.geotransform = (double*)malloc(6 * sizeof(double));
		for (int i = 0; i < 6; i++) {
			Ascii.geotransform[i] = anotherAscii.geotransform[i];
		}
	}

	if (anotherAscii.projection != NULL) {
		int len = strlen(anotherAscii.projection);
		Ascii.projection = (char*)malloc(len * sizeof(char));
		strcpy((char*)Ascii.projection, anotherAscii.projection);
	}

	size_t size = Ascii.nCols * Ascii.nRows * sizeof(float);
	Ascii.elements = (float*)malloc(size);
	size_t idx;
	for (int row = 0; row < Ascii.nRows; row++){
		for (int col = 0; col < Ascii.nCols; col++){
			idx = row * Ascii.nCols + col;
			Ascii.elements[idx] = anotherAscii.elements[idx];
		}
	}

	return Ascii;
}

// ascii raster read from a .asc file
AsciiRaster ReadAsciiRaster(char * asciiFile){
	FILE *f = fopen(asciiFile, "rt");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	AsciiRaster Ascii;

	const int HEADER_LINE_LENGTH = 64;
	char hdrLine[HEADER_LINE_LENGTH];
	char* pch;
	float meta[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

	// read headers
	for (int i = 0; i < 6; i++){
		fgets(hdrLine, HEADER_LINE_LENGTH, f);
		pch = strtok(hdrLine, " \n");
		while (pch != NULL)
		{
			pch = strtok(NULL, "\n");
			meta[i] = atof(pch);
			break;
		}
	}

	Ascii.nCols = (int)meta[0];
	Ascii.nRows = (int)meta[1];
	Ascii.xLLCorner = meta[2];
	Ascii.yLLCorner = meta[3];
	Ascii.cellSize = meta[4];
	Ascii.noDataValue = meta[5];

	Ascii.start = 0;
	Ascii.end = Ascii.nCols * Ascii.nRows;

	Ascii.geotransform = NULL;
	Ascii.projection = NULL;

	Ascii.elements = (float*)malloc(Ascii.nRows * Ascii.nCols * sizeof(float));

	const int DATA_LINE_LENGTH = Ascii.nCols * 32;
	char* datLine = (char*)malloc(DATA_LINE_LENGTH * sizeof(char));

	int row_counter = 0;
	while (fgets(datLine, DATA_LINE_LENGTH, f) != NULL){
		int col_counter = 0;
		pch = strtok(datLine, " \n");
		Ascii.elements[row_counter*Ascii.nCols+col_counter] = atof(pch);
		while (pch != NULL)
		{
			pch = strtok(NULL, " ");
			if (pch != NULL && col_counter < Ascii.nCols - 1){
				col_counter++;
				Ascii.elements[row_counter*Ascii.nCols + col_counter] = atof(pch);
			}
		}
		row_counter++;
	}
	free(datLine);

	fclose(f);

	return Ascii;
}

// ascii raster read from a .asc file
AsciiRaster ReadGeoTIFFRaster(char* geotiffFile) {
	
	AsciiRaster Ascii;

	GeoTIFFReader tiff((const char*)geotiffFile);
	
	int* dims = tiff.GetDimensions();
	Ascii.nCols = dims[0];
	Ascii.nRows = dims[1];

	double* gt = tiff.GetGeoTransform();
	Ascii.geotransform = (double*)malloc(6 * sizeof(double));
	for (int i = 0; i < 6; i++) {
		Ascii.geotransform[i] = gt[i];
	}

	const char* prj = tiff.GetProjection();
	int len = strlen(prj);
	Ascii.projection = (char*)malloc(len * sizeof(char));
	strcpy((char*)Ascii.projection, prj);

	Ascii.xLLCorner = (float)gt[0];
	Ascii.yLLCorner = (float)(gt[3] + dims[0] * gt[5]);
	Ascii.cellSize = (float)gt[1];
	Ascii.noDataValue = (float)tiff.GetNoDataValue();

	Ascii.start = 0;
	Ascii.end = Ascii.nCols * Ascii.nRows;

	Ascii.elements = (float*)malloc(Ascii.nRows * Ascii.nCols * sizeof(float));

	float** data = tiff.GetRasterBand(1);
	for (int row = 0; row < Ascii.nRows; row++) {
		for (int col = 0; col < Ascii.nCols; col++) {
			size_t idx = row * Ascii.nCols + col;
			Ascii.elements[idx] = data[row][col];
		}
	}

	return Ascii;
}

AsciiRaster AsciiRasterFromGeoTIFFTile(double* geotransform, const char* projection, int nrows, int ncols, double nodata, float** data) {
	AsciiRaster Ascii;


	Ascii.nCols = ncols;
	Ascii.nRows = nrows;

	Ascii.geotransform = (double*)malloc(6 * sizeof(double));
	for (int i = 0; i < 6; i++) {
		Ascii.geotransform[i] = geotransform[i];
	}

	int len = strlen(projection);
	Ascii.projection = (char*)malloc(len * sizeof(char));
	strcpy((char*)Ascii.projection, projection);

	Ascii.xLLCorner = (float)geotransform[0];
	Ascii.yLLCorner = (float)(geotransform[3] + ncols * geotransform[5]);
	Ascii.cellSize = (float)geotransform[1];
	Ascii.noDataValue = (float)nodata;

	Ascii.start = 0;
	Ascii.end = Ascii.nCols * Ascii.nRows;

	Ascii.elements = (float*)malloc(Ascii.nRows * Ascii.nCols * sizeof(float));
	
	for (int row = 0; row < Ascii.nRows; row++) {
		for (int col = 0; col < Ascii.nCols; col++) {
			size_t idx = row * Ascii.nCols + col;
			Ascii.elements[idx] = data[row][col];
		}
	}

	return Ascii;
}

void AllocateDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii){
	//Timothy @ 10/16/2020
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		dAscii[i].nCols = hAscii.nCols;
		dAscii[i].nRows = hAscii.nRows;
		dAscii[i].xLLCorner = hAscii.xLLCorner;
		dAscii[i].yLLCorner = hAscii.yLLCorner;
		dAscii[i].cellSize = hAscii.cellSize;
		dAscii[i].noDataValue = hAscii.noDataValue;
		size_t size = hAscii.nCols*hAscii.nRows * sizeof(float);

		printf("size in AllocateDeviceAsciiRaster %llu x 1\n", size);

		dAscii[i].start = 0;
		dAscii[i].end = hAscii.nCols * hAscii.nRows;

		cudaError_t error;
	
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dAscii[i].elements, size);
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceAsciiRaster: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyToDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii){
	
	// Guiming 2021-08-18
	size_t n = hAscii.nCols * hAscii.nRows; //Number of cells on GPU
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of cells evenly
	int div = n / GPU_N; //Division of cells to be divided amongst GPUs
	int divNum = 0; //Tracks our place in the original n number of cells
	int index = 0; //Tracks indexing for our multiple GPUs
	
	size_t size = n * sizeof(float);
	cudaError_t error;
	//Copy raster to all available devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);

		dAscii[i].start = index; //Begin tracking division of points
		dAscii[i].end = index + div; //Tracking end of division

		//If on last GPU, check if GPU_N divided into cells evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((i == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}

		error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR in CopyToDeviceAsciiRaster: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		index = div; //Set starting index of next group of cells to the end of previous group.
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// Guiming 2021-08-18 Combine rasters across GPUs into a single raster and send the update raster back to GPUs
void ReformAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii) {
	
	AsciiRaster tmpAscii = CopyAsciiRaster(hAscii);

	size_t n = hAscii.nCols * hAscii.nRows; //Number of TOTAL cells	
	size_t size = n * sizeof(float);

	cudaError_t error = cudaSuccess;

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);	


		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tmpAscii.elements, dAscii[device].elements, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformAsciiRaster (FROM device): %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (size_t i = dAscii[device].start; i < dAscii[device].end; i++)
		{
			hAscii.elements[i] = tmpAscii.elements[i];
		}
		if (DEBUGREFORMING) printf("......Copying Ascii FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2.%d in ReformAsciiRaster (To device): %s\n", i, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying Ascii TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeAsciiRaster(&tmpAscii);
	if (DEBUGREFORMING) printf("***Reforming Ascii DONE\n");
}


// Guiming 2021-08-30 Add up cell densities from all devices (gpuDen) into one single array
void ReformGPUAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii) {

	AsciiRaster tmpAscii = CopyAsciiRaster(hAscii);

	size_t n = hAscii.nCols * hAscii.nRows; //Number of TOTAL cells
	for (size_t i = 0; i < n; i++) {
		if (hAscii.elements[i] == hAscii.noDataValue) continue;
		hAscii.elements[i] = 0.0f;
	}

	size_t size = n * sizeof(float);

	cudaError_t error = cudaSuccess;

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);

		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tmpAscii.elements, dAscii[device].elements, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformAsciiRaster (FROM device): %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (size_t i = 0; i < n; i++)
		{	if (hAscii.elements[i] == hAscii.noDataValue) continue;
			hAscii.elements[i] += tmpAscii.elements[i];
		}
		if (DEBUGREFORMING) printf("......Copying Ascii FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2.%d in ReformAsciiRaster (To device): %s\n", i, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying Ascii TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeAsciiRaster(&tmpAscii);
	if (DEBUGREFORMING) printf("***Reforming Ascii DONE\n");
}

void CopyFromDeviceAsciiRaster(AsciiRaster hAscii, const AsciiRaster dAscii){
	hAscii.nCols = dAscii.nCols;
	hAscii.nRows = dAscii.nRows;
	hAscii.xLLCorner = dAscii.xLLCorner;
	hAscii.yLLCorner = dAscii.yLLCorner;
	hAscii.cellSize = dAscii.cellSize;
	hAscii.noDataValue = dAscii.noDataValue;

	hAscii.start = dAscii.start;
	hAscii.end = dAscii.end;

	size_t size = dAscii.nCols*dAscii.nRows * sizeof(float);
	cudaError_t error;
	error = cudaMemcpy(hAscii.elements, dAscii.elements, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("ERROR in CopyFromDeviceAsciiRaster: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
        exit(EXIT_FAILURE);
    }
}

// write to .asc file
void WriteAsciiRaster(AsciiRaster* Ascii, const char * asciiFile){

	auto t1 = high_resolution_clock::now();

	FILE *f = fopen(asciiFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "ncols %d\n", Ascii->nCols);
	fprintf(f, "nrows %d\n", Ascii->nRows);
	fprintf(f, "xllcorner %f\n", Ascii->xLLCorner);
	fprintf(f, "yllcorner %f\n", Ascii->yLLCorner);
	fprintf(f, "cellsize %f\n", Ascii->cellSize);
	fprintf(f, "NODATA_value %.0f\n", Ascii->noDataValue);

	for (int row = 0; row < Ascii->nRows; row++){
		for (int col = 0; col < Ascii->nCols; col++){
			fprintf(f, "%.16f ", Ascii->elements[row*Ascii->nCols+col]);
		}
		fprintf(f, "\n");
	}
	fclose(f);

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...writing %s took %f ms\n", asciiFile, ms_double.count());
}


// write to .tif file
void WriteGeoTIFFRaster(AsciiRaster* Ascii, const char* geotiffFile) {
	// reform Ascii->elements to 2D array
	/*float** data = new float*[Ascii->nRows];
	for (int i = 0; i < Ascii->nRows; i++) {
		data[i] = new float[Ascii->nCols];
	}
	for (int row = 0; row < Ascii->nRows; row++) {
		for (int col = 0; col < Ascii->nCols; col++) {
			data[row][col] = Ascii->elements[row * Ascii->nCols + col];
		}
	}*/
	//printf("%d %d\n", Ascii->geotransform == NULL, Ascii->projection == NULL);
	//printf("xLL %f, yLL %f, nRows %d, nCols %d, cellsize %f\n", Ascii->xLLCorner, Ascii->yLLCorner, Ascii->nRows, Ascii->nCols, Ascii->cellSize);
	GeoTIFFWriter tiffw;
	tiffw.WriteGeoTIFF(geotiffFile, Ascii->geotransform, Ascii->projection, (int)Ascii->nRows, (int)Ascii->nCols, (double)Ascii->noDataValue, Ascii->elements);
	//printf("+++\n");
	
	//clean up
	/*
	for (int i = 0; i < Ascii->nRows; i++) {
		delete data[i];
	}
	delete data;
	*/
}


void FreeAsciiRaster(AsciiRaster* Ascii){
	if (Ascii->geotransform != NULL) {
		free(Ascii->geotransform);
		Ascii->geotransform = NULL;
	}
	//printf("in free\n");
	/*
	if (Ascii->projection != NULL) {
		free((void*)(Ascii->projection));
		Ascii->projection = NULL;
	}*/

	free(Ascii->elements);
	Ascii->elements = NULL;
}

void FreeDeviceAsciiRaster(AsciiRaster* Ascii){
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(Ascii[i].elements);
		if (error != cudaSuccess)
		{
			printf("ERROR in FreeDeviceAsciiRaster: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		Ascii[i].elements = NULL;
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// edge effects correction weights at each point, weights is allocated somewhere else
void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights){
	float h2 = h * h;
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew;

	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);

		// By Guiming @ 2016-09-03
		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
			continue;
		}

		p_x = Points.xCoordinates[p];
		p_y = Points.yCoordinates[p];
		ew = 0.0f;

		// added by Guiming @2016-09-11
		// narrow down the row/col range
		int row_lower = 0;
		int row_upper = Ascii.nRows - 1;
		int col_lower = 0;
		int col_upper = Ascii.nCols - 1;
		if(NARROW){
			int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * h, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
			row_lower = MAX(0, r);
			row_upper = MIN(Ascii.nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * h, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize));
			col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * h, Ascii.xLLCorner, Ascii.cellSize));
			col_upper = MIN(Ascii.nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * h, Ascii.xLLCorner, Ascii.cellSize));
		}

		for (int row = row_lower; row <= row_upper; row++){
			for (int col = col_lower; col <= col_upper; col++){
				size_t idx = row * Ascii.nCols + col;
				if (Ascii.elements[idx] != Ascii.noDataValue){
					cell_x = COL_TO_XCOORD(col, Ascii.xLLCorner, Ascii.cellSize);
					cell_y = ROW_TO_YCOORD(row, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
					float d2 = Distance2(p_x, p_y, cell_x, cell_y);
					ew += GaussianKernel(h2, d2) * cellArea;
				}
			}
		}
		weights[p] = 1.0 / ew;
	}
}

void EdgeCorrectionWeightsExact(SamplePoints Points, float* hs, AsciiRaster Ascii, float *weights){
	//float h2 = BandWidth2(Points);
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew, h2;

	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);
		p_x = Points.xCoordinates[p];
		p_y = Points.yCoordinates[p];
		ew = 0.0f;
		h2 = hs[p] * hs[p];

		// By Guiming @ 2016-09-03
		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
			continue;
		}

		// added by Guiming @2016-09-11
		// narrow down the row/col range
		int row_lower = 0;
		int row_upper = Ascii.nRows - 1;
		int col_lower = 0;
		int col_upper = Ascii.nCols - 1;

		if(NARROW){
			int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * hs[p], Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
			row_lower = MAX(0, r);
			row_upper = MIN(Ascii.nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * hs[p], Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize));
			col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * hs[p], Ascii.xLLCorner, Ascii.cellSize));
			col_upper = MIN(Ascii.nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * hs[p], Ascii.xLLCorner, Ascii.cellSize));
		}

		for (int row = row_lower; row <= row_upper; row++){
			for (int col = col_lower; col <= col_upper; col++){
				size_t idx = row * Ascii.nCols + col;
				if (Ascii.elements[idx] != Ascii.noDataValue){
					cell_x = COL_TO_XCOORD(col, Ascii.xLLCorner, Ascii.cellSize);
					cell_y = ROW_TO_YCOORD(row, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
					float d2 = Distance2(p_x, p_y, cell_x, cell_y);
					ew += GaussianKernel(h2, d2) * cellArea;
				}
			}
		}
		weights[p] = 1.0 / ew;
	}
}

float* AllocateEdgeCorrectionWeights(SamplePoints Points){
	
	float* ecweights = (float*)malloc(Points.numberOfPoints * sizeof(float));
	for (int i = 0; i < Points.numberOfPoints; i++) {
		ecweights[i] = 1.0f;
	}
	return ecweights;	

	//return (float*)malloc(Points.numberOfPoints * sizeof(float));

}

void AllocateDeviceEdgeCorrectionWeights(float** dWeights, SamplePoints Points){
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dWeights[i], Points.numberOfPoints * sizeof(float));
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceEdgeCorrectionWeights: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}


void CopyToDeviceWeights(float** dWeights, const float* hWeights, const int n) {
	size_t size = n * sizeof(float);
	cudaError_t error;

	//Copy to each available device
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dWeights[i], hWeights, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR in CopyToDeviceBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START);
}

void FreeEdgeCorrectionWeights(float* weights){
	
	free(weights);
	weights = NULL;
}

void FreeDeviceEdgeCorrectionWeights(float** weights){
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(weights[i]);
		if (error != cudaSuccess)
		{
			printf("ERROR in FreeDeviceEdgeCorrectionWeights: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		weights[i] = NULL;
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

///////// Guiming on 2016-03-16 ///////////////
// the array holding bandwidth at each point
float* AllocateBandwidths(int n){ // n is number of points
	return (float*)malloc(n*sizeof(float));
}

void AllocateDeviceBandwidths(float** dBandwidths, int n){ // n is number of points
	cudaError_t error;
	//Allocate bandwidth accross all available devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dBandwidths[i], n * sizeof(float));
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyToDeviceBandwidths(float** dBandwidth, const float* hBandwidths, const int n) {
	size_t size = n * sizeof(float);
	cudaError_t error;
	
	//Copy to each available device
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dBandwidth[i], hBandwidths, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR in CopyToDeviceBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START);
}

void CopyFromDeviceBandwidths(float* hBandwidth, const float* dBandwidths, const int n){
	size_t size = n * sizeof(float);
	cudaError_t error;
	error = cudaMemcpy(hBandwidth, dBandwidths, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("ERROR in CopyFromDeviceBandwidths: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
        exit(EXIT_FAILURE);
    }
}

void FreeDeviceBandwidths(float** bandwidths){
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(bandwidths[i]);
		if (error != cudaSuccess)
		{
			printf("ERROR in FreeDeviceBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		bandwidths[i] = NULL;
	}
	delete bandwidths;
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void FreeBandwidths(float* bandwidths){
	free(bandwidths);
	bandwidths = NULL;
}

// the array holding inclusive density at each point
float* AllocateDen(int n){ // n is number of points
	return (float*)malloc(n*sizeof(float));
}

void AllocateDeviceDen(float** dDen, int n){ // n is number of points
	cudaError_t error;
	//Timothy @ 10/16/2020
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dDen[i], n * sizeof(float));
		if (error != cudaSuccess)
		{
			printf("ERROR in AllocateDeviceDen: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyToDeviceDen(float** dDen, const float* hDen, const int n){
	size_t size = n * sizeof(float);
	cudaError_t error;
	//Copy accross all available devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dDen[i], hDen, size, cudaMemcpyHostToDevice);
	}
	if (error != cudaSuccess)
    {
        printf("ERROR in CopyToDeviceDen: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyFromDeviceDen(float* hDen, const float* dDen, const int n){
	size_t size = n * sizeof(float);
	cudaError_t error;
	error = cudaMemcpy(hDen, dDen, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("ERROR in CopyFromDeviceDen: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
        exit(EXIT_FAILURE);
    }
}

void CopyDeviceDen(float* dDenTo, const float* dDenFrom, const int n){
	size_t size = n * sizeof(float);
	cudaError_t error = cudaSuccess;
	error = cudaMemcpy(dDenTo, dDenFrom, size, cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess)
    {
        printf("ERROR in CopyDeviceDen: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToDevice);
        exit(EXIT_FAILURE);
    }
}
/*
void FreeDeviceDen(float** den){
	cudaError_t error;
	cudaSetDevice(GPU_START);
	error = cudaFree(den[0]);
	if (error != cudaSuccess)
	{
		printf("ERROR in FreeDeviceDen(Elements): %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	den = NULL;
	error = cudaFree(den);
	if (error != cudaSuccess)
	{
		printf("ERROR in FreeDeviceDen(Pointer): %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}*/

void FreeDeviceDen(float** den) {
	cudaError_t error;
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);
		error = cudaFree(den[i]);
		if (error != cudaSuccess)
		{
			printf("ERROR in FreeDeviceDen(Elements): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		den[i] = NULL;
	}
	delete den;
}

void FreeDen(float* den){
	free(den);
	den = NULL;
}

// compute the optimal Maximum Likelihood Estimation fixed bandwidth
// By Guiming @ 2016-02-26
float MLE_FixedBandWidth(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0, float* den1, bool useGPU, float** dDen0, float** dDen1){
	
	//float hA = max(h/50, meanNNDist);
	//float hD = min(1.5*h, maxNNDist);

	float hA = h / 500;
	float hD = h;

	float width = hD - hA;
	float epsilon = width/200;
	float factor = 1 + sqrtf(5.0f);
	int iteration = 0;

	printf("hA: %f hD: %f width: %f, epsilon: %f\n", hA, hD, width, epsilon); //DEBUG
	while(width > epsilon){

		if(DEBUG){
			printf("iteration: %d ", iteration);
			printf("hD: %.6f ", hD);
			printf("hA: %.6f ", hA);
		}

		float hB = hA + width / factor;
		float hC = hD - width / factor;

		/*if (DEBUG) {
			printf("hB: %.6f ", hB);
			printf("hC: %.6f ", hC);
		}*/

		//ERROR HERE, ONLY WHEN USING GPU
		float LoghB = LogLikelihood(Ascii, Points, edgeWeights, hB, den0, den1, useGPU, dDen0, dDen1);
		float LoghC = LogLikelihood(Ascii, Points, edgeWeights, hC, den0, den1, useGPU, dDen0, dDen1);

		/*if (DEBUG) {
			printf("LoghB: %.6f ", LoghB);
			printf("LoghC: %.6f ", LoghC);
		}*/

		if(LoghB > LoghC){
			hD = hC;
			if(DEBUG) printf("LoghB: %.6f \n", LoghB);
		}
		else{
			hA = hB;
			if(DEBUG) printf("LoghC: %.6f \n", LoghC);
		}

		width = hD - hA;

		iteration += 1;
	}

	return (hA + hD) / 2;
}

// By Guiming @ 2016-05-21
// computed fixed bandwidth kde
void ComputeFixedDensityAtPoints(AsciiRaster Ascii, SamplePoints Points, float* edgeWeights, float h, float* den0, float* den1, float* dDen0, float* dDen1) {
	
	int numPoints = Points.numberOfPoints;
		// update edge correction weights
		if (UPDATEWEIGHTS) {
			EdgeCorrectionWeightsExact(Points, h, Ascii, edgeWeights);
		}

		for (int i = 0; i < numPoints; i++) {
			float pi_x = Points.xCoordinates[i];
			float pi_y = Points.yCoordinates[i];

			float den = 0.0f; //EPSILONDENSITY;
			float den_itself = 0.0f;  //EPSILONDENSITY;
			for (int j = 0; j < numPoints; j++) {
				float pj_x = Points.xCoordinates[j];
				float pj_y = Points.yCoordinates[j];
				float pj_w = Points.weights[j];
				float pj_ew = edgeWeights[j];

				float d2 = Distance2(pi_x, pi_y, pj_x, pj_y);

				if (j == i) {
					den_itself += GaussianKernel(h * h, d2) * pj_w * pj_ew; // / numPoints;
				}
				else {
					den += GaussianKernel(h * h, d2) * pj_w * pj_ew;
				}
			}

			if (den0 != NULL) {
				den0[i] = den + den_itself;
			}
			if (den1 != NULL) {
				den1[i] = den;
			}
		}
}

// By Timothy @ 04-23-2021
// Separated this function into two separate, rather than using the boolean value as was done previously.
//this was done to enable parallel functionality accross multiple GPUs
void ComputeFixedDensityDevice(cudaStream_t* streams, AsciiRaster* Ascii, SamplePoints* Points, float** edgeWeights, float h, float* den0, float** dDen0){
	
	// update edge correction weights
	if (UPDATEWEIGHTS) {
		
		auto t1 = high_resolution_clock::now();

		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);

			int pNum = Points[i].end - Points[i].start;
			int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (h * h, Points[i], Ascii[i], edgeWeights[i]);
		}/*
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}*/

		ReformECWeights(edgeWeights, gedgeWeights);
		//cudaStreamStatus();

		auto t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a double. */
		duration<double, std::milli> ms_double = t2 - t1;
		printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
	}


		//printDdensities(gpuDen);
		//// brute force to search for neighbors
		//DensityAtPoints<<<dimGrid_W, BLOCK_SIZE>>>(h*h, Points, edgeWeights, dDen0, dDen1);
		///// use KD Tree to speedup neighor search
		//CopyToDeviceDen(gpuDen, zeroDen, Points.numberOfPoints);
	
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);

		int numPoints = Points[i].numberOfPoints;
		int NBLOCK_W = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

		InitGPUDen << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuDen[i], numPoints);
	}
	for (int i = 0; i < GPU_N; i++)	{
		cudaSetDevice(i + GPU_START);
		cudaStreamSynchronize(streams[i]);
	}

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);

		int pNum = Points[i].end - Points[i].start;
		int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

		DensityAtPointsKdtr << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, Points[i], edgeWeights[i], gpuDen[i]);
	}
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		cudaStreamSynchronize(streams[i]);
		// have to do this as a separate kernel call due to the need of block synchronization !!!
		// this took me hours to debug!
	}
	//cudaStreamStatus();
	//cudaStreamSynchronize(streams[0]);
	//cudaStreamSynchronize(streams[1]);
	//printDdensities(gpuDen);

	ReformGPUDensities(gpuDen, den0);
	//ReformDensities(dDen0, den0);
	//printDdensities(dDen0);

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);

		int pNum = Points[i].end - Points[i].start;
		int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

		dCopyDensityValues << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (Points[i], edgeWeights[i], h * h, gpuDen[i], dDen0[i], NULL);
	}
	/*
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		cudaStreamSynchronize(streams[i]);
	}*/
	//printDdensities(gpuDen);
	//cudaStreamStatus();

	//printf("***inside ComputeFixedDensityDevice, reform densities \n");
	ReformDensities(dDen0, den0);
	//cudaSetDevice(GPU_START);
}

// By Guiming @ 2016-02-26
// the log likelihood given single bandwidth h
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0, float* den1, bool useGPU, float** dDen0, float** dDen1){
	float logL = 0.0f; // log likelihood
	cudaError error = cudaSuccess;

	if (error != cudaSuccess)
	{
		printf("ERROR 0 in LogLikelihood: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	if (useGPU) { // do it on GPU

		if (UPDATEWEIGHTS) {
			
			auto t1 = high_resolution_clock::now();

			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);

				int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
				int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
				dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

				// update edge correction weights			
				CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (h * h, Points[i], Ascii[i], edgeWeights[i]);			
				//cudaStreamSynchronize(streams[i]);
			}
			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a double. */
			duration<double, std::milli> ms_double = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
		}
		
		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);

			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuDen[i], Points[i].numberOfPoints);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}
		
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			DensityAtPointsKdtr << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, Points[i], edgeWeights[i], gpuDen[i]);
			// have to do this as a separate kernel call due to the need of block synchronization !!!
			// this took me hours to debug!
			

		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			int stack_depth;
			cudaMemcpyFromSymbol(&stack_depth, STACK_DEPTH_MAX, sizeof(int), 0, cudaMemcpyDeviceToHost);
			printf("\nSTACK_DEPTH_MAX = %d\n", stack_depth);
			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den1);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);	

			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (Points[i], edgeWeights[i], h * h, gpuDen[i], NULL, dDen1[i]);
		}
		/*
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}*/
		// Guiming 2021-08-14
		//ReformPoints(Points); // Unnecessary	

		ReformDensities(dDen1, den1);

		//cudaSetDevice(GPU_START);
		// compute likelihood on GPU
		///ReductionSumGPU_V0(dDen1[0], Points[0].numberOfPoints);
		///cudaStreamSynchronize(streams[0]);
		///cudaMemcpyFromSymbol(&logL, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);

		logL = ReductionSumGPU(dDen1, Points[0].numberOfPoints);
		
		//printf("\nreduction result (likelihood) A: %3.4f \n", logL);
		
		//CopyToDeviceSamplePoints(Points, sPoints);
		//Cleanup
		//cudaSetDevice(GPU_START);
		/*FreeSamplePoints(&hostP);*/
	}
	else{ // do it on CPU
		int numPoints = Points[0].numberOfPoints;
		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], h, Ascii[0], edgeWeights[0]);
		}

		// the kd tree appraoch
		float* tmpden = AllocateDen(numPoints);
		float h2 = h * h;
		float range = CUT_OFF_FACTOR * h2;

		for(int i = 0; i < numPoints; i++){
			tmpden[i] = -1.0 * GaussianKernel(h2, 0.0f) *  Points[0].weights[i] * edgeWeights[0][i];
		}

		vector<int> ret_index = vector<int>();
		vector<float> ret_dist = vector<float>(); // squared distance

		for(int i = 0; i < numPoints; i++){
			float pi_x = Points[0].xCoordinates[i];
			float pi_y = Points[0].yCoordinates[i];
			float pj_w = Points[0].weights[i];
			float pj_ew = edgeWeights[0][i];

			// range query
			Point query;
			query.coords[0] = pi_x;
			query.coords[1] = pi_y;
			ret_index.clear();
			ret_dist.clear();
			tree.SearchRange(query, range, ret_index, ret_dist);
		
			//printf("CPU PNT_%d %d NBRS RANGE=%.1f\n", i, ret_index.size(), range);

			/*
			tree.SearchKNN(query, 3, ret_index, ret_dist);
			if (i < 2) {
				printf("\nret_index.size() = %d\n", ret_index.size());
				for (int NN = 0; NN < ret_index.size(); NN++) {
					printf("%d %d %f\n", NN, ret_index[NN], ret_dist[NN]);
				}
				printf("\n");

				for (int j = 0; j < numPoints; j++) {
					printf("dist Point %d - Point %d = %f\n", i, j, Distance2(pi_x, pi_y, Points[0].xCoordinates[j], Points[0].yCoordinates[j]));
				}
				printf("\n");
			}
			exit(0);
			*/
			if(ret_index.size() > MAX_N_NBRS) MAX_N_NBRS = ret_index.size();

			float g = 0.0f;
			int idx;
			for(int j = 0; j < ret_index.size(); j++){
					g = GaussianKernel(h2, ret_dist[j]) * pj_w *pj_ew;
					idx = ret_index[j];
					//float t = tmpden[idx];
					tmpden[idx] += g;
					//if(i == 0) printf("CPU PNT_%d g[%d]=%.5f gpuDen[%d]=%.5f gpuDen[%d]=%.5f\n", i, idx, g, idx, t, idx, tmpden[idx]);
			}
		} // END OF COMPUTING DENSITIES AT POINTS



		for(int i = 0; i < numPoints; i++){
			//printf("CPU H2=%.2f DEN[%d]=%.5f\n", h2, i, tmpden[i]);
			logL += logf(max(tmpden[i], EPSILONDENSITY));
		}

		if(den0 != NULL){
			for(int i = 0; i < numPoints; i++)
				den0[i] = tmpden[i]  + GaussianKernel(h2, 0.0f) * Points[0].weights[i] * edgeWeights[0][i];
		}
		if(den1 != NULL){
			for(int i = 0; i < numPoints; i++)
				den1[i] = tmpden[i];
		}

		FreeDen(tmpden);
	}
	return logL;
}

// the log likelihood given bandwidths hs
// By Guiming @ 2016-02-26
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
//EDIT: Timothy @ 03/12/2021
//Added additional variable passed in, so now when useGPU is TRUE, the function will be handled across however many GPUs are available
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float ** edgeWeights, float* hs, float* den0, float* den1, bool useGPU, float** dHs, float** dDen0, float** dDen1, float h, float alpha, float** dDen0cpy){
	
	float logL = 0.0f; // log likelihood
	cudaError_t error = cudaSuccess;

	if(useGPU){ // do it on GPU

		if (UPDATEWEIGHTS) {

			auto t1 = high_resolution_clock::now();

			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				
				// execution config.
				int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
				int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
				dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
			
				CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (h*h, gpuPoints[i], Ascii[i], edgeWeights[i]);
			}
			/*
			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				cudaStreamSynchronize(streams[i]);
			}*/
			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a double. */
			duration<double, std::milli> ms_double = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
		}
		
		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuDen[i], Points[i].numberOfPoints);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}
		

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			DensityAtPointsKdtr << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, gpuPoints[i], edgeWeights[i], gpuDen[i]);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den1);
		/*
		for (int i = 0; i < Points[0].numberOfPoints; i++) {
			//printf("GPU - den[%d] = %f \n", i, den1[i]);
			printf("%f\n", den1[i]);
		}
		printf("\n");
		//exit(0);
		*/

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuPoints[i], edgeWeights[i], h * h, gpuDen[i], dDen0[i], dDen1[i]);
		}
		/*
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}*/
	
		ReformDensities(dDen0, den0);
		//ReformDensities(dDen1, den1);
		 
		/*
		cudaSetDevice(GPU_START);
		CopyDeviceDen(dDen0cpy[0], dDen0[0], Points[0].numberOfPoints);		
		// compute sum of log densities on GPU
		ReductionSumGPU_V0(dDen0cpy[0], Points[0].numberOfPoints);		
		cudaMemcpyFromSymbol(&reductionSum, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);
		*/

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			CopyDeviceDen(dDen0cpy[i], dDen0[i], Points[i].numberOfPoints);
		}
		reductionSum = ReductionSumGPU(dDen0cpy, Points[0].numberOfPoints);
		
		//printf("reduction result (geometricmean): %f \n", exp(reductionSum/ Points[0].numberOfPoints));		
		//exit(0);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			// update bandwidth on GPU
			CalcVaryingBandwidths << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuPoints[i], dDen0[i], h, alpha, dHs[i], reductionSum);
		}
		/*
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}*/
		//printf("***\n");
		//printDdensities(dHs);

		ReformBandwidths(dHs, hs);
		//printHdensities(hs, numPoints);
		//printf("\n");

			// update edge correction weights
		if (UPDATEWEIGHTS) {

			auto t1 = high_resolution_clock::now();

			for (int i = 0; i < GPU_N; i++)
			{	
				cudaSetDevice(i + GPU_START);

				// execution config.
				int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
				int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
				dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

				CalcEdgeCorrectionWeights << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (dHs[i], gpuPoints[i], Ascii[i], edgeWeights[i]);
			}
			/*
			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				cudaStreamSynchronize(streams[i]);
			}*/

			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a double. */
			duration<double, std::milli> ms_double = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %f ms\n", ms_double.count());
		}

		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);
			
			// execution config.
			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuDen[i], Points[i].numberOfPoints);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			DensityAtPointsKdtr << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, dHs[i], gpuPoints[i], edgeWeights[i], gpuDen[i]);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
			// have to do this as a separate kernel call due to the need of block synchronization !!!
			// this took me hours to debug!			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den0);
		//printDdensities(gpuDen);
		/*
		for (int i = 0; i < 10; i++) {
			printf("GPU - den[%d] = %f \n", i, den0[i]);
		}
		printf("\n");
		*/

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuPoints[i], edgeWeights[i], dHs[i], gpuDen[i], dDen0[i], dDen1[i]);
		}
		/*
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}*/
		
		//ReformPoints(gpuPoints);
		// compute likelihood on GPU
		//ReformDensities(dDen0, den0);
		ReformDensities(dDen1, den1);
		
		/*
		cudaSetDevice(GPU_START);	
		ReductionSumGPU_V0(dDen1[0], Points[0].numberOfPoints);
		cudaMemcpyFromSymbol(&logL, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);
		*/

		logL = ReductionSumGPU(dDen1, Points[0].numberOfPoints);

		//printf("reduction result (likelihood): %f \n", logL);

	}
	else{ // do it on CPU

		int numPoints = Points[0].numberOfPoints;

		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], h, Ascii[0], edgeWeights[0]);
		}

		// kdtree approach
		float h2 = h * h;
		float range = CUT_OFF_FACTOR * h2;
		float* denTmp = AllocateDen(numPoints);
		for(int i = 0; i < numPoints; i++){
			denTmp[i] = 0.0f;
		}

		vector<int> ret_index = vector<int>();
		vector<float> ret_dist = vector<float>(); // squared distance
		for(int i = 0; i < numPoints; i++){
			float pi_x = Points[0].xCoordinates[i];
			float pi_y = Points[0].yCoordinates[i];
			float pj_w = Points[0].weights[i];
			float pj_ew = edgeWeights[0][i];

			// range query
			Point query;
			query.coords[0] = pi_x;
			query.coords[1] = pi_y;
			ret_index.clear();
			ret_dist.clear();
			tree.SearchRange(query, range, ret_index, ret_dist);

			if(ret_index.size() > MAX_N_NBRS) MAX_N_NBRS = ret_index.size();

			int nn = ret_index.size();
			float g = 0.0f;
			int idx;
			for(int j = 0; j < ret_index.size(); j++){
					g = GaussianKernel(h2, ret_dist[j]) * pj_w * pj_ew;
					idx = ret_index[j];
					denTmp[idx] += g;
			}
		} // END OF COMPUTING DENSITIES AT POINTS

		/*
		for (int i = 0; i < numPoints; i++) {
			//printf("CPU - den[%d] = %f \n", i, denTmp[i]);
			printf("%f\n", denTmp[i]);
		}
		printf("\n");
		//exit(0);
		*/
		// update bandwidths
		float gml = compGML(denTmp, numPoints);
		
		//printf("reduction result (geometricmean): %f \n", gml);
		//exit(0);

	    for(int i = 0; i < numPoints; i++){
			hs[i] = h * powf((denTmp[i] / gml), alpha);
			//printf("%d - %f, %f \n", i, denTmp[i], hs[i]);
	    }
		//printf("\n");


		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], hs, Ascii[0], edgeWeights[0]);
		}

		for(int i = 0; i < numPoints; i++){
			float h2 = hs[i] * hs[i];
			denTmp[i] = -1.0 * GaussianKernel(h2, 0.0f) *  Points[0].weights[i] * edgeWeights[0][i];
		}

		for(int i = 0; i < numPoints; i++){
			float pi_x = Points[0].xCoordinates[i];
			float pi_y = Points[0].yCoordinates[i];
			float pj_w = Points[0].weights[i];
			float pj_ew = edgeWeights[0][i];
			float h2 = hs[i] * hs[i];
			float range = CUT_OFF_FACTOR * h2;

			// range query
			Point query;
			query.coords[0] = pi_x;
			query.coords[1] = pi_y;
			ret_index.clear();
			ret_dist.clear();
			tree.SearchRange(query, range, ret_index, ret_dist);

			if(ret_index.size() > MAX_N_NBRS) MAX_N_NBRS = ret_index.size();

			int nn = ret_index.size();
			float g = 0.0f;
			int idx;
			for(int j = 0; j < ret_index.size(); j++){
					g = GaussianKernel(h2, ret_dist[j]) * pj_w * pj_ew;
					idx = ret_index[j];
					denTmp[idx] += g;
			}
		} // END OF COMPUTING DENSITIES AT POINTS

		for(int i = 0; i < numPoints; i++){
			//if(i < 10) printf("CPU - den[%d] = %f \n", i, denTmp[i]);
			logL += logf(max(denTmp[i], EPSILONDENSITY));
		}
		//printf("\n");
		//printf("reduction result (likelihood): %f \n", logL);

		if(den0 != NULL){
			for(int i = 0; i < numPoints; i++){
				float h2 = hs[i] * hs[i];
				den0[i] = denTmp[i] + GaussianKernel(h2, 0.0f) *  Points[0].weights[i] * edgeWeights[0][i];
			}
		}

		if(den1 != NULL){
			for(int i = 0; i < numPoints; i++){
				den1[i] = denTmp[i];
			}
		}

		FreeDen(denTmp);
	}

	return logL;
}

// compute the log likelihood given a center (h0, alpha0) and step (stepH, stepA)
// By Guiming @ 2016-03-06
/*
 return 9 elements log likelihood in float* logLs
**/
void hj_likelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float** edgeWeights, float h0, float alpha0, float stepH, float stepA, int lastdmax, float* logLs, float* hs, float* den0, float* den1, bool useGPU, float** dHs, float** dDen0, float** dDen1, float** dDen0cpy)
{
    //int n = Points.numberOfPoints;

    //float gml;
    // the center (h0, alpha0)
    if(lastdmax == -1){ // avoid unnecessary [expensive] computation
	    //LogLikelihood(Ascii, Points, edgeWeights, h0, den0, den1, useGPU, dDen0, dDen1);
	    float L0 = LogLikelihood(Ascii, Points, gpuPoints, edgeWeights, hs, den0, den1, useGPU, dHs, dDen0, dDen1, h0, alpha0, dDen0cpy);
	    //printf("L0: %.5f\t", L0);
	    logLs[0] = L0;
	}

	//exit(0);
    // (h0 - stepH, alpha0)
    if(lastdmax != 2){ // avoid unnecessary [expensive] computation
	    //LogLikelihood(Ascii, Points, edgeWeights, h0 - stepH, den0, den1, useGPU, dDen0, dDen1);
	    float L1 = LogLikelihood(Ascii, Points, gpuPoints, edgeWeights, hs, den0, den1, useGPU, dHs, dDen0, dDen1, h0 - stepH, alpha0, dDen0cpy);
	    //printf("L1: %.5f\t", L1);
	    logLs[1] = L1;
	}
    // (h0 + stepH, alpha0)
    if(lastdmax != 1){
	    //LogLikelihood(Ascii, Points, edgeWeights, h0 + stepH, den0, den1, useGPU, dDen0, dDen1);
	    float L2 = LogLikelihood(Ascii, Points, gpuPoints, edgeWeights, hs, den0, den1, useGPU, dHs, dDen0, dDen1, h0 + stepH, alpha0, dDen0cpy);
	    //printf("L2: %.5f\t", L2);
	    logLs[2] = L2;
	}
    // (h0, alpha0 + stepA)
    if(lastdmax != 4){
	    //LogLikelihood(Ascii, Points, edgeWeights, h0, den0, den1, useGPU, dDen0, dDen1);
	    float L3 = LogLikelihood(Ascii, Points, gpuPoints, edgeWeights, hs, den0, den1, useGPU, dHs, dDen0, dDen1, h0, alpha0 + stepA, dDen0cpy);
	    //printf("L3: %.5f\t", L3);
	    logLs[3] = L3;
	}
    // (h0, alpha0 - stepA)
    if(lastdmax != 3){
	    //LogLikelihood(Ascii, Points, edgeWeights, h0, den0, den1, useGPU, dDen0, dDen1);
	    float L4 = LogLikelihood(Ascii, Points, gpuPoints, edgeWeights, hs, den0, den1, useGPU, dHs, dDen0, dDen1, h0, alpha0 - stepA, dDen0cpy);
	    //printf("L4: %.5f\n", L4);
	    logLs[4] = L4;
	}
}

// compute the optimal h and alpha (parameters for calculating the optimal adaptive bandwith)
// By Guiming @ 2016-03-06
/*
 return 3 optmal parameters in float* optParas (optH, optAlpha, LogLmax)
//EDIT: Timothy @ 03/10/2021
//Added aditional variable to this, hj_likelihood and LogLikelihood functions to handle array of SamplePoints whenever multiple GPUs are present
**/
void hooke_jeeves(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, float* hs, float* den0, float* den1, bool useGPU, float** dHs, float** dDen0, float** dDen1, float** dDen0cpy){
	float* Ls = (float*)malloc(5 * sizeof(float)); // remember to free at the end
	hj_likelihood(Ascii, Points, gpuPoints, edgeWeights, h0, alpha0, stepH, stepA, -1, Ls, hs, den0, den1, useGPU, dHs, dDen0, dDen1, dDen0cpy);
	//exit(0);
	float Lmax = Ls[0];

	float s = stepH / 200;
	float a = stepA / 200;

	int iteration = 0;
    while ((stepH > s || stepA > a) &&  iteration <= MAX_NUM_ITERATIONS){

        //float Lmax0 = Lmax;
        int dmax = 0;
        for(int i = 0; i < 5; i++){
            if(Ls[i] > Lmax){
            	Lmax = Ls[i];
                dmax = i;
            }
        }
        if(DEBUG)
        	printf ("iteration: %d center: (%.5f %.5f) steps: (%.5f %.5f) dmax: %d Lmax: %.5f \n", iteration, h0, alpha0, stepH, stepA, dmax, Lmax);

        if(dmax == 0){
            stepH = stepH / 2;
            stepA = stepA / 2;
        }

        else{
            if(dmax == 1){
                h0 = h0 - stepH;
                alpha0 = alpha0;
                Ls[2] = Ls[0]; // avoid unnecessary [expensive] computation
                Ls[0] = Ls[1];
            }
            if(dmax == 2){
                h0 = h0 + stepH;
                alpha0 = alpha0;
                Ls[1] = Ls[0];
                Ls[0] = Ls[2];
            }
            if (dmax == 3){
                h0 = h0;
                alpha0 = alpha0 + stepA;
                Ls[3] = Ls[0];
                Ls[0] = Ls[4];
            }
            if(dmax == 4){
                h0 = h0;
                alpha0 = alpha0 - stepA;
                Ls[3] = Ls[0];
                Ls[0] = Ls[4];
            }
        }
	    hj_likelihood(Ascii, Points, gpuPoints, edgeWeights, h0, alpha0, stepH, stepA, dmax, Ls, hs, den0, den1, useGPU, dHs, dDen0, dDen1, dDen0cpy);
	    iteration++;
    }

    optParas[0] = h0;
    optParas[1] = alpha0;
    optParas[2] = Lmax;

	free(Ls);
    Ls = NULL;
}

///////// Guiming on 2016-03-16 ///////////////

// check whether the result from sequential computation and that from parallel computation agree
void CheckResults(AsciiRaster AsciiSEQ, AsciiRaster AsciiPARA){
	float eps = 0.000001f;

	size_t n = AsciiSEQ.nCols * AsciiSEQ.nRows;

	for(size_t i = 0; i < n; i++){
		if(abs(AsciiSEQ.elements[i] - AsciiPARA.elements[i]) > eps){
			printf("TEST FAILED. Result from parallel computation does not match that from sequential computation.\n");
			return;
		}
	}
	printf("TEST PASSED. Result from GPU computation does match that from CPU computation.\n");
}

float compGML(float* den0, int n){
	float gml = 0.0f;
	for(int i = 0; i < n; i++){
		gml = gml + log(den0[i]);
	}
	gml = expf(gml / n);
	return gml;
}

// reduction sum on GPU
float ReductionSumGPU(float** dArray, int n){
	int N = n;
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of items evenly
	int div = n / GPU_N; //Division of items to be divided amongst GPUs
	int idx = 0;
	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((device == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}

		//printf("idx = %d, div = %d \n", idx, div);

	   int numberOfElements = div;  	   
	   int iteration = 0;
	   int NUM_ACTIVE_ITEMS = numberOfElements; // # active items need to be reduced

	   // approx. # of blocks needed
	   int NUM_BLOCKS = (numberOfElements) / BLOCK_SIZE;

	   // decide grid dimension
	   int GRID_SIZE = (int)(sqrtf(NUM_BLOCKS)) + 1;
	   dim3 dimGrid(GRID_SIZE, GRID_SIZE);

	   // call the kernel for the first iteration
	   ReductionSum<<<dimGrid, BLOCK_SIZE, 0, streams[device]>>>(dArray[device], idx, div, N, iteration, NUM_ACTIVE_ITEMS);

	   // update # of items to be reduced in next iteration
	   NUM_ACTIVE_ITEMS = (NUM_ACTIVE_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	   // update numberOfElements (needed for deciding grid dimension)
	   numberOfElements = dimGrid.x * dimGrid.y;

	   // increment iteraton index
	   iteration++;

	   // iterate if needed
	   while(numberOfElements > 1){
		  NUM_BLOCKS = (numberOfElements ) / BLOCK_SIZE;

		  GRID_SIZE = (int)(sqrtf(NUM_BLOCKS)) + 1;
		  dimGrid.x = GRID_SIZE;
		  dimGrid.y = GRID_SIZE;

		  ReductionSum<<<dimGrid, BLOCK_SIZE, 0, streams[device]>>>(dArray[device], idx, div, N, iteration, NUM_ACTIVE_ITEMS);

		  NUM_ACTIVE_ITEMS = (NUM_ACTIVE_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

		  numberOfElements = dimGrid.x * dimGrid.y;

		  iteration++;
		}
	   idx += div;
	}

	// copy back results (partial sums) and compute total sum
	float sum = 0.0f;
	for (int device = 0; device < GPU_N; device++)
	{	
		float tmp = 0.0;
		cudaSetDevice(device + GPU_START);		
		cudaMemcpyFromSymbol(&tmp, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);		
		sum = sum + tmp;
		//printf("partial sum on device %d = %f, current sum = %f \n", device, tmp, sum);
	}
	cudaSetDevice(GPU_START);	

	return sum;
}


// reduction sum on GPU
void ReductionSumGPU_V0(float* dArray, int numberOfElements) {
	unsigned int N = numberOfElements;
	printf("N = %d\n", N);

	int iteration = 0;
	int NUM_ACTIVE_ITEMS = numberOfElements; // # active items need to be reduced

	// approx. # of blocks needed
	int NUM_BLOCKS = (numberOfElements) / BLOCK_SIZE;

	// decide grid dimension
	int GRID_SIZE = (int)(sqrtf(NUM_BLOCKS)) + 1;
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);

	//int GRID_SIZE = NUM_BLOCKS + 1;
	//dim3 dimGrid(0, GRID_SIZE);
	printf("dimGrid.x = %d, dimGrid.y = %d, dimGrid.z = %d, BLOCK_SIZE = %d \n", dimGrid.x, dimGrid.y, dimGrid.z, BLOCK_SIZE);

	//printf("iteration %d NUM_ACTIVE_ITEMS %d GRID_SIZE %d x GRID_SIZE %d\n", iteration, NUM_ACTIVE_ITEMS, GRID_SIZE, GRID_SIZE);

	// call the kernel for the first iteration
	ReductionSum_V0 << <dimGrid, BLOCK_SIZE, 0, streams[0] >> > (dArray, N, iteration, NUM_ACTIVE_ITEMS);

	//ReductionSum << <NUM_BLOCKS + 1, BLOCK_SIZE, 0, streams[0] >> > (dArray, N, iteration, NUM_ACTIVE_ITEMS);
	//printf("NUM_BLOCKS = %d, BLOCK_SIZE = %d\n", NUM_BLOCKS + 1, BLOCK_SIZE);

	/*
	float* tmpArray = AllocateDen(N);
	CopyFromDeviceDen(tmpArray, dArray, N);

	for (int i = 0; i < N; i+= BLOCK_SIZE) {
		printf("iteration %d, %d %f\n", iteration, i, tmpArray[i]);
	}
	printf("\n");
	*/

	// update # of items to be reduced in next iteration
	NUM_ACTIVE_ITEMS = (NUM_ACTIVE_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// update numberOfElements (needed for deciding grid dimension)
	numberOfElements = dimGrid.x * dimGrid.y;

	// increment iteraton index
	iteration++;

	// iterate if needed
	while (numberOfElements > 1) {
		NUM_BLOCKS = (numberOfElements) / BLOCK_SIZE;

		GRID_SIZE = (int)(sqrtf(NUM_BLOCKS)) + 1;
		dimGrid.x = GRID_SIZE;
		dimGrid.y = GRID_SIZE;


		//int GRID_SIZE = NUM_BLOCKS + 1;
		//dimGrid.y = NUM_BLOCKS + 1;

		printf("dimGrid.x = %d, dimGrid.y = %d, dimGrid.z = %d, BLOCK_SIZE = %d \n", dimGrid.x, dimGrid.y, dimGrid.z, BLOCK_SIZE);

		//printf("iteration %d NUM_ACTIVE_ITEMS %d GRID_SIZE %d x GRID_SIZE %d \n", iteration, NUM_ACTIVE_ITEMS, GRID_SIZE, GRID_SIZE);
		ReductionSum_V0 << <dimGrid, BLOCK_SIZE, 0, streams[0] >> > (dArray, N, iteration, NUM_ACTIVE_ITEMS);

		//printf("NUM_BLOCKS = %d, BLOCK_SIZE = %d\n", NUM_BLOCKS + 1, BLOCK_SIZE);
		//ReductionSum << <NUM_BLOCKS + 1, BLOCK_SIZE, 0, streams[0] >> > (dArray, N, iteration, NUM_ACTIVE_ITEMS);

		/*
		CopyFromDeviceDen(tmpArray, dArray, N);

		for (int i = 0; i < N; i+= BLOCK_SIZE) {
			printf("iteration %d %d %f\n", iteration, i, tmpArray[i]);
		}
		printf("\n");

		if(iteration == 2)
		{
		  float tmp;
		  cudaMemcpyFromSymbol(&tmp, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);
		  printf("reduction sum %d: %f \n", iteration, tmp);

		}
		*/
		NUM_ACTIVE_ITEMS = (NUM_ACTIVE_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

		numberOfElements = dimGrid.x * dimGrid.y;

		iteration++;
	}
}

// mark the boundary cells on a raster representing the study area
// By Guiming @ 2016-09-02
void MarkBoundary(AsciiRaster* Ascii, bool useGPU){

	auto t1 = high_resolution_clock::now();

	if(useGPU){ // do it on GPU

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// invoke kernels to mark the boundary of study area
			// execution config.
			int NBLOCK_W = (Ascii[i].end - Ascii[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
			//printf("In Marking Boundary...\n");
			dMarkBoundary << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (Ascii[i]);
		}
		cudaSynchronizeStreams();
		cudaSetDevice(GPU_START);
	}
	else{ // do it on CPU
		for(int row = 0; row < Ascii[0].nRows; row++){
			for(int col = 0; col < Ascii[0].nCols; col++){

				size_t idx = row * Ascii[0].nCols + col;

				if(Ascii[0].elements[idx] == Ascii[0].noDataValue)
					continue;

				if(row == 0 || (row == Ascii[0].nRows - 1) || col == 0 || (col == Ascii[0].nCols - 1)){ // cells in the outmost rows and cols
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}

				if(Ascii[0].elements[(row - 1) * Ascii[0].nCols + col - 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				if(Ascii[0].elements[row * Ascii[0].nCols + col - 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				if(Ascii[0].elements[(row + 1) * Ascii[0].nCols + col - 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}

				if(Ascii[0].elements[(row - 1) * Ascii[0].nCols + col] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				if(Ascii[0].elements[(row + 1) * Ascii[0].nCols + col] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}

				if(Ascii[0].elements[(row - 1) * Ascii[0].nCols + col + 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				if(Ascii[0].elements[row * Ascii[0].nCols + col + 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				if(Ascii[0].elements[(row + 1) * Ascii[0].nCols + col + 1] == Ascii[0].noDataValue){
					Ascii[0].elements[idx] = 1.0f;
					continue;
				}
				Ascii[0].elements[idx] = 0.0f;
			}
		}
	}
	auto t2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...MarkBoundary took %f ms\n", ms_double.count());
}

// compute the closest distances from sample points to study area boundary
// By Guiming @ 2016-09-02
void CalcDist2Boundary(SamplePoints* Points, AsciiRaster* Ascii, bool useGPU){

	// mark the boundary first
	//MarkBoundary(Ascii, useGPU); // either on GPU or CPU

	//printf("Done Marking Boundary!\n");
	auto tt1 = high_resolution_clock::now();
	if(useGPU){ // do it on GPU
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCalcDist2Boundary << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (Points[i], Ascii[i]);
		}
		ReformPoints(Points, sPoints);
		cudaSetDevice(GPU_START);
	}
	else{
		float p_x, p_y, cell_x, cell_y;
		for (int p = 0; p < Points[0].numberOfPoints; p++){
			float minDist = FLOAT_MAX;
			p_x = Points[0].xCoordinates[p];
			p_y = Points[0].yCoordinates[p];

			for (int row = 0; row < Ascii[0].nRows; row++){
				for (int col = 0; col < Ascii[0].nCols; col++){
					if (Ascii[0].elements[row*Ascii[0].nCols+col] == 1.0f){ // cells on boundary
						cell_x = COL_TO_XCOORD(col, Ascii[0].xLLCorner, Ascii[0].cellSize);
						cell_y = ROW_TO_YCOORD(row, Ascii[0].nRows, Ascii[0].yLLCorner, Ascii[0].cellSize);
						float d2 = Distance2(p_x, p_y, cell_x, cell_y);

						if(d2 < minDist){
							minDist = d2;
						}
					}
				}
			}

			Points[0].distances[p] = minDist;
			//printf("p: %d Points.distances[p]: %f minDist: %f\n", p, Points.distances[p]);
		}
	}
	auto tt2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = tt2 - tt1;
	printf("...CalcDist2Boundary took %f ms\n", ms_double.count());
}

// By Guiming @ 2016-09-04
SamplePoints CopySamplePoints(const SamplePoints anotherPoints){ // copy points
	int n = anotherPoints.numberOfPoints;
	SamplePoints Points = AllocateSamplePoints(n);
	Points.numberOfPoints = n;
	for(int p = 0; p < n; p++){
		Points.xCoordinates[p] = anotherPoints.xCoordinates[p];
		Points.yCoordinates[p] = anotherPoints.yCoordinates[p];
		Points.weights[p] = anotherPoints.weights[p];
		Points.distances[p] = anotherPoints.distances[p];
	}
	return Points;
}

// comparison function for sort
// By Guiming @ 2016-09-04
/*
int comparev0 ( const void *pa, const void *pb )
{
    const float *a = (const float *)pa;
    const float *b = (const float *)pb;
    if(a[0] == b[0])
        return a[1] - b[1];
    else
        return a[0] > b[0] ? 1 : -1;
}*/
// comparison function for sort
// By Guiming @ 2021-08-17
int compare(const void* pa, const void* pb)
{
	const float* a = (const float*)pa;
	const float* b = (const float*)pb;
	if (a[0] == b[0])
		return a[1] - b[1];
	else
		return a[0] > b[0] ? 1 : -1;
}

void SortSamplePoints(SamplePoints Points) {

	const int n = Points.numberOfPoints;
	SamplePoints temPoints = CopySamplePoints(Points);

	float* distances = (float*)malloc(2*n*sizeof(float));
	for (int i = 0; i < 2*n - 1; i += 2)
	{
		distances[i] = Points.distances[i/2];
		distances[i + 1] = i/2 * 1.0f;
	}
	/*
	for(int i = 0; i < 2*n; ++i)
	  printf("distance[%d] = %f\n", i, distances[i]);
	*/
	//exit(0);
	//qsort(distances, NPONTS, sizeof(distances[0]), comparev0);
	qsort(distances, n, 2*sizeof(distances[0]), compare);

	//for (int i = 0; i < n; i++)
	for (int i = 0; i < 2*n - 1; i += 2)
	{
		/*int idx = (int)distances[i][1];
		Points.xCoordinates[i] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i] = temPoints.yCoordinates[idx];
		Points.weights[i] = temPoints.weights[idx];
		Points.distances[i] = temPoints.distances[idx];
		*/
		int idx = (int)distances[i + 1];
		Points.xCoordinates[i/2] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i/2] = temPoints.yCoordinates[idx];
		Points.weights[i/2] = temPoints.weights[idx];
		Points.distances[i/2] = temPoints.distances[idx];
	}
	/*
	for(int i = 0; i < n; ++i)
	  printf("%.1f\n", Points.distances[i]);
	*/
	FreeSamplePoints(&temPoints);

}

// if bandwidths are provided in the file, need to adjust the order in hs
void SortSamplePoints(SamplePoints Points, float* hs) {

	const int n = Points.numberOfPoints;
	SamplePoints temPoints = CopySamplePoints(Points);

	//printf("here %d\n", n);

	float* temhs = (float*)malloc(n * sizeof(float));
	
	//printf("there %d\n", n);
	
	for (int i = 0; i < n; i++) {
		//printf("...hs[%d]\n", i, hs[i]);
		temhs[i] = hs[i];
	}

	float* distances = (float*)malloc(2 * n * sizeof(float));
	for (int i = 0; i < 2 * n - 1; i += 2)
	{
		distances[i] = Points.distances[i / 2];
		distances[i + 1] = i / 2 * 1.0f;
	}
	/*
	for(int i = 0; i < 2*n; ++i)
	  printf("distance[%d] = %f\n", i, distances[i]);
	*/
	//exit(0);
	//qsort(distances, NPONTS, sizeof(distances[0]), comparev0);
	qsort(distances, n, 2 * sizeof(distances[0]), compare);

	//for (int i = 0; i < n; i++)
	for (int i = 0; i < 2 * n - 1; i += 2)
	{
		/*int idx = (int)distances[i][1];
		Points.xCoordinates[i] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i] = temPoints.yCoordinates[idx];
		Points.weights[i] = temPoints.weights[idx];
		Points.distances[i] = temPoints.distances[idx];
		*/
		int idx = (int)distances[i + 1];
		Points.xCoordinates[i / 2] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i / 2] = temPoints.yCoordinates[idx];
		Points.weights[i / 2] = temPoints.weights[idx];
		Points.distances[i / 2] = temPoints.distances[idx];
		hs[i / 2] = temhs[idx];
		//printf("hs[%d]\n", i/2, hs[i/2]);
	}
	/*
	for(int i = 0; i < n; ++i)
	  printf("%.1f\n", Points.distances[i]);
	*/
	FreeSamplePoints(&temPoints);
	FreeBandwidths(temhs);

}

// build a KDtree on sample points
// By Guiming @ 2016-09-07
void BuildCPUKDtree (SamplePoints Points){

	auto t1 = high_resolution_clock::now();

	int NPTS = Points.numberOfPoints;
	dataP = vector<Point>(NPTS);
	for(int i = 0; i < NPTS; i++){
		dataP[i].coords[0] = Points.xCoordinates[i];
		dataP[i].coords[1] = Points.yCoordinates[i];
	}
	int max_level = (int)(log(dataP.size())/log(2) / 2) + 1;
	tree.Create(dataP, max_level);

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...building KDTree (CPU) took %f ms\n", ms_double.count());
}

void BuildGPUKDtree ()
{
	auto t1 = high_resolution_clock::now();

	for(int i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i + GPU_START);
		GPU_tree[i].CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), dataP);	
	}
	cudaSetDevice(GPU_START);

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...building KDTree (GPU) took %f ms\n", ms_double.count());
}

//Enable P2P Access Across Devices
//Timothy @ 08/13/2020
void EnableP2P()
{
	cudaError_t error = cudaSuccess;
	for (int id = 0; id < GPU_N; ++id)
	{
		cudaSetDevice(id);
		const int top = id > 0 ? id - 1 : (GPU_N - 1); //Int representing first in list of GPUs
		int capable = 1; //(T/F) P2P Access is enabled between devices 
		error = cudaDeviceCanAccessPeer(&capable, id, top);
		if (error != cudaSuccess)
		{
			printf("ERROR 1 in EnableP2P: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (capable)
		{
			printf("Enabled P2P for Device %d...\n", id);
			cudaDeviceEnablePeerAccess(top, 0);
		}
		else if (!capable){printf("NOT CAPABLE! P2P for Device %d...\n", id);}
		const int bottom = (id + 1) % GPU_N;
		if (top != bottom)
		{
			error = cudaDeviceCanAccessPeer(&capable, id, bottom);
			if (error != cudaSuccess)
			{
				printf("ERROR 2 in EnableP2P: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			if (capable)
			{
				printf("Enabling P2P for Device %d...\n", id);
				cudaDeviceEnablePeerAccess(bottom, 0);
			}
			else if (!capable){printf("NOT CAPABLE! P2P for Device %d...\n", id);}
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

//By Timothy @ 08/14/2020
//Determine next Device to be used based on passed integers assumed to represent their numbers
void nextDev(int numDev, int& curDev)
{
	if (curDev == (numDev - 1))
	{
		curDev = 0;
	}
	else
	{
		curDev++;
	}
}

//Timothy @ 08/24/2020
//Function to check device properties, primarily for troubleshooting purposes
void DevProp()
{
	for (int i = 0; i < GPU_N; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
}

////Timothy @ 12/29/20
////Function which copies each group of points into a temorary place on the host, before copying their values to
////hPoints in order to reform the original group
void ReformPoints(SamplePoints* dPoints, const SamplePoints hPoints)
{
	int n = hPoints.numberOfPoints; //Number of TOTAL points
	/*
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	size_t size; //Size of data chunk being copied to tempPoints
	int index = 0; //Index for the points we are reforming into
	*/
	size_t size = n * sizeof(float);

	cudaError_t error = cudaSuccess;

	SamplePoints tempPoints = AllocateSamplePoints(n);
	if (error != cudaSuccess)
	{
		printf("ERROR 0 in ReformPoints: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		//if ((device == GPU_N - 1) && (rem != 0))
		//{
		//	div += rem;
		//}

		/*
		int NBLOCK_W = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
		cudaSetDevice(device + GPU_START);
		printf("%d:Points...\n", device); //DEBUGGING
		PrintPoints << <dimGrid_W, BLOCK_SIZE, 0, streams[device] >> > (dPoints[device], 100);
		cudaStreamSynchronize(streams[device]);
		*/

		//Copy all data from chunk to tempPoints
		// x, y coordinates and weights do not change; no need to reform ?
		error = cudaMemcpy(tempPoints.xCoordinates, dPoints[device].xCoordinates, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1 in ReformPoints (FROM device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(tempPoints.yCoordinates, dPoints[device].yCoordinates, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformPoints (FROM device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(tempPoints.weights, dPoints[device].weights, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 3 in ReformPoints (FROM device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(tempPoints.distances, dPoints[device].distances, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 4 in ReformPoints (FROM device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		//Loop to merge copied chunk of points into hPoints
		/*for (int i = 0; i < div; i++)
		{
			hPoints.xCoordinates[index] = tempPoints.xCoordinates[index];
			hPoints.yCoordinates[index] = tempPoints.yCoordinates[index];
			hPoints.weights[index] = tempPoints.weights[index];
			hPoints.distances[index] = tempPoints.distances[index];
			index++;
		}*/

		for (int i = dPoints[device].start; i < dPoints[device].end; i++)
		{
			hPoints.xCoordinates[i] = tempPoints.xCoordinates[i];
			hPoints.yCoordinates[i] = tempPoints.yCoordinates[i];
			hPoints.weights[i] = tempPoints.weights[i];
			hPoints.distances[i] = tempPoints.distances[i];
		}

	}

	//Copy reformed points accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);

		error = cudaMemcpy(dPoints[i].xCoordinates, hPoints.xCoordinates, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 1 in ReformPoints (TO device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(dPoints[i].yCoordinates, hPoints.yCoordinates, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformPoints (TO device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(dPoints[i].weights, hPoints.weights, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 3 in ReformPoints (TO device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(dPoints[i].distances, hPoints.distances, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 4 in ReformPoints (TO device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		
	}

	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeSamplePoints(&tempPoints);
}

//Timothy @ 08/13/2021
//Reforms points using indeces rather than actually changing any memory
//We realized when reforming other data values that each GPU already has copies of the full set of point structs
void ReformPoints(SamplePoints* dPoints)
{
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		dPoints[i].numberOfPoints = sPoints.numberOfPoints;
		dPoints[i].start = 0;
		dPoints[i].end = sPoints.numberOfPoints; // 100;
	}
	cudaSetDevice(GPU_START);
}

//Timothy @ 08/13/2021
//Divides points using indeces rather than actually changing any memory
void DividePoints(SamplePoints* dPoints)
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	int index = 0; //Index to track start of each data division
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((i == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}
		dPoints[i].numberOfPoints = div;
		dPoints[i].start = index; //Begin tracking division of points
		index += div; //Add division size to index
		dPoints[i].end = index; //Tracking end of division
	}
	cudaSetDevice(GPU_START);
}

//Guiming @ 08/15/2021
//Reform gpuDen on host and copy back accross devices
void ReformGPUDensities(float** dDen, float* hDen)
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points	
	cudaError_t error = cudaSuccess;
	float* tempDen = (float*)malloc(n * sizeof(float));
	size_t size = n * sizeof(float);

	//Guiming 2021-08-15
	for (int i = 0; i < n; i++)
	{
		hDen[i] = 0.0;
	}

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tempDen, dDen[device], size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformGPUDensities: %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (int i = 0; i < n; i++)
		{
			hDen[i] += tempDen[i];
			//if(i == 10) printf("point[%d] - density = %f - device = %d \n", i, hDen[i], device);
		}
		if (DEBUGREFORMING) printf("......Copying GPUDensities FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dDen[i], hDen, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformGPUDensities: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying GPUDensities TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeDen(tempDen);
	if (DEBUGREFORMING) printf("***Reforming GPUDensities DONE\n");

	//printDdensities(dDen);
}

//Timothy @ 08/10/2021
//Reform density arrays on host and copy back accross devices
void ReformDensities(float** dDen, float* hDen)
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	size_t size; //Size of data chunk being copied to tempPoints
	int index = 0; //Index for the points we are reforming into
	cudaError_t error = cudaSuccess;
	float* tempDen = (float*)malloc(n * sizeof(float));
	size = n * sizeof(float);
	
	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((device == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}
		
		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tempDen, dDen[device], size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformDensities: %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (int i = 0; i < div; i++)
		{
			hDen[index] = tempDen[index];
			index++;
		}
		if (DEBUGREFORMING) printf("......Copying Densities FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dDen[i], hDen, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformDensities: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying Densities TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeDen(tempDen);
	if (DEBUGREFORMING) printf("***Reforming Densities DONE\n");
}

//Timothy @ 08/13/2021
//Reform bandwidth arrays on host and copy back accross devices
void ReformBandwidths(float** dBand, float* hBand) 
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	size_t size; //Size of data chunk being copied to tempPoints
	int index = 0; //Index for the points we are reforming into
	cudaError_t error = cudaSuccess;
	float* tempBand = (float*)malloc(n * sizeof(float));
	size = n * sizeof(float);

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((device == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}

		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tempBand, dBand[device], size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformBandwidths: %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (int i = 0; i < div; i++)
		{
			hBand[index] = tempBand[index];
			index++;
		}
		if (DEBUGREFORMING) printf("......Copying Bandwidth FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dBand[i], hBand, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying Bandwidth TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp bands
	FreeBandwidths(tempBand);
	if (DEBUGREFORMING) printf("***Reforming Bandwidths DONE \n");

	//printDdensities(dBand);
}

//Timothy @ 08/13/2021
//Reform EC Weight arrays on host and copy back accross devices
void ReformECWeights(float** dWeights, float* hWeights)
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points
	int rem = n % GPU_N; //Remainder to determine if number of GPUs divides Number of Points evenly
	int div = n / GPU_N; //Division of points to be divided amongst GPUs
	size_t size; //Size of data chunk being copied to tempPoints
	int index = 0; //Index for the points we are reforming into
	cudaError_t error = cudaSuccess;
	float* tempWeights = (float*)malloc(n * sizeof(float));
	size = n * sizeof(float);

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);
		//If on last GPU, check if GPU_N divided into points evenly (rem==0) 
		//if not add remainder to size on final GPU
		if ((device == GPU_N - 1) && (rem != 0))
		{
			div += rem;
		}

		//Copy all data from chunk to tempPoints
		error = cudaMemcpy(tempWeights, dWeights[device], size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 1.%d in ReformBandwidths: %s\n", device, cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		//Loop to merge copied chunk of points into hPoints
		for (int i = 0; i < div; i++)
		{
			hWeights[index] = tempWeights[index];
			index++;
		}
		if (DEBUGREFORMING) printf("......Copying ECWeigths FROM Device %d \n", device);
	}
	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMemcpy(dWeights[i], hWeights, size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("ERROR 2 in ReformBandwidths: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		if (DEBUGREFORMING) printf("......Copying ECWeigths TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp weights
	FreeEdgeCorrectionWeights(tempWeights);
	if(DEBUGREFORMING) printf("***Reforming ECWeights DONE \n");

	//printDdensities(dWeights);
}

void cudaSynchronizeStreams() {
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		cudaStreamSynchronize(streams[i]);
	}
}

void cudaStreamStatus() 
{
	for (int i = 0; i < GPU_N; i++)
	{
		//cudaSetDevice(i + GPU_START);
		printf("strem %d done? %s\n", i, cudaGetErrorString(cudaStreamQuery(streams[i])));
	}
	
}

void printDdensities(float** gpuDen) {

	int n = sPoints.numberOfPoints;
	int NBLOCK_W = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
	dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
	for (int i = 0; i < GPU_N; i++) {
		printf("device[%d]:\n\n", i);
		cudaSetDevice(i + GPU_START);
		PrintDen << <dimGrid_W, BLOCK_SIZE, 0, streams[i] >> > (gpuDen[i], n);
		cudaStreamSynchronize(streams[i]);

	}
}

void printHpoints() 
{
	for (int i = 0; i < sPoints.numberOfPoints; i++) 
	{
		printf("pt %d x %f y %f w %f d %f \n", i, sPoints.xCoordinates[i], sPoints.yCoordinates[i], sPoints.weights[i], sPoints.distances[i]);
	}
}

void printHascii(AsciiRaster ascii) 
{
	for (int row = 0; row < ascii.nRows; row++) 
	{
		for (int col = 0; col < ascii.nCols; col++) 
		{
			printf("%f ", ascii.elements[row * ascii.nCols + col]);
		}
		printf("\n");
	}
}

void printHdensities(float* hden, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d - %f\n", i, hden[i]);
	}
}

/// <summary>
/// Testing if ReductionSumGPU is working properly
/// </summary>
/// <param name="n">number of iterm to sum</param>
/// <returns></returns>
int test_ReductionSumGPU(int n) {
	
	printf("testing ReductionSumGPU on %d 1's...\n", n);
	
	float* nums = AllocateDen(n);
	for (int i = 0; i < n; i++) {
		nums[i] = 1.0;
	}
	
	float** dnums = new float* [GPU_N];
	AllocateDeviceDen(dnums, n);
	
	CopyToDeviceDen(dnums, nums, n);
	/*
	cudaSetDevice(GPU_START);
	ReductionSumGPU(dnums[0], n);
	
	cudaStreamSynchronize(streams[0]);
	
	float tmp;
	cudaMemcpyFromSymbol(&tmp, dReductionSum, sizeof(float), 0, cudaMemcpyDeviceToHost);
	printf("...reduction sum: %d vs. sum %d \n", (int)tmp, n);
	*/

	float tmp = ReductionSumGPU(dnums, n);
	printf("...reduction sum: %d vs. sum %d \n", (int)tmp, n);

	FreeDen(nums);
	FreeDeviceDen(dnums);

	if ((int)tmp != n) {
		return 1;
	}
	else {
		printf("...:) test passed\n");
		return 0;
	}
	

}

void ComputeNearestNeighborDistances(float& meanNNDist, float& minNNDist, float& maxNNDist) {
	
	auto t1 = high_resolution_clock::now();

	int ret_index = 0;
	float ret_dist = 0;
	float dist_sum = 0.0f;
	minNNDist = FLT_MAX;
	maxNNDist = FLT_MIN;
	int i;

	omp_set_dynamic(0);
	omp_set_num_threads(omp_get_max_threads()-1);	
	//printf("%d/%d threads are working on finding nearest neighbors\n", omp_get_num_threads(), omp_get_max_threads());
	#pragma omp parallel for private(ret_index, ret_dist, i) reduction(+:dist_sum) 
	for (int i = 0; i < sPoints.numberOfPoints; i++) {
		float pi_x = sPoints.xCoordinates[i];
		float pi_y = sPoints.yCoordinates[i];

		// range query
		Point query;
		query.coords[0] = pi_x;
		query.coords[1] = pi_y;
		tree.Search(query, &ret_index, &ret_dist);
		if (i % 1000000 == 0) printf("Thread %d - Point %d's NN: Point %d, Dist %f\n", omp_get_thread_num(), i, ret_index, sqrt(ret_dist));
		//if(i % 1000000 == 0) printf("Point %d's NN: Point %d, Dist %f\n", i, ret_index[0], sqrt(ret_dist[0]));
		float dist = sqrt(ret_dist);
		dist_sum += dist;

		#pragma omp critical
		{
			maxNNDist = max(dist, maxNNDist);
			minNNDist = min(dist, minNNDist);
		}
	}

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...computing nearest neighbor distances took %f ms\n", ms_double.count());

	meanNNDist = dist_sum/sPoints.numberOfPoints;

	printf("...mean = %f min = %f max = %f\n", meanNNDist, minNNDist, maxNNDist);
}

void OMP_TEST() {
	float mean = 0.0f;
	float minv = FLT_MAX;
	float maxv = FLT_MIN;

	int i;
	
	omp_set_dynamic(0);
	omp_set_num_threads(omp_get_max_threads() - 1);
	#pragma omp parallel for reduction(+:mean)
	for (i = 0; i < 100; i++) {
		printf("%d threads, Thread = %d, i = %d\n", omp_get_num_threads(), omp_get_thread_num(), i);
		mean += i;
#pragma omp critical
		{
			
			maxv = max((float)i, maxv);
			minv = min((float)i, minv);
		}
	}
	printf("mean = %f min = %f max = %f\n", mean, minv, maxv);
}

void ComputeKNearestNeighborDistances(int k, float* knn_dist) {

	auto t1 = high_resolution_clock::now();

	vector<int> ret_index = vector<int>();
	vector<float> ret_dist = vector<float>(); 
	
	int i;

	omp_set_dynamic(0);
	omp_set_num_threads(omp_get_max_threads() - 1);
	//omp_set_num_threads(1);

	#pragma omp parallel for private(ret_index, ret_dist, i)
	for (int i = 0; i < sPoints.numberOfPoints; i++) {
		float pi_x = sPoints.xCoordinates[i];
		float pi_y = sPoints.yCoordinates[i];

		// range query
		Point query;
		query.coords[0] = pi_x;
		query.coords[1] = pi_y;

		ret_index.clear();
		ret_dist.clear();
		
		tree.SearchKNN(query, k, ret_index, ret_dist);
		
		//if(i == 0) printf("\nThread %d - Point[%d] ", omp_get_thread_num(), i);

		float dist = 0.0f;
		for (int j = 0; j < ret_dist.size(); j++) {
			dist = max(dist, ret_dist[j]);
			//if (i == 0) printf(" dist[%d] = %f", ret_index[j], sqrt(ret_dist[j]));
		}
		//if (i == 0) printf("\n");
		knn_dist[i] = sqrt(dist);

		if (i % 1000000 == 0) {
			printf("Thread %d - Point[%d] %d's NN dist = %f\n", omp_get_thread_num(), i, k, knn_dist[i]);
		}
		//break;
	}

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	printf("...computing %d nearest neighbor distances took %f ms\n", k, ms_double.count());
}
