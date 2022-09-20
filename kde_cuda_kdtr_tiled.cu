// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
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
#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>
#include <windows.h>

#include "SamplePoints.h"
#include "AsciiRaster.h"
#include "Utilities.h"

#include "KDtree.h"
#include "CUDA_KDtree.h"

#include "kde_kernel_kdtr.cu"

#include "GeoTIFF.cpp"

using namespace std;
// for timing
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

unsigned long long getTotalSystemMemory()
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	return status.ullTotalPhys;
}

// distance squared between two points
inline double Distance2(float x0, float y0, float x1, float y1){
	double dx = x1 - x0;
	double dy = y1 - y0;
	//printf("dx=%f dy=%f dx*dx=%f dy*dy=%f dx*dx+dy*dy=%f %f float_MAX=%f float_MAX=%F\n", dx, dy, dx*dx, dy*dy, dx * dx + dy * dy, float(dx * dx + dy * dy), float_MAX, DBL_MAX);
	return dx*dx + dy*dy;
}

// mean center of points
void MeanCenter(SamplePoints Points, float &mean_x, float &mean_y);

// (squared) standard distance of points
void StandardDistance2(SamplePoints Points, double &d2);

// bandwidth squared
inline double BandWidth2(SamplePoints Points){
	double d2;
	StandardDistance2(Points, d2);
	return sqrtf(2.0f / (3 * Points.numberOfPoints)) * d2;
}

// Gaussian kernel
inline float GaussianKernel(double h2, double d2){
	if(d2 >= CUT_OFF_FACTOR * h2){
		return 0.0;
	}
	return expf(d2 / (-2.0 * h2)) / (h2*TWO_PI);
}

//EDIT: Changed AllocateDeviceSamplePoints to return void, and instead utilize pointers to an array
//Changed all functions to utilize the array of pointers
void AllocateDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints Points);
void CopyToDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints hPoints);
void CopyFromDeviceSamplePoints(SamplePoints hPoints, const SamplePoints* dPoints);
SamplePoints AllocateSamplePoints(int n); // random points

// if bandwidths is True, bandwidths are provided in the file (Hoption = -1)
SamplePoints ReadSamplePoints(const char *csvFile, bool bandwidths); // points read from a .csv file: x, y, [h,] w

SamplePoints CopySamplePoints(const SamplePoints Points);
void FreeDeviceSamplePoints(SamplePoints* dPoints);
void FreeSamplePoints(SamplePoints* Points);
void WriteSamplePoints(SamplePoints* Points, const char * csvFile);
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, const char * csvFile);
void ReformPoints(SamplePoints* dPoints, const SamplePoints hPoints); 
void ReformPoints(SamplePoints* dPoints); 
void DividePoints(SamplePoints* dPoints); 

void AllocateDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster Ascii);
void CopyToDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii);
void CopyFromDeviceAsciiRaster(AsciiRaster hAscii, const AsciiRaster dAscii);
AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue, bool serialized=false, bool compute_serialized=false);
AsciiRaster ReadGeoTIFFRaster(char* geotiffFile, bool data_serialized=false, bool compute_serialized=false); // geotiff raster read from a .tif file
void AsciiRasterSwitchDataSerialization(AsciiRaster* Ascii, bool data_serialized);
void AsciiRasterSwitchComputeSerialization(AsciiRaster* Ascii, bool compute_serialized);
//Construct AsciiRaster from a tile read from a GeoTIFF
AsciiRaster AsciiRasterFromGeoTIFFTile(double* geotransform, const char* projection, int nrows, int ncols, float nodata, float** data, bool data_serialized = false, bool compute_serialized = false);
AsciiRaster CopyAsciiRaster(const AsciiRaster Ascii);
void FreeDeviceAsciiRaster(AsciiRaster* Ascii);
void FreeAsciiRaster(AsciiRaster* Ascii);
void WriteGeoTIFFRaster(AsciiRaster* Ascii, const char* geotiffFile);

void ReformAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii); //Combine rasters amongst GPUs into a single raster
void ReformGPUAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii); //Add up cell densities from all devices (gpuDen) into one single array

float* AllocateEdgeCorrectionWeights(SamplePoints Points);
void CopyToDeviceWeights(float** dWeights, const float* hWeights, const int n);
void FreeEdgeCorrectionWeights(float* weights);
void ReformECWeights(float** dWeights, float* hWeights);

void AllocateDeviceEdgeCorrectionWeights(float** dWeights, SamplePoints Points);
void FreeDeviceEdgeCorrectionWeights(float** weights);

// the array holding bandwidth at each point
float* AllocateBandwidths(int n); // n is number of points
double* AllocateDistances(int n); // n is number of points
//Allocation on device now done with pointers instead of return
void AllocateDeviceBandwidths(float** dBandwidths, int n); // n is number of points
void CopyToDeviceBandwidths(float** dBandwidth, const float* hBandwidths, const int n);
void CopyFromDeviceBandwidths(float* hBandwidth, const float* dBandwidths, const int n);
void FreeDeviceBandwidths(float** bandwidths);
void FreeBandwidths(float* bandwidths);
void FreeDistances(double* distances);
void ReformBandwidths(float** dBand, float* hBand); //Reform bandwidth arrays on host and copy back accross devices

// the array holding inclusive/exclusive density at each point
float* AllocateDen(int n); // n is number of points
void AllocateDeviceDen(float** dDen, int n); // n is number of points
void CopyToDeviceDen(float** dDen, const float* hDen, const int n);
void CopyFromDeviceDen(float* hDen, const float* dDen, const int n);
void CopyDeviceDen(float* dDenTo, const float* dDenFrom, const int n);
void FreeDeviceDen(float** den);
void FreeDen(float* den);
void ReformDensities(float** dDen, float* den); //Reforms densities from all devices back into one single array
void ReformGPUDensities(float** gpuDen, float* den); //Add up densities from all devices (gpuDen) into one single array

// compute the optimal Maximum Likelihood Estimation fixed bandwidth
float MLE_FixedBandWidth(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dDen0 = NULL, float** dDen1 = NULL);

// compute fixed bandwidth density at sample points
void ComputeFixedDensityAtPoints(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, float* dDen0 = NULL, float* dDen1 = NULL);

// compute the log likelihood given single bandwidth h
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dDen0 = NULL, float** dDen1 = NULL);

// compute the log likelihood given bandwidths hs
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float* hs, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float h = 1.0f, float alpha = -0.5f, float** dDen0cpy = NULL);

// compute the log likelihood given a center (h0, alpha0) and step (stepH, stepA)
void hj_likelihood(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, int lastdmax, float* logLs, float* hs = NULL, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float** dDen0cpy = NULL);

// compute the optimal h and alpha (parameters for calculating the optimal adaptive bandwith)
void hooke_jeeves(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, float* hs = NULL, float* den0 = NULL, float* den1 = NULL, bool useGPU = false, float** dHs = NULL, float** dDen0 = NULL, float** dDen1 = NULL, float** dDen0cpy = NULL);

float compGML(float* den0, int n);

// exact edge effects correction (Diggle 1985)
void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights);
void EdgeCorrectionWeightsExact(SamplePoints Points, float *hs, AsciiRaster Ascii, float *weights);

// check whether the result from sequential computation and that from parallel computation agree
void CheckResults(AsciiRaster AsciiSEQ, AsciiRaster AsciiPARA);

// reduction an array on GPU
void ReductionSumGPU_V0(float* dArray, int numberOfElements);
float ReductionSumGPU(float** dArray, int numberOfElements);

// extract study area boundary from a raster
// the second parameter tempAscii is only needed for gpu computing
void MarkBoundary(AsciiRaster* Ascii, AsciiRaster& tmpAscii, bool useGPU = false);

// compute the closest distances from sample points to study area boundary
void CalcDist2Boundary(SamplePoints* Points, AsciiRaster* Ascii, bool useGPU = false);

// sort the sample points on their distances to study area boundary
void SortSamplePoints(SamplePoints Points);

// sort the sample points on their distances to study area boundary
// if bandwidths are provided in file (Hoption = -1), need to adjust the order of bandwidths
void SortSamplePoints(SamplePoints Points, float* hs);

// comparison function for sort
int compare ( const void *pa, const void *pb );

void BuildCPUKDtree (SamplePoints Points);
void BuildGPUKDtree ();

void DevProp(); //Check device properties, primarily for troubleshooting purposes

//This performs the same tasks as ComputeFixedDensityAtPoints function, however it is designed specifically to run accross multiple GPUs
void ComputeFixedDensityDevice(cudaStream_t* streams, AsciiRaster* Ascii, SamplePoints* Points, float** edgeWeights, float h, float* den0, float** dDen0);

// Extract points that contribute density to cells in the current raster tile
SamplePoints ExtractSamplePointsCurTile(AsciiRaster ascTile, float* hs, vector<int>& index);

/* Run in 2 modes
 *
 * Mode 0: Do not read points and mask from files.
 *         User specify # of points, cell size of the mask for edge correction, and cell size of the estimated intensity surface.
 *         Random points with x, y coordinates in the range [0,100] will be generated.
 *         The cell size (must be less than 100) determines how many cells in the mask raster and the density surface raster.
 *
 *         mGPUKDE_tiled.exe [mode] [#points] [cellsize_edge_correction] [cellsize_density] [bwoption] [enable_edge_corection] [enable_sample_weight] [skipOMP] [skipCUDA] [num_cpu_threads] [num_gpu] [denFN_omp] [denFN_cuda] [serialized_mode]
 *         e.g., mGPUKDE_tiled.exe 0 100 1.0 0.5 2 0 0 1 0 12 2 omp.tif cuda.tif 0
 *
 * Mode 1: Read points and mask from files.
 *
 *         mGPUKDE_tiled.exe [mode] [points_file] [mask_file_edge_correction] [mask_file_density] [bwoption] [enable_edge_corection] [enable_sample_weight] [skipOMP] [skipCUDA] [num_cpu_threads] [num_gpu] [denFN_omp] [denFN_cuda] [serialized_mode]
 *         e.g., mGPUKDE_tiled.exe 1 Points.csv Mask_edge_correction.tif Mask_density.tif 2 0 0 1 0 12 2 omp.tif cuda.tif 0
 *
*/

/* be very careful with these global variables
 * they are declared in this way to avoid passing additional parameters in functions
*/

int GPU_N = 2;
int GPU_START = 0;

int NCPU_THREADS = 1;

KDtree tree; // pointer to the kd tree, can be accessed in any function
//CUDA_KDTree GPU_tree[GPU_N]; //pointer to the GPU kd tree, can be accessed in any function. EDIT: A copy of the tree 
CUDA_KDTree* GPU_tree;
//is now kept on each GPU with each of these pointers corresponding to a GPU.

vector <Point> dataP; // pointer to the vector to hold data points in kd tree, initilized when building kd tree
float** gpuDen; // this is a global array allocated on gpu to store density values. Used in DensityAtPointsKdtr
int MAX_N_NBRS = 0;

//Streams to be used for parallelism
cudaStream_t* streams;

SamplePoints sPoints; // sample of point events

float* gedgeWeights;  // edge effect correct weights (for each point in the sample)

float* hs; // bandwidth for points on host
float* den0;
float* den1;
float reductionSum;

bool UPDATEWEIGHTS = 0; // conduct edge effect correction
bool SAMPLEWEIGHTS = 0; // weigh samples
bool SERIALIZED_MODE = 0; // A special mode for internal raster representation to save memory space. If true (1), no-data values are stripped off and raster is serialized into a sequential 1d array
						  // experiments show it is not beneficial - as it invoid the optimization coming with point-based parallelization
						  // advised to always set to 0 (passed from the last command arguements)

void OMP_TEST();


int main(int argc, char *argv[]){

		auto T1 = high_resolution_clock::now();

		int NPNTS = 100;                // default # of points
		float CELLSIZE = 1.0f;          // default cellsize (for edge correction)
		float CELLSIZE_density = 0.5f;    // default cellsize (for density estimation)
		char* pntFn = "Points.csv";  // default points file
		char* maskFn = "Mask.tif";   // default mask file (for edge correction)
		char* maskFn_density = "Mask.tif";   // default mask file (for density estimation)
		bool fromFiles = true;          // by default, read Points and Mask from files

		int NGPU = 1;
		int NCPU_THREADS = 1;

		int SKIPOMP = 0;                // by default, do not skip OpenMP execution
		int SKIPCUDA = 0;               // by default, do not skip CUDA execution

		//bandwidth option
		int Hoption = 0; // 0 for rule of thumb
						 // 1 for h optimal
						 // 2 for h adaptive
						 // -1 use whatever bandwidths provided in the input file

		char* denOMPfn = "den_OMP.tif";
		char* denCUDAfn = "den_CUDA.tif";

		// parse commandline arguments
		if (argc != 15) {
			printf("Incorrect arguments provided. Exiting...\n");
			printf("Run in mode 0:\n ./kde_cuda 0 #points cellsize_edge_correction cellsize_density h_option enable_edge_corection enable_sample_weight skip_omp_parallel skip_gpu_parallel num_cpu_threads num_gpu denfn_omp denfn_cuda\n");
			printf("Run in mode 1:\n ./kde_cuda 1 points_file mask_file_edge_correction mask_file_density h_option enable_edge_corection enable_sample_weight skip_omp_parallel skip_gpu_parallel num_cpu_threads num_gpu denfn_omp denfn_cuda\n");
			
			return 1;
		}
		else {
			int mode = atoi(argv[1]);
			if (mode == 0) {
				fromFiles = false;
				NPNTS = atoi(argv[2]);
				CELLSIZE = (float)atof(argv[3]);
				CELLSIZE_density = (float)atof(argv[4]);
				Hoption = atoi(argv[5]);

				if (Hoption == -1) {
					printf("***Error - should never use bandwidth option -1 in mode 0 (i.e., randomly generately points). Exiting...");
					return 1;
				}

				UPDATEWEIGHTS = atoi(argv[6]);
				SAMPLEWEIGHTS = atoi(argv[7]);

				SKIPOMP = atoi(argv[8]);
				SKIPCUDA = atoi(argv[9]);

				NGPU = atoi(argv[11]);
				NCPU_THREADS = atoi(argv[10]);

				denOMPfn = argv[12];
				denCUDAfn = argv[13];

				SERIALIZED_MODE = atoi(argv[14]);
			}
			else if (mode == 1) {
				pntFn = argv[2];
				maskFn = argv[3];
				maskFn_density = argv[4];
				Hoption = atoi(argv[5]);
				
				UPDATEWEIGHTS = atoi(argv[6]);
				SAMPLEWEIGHTS = atoi(argv[7]);

				SKIPOMP = atoi(argv[8]);
				SKIPCUDA = atoi(argv[9]);
				
				NGPU = atoi(argv[11]);
				NCPU_THREADS = atoi(argv[10]);

				denOMPfn = argv[12];
				denCUDAfn = argv[13];

				SERIALIZED_MODE = atoi(argv[14]);

				GeoTIFFReader tiffreadertmp((const char*)maskFn_density);
				double* geo = tiffreadertmp.GetGeoTransform();
				CELLSIZE_density = geo[1];

			}
			else {
				printf("***Incorrect arguments provided. Exiting...\n");
				printf("***Run in mode 0:\n ./kde_cuda 0 #points cellsize_edge_correction cellsize_density h_option enable_edge_corection enable_sample_weight skip_omp_parallel skip_gpu_parallel num_cpu_threads num_gpu denfn_seq, denfn_cuda\n");
				printf("***Run in mode 1:\n ./kde_cuda 1 points_file mask_file_edge_correction mask_file_density h_option enable_edge_corection enable_sample_weight skip_omp_parallel skip_gpu_parallel num_cpu_threads num_gpu denfn_seq, denfn_cuda\n");
				return 1;
			}

		}

		printf("===Number of requested CPU thread(s): %d\n", NCPU_THREADS);
		if (omp_get_max_threads() < NCPU_THREADS) {
			printf("===Number of requested CPU threads (%d) EXCEEDS number of available threads (%d)\n", NCPU_THREADS, omp_get_max_threads());
		}
		NCPU_THREADS = min(NCPU_THREADS, omp_get_max_threads());
		printf("===%d CPU thread(s) out of %d threads available can be used\n\n", NCPU_THREADS, omp_get_max_threads());

		//Assign and print number of Compute Capable Devices
		int nGPU;
		cudaGetDeviceCount(&nGPU);
		printf("===Number of Capable Devices: %d\n", nGPU);
		if (nGPU < NGPU) {
			printf("===Number of requested devices (%d) EXCEEDS number of available devices (%d)\n", NGPU, nGPU);
		}
		GPU_N = min(nGPU, NGPU);
		printf("===%d device(s) out of %d devices available can be used\n\n", GPU_N, nGPU);

		// skip the first GPU where possible as it's may be used by the OS for other purposes
		if (GPU_N < nGPU) {
			GPU_START = 0;
		}
		
		DevProp();

		streams = new cudaStream_t[GPU_N];

		cudaError_t error = cudaSuccess;


		//Create streams for each available device

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i+GPU_START);
			error = cudaStreamCreate(&streams[i]);
		}
		if (error != cudaSuccess)
		{
			printf("***Failed to create streams (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		cudaSetDevice(GPU_START); //Reset device to first GPU

		
		AsciiRaster Mask;    // a mask indicating the extent of study area
		AsciiRaster MaskBoundary; // used only for computing point distance to boundary. 
								  // cell values are serialized to an 1d array, with nodata values removed
		bool correction = true; // enable edge effect correction
		srand(100); // If not read from files, generate random points

		auto t1 = high_resolution_clock::now();
		//Read or generate points
		if (fromFiles) {
			Mask = ReadGeoTIFFRaster(maskFn);
			if (Hoption == -1) { 
				sPoints = ReadSamplePoints(pntFn, true);
				hs = AllocateBandwidths(sPoints.numberOfPoints);
				for (int i = 0; i < sPoints.numberOfPoints; i++) {
					hs[i] = sPoints.distances[i];
					sPoints.distances[i] = 0.0;
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
		/* Getting number of milliseconds as a float. */
		duration<float, std::milli> ms_float = t2 - t1;
		printf("...reading in data took %.0f ms\n", ms_float.count());

		if (Hoption > -1) {
			hs = AllocateBandwidths(sPoints.numberOfPoints);
		}
		gedgeWeights = AllocateEdgeCorrectionWeights(sPoints);
		den0 = AllocateDen(sPoints.numberOfPoints);
		den1 = AllocateDen(sPoints.numberOfPoints);

		//DenSurf = CopyAsciiRaster(Mask);

		// parameters
		int numPoints = sPoints.numberOfPoints;
		int nCols = Mask.nCols;
		int nRows = Mask.nRows;
		float xLLCorner = Mask.xLLCorner;
		float yLLCorner = Mask.yLLCorner;
		float noDataValue = Mask.noDataValue;
		float cellSize = Mask.cellSize;

		printf("===Number of points: %d\n", numPoints);
		printf("===Cell size (edge correction): %f\n", cellSize);		
		printf("===Number of cells (edge correction): %llu\n", (size_t)nCols * nRows);
		printf("===Cell size (density estimation): %f\n", CELLSIZE_density);
		int factor = int(ceil(cellSize / CELLSIZE_density));
		printf("===Number of cells (density estimation): %llu\n", factor*factor*(size_t)nCols * (size_t)nRows);

		printf("===Bandwidth option: %d\n", Hoption);
		printf("===Enable edge correction: %d\n", UPDATEWEIGHTS);
		printf("===Enable sample weight: %d\n", SAMPLEWEIGHTS);

		printf("===Skip parallel computing on multi-core CPU (OPENMP)? %d\n", SKIPOMP);
		printf("===Skip parallel computing on GPUs (CUDA)? %d\n", SKIPCUDA);

		printf("===Serialize raster cells for compute: %d\n", SERIALIZED_MODE);
	
		printf("===Number of threads per block: %d\n", BLOCK_SIZE);

		// do the work
		float cell_x; // x coord of cell
		float cell_y; // y coord of cell
		float p_x;    // x coord of point
		float p_y;    // x coord of point
		float p_w;    // weight of point
		float e_w = 1.0;    // edge effect correction weight

		float h = sqrtf(BandWidth2(sPoints));
		printf("===Rule of thumb bandwidth h0: %.5f\n", h);

		if (SKIPOMP == 0) {
			
			// only need to setup once for the whole program
			omp_set_dynamic(0);
			printf("...omp_get_max_threads()= %d\n", omp_get_max_threads());
			//omp_set_num_threads(max(1, omp_get_max_threads() - 1)); 
			omp_set_num_threads(NCPU_THREADS);
			printf("...omp_get_num_threads()= %d\n", omp_get_num_threads());
			auto t1_cpu = high_resolution_clock::now();

		///////////////////////// Parallel computing on multi-core CPU (OpenMP) /////////////////////////////////

			if(UPDATEWEIGHTS){
				
				MaskBoundary = CopyAsciiRaster(Mask); // used only for computing point distance to boundary. 
																  // cell values are serialized to an 1d array, with nodata values removed
				MarkBoundary(&MaskBoundary, Mask); // either on GPU or CPU				

				CalcDist2Boundary(&sPoints, &MaskBoundary);
				FreeAsciiRaster(&MaskBoundary);

				if (Hoption == -1) {
					//printf("***\n");
					SortSamplePoints(sPoints, hs);
					//printf("***\n");
				}
				else {
					SortSamplePoints(sPoints);
				}
				
			}

			BuildCPUKDtree(sPoints);

			if (Hoption == -1) {
				if (UPDATEWEIGHTS) {
					// compute edge effect correction weights
					EdgeCorrectionWeightsExact(sPoints, hs, Mask, gedgeWeights);
				}
			}
			else {

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
					
					float h0 = h;
					float alpha0 = -0.5;
					float stepH = h0 / 10;
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

					// update edge correction weights
					if (UPDATEWEIGHTS) {
						EdgeCorrectionWeightsExact(sPoints, hs, Mask, gedgeWeights);
					}
				}
			}

			////////////////////////////// KDE //////////////////////

			
			auto t1_kde = high_resolution_clock::now();

			// figure out how much memory is left on the host
			size_t hostMem_free = getTotalSystemMemory();
			printf("...hostMem_free=%.2f GB\n", hostMem_free / 1024.0 / 1024.0 / 1024.0);

			// figure out GeoTIFF block size
			int BLOCK_XSIZE, BLOCK_YSIZE;
			int _NCOLS, _NROWS;

			if (fromFiles) {
				GeoTIFFReader tiffreadertmp((const char*)maskFn_density);
				tiffreadertmp.GetBlockXYSize(BLOCK_XSIZE, BLOCK_YSIZE);
				int* dims = tiffreadertmp.GetDimensions();
				_NCOLS = dims[0];
				_NROWS = dims[1];
			}
			else {
				BLOCK_XSIZE = 128;
				BLOCK_YSIZE = 128;
				_NCOLS = int(ceil(100.0 / CELLSIZE_density));
				_NROWS = int(ceil(100.0 / CELLSIZE_density));
			}

			// size of a single block
			size_t nBytes_BLOCK = (sizeof(float) * (size_t)BLOCK_XSIZE * (size_t)BLOCK_YSIZE) * 3;			
			// can read in up to nBlocks blocks
			int nBlocks = (int)floor(hostMem_free * 0.2 / nBytes_BLOCK);
			printf("...Can read in %d blocks at once\n", nBlocks);
			
			// how many blocks row-wise and col-wise
			int nBlocksX = (int)ceil(_NCOLS * 1.0 / BLOCK_XSIZE);
			int nBlocksY = (int)ceil(_NROWS * 1.0 / BLOCK_YSIZE);

			printf("...GeoTIFF: %d blocks col-wise,  %d blocks row-wise, %d blocks in total\n", nBlocksX, nBlocksY, nBlocksX* nBlocksY);

			// read in the whole GeoTIFF, by default
			int TILE_XSIZE = _NCOLS;
			int TILE_YSIZE = _NROWS;

			if (nBlocks < nBlocksX * nBlocksY) {
				int TILE_nBLOCKS_Y = 1; // tile height in # of blocks 
				int TILE_nBLOCKS_X = 1; // tile width in # of blocks

				while (TILE_nBLOCKS_Y < nBlocksY / 2) {

					if (TILE_nBLOCKS_Y * TILE_nBLOCKS_X < nBlocks) {
						TILE_nBLOCKS_Y += 1;

						while (TILE_nBLOCKS_X < nBlocksX / 2) {

							if (TILE_nBLOCKS_Y * TILE_nBLOCKS_X < nBlocks) {
								TILE_nBLOCKS_X += 1;
								break;
							}
							else {
								TILE_nBLOCKS_X -= 1;
								break;
							}
						}
					}
					else {
						TILE_nBLOCKS_Y -= 1;
						break;
					}
				}
				TILE_XSIZE = TILE_nBLOCKS_X * BLOCK_XSIZE;
				TILE_YSIZE = TILE_nBLOCKS_Y * BLOCK_YSIZE;

				printf("...TILE_nBLOCKS_X=%d, TILE_nBLOCKS_Y=%d\n", TILE_nBLOCKS_X, TILE_nBLOCKS_Y);
			}
			printf("...TILE_XSIZE=%d, TILE_YSIZE=%d\n", TILE_XSIZE, TILE_YSIZE);

			if (fromFiles) {	

				GeoTIFFReader tiffreader((const char*)maskFn_density, TILE_XSIZE, TILE_YSIZE);

				GeoTIFFWriter tiffwriter = GeoTIFFWriter((const char*)denOMPfn, tiffreader.GetGeoTransform(), tiffreader.GetProjection(), _NROWS, _NCOLS, tiffreader.GetNoDataValue());

				float** tileData;
				tileData = tiffreader.GetRasterBand_NextTile(1);

				while (tiffreader.GetNTiles() > 0) {
					//printf("***tiffreader.GetNTiles() = %d\n", tiffreader.GetNTiles());
					t1 = high_resolution_clock::now();

					printf("...working on tile %d / %d\n", tiffreader.GetNTiles(), tiffreader.GetNTilesTotal());

					int* para = tiffreader.GetCurTileParam();
					int XOFF = para[0];
					int YOFF = para[1];
					int XSIZE = para[2];
					int YSIZE = para[3];

					AsciiRaster AsciiTile = AsciiRasterFromGeoTIFFTile(tiffreader.GetGeoTransformTile(), tiffreader.GetProjection(), YSIZE, XSIZE, tiffreader.GetNoDataValue(), tileData, SERIALIZED_MODE, SERIALIZED_MODE);
					if (AsciiTile.nVals > 0) {

						if (AsciiTile.compute_serialized){
							#pragma omp parallel for private(p_x, p_y, p_w, cell_x, cell_y, e_w)
							for (long long int idx = 0; idx < AsciiTile.nVals; idx++) {
								int row = AsciiTile.rowcolIdx[idx] / AsciiTile.nCols;
								int col = AsciiTile.rowcolIdx[idx] % AsciiTile.nCols;
								cell_y = ROW_TO_YCOORD(row, AsciiTile.nRows, AsciiTile.yLLCorner, AsciiTile.cellSize);
								cell_x = COL_TO_XCOORD(col, AsciiTile.xLLCorner, AsciiTile.cellSize);

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
									double d2 = Distance2(p_x, p_y, cell_x, cell_y);
									den += GaussianKernel(hp * hp, d2) * p_w * e_w;
								}
								AsciiTile.elementsVals[idx] = den; // intensity, not probability
								AsciiTile.elements[row * AsciiTile.nCols + col] = den; // intensity, not probability
							}
						}
						else {
							#pragma omp parallel for private(p_x, p_y, p_w, cell_x, cell_y, e_w)
							for (int row = 0; row < AsciiTile.nRows; row++) {
								cell_y = ROW_TO_YCOORD(row, AsciiTile.nRows, AsciiTile.yLLCorner, AsciiTile.cellSize);
								for (int col = 0; col < AsciiTile.nCols; col++) {
									cell_x = COL_TO_XCOORD(col, AsciiTile.xLLCorner, AsciiTile.cellSize);
									//int idx = row * nCols + col;
									size_t idx = row * AsciiTile.nCols + col;
									if (AsciiTile.elements[idx] != AsciiTile.noDataValue) {

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
											double d2 = Distance2(p_x, p_y, cell_x, cell_y);
											den += GaussianKernel(hp * hp, d2) * p_w * e_w;
										}
										AsciiTile.elements[idx] = den; // intensity, not probability
									}
								}
							}
						}
					}

					t2 = high_resolution_clock::now();
					/* Getting number of milliseconds as a float. */
					ms_float = t2 - t1;
					printf("...KernelDesityEstimation on tile %d took %.0f ms\n", tiffreader.GetNTiles(), ms_float.count());

					// write results to file
					tiffwriter.WriteGeoTIFF_NextTile(XOFF, YOFF, XSIZE, YSIZE, AsciiTile.elements);

					// clean up
					FreeAsciiRaster(&AsciiTile);
					delete tileData;

					// retrieve the next tile
					tileData = tiffreader.GetRasterBand_NextTile(1);
				}
				delete tileData;
				//printf("*** AI made it here\n");
			}
			else { // from points generated on the fly

				double geotransform[6] = { 0, CELLSIZE_density, 0, 100.0, 0, -1.0 * CELLSIZE_density };
				GeoTIFFWriter tiffwritergpu = GeoTIFFWriter((const char*)denOMPfn, geotransform, NULL, _NROWS, _NCOLS, -9999.0f);

				int xoff = 0;
				int yoff = 0;

				int cnt = 1;
				int NTILES_TOTAL = (int)ceil(_NCOLS * 1.0 / TILE_XSIZE) * (int)ceil(_NROWS * 1.0 / TILE_YSIZE);
				while (xoff < _NCOLS && yoff < _NROWS) {

					printf("...working on tile %d / %d\n", cnt, NTILES_TOTAL);

					int _TILE_YSIZE = TILE_YSIZE;
					int _TILE_XSIZE = TILE_XSIZE;

					if (TILE_YSIZE > _NROWS - yoff) {
						_TILE_YSIZE = _NROWS - yoff;
					}

					if (TILE_XSIZE > _NCOLS - xoff) {
						_TILE_XSIZE = _NCOLS - xoff;
					}
					
					AsciiRaster AsciiTile = AllocateAsciiRaster(_TILE_XSIZE, _TILE_YSIZE, (float)xoff * CELLSIZE_density, 100.0f - (yoff + _TILE_YSIZE) * CELLSIZE_density, CELLSIZE_density, -9999.0f, SERIALIZED_MODE, SERIALIZED_MODE);
					if (AsciiTile.compute_serialized) {
						#pragma omp parallel for private(p_x, p_y, p_w, cell_x, cell_y, e_w)
						for (long long int idx = 0; idx < AsciiTile.nVals; idx++) {
							int row = AsciiTile.rowcolIdx[idx] / AsciiTile.nCols;
							int col = AsciiTile.rowcolIdx[idx] % AsciiTile.nCols;
							cell_y = ROW_TO_YCOORD(row, AsciiTile.nRows, AsciiTile.yLLCorner, AsciiTile.cellSize);							
							cell_x = COL_TO_XCOORD(col, AsciiTile.xLLCorner, AsciiTile.cellSize);							

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
								double d2 = Distance2(p_x, p_y, cell_x, cell_y);
								den += GaussianKernel(hp * hp, d2) * p_w * e_w;
							}
							AsciiTile.elementsVals[idx] = den; // intensity, not probability
							AsciiTile.elements[row * AsciiTile.nCols + col] = den; // intensity, not probability
						}					
					}
					else {
						#pragma omp parallel for private(p_x, p_y, p_w, cell_x, cell_y, e_w)
						for (int row = 0; row < AsciiTile.nRows; row++) {
							cell_y = ROW_TO_YCOORD(row, AsciiTile.nRows, AsciiTile.yLLCorner, AsciiTile.cellSize);
							for (int col = 0; col < AsciiTile.nCols; col++) {
								cell_x = COL_TO_XCOORD(col, AsciiTile.xLLCorner, AsciiTile.cellSize);
								//int idx = row * nCols + col;
								size_t idx = (size_t)row * AsciiTile.nCols + (size_t)col;
								if (AsciiTile.elements[idx] != noDataValue) {

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
										double d2 = Distance2(p_x, p_y, cell_x, cell_y);
										den += GaussianKernel(hp * hp, d2) * p_w * e_w;
									}
									AsciiTile.elements[idx] = den; // intensity, not probability
								}
							}
						}
					}
					
					auto t2 = high_resolution_clock::now();
					duration<float, std::milli> ms_float = t2 - t1;
					printf("...KernelDesityEstimation on tile %d took %.0f ms\n", cnt, ms_float.count());
					
					tiffwritergpu.WriteGeoTIFF_NextTile(xoff, yoff, _TILE_XSIZE, _TILE_YSIZE, AsciiTile.elements);

					// clean up
					FreeAsciiRaster(&AsciiTile);

					cnt += 1;

					// set up for next tile
					if (xoff + TILE_XSIZE < _NCOLS) {
						xoff += TILE_XSIZE;
					}
					else {
						if (yoff + TILE_YSIZE < _NROWS) {
							xoff = 0;
							yoff += TILE_YSIZE;
						}
						else {
							xoff = _NCOLS;
							yoff = _NROWS;
						}
					}
				}
			}

			auto t2_kde = high_resolution_clock::now();
			ms_float = t2_kde - t1_kde;
			printf("KernelDesityEstimation took %.0f ms\n", ms_float.count());
			
			// write results to file
			//WriteGeoTIFFRaster(&DenSurf, denOMPfn);
			WriteSamplePoints(&sPoints, hs, gedgeWeights, "pntsCPU.csv");

			auto t2_cpu = high_resolution_clock::now();
			ms_float = t2_cpu - t1_cpu;
			printf("\n>>>>>>Computation on CPU took %.0f ms in total\n\n", ms_float.count());
		}
		////////////////////////// END OF PARALLEL COMPUTER ON MULTI-CORE CPU (OpenMP) //////////////////////////////

		//////////////////////////  PARALLEL COMPUTING ON GPUs (CUDA)  /////////////////////////////////////////
		if (SKIPCUDA == 0) {

			auto t1_gpu = high_resolution_clock::now();

			GPU_tree = new CUDA_KDTree[GPU_N];
			gpuDen = new float* [GPU_N];

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

			float* zeroDen = AllocateDen(sPoints.numberOfPoints);

			for (int i = 0; i < numPoints; i++) {
				if (Hoption > -1) hs[i] = h;
				zeroDen[i] = 0.0f;
			}

			float** dHs = new float* [GPU_N];
			AllocateDeviceBandwidths(dHs, sPoints.numberOfPoints);

			float** dDen0 = new float* [GPU_N];
			AllocateDeviceDen(dDen0, sPoints.numberOfPoints);

			float** dDen0cpy = new float* [GPU_N];
			AllocateDeviceDen(dDen0cpy, sPoints.numberOfPoints);

			float** dDen1 = new float* [GPU_N];
			AllocateDeviceDen(dDen1, sPoints.numberOfPoints);

			AllocateDeviceDen(gpuDen, sPoints.numberOfPoints);

			CopyToDeviceBandwidths(dHs, hs, sPoints.numberOfPoints);

			CopyToDeviceSamplePoints(dPoints, sPoints);

			for (int i = 0; i < GPU_N; i++) {
				printf("...Device %d start %d end %d [points]\n", i, dPoints[i].start, dPoints[i].end);

			}

			CopyToDeviceAsciiRaster(dAscii, Mask);

			CopyToDeviceDen(gpuDen, zeroDen, sPoints.numberOfPoints);

			if (UPDATEWEIGHTS) {
				MaskBoundary = CopyAsciiRaster(Mask); // used only for computing point distance to boundary. 

				AsciiRaster* dAsciiBoundary = new AsciiRaster[GPU_N];
				AllocateDeviceAsciiRaster(dAsciiBoundary, MaskBoundary);
				CopyToDeviceAsciiRaster(dAsciiBoundary, MaskBoundary);

				MarkBoundary(dAsciiBoundary, MaskBoundary, true); // either on GPU or CPU

				FreeDeviceAsciiRaster(dAsciiBoundary);
				
				AsciiRaster* dAsciiBoundary1 = new AsciiRaster[GPU_N];

				AllocateDeviceAsciiRaster(dAsciiBoundary1, MaskBoundary);
				CopyToDeviceAsciiRaster(dAsciiBoundary1, MaskBoundary);

				CalcDist2Boundary(dPoints, dAsciiBoundary1, true);

				FreeDeviceAsciiRaster(dAsciiBoundary1);

				if (Hoption == -1) {
					SortSamplePoints(sPoints, hs);
					CopyToDeviceBandwidths(dHs, hs, sPoints.numberOfPoints);
				}
				else {
					SortSamplePoints(sPoints);
				}

				//When adding back sorted points, divide points as they are copied accross GPUs
				CopyToDeviceSamplePoints(dPoints, sPoints);
			}

			if (SKIPOMP == 1) {
				//printf("bulding kdtree on CPU since it has not been built yet\n");
				BuildCPUKDtree(sPoints);
			}
			BuildGPUKDtree(); // needs to build the CPUKDtree first

			for (int i = 0; i < GPU_N; i++) {
				printf("...Device %d start %d end %d [points]\n", i, dPoints[i].start, dPoints[i].end);
			}

			if (Hoption == -1) { // use bandwidth provided in file
				if (UPDATEWEIGHTS)
				{	
					t1 = high_resolution_clock::now();
					//Run Kernel Asynchronously accross GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
					}
					ReformECWeights(dWeights, gedgeWeights);

					t2 = high_resolution_clock::now();
					ms_float = t2 - t1;
					printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
				}
			}
			else{
				if (UPDATEWEIGHTS)
				{	
					t1 = high_resolution_clock::now();
					//Run Kernel Asynchronously accross GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (h * h, dPoints[i], dAscii[i], dWeights[i]);
					}
					ReformECWeights(dWeights, gedgeWeights);
					t2 = high_resolution_clock::now();
					ms_float = t2 - t1;
					printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
				}

				/////////////////////////////////////////////////////////////////////////////////////////
				int numPoints = sPoints.numberOfPoints;

					//////////////////////////////////////////////////
				if (Hoption == 1) {
					float hopt = MLE_FixedBandWidth(dAscii, dPoints, dWeights, h, NULL, den1, true, NULL, dDen1);

					printf("cross validated optimal fixed bandwidth hopt: %.5f\n", hopt);
					//Running following kernels accross all GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						CalcVaryingBandwidths <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dPoints[i], hopt, dHs[i]);
					}

					ReformBandwidths(dHs, hs);

					if (UPDATEWEIGHTS)
					{	
						t1 = high_resolution_clock::now();
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);

							int pNum = dPoints[i].end - dPoints[i].start;
							int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
							dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

							CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
						}
						ReformECWeights(dWeights, gedgeWeights);

						t2 = high_resolution_clock::now();
						ms_float = t2 - t1;
						printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
					}
				}

				if (Hoption == 2) {
					// these parameters may need to be change based on the characteristics of a dataset
					float h0 = h;
					float alpha0 = -0.5;
					float stepH = h0 / 10;
					float stepA = 0.1;
					float* optParas = (float*)malloc(3 * sizeof(float));

					hooke_jeeves(dAscii, dPoints, dPoints, dWeights, h0, alpha0, stepH, stepA, optParas, hs, den0, den1, true, dHs, dDen0, dDen1, dDen0cpy);
					h0 = optParas[0];
					alpha0 = optParas[1];
					float logL = optParas[2];

					if (DEBUG) printf("h0: %.5f alpha0: %.5f Lmax: %.5f\n", h0, alpha0, logL);
					free(optParas);
					optParas = NULL;

					ComputeFixedDensityDevice(streams, dAscii, dPoints, dWeights, h0, den0, dDen0);

					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);
						CopyDeviceDen(dDen0cpy[i], dDen0[i], numPoints);
					}	

					reductionSum = ReductionSumGPU(dDen0cpy, numPoints);

					// update bandwidth on GPU
					//Running following kernels accross all GPUs
					for (int i = 0; i < GPU_N; i++)
					{
						printf("...CalcVaryingBandwidths on device %d\n", i + GPU_START);
						cudaSetDevice(i + GPU_START);

						int pNum = dPoints[i].end - dPoints[i].start;
						int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
						dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

						CalcVaryingBandwidths <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dPoints[i], dDen0[i], h0, alpha0, dHs[i], reductionSum);
					}
					ReformBandwidths(dHs, hs);
				
					// update weights
					if (UPDATEWEIGHTS) {
						t1 = high_resolution_clock::now();
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);

							int pNum = dPoints[i].end - dPoints[i].start;
							int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
							dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

							CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dHs[i], dPoints[i], dAscii[i], dWeights[i]);
						}

						ReformECWeights(dWeights, gedgeWeights);

						t2 = high_resolution_clock::now();
						ms_float = t2 - t1;
						printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
					}
				}
			}

			t1 = high_resolution_clock::now();
			// clean up
			FreeDeviceSamplePoints(dPoints);
			FreeDeviceEdgeCorrectionWeights(dWeights);
			FreeDeviceAsciiRaster(dAscii);
			FreeDeviceBandwidths(dHs);
			FreeDeviceDen(dDen0);
			FreeDeviceDen(dDen0cpy);
			FreeDeviceDen(dDen1);
			FreeDen(zeroDen);
			FreeDeviceDen(gpuDen);
			t2 = high_resolution_clock::now();
			ms_float = t2 - t1;
			printf("...cleaning up took %.0f ms\n", ms_float.count());

/////////////////////////////////////////////START::::KERNEL DENSITY ESTIMATION IN TILE FASHION/////////////////////////////////////////////////////////////
			
			auto t1_kde = high_resolution_clock::now();

			// figure out how much memory is left on GPUs (the minimum of all GPUs)
			size_t gpuMem_free_min = ULLONG_MAX;
			size_t gpuMem_free;
			size_t gpuMem_total;
			for (int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(i + GPU_START);
				cudaMemGetInfo(&gpuMem_free, &gpuMem_total);
				if (gpuMem_free < gpuMem_free_min) {
					gpuMem_free_min = gpuMem_free;
				}
				printf("...Device %d: gpuMem_free=%.2f GB, gpuMem_total=%.2f GB\n", i + GPU_START, gpuMem_free / 1024.0 / 1024.0 / 1024.0, gpuMem_total / 1024.0 / 1024.0 / 1024.0);
			}
			printf("...gpuMem_free_min=%.2f GB\n", gpuMem_free_min / 1024.0 / 1024.0 / 1024.0);
			cudaSetDevice(GPU_START);

			int BLOCK_XSIZE, BLOCK_YSIZE;
			int _NCOLS, _NROWS;

			if (fromFiles) {
				// figure out GeoTIFF block size
				GeoTIFFReader tiffreadertmp((const char*)maskFn_density);
				tiffreadertmp.GetBlockXYSize(BLOCK_XSIZE, BLOCK_YSIZE);
				int* dims = tiffreadertmp.GetDimensions();
				_NCOLS = dims[0];
				_NROWS = dims[1];
			}
			else {
				BLOCK_XSIZE = 128;
				BLOCK_YSIZE = 128;
				_NCOLS = int(ceil(100.0 / CELLSIZE_density));
				_NROWS = int(ceil(100.0 / CELLSIZE_density));
			}
			// size of a single block
			size_t nBytes_BLOCK = sizeof(float) * (size_t)BLOCK_XSIZE * (size_t)BLOCK_YSIZE * 3;
			// can read in up to nBlocks blocks
			int nBlocks = (int)floor(gpuMem_free_min * 0.2 / nBytes_BLOCK);
			printf("...can read in %d blocks at once\n", nBlocks);

			// how many blocks row-wise and col-wise
			int nBlocksX = (int)ceil(_NCOLS * 1.0 / BLOCK_XSIZE);
			int nBlocksY = (int)ceil(_NROWS * 1.0 / BLOCK_YSIZE);

			printf("...GeoTIFF: %d blocks col-wise,  %d blocks row-wise, %d blocks in total\n", nBlocksX, nBlocksY, nBlocksX* nBlocksY);

			// read in the whole GeoTIFF, by default
			int TILE_XSIZE = _NCOLS; // 10 * BLOCK_XSIZE
			int TILE_YSIZE = _NROWS; // 10 * BLOCK_YSIZE

			if (nBlocks < nBlocksX * nBlocksY) {

				int TILE_nBLOCKS_Y = 1; // tile height in # of blocks 
				int TILE_nBLOCKS_X = 1; // tile width in # of blocks

				while (TILE_nBLOCKS_Y < nBlocksY / 2) {

					if (TILE_nBLOCKS_Y * TILE_nBLOCKS_X < nBlocks) {
						TILE_nBLOCKS_Y += 1;

						while (TILE_nBLOCKS_X <= nBlocksX / 2) {

							if (TILE_nBLOCKS_Y * TILE_nBLOCKS_X < nBlocks) {
								TILE_nBLOCKS_X += 1;
								break;
							}
							else {
								TILE_nBLOCKS_X -= 1;
								break;
							}
						}
					}
					else {
						TILE_nBLOCKS_Y -= 1;
						break;
					}
				}
				TILE_XSIZE = TILE_nBLOCKS_X * BLOCK_XSIZE;
				TILE_YSIZE = TILE_nBLOCKS_Y * BLOCK_YSIZE;

				printf("...TILE_nBLOCKS_X=%d, TILE_nBLOCKS_Y=%d\n", TILE_nBLOCKS_X, TILE_nBLOCKS_Y);
			}
			printf("...TILE_XSIZE=%d, TILE_YSIZE=%d\n", TILE_XSIZE, TILE_YSIZE);

			if (fromFiles) {				
				
				GeoTIFFReader tiffreadergpu((const char*)maskFn_density, TILE_XSIZE, TILE_YSIZE);

				GeoTIFFWriter tiffwritergpu = GeoTIFFWriter((const char*)denCUDAfn, tiffreadergpu.GetGeoTransform(), tiffreadergpu.GetProjection(), _NROWS, _NCOLS, tiffreadergpu.GetNoDataValue());

				float** tileDatagpu;
				tileDatagpu = tiffreadergpu.GetRasterBand_NextTile(1);

				while (tiffreadergpu.GetNTiles() > 0) {

					t1 = high_resolution_clock::now();

					printf("...working on tile %d / %d\n", tiffreadergpu.GetNTiles(), tiffreadergpu.GetNTilesTotal());

					int* para = tiffreadergpu.GetCurTileParam();
					int XOFF = para[0];
					int YOFF = para[1];
					int XSIZE = para[2];
					int YSIZE = para[3];

					AsciiRaster AsciiTile = AsciiRasterFromGeoTIFFTile(tiffreadergpu.GetGeoTransformTile(), tiffreadergpu.GetProjection(), YSIZE, XSIZE, tiffreadergpu.GetNoDataValue(), tileDatagpu, SERIALIZED_MODE, SERIALIZED_MODE);
					if (AsciiTile.nVals > 0) {

						AsciiRaster* dAsciiTile = new AsciiRaster[GPU_N];
						AllocateDeviceAsciiRaster(dAsciiTile, AsciiTile);
						CopyToDeviceAsciiRaster(dAsciiTile, AsciiTile);

						// set cell densities in dAscii to 0.0f
						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							// invoke kernel to do density estimation
							size_t n_cells;
							if (dAsciiTile[i].compute_serialized) {
								n_cells = dAsciiTile[i].nVals;
							}
							else {
								n_cells = dAsciiTile[i].nRows * dAsciiTile[i].nCols;
							}
							int NBLOCK_K = (n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
							dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);
							InitCellDensities <<<dimGrid_K, BLOCK_SIZE, 0, streams[i] >>> (dAsciiTile[i]);
						}

						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							cudaStreamSynchronize(streams[i]);
						}

						/////////////////////////////////////////////////////////////////////////////////

						vector<int> index;						
						SamplePoints pointsTile = ExtractSamplePointsCurTile(AsciiTile, hs, index);						
						int N = index.size();
						if (N > 0) {

							SamplePoints* dPointsTile = new SamplePoints[GPU_N];
							AllocateDeviceSamplePoints(dPointsTile, pointsTile);
							CopyToDeviceSamplePoints(dPointsTile, pointsTile);

							float* hsTile = AllocateBandwidths(N);
							float* ewTile = AllocateEdgeCorrectionWeights(pointsTile);
							#pragma omp parallel for
							for (int i = 0; i < N; i++) {
								int idx = index[i];
								hsTile[i] = hs[idx];
								ewTile[i] = gedgeWeights[idx];
							}
							float** dWeightsTile = new float* [GPU_N];
							AllocateDeviceEdgeCorrectionWeights(dWeightsTile, pointsTile);
							CopyToDeviceWeights(dWeightsTile, ewTile, pointsTile.numberOfPoints);

							float** dHsTile = new float* [GPU_N];
							AllocateDeviceBandwidths(dHsTile, pointsTile.numberOfPoints);
							CopyToDeviceBandwidths(dHsTile, hsTile, pointsTile.numberOfPoints);

							size_t n_cells;
							if (AsciiTile.compute_serialized) {
								n_cells = AsciiTile.nVals;
							}
							else {
								n_cells = AsciiTile.nRows * AsciiTile.nCols;
							}

							for (int i = 0; i < GPU_N; i++)
							{
								printf("...KernelDesityEstimation on device %d\n", i + GPU_START);
								cudaSetDevice(i + GPU_START);
																
								printf("---Parallel on sample points\n");
								// invoke kernel to do density estimation
								int NBLOCK_K = (dPointsTile[i].end - dPointsTile[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
								int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
								dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);

								KernelDesityEstimation_pPoints <<<dimGrid_K, BLOCK_SIZE, 0, streams[i] >> > (dHsTile[i], dPointsTile[i], dAsciiTile[i], dWeightsTile[i]);
								

							}

							for (int i = 0; i < GPU_N; i++)
							{
								cudaSetDevice(i + GPU_START);
								cudaStreamSynchronize(streams[i]);
							}

							FreeDeviceSamplePoints(dPointsTile);
							FreeDeviceBandwidths(dHsTile);
							FreeDeviceEdgeCorrectionWeights(dWeightsTile);							
							FreeBandwidths(hsTile);
							FreeEdgeCorrectionWeights(ewTile);
						}
						FreeSamplePoints(&pointsTile);
						//cudaSetDevice(GPU_START);

						ReformGPUAsciiRaster(dAsciiTile, AsciiTile);

						// clean up
						FreeDeviceAsciiRaster(dAsciiTile);
						

					}
					t2 = high_resolution_clock::now();
					duration<float, std::milli> ms_float = t2 - t1;
					printf("...KernelDesityEstimation on tile %d took %.0f ms\n", tiffreadergpu.GetNTiles(), ms_float.count());

					// write results to file
					tiffwritergpu.WriteGeoTIFF_NextTile(XOFF, YOFF, XSIZE, YSIZE, AsciiTile.elements);

					// clean up
					FreeAsciiRaster(&AsciiTile);
					delete tileDatagpu;

					// retrieve the next tile
					tileDatagpu = tiffreadergpu.GetRasterBand_NextTile(1);
				}
			}
			else { // based on data generated on the fly				

				double geotransform[6] = {0, CELLSIZE_density, 0, 100.0, 0, -1.0* CELLSIZE_density};
				GeoTIFFWriter tiffwritergpu = GeoTIFFWriter((const char*)denCUDAfn, geotransform, NULL, _NROWS, _NCOLS, -9999.0f);

				int xoff = 0;
				int yoff = 0;

				int cnt = 1;
				int NTILES_TOTAL = (int)ceil(_NCOLS * 1.0 / TILE_XSIZE) * (int)ceil(_NROWS * 1.0 / TILE_YSIZE);
				while (xoff < _NCOLS && yoff < _NROWS) {

					printf("...working on tile %d / %d\n", cnt, NTILES_TOTAL);
					t1 = high_resolution_clock::now();
					int _TILE_YSIZE = TILE_YSIZE;
					int _TILE_XSIZE = TILE_XSIZE;
					
					if (TILE_YSIZE > _NROWS - yoff) {
						_TILE_YSIZE = _NROWS - yoff;
					}

					if (TILE_XSIZE > _NCOLS - xoff) {
						_TILE_XSIZE = _NCOLS - xoff;
					}

					AsciiRaster AsciiTile = AllocateAsciiRaster(_TILE_XSIZE, _TILE_YSIZE, (float)xoff * CELLSIZE_density, 100.0f - (yoff + _TILE_YSIZE) * CELLSIZE_density, CELLSIZE_density, -9999.0f, SERIALIZED_MODE, SERIALIZED_MODE);

					AsciiRaster* dAsciiTile = new AsciiRaster[GPU_N];
					AllocateDeviceAsciiRaster(dAsciiTile, AsciiTile);
					CopyToDeviceAsciiRaster(dAsciiTile, AsciiTile);
					
					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);
						// invoke kernel to do density estimation
						size_t n_cells;
						if (dAsciiTile[i].compute_serialized) {
							n_cells = dAsciiTile[i].nVals;
						}
						else {
							n_cells = dAsciiTile[i].nRows * dAsciiTile[i].nCols;
						}

						int NBLOCK_K = (n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
						int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
						dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);
						InitCellDensities <<< dimGrid_K, BLOCK_SIZE, 0, streams[i] >>> (dAsciiTile[i]);
					}

					for (int i = 0; i < GPU_N; i++)
					{
						cudaSetDevice(i + GPU_START);
						cudaStreamSynchronize(streams[i]);
					}

					/////////////////////////////////////////////////////////////////////////////////

					vector<int> index;
					SamplePoints pointsTile = ExtractSamplePointsCurTile(AsciiTile, hs, index);

					int N = index.size();
					if (N > 0) {

						SamplePoints* dPointsTile = new SamplePoints[GPU_N];
						AllocateDeviceSamplePoints(dPointsTile, pointsTile);
						CopyToDeviceSamplePoints(dPointsTile, pointsTile);

						float* hsTile = AllocateBandwidths(N);
						float* ewTile = AllocateEdgeCorrectionWeights(pointsTile);
						#pragma omp parallel for
						for (int i = 0; i < N; i++) {
							int idx = index[i];
							hsTile[i] = hs[idx];
							ewTile[i] = gedgeWeights[idx];
						}
						float** dWeightsTile = new float* [GPU_N];
						AllocateDeviceEdgeCorrectionWeights(dWeightsTile, pointsTile);
						CopyToDeviceWeights(dWeightsTile, ewTile, pointsTile.numberOfPoints);

						float** dHsTile = new float* [GPU_N];
						AllocateDeviceBandwidths(dHsTile, pointsTile.numberOfPoints);
						CopyToDeviceBandwidths(dHsTile, hsTile, pointsTile.numberOfPoints);

						for (int i = 0; i < GPU_N; i++)
						{
							printf("...KernelDesityEstimation on device %d\n", i + GPU_START);
							cudaSetDevice(i + GPU_START);
							
							printf("---Parallel on sample points\n");
							// invoke kernel to do density estimation
							int NBLOCK_K = (dPointsTile[i].end - dPointsTile[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
							int GRID_SIZE_K = (int)(sqrtf(NBLOCK_K)) + 1;
							dim3 dimGrid_K(GRID_SIZE_K, GRID_SIZE_K);

							KernelDesityEstimation_pPoints <<<dimGrid_K, BLOCK_SIZE, 0, streams[i] >> > (dHsTile[i], dPointsTile[i], dAsciiTile[i], dWeightsTile[i]);
							
						}

						for (int i = 0; i < GPU_N; i++)
						{
							cudaSetDevice(i + GPU_START);
							cudaStreamSynchronize(streams[i]);
						}

						FreeDeviceSamplePoints(dPointsTile);
						FreeDeviceBandwidths(dHsTile);
						FreeDeviceEdgeCorrectionWeights(dWeightsTile);
						FreeBandwidths(hsTile);
						FreeEdgeCorrectionWeights(ewTile);
					}

					auto t2 = high_resolution_clock::now();
					duration<float, std::milli> ms_float = t2 - t1;
					printf("...KernelDesityEstimation on tile %d took %.0f ms\n", cnt, ms_float.count());
					
					FreeSamplePoints(&pointsTile);

					//cudaSetDevice(GPU_START);
					ReformGPUAsciiRaster(dAsciiTile, AsciiTile);
					
					// clean up
					FreeDeviceAsciiRaster(dAsciiTile);

					tiffwritergpu.WriteGeoTIFF_NextTile(xoff, yoff, _TILE_XSIZE, _TILE_YSIZE, AsciiTile.elements);
					
					FreeAsciiRaster(&AsciiTile);
					
					cnt += 1;

					// set up for next tile
					if (xoff + TILE_XSIZE < _NCOLS) {
						xoff += TILE_XSIZE;
					}
					else {
						if (yoff + TILE_YSIZE < _NROWS) {
							xoff = 0;
							yoff += TILE_YSIZE;
						}
						else {
							xoff = _NCOLS;
							yoff = _NROWS;
						}
					}
				}
			}
			auto t2_kde = high_resolution_clock::now();
			ms_float = t2_kde - t1_kde;
			printf("KernelDensityEstimation took %.0f ms\n", ms_float.count());

/////////////////////////////////////////////END::::KERNEL DENSITY ESTIMATION IN TILE FASHION/////////////////////////////////////////////////////////////

			WriteSamplePoints(&sPoints, hs, gedgeWeights, (const char*)"pntsCUDA.csv");

/////////////////////////////////////////////////////////////////////////////	

			auto t2_gpu = high_resolution_clock::now();
			/* Getting number of milliseconds as a float. */
			ms_float = t2_gpu - t1_gpu;
			printf("\n>>>>>>Computation on GPU took %.0f ms in total\n\n", ms_float.count());
			
		}
		t1 = high_resolution_clock::now();

		FreeDen(den0);
		FreeDen(den1);

		FreeAsciiRaster(&Mask);
		FreeSamplePoints(&sPoints);


		FreeBandwidths(hs);
		FreeEdgeCorrectionWeights(gedgeWeights);

		t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a float. */
		ms_float = t2 - t1;
		printf("...cleaning up took %.0f ms\n", ms_float.count());

		//printf("MAX_N_NBRS=%d\n", MAX_N_NBRS);
		printf("Done...\n\n");

		auto T2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a float. */
		duration<float, std::milli> MS_float = T2 - T1;
		printf("+++In total it took %.0f ms\n", MS_float.count());

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
void StandardDistance2(SamplePoints Points, double &d2){

	float mean_x, mean_y;
	MeanCenter(Points, mean_x, mean_y);

	double sum2 = 0.0;
	double sum_w = 0.0;
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
	Points.distances = (double*)malloc(n * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		Points.xCoordinates[i] = rand() * 100.0f / RAND_MAX;
		Points.yCoordinates[i] = rand() * 100.0f / RAND_MAX;
		Points.weights[i] = 1.0f;
		Points.distances[i] = 0.0; 
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
	//n = 1000;
	Points.numberOfPoints = n;
	Points.xCoordinates = (float*)malloc(n*sizeof(float));
	Points.yCoordinates = (float*)malloc(n*sizeof(float));
	Points.weights = (float*)malloc(n*sizeof(float));
	Points.distances = (double*)malloc(n*sizeof(double));

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
		
		if (bandwidths) { 
			Points.distances[counter] = h;
		}
		else{
			Points.distances[counter] = 0.0; 
		}

		counter++;
	}

	fclose(f);

	return Points;
}

void AllocateDeviceSamplePoints(SamplePoints* dPoints, const SamplePoints Points){
	//Changing dPoints to be a array of pointers to each set of points on each device.
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
			printf("***ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc((void**)&dPoints[i].yCoordinates, size);
		if (error != cudaSuccess)
		{
			printf("***ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc((void**)&dPoints[i].weights, size);
		if (error != cudaSuccess)
		{
			printf("***ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaMalloc((void**)&dPoints[i].distances, Points.numberOfPoints * sizeof(double));
		if (error != cudaSuccess)
		{
			printf("***ERROR in AllocateDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

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
		error = cudaMemcpy(dPoints[i].distances, hPoints.distances, hPoints.numberOfPoints * sizeof(double), cudaMemcpyHostToDevice);
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

	error = cudaMemcpy(hPoints.distances, dPoints[0].distances, dPoints[0].numberOfPoints * sizeof(double), cudaMemcpyDeviceToHost);
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


void FreeSamplePoints(SamplePoints* Points) {
	free(Points->xCoordinates);
	Points->xCoordinates = NULL;

	free(Points->yCoordinates);
	Points->yCoordinates = NULL;

	free(Points->weights);
	Points->weights = NULL;

	free(Points->distances);
	Points->distances = NULL;
}

void FreeDeviceSamplePoints(SamplePoints* dPoints){
	cudaError_t error;
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

		error = cudaFree(dPoints[i].distances);
		if (error != cudaSuccess)
		{
			printf("***ERROR in FreeDeviceSamplePoints: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		dPoints->distances = NULL;
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// this is a mask
AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue, bool data_serialized, bool compute_serialized){
	printf("...starting AllocateAsciiRaster()\n");
	if (compute_serialized && !data_serialized) {
		printf("***ERROR: data_serialized has to be true in order to set compute_serialized true: AllocateAsciiRaster()\n");
		exit(1);
	}
	
	AsciiRaster Ascii;
	Ascii.data_serialized = data_serialized;
	Ascii.compute_serialized = compute_serialized;

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
	
	Ascii.nVals = Ascii.nCols * Ascii.nRows;
	if (Ascii.data_serialized) {
		Ascii.rowcolIdx = (int*)malloc(Ascii.nVals * sizeof(int));  // Array to hold the seqential index of non-nodata values, row * nCol + col
		Ascii.elementsVals = (float*)malloc(Ascii.nVals * sizeof(float));  // Array to hold the non-nodata values	
		if (Ascii.compute_serialized) {
			Ascii.start = 0;
			Ascii.end = Ascii.nVals;
		}
	}
	
	size_t size = Ascii.nCols * Ascii.nRows;
	Ascii.elements = (float*)malloc(size * sizeof(float));
	size_t idx;
	for (size_t row = 0; row < Ascii.nRows; row++){
		for (size_t col = 0; col < Ascii.nCols; col++){
			idx = row * Ascii.nCols + col;
			Ascii.elements[idx] = 0.0f;

			// Guiming 2022-01-20
			if (Ascii.data_serialized) {
				Ascii.rowcolIdx[idx] = idx;
				Ascii.elementsVals[idx] = 0.0f;
			}
		}
	}
	printf("...done AllocateAsciiRaster()\n");
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

	Ascii.nVals = anotherAscii.nVals;
	Ascii.data_serialized = anotherAscii.data_serialized;
	Ascii.compute_serialized = anotherAscii.compute_serialized;

	if (Ascii.data_serialized) {
		Ascii.rowcolIdx = (int*)malloc(Ascii.nVals * sizeof(int));
		Ascii.elementsVals = (float*)malloc(Ascii.nVals * sizeof(float));
		for (size_t rcidx = 0; rcidx < Ascii.nVals; rcidx++) {
			Ascii.rowcolIdx[rcidx] = anotherAscii.rowcolIdx[rcidx];
			Ascii.elementsVals[rcidx] = anotherAscii.elementsVals[rcidx];
		}
	}

	return Ascii;
}

// ascii raster read from a .asc file
AsciiRaster ReadGeoTIFFRaster(char* geotiffFile, bool data_serialized, bool compute_serialized) {
	
	if (compute_serialized && !data_serialized) {
		printf("***ERROR: data_serialized has to be true in order to set compute_serialized true: ReadGeoTIFFRaster()\n");
		exit(1);
	}

	AsciiRaster Ascii;
	Ascii.data_serialized = data_serialized;
	Ascii.compute_serialized = compute_serialized;

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
	Ascii.yLLCorner = (float)(gt[3] + dims[1] * gt[5]);

	//printf("Ascii.yLLCorner=%f compute=%f\n", Ascii.yLLCorner, gt[3] + dims[1] * gt[5]);

	Ascii.cellSize = (float)gt[1];
	Ascii.noDataValue = (float)tiff.GetNoDataValue();

	Ascii.start = 0;
	Ascii.end = Ascii.nCols * Ascii.nRows;

	Ascii.elements = (float*)malloc(Ascii.nRows * Ascii.nCols * sizeof(float));

	int nVals = 0; // # of non-nodata values

	float** data = tiff.GetRasterBand(1);
	for (int row = 0; row < Ascii.nRows; row++) {
		for (int col = 0; col < Ascii.nCols; col++) {
			size_t idx = row * Ascii.nCols + col;
			Ascii.elements[idx] = data[row][col];

			if (Ascii.elements[idx] != Ascii.noDataValue) 
				nVals += 1;
		}
	}

	Ascii.nVals = nVals;
	if (Ascii.data_serialized) {
		Ascii.rowcolIdx = (int*)malloc(Ascii.nVals * sizeof(int));
		Ascii.elementsVals = (float*)malloc(Ascii.nVals * sizeof(float));
		size_t rcidx = 0;
		for (int row = 0; row < Ascii.nRows; row++) {
			for (int col = 0; col < Ascii.nCols; col++) {
				if (data[row][col] != Ascii.noDataValue) {
					Ascii.rowcolIdx[rcidx] = row * Ascii.nCols + col;
					Ascii.elementsVals[rcidx] = data[row][col];
					rcidx += 1;
				}
			}
		}
		if (Ascii.compute_serialized) {
			Ascii.start = 0;
			Ascii.end = Ascii.nVals;
		}	
	}

	return Ascii;
}

void AsciiRasterSwitchDataSerialization(AsciiRaster* Ascii, bool data_serialized) {
	if (Ascii -> data_serialized == data_serialized) {
		return;
	}
	else {
		if (Ascii->data_serialized && !data_serialized) {
			
			if (Ascii->rowcolIdx != NULL) {
				free(Ascii->rowcolIdx);
				Ascii->rowcolIdx = NULL;
			}

			if (Ascii->elementsVals != NULL) {
				free(Ascii->elementsVals);
				Ascii->elementsVals = NULL;
			}
			Ascii->data_serialized = false;
			
			Ascii->start = 0;
			Ascii->start = Ascii->nRows * Ascii->nCols;
			Ascii->compute_serialized = false;
		}
		else { //(!Ascii->data_serialized) && data_serialized
			
			Ascii->rowcolIdx = (int*)malloc(Ascii->nVals * sizeof(int));
			Ascii->elementsVals = (float*)malloc(Ascii->nVals * sizeof(float));
			
			//printf(".....%d\n", Ascii->rowcolIdx[0]);

			int rcidx = 0;
			for (int row = 0; row < Ascii->nRows; row++) {
				for (int col = 0; col < Ascii->nCols; col++) {
					int idx = row * Ascii->nCols + col;
					
					float val = Ascii->elements[idx];	

					if (val != Ascii->noDataValue) {
						(Ascii->rowcolIdx)[rcidx] = idx;
						Ascii->elementsVals[rcidx] = val;
						rcidx += 1;
					}
				}
			}
			Ascii->data_serialized = true;
		}
	}
}

void AsciiRasterSwitchComputeSerialization(AsciiRaster* Ascii, bool compute_serialized) {
	
	if (Ascii->compute_serialized == compute_serialized) {
		return;
	}
	else {
		if (Ascii->compute_serialized && !compute_serialized) {
			Ascii->compute_serialized = false;
			Ascii->start = 0;
			Ascii->end = Ascii->nRows * Ascii->nCols;
		}
		else { //!Ascii->compute_serialized && compute_serialized
			
			if (!Ascii->data_serialized) {
				
				AsciiRasterSwitchDataSerialization(Ascii, true);
				
			}
			Ascii->compute_serialized = true;
			Ascii->start = 0;
			Ascii->end = Ascii->nVals;
		}
	}
}

AsciiRaster AsciiRasterFromGeoTIFFTile(double* geotransform, const char* projection, int nrows, int ncols, float nodata, float** data, bool data_serialized, bool compute_serialized) {
	
	printf("...starting AsciiRasterFromGeoTIFFTile()\n");

	if (compute_serialized && !data_serialized) {
		printf("***ERROR: data_serialized has to be true in order to set compute_serialized true: AsciiRasterFromGeoTIFFTile()\n");
		exit(1);
	}
	
	AsciiRaster Ascii;
	Ascii.data_serialized = data_serialized;
	Ascii.compute_serialized = compute_serialized;

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
	Ascii.yLLCorner = (float)(geotransform[3] + nrows * geotransform[5]);
	Ascii.cellSize = (float)geotransform[1];
	Ascii.noDataValue = (float)nodata;

	Ascii.start = 0;
	Ascii.end = Ascii.nCols * Ascii.nRows;

	Ascii.elements = (float*)malloc((size_t)Ascii.nRows * (size_t)Ascii.nCols * sizeof(float));
	
	int nVals = 0;

	for (int row = 0; row < Ascii.nRows; row++) {
		for (int col = 0; col < Ascii.nCols; col++) {
			size_t idx = (size_t)row * (size_t)Ascii.nCols + (size_t)col;
			Ascii.elements[idx] = data[row][col];

			if (Ascii.elements[idx] != Ascii.noDataValue) {
				nVals += 1;
			}
		}
	}

	Ascii.nVals = nVals;
	if (Ascii.data_serialized) {
		Ascii.rowcolIdx = (int*)malloc((size_t)Ascii.nVals * sizeof(int));
		Ascii.elementsVals = (float*)malloc((size_t)Ascii.nVals * sizeof(float));
		size_t rcidx = 0;
		for (int row = 0; row < Ascii.nRows; row++) {
			for (int col = 0; col < Ascii.nCols; col++) {
				if (data[row][col] != Ascii.noDataValue) {
					Ascii.rowcolIdx[rcidx] = (size_t)row * (size_t)Ascii.nCols + (size_t)col;
					Ascii.elementsVals[rcidx] = data[row][col];
					rcidx += 1;
				}
			}
		}
		if (Ascii.compute_serialized) {
			Ascii.start = 0;
			Ascii.end = Ascii.nVals;
		}
	}
	printf("...done AsciiRasterFromGeoTIFFTile()\n");
	return Ascii;
}

void AllocateDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii){
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		dAscii[i].nCols = hAscii.nCols;
		dAscii[i].nRows = hAscii.nRows;
		dAscii[i].xLLCorner = hAscii.xLLCorner;
		dAscii[i].yLLCorner = hAscii.yLLCorner;
		dAscii[i].cellSize = hAscii.cellSize;
		dAscii[i].noDataValue = hAscii.noDataValue;
		
		dAscii[i].nVals = hAscii.nVals;
		dAscii[i].start = hAscii.start;
		dAscii[i].end = hAscii.end;
		dAscii[i].data_serialized = hAscii.data_serialized;
		dAscii[i].compute_serialized = hAscii.compute_serialized;
		
		cudaError_t error;
		cudaSetDevice(i + GPU_START);

		size_t size;
		if (hAscii.compute_serialized) {
			size = hAscii.nVals * sizeof(float);
			printf("...size in AllocateDeviceAsciiRaster %llu (%d x %llu) x 2: %.2f MB\n", size, hAscii.nVals, sizeof(float), float(size * 2.0 / 1024.0/1024.0));
			error = cudaMalloc((void**)&dAscii[i].elementsVals, hAscii.nVals * sizeof(float));
			if (error != cudaSuccess)
			{
				printf("***ERROR in AllocateDeviceAsciiRaster->elementsVals: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			error = cudaMalloc((void**)&dAscii[i].rowcolIdx, size);
			if (error != cudaSuccess)
			{
				printf("***ERROR in AllocateDeviceAsciiRaster->rowcolIdx: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}		
		else {
			size = hAscii.nCols * hAscii.nRows * sizeof(float);
			printf("...size in AllocateDeviceAsciiRaster %llu (%d x %llu) x 1: %.2f MB\n", size, hAscii.nCols * hAscii.nRows, sizeof(float), float(size / 1024.0 / 1024.0));
			error = cudaMalloc((void**)&dAscii[i].elements, size);
			if (error != cudaSuccess)
			{
				printf("***ERROR in AllocateDeviceAsciiRaster->elements: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}	
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

void CopyToDeviceAsciiRaster(AsciiRaster* dAscii, const AsciiRaster hAscii){

	size_t n;
	if (!hAscii.compute_serialized) {
		n = hAscii.nCols * hAscii.nRows; //Number of cells on GPU
	}
	else {
		n = hAscii.nVals;
	}
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

		if (!hAscii.compute_serialized) {
			error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("***ERROR in CopyToDeviceAsciiRaster->elements: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		else {
			error = cudaMemcpy(dAscii[i].elementsVals, hAscii.elementsVals, n * sizeof(float), cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("ERROR in CopyToDeviceAsciiRaster->elementsVals: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}

			error = cudaMemcpy(dAscii[i].rowcolIdx, hAscii.rowcolIdx, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("ERROR in CopyToDeviceAsciiRaster->rowcolIdx: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		index = div; //Set starting index of next group of cells to the end of previous group.
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// Combine rasters across GPUs into a single raster and send the update raster back to GPUs
void ReformAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii) {
	
	AsciiRasterSwitchDataSerialization(&hAscii, dAscii[0].data_serialized);
	AsciiRasterSwitchComputeSerialization(&hAscii, dAscii[0].compute_serialized);

	AsciiRaster tmpAscii = CopyAsciiRaster(hAscii);

	size_t n;
	if (dAscii[0].compute_serialized) {
		n = hAscii.nVals;
	}
	else {
		n = hAscii.nCols * hAscii.nRows; //Number of TOTAL cells	
	}
	size_t size = n * sizeof(float);

	cudaError_t error = cudaSuccess;

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);	

		if (hAscii.compute_serialized) {
			error = cudaMemcpy(tmpAscii.elementsVals, dAscii[device].elementsVals, size, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("***ERROR 1.%d in ReformAsciiRaster (FROM device)->elementsVals: %s\n", device, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//Loop to merge copied chunk of points into hPoints
			for (size_t i = dAscii[device].start; i < dAscii[device].end; i++)
			{
				hAscii.elementsVals[i] = tmpAscii.elementsVals[i];
			}
		}
		else {
			//Copy all data from chunk to tempPoints
			error = cudaMemcpy(tmpAscii.elements, dAscii[device].elements, size, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("***ERROR 1.%d in ReformAsciiRaster (FROM device)->elements: %s\n", device, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//Loop to merge copied chunk of points into hPoints
			for (size_t i = dAscii[device].start; i < dAscii[device].end; i++)
			{
				hAscii.elements[i] = tmpAscii.elements[i];
			}
		}
		if (DEBUGREFORMING) printf("......Copying Ascii FROM Device %d \n", device);
	}

	if (hAscii.compute_serialized) {
		for (size_t i = 0; i < hAscii.nRows * hAscii.nCols; i++) {
			hAscii.elements[i] = hAscii.noDataValue;
		}
		for (size_t i = 0; i < hAscii.nVals; i++) {
			//hAscii.elements[hAscii.rowIdx[i] * hAscii.nCols + hAscii.colIdx[i]] = hAscii.elementsVals[i];
			hAscii.elements[hAscii.rowcolIdx[i]] = hAscii.elementsVals[i];
		}
	}

	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		if (hAscii.compute_serialized) {
			error = cudaMemcpy(dAscii[i].elementsVals, hAscii.elementsVals, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("***ERROR 2.%d in ReformAsciiRaster (To device)->elementsVals: %s\n", i, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		else {
			error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("***ERROR 2.%d in ReformAsciiRaster (To device)->elements: %s\n", i, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		if (DEBUGREFORMING) printf("......Copying Ascii TO Device %d \n", i);
	}
	//Cleanup
	cudaSetDevice(GPU_START); //Reset device to first GPU
	//Free temp points
	FreeAsciiRaster(&tmpAscii);
	if (DEBUGREFORMING) printf("***Reforming Ascii DONE\n");
}

// Guiming 2021-08-30 Add up cell densities from all devices (dAscii) into one single array
void ReformGPUAsciiRaster(AsciiRaster* dAscii, AsciiRaster hAscii) {

	AsciiRasterSwitchDataSerialization(&hAscii, dAscii[0].data_serialized);
	AsciiRasterSwitchComputeSerialization(&hAscii, dAscii[0].compute_serialized);

	AsciiRaster tmpAscii = CopyAsciiRaster(hAscii);

	size_t n;
	if (dAscii[0].compute_serialized) {
		n = hAscii.nVals;
		for (size_t i = 0; i < n; i++) {
			hAscii.elementsVals[i] = 0.0f;
		}
	}
	else {
		n = hAscii.nCols * hAscii.nRows; //Number of TOTAL cells	
		for (size_t i = 0; i < n; i++) {
			if (hAscii.elements[i] == hAscii.noDataValue) continue;
			hAscii.elements[i] = 0.0f;
		}
	}	

	size_t size = n * sizeof(float);

	cudaError_t error = cudaSuccess;

	for (int device = 0; device < GPU_N; device++)
	{
		cudaSetDevice(device + GPU_START);

		if (hAscii.compute_serialized) {
			error = cudaMemcpy(tmpAscii.elementsVals, dAscii[device].elementsVals, size, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("***ERROR 1.%d in ReformGPUAsciiRaster (FROM device)->elementsVals: %s\n", device, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//Loop to merge copied chunk of points into hPoints
			for (size_t i = 0; i < n; i++)
			{
				hAscii.elementsVals[i] += tmpAscii.elementsVals[i];
			}
		}
		else {

			//Copy all data from chunk to tempPoints
			error = cudaMemcpy(tmpAscii.elements, dAscii[device].elements, size, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("***ERROR 1.%d in ReformGPUAsciiRaster (FROM device)->elements: %s\n", device, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			//Loop to merge copied chunk of points into hPoints
			for (size_t i = 0; i < n; i++)
			{
				if (hAscii.elements[i] == hAscii.noDataValue) continue;
				hAscii.elements[i] += tmpAscii.elements[i];
			}
		}
		if (DEBUGREFORMING) printf("......Copying Ascii FROM Device %d \n", device);
	}

	if (hAscii.compute_serialized) {
		for (size_t i = 0; i < hAscii.nRows * hAscii.nCols; i++) {
			hAscii.elements[i] = hAscii.noDataValue;
		}
		for (size_t i = 0; i < hAscii.nVals; i++) {
			hAscii.elements[hAscii.rowcolIdx[i]] = hAscii.elementsVals[i];
		}
	}

	//Copy reformed dDen accross GPUs
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);

		if (hAscii.compute_serialized) {
			error = cudaMemcpy(dAscii[i].elementsVals, hAscii.elementsVals, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("***ERROR 2.%d in ReformGPUAsciiRaster (To device)->elementsVals: %s\n", i, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
		}
		else {
			error = cudaMemcpy(dAscii[i].elements, hAscii.elements, size, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("***ERROR 2.%d in ReformGPUAsciiRaster (To device)->elements: %s\n", i, cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
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
	
	AsciiRasterSwitchDataSerialization(&hAscii, dAscii.data_serialized);
	AsciiRasterSwitchComputeSerialization(&hAscii, dAscii.compute_serialized);
	
	hAscii.nCols = dAscii.nCols;
	hAscii.nRows = dAscii.nRows;
	hAscii.xLLCorner = dAscii.xLLCorner;
	hAscii.yLLCorner = dAscii.yLLCorner;
	hAscii.cellSize = dAscii.cellSize;
	hAscii.noDataValue = dAscii.noDataValue;

	hAscii.start = dAscii.start;
	hAscii.end = dAscii.end;

	hAscii.nVals = dAscii.nVals;

	size_t size;
	cudaError_t error;

	if (hAscii.compute_serialized) {
		size = dAscii.nVals * sizeof(float);
		error = cudaMemcpy(hAscii.elementsVals, dAscii.elementsVals, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("***ERROR in CopyFromDeviceAsciiRaster->elementsVals: %s\n", cudaGetErrorString(error));
			printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
			exit(EXIT_FAILURE);
		}

		error = cudaMemcpy(hAscii.rowcolIdx, dAscii.rowcolIdx, dAscii.nVals * sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("***ERROR in CopyFromDeviceAsciiRaster->rowcolIdx: %s\n", cudaGetErrorString(error));
			printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
			exit(EXIT_FAILURE);
		}

		for (size_t i = 0; i < hAscii.nRows * hAscii.nCols; i++) {
			hAscii.elements[i] = hAscii.noDataValue;
		}

		for (size_t i = 0; i < hAscii.nVals; i++) {
			hAscii.elements[hAscii.rowcolIdx[i]] = hAscii.elementsVals[i];
		}
	}
	else {
		size = dAscii.nCols * dAscii.nRows * sizeof(float);
		error = cudaMemcpy(hAscii.elements, dAscii.elements, size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("***ERROR in CopyFromDeviceAsciiRaster->elements: %s\n", cudaGetErrorString(error));
			printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
			exit(EXIT_FAILURE);
		}
	}
}

// write to .tif file
void WriteGeoTIFFRaster(AsciiRaster* Ascii, const char* geotiffFile) {
	GeoTIFFWriter tiffw;
	tiffw.WriteGeoTIFF(geotiffFile, Ascii->geotransform, Ascii->projection, (int)Ascii->nRows, (int)Ascii->nCols, (float)Ascii->noDataValue, Ascii->elements);
	
}


void FreeAsciiRaster(AsciiRaster* Ascii){
	if (Ascii->geotransform != NULL) {
		free(Ascii->geotransform);
		Ascii->geotransform = NULL;
	}
	if (Ascii->elements != NULL) {
		free(Ascii->elements);
		Ascii->elements = NULL;
	}

	if (Ascii->elementsVals != NULL) {
		free(Ascii->elementsVals);
		Ascii->elementsVals = NULL;
	}

	if (Ascii->rowcolIdx != NULL) {
		free(Ascii->rowcolIdx);
		Ascii->rowcolIdx = NULL;
	}
}

void FreeDeviceAsciiRaster(AsciiRaster* Ascii){
	cudaError_t error;
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);

		if (Ascii[i].compute_serialized) {
			error = cudaFree(Ascii[i].elementsVals);
			if (error != cudaSuccess)
			{
				printf("***ERROR in FreeDeviceAsciiRaster->elementsVals: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			Ascii[i].elementsVals = NULL;

			error = cudaFree(Ascii[i].rowcolIdx);
			if (error != cudaSuccess)
			{
				printf("***ERROR in FreeDeviceAsciiRaster->rowcolIdx: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			Ascii[i].rowcolIdx = NULL;
		}
		else {
			error = cudaFree(Ascii[i].elements);
			if (error != cudaSuccess)
			{
				printf("***ERROR in FreeDeviceAsciiRaster: %s\n", cudaGetErrorString(error));
				exit(EXIT_FAILURE);
			}
			Ascii[i].elements = NULL;
		}
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// edge effects correction weights at each point, weights is allocated somewhere else
void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights){
	
	auto t1 = high_resolution_clock::now();
	
	double h2 = h * h;
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew;	

	#pragma omp parallel for private(p_x, p_y, cell_x, cell_y, ew) firstprivate(h2)
	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);
		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
			continue;
		}

		p_x = Points.xCoordinates[p];
		p_y = Points.yCoordinates[p];
		ew = 0.0f;

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
					double d2 = Distance2(p_x, p_y, cell_x, cell_y);
					ew += GaussianKernel(h2, d2) * cellArea;
				}
			}
		}
		weights[p] = 1.0 / ew;
	}
	#pragma omp barrier

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...EdgeCorrectionWeightsExact took %.0f ms\n", ms_float.count());
}

void EdgeCorrectionWeightsExact(SamplePoints Points, float* hs, AsciiRaster Ascii, float *weights){
	
	auto t1 = high_resolution_clock::now();
	
	//float h2 = BandWidth2(Points);
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew;
	double h2;

	#pragma omp parallel for private(p_x, p_y, cell_x, cell_y, ew, h2)
	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);
		p_x = Points.xCoordinates[p];
		p_y = Points.yCoordinates[p];
		ew = 0.0f;
		h2 = hs[p] * hs[p];

		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
			continue;
		}

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
					double d2 = Distance2(p_x, p_y, cell_x, cell_y);
					ew += GaussianKernel(h2, d2) * cellArea;
				}
			}
		}
		weights[p] = 1.0 / ew;
	}
	#pragma omp barrier

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...EdgeCorrectionWeightsExact took %.0f ms\n", ms_float.count());
}

float* AllocateEdgeCorrectionWeights(SamplePoints Points){
	
	float* ecweights = (float*)malloc(Points.numberOfPoints * sizeof(float));
	for (int i = 0; i < Points.numberOfPoints; i++) {
		ecweights[i] = 1.0f;
	}
	return ecweights;	

}

void AllocateDeviceEdgeCorrectionWeights(float** dWeights, SamplePoints Points){
	cudaError_t error;
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dWeights[i], Points.numberOfPoints * sizeof(float));
		if (error != cudaSuccess)
		{
			printf("***ERROR in AllocateDeviceEdgeCorrectionWeights: %s\n", cudaGetErrorString(error));
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
			printf("***ERROR in CopyToDeviceWeights: %s\n", cudaGetErrorString(error));
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
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(weights[i]);
		if (error != cudaSuccess)
		{
			printf("***ERROR in FreeDeviceEdgeCorrectionWeights: %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		weights[i] = NULL;
	}
	cudaSetDevice(GPU_START); //Reset device to first GPU
}

// the array holding bandwidth at each point
float* AllocateBandwidths(int n){ // n is number of points
	return (float*)malloc(n*sizeof(float));
}

// the array holding bandwidth at each point
double* AllocateDistances(int n) { // n is number of points
	return (double*)malloc(n * sizeof(double));
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
			printf("***ERROR in AllocateDeviceBandwidths: %s\n", cudaGetErrorString(error));
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
			printf("***ERROR in CopyToDeviceBandwidths: %s\n", cudaGetErrorString(error));
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
        printf("***ERROR in CopyFromDeviceBandwidths: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToHost);
        exit(EXIT_FAILURE);
    }
}

void FreeDeviceBandwidths(float** bandwidths){
	cudaError_t error;
	//Free Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaFree(bandwidths[i]);
		if (error != cudaSuccess)
		{
			printf("***ERROR in FreeDeviceBandwidths: %s\n", cudaGetErrorString(error));
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

void FreeDistances(double* distances) {
	free(distances);
	distances = NULL;
}

// the array holding inclusive density at each point
float* AllocateDen(int n){ // n is number of points
	return (float*)malloc(n*sizeof(float));
}

void AllocateDeviceDen(float** dDen, int n){ // n is number of points
	cudaError_t error;
	//Allocate Memory Across All Devices
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		error = cudaMalloc((void**)&dDen[i], n * sizeof(float));
		if (error != cudaSuccess)
		{
			printf("***ERROR in AllocateDeviceDen: %s\n", cudaGetErrorString(error));
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
        printf("***ERROR in CopyToDeviceDen: %s\n", cudaGetErrorString(error));
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
        printf("***ERROR in CopyFromDeviceDen: %s\n", cudaGetErrorString(error));
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
        printf("***ERROR in CopyDeviceDen: %s\n", cudaGetErrorString(error));
		printf("size=%d mode=%d\n", size, cudaMemcpyDeviceToDevice);
        exit(EXIT_FAILURE);
    }
}

void FreeDeviceDen(float** den) {
	cudaError_t error;
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);
		error = cudaFree(den[i]);
		if (error != cudaSuccess)
		{
			printf("***ERROR in FreeDeviceDen(Elements): %s\n", cudaGetErrorString(error));
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
float MLE_FixedBandWidth(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0, float* den1, bool useGPU, float** dDen0, float** dDen1){
	
	// hA, hD, and epsilon may be changed based on characteristics of the dataset
	float hA = h / 20;
	float hD = 2*h;

	float width = hD - hA;
	float epsilon = width / 100;

	float factor = 1 + sqrtf(5.0f);
	int iteration = 0;

	printf("+++hA: %f hD: %f width: %f, epsilon: %f\n", hA, hD, width, epsilon); //DEBUG
	while(width > epsilon && iteration < MAX_NUM_ITERATIONS){

		float hD0 = hD;
		float hA0 = hA;

		float hB = hA + width / factor;
		float hC = hD - width / factor;

		float LoghB = LogLikelihood(Ascii, Points, edgeWeights, hB, den0, den1, useGPU, dDen0, dDen1);
		float LoghC = LogLikelihood(Ascii, Points, edgeWeights, hC, den0, den1, useGPU, dDen0, dDen1);

		if(LoghB > LoghC){
			hD = hC;
			if (DEBUG) { 
				printf("++iteration: %d ", iteration);
				printf("hD: %.6f ", hD0);
				printf("hA: %.6f ", hA0);
				printf("LoghB: %.6f \n", LoghB);			
			}
		}
		else{
			hA = hB;
			if (DEBUG) { 
				printf("++iteration: %d ", iteration);
				printf("hD: %.6f ", hD0);
				printf("hA: %.6f ", hA0);
				printf("LoghC: %.6f \n", LoghC); 
			}
		}

		width = hD - hA;

		iteration += 1;
	}

	return (hA + hD) / 2;
}

// computed fixed bandwidth kde
void ComputeFixedDensityAtPoints(AsciiRaster Ascii, SamplePoints Points, float* edgeWeights, float h, float* den0, float* den1, float* dDen0, float* dDen1) {
	
	int numPoints = Points.numberOfPoints;
	// update edge correction weights
	if (UPDATEWEIGHTS) {
		EdgeCorrectionWeightsExact(Points, h, Ascii, edgeWeights);
	}

	#pragma omp parallel for
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

			double d2 = Distance2(pi_x, pi_y, pj_x, pj_y);

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

			CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (h * h, Points[i], Ascii[i], edgeWeights[i]);
		}

		ReformECWeights(edgeWeights, gedgeWeights);

		auto t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as a float. */
		duration<float, std::milli> ms_float = t2 - t1;
		printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
	}
	
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);

		int numPoints = Points[i].numberOfPoints;
		int NBLOCK_W = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

		InitGPUDen <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuDen[i], numPoints);
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

		DensityAtPointsKdtr <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, Points[i], edgeWeights[i], gpuDen[i]);
	}
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(i + GPU_START);
		cudaStreamSynchronize(streams[i]);
		// have to do this as a separate kernel call due to the need of block synchronization !!!
		// this took me hours to debug!
	}

	ReformGPUDensities(gpuDen, den0);

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i + GPU_START);

		int pNum = Points[i].end - Points[i].start;
		int NBLOCK_W = (pNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
		dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

		dCopyDensityValues <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (Points[i], edgeWeights[i], h * h, gpuDen[i], dDen0[i], NULL);
	}

	ReformDensities(dDen0, den0);
}

// the log likelihood given single bandwidth h
float LogLikelihood(AsciiRaster* Ascii, SamplePoints* Points, float **edgeWeights, float h, float* den0, float* den1, bool useGPU, float** dDen0, float** dDen1){
	float logL = 0.0f; // log likelihood
	cudaError error = cudaSuccess;

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
				CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (h * h, Points[i], Ascii[i], edgeWeights[i]);			
				//cudaStreamSynchronize(streams[i]);
			}
			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as a float. */
			duration<float, std::milli> ms_float = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
		}
		
		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);

			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuDen[i], Points[i].numberOfPoints);
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

			DensityAtPointsKdtr <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, Points[i], edgeWeights[i], gpuDen[i]);
			// have to do this as a separate kernel call due to the need of block synchronization !!!
			// this took me hours to debug!
			

		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			int stack_depth;
			cudaMemcpyFromSymbol(&stack_depth, STACK_DEPTH_MAX, sizeof(int), 0, cudaMemcpyDeviceToHost);
			printf("---STACK_DEPTH_MAX = %d\n", stack_depth);
			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den1);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);	

			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (Points[i], edgeWeights[i], h * h, gpuDen[i], NULL, dDen1[i]);
		}

		ReformDensities(dDen1, den1);

		logL = ReductionSumGPU(dDen1, Points[0].numberOfPoints);

	}
	else{ // do it on CPU
		int numPoints = Points[0].numberOfPoints;
		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], h, Ascii[0], edgeWeights[0]);
		}

		// the kd tree appraoch
		float* tmpden = AllocateDen(numPoints);
		double h2 = h * h;
		double range = CUT_OFF_FACTOR * h2;

		for(int i = 0; i < numPoints; i++){
			tmpden[i] = -1.0 * GaussianKernel(h2, 0.0) *  Points[0].weights[i] * edgeWeights[0][i];
		}

		vector<int> ret_index = vector<int>();
		vector<double> ret_dist = vector<double>(); // squared distance

		#pragma omp parallel for private(ret_index, ret_dist)
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
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
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
			
				CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (h*h, gpuPoints[i], Ascii[i], edgeWeights[i]);
			}
			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			duration<float, std::milli> ms_float = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
		}
		
		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuDen[i], Points[i].numberOfPoints);
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

			DensityAtPointsKdtr <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, h * h, gpuPoints[i], edgeWeights[i], gpuDen[i]);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den1);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuPoints[i], edgeWeights[i], h * h, gpuDen[i], dDen0[i], dDen1[i]);
		}
	
		ReformDensities(dDen0, den0);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			CopyDeviceDen(dDen0cpy[i], dDen0[i], Points[i].numberOfPoints);
		}
		reductionSum = ReductionSumGPU(dDen0cpy, Points[0].numberOfPoints);
		
		//printf("reduction result (geometricmean): %f \n", exp(reductionSum/ Points[0].numberOfPoints));		

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			// update bandwidth on GPU
			CalcVaryingBandwidths <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuPoints[i], dDen0[i], h, alpha, dHs[i], reductionSum);
		}

		ReformBandwidths(dHs, hs);

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

				CalcEdgeCorrectionWeights <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (dHs[i], gpuPoints[i], Ascii[i], edgeWeights[i]);
			}

			ReformECWeights(edgeWeights, gedgeWeights);

			auto t2 = high_resolution_clock::now();
			duration<float, std::milli> ms_float = t2 - t1;
			printf("...CalcEdgeCorrectionWeights took %.0f ms\n", ms_float.count());
		}

		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(i + GPU_START);
			
			// execution config.
			int NBLOCK_W = (Points[i].numberOfPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			InitGPUDen <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuDen[i], Points[i].numberOfPoints);
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

			DensityAtPointsKdtr <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (GPU_tree[i].m_gpu_nodes, GPU_tree[i].m_gpu_indexes, GPU_tree[i].m_gpu_points, dHs[i], gpuPoints[i], edgeWeights[i], gpuDen[i]);
		}
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);
			cudaStreamSynchronize(streams[i]);
			// have to do this as a separate kernel call due to the need of block synchronization !!!
			// this took me hours to debug!			cudaStreamSynchronize(streams[i]);
		}

		ReformGPUDensities(gpuDen, den0);

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// execution config.
			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCopyDensityValues <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (gpuPoints[i], edgeWeights[i], dHs[i], gpuDen[i], dDen0[i], dDen1[i]);
		}

		ReformDensities(dDen1, den1);

		logL = ReductionSumGPU(dDen1, Points[0].numberOfPoints);
	}
	else{ // do it on CPU

		int numPoints = Points[0].numberOfPoints;

		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], h, Ascii[0], edgeWeights[0]);
		}

		// kdtree approach
		double h2 = h * h;
		double range = CUT_OFF_FACTOR * h2;
		float* denTmp = AllocateDen(numPoints);
		for(int i = 0; i < numPoints; i++){
			denTmp[i] = 0.0f;
		}

		vector<int> ret_index = vector<int>();
		vector<double> ret_dist = vector<double>(); // squared distance

		#pragma omp parallel for private(ret_index, ret_dist)
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

		// update bandwidths
		float gml = compGML(denTmp, numPoints);
		
		//printf("reduction result (geometricmean): %f \n", gml);
		//exit(0);

	    for(int i = 0; i < numPoints; i++){
			hs[i] = h * powf((denTmp[i] / gml), alpha);
	    }

		// update edge correction weights
		if(UPDATEWEIGHTS){
			EdgeCorrectionWeightsExact(Points[0], hs, Ascii[0], edgeWeights[0]);
		}

		for(int i = 0; i < numPoints; i++){
			double h2 = hs[i] * hs[i];
			denTmp[i] = -1.0 * GaussianKernel(h2, 0.0) *  Points[0].weights[i] * edgeWeights[0][i];
		}

		#pragma omp parallel for private(ret_index, ret_dist)
		for(int i = 0; i < numPoints; i++){
			float pi_x = Points[0].xCoordinates[i];
			float pi_y = Points[0].yCoordinates[i];
			float pj_w = Points[0].weights[i];
			float pj_ew = edgeWeights[0][i];
			double h2 = hs[i] * hs[i];
			double range = CUT_OFF_FACTOR * h2;

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

		if(den0 != NULL){
			for(int i = 0; i < numPoints; i++){
				double h2 = hs[i] * hs[i];
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
/*
 return 5 elements log likelihood in float* logLs
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
/*
 return 3 optmal parameters in float* optParas (optH, optAlpha, LogLmax)
//Added aditional variable to this, hj_likelihood and LogLikelihood functions to handle array of SamplePoints whenever multiple GPUs are present
**/
void hooke_jeeves(AsciiRaster* Ascii, SamplePoints* Points, SamplePoints* gpuPoints, float **edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, float* hs, float* den0, float* den1, bool useGPU, float** dHs, float** dDen0, float** dDen1, float** dDen0cpy){
	float* Ls = (float*)malloc(5 * sizeof(float)); // remember to free at the end
	hj_likelihood(Ascii, Points, gpuPoints, edgeWeights, h0, alpha0, stepH, stepA, -1, Ls, hs, den0, den1, useGPU, dHs, dDen0, dDen1, dDen0cpy);

	float Lmax = Ls[0];

	float s = stepH / 10;
	float a = stepA / 10;

	int iteration = 0;
    while ((stepH > s || stepA > a) && (h0 - stepH > 0) &&  iteration <= MAX_NUM_ITERATIONS){

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

	auto t1 = high_resolution_clock::now();

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

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...ReductionSum took %.0f ms in total\n", ms_float.count());

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

	printf("dimGrid.x = %d, dimGrid.y = %d, dimGrid.z = %d, BLOCK_SIZE = %d \n", dimGrid.x, dimGrid.y, dimGrid.z, BLOCK_SIZE);

	//printf("iteration %d NUM_ACTIVE_ITEMS %d GRID_SIZE %d x GRID_SIZE %d\n", iteration, NUM_ACTIVE_ITEMS, GRID_SIZE, GRID_SIZE);

	// call the kernel for the first iteration
	ReductionSum_V0 <<<dimGrid, BLOCK_SIZE, 0, streams[0] >>> (dArray, N, iteration, NUM_ACTIVE_ITEMS);

	
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

		printf("dimGrid.x = %d, dimGrid.y = %d, dimGrid.z = %d, BLOCK_SIZE = %d \n", dimGrid.x, dimGrid.y, dimGrid.z, BLOCK_SIZE);

		//printf("iteration %d NUM_ACTIVE_ITEMS %d GRID_SIZE %d x GRID_SIZE %d \n", iteration, NUM_ACTIVE_ITEMS, GRID_SIZE, GRID_SIZE);
		ReductionSum_V0 <<<dimGrid, BLOCK_SIZE, 0, streams[0] >>> (dArray, N, iteration, NUM_ACTIVE_ITEMS);

		
		NUM_ACTIVE_ITEMS = (NUM_ACTIVE_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

		numberOfElements = dimGrid.x * dimGrid.y;

		iteration++;
	}
}

// mark the boundary cells on a raster representing the study area
// the second parameter tempAscii is only needed for gpu computing
void MarkBoundary(AsciiRaster* Ascii, AsciiRaster& tmpAscii, bool useGPU){

	auto t1 = high_resolution_clock::now();
	//printf("...before MarkBoundary nVals = %d \n", Ascii[0].nVals);
	if(useGPU){ // do it on GPU

		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			// invoke kernels to mark the boundary of study area
			// execution config.
			int NBLOCK_W = (Ascii[i].end - Ascii[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);
			dMarkBoundary <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (Ascii[i]);
		}
		//cudaSetDevice(GPU_START);

		ReformAsciiRaster(Ascii, tmpAscii);
		#pragma omp parallel for
		for (int row = 0; row < tmpAscii.nRows; row++) {
			for (int col = 0; col < tmpAscii.nCols; col++) {
				float val = tmpAscii.elements[row * tmpAscii.nCols + col];
				if (val == 0.0f) {
					tmpAscii.elements[row * tmpAscii.nCols + col] = tmpAscii.noDataValue;
					#pragma omp atomic
					tmpAscii.nVals -= 1;
				}
			}
		}
		
		// data have changed, call these two lines to refresh
		AsciiRasterSwitchDataSerialization(&tmpAscii, false); // so stale serialized data can be removed, if any
		AsciiRasterSwitchComputeSerialization(&tmpAscii, true);

	}
	else{ // do it on CPU

		#pragma omp parallel for
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
				#pragma omp atomic
				Ascii[0].nVals -= 1;
			}
		}
		#pragma omp barrier
		// replace 0.0 values (interior) with no-data value

		#pragma omp parallel for
		for (int row = 0; row < Ascii[0].nRows; row++) {
			for (int col = 0; col < Ascii[0].nCols; col++) {
				float val = Ascii[0].elements[row * Ascii[0].nCols + col];
				if (val == 0.0f) {
					Ascii[0].elements[row * Ascii[0].nCols + col] = Ascii[0].noDataValue;
				}
			}
		}
		#pragma omp barrier
		// data have changed, call these two lines to refresh
		AsciiRasterSwitchDataSerialization(&Ascii[0], false); // so stale serialized data can be removed, if any
		AsciiRasterSwitchComputeSerialization(&Ascii[0], true);
	}
	
	auto t2 = high_resolution_clock::now();
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...MarkBoundary took %.0f ms\n", ms_float.count());
}

// compute the closest distances from sample points to study area boundary
void CalcDist2Boundary(SamplePoints* Points, AsciiRaster* Ascii, bool useGPU){

	printf("...starting CalcDist2Boundary\n");

	auto t1 = high_resolution_clock::now();
	if(useGPU){ // do it on GPU
		for (int i = 0; i < GPU_N; i++)
		{
			cudaSetDevice(i + GPU_START);

			int NBLOCK_W = (Points[i].end - Points[i].start + BLOCK_SIZE - 1) / BLOCK_SIZE;
			int GRID_SIZE_W = (int)(sqrtf(NBLOCK_W)) + 1;
			dim3 dimGrid_W(GRID_SIZE_W, GRID_SIZE_W);

			dCalcDist2Boundary <<<dimGrid_W, BLOCK_SIZE, 0, streams[i] >>> (Points[i], Ascii[i]);
		}
		ReformPoints(Points, sPoints);
		cudaSetDevice(GPU_START);
	}
	else{
		float p_x, p_y, cell_x, cell_y;
		#pragma omp parallel for private(p_x, p_y, cell_x, cell_y)
		for (int p = 0; p < Points[0].numberOfPoints; p++){
			double minDist = DBL_MAX;
			p_x = Points[0].xCoordinates[p];
			p_y = Points[0].yCoordinates[p];			

			if (Ascii[0].compute_serialized) {
				for (int idx = 0; idx < Ascii[0].nVals; idx++) {
					if (Ascii[0].elementsVals[idx] == 1.0f) {
						int row = Ascii[0].rowcolIdx[idx] / Ascii[0].nCols;
						int col = Ascii[0].rowcolIdx[idx] % Ascii[0].nCols;

						cell_x = COL_TO_XCOORD(col, Ascii[0].xLLCorner, Ascii[0].cellSize);
						cell_y = ROW_TO_YCOORD(row, Ascii[0].nRows, Ascii[0].yLLCorner, Ascii[0].cellSize);
						double d2 = Distance2(p_x, p_y, cell_x, cell_y);

						if (d2 < minDist) {
							minDist = d2;
						}
					}
				}
			}
			else {
				for (int row = 0; row < Ascii[0].nRows; row++) {
					for (int col = 0; col < Ascii[0].nCols; col++) {
						if (Ascii[0].elements[row * Ascii[0].nCols + col] == 1.0f) { // cells on boundary
							cell_x = COL_TO_XCOORD(col, Ascii[0].xLLCorner, Ascii[0].cellSize);
							cell_y = ROW_TO_YCOORD(row, Ascii[0].nRows, Ascii[0].yLLCorner, Ascii[0].cellSize);
							double d2 = Distance2(p_x, p_y, cell_x, cell_y);

							if (d2 < minDist) {
								minDist = d2;
							}
						}
					}
				}
			}	
			Points[0].distances[p] = minDist;
		}
		#pragma omp barrier
	}
	auto t2 = high_resolution_clock::now();
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...CalcDist2Boundary took %.0f ms\n", ms_float.count());
}

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

	double* distances = (double*)malloc(2*n*sizeof(double));
	for (int i = 0; i < 2*n - 1; i += 2)
	{
		distances[i] = Points.distances[i/2];
		distances[i + 1] = i/2 * 1.0f;
	}
	
	qsort(distances, n, 2*sizeof(distances[0]), compare);

	for (int i = 0; i < 2*n - 1; i += 2)
	{

		int idx = (int)distances[i + 1];
		Points.xCoordinates[i/2] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i/2] = temPoints.yCoordinates[idx];
		Points.weights[i/2] = temPoints.weights[idx];
		Points.distances[i/2] = temPoints.distances[idx];
	}
	FreeSamplePoints(&temPoints);

}

// if bandwidths are provided in the file, need to adjust the order in hs
void SortSamplePoints(SamplePoints Points, float* hs) {
	
	auto t1 = high_resolution_clock::now();
	
	const int n = Points.numberOfPoints;
	SamplePoints temPoints = CopySamplePoints(Points);

	//printf("here %d\n", n);

	float* temhs = (float*)malloc(n * sizeof(float));
	
	//printf("there %d\n", n);
	
	for (int i = 0; i < n; i++) {
		//printf("...hs[%d]\n", i, hs[i]);
		temhs[i] = hs[i];
	}

	double* distances = (double*)malloc(2 * n * sizeof(double));
	for (int i = 0; i < 2 * n - 1; i += 2)
	{
		distances[i] = Points.distances[i / 2];
		distances[i + 1] = i / 2 * 1.0f;
	}

	qsort(distances, n, 2 * sizeof(distances[0]), compare);

	for (int i = 0; i < 2 * n - 1; i += 2)
	{
		int idx = (int)distances[i + 1];
		Points.xCoordinates[i / 2] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i / 2] = temPoints.yCoordinates[idx];
		Points.weights[i / 2] = temPoints.weights[idx];
		Points.distances[i / 2] = temPoints.distances[idx];
		hs[i / 2] = temhs[idx];
		//printf("hs[%d]\n", i/2, hs[i/2]);
	}

	FreeSamplePoints(&temPoints);
	FreeBandwidths(temhs);

	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...SortSamplePoints took %.0f ms\n", ms_float.count());

}

// build a KDtree on sample points
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
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...building KDTree (CPU) took %.0f ms\n", ms_float.count());
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
	/* Getting number of milliseconds as a float. */
	duration<float, std::milli> ms_float = t2 - t1;
	printf("...building KDTree (GPU) took %.0f ms\n", ms_float.count());
}

//Function to check device properties, primarily for troubleshooting purposes
void DevProp()
{
	for (int i = 0; i < GPU_N; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Device Memory (bytes): %llu - (GB): %.2f\n", prop.totalGlobalMem, prop.totalGlobalMem/1024/1024/1024.0);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Clock Rate (KHz): %d\n", prop.clockRate);
		printf("  multiProcessorCount: %d\n\n", prop.multiProcessorCount);
	}
}

////Function which copies each group of points into a temorary place on the host, before copying their values to
////hPoints in order to reform the original group
void ReformPoints(SamplePoints* dPoints, const SamplePoints hPoints)
{
	int n = hPoints.numberOfPoints; //Number of TOTAL points
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

		error = cudaMemcpy(tempPoints.distances, dPoints[device].distances, n * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("ERROR 4 in ReformPoints (FROM device): %s\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

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

		error = cudaMemcpy(dPoints[i].distances, hPoints.distances, n * sizeof(double), cudaMemcpyHostToDevice);
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

//Reform gpuDen on host and copy back accross devices
void ReformGPUDensities(float** dDen, float* hDen)
{
	int n = sPoints.numberOfPoints; //Number of TOTAL points	
	cudaError_t error = cudaSuccess;
	float* tempDen = (float*)malloc(n * sizeof(float));
	size_t size = n * sizeof(float);

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
}

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
}

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
}

void OMP_TEST() {
	float mean = 0.0f;
	float minv = FLT_MAX;
	float maxv = FLT_MIN;

	int i;
	
	omp_set_dynamic(0);
	printf("K -> omp_get_max_threads()= %d\n", omp_get_max_threads());
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
	printf("mean = %.2f min = %.2f max = %.2f\n", mean, minv, maxv);
}

SamplePoints ExtractSamplePointsCurTile(AsciiRaster ascTile, float* hs, vector<int> &index) {

	float xll = ascTile.xLLCorner;
	float yll = ascTile.yLLCorner;
	float xur = xll + ascTile.cellSize * ascTile.nCols;
	float yur = yll + ascTile.cellSize * ascTile.nRows;

	#pragma omp parallel for
	for (int i = 0; i < sPoints.numberOfPoints; i++) {
		float p_x = sPoints.xCoordinates[i];
		float p_y = sPoints.yCoordinates[i];
		double p_h2 = (double)hs[i] * (double)hs[i];
		double dist2bbox2 = 0;
		if (xll <= p_x && p_x <= xur && yll <= p_y && p_y <= yur) {
			dist2bbox2 = 0;
		}
		else if (p_x < xll && p_y > yur) {
			dist2bbox2 = Distance2(p_x, p_y, xll, yur);
		}		
		else if (xll < p_x && p_x < xur && p_y > yur) {
			double diff = (double)(p_y - yur);
			dist2bbox2 = diff * diff;
		}
		else if (p_x > xur && p_y > yur) {
			dist2bbox2 = Distance2(p_x, p_y, xur, yur);
		}
		else if (yll < p_y && p_y < yur && p_x > xur) {
			double diff = (double)(p_x - xur);
			dist2bbox2 = diff * diff;
		}
		else if (xur < p_x && p_y < yll) {
			dist2bbox2 = Distance2(p_x, p_y, xur, yll);
		}
		else if (xll < p_x && p_x < xur && p_y < yll) {
			double diff = (double)(p_y - yll);
			dist2bbox2 = diff * diff;
		}
		else if (p_x < xll && p_y < yll) {
			dist2bbox2 = Distance2(p_x, p_y, xll, yll);
		}
		else { // (yll < p_y && p_y < yur && p_x < xll) 
			double diff = (double)(p_x - xll);
			dist2bbox2 = diff * diff;
		}

		if (dist2bbox2 < CUT_OFF_FACTOR * p_h2) {
			#pragma omp critical	
			index.push_back(i);
		}
		
	}
	#pragma omp barrier
	int N = index.size();
	printf("---%d points in this tile\n", N);

	SamplePoints pointsTile;
	pointsTile.numberOfPoints = N;
	pointsTile.start = 0;
	pointsTile.end = N;
	pointsTile.xCoordinates = (float*)malloc(N * sizeof(float));
	pointsTile.yCoordinates = (float*)malloc(N * sizeof(float));
	pointsTile.weights = (float*)malloc(N * sizeof(float));
	pointsTile.distances = (double*)malloc(N * sizeof(double));	
	
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		int idx = index[i];

		pointsTile.xCoordinates[i] = sPoints.xCoordinates[idx];
		pointsTile.yCoordinates[i] = sPoints.yCoordinates[idx];
		pointsTile.weights[i] = sPoints.weights[idx];
		pointsTile.distances[i] = sPoints.distances[idx];
	}

	return pointsTile;
}
