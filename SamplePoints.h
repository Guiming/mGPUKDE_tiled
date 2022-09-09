// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

#ifndef _SAMPLEPOINTS_H_
#define _SAMPLEPOINTS_H_


// SamplePoints Structure declaration
typedef struct {
	unsigned int numberOfPoints; //Total number of points
	unsigned int start; //Used to identify starting index when points are being worked with accross multiple GPUs, inclusive
	unsigned int end; //Ending index, exclusive
	float* xCoordinates;
	float* yCoordinates;
	float* weights;
	double* distances; // closest distances (squared) to study area boundary (by Guiming @ 2016-09-02)
} SamplePoints;


#endif // _SAMPLEPOINTS_H_
