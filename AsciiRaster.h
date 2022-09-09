// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license

#ifndef _ASCIIRASTER_H_
#define _ASCIIRASTER_H_


// AsciiRaster Structure declaration
typedef struct {
	size_t nCols;
	size_t nRows;

	// these additional data structures are for saving memory space and for speeding up computation by avoiding nodata celss (e.g, calculating distance to boundary) 
	bool data_serialized = false; // a flag indicating if cell values are serialized
	size_t nVals; // Number of non-nodata values
	int* rowcolIdx = NULL;  // Array to hold the seqential index of non-nodata values, row * nCol + col
	float* elementsVals = NULL; // Array to hold the serialized non-nodata values

	float xLLCorner;
	float yLLCorner;
	float cellSize;
	float noDataValue;
	float* elements;
	
	bool compute_serialized = false; // a flag indicating if compute is serialized
									 // if true, parallel computing is on serialized data values (data_serialized has to be true)							
	size_t start; //Used to identify starting index (in elements) when cells are being worked with accross multiple GPUs, inclusive
	size_t end; //Ending index, exclusive

	double* geotransform = NULL;
	const char* projection = NULL;

} AsciiRaster;


#endif // _ASCIIRASTER_H_