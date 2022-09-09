// Copyright 2022 Guiming Zhang (guiming.zhang@du.edu)
// Distributed under GNU General Public License (GPL) license
// 
// Revised based on https://gerasimosmichalitsianos.wordpress.com/2018/11/30/431/

#include <iostream>
#include <string>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <gdalwarper.h>
#include <stdlib.h>
#include <chrono>
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;
//typedef std::string String;

class GeoTIFFReader {

private: // NOTE: "private" keyword is redundant here.  
         // we place it here for emphasis. Because these
         // variables are declared outside of "public", 
         // they are private. 

    const char* filename;        // name of Geotiff
    GDALDataset* geotiffDataset; // Geotiff GDAL datset object. 
    double geotransform[6];      // 6-element geotranform array.
    int dimensions[3];           // X,Y, and Z dimensions. 
    int NROWS, NCOLS, NLEVELS;     // dimensions of data in Geotiff. 

    //float NODATAVALUE;
    //const char* projection;

    int NTILES;
    int NTILES_TOTAL;
    int XOFF;
    int YOFF;
    int XSIZE;
    int YSIZE;

    double geotransform_tile[6];
    
    int CUR_TILE_PARAM[4];
    int TILE_PARAM[4];

    int BLOCK_XSIZE;
    int BLOCK_YSIZE;

public:

    // define constructor function to instantiate object
    // of this Geotiff class. 
    GeoTIFFReader(const char* tiffname) {
        filename = tiffname;
        GDALAllRegister();

        // set pointer to Geotiff dataset as class member.  
        geotiffDataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);

        // set the dimensions of the Geotiff 
        NROWS = GDALGetRasterYSize(geotiffDataset);
        NCOLS = GDALGetRasterXSize(geotiffDataset);
        NLEVELS = GDALGetRasterCount(geotiffDataset);
        printf("...GeoTIFF: NCOLS=%d, NROWS=%d\n", NCOLS, NROWS);
        NTILES = 0;
        XOFF = 0;
        YOFF = 0;
        XSIZE = NCOLS;
        YSIZE = NROWS;

        NTILES_TOTAL = 1;
        
        geotiffDataset->GetRasterBand(1)->GetBlockSize(&BLOCK_XSIZE, &BLOCK_YSIZE);
        printf("...GeoTIFF: BLOCK_XSIZE=%d, BLOCK_YSIZE=%d, NTILES_TOTAL=%d\n", BLOCK_XSIZE, BLOCK_YSIZE, NTILES_TOTAL);

    }

    GeoTIFFReader(const char* tiffname, int tile_xsize, int tile_ysize) {
        filename = tiffname;
        GDALAllRegister();

        // set pointer to Geotiff dataset as class member.  
        geotiffDataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);

        // set the dimensions of the Geotiff 
        NROWS = GDALGetRasterYSize(geotiffDataset);
        NCOLS = GDALGetRasterXSize(geotiffDataset);
        NLEVELS = GDALGetRasterCount(geotiffDataset);

        printf("...GeoTIFF: NCOLS=%d, NROWS=%d\n", NCOLS, NROWS);

        NTILES = 0;
        XOFF = 0;
        YOFF = 0;
        XSIZE = min(tile_xsize, NCOLS);
        YSIZE = min(tile_ysize, NROWS);

        NTILES_TOTAL = (int)ceil(NCOLS * 1.0 / XSIZE) * (int)ceil(NROWS * 1.0 / YSIZE);


        printf("...TILE: TILE_XSIZE=%d, TILE_YSIZE=%d, NTILES_TOTAL=%d\n", XSIZE, YSIZE, NTILES_TOTAL);

        geotiffDataset->GetRasterBand(1)->GetBlockSize(&BLOCK_XSIZE, &BLOCK_YSIZE);
        printf("...GeoTIFF: BLOCK_XSIZE=%d, BLOCK_YSIZE=%d\n", BLOCK_XSIZE, BLOCK_YSIZE);
    }

    // define destructor function to close dataset, 
    // for when object goes out of scope or is removed
    // from memory. 
    ~GeoTIFFReader() {

        // close the Geotiff dataset, free memory for array.  
        //GDALClose(geotiffDataset); // this causes problems

        //printf("... starting ~GeoTIFFReader()\n");

        GDALDestroyDriverManager();
        //printf("...~GeoTIFFReader()\n");
    }

    void GetBlockXYSize(int& blockxsize, int& blockysize) {
        blockxsize = BLOCK_XSIZE;
        blockysize = BLOCK_YSIZE;
    }

    int GetNTiles() {
        return NTILES;
    }

    int GetNTilesTotal() {
        return NTILES_TOTAL;
    }

    int* GetTileParam() {
        TILE_PARAM[0] = XOFF;
        TILE_PARAM[1] = YOFF;
        TILE_PARAM[2] = XSIZE;
        TILE_PARAM[3] = YSIZE;
        return TILE_PARAM;
    }

    void SetCurTileParam(int xoff, int yoff, int xsize, int ysize) {
        CUR_TILE_PARAM[0] = xoff;
        CUR_TILE_PARAM[1] = yoff;
        CUR_TILE_PARAM[2] = xsize;
        CUR_TILE_PARAM[3] = ysize;        
    }

    int* GetCurTileParam() {
        return CUR_TILE_PARAM;
    }



    const char* GetFileName() {
        /*
         * function GetFileName()
         * This function returns the filename of the Geotiff.
         */
        return filename;
    }

    const char* GetProjection() {
        /* function const char* GetProjection():
         *  This function returns a character array (string)
         *  for the projection of the geotiff file. Note that
         *  the "->" notation is used. This is because the
         *  "geotiffDataset" class variable is a pointer
         *  to an object or structure, and not the object
         *  itself, so the "." dot notation is not used.
         */
        return geotiffDataset->GetProjectionRef();
    }

    double* GetGeoTransform() {
        /*
         * function float *GetGeoTransform()
         *  This function returns a pointer to a float that
         *  is the first element of a 6 element array that holds
         *  the geotransform of the geotiff.
         */
        geotiffDataset->GetGeoTransform(geotransform);
        return geotransform;
    }

    void SetGeoTransformTile(int xoff, int yoff) {
        GetGeoTransform();
        geotransform_tile[0] = geotransform[0] + geotransform[1] * xoff;
        geotransform_tile[1] = geotransform[1];
        geotransform_tile[2] = geotransform[2];
        geotransform_tile[3] = geotransform[3] + geotransform[5] * yoff;;
        geotransform_tile[4] = geotransform[4];
        geotransform_tile[5] = geotransform[5];
    }

    double* GetGeoTransformTile() {
        return geotransform_tile;
    }

    float GetNoDataValue() {
        /*
         * function GetNoDataValue():
         *  This function returns the NoDataValue for the Geotiff dataset.
         *  Returns the NoData as a float.
         */
        return (float)geotiffDataset->GetRasterBand(1)->GetNoDataValue();
    }

    int* GetDimensions() {
        /*
         * int *GetDimensions():
         *
         *  This function returns a pointer to an array of 3 integers
         *  holding the dimensions of the Geotiff. The array holds the
         *  dimensions in the following order:
         *   (1) number of columns (x size)
         *   (2) number of rows (y size)
         *   (3) number of bands (number of bands, z dimension)
         */
        dimensions[0] = NCOLS;
        dimensions[1] = NROWS;
        dimensions[2] = NLEVELS;
        return dimensions;
    }

    float** GetRasterBand(int z) {

        /*
         * function float** GetRasterBand(int z):
         * This function reads a band from a geotiff at a
         * specified vertical level (z value, 1 ...
         * n bands). To this end, the Geotiff's GDAL
         * data type is passed to a switch statement,
         * and the template function GetArray2D (see below)
         * is called with the appropriate C++ data type.
         * The GetArray2D function uses the passed-in C++
         * data type to properly read the band data from
         * the Geotiff, cast the data to float**, and return
         * it to this function. This function returns that
         * float** pointer.
         */

        float** bandLayer = new float* [NROWS];
        switch (GDALGetRasterDataType(geotiffDataset->GetRasterBand(z))) {
        case 0:
            return NULL; // GDT_Unknown, or unknown data type.
        case 1:
            // GDAL GDT_Byte (-128 to 127) - unsigned  char
            return GetArray2D<unsigned char>(z, bandLayer);
        case 2:
            // GDAL GDT_UInt16 - short
            return GetArray2D<unsigned short>(z, bandLayer);
        case 3:
            // GDT_Int16
            return GetArray2D<short>(z, bandLayer);
        case 4:
            // GDT_UInt32
            return GetArray2D<unsigned int>(z, bandLayer);
        case 5:
            // GDT_Int32
            return GetArray2D<int>(z, bandLayer);
        case 6:
            // GDT_Float32
            return GetArray2D<float>(z, bandLayer);
        case 7:
            // GDT_Float64
            return GetArray2D<float>(z, bandLayer);
        default:
            break;
        }

        return NULL;
    }

    template<typename T>
    float** GetArray2D(int layerIndex, float** bandLayer) {

        /*
         * function float** GetArray2D(int layerIndex):
         * This function returns a pointer (to a pointer)
         * for a float array that holds the band (array)
         * data from the geotiff, for a specified layer
         * index layerIndex (1,2,3... for GDAL, for Geotiffs
         * with more than one band or data layer, 3D that is).
         *
         * Note this is a template function that is meant
         * to take in a valid C++ data type (i.e. char,
         * short, int, float), for the Geotiff in question
         * such that the Geotiff band data may be properly
         * read-in as numbers. Then, this function casts
         * the data to a float data type automatically.
         */
        auto t1 = high_resolution_clock::now();

         // get the raster data type (ENUM integer 1-12, 
         // see GDAL C/C++ documentation for more details)        
        GDALDataType bandType = GDALGetRasterDataType(geotiffDataset->GetRasterBand(layerIndex));

        // get number of bytes per pixel in Geotiff
        int nbytes = GDALGetDataTypeSizeBytes(bandType);

        // allocate pointer to memory block for one row (scanline) 
        // in 2D Geotiff array.  
        T* rowBuff = (T*)CPLMalloc(nbytes * NCOLS);

        for (int row = 0; row < NROWS; row++) {     // iterate through rows

          // read the scanline into the dynamically allocated row-buffer       
            CPLErr e = geotiffDataset->GetRasterBand(layerIndex)->RasterIO(GF_Read, 0, row, NCOLS, 1, rowBuff, NCOLS, 1, bandType, 0, 0);
            if (!(e == 0)) {
                cout << "Warning: Unable to read scanline in Geotiff!" << endl;
                exit(1);
            }

            bandLayer[row] = new float[NCOLS];
            for (int col = 0; col < NCOLS; col++) { // iterate through columns
                bandLayer[row][col] = (float)rowBuff[col];
            }
        }
        CPLFree(rowBuff);

        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a float. */
        duration<float, std::milli> ms_float = t2 - t1;
        printf("...reading GeoTiff took %f ms\n\n", ms_float.count());

        return bandLayer;
    }


    float** GetRasterBand_NextTile(int z) {

        /*
         * function float** GetRasterBand(int z):
         * This function reads a band from a geotiff at a
         * specified vertical level (z value, 1 ...
         * n bands). To this end, the Geotiff's GDAL
         * data type is passed to a switch statement,
         * and the template function GetArray2D (see below)
         * is called with the appropriate C++ data type.
         * The GetArray2D function uses the passed-in C++
         * data type to properly read the band data from
         * the Geotiff, cast the data to float**, and return
         * it to this function. This function returns that
         * float** pointer.
         */

        if (XOFF == NCOLS && YOFF == NROWS) {
            NTILES = -1;
            printf("***no more tiles to read.\n");
            return NULL;
        }
        
        int _YSIZE = YSIZE;
        int _XSIZE = XSIZE;
        if (YSIZE > NROWS - YOFF) {
            _YSIZE = NROWS - YOFF;
        } 

        if (XSIZE > NCOLS - XOFF) {
            _XSIZE = NCOLS - XOFF;
        }

        SetCurTileParam(XOFF, YOFF, _XSIZE, _YSIZE);
        SetGeoTransformTile(XOFF, YOFF);

        float** bandLayer = new float* [_YSIZE];
        switch (GDALGetRasterDataType(geotiffDataset->GetRasterBand(z))) {
        case 0:
            return NULL; // GDT_Unknown, or unknown data type.
        case 1:
            // GDAL GDT_Byte (-128 to 127) - unsigned  char
            return GetArray2D_NextTile<unsigned char>(z, bandLayer);
        case 2:
            // GDAL GDT_UInt16 - short
            return GetArray2D_NextTile<unsigned short>(z, bandLayer);
        case 3:
            // GDT_Int16
            return GetArray2D_NextTile<short>(z, bandLayer);
        case 4:
            // GDT_UInt32
            return GetArray2D_NextTile<unsigned int>(z, bandLayer);
        case 5:
            // GDT_Int32
            return GetArray2D_NextTile<int>(z, bandLayer);
        case 6:
            // GDT_Float32
            return GetArray2D_NextTile<float>(z, bandLayer);
        case 7:
            // GDT_Float64
            return GetArray2D_NextTile<float>(z, bandLayer);
        default:
            break;
        }

        return NULL;
    }

    template<typename T>
    float** GetArray2D_NextTile(int layerIndex, float** bandLayer) {

        /*
         * function float** GetArray2D(int layerIndex):
         * This function returns a pointer (to a pointer)
         * for a float array that holds the band (array)
         * data from the geotiff, for a specified layer
         * index layerIndex (1,2,3... for GDAL, for Geotiffs
         * with more than one band or data layer, 3D that is).
         *
         * Note this is a template function that is meant
         * to take in a valid C++ data type (i.e. char,
         * short, int, float), for the Geotiff in question
         * such that the Geotiff band data may be properly
         * read-in as numbers. Then, this function casts
         * the data to a float data type automatically.
         */

        auto t1 = high_resolution_clock::now();

         // get the raster data type (ENUM integer 1-12, 
         // see GDAL C/C++ documentation for more details)        
        GDALDataType bandType = GDALGetRasterDataType(geotiffDataset->GetRasterBand(layerIndex));

        // get number of bytes per pixel in Geotiff
        int nbytes = GDALGetDataTypeSizeBytes(bandType);

        // allocate pointer to memory block for one row (scanline) 
        // in 2D Geotiff array.  
        //T* rowBuff = (T*)CPLMalloc(nbytes * NCOLS);

        int* pos = GetCurTileParam();
        int _XSIZE = pos[2];
        int _YSIZE = pos[3];       
        //printf("...%d, %d, %d, %ull\n", nbytes, _XSIZE, _YSIZE, (size_t)nbytes * (size_t)_YSIZE * (size_t)_XSIZE);
        T* tileBuff = (T*)CPLMalloc((size_t)((size_t)nbytes * (size_t)_YSIZE * (size_t)_XSIZE));
        
        // read the tile into the dynamically allocated tile-buffer       
        CPLErr e = geotiffDataset->GetRasterBand(layerIndex)->RasterIO(GF_Read, XOFF, YOFF, _XSIZE, _YSIZE, tileBuff, _XSIZE, _YSIZE, bandType, 0, 0);
        if (!(e == 0)) {
            cout << "Warning: Unable to read tile in Geotiff!" << endl;
            exit(1);
        }

        for (int row = 0; row < _YSIZE; row++) {     // iterate through rows
            bandLayer[row] = new float[_XSIZE];
            for (int col = 0; col < _XSIZE; col++) { // iterate through columns
                bandLayer[row][col] = (float)tileBuff[row * _XSIZE + col];
            }
        }
        CPLFree(tileBuff);
        
        if (XOFF + _XSIZE < NCOLS) {
            XOFF += _XSIZE;
        } 
        else{
            if (YOFF + _YSIZE < NROWS) {
                XOFF = 0;
                YOFF += _YSIZE;
            }
            else {
                XOFF = NCOLS;
                YOFF = NROWS;
            }
        }     

        NTILES += 1;

        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a float. */
        duration<float, std::milli> ms_float = t2 - t1;
        printf("...reading GeoTiff tile took %f ms\n\n", ms_float.count());

        return bandLayer;
    }
};

class GeoTIFFWriter
{
private:
    GDALDriver* driverGeotiff;
    GDALDataset* geotiffDataset;

public:
    
    GeoTIFFWriter() {
        GDALAllRegister();
        driverGeotiff = GetGDALDriverManager()->GetDriverByName("GTiff");
        geotiffDataset = NULL;
        //printf("...GeoTIFFWriter()\n");
    }
    
    GeoTIFFWriter(const char* tiffname, double* geotransform, const char* projection, int nrows, int ncols, float nodata) {
        GDALAllRegister();
        driverGeotiff = GetGDALDriverManager()->GetDriverByName("GTiff");
        char** papszOptions = NULL;
        papszOptions = CSLSetNameValue(papszOptions, "BIGTIFF", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
        geotiffDataset = driverGeotiff->Create(tiffname, ncols, nrows, 1, GDT_Float32, papszOptions);
        if (geotransform != NULL) geotiffDataset->SetGeoTransform(geotransform);
        if (projection != NULL) geotiffDataset->SetProjection(projection);
        geotiffDataset->GetRasterBand(1)->SetNoDataValue(nodata);

        //printf("...GeoTIFFWriter()\n");
    }

    ~GeoTIFFWriter() {
        
        //compute and set statistics
        double bMin, bMax, bMean, bStd;
        geotiffDataset->GetRasterBand(1)->ComputeStatistics(true, &bMin, &bMax, &bMean, &bStd, NULL, NULL);
        geotiffDataset->GetRasterBand(1)->SetStatistics(bMin, bMax, bMean, bStd);
        
        // close the Geotiff dataset, free memory for array.  
        GDALClose(geotiffDataset);

        GDALDestroyDriverManager();

        //printf("...~GeoTIFFWriter()\n");
    }
    
    void WriteGeoTIFF(const char* tiffname, double* geotransform, const char* projection, int nrows, int ncols, float nodata, float* banddata) {
        
        auto t1 = high_resolution_clock::now();

        //GDALDataset* geotiffDataset;

        char** papszOptions = NULL;
        papszOptions = CSLSetNameValue(papszOptions, "BIGTIFF", "YES");
        papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");

        geotiffDataset = driverGeotiff->Create(tiffname, ncols, nrows, 1, GDT_Float32, papszOptions);

        if (geotransform != NULL) geotiffDataset->SetGeoTransform(geotransform);
        if (projection != NULL) geotiffDataset->SetProjection(projection);
        geotiffDataset->GetRasterBand(1)->SetNoDataValue(nodata);

        CPLErr e = geotiffDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, banddata, ncols, nrows, GDT_Float32, 0, 0);
        if (!(e == 0)) {
            cout << "Warning: Unable to write tile in Geotiff!" << endl;
            exit(1);
        }
        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a float. */
        duration<float, std::milli> ms_float = t2 - t1;
        printf("...writing GeoTiff took %f ms\n\n", ms_float.count());

    }


    void WriteGeoTIFF_NextTile(int xoff, int yoff, int ncols, int nrows, float* banddata) {

        auto t1 = high_resolution_clock::now();
       
        //printf("banddata[-1]=%f\n", banddata[ncols * nrows - 1]);
        //printf("xoff=%d\n", xoff);
        //printf("yoff=%d\n", yoff);
        //printf("ncols=%d\n", ncols);
        //printf("nrows=%d\n", nrows);
        CPLErr e = geotiffDataset->GetRasterBand(1)->RasterIO(GF_Write, xoff, yoff, ncols, nrows, banddata, ncols, nrows, GDT_Float32, 0, 0);
        //printf("%d %s\n", e, e);
        if (!(e == 0)) {
            
            cout << "Warning: Unable to write tile in Geotiff!" << endl;
            exit(1);
        }
        //printf("got here\n");
        geotiffDataset->FlushCache();
        auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a float. */
        duration<float, std::milli> ms_float = t2 - t1;
        printf("...writing GeoTiff tile took %f ms\n\n", ms_float.count());

    }

};