#ifndef Hit_h
#define Hit_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <cmath>
#include <vector>

#include "MathUtil.cuh"
#include "PrintUtil.h"
#include "Module.cuh"
#include "GeometryUtil.cuh"

struct hits
{
    unsigned int *nHits; //single number
    unsigned int *n2SHits;
    float *xs;
    float *ys;
    float *zs;

    unsigned int* moduleIndices;
    
    float *rts;
    float* phis;

    int *edge2SMap;
    float *highEdgeXs;
    float *highEdgeYs;
    float *lowEdgeXs;
    float *lowEdgeYs;

};

void createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int maxHits, unsigned int max2SHits);
void addHitToMemory(struct hits& hitsInGPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId);
inline float phi(float x, float y, float z);
inline float Atan2(float y, float x);
inline float phi_mpi_pi(float phi);
float deltaPhi(float x1, float y1, float z1, float x2, float y2, float z2);
float deltaPhiChange(float x1, float y1, float z1, float x2, float y2, float z2);
void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);
#endif