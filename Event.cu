# include "Event.cuh"


const unsigned int N_MAX_HITS_PER_MODULE = 100;
const unsigned int N_MAX_MD_PER_MODULES = 100;
const unsigned int N_MAX_SEGMENTS_PER_MODULE = 600; //WHY!
const unsigned int MAX_CONNECTED_MODULES = 40;

struct SDL::modules* SDL::modulesInGPU = nullptr;
unsigned int SDL::nModules;

SDL::Event::Event()
{
    hitsInGPU = nullptr;
    mdsInGPU = nullptr;
    segmentsInGPU = nullptr;
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        n_segments_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
	    n_segments_by_layer_endcap_[i] = 0;
        }
    }
    resetObjectsInModule();
}

SDL::Event::~Event()
{
    mdsInGPU->freeMemory();
    cudaFree(mdsInGPU);
    hitsInGPU->freeMemory();
    cudaFree(hitsInGPU);
    segmentsInGPU->freeMemory();
    cudaFree(segmentsInGPU);
}

void SDL::initModules()
{
    cudaMallocManaged(&modulesInGPU, sizeof(struct SDL::modules));
    if((modulesInGPU->detIds) == nullptr) //check for nullptr and create memory
    {
        loadModulesFromFile(*modulesInGPU,nModules); //nModules gets filled here
    }
    resetObjectRanges(*modulesInGPU,nModules);
}

void SDL::cleanModules()
{
  freeModulesInUnifiedMemory(*modulesInGPU);
  cudaFree(modulesInGPU);
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(*modulesInGPU,nModules);
}

void SDL::Event::addHitToEvent(float x, float y, float z, unsigned int detId)
{
    const int HIT_MAX = 1000000;
    const int HIT_2S_MAX = 100000;

    if(hitsInGPU == nullptr)
    {
        cudaMallocManaged(&hitsInGPU, sizeof(SDL::hits));
        createHitsInUnifiedMemory(*hitsInGPU,HIT_MAX,HIT_2S_MAX);
    }
    //calls the addHitToMemory function
    addHitToMemory(*hitsInGPU, *modulesInGPU, x, y, z, detId);

    unsigned int moduleLayer = modulesInGPU->layers[(*detIdToIndex)[detId]];
    unsigned int subdet = modulesInGPU->subdets[(*detIdToIndex)[detId]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer-1]++;
    }
    else
    {
        n_hits_by_layer_endcap_[moduleLayer-1]++;
    }

}

void SDL::Event::addMiniDoubletsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(mdsInGPU->nMDs[idx] == 0 or modulesInGPU->hitRanges[idx * 2] == -1)
        {
            modulesInGPU->mdRanges[idx * 2] = -1;
            modulesInGPU->mdRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->mdRanges[idx * 2] = idx * N_MAX_MD_PER_MODULES;
            modulesInGPU->mdRanges[idx * 2 + 1] = (idx * N_MAX_MD_PER_MODULES) + mdsInGPU->nMDs[idx] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_minidoublets_by_layer_barrel_[modulesInGPU->layers[idx] -1] += mdsInGPU->nMDs[idx];
            }
            else
            {
                n_minidoublets_by_layer_endcap_[modulesInGPU->layers[idx] - 1] += mdsInGPU->nMDs[idx];
            }

        }
    }
}

void SDL::Event::addSegmentsToEvent()
{
    unsigned int idx;
    for(unsigned int i = 0; i<*(SDL::modulesInGPU->nLowerModules); i++)
    {
        idx = SDL::modulesInGPU->lowerModuleIndices[i];
        if(segmentsInGPU->nSegments[idx] == 0)
        {
            modulesInGPU->segmentRanges[idx * 2] = -1;
            modulesInGPU->segmentRanges[idx * 2 + 1] = -1;
        }
        else
        {
            modulesInGPU->segmentRanges[idx * 2] = idx * N_MAX_SEGMENTS_PER_MODULE;
            modulesInGPU->segmentRanges[idx * 2 + 1] = idx * N_MAX_SEGMENTS_PER_MODULE + segmentsInGPU->nSegments[idx] - 1;

            if(modulesInGPU->subdets[idx] == Barrel)
            {
                n_segments_by_layer_barrel_[modulesInGPU->layers[idx] - 1] += segmentsInGPU->nSegments[idx];
            }
            else
            {
                n_segments_by_layer_endcap_[modulesInGPU->layers[idx] -1] += segmentsInGPU->nSegments[idx];
            }
        }
    }
}

void SDL::Event::createMiniDoublets()
{
    cudaDeviceSynchronize();
    auto memStart = std::chrono::high_resolution_clock::now();
    if(mdsInGPU == nullptr)
    {
        cudaMallocManaged(&mdsInGPU, sizeof(SDL::miniDoublets));
    	createMDsInUnifiedMemory(*mdsInGPU, N_MAX_MD_PER_MODULES, nModules);
    }
    cudaDeviceSynchronize();
    auto memStop = std::chrono::high_resolution_clock::now();
    auto memDuration = std::chrono::duration_cast<std::chrono::milliseconds>(memStop - memStart); //in milliseconds
    std::cout<<"memory allocation took "<<memDuration.count()<<" ms"<<std::endl;

    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

//    dim3 nThreads(1,16,16);
//    dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nLowerModules/nThreads.x : nLowerModules/nThreads.x + 1),(N_MAX_HITS_PER_MODULE % nThreads.y == 0 ? N_MAX_HITS_PER_MODULE/nThreads.y : N_MAX_HITS_PER_MODULE/nThreads.y + 1), (N_MAX_HITS_PER_MODULE % nThreads.z == 0 ? N_MAX_HITS_PER_MODULE/nThreads.z : N_MAX_HITS_PER_MODULE/nThreads.z + 1));
    //std::cout<<nBlocks.x<<" "<<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;

    int nThreads = 1;
    int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    cudaDeviceSynchronize();
    auto syncStart = std::chrono::high_resolution_clock::now();

    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU,*hitsInGPU,*mdsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    auto syncStop = std::chrono::high_resolution_clock::now();

    auto syncDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(syncStop - syncStart);
    std::cout<<"sync took "<<syncDuration.count()<<" ms"<<std::endl;

    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }

    addMiniDoubletsToEvent();


}

void SDL::Event::createSegmentsWithModuleMap()
{
    if(segmentsInGPU == nullptr)
    {
        cudaMallocManaged(&segmentsInGPU, sizeof(SDL::segments));
        createSegmentsInUnifiedMemory(*segmentsInGPU, N_MAX_SEGMENTS_PER_MODULE, nModules);
    }
    unsigned int nLowerModules = *modulesInGPU->nLowerModules;

//    dim3 nThreads(1,16,16);
//    dim3 nBlocks(((nLowerModules * MAX_CONNECTED_MODULES)  % nThreads.x == 0 ? (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x : (nLowerModules * MAX_CONNECTED_MODULES)/nThreads.x + 1),(N_MAX_MD_PER_MODULES % nThreads.y == 0 ? N_MAX_MD_PER_MODULES/nThreads.y : N_MAX_MD_PER_MODULES/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0 ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));

    unsigned int nThreads = 1;
    unsigned int nBlocks = nLowerModules % nThreads == 0 ? nLowerModules/nThreads : nLowerModules/nThreads + 1;

    createSegmentsInGPU<<<nBlocks,nThreads>>>(*modulesInGPU, *hitsInGPU, *mdsInGPU, *segmentsInGPU);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess)
    {
        std::cout<<"sync failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;
    }
    addSegmentsToEvent();

}

/*__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;
    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);

    if(success)
    {
        unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;

        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
    }
}*/

__global__ void createMiniDoubletsFromLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, unsigned int lowerModuleIndex, unsigned int upperModuleIndex, unsigned int nLowerHits, unsigned int nUpperHits)
{
    unsigned int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);

    if(success)
    {
        unsigned int mdModuleIdx = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        unsigned int mdIdx = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIdx;

        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIdx);
    }
}


__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;

    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

    dim3 nThreads(1,16,16);
    dim3 nBlocks(1,nLowerHits % nThreads.y == 0 ? nLowerHits/nThreads.y : nLowerHits/nThreads.y + 1, nUpperHits % nThreads.z == 0 ? nUpperHits/nThreads.z : nUpperHits/nThreads.z + 1);

    createMiniDoubletsFromLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, lowerModuleIndex, upperModuleIndex, nLowerHits, nUpperHits);


}

/*__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
    int xAxisIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int innerMDArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int outerMDArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;

    int innerLowerModuleArrayIdx = xAxisIdx/MAX_CONNECTED_MODULES;
    int outerLowerModuleArrayIdx = xAxisIdx % MAX_CONNECTED_MODULES; //need this index from the connected module array

    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIdx];

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

    if(outerLowerModuleArrayIdx >= nConnectedModules) return;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

    if(innerMDArrayIdx >= nInnerMDs) return;
    if(outerMDArrayIdx >= nOuterMDs) return;

    unsigned int innerMDIndex = modulesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
    unsigned int outerMDIndex = modulesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;

        addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
    }


}*/


__global__ void createSegmentsFromInnerLowerModule(struct SDL::modules&modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, unsigned int innerLowerModuleIndex, unsigned int nInnerMDs)
{
    unsigned int outerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int innerMDArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int outerMDArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIndex];

    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];
    if(innerMDArrayIndex >= nInnerMDs) return;
    if(outerMDArrayIndex >= nOuterMDs) return;

    unsigned int innerMDIndex = innerLowerModuleIndex * N_MAX_MD_PER_MODULES + innerMDArrayIndex;
    unsigned int outerMDIndex = outerLowerModuleIndex * N_MAX_MD_PER_MODULES + outerMDArrayIndex;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;

        addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD,segmentIdx);
    }

}

__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
    int innerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIndex];
    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];
    if(nConnectedModules == 0) return;
    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
    if(nInnerMDs == 0) return;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nConnectedModules % nThreads.x == 0 ? nConnectedModules/nThreads.x : nConnectedModules/nThreads.x + 1), (nInnerMDs % nThreads.y == 0 ? nInnerMDs/nThreads.y : nInnerMDs/nThreads.y + 1), (N_MAX_MD_PER_MODULES % nThreads.z == 0 ? N_MAX_MD_PER_MODULES/nThreads.z : N_MAX_MD_PER_MODULES/nThreads.z + 1));
    createSegmentsFromInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerLowerModuleIndex,nInnerMDs);

}


unsigned int SDL::Event::getNumberOfHits()
{
    unsigned int hits = 0;
    for(auto &it:n_hits_by_layer_barrel_)
    {
        hits += it;
    }
    for(auto& it:n_hits_by_layer_endcap_)
    {
        hits += it;
    }

    return hits;
}

unsigned int SDL::Event::getNumberOfHitsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_hits_by_layer_barrel_[layer];
    else
        return n_hits_by_layer_barrel_[layer] + n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerBarrel(unsigned int layer)
{
    return n_hits_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerEndcap(unsigned int layer)
{
    return n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoublets()
{
     unsigned int miniDoublets = 0;
    for(auto &it:n_minidoublets_by_layer_barrel_)
    {
        miniDoublets += it;
    }
    for(auto &it:n_minidoublets_by_layer_endcap_)
    {
        miniDoublets += it;
    }

    return miniDoublets;

}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayer(unsigned int layer)
{
     if(layer == 6)
        return n_minidoublets_by_layer_barrel_[layer];
    else
        return n_minidoublets_by_layer_barrel_[layer] + n_minidoublets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer)
{
    return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer)
{
    return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfSegments()
{
     unsigned int segments = 0;
    for(auto &it:n_segments_by_layer_barrel_)
    {
        segments += it;
    }
    for(auto &it:n_segments_by_layer_endcap_)
    {
        segments += it;
    }

    return segments;

}

unsigned int SDL::Event::getNumberOfSegmentsByLayer(unsigned int layer)
{
     if(layer == 6)
        return n_segments_by_layer_barrel_[layer];
    else
        return n_segments_by_layer_barrel_[layer] + n_segments_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfSegmentsByLayerBarrel(unsigned int layer)
{
    return n_segments_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfSegmentsByLayerEndcap(unsigned int layer)
{
    return n_segments_by_layer_endcap_[layer];
}
