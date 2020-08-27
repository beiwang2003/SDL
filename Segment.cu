# include "Segment.cuh"

void SDL::createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules)
{
    cudaMallocManaged(&segmentsInGPU.mdIndices, maxSegments * nModules * 2 * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.innerLowerModuleIndices, maxSegments * nModules * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.outerLowerModuleIndices, maxSegments * nModules * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.innerMiniDoubletAnchorHitIndices, maxSegments * nModules *sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.outerMiniDoubletAnchorHitIndices, maxSegments * nModules * sizeof(unsigned int));
    cudaMallocManaged(&segmentsInGPU.nSegments, nModules * sizeof(unsigned int));
#pragma omp parallel for default(shared)
    for(size_t i = 0; i < nModules; i++)
    {
        segmentsInGPU.nSegments[i] = 0;
    }

    cudaMallocManaged(&segmentsInGPU.dPhis, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dPhiMins, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dPhiMaxs, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dPhiChanges, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dPhiChangeMins, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dPhiChangeMaxs, maxSegments * nModules * sizeof(float));

    cudaMallocManaged(&segmentsInGPU.zIns, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.zOuts, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.rtIns, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.rtOuts, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaInnerMDSegments, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaOuterMDSegments, maxSegments * nModules * sizeof(float));
    cudaMallocManaged(&segmentsInGPU.dAlphaInnerMDOuterMDs, maxSegments * nModules * sizeof(float));
}

SDL::segments::segments()
{
    mdIndices = nullptr;
    innerLowerModuleIndices = nullptr;
    outerLowerModuleIndices = nullptr;
    innerMiniDoubletAnchorHitIndices = nullptr;
    outerMiniDoubletAnchorHitIndices = nullptr;

    nSegments = nullptr;
    dPhis = nullptr;
    dPhiMins = nullptr;
    dPhiMaxs = nullptr;
    dPhiChanges = nullptr;
    dPhiChangeMins = nullptr;
    dPhiChangeMaxs = nullptr;

    zIns = nullptr;
    zOuts = nullptr;
    rtIns = nullptr;
    rtOuts = nullptr;
    dAlphaInnerMDSegments = nullptr;
    dAlphaOuterMDSegments = nullptr;
    dAlphaInnerMDOuterMDs = nullptr;
}

SDL::segments::~segments()
{
    cudaFree(mdIndices);
    cudaFree(innerLowerModuleIndices);
    cudaFree(outerLowerModuleIndices);
    cudaFree(innerMiniDoubletAnchorHitIndices);
    cudaFree(outerMiniDoubletAnchorHitIndices);

    cudaFree(nSegments);
    cudaFree(dPhis);
    cudaFree(dPhiMins);
    cudaFree(dPhiMaxs);
    cudaFree(dPhiChanges);
    cudaFree(dPhiChangeMins);
    cudaFree(dPhiChangeMaxs);

    cudaFree(zIns);
    cudaFree(zOuts);
    cudaFree(rtIns);
    cudaFree(rtOuts);
    cudaFree(dAlphaInnerMDSegments);
    cudaFree(dAlphaOuterMDSegments);
    cudaFree(dAlphaInnerMDOuterMDs);

}

__device__ void SDL::addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float& dAlphaInnerMDOuterMD, unsigned int idx)
{
    //idx will be computed in the kernel, which is the index into which the 
    //segment will be written
    //nSegments will be incremented in the kernel

    segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;

    segmentsInGPU.dPhis[idx] = dPhi;
    segmentsInGPU.dPhiMins[idx] = dPhiMin;
    segmentsInGPU.dPhiMaxs[idx] = dPhiMax;
    segmentsInGPU.dPhiChanges[idx] = dPhiChange;
    segmentsInGPU.dPhiChangeMins[idx] = dPhiChangeMin;
    segmentsInGPU.dPhiChangeMaxs[idx] = dPhiChangeMax;

    segmentsInGPU.zIns[idx] = zIn;
    segmentsInGPU.zOuts[idx] = zOut;
    segmentsInGPU.rtIns[idx] = rtIn;
    segmentsInGPU.rtOuts[idx] = rtOut;
    segmentsInGPU.dAlphaInnerMDSegments[idx] = dAlphaInnerMDSegment;
    segmentsInGPU.dAlphaOuterMDSegments[idx] = dAlphaOuterMDSegment;
    segmentsInGPU.dAlphaInnerMDOuterMDs[idx] = dAlphaInnerMDOuterMD;
}

__device__ void SDL::dAlphaThreshold(float* dAlphaThresholdValues, struct hits& hitsInGPU, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex)
{
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;

    // BField dAlpha

    float innerMiniDoubletAnchorHitRt = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    float outerMiniDoubletAnchorHitRt = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    float innerMiniDoubletAnchorHitZ = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    float outerMiniDoubletAnchorHitZ = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    //more accurate then outer rt - inner rt

    float segmentDr = sqrt(pow(hitsInGPU.ys[outerMiniDoubletAnchorHitIndex] - hitsInGPU.ys[innerMiniDoubletAnchorHitIndex],2) + pow(hitsInGPU.xs[outerMiniDoubletAnchorHitIndex]- hitsInGPU.xs[innerMiniDoubletAnchorHitIndex],2));
    

    const float dAlpha_Bfield = asin(min(segmentDr * k2Rinv1GeVf/ptCut, sinAlphaMax));
    const float pixelPSZpitch = 0.15;

    bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] != SDL::Center;
    bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] != SDL::Center;
    const float drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
    const float drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];

    float innerModuleGapSize = SDL::moduleGapSize(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = SDL::moduleGapSize(modulesInGPU, outerLowerModuleIndex);

    const float innerminiTilt = isInnerTilted ? (0.5f * pixelPSZpitch * drdzInner / sqrt(1.f + drdzInner * drdzInner) / innerModuleGapSize) : 0;

    const float outerminiTilt = isOuterTilted ? (0.5f * pixelPSZpitch * drdzOuter / sqrt(1.f + drdzOuter * drdzOuter) / outerModuleGapSize) : 0;

    float miniDelta = innerModuleGapSize; 
 

    float sdLumForInnerMini;    
    float sdLumForOuterMini;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
    {
        sdLumForInnerMini = innerminiTilt * dAlpha_Bfield;
    }
    else
    {
        sdLumForInnerMini = mdsInGPU.dphis[innerMDIndex] * 15.0 / mdsInGPU.dzs[innerMDIndex];
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
    {
        sdLumForOuterMini = outerminiTilt * dAlpha_Bfield;
    }
    else
    {
        sdLumForOuterMini = mdsInGPU.dphis[outerMDIndex] * 15.0 / mdsInGPU.dzs[outerMDIndex];
    }


    //Unique stuff for the segment dudes alone

    float dAlpha_res_inner = 0.02f/miniDelta;
    float dAlpha_res_outer = 0.02f/miniDelta;

    if(modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Endcap)
    {
        dAlpha_res_inner *= fabs(innerMiniDoubletAnchorHitZ/innerMiniDoubletAnchorHitRt);
    }

    if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap)
    {
        dAlpha_res_outer *= fabs(outerMiniDoubletAnchorHitZ/outerMiniDoubletAnchorHitRt);
    }


    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[innerLowerModuleIndex] == SDL::Center)
    {
        dAlphaThresholdValues[0] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));
    }
    else
    {
        dAlphaThresholdValues[0] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2) + pow(sdLumForInnerMini,2));
    }

    if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and modulesInGPU.sides[outerLowerModuleIndex] == SDL::Center)
    {
        dAlphaThresholdValues[1] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));
    }
    else
    {
        dAlphaThresholdValues[1] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2) + pow(sdLumForOuterMini,2));
    }

    //Inner to outer 
    dAlphaThresholdValues[2] = dAlpha_Bfield + sqrt(pow(dAlpha_res,2) + pow(sdMuls,2));

}

__device__ bool SDL::runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    bool pass = true;
    
//    unsigned int innerMiniDoubletAnchorHitIndex;
//    unsigned int outerMiniDoubletAnchorHitIndex;

    if(mdsInGPU.pixelModuleFlag[innerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[innerMDIndex] == 0)
        {    
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2]; 
        }
        else
        {
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2 + 1];
        }
    }
    else
    {
        innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2];
    }

    if(mdsInGPU.pixelModuleFlag[outerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[outerMDIndex] == 0)
        {    
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2]; 
        }
        else
        {
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2 + 1];
        }
    }
    else
    {
        outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2];
    }

    rtIn = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    rtOut = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    zIn = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    zOut = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    bool outerLayerEndcapTwoS = hitsInGPU.edge2SMap[outerMiniDoubletAnchorHitIndex] >= 0;
    
    float sdSlope = std::asin(min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1/rtOut;
    float pixelPSZpitch = 0.15;
    float strip2SZpitch = 5.0;
    float disks2SMinRadius = 60.f;

    float rtGeom =  ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius) ? (2.f * pixelPSZpitch)
            : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
            : (2.f * strip2SZpitch)));


    //cut 0 - z compatibility
    if(zIn * zOut < 0)
    {
        pass = false;
    }

    float dz = zOut - zIn;
    float dLum = std::copysign(deltaZLum, zIn);
    float drtDzScale = std::tan(sdSlope)/sdSlope;

    float rtLo = max(rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom,  rtIn - 0.5f * rtGeom); //rt should increase
    float rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom; //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction
    
    if(not(rtOut >= rtLo and rtOut <= rtHi))
    {
        pass = false;
    }

    dPhi = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

    float sdCut = sdSlope;
    unsigned int outerEdgeIndex;
    if(outerLayerEndcapTwoS)
    {
        outerEdgeIndex = hitsInGPU.edge2SMap[outerMiniDoubletAnchorHitIndex];

        float dPhiPos_high = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.highEdgeXs[outerEdgeIndex], hitsInGPU.highEdgeYs[outerEdgeIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

        float dPhiPos_low = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.lowEdgeXs[outerEdgeIndex], hitsInGPU.lowEdgeYs[outerEdgeIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

        dPhiMax = abs(dPhiPos_high) > abs(dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
        dPhiMin = abs(dPhiPos_high) > abs(dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    }
    else
    {
        dPhiMax = dPhi;
        dPhiMin = dPhi;
    }

    if(fabs(dPhi) > sdCut)
    {
        pass = false;
    }

    float dzFrac = dz/zIn;
    dPhiChange = dPhi/dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin/dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax/dzFrac * (1.f + dzFrac);

    if(fabs(dPhiChange) > sdCut)
    {
        pass = false;
    }

    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, hitsInGPU, modulesInGPU, mdsInGPU, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;
    
    if(fabs(dAlphaInnerMDSegment) >= dAlphaThresholdValues[0])
    {
        pass = false;
    }

    if(fabs(dAlphaOuterMDSegment) >= dAlphaThresholdValues[1])
    {
        pass = false;
    }

    if(fabs(dAlphaInnerMDOuterMD) >= dAlphaThresholdValues[2])
    {
        pass = false;
    }


    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    bool pass = true;
   
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex]-1] * 3.f/ptCut;


//    unsigned int innerMiniDoubletAnchorHitIndex;
//    unsigned int outerMiniDoubletAnchorHitIndex;

    if(mdsInGPU.pixelModuleFlag[innerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[innerMDIndex] == 0)
        {    
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2]; 
        }
        else
        {
            innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2 + 1];
        }
    }
    else
    {
        innerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[innerMDIndex * 2];
    }

    if(mdsInGPU.pixelModuleFlag[outerMDIndex] >= 0)
    {
        if(mdsInGPU.pixelModuleFlag[outerMDIndex] == 0)
        {    
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2]; 
        }
        else
        {
            outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2 + 1];
        }
    }
    else
    {
        outerMiniDoubletAnchorHitIndex = mdsInGPU.hitIndices[outerMDIndex * 2];
    }


    rtIn = hitsInGPU.rts[innerMiniDoubletAnchorHitIndex];
    rtOut = hitsInGPU.rts[outerMiniDoubletAnchorHitIndex];
    zIn = hitsInGPU.zs[innerMiniDoubletAnchorHitIndex];
    zOut = hitsInGPU.zs[outerMiniDoubletAnchorHitIndex];

    float sdSlope = std::asin(min(rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1f/rtOut;
    float dzDrtScale = std::tan(sdSlope)/sdSlope; //FIXME: need appropriate value
    float pixelPSZpitch = 0.15;
    float strip2SZpitch = 5.0;

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch;

    float zLo = zIn + (zIn - deltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom; //slope-correction only on outer end
    float zHi = zIn + (zIn + deltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;


    //cut 1 - z compatibility
    if(not(zOut >= zLo and zOut <= zHi))
    {
        pass = false;
    }


    dPhi = deltaPhi(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);
    float sdCut = sdSlope + sqrt(sdMuls * sdMuls + sdPVoff * sdPVoff);

    if(not( fabs(dPhi) <= sdCut ))
    {
        pass = false;
    }

    dPhiChange = deltaPhiChange(hitsInGPU.xs[innerMiniDoubletAnchorHitIndex], hitsInGPU.ys[innerMiniDoubletAnchorHitIndex], hitsInGPU.zs[innerMiniDoubletAnchorHitIndex], hitsInGPU.xs[outerMiniDoubletAnchorHitIndex], hitsInGPU.ys[outerMiniDoubletAnchorHitIndex], hitsInGPU.zs[outerMiniDoubletAnchorHitIndex]);

    if(not( fabs(dPhiChange) <= sdCut ))
    {
        pass = false;
    }
    
    float dAlphaThresholdValues[3];
    dAlphaThreshold(dAlphaThresholdValues, hitsInGPU, modulesInGPU, mdsInGPU, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;
    
    if(fabs(dAlphaInnerMDSegment) >= dAlphaThresholdValues[0])
    {
        pass = false;
    }

    if(fabs(dAlphaOuterMDSegment) >= dAlphaThresholdValues[1])
    {
        pass = false;
    }

    if(fabs(dAlphaInnerMDOuterMD) >= dAlphaThresholdValues[2])
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runSegmentDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex)
{
    bool pass = true;

    if(modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel)
    {
        pass = runSegmentDefaultAlgoBarrel(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
    }

    else
    {
        pass = runSegmentDefaultAlgoEndcap(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);
    }

    return pass;
}
