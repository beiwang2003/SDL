# include "MiniDoublet.cuh"

void createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs, unsigned int nModules)
{
    cudaMallocManaged(&mdsInGPU.hitIndices, maxMDs * 2 * sizeof(unsigned int));
    cudaMallocManaged(&mdsInGPU.moduleIndices, maxMDs * sizeof(unsigned int));
    cudaMallocManaged(&mdsInGPU.pixelModuleFlag, maxMDs * sizeof(short));
    cudaMallocManaged(&mdsInGPU.dphichanges, maxMDs * sizeof(float));

    cudaMallocManaged(&mdsInGPU.nMDs, nModules * sizeof(int));

    cudaMallocManaged(&mdsInGPU.dzs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.dphis, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedXs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedYs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedZs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDz, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDphis, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDphiChanges, maxMDs * sizeof(float));
}

void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx)
{
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here
    
    mdsInGPU.hitIndices[idx * 2] = lowerHitIdx;
    mdsInGPU.hitIndices[idx * 2 + 1] = upperHitIdx;
    mdsInGPU.moduleIndices[idx] = lowerModuleIdx;
    if(modulesInGPU.moduleType(lowerModuleIdx) == PS
    {
        if(modulesInGPU.moduleLayerType(lowerModuleIdx) == Pixel)
        {
            pixelModuleFlag = 0;
        }
        else
        {
            pixelModuleFlag = 1;
        }
    }
    else
    {
        pixelModuleFlag = -1;
    }

    mdsInGPU.dphichanges[idx] = dphichange;

    mdsInGPU.dphis[idx] = dphi;
    mdsInGPU.dzs[idx] = dz;
    mdsInGPU.shiftedXs[idx] = shiftedX;
    mdsInGPU.shiftedYs[idx] = shiftedY;
    mdsInGPU.shiftedZs[idx] = shiftedZ;

    mdsInGPU.noShiftedDzs[idx] = noshiftedDz;
    mdsInGPU.noShiftedDphis[idx] = noShiftedDphi;
    mdsInGPU.noShfitedDphiChanges[idx] = noShiftedDphiChange;
}

bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphichange)
{
    float xLower = hitsInGPU.xs[lowerHitIndex];
    float yLower = hitsInGPU.ys[lowerHitIndex];
    float zLower = hitsInGPU.zs[lowerHitIndex];

    float xUpper = hitsInGPU.xs[upperHitIndex];
    float yUpper = hitsInGPU.ys[upperHitIndex];
    float zUpper = hitsInGPU.zs[upperHitIndex];

    bool pass = true; 
    dz = zLower - zUpper;     
    const float dzCut = modulesInGPU.moduleType(lowerModuleIndex) == PS ? 2.f : 10.f;
    const float sign = ((dz > 0) - (dz < 0)) * ((hitsInGPU.zs[lowerHitIndex] > 0) - (hitsInGPU.zs[lowerHitIndex] < 0));
    const float invertedcrossercut = (fabs(dz) > 2) * sign;


    //cut convention - when a particular cut fails, the pass variable goes to false
    //but all cuts will be checked even if a previous cut has failed, this is
    //to prevent thread divergence

    if (not (fabs(dz) < dzCut and invertedcrossercut <= 0)) // Adding inverted crosser rejection
    {
        pass = false;
    }

    float miniCut = 0;

    if (modulesInGPU.moduleLayerType(lowerModule) == Pixel)
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, lowerHitIndex, lowerModuleIndex);
    }
    else
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, upperHitIndex, lowerModuleIndex);
    }

    // Cut #2: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
    float xn = 0, yn = 0 , zn = 0;
    float shiftedRt;
    if (modulesInGPU.sides[lowerModuleIndex] != Center) // If barrel and not center it is tilted
    {
        // Shift the hits and calculate new xn, yn position
        float shiftedCoords[3];
        shiftStripHits(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, shiftedCoords);
        xn = shiftedCoords[0];
        yn = shiftedCoords[1];
        zn = shiftedCoords[2];

        // Lower or the upper hit needs to be modified depending on which one was actually shifted
        if (modulesInGPU.moduleLayerType(lowerModuleIndex) == Pixel)
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            shiftedRt = sqrt(xn * xn + yn * yn);

            dPhi = deltaPhi(xLower,yLower,zLower,shiftedX, shiftedY, shiftedZ); //function from Hit.cu
            noShiftedDphi = deltaPhi(x,y,z);
        }
        else
        {
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zLower;
            shiftedRt = sqrt(xn * xn + yn * yn);
            dPhi = deltaPhi(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper);
            noShiftedDphi = deltaPhi(x,y,z)

        }
    }
    else
    {
        dPhi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        noShiftedDphi = dPhi;
    }


    if (not (fabs(dPhi) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #3: The dphi change going from lower Hit to upper Hit
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
    if (modulesInGPU.sides[lowerModule]!= Center)
    {
        // When it is tilted, use the new shifted positions
        if (modulesInGPU.moduleLayerType(lowerModuleIndex) == Pixel)
        {
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)
            // setDeltaPhiChange(lowerHit.rt() < upperHitMod.rt() ? lowerHit.deltaPhiChange(upperHitMod) : upperHitMod.deltaPhiChange(lowerHit));


            dPhiChange = (hitsInGPU.rt[lowerHitIndex] < shiftedRt) ? deltaPhiChange(xLower, yLower, zLower, shiftedX, shiftedY, shiftedZ) : deltaPhiChange(shfitedX, shiftedY, shiftedZ, xLower, yLower, zLower); 
            noShiftedDphichange = hitsInGPU.rts[lowerHitIndex] < hitsInGPU.rts[upperHitIndex] ? deltaPhiChange(xLower,yLower, zLower, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, xLower, yLower, zLower);
        }
        else
        {
            // dPhi Change should be calculated so that the upper hit has higher rt.
            // In principle, this kind of check rt_lower < rt_upper should not be necessary because the hit shifting should have taken care of this.
            // (i.e. the strip hit is shifted to be aligned in the line of sight from interaction point to pixel hit of PS module guaranteeing rt ordering)
            // But I still placed this check for safety. (TODO: After cheking explicitly if not needed remove later?)

            dPhiChange = (shiftedRt < hitsInGPU.rts[upperHitIdx]) ? deltaPhiChange(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, shiftedX, shiftedY, shiftedZ);
            noShiftedDphichange = hitsInGPU.rts[lowerHitIndex] < hitsInGPU.rts[upperHitIndex] ? deltaPhiChange(xLower,yLower, zLower, xUpper, yUpper, zUpper) : deltaPhiChange(xUpper, yUpper, zUpper, xLower, yLower, zLower);
        }
    }
    else
    {
        // When it is flat lying module, whichever is the lowerSide will always have rt lower
        dPhiChange = deltaPhiChange(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        noShfitedDphiChange = dPhiChange;
    }

    if (not (fabs(dPhiChange) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    return pass;
}

bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& drt, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphichange)
{
    float xLower = hitsInGPU.xs[lowerHitIndex];
    float yLower = hitsInGPU.ys[lowerHitIndex];
    float zLower = hitsInGPU.zs[lowerHitIndex];

    float xUpper = hitsInGPU.xs[upperHitIndex];
    float yUpper = hitsInGPU.ys[upperHitIndex];
    float zUpper = hitsInGPU.zs[upperHitIndex];

    bool pass = true; 

    // There are series of cuts that applies to mini-doublet in a "endcap" region

    // Cut #1 : dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
    // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.
    // This is because the 10.f cut is meant more for sanity check (most will pass this cut anyway) (TODO: Maybe revisit this cut later?)

    float dz = zLower - zUpper; // Not const since later it might change depending on the type of module

    const float dzCut = ((modulesInGPU.sides[lowerModuleIndex] == Endcap) ?  1.f : 10.f);
    if (not (fabs(dz) < dzCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #2 : drt cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
    const float drtCut = modulesInGPU.moduleType(lowerModuleIndex) == PS ? 2.f : 10.f;
    drt = hitsInGPU.rts[lowerHitIndex] - hitsInGPU.rts[upperHitIndex];
    if (not (fabs(drt) < drtCut)) // If cut fails continue
    {
        pass = false;
    }

    // Calculate the cut thresholds for the selection
    
    // Cut #3: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3111
    // // Old comments ----
    // // Old comments Slava, 6:17 PM
    // // Old comments here for the code to work you would need to slide (line extrapolate) the lower or the upper  hit along the strip direction to the radius of the other
    // // Old comments you'll get it almost right by assuming radial strips and just add the d_rt*(cosPhi, sinPhi)
    // // Old comments ----
    // // Old comments The algorithm assumed that the radial position is ~close according to Slava.
    // // Old comments However, for PS modules, it is not the case.
    // // Old comments So we'd have to move the hits to be in same position as the other.
    // // Old comments We'll move the pixel along the radial direction (assuming the radial direction is more or less same as the strip direction)
    // ----
    // The new scheme shifts strip hits to be "aligned" along the line of sight from interaction point to the pixel hit (if it is PS modules)
    float xn = 0, yn = 0, zn = 0;
    float shiftedRt;

    float shiftedCoords[3];
    shiftStripHits(lowerHit, upperHit, lowerModule, shiftedCoords, logLevel);
    xn = shiftedCoords[0];
    yn = shiftedCoords[1];
    zn = shiftedCoords[2];

    if (modulesInGPU.moduleType(lowerModuleIndex) == PS)
    {
        // Appropriate lower or upper hit is modified after checking which one was actually shifted
        if (modulesInGPU.moduleLayerType(lowerModuleIndex) == Pixel)
        {
            // SDL::Hit upperHitMod(upperHit);
            // upperHitMod.setXYZ(xn, yn, upperHit.z());
            // setDeltaPhi(lowerHit.deltaPhi(upperHitMod));
            setUpperShiftedHit(xn, yn, upperHit.z());
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zUpper;
            dPhi = deltaPhi(xLower, yLower, zLower, shiftedX, shiftedY, shiftedZ);
            noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        }
        else
        {
            // SDL::Hit lowerHitMod(lowerHit);
            // lowerHitMod.setXYZ(xn, yn, lowerHit.z());
            // setDeltaPhi(lowerHitMod.deltaPhi(upperHit));
            shiftedX = xn;
            shiftedY = yn;
            shiftedZ = zLower;
            dPhi = deltaPhi(shiftedX, shiftedY, shiftedZ, xUpper, yUpper, zUpper);
            noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
        }
    }
    else
    {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        dPhi = deltaPhi(xLower, yLower, zLower, xn, yn, zUpper);
        noShiftedDphi = deltaPhi(xLower, yLower, zLower, xUpper, yUpper, zUpper);
    }

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (modulesInGPU.moduleType(lowerModuleIndex) == PS)
    {
        if (modulesInGPU.moduleLayerType(lowerModuleIndex) == Pixel)
        {
            dz = zLower - zn;
        }
        else
        {
            dz = zUpper - zn;
        }
    }

    float miniCut = 0;
    if(modulesInGPU.moduleLayerType(lowerModuleIndex) == Pixel)
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, lowerHitIndex, lowerModuleIndex,dPhi, dz);
    }
    else
    {
        miniCut = dPhiThreshold(hitsInGPU, modulesInGPU, upperHitIndex, lowerModuleIndex, dPhi, dz);
    }

    if (not (fabs(dPhi) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    // Cut #4: Another cut on the dphi after some modification
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    
    float dzFrac = fabs(dz) / fabs(zLower);
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);
    if (not (fabs(dPhiChange) < miniCut)) // If cut fails continue
    {
        pass = false;
    }

    return pass;
}

bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange)
{
    bool pass;
   if(lowerModule.subdets[lowerModuleIndex] == Barrel)
   {
        pass = runMiniDoubletDefaultAlgoBarrel(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphichange);
   } 
   else
   {
       pass = runMiniDoubletDefaultAlgoEndcap(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitIndex, upperHitIndex, dz, dPhi, dPhiChange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphichange);

   }
   return pass;
}

float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex, float dPhi, float dz)
{
    // =================================================================
    // Various constants
    // =================================================================
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    // const float ptCut = PTCUT;
    // const float sinAlphaMax = 0.95;
    float ptCut = 1;
    float sinAlphaMax = 0.95;
    float miniMulsPtScaleBarrel[] = {0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
    float miniMulsPtScaleEndcap[] =  {0.006, 0.006, 0.006, 0.006, 0.006}; //inter/extra-polated from L11 and L13 both roughly 0.006 [larger R have smaller value by ~50%]
    //mean of the horizontal layer position in y; treat this as R below
    float miniRminMeanBarrel[] = {21.8, 34.6, 49.6, 67.4, 87.6, 106.8}; // TODO: Update this with newest geometry
    float miniRminMeanEndcap[] = {131.4, 156.2, 185.6, 220.3, 261.5};// use z for endcaps // TODO: Update this with newest geometry

    // =================================================================
    // Computing some components that make up the cut threshold
    // =================================================================

    float rt = hitsInGPU.rts[hitIndex];
    unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
    const float miniSlope = std::asin(min(rt * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float rLayNominal = ((modulesInGPU.subdets[moduleIndex]== Barrel) ? miniRminMeanBarrel[iL] : miniRminMeanEndcap[iL]);
    const float miniPVoff = 0.1 / rLayNominal;
    const float miniMuls = ((modulesInGPU.subdets[moduleIndex] == Barrel) ? miniMulsPtScaleBarrel[iL] * 3.f / ptCut : miniMulsPtScaleEndcap[iL] * 3.f / ptCut);
    const bool isTilted = modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] != Center;
    const bool tiltedOT123 = true;
    const float pixelPSZpitch = 0.15;
    const float drdz = isTilted && tiltedOT123 ? modulesInGPU.drdzs[moduleIndex] : 0;
    const float miniTilt = ((isTilted && tiltedOT123) ? 0.5f * pixelPSZpitch * drdz / sqrt(1.f + drdz * drdz) / moduleGapSize(modulesInGPU,moduleIndex) : 0);

    // Compute luminous region requirement for endcap
    const float deltaZLum = 15.f;
    const float miniLum = fabs(dPhi * deltaZLum/dz); // Balaji's new error
    // const float miniLum = abs(deltaZLum / lowerHit.z()); // Old error


    // =================================================================
    // Return the threshold value
    // =================================================================
    // Following condition is met if the module is central and flatly lying
    if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] == Center)
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2));
    }
    // Following condition is met if the module is central and tilted
    else if (modulesInGPU.subdets[moduleIndex] == Barrel and modulesInGPU.sides[moduleIndex] != Center) //all types of tilted modules
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2) + pow(miniTilt * miniSlope, 2));
    }
    // If not barrel, it is Endcap
    else
    {
        return miniSlope + sqrt(pow(miniMuls, 2) + pow(miniPVoff, 2) + pow(miniLum, 2));
    }

}

inline float isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    if (
           (subdet == Barrel and side != Center and layer== 3)
           or (subdet == Barrel and side == NegZ and layer == 2 and rod > 5)
           or (subdet == Barrel and side == PosZ and layer == 2 and rod < 8)
           or (subdet == Barrel and side == NegZ and layer == 1 and rod > 9)
           or (subdet == Barrel and side == PosZ and layer == 1 and rod < 4)
       )
        return true;
    else
        return false;

}

inline float moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    float miniDeltaTilted[] = {0.26, 0.26, 0.26};
    float miniDeltaLooseTilted[] =  {0.4,0.4,0.4};
    float miniDeltaFlat[] =  {0.26, 0.16, 0.16, 0.18, 0.18, 0.18};
    float miniDeltaEndcap[5][15];

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
        }
    }

    unsigned int iL = modulesInGPU.layers[moduleIndex]-1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules(modulesInGPU, moduleIndex))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (subdet == Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }
}

void shiftStripHits(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords)
{

    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry
    // SDL::tiltedGeometry

    // Some variables relevant to the function
    float xp; // pixel x (pixel hit x)
    float yp; // pixel y (pixel hit y)
    float xa; // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya; // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo; // old x (before the strip hit is moved up or down)
    float yo; // old y (before the strip hit is moved up or down)
    float xn; // new x (after the strip hit is moved up or down)
    float yn; // new y (after the strip hit is moved up or down)
    float abszn; // new z in absolute value
    float zn; // new z with the sign (+/-) accounted
    float angleA; // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB; // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    bool isEndcap; // If endcap, drdz = infinity
    unsigned int pixelHitIndex; // Pointer to the pixel hit
    unsigned int stripHitIndex; // Pointer to the strip hit
    float moduleSeparation;
    float drprime; // The radial shift size in x-y plane projection
    float drprime_x; // x-component of drprime
    float drprime_y; // y-component of drprime
    float slope; // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime; // The distance between the two points after shifting
    float drdz_;

    // Assign hit pointers based on their hit type
    if (modulesInGPU.moduleType(lowerModuleIndex) == PS)
    {
        if (modulesInGPU.moduleLayerType(lowerModuleIndex)== Pixel)
        {
            pixelHitIndex = lowerHitIndex;
            stripHitIndex = upperHitIndex;
        }
        else
        {
            pixelHitIndex = upperHitIndex;
            stripHitIndex = lowerHitIndex;
        }
    }
    else // if (lowerModule.moduleType() == SDL::Module::TwoS) // If it is a TwoS module (if this is called likely an endcap module) then anchor the inner hit and shift the outer hit
    {
        pixelHitIndex = lowerHitIndex; // Even though in this case the "pixelHitPtr" is really just a strip hit, we pretend it is the anchoring pixel hit
        stripHitIndex = upperHitIndex;
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = modulesInGPU.subdets[lowerModuleIndex]== SDL::Module::Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time

    // The pixel hit is used to compute the angleA which is the theta in polar coordinate
    // angleA = std::atan(pixelHitPtr->rt() / pixelHitPtr->z() + (pixelHitPtr->z() < 0 ? M_PI : 0)); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2

    angleA = fabs(std::atan(hitsInGPU.rts[pixelHitIndex] / hitsInGPU.zs[pixelHitIndex]));
    // angleB = isEndcap ? M_PI / 2. : -std::atan(tiltedGeometry.getDrDz(detid) * (lowerModule.side() == SDL::Module::PosZ ? -1 : 1)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa
    drdz_ = modulesInGPU.drdzs[lowerModuleIndex];
    angleB = ((isEndcap) ? M_PI / 2. : atan(drdz_)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa


    moduleSeparation = moduleGapSize(modulesInGPU, moduleIndex);

    // Sign flips if the pixel is later layer
    if (modulesInGPU.moduleType(lowerModuleIndex) == PS and modulesInGPU.moduleLayerType(lowerModuleIndex) != Pixel)
    {
        moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / std::sin(angleA + angleB)) * std::sin(angleA);
    slope = modulesInGPU.slopes[moduleIndex];

    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != SDL_INF) ? fabs(std::atan(slope)) : M_PI / 2); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

    // The pixel hit position
    xp = hitsInGPU.xs[pixelHitIndex];
    yp = hitsInGPU.ys[pixelHitIndex];

    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0)
    {
        angleM = absArctanSlope;
    }
    else if (xp > 0 and yp < 0)
    {
        angleM = M_PI - absArctanSlope;
    }
    else if (xp < 0 and yp < 0)
    {
        angleM = M_PI + absArctanSlope;
    }
    else // if (xp < 0 and yp > 0)
    {
        angleM = 2 * M_PI - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * std::sin(angleM);
    drprime_y = drprime * std::cos(angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // The original strip hit position
    xo = hitsInGPU.xs[stripHitIndex];
    yo = hitsInGPU.ys[stripHitIndex];

    // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
    if (slope == SDL_INF) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
    {
        xn = xa; // New x point is simply where the anchor is
        yn = yo; // No shift in y
    }
    else if (slope == 0)
    {
        xn = xo; // New x point is simply where the anchor is
        yn = ya; // No shift in y
    }
    else
    {
        xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
        yn = (xn - xa) * slope + ya; // new yn
    }

    // Computing new Z position
    absdzprime = fabs(moduleSeparation / std::sin(angleA + angleB) * std::cos(angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        abszn = std::abs(hitsInGPU.zs[pixelHitIndex]) + absdzprime;
    }
    else
    {
        abszn = std::abs(hitsInGPU.zs[pixelHitIndex]) - absdzprime;
    }

    zn = abszn * ((hitsInGPU.zs[pixelHitIndex] > 0) ? 1 : -1); // Apply the sign of the zn

/*    if (logLevel == SDL::Log_Debug3)
    {
        SDL::cout << upperHit << std::endl;
        SDL::cout << lowerHit << std::endl;
        SDL::cout <<  " lowerModule.moduleType()==SDL::Module::PS: " << (lowerModule.moduleType()==SDL::Module::PS) <<  std::endl;
        SDL::cout <<  " lowerModule.moduleLayerType()==SDL::Module::Pixel: " << (lowerModule.moduleLayerType()==SDL::Module::Pixel) <<  std::endl;
        SDL::cout <<  " pixelHitPtr: " << pixelHitPtr <<  std::endl;
        SDL::cout <<  " stripHitPtr: " << stripHitPtr <<  std::endl;
        SDL::cout <<  " detid: " << detid <<  std::endl;
        SDL::cout <<  " isEndcap: " << isEndcap <<  std::endl;
        SDL::cout <<  " pixelHitPtr->rt(): " << pixelHitPtr->rt() <<  std::endl;
        SDL::cout <<  " pixelHitPtr->z(): " << pixelHitPtr->z() <<  std::endl;
        SDL::cout <<  " angleA: " << angleA <<  std::endl;
        SDL::cout <<  " angleB: " << angleB <<  std::endl;
        SDL::cout <<  " moduleSeparation: " << moduleSeparation <<  std::endl;
        SDL::cout <<  " drprime: " << drprime <<  std::endl;
        SDL::cout <<  " slope: " << slope <<  std::endl;
        SDL::cout <<  " absArctanSlope: " << absArctanSlope <<  std::endl;
        SDL::cout <<  " angleM: " << angleM <<  std::endl;
        SDL::cout <<  " drprime_x: " << drprime_x <<  std::endl;
        SDL::cout <<  " drprime_y: " << drprime_y <<  std::endl;
        SDL::cout <<  " xa: " << xa <<  std::endl;
        SDL::cout <<  " ya: " << ya <<  std::endl;
        SDL::cout <<  " xo: " << xo <<  std::endl;
        SDL::cout <<  " yo: " << yo <<  std::endl;
        SDL::cout <<  " xn: " << xn <<  std::endl;
        SDL::cout <<  " yn: " << yn <<  std::endl;
        SDL::cout <<  " absdzprime: " << absdzprime <<  std::endl;
        SDL::cout <<  " zn: " << zn <<  std::endl;
    }*/

    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;

}

miniDoublets::~miniDoublets()
{
    cudaFree(hitIndices);
    cudaFree(moduleIndices);
    cudaFree(pixelModuleFlag);
    cudaFree(nMDs);
    cudaFree(dphichanges);

    cudaFree(dzs);
    cudaFree(dphis);

    cudaFree(shiftedXs);
    cudaFree(shiftedYs);
    cudaFree(shiftedZs);
    cudaFree(noShiftedDzs);
    cudaFree(noShiftedDphis);
    cudaFree(noShiftedDphiChanges);
}