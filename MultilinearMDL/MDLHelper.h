/*************************************************************************************************************************/
// This source is provided for NON-COMMERCIAL RESEARCH PURPOSES only, and is provided “as is” WITHOUT ANY WARRANTY; 
// without even the implied warranty of fitness for a particular purpose. The redistribution of the code is not permitted.
//
// If you use the source or part of it in a publication, cite the following paper:
// 
// T. Bolkart, S. Wuhrer
// A Groupwise Multilinear Correspondence Optimization for 3D Faces
// International Conference on Computer Vision (ICCV), 2015
//
// Copyright (c) 2015 Timo Bolkart, Stefanie Wuhrer
/*************************************************************************************************************************/

#ifndef MDLHELPER_H
#define MDLHELPER_H

#include "DataContainer.h"
#include <stdlib.h>

#include <vnl/vnl_cost_function.h>
#include <vnl/algo/vnl_lbfgsb.h>

#include <vector>
#include <string>

class MDLHelper
{
public:
	static bool computeThinPlateSplines(const std::string& sstrFileCollectionName, const std::string& sstrTextureCoordsFileName, const std::string& sstrOutFolder, const std::string& sstrTpsFCFileName);

private:
	static void updateShapeParameter(const size_t iShape, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation);

	static void updateParameter(const size_t numSamples, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation);

	static void updateShapeData(const size_t iShape, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec
										, const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data);

	static void updateData(std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
								  ,const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data);

	static void computeShapeExcludedMean(const std::vector<double>& data, const size_t shapeDim, const size_t iExcludedShape, std::vector<double>& excludedShapeMean);

	static bool alignShapeData(const size_t iShape, const std::vector<double>& target, std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs);

	static bool procrustesAlignShapeData(std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, const size_t numIter);

public:
	static void outputParameterVariations(const size_t numSamples, const size_t numVertices, const std::vector<double>& parameterVariation, const std::string& sstrOutFolder, const std::string& sstrFileName);

	static void outputShapeData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames, const size_t iSample);

	static void outputData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames);

	static void precomputeBiLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights);

	static void precomputeLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights);

	static void precomputeHybridSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights);

	static void optimizeShapeWise(const std::string& sstrFileCollectionName, const std::string& sstrTpsFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string& sstrLmkFileCollectionName 
											, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string& sstrOutFolder);

private:
	static bool loadMDLData(const std::string& sstrFileCollectionName, const std::string& sstrTpsFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string& sstrLmkFileCollectionName
									, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName
									, std::vector<double>& data, DataContainer& mesh, std::vector<std::string>& geometryFileNames
									, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec, std::vector<std::string>& tpsFileNames
									, size_t& numIdentities, size_t& numExpressions
									, std::vector<size_t>& lmkIndices, std::vector<double>& lmks
									, std::vector<size_t>& outerBoundaryVertexIDs, std::vector<size_t>& innerBoundaryVertexIDs);

	static bool computeOptimizationBounds(const std::vector<double>& initialParam, const size_t numSamples, const std::vector<size_t>& outerBoundaryVertexIDs, const std::vector<size_t>& innerBoundaryVertexIDs
													, vnl_vector<long>& boundSelection, std::vector<vnl_vector<double>>& lowerShapeBounds, std::vector<vnl_vector<double>>& upperShapeBounds);

	static void getBoundaryNeighbors(const DataContainer& mesh, const std::vector<size_t>& boundary, std::vector<size_t>& neighbors);

	static void outputConfig(const std::string& sstrOutFolder, const bool bGroupWise);

public:
	static void outputCompactness(const std::vector<double>& data, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName);
	static void outputCompactness(const std::vector<double>& data, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName, double& sum);

private:
	static std::string getConfigOutFolder(const std::string& sstrOutFolder);
};

#endif