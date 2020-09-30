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

#include "MDLHelper.h"
#include "MDLShapeCostFunction.h"
#include "MDLShapeCostFunctionPCA.h"
#include "FileLoader.h"
#include "FileWriter.h"
#include "MathHelper.h"

#include <iostream>
#include <fstream>

#include <vector>
#include <set>
#include <map>

#define OUTPUT_INITIAL_ALIGNED_DATA
//#define OUTPUT_ITERATION_AVG_POINT_DIST
#define OUTPUT_TIME
#define OUTPUT_COMPACTNESS
#define OUTPUT_ITERATION_RESULTS
//#define OUTPUT_MEAN_FACE

//If disabled, Laplacian smoothness is used instead
#define USE_BI_LAPLACIAN_SMOOTHNESS

//If enabled, the PCA optimization gradient is used (w_EXP = 0)
//#define USE_PCA_OPTIMIZATION

#ifdef OUTPUT_COMPACTNESS
#include "MultilinearModel.h"
#endif

#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

#include <iomanip>  

//Weight of identity and expression compactness energy
const double IDENTITY_WEIGHT = 1.0;
#ifndef USE_PCA_OPTIMIZATION
const double EXPRESSION_WEIGHT = 1.0;
#else
const double EXPRESSION_WEIGHT = 0.0; 
#endif

//Weight of the regularization energy 
//(we choose 0.5 for the large databases in our submission, for much smaller datasets we recommend using larger values)
const double SMOOTHNESS_WEIGHT = 20.0;

const double LMK_WEIGHT = 0.0;
const double MAX_PARAMETER_VARIATION = 1.0;

//Number of iterations where one iteration means optimizing over each shapes once
const size_t NUM_ITERATION = 15;
//Number of function evaluations during optimization of one shape
const size_t NUM_FKT_EVAL = 50;
//Number of function evaluations before re-computing the shape alignment (should be fraction of NUM_FKT_EVAL)
const size_t NUM_NUM_FKT_EVAL_ALIGNMENT = 10;

const double OPTIMIZATION_DOMAIN_MIN = -0.2;
const double OPTIMIZATION_DOMAIN_MAX = 1.0-OPTIMIZATION_DOMAIN_MIN;

//Boundary vertices are allowed to vary within [p-delta, p+delta]
//If value is zero, boundary vertex is fixed during optimization
const double MAX_OUTER_BOUNDARY_VARIATION = 0.1; // ~19 mm
const double MAX_INNER_BOUNDARY_VARIATION = 0.0;

bool MDLHelper::computeThinPlateSplines(const std::string& sstrFileCollectionName, const std::string& sstrTextureCoordsFileName, const std::string& sstrOutFolder, const std::string& sstrTpsFCFileName)
{
	std::cout << "outputThinPlateSplines start" << std::endl;

	//Load meshes
	FileLoader loader;

	std::vector<double> data;
	DataContainer mesh;
	std::vector<std::string> fileNames;
	size_t numExpressions(0);
	size_t numIdentities(0);

	if(!loader.loadFileCollection(sstrFileCollectionName, data, mesh, fileNames, numExpressions, numIdentities))
	{
		std::cout << "Unable to load file collection " << sstrFileCollectionName << std::endl;
		return false;
	}

	const size_t dim = data.size()/(numExpressions*numIdentities);

	//Load texture coordinates
	std::vector<double> textureCoords;
	if(!loader.loadDataFile(sstrTextureCoordsFileName, textureCoords))
	{
		std::cout << "Unable to load texture coordinates " << sstrTextureCoordsFileName << std::endl;
		return false;
	}

	if(textureCoords.size() != 2*(dim/3))
	{
		std::cout << "Texture coordinates dimension does not fit " << textureCoords.size() << " != " << 2*(dim/3) << std::endl;
		return false;
	}

	const int numFiles = static_cast<int>(numIdentities*numExpressions);

	//Output thin-plate spline files
	std::vector<std::string> tpsFileNames;
	tpsFileNames.resize(numFiles);

#pragma omp parallel for
	for(int iFile = 0; iFile < numFiles; ++iFile)
	{
		std::cout << "File " << iFile+1 << " / " << numFiles << std::endl;

		const size_t startIndex = iFile*dim;

		std::vector<double> currData;
		currData.resize(dim, 0.0);
		
		for(size_t i = 0; i < dim; ++i)
		{
			currData[i] = data[startIndex+i];
		}

		std::vector<double> vecC; 
		std::vector<double> matA;
		std::vector<double> matW;
		if(!MathHelper::computeInterpolationBasis(textureCoords, 2, currData, 3, vecC, matA, matW))
		{
			std::cout << "Unable to compute thin-plate spline" << std::endl;
			continue;
		}

		const std::string sstrGeometryFileName = fileNames[iFile];
		const std::string sstrOutFileName = FileLoader::getFileName(sstrGeometryFileName) + ".tps";
		tpsFileNames[iFile] = sstrOutFileName;

		std::cout << "sstrGeometryFileName " << sstrGeometryFileName << std::endl;
		std::cout << "sstrOutFileName " << sstrOutFileName << std::endl;

		if(!FileWriter::saveThinPlateSpline(sstrOutFolder + "/" + sstrOutFileName, vecC, matA, matW, textureCoords))
		{
			std::cout << "Unable to save thin-plate spline " << sstrOutFileName << std::endl;
			continue;
		}
	}

	if(tpsFileNames.size() != numFiles)
	{
		std::cout << "Wrong number of thin-plate spline files" << std::endl;
		return false;
	}

	//Output thin-plate splines file collection
	const std::string sstrOutFileName = sstrTpsFCFileName;
	std::fstream output(sstrOutFileName, std::ios::out);	

	output << numExpressions << "#Expressions" << std::endl;
	output << numIdentities << "#Identities" << std::endl;
	for(size_t iFile = 0; iFile < numFiles; ++iFile)
	{
		output << tpsFileNames[iFile] << std::endl;
	}

	output.close();

	std::cout << "outputThinPlateSplines end" << std::endl;
	return true;
}

void MDLHelper::updateShapeParameter(const size_t iShape, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation)
{
	bool bSingleShapeParameters = (x.size() == 2*numVertices);

	const size_t paramOffset = iShape*2*numVertices;
	const size_t paramXOffset = bSingleShapeParameters ? 0 : paramOffset;

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpParamStartIndex = 2*iVertex;

		double v1 = x[paramXOffset+tmpParamStartIndex];
		double v2 = x[paramXOffset+tmpParamStartIndex+1];
		const double tmpLength = sqrt(std::pow(v1,2) + std::pow(v2, 2));
		if(tmpLength > maxParamVariation)
		{
			const double factor = (maxParamVariation/tmpLength);
			v1 *= factor;
			v2 *= factor;
		}

		paramVariation[paramOffset+tmpParamStartIndex] = v1;
		paramVariation[paramOffset+tmpParamStartIndex+1] = v2;
	}
}

void MDLHelper::updateParameter(const size_t numSamples, const size_t numVertices, const double maxParamVariation, const vnl_vector<double>& x, std::vector<double>& paramVariation)
{
	if(paramVariation.size() != 2*numSamples*numVertices)
	{
		std::cout << "Unable to update parameter " << paramVariation.size() << " != " << 2*numSamples*numVertices << std::endl;
	}

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		updateShapeParameter(iSample, numVertices, maxParamVariation, x, paramVariation);
	}
}

void MDLHelper::updateShapeData(const size_t iShape, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec
										, const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data)
{
	const size_t numSamples = vecCs.size();
	const size_t numVertices = data.size()/(3*numSamples);

	const size_t dataOffset = iShape*3*numVertices;
	const size_t paramOffset = iShape*2*numVertices;

	const std::vector<double>& currVecC = vecCs[iShape]; 
	const std::vector<double>& currMatA = matAs[iShape];
	const std::vector<double>& currMatW = matWs[iShape];
	const std::vector<double>& currSourcePoints = sourcePointsVec[iShape];

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		const size_t tmpParamStartIndex = 2*iVertex;

		std::vector<double> paramVertex;
		paramVertex.resize(2, 0.0);

		paramVertex[0] = initialParametrization[paramOffset+tmpParamStartIndex]+paramVariation[paramOffset+tmpParamStartIndex];
		paramVertex[1] = initialParametrization[paramOffset+tmpParamStartIndex+1]+paramVariation[paramOffset+tmpParamStartIndex+1];

		std::vector<double> outDataVertex;
		if(!MathHelper::evaluateInterpolation(currVecC, currMatA, currMatW, currSourcePoints, paramVertex, outDataVertex))
		{
			std::cout << "MDLHelper::updateShapeData(...) - unable to compute tps point" << std::endl;
			return;
		}

		data[dataOffset+tmpDataStartIndex] = outDataVertex[0];
		data[dataOffset+tmpDataStartIndex+1] = outDataVertex[1];
		data[dataOffset+tmpDataStartIndex+2] = outDataVertex[2];
	}
}

void MDLHelper::updateData(std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
								  , const std::vector<double>& initialParametrization, const std::vector<double>& paramVariation, std::vector<double>& data)
{
	const size_t numSamples = vecCs.size();
	const size_t numVertices = data.size()/(3*numSamples);
	
	if(data.size() != 3*numSamples*numVertices)
	{
		std::cout << "Unable to update data " << data.size() << " != " << 3*numSamples*numVertices << std::endl;
		return;
	}

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		MDLHelper::updateShapeData(iSample, vecCs, matAs, matWs, sourcePointsVec, initialParametrization, paramVariation, data);
	}
}

void MDLHelper::computeShapeExcludedMean(const std::vector<double>& data, const size_t shapeDim, const size_t iExcludedShape, std::vector<double>& excludedShapeMean)
{
	excludedShapeMean.clear();
	excludedShapeMean.resize(shapeDim, 0.0);

	const size_t numSamples = data.size()/shapeDim;
	if(numSamples == 0)
	{
		return;
	}
	else if(numSamples == 1)
	{
		excludedShapeMean = data;
		return;
	}

	for(size_t i = 0; i < numSamples; ++i)
	{
		if(i == iExcludedShape)
		{
			continue;
		}

		const size_t startIndex = i*shapeDim;

		for(size_t j = 0; j < shapeDim; ++j)
		{
			const size_t currIndex = startIndex+j;
			excludedShapeMean[j] += data[currIndex];
		}
	}

	const double factor = 1.0 / static_cast<double>(numSamples-1);
	for(size_t i = 0; i < shapeDim; ++i)
	{
		excludedShapeMean[i] *= factor;
	}
}

bool MDLHelper::alignShapeData(const size_t iShape, const std::vector<double>& target, std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs)
{
	const size_t shapeDim = target.size();
	const size_t shapeOffset = iShape*shapeDim;

	std::vector<double> shapeData;
	shapeData.resize(shapeDim, 0.0);

	for(size_t j = 0; j < shapeDim; ++j)	
	{
		const size_t currIndex = shapeOffset+j;
		shapeData[j] = data[currIndex];
	}

	double s(0.0);
	std::vector<double> R; 
	std::vector<double> t;
	if(!MathHelper::computeAlignmentTrafo(shapeData, target, s, R, t, false))
	{
		std::cout << "Unable to compute alignment of shape " << iShape << std::endl;
		return false;
	}

	MathHelper::transformData(s, R, "N", t, "+", shapeData);
	MathHelper::transformThinPlateSpline(s, R, "N", t, "+", vecCs[iShape], matAs[iShape], matWs[iShape]);

	for(size_t j = 0; j < shapeDim; ++j)
	{
		const size_t currIndex = shapeOffset+j;
		data[currIndex] = shapeData[j];
	}

	return true;
}

bool MDLHelper::procrustesAlignShapeData(std::vector<double>& data, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, const size_t numIter)
{
	const size_t numShapes = vecCs.size();
	const size_t dataDim = data.size()/numShapes;

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Compute mean
		std::vector<double> procrustesMean;
		MathHelper::computeMean(data, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		for(size_t iShape = 0; iShape < numShapes; ++iShape)
		{
			if(!MDLHelper::alignShapeData(iShape, procrustesMean, data, vecCs, matAs, matWs))
			{
				std::cout << "Unable to align shape data " << iShape << std::endl;
				return false;
			}
		}
	}

	return true;
}

void MDLHelper::outputParameterVariations(const size_t numSamples, const size_t numVertices, const std::vector<double>& parameterVariation, const std::string& sstrOutFolder, const std::string& sstrFileName)
{
	std::fstream outVariationStream(sstrOutFolder + "/" + sstrFileName, std::ios::out);

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		const size_t paramOffset = iSample*2*numVertices;

		for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
		{
			const size_t tmpParamStartIndex = 2*iVertex;
			outVariationStream << parameterVariation[paramOffset+tmpParamStartIndex] << " " << parameterVariation[paramOffset+tmpParamStartIndex+1] << "   ";
		}

		outVariationStream << std::endl;
	}

	outVariationStream.close();
}

void MDLHelper::outputShapeData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames, const size_t iSample)
{
	const size_t numVertices = mesh.getNumVertices();
	const size_t numSamples = data.size()/(3*numVertices);
	if(fileNames.size() != numSamples)
	{
		std::cout << "Unable to output data " << fileNames.size() << " != " << numSamples << std::endl;
		return;
	}

	DataContainer currMesh = mesh;
	std::vector<double> vertices = currMesh.getVertexList();

	const size_t dataOffset = iSample*3*numVertices;

	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		vertices[tmpDataStartIndex] = data[dataOffset+tmpDataStartIndex];
		vertices[tmpDataStartIndex+1] = data[dataOffset+tmpDataStartIndex+1];
		vertices[tmpDataStartIndex+2] = data[dataOffset+tmpDataStartIndex+2];
	}

	currMesh.setVertexList(vertices);

	const std::string sstrCurrOutFileName = sstrOutFolder + "/" + fileNames[iSample];
	FileWriter::saveFile(sstrCurrOutFileName, currMesh);
}

void MDLHelper::outputData(const DataContainer& mesh, const std::vector<double>& data, const std::string& sstrOutFolder, const std::vector<std::string>& fileNames)
{
	const size_t numVertices = mesh.getNumVertices();
	const size_t numSamples = data.size()/(3*numVertices);
	if(fileNames.size() != numSamples)
	{
		std::cout << "Unable to output data " << fileNames.size() << " != " << numSamples << std::endl;
		return;
	}

	//TODO Run in parallel
	//for(size_t iSample = 0; iSample < numSamples; ++iSample)
#pragma omp parallel for
	for(int iSample = 0; iSample < numSamples; ++iSample)
	{
		outputShapeData(mesh, data, sstrOutFolder, fileNames, iSample);
	}

#ifdef OUTPUT_MEAN_FACE
	std::vector<double> mean;
	MathHelper::computeMean(data, 3*numVertices, mean);
	
	DataContainer currMesh = mesh;
	currMesh.setVertexList(mean);

	FileWriter::saveFile(sstrOutFolder + "/MeanFace.wrl", currMesh);
#endif

}

void MDLHelper::precomputeBiLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights)
{
	//Compute for each vertex all its neighbors
	const size_t numVertices = mesh.getNumVertices();
	precomputedSmoothnessIndices.resize(numVertices);
	precomputedSmoothnessWeights.resize(numVertices);

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	//Remove points that are in the 2-ring neighborhood of the boundary
	std::vector<bool> invalidSmoothNeighbors;
	invalidSmoothNeighbors.resize(numVertices, false);

	std::vector<size_t> boundaryVertices;
	MathHelper::computeBoundaryVertices(mesh, boundaryVertices);

	const size_t numBoundaryVertices = boundaryVertices.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currVertex = boundaryVertices[i];
		invalidSmoothNeighbors[currVertex] = true;

		const std::set<int>& borderNeighbors = vertexNeighbors[currVertex];
		std::set<int>::const_iterator currNeighborIter = borderNeighbors.begin();
		const std::set<int>::const_iterator endNeighborIter = borderNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			//Boundary points in the 1-ring neighborhood.
			const size_t borderNeighborID = *currNeighborIter;
			invalidSmoothNeighbors[borderNeighborID] = true;

			//const std::set<int>& border2RingNeighbors = vertexNeighbors[borderNeighborID];
			//std::set<int>::const_iterator curr2RingNeighborIter = border2RingNeighbors.begin();
			//const std::set<int>::const_iterator end2RingNeighborIter = border2RingNeighbors.end();
			//for(; curr2RingNeighborIter != end2RingNeighborIter; ++curr2RingNeighborIter)
			//{
			//	//Boundary points in the 2-ring neighborhood.
			//	const size_t border2RingNeighborID = *curr2RingNeighborIter;
			//	invalidSmoothNeighbors[border2RingNeighborID] = true;
			//}
		}
	}

	//Compute weights for all vertices
	for(int iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		if(invalidSmoothNeighbors[iVertex])
		{
			continue;
		}

		const std::set<int>& currNeighbors = vertexNeighbors[iVertex];
		if(currNeighbors.empty())
		{
			continue;
		}

		std::map<int, double> indexWeightMap; 
		indexWeightMap.insert(std::make_pair(iVertex, 1.0));
		
		std::map<int, double>::iterator mapIter;

		const double neighborFactor = -2.0/static_cast<double>(currNeighbors.size());

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const int currNeighborIndex = *currNeighborIter;

			mapIter = indexWeightMap.find(currNeighborIndex);
			if(mapIter != indexWeightMap.end())
			{
				mapIter->second += neighborFactor;
			}
			else
			{
				indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
			}

			//Iterate over all neighbor neighbors
			const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
			if(neighborNeighbors.empty())
			{
				continue;
			}

			const double neighborNeighborFactor = 1.0/static_cast<double>(currNeighbors.size()*neighborNeighbors.size());

			std::set<int>::const_iterator neighborNeighborIter = neighborNeighbors.begin();
			std::set<int>::const_iterator endNeighborNeighborIter = neighborNeighbors.end();
			for(; neighborNeighborIter != endNeighborNeighborIter; ++neighborNeighborIter)
			{
				const int neighborNeighborIndex = *neighborNeighborIter;

				mapIter = indexWeightMap.find(neighborNeighborIndex);
				if(mapIter != indexWeightMap.end())
				{
					mapIter->second += neighborNeighborFactor;
				}
				else
				{
					indexWeightMap.insert(std::make_pair(neighborNeighborIndex, neighborNeighborFactor));
				}
			}
		}

		const size_t numElements = indexWeightMap.size();
		std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		currSmoothnessIndices.reserve(numElements);

		std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];
		currSmoothnessWeights.reserve(numElements);

		for(mapIter=indexWeightMap.begin(); mapIter != indexWeightMap.end(); ++mapIter)
		{
			currSmoothnessIndices.push_back(mapIter->first);
			currSmoothnessWeights.push_back(mapIter->second);
		}
	}
}

void MDLHelper::precomputeLaplacianSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights)
{
	//Compute for each vertex all its neighbors
	const size_t numVertices = mesh.getNumVertices();
	precomputedSmoothnessIndices.resize(numVertices);
	precomputedSmoothnessWeights.resize(numVertices);

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	//Remove points that are in the 2-ring neighborhood of the boundary
	std::vector<bool> invalidSmoothNeighbors;
	invalidSmoothNeighbors.resize(numVertices, false);

	std::vector<size_t> boundaryVertices;
	MathHelper::computeBoundaryVertices(mesh, boundaryVertices);

	const size_t numBoundaryVertices = boundaryVertices.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currVertex = boundaryVertices[i];
		invalidSmoothNeighbors[currVertex] = true;

		//const std::set<int>& borderNeighbors = vertexNeighbors[currVertex];
		//std::set<int>::const_iterator currNeighborIter = borderNeighbors.begin();
		//const std::set<int>::const_iterator endNeighborIter = borderNeighbors.end();
		//for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		//{
		//	//Boundary points in the 1-ring neighborhood.
		//	const size_t borderNeighborID = *currNeighborIter;
		//	invalidSmoothNeighbors[borderNeighborID] = true;
		//}
	}

	//Compute weights for all vertices
	for(int iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		if(invalidSmoothNeighbors[iVertex])
		{
			continue;
		}

		const std::set<int>& currNeighbors = vertexNeighbors[iVertex];
		if(currNeighbors.empty())
		{
			continue;
		}

		std::map<int, double> indexWeightMap; 
		indexWeightMap.insert(std::make_pair(iVertex, -1.0));
		
		std::map<int, double>::iterator mapIter;

		const double neighborFactor = 1.0/static_cast<double>(currNeighbors.size());

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const int currNeighborIndex = *currNeighborIter;

			mapIter = indexWeightMap.find(currNeighborIndex);
			if(mapIter != indexWeightMap.end())
			{
				mapIter->second += neighborFactor;
			}
			else
			{
				indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
			}

			//Iterate over all neighbor neighbors
			const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
			if(neighborNeighbors.empty())
			{
				continue;
			}
		}

		const size_t numElements = indexWeightMap.size();
		std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		currSmoothnessIndices.reserve(numElements);

		std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];
		currSmoothnessWeights.reserve(numElements);

		for(mapIter=indexWeightMap.begin(); mapIter != indexWeightMap.end(); ++mapIter)
		{
			currSmoothnessIndices.push_back(mapIter->first);
			currSmoothnessWeights.push_back(mapIter->second);
		}
	}
}

void MDLHelper::precomputeHybridSmoothnessWeights(const DataContainer& mesh, std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, std::vector<std::vector<double>>& precomputedSmoothnessWeights)
{
	//Compute for each vertex all its neighbors
	const size_t numVertices = mesh.getNumVertices();
	precomputedSmoothnessIndices.resize(numVertices);
	precomputedSmoothnessWeights.resize(numVertices);

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	//Remove points that are in the 2-ring neighborhood of the boundary
	std::vector<bool> invalidBoundaryVertices;
	invalidBoundaryVertices.resize(numVertices, false);

	std::vector<bool> laplaceVertices;
	laplaceVertices.resize(numVertices, false);

	std::vector<size_t> boundaryVertices;
	MathHelper::computeBoundaryVertices(mesh, boundaryVertices);

	const size_t numBoundaryVertices = boundaryVertices.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currVertex = boundaryVertices[i];
		invalidBoundaryVertices[currVertex] = true;

		const std::set<int>& borderNeighbors = vertexNeighbors[currVertex];
		std::set<int>::const_iterator currNeighborIter = borderNeighbors.begin();
		const std::set<int>::const_iterator endNeighborIter = borderNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			//Boundary points in the 1-ring neighborhood.
			const size_t borderNeighborID = *currNeighborIter;
			laplaceVertices[borderNeighborID] = true;
		}
	}

	//Compute weights for all vertices
	for(int iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		if(invalidBoundaryVertices[iVertex])
		{
			continue;
		}

		const std::set<int>& currNeighbors = vertexNeighbors[iVertex];
		if(currNeighbors.empty())
		{
			continue;
		}

		std::map<int, double> indexWeightMap; 
		std::map<int, double>::iterator mapIter;

		if(laplaceVertices[iVertex])
		{
			indexWeightMap.insert(std::make_pair(iVertex, -1.0));
			const double neighborFactor = 1.0/static_cast<double>(currNeighbors.size());

			//Iterate over all neighbors
			std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
			std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
			for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
			{
				const int currNeighborIndex = *currNeighborIter;

				mapIter = indexWeightMap.find(currNeighborIndex);
				if(mapIter != indexWeightMap.end())
				{
					mapIter->second += neighborFactor;
				}
				else
				{
					indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
				}

				//Iterate over all neighbor neighbors
				const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
				if(neighborNeighbors.empty())
				{
					continue;
				}
			}
		}
		else
		{
			indexWeightMap.insert(std::make_pair(iVertex, 1.0));

			const double neighborFactor = -2.0/static_cast<double>(currNeighbors.size());

			//Iterate over all neighbors
			std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
			std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
			for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
			{
				const int currNeighborIndex = *currNeighborIter;

				mapIter = indexWeightMap.find(currNeighborIndex);
				if(mapIter != indexWeightMap.end())
				{
					mapIter->second += neighborFactor;
				}
				else
				{
					indexWeightMap.insert(std::make_pair(currNeighborIndex, neighborFactor));
				}

				//Iterate over all neighbor neighbors
				const std::set<int>& neighborNeighbors = vertexNeighbors[currNeighborIndex];
				if(neighborNeighbors.empty())
				{
					continue;
				}

				const double neighborNeighborFactor = 1.0/static_cast<double>(currNeighbors.size()*neighborNeighbors.size());

				std::set<int>::const_iterator neighborNeighborIter = neighborNeighbors.begin();
				std::set<int>::const_iterator endNeighborNeighborIter = neighborNeighbors.end();
				for(; neighborNeighborIter != endNeighborNeighborIter; ++neighborNeighborIter)
				{
					const int neighborNeighborIndex = *neighborNeighborIter;

					mapIter = indexWeightMap.find(neighborNeighborIndex);
					if(mapIter != indexWeightMap.end())
					{
						mapIter->second += neighborNeighborFactor;
					}
					else
					{
						indexWeightMap.insert(std::make_pair(neighborNeighborIndex, neighborNeighborFactor));
					}
				}
			}
		}

		const size_t numElements = indexWeightMap.size();
		std::vector<size_t>& currSmoothnessIndices = precomputedSmoothnessIndices[iVertex];
		currSmoothnessIndices.reserve(numElements);

		std::vector<double>& currSmoothnessWeights = precomputedSmoothnessWeights[iVertex];
		currSmoothnessWeights.reserve(numElements);

		for(mapIter=indexWeightMap.begin(); mapIter != indexWeightMap.end(); ++mapIter)
		{
			currSmoothnessIndices.push_back(mapIter->first);
			currSmoothnessWeights.push_back(mapIter->second);
		}
	}

}

void MDLHelper::optimizeShapeWise(const std::string& sstrFileCollectionName, const std::string& sstrTpsFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string& sstrLmkFileCollectionName
											, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName, const std::string& sstrOutFolder)
{
	const std::string sstrCurrOutFolder = MDLHelper::getConfigOutFolder(sstrOutFolder);
	//CreateDirectoryA(sstrCurrOutFolder.c_str(), NULL);
	FileWriter::makeDirectory(sstrCurrOutFolder);

	//MDLHelper::outputConfig(sstrOutFolder, false);
	MDLHelper::outputConfig(sstrCurrOutFolder, false);
	
	std::vector<double> data;
	DataContainer mesh;
	std::vector<std::string> geometryFileNames;

	std::vector<std::vector<double>> vecCs;
	std::vector<std::vector<double>> matAs; 
	std::vector<std::vector<double>> matWs;
	std::vector<std::vector<double>> sourcePointsVec;
	std::vector<std::string> tpsFileNames;
	
	size_t numExpressions(0);
	size_t numIdentities(0);

	std::vector<size_t> lmkIndices;
	std::vector<double> lmks;

	std::vector<size_t> outerBoundaryVertexIDs;
	std::vector<size_t> innerBoundaryVertexIDs;

	if(!loadMDLData(sstrFileCollectionName, sstrTpsFileCollectionName, sstrLmkIndexFileName, sstrLmkFileCollectionName, sstrOuterBoundaryIndexFileName, sstrInnerBoundaryIndexFileName
									, data, mesh, geometryFileNames, vecCs, matAs, matWs, sourcePointsVec, tpsFileNames, numIdentities, numExpressions, lmkIndices, lmks, outerBoundaryVertexIDs, innerBoundaryVertexIDs))
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Unable to load required mdl data" << std::endl;
#endif
		return;
	}

	const size_t numSamples = numExpressions*numIdentities;
	const size_t numSampleVertices = data.size()/(3*numSamples);
	const size_t sampleDataDim = 3*numSampleVertices;
	const size_t sampleParameterDim = 2*numSampleVertices;
	
	std::vector<double> initialParam;
	initialParam.resize(numSamples*sampleParameterDim, 0.0);

	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		const size_t sampleParamOffset = iSample*sampleParameterDim;

		const std::vector<double>& currSourcePoints = sourcePointsVec[iSample];
		for(size_t i = 0; i < sampleParameterDim; ++i)
		{
			initialParam[sampleParamOffset+i] = currSourcePoints[i];
		}
	}

/**/
	//Compute 1-ring neihbors of inner boundary and fix these vertices
	std::vector<size_t> innerBoundaryNeighborIDs;
	getBoundaryNeighbors(mesh, innerBoundaryVertexIDs, innerBoundaryNeighborIDs);

	innerBoundaryVertexIDs = innerBoundaryNeighborIDs;
/**/

/**/
	//Compute 1-ring neihbors of outer boundary and set threshold for these vertices
	std::vector<size_t> outerBoundaryNeighborIDs;
	getBoundaryNeighbors(mesh, outerBoundaryVertexIDs, outerBoundaryNeighborIDs);

	outerBoundaryVertexIDs = outerBoundaryNeighborIDs;
/**/

	//Compute Procrustes alignment for the initial shapes
	if(!MDLHelper::procrustesAlignShapeData(data, vecCs, matAs, matWs, 10))
	{
		std::cout << "Unable to compute procrustes shape alignment" << std::endl;
		return;
	}

#ifdef OUTPUT_INITIAL_ALIGNED_DATA
	const std::string sstrInitialOutFolder = sstrCurrOutFolder + "/Initial";
	FileWriter::makeDirectory(sstrInitialOutFolder);
	
	MDLHelper::outputData(mesh, data, sstrInitialOutFolder, geometryFileNames);
#endif

	const size_t dataDim = data.size()/(numExpressions*numIdentities);
	std::vector<double> mean;
	MathHelper::computeMean(data, dataDim, mean);
	mesh.setVertexList(mean);

	std::vector<double> vertexAreas;
	double meanShapeArea(1.0);
	vertexAreas.resize(data.size()/(3*numExpressions*numIdentities), 1.0);
	//MathHelper::computeVertexAreas(mesh, vertexAreas, meanShapeArea);

	std::vector<std::vector<size_t>> precomputedSmoothnessIndices;
	std::vector<std::vector<double>> precomputedSmoothnessWeights;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	precomputeBiLaplacianSmoothnessWeights(mesh, precomputedSmoothnessIndices, precomputedSmoothnessWeights);
#else
	precomputeLaplacianSmoothnessWeights(mesh, precomputedSmoothnessIndices, precomputedSmoothnessWeights);
#endif

	//Compute boundaries for bounded Quasi-Newton
	vnl_vector<long> boundSelection;
	std::vector<vnl_vector<double>> lowerShapeBounds;
	std::vector<vnl_vector<double>> upperShapeBounds;
	if(!computeOptimizationBounds(initialParam, numSamples, outerBoundaryVertexIDs, innerBoundaryVertexIDs, boundSelection, lowerShapeBounds, upperShapeBounds))
	{
		return;
	}

	std::vector<double> parameterVariation;
	parameterVariation.resize(numSamples*sampleParameterDim, 0.0);

#ifdef OUTPUT_ITERATION_AVG_POINT_DIST
	std::vector<double> tmpData = data;
#endif

#ifdef OUTPUT_COMPACTNESS
	{
		const size_t d1 = 3*mesh.getNumVertices();
		const size_t d2 = numIdentities;
		const size_t d3 = numExpressions;
		double sum(0.0);
		MDLHelper::outputCompactness(data, d1, d2, d3, sstrCurrOutFolder + "/CompactnessStart.txt", sum);

		std::fstream outStream(sstrCurrOutFolder + "/CompactnessSum.txt", std::ios::out);
		outStream << "0  " << sum << std::endl;
		outStream.close();
	}
#endif

#ifdef OUTPUT_GRADIENT_NORM
	std::fstream outGradientStream(sstrCurrOutFolder + "/GradientNorm.txt", std::ios::out);
	outGradientStream.close();
#endif

#ifdef OUTPUT_TIME
	const double t1 = PerformanceCounter::getTime();

	std::fstream outTime(sstrCurrOutFolder + "/Time.txt", std::ios::out);
	outTime.close();
#endif

	const size_t numProcessedSamples = numSamples;

	for(size_t iter = 0; iter < NUM_ITERATION; ++iter)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "*******************************" << std::endl;
		std::cout << "*******************************" << std::endl;
		std::cout << "Iteration " << iter+1 << std::endl;
		std::cout << "*******************************" << std::endl;
		std::cout << "*******************************" << std::endl;
#endif

#ifdef OUTPUT_TIME
	const double iterStart = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_ITERATION_RESULTS
		std::stringstream out;
		out << iter+1;
		std::string sstrIter = "Iter_" + out.str();

		const std::string sstrIterOutFolder = sstrCurrOutFolder + "/" + sstrIter;
		FileWriter::makeDirectory(sstrIterOutFolder);
#endif

		std::vector<int> permutedShapeIndices;
		MathHelper::getRandomlyPermutedInteger(0, numSamples-1, permutedShapeIndices);

		for(size_t i = 0; i < numProcessedSamples; ++i)
		{
#ifdef OUTPUT_TIME
			const double tmpt1 = PerformanceCounter::getTime();
#endif

			const size_t iShape = permutedShapeIndices[i];

#ifdef DEBUG_OUTPUT
			std::cout << std::endl;
			std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
			std::cout << "Optimization of shape " << iShape << " (" << i+1 << " of " << numSamples << ")" << std::endl;
			std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
			std::cout << std::endl;
#endif

			//Compute mean for all faces except shape iShape
			std::vector<double> excludedShapeMean;
			computeShapeExcludedMean(data, sampleDataDim, iShape, excludedShapeMean);

			const size_t numSubIter = NUM_FKT_EVAL/NUM_NUM_FKT_EVAL_ALIGNMENT;
			for(size_t subIter = 0; subIter < numSubIter; ++subIter)
			{
#ifndef USE_PCA_OPTIMIZATION
				MDLShapeCostFunction fkt(data, initialParam, numIdentities, numExpressions, vecCs, matAs, matWs, sourcePointsVec, vertexAreas, meanShapeArea, IDENTITY_WEIGHT, EXPRESSION_WEIGHT, MAX_PARAMETER_VARIATION, iShape);
#else
				MDLShapeCostFunctionPCA fkt(data, initialParam, numIdentities, numExpressions, vecCs, matAs, matWs, sourcePointsVec, vertexAreas, meanShapeArea, IDENTITY_WEIGHT, EXPRESSION_WEIGHT, MAX_PARAMETER_VARIATION, iShape);
#endif
				fkt.setLandmarks(lmkIndices, lmks, LMK_WEIGHT);
				fkt.setSmoothnessValues(precomputedSmoothnessIndices, precomputedSmoothnessWeights, SMOOTHNESS_WEIGHT);

				vnl_lbfgsb minimizer(fkt);
				minimizer.set_cost_function_convergence_factor(10000000); 
				minimizer.set_projected_gradient_tolerance(0.00001);		
				minimizer.set_max_function_evals(NUM_FKT_EVAL/numSubIter);

				minimizer.set_bound_selection(boundSelection);
				minimizer.set_lower_bound(lowerShapeBounds[iShape]);
				minimizer.set_upper_bound(upperShapeBounds[iShape]);

#ifdef DEBUG_OUTPUT
				minimizer.set_trace(true);
#endif

				const size_t paramOffset = iShape*sampleParameterDim;

				vnl_vector<double> x(sampleParameterDim, 0.0);
				for(size_t j = 0; j < sampleParameterDim; ++j)
				{
					x[j] = parameterVariation[paramOffset+j];
				}		

				minimizer.minimize(x);

				if(minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_FTOL
					|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XTOL
					|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_XFTOL
					|| minimizer.get_failure_code() == vnl_lbfgsb::CONVERGED_GTOL)
				{
					MDLHelper::updateShapeParameter(iShape, numSampleVertices, MAX_PARAMETER_VARIATION, x, parameterVariation);
				}
				else if(minimizer.get_failure_code() == vnl_lbfgsb::FAILED_TOO_MANY_ITERATIONS)
				{
					std::cout << "Reached maximum number of function evaluations " << minimizer.get_failure_code() << std::endl;
					if(minimizer.obj_value_reduced())
					{
						std::cout << "Function value reduced" << std::endl;
						MDLHelper::updateShapeParameter(iShape, numSampleVertices, MAX_PARAMETER_VARIATION, x, parameterVariation);
					}
					else
					{
						std::cout << "Function value not reduced" << std::endl;
					}
				}
				else
				{
					std::cout << "Minimizer failed convergence " << minimizer.get_failure_code() << std::endl;
				}

#ifdef OUTPUT_GRADIENT_NORM
				const double gradientNorm = fkt.getGradientNorm();

				std::fstream outGradientStream(sstrCurrOutFolder + "/GradientNorm.txt", std::ios::app);
				outGradientStream << "Iter " << iter << " Shape " << iShape << " SubIter " << subIter << " |g| " << gradientNorm << std::endl;
				outGradientStream.close();
#endif

				MDLHelper::updateShapeData(iShape, vecCs, matAs, matWs, sourcePointsVec, initialParam, parameterVariation, data);

				//Compute rigid alignment of iShape to mean of all other shapes
				if(!MDLHelper::alignShapeData(iShape, excludedShapeMean, data, vecCs, matAs, matWs))
				{
					std::cout << "Failed to align shape after optimization" << std::endl;
					return;
				}
			}

#ifdef OUTPUT_TIME
			const double tmpt2 = PerformanceCounter::getTime();
			const double tmpDiff = tmpt2-tmpt1;
			std::cout << std::endl;
			std::cout << "Shape optimization time " << tmpDiff <<  "s" << std::endl;
			std::cout << "Shape optimization time " << tmpDiff/60.0 <<  "min" << std::endl;
			std::cout << std::endl;

			std::fstream outShapeTime(sstrCurrOutFolder + "/ShapeTime.txt", std::ios::app);
			outShapeTime << "Shape " <<  iShape << " (" << i+1 << " of " << numSamples << ")" << std::endl;
			outShapeTime << "Shape optimization time " << tmpDiff <<  "s" << std::endl;
			outShapeTime << "Shape optimization time " << tmpDiff/60.0 <<  "min" << std::endl;
			outShapeTime.close();
#endif

#ifdef OUTPUT_ITERATION_RESULTS
			outputShapeData( mesh, data, sstrIterOutFolder, geometryFileNames, iShape);
#endif

#ifdef OUTPUT_ITERATION_AVG_POINT_DIST
			const size_t numVertices = data.size()/(3*numSamples);

			double tmpDist(0.0);
			for(size_t iSample = 0; iSample < numSamples; ++iSample)
			{
				const size_t sampleParamOffset = iSample*sampleParameterDim;
				for(size_t j = 0; j < numVertices; ++j)
				{
					const size_t currVertexOffset = 3*j;
					const double diffX = data[sampleParamOffset+currVertexOffset] - tmpData[sampleParamOffset+currVertexOffset];
					const double diffY = data[sampleParamOffset+currVertexOffset+1] - tmpData[sampleParamOffset+currVertexOffset+1];
					const double diffZ = data[sampleParamOffset+currVertexOffset+2] - tmpData[sampleParamOffset+currVertexOffset+2];
					tmpDist += sqrt(std::pow(diffX, 2) + std::pow(diffY, 2) + std::pow(diffZ, 2));
				}
			}

			tmpDist /= static_cast<double>(numSamples*numVertices);
			std::cout << "Avg point distance: " << tmpDist << std::endl;
#endif
		}

#ifdef OUTPUT_ITERATION_RESULTS
		MDLHelper::outputParameterVariations(numSamples, numSampleVertices, parameterVariation, sstrIterOutFolder, "VariationIter.txt");

#ifdef OUTPUT_COMPACTNESS
		{
			const size_t d1 = 3*mesh.getNumVertices();
			const size_t d2 = numIdentities;
			const size_t d3 = numExpressions;
			double sum(0.0);
			MDLHelper::outputCompactness(data, d1, d2, d3, sstrIterOutFolder + "/CompactnessIter.txt", sum);

			std::fstream outStream(sstrCurrOutFolder + "/CompactnessSum.txt", std::ios::app);
			outStream << iter+1 << "  " << sum << std::endl;
			outStream.close();
		}
#endif		
#endif

#ifdef OUTPUT_TIME
		const double iterEnd = PerformanceCounter::getTime();
		const double diff = iterEnd-iterStart; 

		std::fstream outIterTime(sstrIterOutFolder + "/TimeIter.txt", std::ios::out);
		outIterTime << "Iteration time " << diff <<  "s" << std::endl;
		outIterTime << "Iteration time " << diff/60.0 <<  "min" << std::endl;
		outIterTime << "Iteration time " << diff/(3600.0) <<  "h" << std::endl;
		outIterTime.close();

		outTime.open(sstrCurrOutFolder + "/Time.txt", std::ios::app);
		outTime << "Iteration " << iter+1 << std::endl;
		outTime << "Iteration time " << diff <<  "s" << std::endl;
		outTime << "Iteration time " << diff/60.0 <<  "min" << std::endl;
		outTime << "Iteration time " << diff/(3600.0) <<  "h" << std::endl;
		outTime << std::endl;
		outTime.close();
#endif
	}

#ifdef OUTPUT_TIME
	const double t2 = PerformanceCounter::getTime();
	const double diff = t2-t1; 
	std::cout << std::endl;
	std::cout << "Overall time " << diff <<  "s" << std::endl;
	std::cout << "Overall time " << diff/60.0 <<  "min" << std::endl;
	std::cout << "Overall time " << diff/(3600.0) <<  "h" << std::endl;
	std::cout << std::endl;

	outTime.open(sstrCurrOutFolder + "/Time.txt", std::ios::app);
	outTime << "Overall time " << diff <<  "s" << std::endl;
	outTime << "Overall time " << diff/60.0 <<  "min" << std::endl;
	outTime << "Overall time " << diff/(3600.0) <<  "h" << std::endl;
	outTime.close();
#endif

#ifdef OUTPUT_COMPACTNESS
	const size_t d1 = 3*mesh.getNumVertices();
	const size_t d2 = numIdentities;
	const size_t d3 = numExpressions;
	MDLHelper::outputCompactness(data, d1, d2, d3, sstrCurrOutFolder + "/CompactnessEnd.txt");
#endif

	MDLHelper::outputParameterVariations(numSamples, numSampleVertices, parameterVariation, sstrCurrOutFolder, "VariationX.txt");
	MDLHelper::outputData(mesh, data, sstrCurrOutFolder, geometryFileNames);
}

bool MDLHelper::loadMDLData(const std::string& sstrFileCollectionName, const std::string& sstrTpsFileCollectionName, const std::string& sstrLmkIndexFileName, const std::string& sstrLmkFileCollectionName
									, const std::string& sstrOuterBoundaryIndexFileName, const std::string& sstrInnerBoundaryIndexFileName
									, std::vector<double>& data, DataContainer& mesh, std::vector<std::string>& geometryFileNames
									, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec, std::vector<std::string>& tpsFileNames
									, size_t& numIdentities, size_t& numExpressions
									, std::vector<size_t>& lmkIndices, std::vector<double>& lmks
									, std::vector<size_t>& outerBoundaryVertexIDs, std::vector<size_t>& innerBoundaryVertexIDs)
{
	FileLoader loader;

	if(!loader.loadFileCollection(sstrFileCollectionName, data, mesh, geometryFileNames, numExpressions, numIdentities))
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Unable to load file collection " << sstrFileCollectionName << std::endl;
#endif
		return false;
	}

	size_t tmpNumExpressions(0);
	size_t tmpNumIdentities(0);
	if(!loader.loadTpsFileCollection(sstrTpsFileCollectionName, vecCs, matAs, matWs, sourcePointsVec, tpsFileNames, tmpNumExpressions, tmpNumIdentities))
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Unable to load tps file collection " << sstrTpsFileCollectionName << std::endl;
#endif

		return false;
	}

	if(numIdentities != tmpNumIdentities || numExpressions != tmpNumExpressions)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Number of identies or expressions do not match " << numIdentities << " != " << tmpNumIdentities << " || " << numExpressions << " != " << tmpNumExpressions << std::endl;
#endif

		return false;
	}

	if(!sstrLmkIndexFileName.empty() && !sstrLmkFileCollectionName.empty())
	{
		if(!loader.loadIndexFile(sstrLmkIndexFileName, lmkIndices))
		{
#ifdef DEBUG_OUTPUT
			std::cout << "Unable to load landmark index file " << sstrLmkIndexFileName << std::endl;
#endif
			return false;
		}

		size_t numFiles(0);
		size_t vertexDataDim(0);
		if(!loader.loadVertexDataFileCollection(sstrLmkFileCollectionName, lmks, numFiles, vertexDataDim))
		{
#ifdef DEBUG_OUTPUT
			std::cout << "Unable to load landmarks " << sstrLmkFileCollectionName << std::endl;
#endif
			return false;
		}

		if(numFiles != numIdentities*numExpressions)
		{
			return false;
		}
	}
	else
	{
#ifdef DEBUG_OUTPUT
		std::cout << "WARNING: No landmarks loaded!!!" << std::endl;
#endif
	}

	if(!sstrOuterBoundaryIndexFileName.empty())
	{
		if(!loader.loadIndexFile(sstrOuterBoundaryIndexFileName, outerBoundaryVertexIDs))
		{
#ifdef DEBUG_OUTPUT
			std::cout << "Outer boundary vertex IDs not loaded " << sstrOuterBoundaryIndexFileName << std::endl;
#endif
			return false;
		}
	}
	else
	{
#ifdef DEBUG_OUTPUT
		std::cout << "WARNING: No outer boundary vertex IDs loaded " << sstrOuterBoundaryIndexFileName << std::endl;
#endif
	}

	if(!sstrInnerBoundaryIndexFileName.empty())
	{
		if(!loader.loadIndexFile(sstrInnerBoundaryIndexFileName, innerBoundaryVertexIDs))
		{
#ifdef DEBUG_OUTPUT
			std::cout << "Inner boundary vertex IDs not loaded " << sstrInnerBoundaryIndexFileName << std::endl;
#endif
			return false;
		}
	}
	else
	{
#ifdef DEBUG_OUTPUT
		std::cout << "WARNING: No inner boundary vertex IDs loaded " << sstrOuterBoundaryIndexFileName << std::endl;
#endif
	}

	return true;
}

bool MDLHelper::computeOptimizationBounds(const std::vector<double>& initialParam, const size_t numSamples, const std::vector<size_t>& outerBoundaryVertexIDs, const std::vector<size_t>& innerBoundaryVertexIDs
														, vnl_vector<long>& boundSelection, std::vector<vnl_vector<double>>& lowerShapeBounds, std::vector<vnl_vector<double>>& upperShapeBounds)
{
	if(initialParam.size()%numSamples != 0)
	{
		return false;
	}

	const size_t sampleParameterDim = initialParam.size()/numSamples;
	boundSelection = vnl_vector<long>(sampleParameterDim, 2);

	lowerShapeBounds.clear();
	lowerShapeBounds.resize(numSamples);

	upperShapeBounds.clear();
	upperShapeBounds.resize(numSamples);
	
	for(size_t iSample = 0; iSample < numSamples; ++iSample)
	{
		vnl_vector<double> lowerBounds(sampleParameterDim, 0.0);
		vnl_vector<double> upperBounds(sampleParameterDim, 0.0);

		const size_t sampleParamOffset = iSample*sampleParameterDim;
		for(size_t i = 0; i < sampleParameterDim; ++i)
		{
			const double currMin = OPTIMIZATION_DOMAIN_MIN-initialParam[sampleParamOffset+i];
			lowerBounds[i] = currMin;

			const double currMax = OPTIMIZATION_DOMAIN_MAX-initialParam[sampleParamOffset+i];
			upperBounds[i] = currMax;
		}

		const size_t numOuterBoundaryVertices = outerBoundaryVertexIDs.size();
		for(size_t i = 0; i < numOuterBoundaryVertices; ++i)
		{
			const size_t currVertexID = outerBoundaryVertexIDs[i];
			const size_t startIndex = 2*currVertexID;
			lowerBounds[startIndex] = -MAX_OUTER_BOUNDARY_VARIATION;
			lowerBounds[startIndex+1] = -MAX_OUTER_BOUNDARY_VARIATION;

			upperBounds[startIndex] = MAX_OUTER_BOUNDARY_VARIATION;
			upperBounds[startIndex+1] = MAX_OUTER_BOUNDARY_VARIATION;
		}

		const size_t numInnerBoundaryVertices = innerBoundaryVertexIDs.size();
		for(size_t i = 0; i < numInnerBoundaryVertices; ++i)
		{
			const size_t currVertexID = innerBoundaryVertexIDs[i];
			const size_t startIndex = 2*currVertexID;
			lowerBounds[startIndex] = -MAX_INNER_BOUNDARY_VARIATION;
			lowerBounds[startIndex+1] = -MAX_INNER_BOUNDARY_VARIATION;

			upperBounds[startIndex] = MAX_INNER_BOUNDARY_VARIATION;
			upperBounds[startIndex+1] = MAX_INNER_BOUNDARY_VARIATION;
		}

		lowerShapeBounds[iSample] = lowerBounds;
		upperShapeBounds[iSample] = upperBounds;
	}

	return true;
}

void MDLHelper::getBoundaryNeighbors(const DataContainer& mesh, const std::vector<size_t>& boundaryVertexIDs, std::vector<size_t>& neighborsVertexIDs)
{
	const size_t numVertices = mesh.getNumVertices();

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	const size_t numTriangles = vertexIndexList.size();

	std::vector<std::set<int>> vertexNeighbors;
	vertexNeighbors.resize(numVertices);

	for(size_t i = 0; i < numTriangles; ++i)
	{
		const std::vector<int>& currTriangleIndices = vertexIndexList[i];
		const int i1 = currTriangleIndices[0];
		const int i2 = currTriangleIndices[1];
		const int i3 = currTriangleIndices[2];

		vertexNeighbors[i1].insert(i2);
		vertexNeighbors[i1].insert(i3);

		vertexNeighbors[i2].insert(i1);
		vertexNeighbors[i2].insert(i3);

		vertexNeighbors[i3].insert(i1);
		vertexNeighbors[i3].insert(i2);
	}

	std::set<size_t> tmpNeighbors;

	const size_t numBoundaryVertices = boundaryVertexIDs.size();
	for(size_t i = 0; i < numBoundaryVertices; ++i)
	{
		const size_t currIndex = boundaryVertexIDs[i];

		const std::set<int>& currNeighbors = vertexNeighbors[currIndex];
		if(currNeighbors.empty())
		{
			continue;
		}

		//Iterate over all neighbors
		std::set<int>::const_iterator currNeighborIter = currNeighbors.begin();
		std::set<int>::const_iterator endNeighborIter = currNeighbors.end();
		for(; currNeighborIter != endNeighborIter; ++currNeighborIter)
		{
			const size_t currNeighborIndex = static_cast<size_t>(*currNeighborIter);
			tmpNeighbors.insert(currNeighborIndex);
		}
	}

	neighborsVertexIDs.reserve(tmpNeighbors.size());

	std::set<size_t>::const_iterator tmpNeighborIter = tmpNeighbors.begin();
	std::set<size_t>::const_iterator endTmpNeighborIter = tmpNeighbors.end();
	for(; tmpNeighborIter != endTmpNeighborIter; ++tmpNeighborIter)
	{
		neighborsVertexIDs.push_back(*tmpNeighborIter);
	}
}

void MDLHelper::outputConfig(const std::string& sstrOutFolder, const bool bGroupWise)
{
	const std::string sstrFileName = sstrOutFolder + "/Config.txt";
	std::fstream outStream(sstrFileName, std::ios::out);

	if(bGroupWise)
	{
		outStream << "Group-wise optimization" << std::endl;
	}
	else
	{
		outStream << "Shape-wise optimization" << std::endl;
	}

	outStream << std::endl;
	outStream << "Number of Iterations: " << NUM_ITERATION << std::endl;
	outStream << "Number of Function Evaluation: " << NUM_FKT_EVAL << std::endl;
	outStream << "Recompute Alignment after " << NUM_NUM_FKT_EVAL_ALIGNMENT << " Function Evaluation" << std::endl;
	outStream << std::endl;
	outStream << "Identity weight: " << IDENTITY_WEIGHT << std::endl;
	outStream << "Expression weight: " << EXPRESSION_WEIGHT << std::endl;
	outStream << "Landmark weight: " << LMK_WEIGHT << std::endl;
	outStream << "Smoothness weight: " << SMOOTHNESS_WEIGHT << std::endl;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	outStream << "Use of Bi-Laplacian Smoothing" << std::endl;
#else
	outStream << "Use of Laplacian Smoothing" << std::endl;
#endif
	
	outStream << std::endl;
	outStream << "Max parameter variation: " << MAX_PARAMETER_VARIATION << std::endl;
	outStream << "Maximum variation outer boundary: " << MAX_OUTER_BOUNDARY_VARIATION << std::endl;
	outStream << "Maximum variation inner boundary: " << MAX_INNER_BOUNDARY_VARIATION << std::endl;

	outStream << std::endl;
#ifdef USE_PCA_OPTIMIZATION
	outStream << "PCA gradient optimization is used" << std::endl;
#endif
	outStream.close();
}

void MDLHelper::outputCompactness(const std::vector<double>& data, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName)
{
	double sum(0.0);
	MDLHelper::outputCompactness(data, d1, d2, d3, sstrOutFileName, sum);
}

void MDLHelper::outputCompactness(const std::vector<double>& data, const size_t d1, const size_t d2, const size_t d3, const std::string& sstrOutFileName, double& sum)
{
	std::fstream compactnessStream(sstrOutFileName, std::ios::out);

#ifdef DEBUG_OUTPUT
	std::cout << "Output compactness" << std::endl;
#endif

	std::vector<double> centeredData;
	std::vector<double> mean;
	MathHelper::centerData(data, d1, centeredData, mean);

	Tensor dataTensor;
	dataTensor.init(centeredData, d1, d2, d3);


	compactnessStream << "Mode 2" << std::endl;

#ifdef DEBUG_OUTPUT
	std::cout << "Mode 2" << std::endl;
#endif

	//Multiplication mode 2
	std::vector<double> A2; // d2 x d1*d3
	dataTensor.unfold(2, A2);

	std::vector<double> U2; // d2 x d2	(d2 x min(d2, d1*d3))
	std::vector<double> S2; // d2 x d1*d3 (vector of size min(d2, d1*d3))
	//std::vector<double> V2; // d1*d3 x d1*d3
	MathHelper::computeLeftSingularVectors(A2, d2, d1*d3, U2, S2);

#ifdef DEBUG_OUTPUT
	std::cout << "Num eigenvalues: " << S2.size() << std::endl;
	std::cout << "EV: ";
#endif
	double sumMode2(0.0);
	for(size_t i = 0; i < S2.size(); ++i)
	{
		const double currEV = std::pow(S2[i], 2.0)/static_cast<double>(d3);
		sumMode2 += currEV;

#ifdef DEBUG_OUTPUT
		std::cout << std::setprecision(3) << currEV << " ";
#endif	
	}

#ifdef DEBUG_OUTPUT
	std::cout << std::endl;
#endif

	double tmpSumMode2(0.0);
	for(size_t i = 0; i < S2.size(); ++i)
	{
		const double currEV = std::pow(S2[i], 2.0)/static_cast<double>(d3);
		tmpSumMode2 += currEV;

		const double currValue = 100.0*tmpSumMode2/sumMode2;
		compactnessStream <<  tmpSumMode2 << " (" << currValue << ") " << std::endl;

#ifdef DEBUG_OUTPUT
		std::cout << std::setprecision(3) << tmpSumMode2 << " (" << currValue << ") " << std::endl;
#endif
	}

	std::cout << std::endl;

	compactnessStream << "Mode 3" << std::endl;

	//Multiplication mode 3
	std::vector<double> A3; // d3 x d1*d2
	dataTensor.unfold(3, A3);

	std::vector<double> U3;	// d3 x d3	(d3 x min(d3, d1*d2))
	std::vector<double> S3; // d3 x d1*d2 (vector of size d3 x min(d3, d1*d2))
	//std::vector<double> V3; // d1*d2 x d1*d2
	MathHelper::computeLeftSingularVectors(A3, d3, d1*d2, U3, S3);

#ifdef DEBUG_OUTPUT
	std::cout << "Num eigenvalues: " << S3.size() << std::endl;
	std::cout << "EV: ";
#endif

	double sumMode3(0.0);
	for(size_t i = 0; i < S3.size(); ++i)
	{
		const double currEV = std::pow(S3[i], 2.0)/static_cast<double>(d2);
		sumMode3 += currEV;

#ifdef DEBUG_OUTPUT
		std::cout << std::setprecision(3) << currEV << " ";
#endif	
	}

#ifdef DEBUG_OUTPUT
	std::cout << std::endl;
#endif

	double tmpSumMode3(0.0);
	for(size_t i = 0; i < S3.size(); ++i)
	{
		const double currEV = std::pow(S3[i], 2.0)/static_cast<double>(d2);
		tmpSumMode3 += currEV;

		const double currValue = 100.0*tmpSumMode3/sumMode3;
		compactnessStream <<  tmpSumMode3 << " (" << currValue << ") " << std::endl;

#ifdef DEBUG_OUTPUT
		std::cout << std::setprecision(3) << tmpSumMode3 << " (" << currValue << ") " << std::endl;
#endif
	}

	std::cout << std::endl;


	compactnessStream << std::endl;
	compactnessStream << "Compactness (sum): " << sumMode2+sumMode3 << std::endl;
	/**/sum = sumMode2+sumMode3;
	compactnessStream.close();
}

std::string MDLHelper::getConfigOutFolder(const std::string& sstrOutFolder)
{
	std::stringstream outIdentityWeight;
	outIdentityWeight << IDENTITY_WEIGHT;

	std::stringstream outExpressionWeight;
	outExpressionWeight << EXPRESSION_WEIGHT;

	std::stringstream outLmkWeight;
	outLmkWeight << LMK_WEIGHT;

	std::stringstream outSmoothnessWeight;
	outSmoothnessWeight << SMOOTHNESS_WEIGHT;

#ifdef USE_BI_LAPLACIAN_SMOOTHNESS
	std::string sstrSmoothnessPrefix = "_bls";
#else
	std::string sstrSmoothnessPrefix = "_ls";
#endif

	return sstrOutFolder + "_id" + outIdentityWeight.str() + "_exp" + outExpressionWeight.str() + "_l" + outLmkWeight.str() + sstrSmoothnessPrefix + outSmoothnessWeight.str();
}