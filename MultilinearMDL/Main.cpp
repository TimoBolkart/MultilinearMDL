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

#include "DataContainer.h"
#include "FileLoader.h"
#include "FileWriter.h"
#include "MDLHelper.h"
#include "MathHelper.h"

#include <string>
#include <iostream>
#include <vector>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

void visualizeVertexMotion(const std::string& sstrSourceGeometryFileName, const std::string& sstrTargetFolder, const std::vector<std::string>& fileNames, const std::string& sstrOutFolder, const std::string& outFileAppendix)
{
	FileWriter::makeDirectory(sstrOutFolder);

	FileLoader loader;

	DataContainer sourceMesh;
	if(!loader.loadFile(sstrSourceGeometryFileName, sourceMesh))
	{
		std::cout << "Unable to load source mesh " << sstrSourceGeometryFileName << std::endl;
		return;
	}

	const std::vector<double>& sourceVertices = sourceMesh.getVertexList();

	double minX(DBL_MAX);
	double maxX(-DBL_MAX);
	double minY(DBL_MAX);
	double maxY(-DBL_MAX);

	for(size_t i = 0; i <sourceMesh.getNumVertices(); ++i)
	{
		const double currX = sourceVertices[3*i];
		const double currY = sourceVertices[3*i+1];

		if(currX < minX)
		{
			minX = currX;
		}

		if(currX > maxX)
		{
			maxX = currX;
		}

		if(currY < minY)
		{
			minY = currY;
		}

		if(currY > maxY)
		{
			maxY = currY;
		}
	}

	const size_t numXSteps = 10;
	const size_t numYSteps = 10;

	const double xStepSize = (maxX-minX)/static_cast<double>(numXSteps);
	const double yStepSize = (maxY-minY)/static_cast<double>(numYSteps);

	std::vector<double> vertexColors;
	vertexColors.resize(3*sourceMesh.getNumVertices(), 0.0);

	for(size_t i = 0; i < sourceMesh.getNumVertices(); ++i)
	{
		const double currX = sourceVertices[3*i];
		const double deltaX = fmod(currX-minX, xStepSize);
		const double normDeltaX = deltaX/xStepSize;

		const double currY = sourceVertices[3*i+1];
		const double deltaY = fmod(currY-minY, yStepSize);
		const double normDeltaY = deltaY/yStepSize;

		vertexColors[3*i] = 1.0-normDeltaY;
		vertexColors[3*i+1] = normDeltaX;
		vertexColors[3*i+2] = 0.5*(normDeltaX+normDeltaY);
	}

	const int numFiles = static_cast<int>(fileNames.size());

#pragma omp parallel for
	for(int j = 0; j < numFiles; ++j)
	{
		std::cout << j+1 << " of " << fileNames.size() << std::endl;

		const std::string sstrTargetGeometryFileName = sstrTargetFolder + "/" + fileNames[j];
		
		const std::string sstrGeometryFileName = FileLoader::getFileName(fileNames[j]);
		const std::string sstrOutFileName = sstrOutFolder + "/" + sstrGeometryFileName + "_" + outFileAppendix + ".wrl";

		DataContainer targetMesh;
		if(!loader.loadFile(sstrTargetGeometryFileName, targetMesh))
		{
			std::cout << "Unable to load target mesh " << sstrTargetGeometryFileName << std::endl;
		}

		if(!FileWriter::saveFile(sstrOutFileName, targetMesh, targetMesh.getVertexList(), vertexColors))
		{
			std::cout << "Unable to save file " << sstrOutFileName << std::endl;
		}
	}
}

void visualizeSampledThinPlateSpline(const std::string& sstrOutMeshFileName, const std::vector<double>& vecC, const std::vector<double>& matA, const std::vector<double>& matW, std::vector<double>& sourcePoints
												, const double startValue, const double endValue, const size_t numSteps)
{
	const double stepSize = (endValue-startValue)/static_cast<double>(numSteps-1);

	std::vector<double> vertices;

	for(size_t i = 0; i < numSteps; ++i)
	{
		const double u = startValue+i*stepSize;

		for(size_t j = 0; j < numSteps; ++j)
		{
			const double v = startValue+j*stepSize;

			std::vector<double> inSourcePoint;
			inSourcePoint.push_back(u);
			inSourcePoint.push_back(v);

			std::vector<double> outTargetPoint;
			if(!MathHelper::evaluateInterpolation(vecC, matA, matW, sourcePoints, inSourcePoint, outTargetPoint))
			{
				std::cout << "Unable to compute tps evaluation" << std::endl;
				continue;
			}

			vertices.push_back(outTargetPoint[0]);
			vertices.push_back(outTargetPoint[1]);
			vertices.push_back(outTargetPoint[2]);
		}
	}

	std::vector<std::vector<int>> vertexIndices;

	for(size_t i = 0; i < numSteps-1; ++i)
	{
		for(size_t j = 0; j < numSteps-1; ++j)
		{
			const size_t i1 = j*numSteps+i;
			const size_t i2 = j*numSteps+i+1;
			const size_t i3 = (j+1)*numSteps+i+1;
			const size_t i4 = (j+1)*numSteps+i;
			
			std::vector<int> triangle1;
			triangle1.push_back(i1);
			triangle1.push_back(i3);
			triangle1.push_back(i2);

			std::vector<int> triangle2;
			triangle2.push_back(i1);
			triangle2.push_back(i4);
			triangle2.push_back(i3);

			vertexIndices.push_back(triangle1);
			vertexIndices.push_back(triangle2);
		}
	}

	DataContainer outMesh;
	outMesh.setVertexList(vertices);
	outMesh.setVertexIndexList(vertexIndices);

	FileWriter::saveFile(sstrOutMeshFileName, outMesh);
}

int main(int argc, char* argv[])
{
	if(argc != 6 && argc != 8)
	{
		std::cout << "Wrong number of parameters " << argc << std::endl;
		return 1;
	}

	size_t index = 0;
	std::string sstrIdentifier(argv[++index]);
	if(sstrIdentifier == "-tps" || sstrIdentifier == "tps")
	{
		const std::string sstrDataFolder(argv[++index]);
		const std::string sstrFileCollectionName(argv[++index]);
		const std::string sstrTpsFileCollectionName(argv[++index]);
		const std::string sstrTextureCoordsFileName(argv[++index]);

		//Output thin-plate splines
		if(!MDLHelper::computeThinPlateSplines(sstrDataFolder + "/" + sstrFileCollectionName, sstrTextureCoordsFileName,  sstrDataFolder, sstrDataFolder + "/" + sstrTpsFileCollectionName))
		{
			std::cout << "Computing thin-plate splines failed" << std::endl;
			return 1;
		}

		return 0;
	}
	else if(sstrIdentifier == "-opt" || sstrIdentifier == "opt")
	{
		const std::string sstrDataFolder(argv[++index]);
		const std::string sstrFileCollectionName(argv[++index]);
		const std::string sstrTpsFileCollectionName(argv[++index]);
		const std::string sstrOuterBoundaryIndexFileName(argv[++index]);
		const std::string sstrInnerBoundaryIndexFileName(argv[++index]);
		const std::string sstrOutFolder(argv[++index]);

		MDLHelper::optimizeShapeWise(sstrDataFolder + "/" + sstrFileCollectionName, sstrDataFolder + "/" + sstrTpsFileCollectionName, "", "", sstrOuterBoundaryIndexFileName, sstrInnerBoundaryIndexFileName, sstrOutFolder);
		return 0;
	}
	else
	{
		std::cout << "Identifier not found" << std::endl;
		return 1;
	}
}
