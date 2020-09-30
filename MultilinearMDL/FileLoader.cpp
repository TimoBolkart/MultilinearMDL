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

#include "FileLoader.h"
#include "MathHelper.h"

#include <fstream>
#include <string>
#include <iostream>

std::string FileLoader::getFileExtension(const std::string& sstrFileName)
{
	if(sstrFileName.empty())
	{
		return "";
	}

	const size_t pos = sstrFileName.rfind(".");
	if(pos != std::string::npos)
	{
		return sstrFileName.substr(pos+1);
	}

	return "";
}

bool FileLoader::fileExist(const std::string& sstrFileName)
{
	if(sstrFileName.empty())
	{
		return false;
	}

	std::fstream inStream;
	inStream.open(sstrFileName, std::ios::in);

	if(inStream.is_open())
	{
		inStream.close();
		return true;
	}
	
	return false;
}

std::string FileLoader::getFileName(const std::string& sstrFullFileName)
{
	std::string sstrFileName = "";

	if(sstrFullFileName.empty())
	{
		return sstrFileName;
	}

	//if(!fileExist(sstrFullFileName))
	//{
	//	return sstrFileName;
	//}

	size_t posEnd = sstrFullFileName.rfind(".");
	if(posEnd == std::string::npos)
	{
		posEnd = sstrFullFileName.size();
	}

	size_t posStart = sstrFullFileName.rfind("/");
	if(posStart != std::string::npos)
	{
		sstrFileName = sstrFullFileName.substr(posStart+1, posEnd-posStart-1);
		return sstrFileName;
	}

	posStart = sstrFullFileName.rfind("\\");
	if(posStart != std::string::npos)
	{
		sstrFileName = sstrFullFileName.substr(posStart+1, posEnd-posStart-1);
		return sstrFileName;
	}
	else
	{
		sstrFileName = sstrFullFileName.substr(0, posEnd);
	}

	return sstrFileName;
}

std::string FileLoader::getFilePath(const std::string& sstrFileName)
{
	std::string sstrFilePath = "";

	if(sstrFileName.empty())
	{
		return sstrFilePath;
	}

	if(!fileExist(sstrFileName))
	{
		return sstrFilePath;
	}

	size_t pos = sstrFileName.rfind("/");
	if(pos != std::string::npos)
	{
		sstrFilePath = sstrFileName.substr(0, pos);
		return sstrFilePath;
	}

	pos = sstrFileName.rfind("\\");
	if(pos != std::string::npos)
	{
		sstrFilePath = sstrFileName.substr(0, pos);
		return sstrFilePath;
	}

	return sstrFilePath;
}

bool FileLoader::loadFile(const std::string& sstrFileName, DataContainer& outData)
{
	if(!fileExist(sstrFileName))
	{
		return false;
	}

	bool bReturn(false);

	const std::string sstrSuffix = FileLoader::getFileExtension(sstrFileName);
	if(sstrSuffix=="off")
	{
		bReturn = loadOFF(sstrFileName, outData);
		if(bReturn)
		{
			MathHelper::cleanMesh(outData);
		}
	}
	else if(sstrSuffix=="wrl")
	{
		bReturn = loadWRL(sstrFileName, outData);
		if(bReturn)
		{
			MathHelper::cleanMesh(outData);
		}
	}
	else if (sstrSuffix=="obj")
	{
		bReturn = loadObj(sstrFileName, outData);
		if(bReturn)
		{
			MathHelper::cleanMesh(outData);
		}
	}

	return bReturn;
}

bool FileLoader::readFileCollection(const std::string& sstrFileName, std::vector<std::string>& fileNames)
{
	if(!fileExist(sstrFileName))
	{
		std::cout << "File does not exist" << std::endl;
		return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		std::cout << "Unable to get file path" << std::endl; 
		return false;
	}

	FILE* pFile = fopen(sstrFileName.c_str(), "r");
	if(pFile==NULL)
	{
		std::cout << "Unable to open file" << std::endl;
		return false;
	}

	bool bEnd(false);
	while(!bEnd)
	{
		char strFileName[1000];
		bEnd = readNextNode(pFile, strFileName);
		if(bEnd)
		{
			break;
		}

		const std::string sstrTmpFileName(strFileName);
		fileNames.push_back(sstrTmpFileName);
	}

	return true;
}

bool FileLoader::readFileCollection(const std::string& sstrFileName, std::vector<std::string>& fileNames, size_t& numExpressions, size_t& numIdentities)
{
	if(!fileExist(sstrFileName))
	{
		std::cout << "File does not exist" << std::endl;
		return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		std::cout << "Unable to get file path" << std::endl; 
		return false;
	}

	FILE* pFile = fopen(sstrFileName.c_str(), "r");
	if(pFile==NULL)
	{
		std::cout << "Unable to open file" << std::endl;
		return false;
	}

	char strOutput[1000];
	bool bSuccess = readNextNumber(pFile, INTEGER, strOutput, numExpressions);
	bSuccess &= readNextNumber(pFile, INTEGER, strOutput, numIdentities);
	if(!bSuccess)
	{
		return false;
	}	

	bool bEnd(false);
	while(!bEnd)
	{
		char strFileName[1000] = "";
		bEnd = readNextNode(pFile, strFileName);
		if(bEnd && strcmp(strFileName, "") == 0)
		{
			break;
		}

		const std::string sstrTmpFileName(strFileName);
		fileNames.push_back(sstrTmpFileName);
	}

	return true;
}

bool FileLoader::loadFileCollection(const std::string& sstrFileName, std::vector<DataContainer*>& meshes, std::vector<std::string>& fileNames, size_t& numExpressions, size_t& numIdentities)
{
	if(!readFileCollection(sstrFileName, fileNames, numExpressions, numIdentities))
	{
		return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		return false;
	}

	std::vector<std::string>::const_iterator currFileNameIter = fileNames.begin();
	std::vector<std::string>::const_iterator endFileNameIter = fileNames.end();
	for(; currFileNameIter != endFileNameIter; ++currFileNameIter)
	{
		const std::string& sstrCurrName = *currFileNameIter;
		const std::string sstrCurrFileName = sstrFilePath + "/" + sstrCurrName;
						
		if(!fileExist(sstrCurrFileName))
		{
			return false;
		}

		DataContainer* pNewMesh = new DataContainer();
		if(!loadFile(sstrCurrFileName, *pNewMesh))
		{
			delete pNewMesh;
			return false;
		}

		meshes.push_back(pNewMesh);
	}

	if(numExpressions*numIdentities!=meshes.size())
	{
		for(size_t i = 0; i < meshes.size(); ++i)
		{
			delete meshes[i];
		}

		meshes.clear();
		return false;
	}

	return meshes.size() == fileNames.size();
}

bool FileLoader::loadFileCollection(const std::string& sstrFileName, std::vector<double>& data, DataContainer& mesh, std::vector<std::string>& fileNames, size_t& numExpressions, size_t& numIdentities)
{
	if(!readFileCollection(sstrFileName, fileNames, numExpressions, numIdentities))
	{
	  std::cout << "Unable to read file follection " << std::endl;  
	  return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		std::cout << "File path is empty " << std::endl;  
		return false;
	}

	std::vector<std::string>::const_iterator currFileNameIter = fileNames.begin();
	std::vector<std::string>::const_iterator endFileNameIter = fileNames.end();
	for(; currFileNameIter != endFileNameIter; ++currFileNameIter)
	{
		const std::string& sstrCurrName = *currFileNameIter;
		const std::string sstrCurrFileName = sstrFilePath + "/" + sstrCurrName;
						
		if(!fileExist(sstrCurrFileName))
		{
			std::cout << "File does not exist " << sstrCurrFileName << std::endl;		  
			return false;
		}

		DataContainer currMesh;
		if(!loadFile(sstrCurrFileName, currMesh))
		{
			std::cout << "Unable to load file " << sstrCurrFileName << std::endl;
			return false;
		}

		const size_t numMeshVertices = currMesh.getNumVertices();
		if(currFileNameIter == fileNames.begin())
		{
			data.reserve(numIdentities*numExpressions*numMeshVertices*3);
			mesh = currMesh;
		}

		const std::vector<double>& meshVertices = currMesh.getVertexList();

		for(size_t i = 0; i < numMeshVertices; ++i)
		{
			data.push_back(meshVertices[3*i]);
			data.push_back(meshVertices[3*i+1]);
			data.push_back(meshVertices[3*i+2]);
		}
	}

	return (data.size()%numIdentities == 0) && (data.size()%numExpressions == 0) && (fileNames.size() == numIdentities*numExpressions);
}

bool FileLoader::loadTpsFileCollection(const std::string& sstrFileName, std::vector<std::vector<double>>& vecCs, std::vector<std::vector<double>>& matAs, std::vector<std::vector<double>>& matWs, std::vector<std::vector<double>>& sourcePointsVec
													, std::vector<std::string>& fileNames, size_t& numExpressions, size_t& numIdentities)
{
	if(!readFileCollection(sstrFileName, fileNames, numExpressions, numIdentities))
	{
		return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		return false;
	}

	std::vector<std::string>::const_iterator currFileNameIter = fileNames.begin();
	std::vector<std::string>::const_iterator endFileNameIter = fileNames.end();
	for(; currFileNameIter != endFileNameIter; ++currFileNameIter)
	{
		const std::string& sstrCurrName = *currFileNameIter;
		const std::string sstrCurrFileName = sstrFilePath + "/" + sstrCurrName;
						
		if(!fileExist(sstrCurrFileName))
		{
			return false;
		}

		std::vector<double> vecC;
		std::vector<double> matA; 
		std::vector<double> matW; 
		std::vector<double> sourcePoints;
		if(!loadThinPlateSpline(sstrCurrFileName, vecC, matA, matW, sourcePoints))
		{
			return false;
		}

		vecCs.push_back(vecC);
		matAs.push_back(matA);
		matWs.push_back(matW);
		sourcePointsVec.push_back(sourcePoints);
	}	

	return true;
}

bool FileLoader::loadVertexDataFileCollection(const std::string& sstrFileName, std::vector<double>& vertexData, size_t& numFiles, size_t& vertexDataDim)
{
	std::vector<std::string> fileNames; 
	size_t numExpressions(0);
	size_t numIdentities(0);
	if(!readFileCollection(sstrFileName, fileNames, numExpressions, numIdentities))
	{
		return false;
	}

	std::string sstrFilePath = getFilePath(sstrFileName);
	if(sstrFilePath.empty())
	{
		return false;
	}

	numFiles = fileNames.size();

	std::vector<std::string>::const_iterator currFileNameIter = fileNames.begin();
	std::vector<std::string>::const_iterator endFileNameIter = fileNames.end();
	for(; currFileNameIter != endFileNameIter; ++currFileNameIter)
	{
		const std::string& sstrCurrName = *currFileNameIter;
		const std::string sstrCurrFileName = sstrFilePath + "/" + sstrCurrName;
						
		if(!fileExist(sstrCurrFileName))
		{
			return false;
		}


		std::vector<double> data;
		if(!loadVertexDataFile(sstrCurrFileName, false, data))
		{
			return false;
		}

		if(currFileNameIter == fileNames.begin())
		{
			vertexDataDim = data.size();
		}
		else
		{
			if(vertexDataDim != data.size())
			{
				return false;
			}
		}

		for(size_t i = 0; i < vertexDataDim; ++i)
		{
			vertexData.push_back(data[i]);
		}
	}	

	return true;
}

bool FileLoader::loadLandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& landmarksLoaded)
{
	if(sstrFileName.empty())
	{
		return false;
	}

	const std::string suffix = FileLoader::getFileExtension(sstrFileName);
	if(suffix==std::string("lm3"))
	{
		return loadBosphorusLandmarks(sstrFileName, landmarks, landmarksLoaded);
	}
	else if(suffix==std::string("txt"))
	{
		return loadSimpleLandmarks(sstrFileName, landmarks, landmarksLoaded);
	}
	else if(suffix==std::string("tlmk"))
	{
		return loadSpecifiedTLandmarks(sstrFileName, landmarks, landmarksLoaded);
	}
	else if(suffix == std::string("bnd"))
	{
		return loadBNDLandmarks(sstrFileName, landmarks, landmarksLoaded);
	}
	else if(suffix == std::string("pse"))
	{
		return loadPSELandmarks(sstrFileName, landmarks, landmarksLoaded);
	}

	return false;
}

bool FileLoader::loadMultilinearModel(const std::string& sstrFileName, std::vector<size_t>& modeDims, std::vector<size_t>& truncModeDims, std::vector<double>& multModel
												  , std::vector<double>& uMatrices, std::vector<double>& sVectors, std::vector<double>& mean)
{
	if(sstrFileName.empty())
	{
		return false;
	}

	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	char strOutput[1000];
	int numModes(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, numModes))
	{
		return false;
	}

	for(int i = 0; i < numModes; ++i)
	{
		char strOutput[1000];
		size_t modeDim(0);
		if(!readNextNumber(pFile, INTEGER, strOutput, modeDim))
		{
			return false;
		}

		modeDims.push_back(modeDim);
	}

	for(int i = 0; i < numModes; ++i)
	{
		char strOutput[1000];
		size_t truncatedModeDimension(0);
		if(!readNextNumber(pFile, INTEGER, strOutput, truncatedModeDimension))
		{
			return false;
		}

		truncModeDims.push_back(truncatedModeDimension);
	}

	if(numModes<1)
	{
		return false;
	}

	size_t numTensorElements = modeDims[0];
	size_t numUMatrixElements = 0;
	size_t numSVectorElements = 0;
	size_t meanSize = modeDims[0];

	for(int i = 1; i < numModes; ++i)
	{
		const size_t d_i = modeDims[i];
		const size_t m_i = truncModeDims[i];

		numTensorElements *= d_i;
		numUMatrixElements += d_i*m_i;
		numSVectorElements += m_i;
	}

	multModel.reserve(numTensorElements);
	uMatrices.reserve(numUMatrixElements);
	sVectors.reserve(numSVectorElements);
	mean.reserve(meanSize);

	bool bEnd(false);
	while(!bEnd)
	{
		char strOutput[1000];
		
		while(!bEnd 
			&& !isEqual(strOutput, "MultilinearModel") 
			&& !isEqual(strOutput, "UMatrix")
			&& !isEqual(strOutput, "SVector")
			&& !isEqual(strOutput, "Mean"))
		{
			bEnd = readNextNode(pFile, strOutput);
		}

		if(!bEnd && isEqual(strOutput, "MultilinearModel"))
		{
			processBlockStructure(pFile, strOutput, multModel);
		}

		if(!bEnd && isEqual(strOutput, "UMatrix"))
		{
			processBlockStructure(pFile, strOutput, uMatrices);
		}

		if(!bEnd && isEqual(strOutput, "SVector"))
		{
			processBlockStructure(pFile, strOutput, sVectors);
		}

		if(!bEnd && isEqual(strOutput, "Mean"))
		{
			processBlockStructure(pFile, strOutput, mean);
		}
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadRestrictedMultilinearModel(const std::string& sstrFileName, std::vector<size_t>& modeDims, std::vector<size_t>& truncModeDims, std::vector<double>& multModel
																, std::vector<double>& modeMean, std::vector<double>& neutralModelMean, std::vector<double>& mean)
{
	if(sstrFileName.empty())
	{
		return false;
	}

	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	char strOutput[1000];
	int numModes(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, numModes))
	{
		return false;
	}

	for(int i = 0; i < numModes; ++i)
	{
		char strOutput[1000];
		size_t modeDim(0);
		if(!readNextNumber(pFile, INTEGER, strOutput, modeDim))
		{
			return false;
		}

		modeDims.push_back(modeDim);
	}

	for(int i = 0; i < numModes; ++i)
	{
		char strOutput[1000];
		size_t truncatedModeDimension(0);
		if(!readNextNumber(pFile, INTEGER, strOutput, truncatedModeDimension))
		{
			return false;
		}

		truncModeDims.push_back(truncatedModeDimension);
	}

	if(numModes<1)
	{
		return false;
	}

	size_t numTensorElements = modeDims[0];
	size_t modeMeanSize = 0;
	const size_t meanSize = modeDims[0];

	for(int i = 1; i < numModes; ++i)
	{
		const size_t d_i = modeDims[i];
		const size_t m_i = truncModeDims[i];

		numTensorElements *= d_i;
		modeMeanSize += m_i;
	}

	multModel.reserve(numTensorElements);
	modeMean.reserve(modeMeanSize);
	mean.reserve(meanSize);

	bool bEnd(false);
	while(!bEnd)
	{
		char strOutput[1000];
		
		while(!bEnd 
			&& !isEqual(strOutput, "MultilinearModel") 
			&& !isEqual(strOutput, "ModeMean")
			&& !isEqual(strOutput, "NeutralModeMean")
			&& !isEqual(strOutput, "Mean"))
		{
			bEnd = readNextNode(pFile, strOutput);
		}

		if(!bEnd && isEqual(strOutput, "MultilinearModel"))
		{
			processBlockStructure(pFile, strOutput, multModel);
		}

		if(!bEnd && isEqual(strOutput, "ModeMean"))
		{
			processBlockStructure(pFile, strOutput, modeMean);
		}

		if(!bEnd && isEqual(strOutput, "NeutralModeMean"))
		{
			processBlockStructure(pFile, strOutput, neutralModelMean);
		}

		if(!bEnd && isEqual(strOutput, "Mean"))
		{
			processBlockStructure(pFile, strOutput, mean);
		}
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadPCAModel(const std::string& sstrFileName, size_t& inputDimension, size_t& outputDimension, std::vector<double>& mean, std::vector<double>& basis, std::vector<double>& singularValues)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	char strOutput[100];

	int inputDim(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, inputDim))
	{
		return false;
	}

	inputDimension = static_cast<size_t>(inputDim);

	int outputDim(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, outputDim))
	{
		return false;
	}

	outputDimension = static_cast<size_t>(outputDim);

	std::vector<double> transposedBasis;

	bool bEnd(false);
	while(!bEnd)
	{	
		while(!bEnd 
			&& !isEqual(strOutput, "Mean") 
			&& !isEqual(strOutput, "Basis")
			&& !isEqual(strOutput, "SingularValues"))
		{
			bEnd = readNextNode(pFile, strOutput);
		}

		if(!bEnd && isEqual(strOutput, "Mean"))
		{
			processBlockStructure(pFile, strOutput, mean);
		}

		if(!bEnd && isEqual(strOutput, "Basis"))
		{
			processBlockStructure(pFile, strOutput, transposedBasis);
		}

		if(!bEnd && isEqual(strOutput, "SingularValues"))
		{
			processBlockStructure(pFile, strOutput, singularValues);
		}
	}

	if(transposedBasis.size() == inputDimension*outputDimension)
	{
		basis.clear();
		basis.resize(inputDimension*outputDimension);

		for(size_t row = 0; row < inputDimension; ++row)
		{
			for(size_t col = 0; col < outputDimension; ++col)
			{
				basis[col*inputDimension+row] = transposedBasis[row*outputDimension+col];
			}
		}
	}

	return (fclose(pFile)==0) &&  (mean.size() == inputDimension) && (basis.size() == inputDimension*outputDimension) && (singularValues.size() == outputDimension);
}

bool FileLoader::loadSequenceWeights(const std::string& sstrShapeWeightFileName, const std::string& sstrExpressionWeightFileName, std::vector<double>& sequenceWeights, size_t& shapeWeightDim,size_t& expWeightDim, size_t& numFrames)
{
	size_t numShapeWeightVecs(0);
	std::vector<double> shapeWeights;
	bool bLoadShapeWeights = loadWeights(sstrShapeWeightFileName, shapeWeights, shapeWeightDim, numShapeWeightVecs);
	if(!bLoadShapeWeights)
	{
		return false;
	}

	if(numShapeWeightVecs != 1)
	{
		return false;
	}

	std::vector<double> expSequenceWeights;
	bool bLoadExpWeights = loadWeights(sstrExpressionWeightFileName, expSequenceWeights, expWeightDim, numFrames);
	if(!bLoadExpWeights)
	{
		return false;
	}

	if(shapeWeightDim == 0 || expWeightDim == 0)
	{
		return false;
	}

	sequenceWeights.reserve(shapeWeights.size()+expSequenceWeights.size());
	for(size_t i = 0; i < shapeWeights.size(); ++i)
	{
		sequenceWeights.push_back(shapeWeights[i]);
	}

	for(size_t i = 0; i < expSequenceWeights.size(); ++i)
	{
		sequenceWeights.push_back(expSequenceWeights[i]);
	}

	return true;
}

bool FileLoader::loadShapeWeights(const std::string& sstrShapeWeightFileName, std::vector<double>& sequenceShapeWeight)
{
	size_t dim(0);
	size_t numShapeWeightVecs(0);
	bool bLoadShapeWeights = loadWeights(sstrShapeWeightFileName, sequenceShapeWeight, dim, numShapeWeightVecs);
	if(!bLoadShapeWeights || numShapeWeightVecs != 1)
	{
		return false;
	}

	return true;
}

bool FileLoader::loadSequenceExpressionWeights(const std::string& sstrExpressionWeightFileName, std::vector<double>& sequenceExpressionWeights, size_t& expWeightDim, size_t& numFrames)
{
	return loadWeights(sstrExpressionWeightFileName, sequenceExpressionWeights, expWeightDim, numFrames);
}

bool FileLoader::loadSampleFile(const std::string& sstrSampleFileName, const size_t numPoints, std::vector<bool>& validVec)
{
	const char* cstrFileName = sstrSampleFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	validVec.reserve(numPoints);
	for(size_t i = 0; i < numPoints; ++i)
	{
		validVec.push_back(false);
	}

	while(true)
	{
		int val(0);
		char strOutput[1000];
		if(!readNextNumber(pFile, INTEGER, strOutput, val))
		{
			break;
		}

		if(val<0 || val>=numPoints)
		{
			return false;
		}

		validVec[val] = true;
	}

	return fclose(pFile)==0;;
}

bool FileLoader::loadIndexFile(const std::string& sstrIndexFileName, std::vector<size_t>& indices)
{
	const char* cstrFileName = sstrIndexFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	while(true)
	{
		int val(0);
		char strOutput[1000];
		if(!readNextNumber(pFile, INTEGER, strOutput, val))
		{
			break;
		}

		if(val < 0)
		{
			return false;
		}

		indices.push_back(static_cast<size_t>(val));
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadDataFile(const std::string& sstrDataFileName, std::vector<double>& data)
{
	const char* cstrFileName = sstrDataFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	while(true)
	{
		double val(0.0);
		char strOutput[1000];
		if(!readNextNumber(pFile, FLOATING_POINT, strOutput, val))
		{
			break;
		}

		data.push_back(val);
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadVertexDataFile(const std::string& sstrDataFileName, bool bIgnoreFirstColumn, std::vector<double>& data)
{
	const char* cstrFileName = sstrDataFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	std::vector<double> tmpData;

	char strOutput[1000];
	while(true)
	{
		double val(0.0);
		if(!readNextNumber(pFile, FLOATING_POINT, strOutput, val))
		{
			break;
		}

		tmpData.push_back(val);
	}

	if(bIgnoreFirstColumn)
	{
		const size_t numData = tmpData.size()/4;
		data.reserve(3*numData);

		for(size_t i = 0; i < tmpData.size(); ++i)
		{
			if(i%4 == 0)
			{
				continue;
			}

			data.push_back(tmpData[i]);
		}
	}
	else
	{
		data = tmpData;
	}

	return fclose(pFile)==0 && data.size()%3 == 0;
}

bool FileLoader::loadWRL(const std::string& sstrFileName, DataContainer& outData)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	bool bEnd(false);
	while(!bEnd)
	{
		char output[1000] = "";
		
		while(!bEnd && !isEqual(output, "Shape") && !isEqual(output, "Shape"))
		{
			bEnd = readNextNode(pFile, output);
		}

		if(!bEnd && isEqual(output, "Shape"))
		{
			bEnd = processShapeNode(pFile, output, outData);
		}

		if(!bEnd && isEqual(output, "Transform"))
		{
			//bEnd = processTransformNode(pFile, output, ...);
		}
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadOFF(const std::string& sstrFileName, DataContainer& outData)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	char output[1000] = "";
	readNextNode(pFile, output);

	bool bColorOff(false);
	if(strcmp(output, "OFF") == 0)
	{
		bColorOff = false;
	}
	else if(strcmp(output, "COFF") == 0)
	{
		bColorOff = true;
	}
	else
	{
		return false;
	}

	int numVertices(0);
	readNextNumber(pFile, INTEGER, output, numVertices);

	int numFaces(0);
	readNextNumber(pFile, INTEGER, output, numFaces);

	int numEdges(0);
	readNextNumber(pFile, INTEGER, output, numEdges);

	if(numVertices < 1 /*|| numFaces < 1*/)
	{
		return false;
	}

	std::vector<double> vertexList;
	std::vector<std::vector<int>> vertexIndexList;
	std::vector<double> vertexColorList;

	bool bEnd(false);
	for(int vertex = 0; vertex < numVertices; ++vertex)
	{
		char tmpOutput[1000];
	
		for(int i = 0; i < 3; ++i)
		{
			double number(0.0);		
			if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, number))
			{
				bEnd = true;
				break;
			}

			vertexList.push_back(number);
		}

		if(bEnd || vertexList.size()%3 != 0)
		{
			std::cout << "Loaded vertex list of wrong dimension" << std::endl;
			return false;
		}

		if(bColorOff)
		{
			for(int i = 0; i < 3; ++i)
			{
				double colorValue(0);		
				if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, colorValue))
				{
					bEnd = true;
					break;
				}

				vertexColorList.push_back(colorValue);
			}

			if(bEnd || vertexColorList.size()%3 != 0)
			{
				std::cout << "Loaded vertex color list of wrong dimension" << std::endl;
				return false;
			}

			double alphaValue(0);		
			if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, alphaValue))
			{
				return false;
			}
		}
	}

	for(int face = 0; face < numFaces; ++face)
	{
		char tmpOutput[1000];

		int numPolyPoints(0);		
		if(!readNextNumber(pFile, INTEGER, tmpOutput, numPolyPoints))
		{
			break;
		}

		if(numPolyPoints!=3)
		{
#ifdef DEBUG_OUTPUT
			std::cout << "loadOff() - only triangles supported" << std::endl; 
#endif
			return false;
		}

		std::vector<int> polyIndices;
		for(int i = 0; i < 3; ++i)
		{
			int vertexIndex(0);		
			if(!readNextNumber(pFile, INTEGER, tmpOutput, vertexIndex))
			{
				bEnd = true;
				break;
			}

			polyIndices.push_back(vertexIndex);
		}

		if(bEnd || polyIndices.size() % 3 != 0) 
		{
			std::cout << "Polygon not considered" << std::endl;
			break;
		}

		vertexIndexList.push_back(polyIndices);
	}


	if(!outData.setVertexList(vertexList))
	{
		return false;
	}

	outData.setVertexIndexList(vertexIndexList);
	outData.setVertexColorList(vertexColorList); 

	return fclose(pFile)==0;
}

bool FileLoader::loadObj(const std::string& sstrFileName, DataContainer& outData)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	std::vector<double> vertexList;
	std::vector<std::vector<int>> vertexIndexList;
	std::vector<double> vertexColors;

	bool bEnd(false);

	char output[1000];
	while(strcmp(output, "v") != 0 
			&& strcmp(output, "vn") != 0 
			&& strcmp(output, "vt") != 0
			&& strcmp(output, "f") != 0 
			&& !bEnd)
	{
		bEnd = readNextNode(pFile, output);
	}

	//Read vertices
	while(strcmp(output, "v") == 0 && !bEnd)
	{
		char tmpOutput[1000];
	
		Vec3d* pVertex = new Vec3d(0.0,0.0,0.0);
		for(int i = 0; i < 3; ++i)
		{
			double number(0.0);		
			if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, number))
			{
				bEnd = true;
				break;
			}

			vertexList.push_back(number);
		}

		if(bEnd || vertexList.size()%3 != 0)
		{
			return false;
		}

		bEnd = readNextNode(pFile, output);
	}

	while(strcmp(output, "v") != 0 
			&& strcmp(output, "vn") != 0 
			&& strcmp(output, "vt") != 0
			&& strcmp(output, "f") != 0 
			&& !bEnd)
	{
		bEnd = readNextNode(pFile, output);
	}

	//Read normals
	while(strcmp(output, "vn") == 0 && !bEnd)
	{
		char tmpOutput[1000];

		for(int i = 0; i < 3; ++i)
		{
			double number(0.0);		
			if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, number))
			{
				bEnd = true;
				break;
			}
		}

		if(bEnd)
		{
			return false;
		}

		bEnd = readNextNode(pFile, output);
	}

	while(strcmp(output, "v") != 0 
			&& strcmp(output, "vn") != 0 
			&& strcmp(output, "vt") != 0
			&& strcmp(output, "f") != 0 
			&& !bEnd)
	{
		bEnd = readNextNode(pFile, output);
	}

	//Read texture coordinates
	while(strcmp(output, "vt") == 0 && !bEnd)
	{
		char tmpOutput[1000];

		for(int i = 0; i < 2; ++i)
		{
			double number(0.0);		
			if(!readNextNumber(pFile, FLOATING_POINT, tmpOutput, number))
			{
				bEnd = true;
				break;
			}
		}

		if(bEnd)
		{
			return false;
		}

		bEnd = readNextNode(pFile, output);
	}

	while(strcmp(output, "v") != 0 
			&& strcmp(output, "vn") != 0 
			&& strcmp(output, "vt") != 0
			&& strcmp(output, "f") != 0 
			&& !bEnd)
	{
		bEnd = readNextNode(pFile, output);
	}

	//Read faces
	while(strcmp(output, "f") == 0 && !bEnd)
	{
		char tmpOutput[1000];

		//Vec3i* pPoly = new Vec3i(0,0,0);
		std::vector<int> polygon;
		for(size_t i = 0; i < 3; ++i)
		{
			bEnd = readNextNode(pFile, tmpOutput);
			if(bEnd)
			{
				//delete pPoly;
				return false;
			}

			const std::string sstrTmpOutput(tmpOutput);

			std::string sstrVertexNum("");
			std::string sstrTexturNum("");
			std::string sstrNormalNum("");

			const size_t pos1 = sstrTmpOutput.find_first_of("/");
			if(pos1 != std::string::npos)
			{
				sstrVertexNum = sstrTmpOutput.substr(0, pos1);
			}
			else
			{
				sstrVertexNum = sstrTmpOutput;
			}

			//const size_t pos2 = sstrTmpOutput.find_first_of("/", pos1+1);
			//if(pos2 == std::string::npos)
			//{

			//}


			//sstrTexturNum = sstrTmpOutput.substr(pos1+1, pos2-pos1-1);
			//sstrNormalNum = sstrTmpOutput.substr(pos2+1);

			if(sstrVertexNum.empty())
			{
				//delete pPoly;
				return false;
			}

			const int vertexNum = atoi(sstrVertexNum.c_str());
			//const int num2 = atoi(sstrTexturNum.c_str());
			//const int num3 = atoi(sstrNormalNum.c_str());

			//(*pPoly)[i] = vertexNum-1;
			polygon.push_back(vertexNum-1);
		}

		//vertexIndexList.push_back(pPoly);
		vertexIndexList.push_back(polygon);
		bEnd = readNextNode(pFile, output);
	}

	outData.setVertexList(vertexList);
	outData.setVertexIndexList(vertexIndexList);
	outData.setVertexColorList(vertexColors);

	return fclose(pFile)==0;
}

bool FileLoader::loadBosphorusLandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& loaded)
{
	return loadSpecifiedTLandmarks(sstrFileName, landmarks, loaded);
}

bool FileLoader::loadSpecifiedTLandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& loaded)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	const int numInterested = 13;
	loaded.resize(numInterested);

	for(int i = 0; i < numInterested; ++i)
	{
		loaded[i] = false;
	}

	char buffer[300];
	fscanf(pFile, "%[^\n] ", &buffer);

	int numLnd(0);
	fscanf(pFile, "%d %*s ", &numLnd);

	landmarks.resize(numInterested*3);

	for(int i = 0; i < numLnd; ++i)
	{
		fscanf(pFile, "%[^\n] ", &buffer);

		float readValX(0.0f), readValY(0.0f), readValZ(0.0f);
		fscanf(pFile, "%f %f %f ", &readValX, &readValY, &readValZ);

		if(strcmp(buffer, "Outer left eye corner") == 0)
		{
			landmarks[0] = static_cast<double>(readValX);
			landmarks[1] = static_cast<double>(readValY);
			landmarks[2] = static_cast<double>(readValZ);
			loaded[0] = true;
		}
		else if(strcmp(buffer, "Inner left eye corner") == 0)
		{
			landmarks[3] = static_cast<double>(readValX);
			landmarks[4] = static_cast<double>(readValY);
			landmarks[5] = static_cast<double>(readValZ);
			loaded[1] = true;
		}
		else if(strcmp(buffer, "Inner right eye corner") == 0)
		{
			landmarks[6] = static_cast<double>(readValX);
			landmarks[7] = static_cast<double>(readValY);
			landmarks[8] = static_cast<double>(readValZ);
			loaded[2] = true;
		}
		else if(strcmp(buffer, "Outer right eye corner") == 0)
		{
			landmarks[9] = static_cast<double>(readValX);
			landmarks[10] = static_cast<double>(readValY);
			landmarks[11] = static_cast<double>(readValZ);
			loaded[3] = true;
		}
		else if(strcmp(buffer, "Nose tip") == 0)
		{
			landmarks[12] = static_cast<double>(readValX);
			landmarks[13] = static_cast<double>(readValY);
			landmarks[14] = static_cast<double>(readValZ);
			loaded[4] = true;
		}
		else if(strcmp(buffer, "Left nose peak") == 0)
		{
			landmarks[15] = static_cast<double>(readValX);
			landmarks[16] = static_cast<double>(readValY);
			landmarks[17] = static_cast<double>(readValZ);
			loaded[5] = true;
		}
		else if(strcmp(buffer, "Right nose peak") == 0)
		{
			landmarks[18] = static_cast<double>(readValX);
			landmarks[19] = static_cast<double>(readValY);
			landmarks[20] = static_cast<double>(readValZ);
			loaded[6] = true;
		}
		else if(strcmp(buffer, "Subnasal point") == 0)
		{
			landmarks[21] = static_cast<double>(readValX);
			landmarks[22] = static_cast<double>(readValY);
			landmarks[23] = static_cast<double>(readValZ);
			loaded[7] = true;
		}
		else if(strcmp(buffer, "Left mouth corner") == 0)
		{
			landmarks[24] = static_cast<double>(readValX);
			landmarks[25] = static_cast<double>(readValY);
			landmarks[26] = static_cast<double>(readValZ);
			loaded[8] = true;
		}
		else if(strcmp(buffer, "Right mouth corner") == 0)
		{
			landmarks[27] = static_cast<double>(readValX);
			landmarks[28] = static_cast<double>(readValY);
			landmarks[29] = static_cast<double>(readValZ);
			loaded[9] = true;
		}
		else if(strcmp(buffer, "Upper lip outer middle") == 0)
		{
			landmarks[30] = static_cast<double>(readValX);
			landmarks[31] = static_cast<double>(readValY);
			landmarks[32] = static_cast<double>(readValZ);
			loaded[10] = true;
		}
		else if(strcmp(buffer, "Lower lip outer middle") == 0)
		{
			landmarks[33] = static_cast<double>(readValX);
			landmarks[34] = static_cast<double>(readValY);
			landmarks[35] = static_cast<double>(readValZ);
			loaded[11] = true;
		}
		else if(strcmp(buffer, "Chin middle") == 0)
		{
			landmarks[36] = static_cast<double>(readValX);
			landmarks[37] = static_cast<double>(readValY);
			landmarks[38] = static_cast<double>(readValZ);
			loaded[12] = true;
		}
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadBNDLandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& loaded)
{
	std::vector<double> tmpLandmarks;
	if(!loadVertexDataFile(sstrFileName, true, tmpLandmarks))
	{
		return false;
	}

	const size_t numLoadedLandmarks = tmpLandmarks.size()/3;
	if(numLoadedLandmarks != 83)
	{
		return false;
	}

	//4
	landmarks.push_back(tmpLandmarks[12]);
	landmarks.push_back(tmpLandmarks[13]);
	landmarks.push_back(tmpLandmarks[14]);
	loaded.push_back(true);

	//0
	landmarks.push_back(tmpLandmarks[0]);
	landmarks.push_back(tmpLandmarks[1]);
	landmarks.push_back(tmpLandmarks[2]);
	loaded.push_back(true);

	//8
	landmarks.push_back(tmpLandmarks[24]);
	landmarks.push_back(tmpLandmarks[25]);
	landmarks.push_back(tmpLandmarks[26]);
	loaded.push_back(true);

	//12
	landmarks.push_back(tmpLandmarks[36]);
	landmarks.push_back(tmpLandmarks[37]);
	landmarks.push_back(tmpLandmarks[38]);
	loaded.push_back(true);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//39
	landmarks.push_back(tmpLandmarks[117]);
	landmarks.push_back(tmpLandmarks[118]);
	landmarks.push_back(tmpLandmarks[119]);
	loaded.push_back(true);

	//44
	landmarks.push_back(tmpLandmarks[132]);
	landmarks.push_back(tmpLandmarks[133]);
	landmarks.push_back(tmpLandmarks[134]);
	loaded.push_back(true);

	//(42)
	//landmarks.push_back(tmpLandmarks[126]);
	//landmarks.push_back(tmpLandmarks[127]);
	//landmarks.push_back(tmpLandmarks[128]);
	//loaded.push_back(true);

	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//48
	landmarks.push_back(tmpLandmarks[144]);
	landmarks.push_back(tmpLandmarks[145]);
	landmarks.push_back(tmpLandmarks[146]);
	loaded.push_back(true);

	//54
	landmarks.push_back(tmpLandmarks[162]);
	landmarks.push_back(tmpLandmarks[163]);
	landmarks.push_back(tmpLandmarks[164]);
	loaded.push_back(true);

	//51
	landmarks.push_back(tmpLandmarks[153]);
	landmarks.push_back(tmpLandmarks[154]);
	landmarks.push_back(tmpLandmarks[155]);
	loaded.push_back(true);

	//57
	landmarks.push_back(tmpLandmarks[171]);
	landmarks.push_back(tmpLandmarks[172]);
	landmarks.push_back(tmpLandmarks[173]);
	loaded.push_back(true);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	return true;
}

bool FileLoader::loadPSELandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& loaded)
{
	std::vector<double> tmpLandmarks;
	if(!loadVertexDataFile(sstrFileName, true, tmpLandmarks))
	{
		return false;
	}

	const size_t numLoadedLandmarks = tmpLandmarks.size()/3;
	if(numLoadedLandmarks != 24)
	{
		return false;
	}

	//1
	landmarks.push_back(tmpLandmarks[3]);
	landmarks.push_back(tmpLandmarks[4]);
	landmarks.push_back(tmpLandmarks[5]);
	loaded.push_back(true);

	//0
	landmarks.push_back(tmpLandmarks[0]);
	landmarks.push_back(tmpLandmarks[1]);
	landmarks.push_back(tmpLandmarks[2]);
	loaded.push_back(true);

	//2
	landmarks.push_back(tmpLandmarks[6]);
	landmarks.push_back(tmpLandmarks[7]);
	landmarks.push_back(tmpLandmarks[8]);
	loaded.push_back(true);

	//3
	landmarks.push_back(tmpLandmarks[9]);
	landmarks.push_back(tmpLandmarks[10]);
	landmarks.push_back(tmpLandmarks[11]);
	loaded.push_back(true);

	//6
	landmarks.push_back(tmpLandmarks[18]);
	landmarks.push_back(tmpLandmarks[19]);
	landmarks.push_back(tmpLandmarks[20]);
	loaded.push_back(true);

	//4
	landmarks.push_back(tmpLandmarks[12]);
	landmarks.push_back(tmpLandmarks[13]);
	landmarks.push_back(tmpLandmarks[14]);
	loaded.push_back(true);

	//5
	landmarks.push_back(tmpLandmarks[15]);
	landmarks.push_back(tmpLandmarks[16]);
	landmarks.push_back(tmpLandmarks[17]);
	loaded.push_back(true);

	//7
	landmarks.push_back(tmpLandmarks[21]);
	landmarks.push_back(tmpLandmarks[22]);
	landmarks.push_back(tmpLandmarks[23]);
	loaded.push_back(true);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	return true;
}

bool FileLoader::loadBosphorusLandmarksFull(const std::string& sstrFileName, std::vector<double>& lmks, std::vector<bool>& loadedLmks)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	const size_t maxNumLmks(22);
	lmks.clear();
	lmks.resize(3*maxNumLmks);

	loadedLmks.clear();
	loadedLmks.resize(maxNumLmks);

	for(size_t i = 0; i < maxNumLmks; ++i)
	{
		loadedLmks[i] = false;
	}

	char buffer[300];
	fscanf(pFile, "%[^\n] ", &buffer);

	int numLnd(0);
	fscanf(pFile, "%d %*s ", &numLnd);

	for(int i = 0; i < numLnd; ++i)
	{
		fscanf(pFile, "%[^\n] ", &buffer);

		float readValX(0.0f), readValY(0.0f), readValZ(0.0f);
		fscanf(pFile, "%f %f %f ", &readValX, &readValY, &readValZ);

		if(strcmp(buffer, "Outer left eyebrow") == 0) //1
		{
			lmks[0] = static_cast<double>(readValX);
			lmks[1] = static_cast<double>(readValY);
			lmks[2] = static_cast<double>(readValZ);
			loadedLmks[0] = true;
		}
		else if(strcmp(buffer, "Middle left eyebrow") == 0) //2
		{
			lmks[3] = static_cast<double>(readValX);
			lmks[4] = static_cast<double>(readValY);
			lmks[5] = static_cast<double>(readValZ);
			loadedLmks[1] = true;
		}
		else if(strcmp(buffer, "Inner left eyebrow") == 0) //3
		{
			lmks[6] = static_cast<double>(readValX);
			lmks[7] = static_cast<double>(readValY);
			lmks[8] = static_cast<double>(readValZ);
			loadedLmks[2] = true;
		}
		else if(strcmp(buffer, "Inner right eyebrow") == 0) //4
		{
			lmks[9] = static_cast<double>(readValX);
			lmks[10] = static_cast<double>(readValY);
			lmks[11] = static_cast<double>(readValZ);
			loadedLmks[3] = true;
		}
		else if(strcmp(buffer, "Middle right eyebrow") == 0) //5
		{
			lmks[12] = static_cast<double>(readValX);
			lmks[13] = static_cast<double>(readValY);
			lmks[14] = static_cast<double>(readValZ);
			loadedLmks[4] = true;
		}
		else if(strcmp(buffer, "Outer right eyebrow") == 0) //6
		{
			lmks[15] = static_cast<double>(readValX);
			lmks[16] = static_cast<double>(readValY);
			lmks[17] = static_cast<double>(readValZ);
			loadedLmks[5] = true;
		}
		else if(strcmp(buffer, "Outer left eye corner") == 0) //7
		{
			lmks[18] = static_cast<double>(readValX);
			lmks[19] = static_cast<double>(readValY);
			lmks[20] = static_cast<double>(readValZ);
			loadedLmks[6] = true;
		}
		else if(strcmp(buffer, "Inner left eye corner") == 0) //8
		{
			lmks[21] = static_cast<double>(readValX);
			lmks[22] = static_cast<double>(readValY);
			lmks[23] = static_cast<double>(readValZ);
			loadedLmks[7] = true;
		}
		else if(strcmp(buffer, "Inner right eye corner") == 0) //9
		{
			lmks[24] = static_cast<double>(readValX);
			lmks[25] = static_cast<double>(readValY);
			lmks[26] = static_cast<double>(readValZ);
			loadedLmks[8] = true;
		}
		else if(strcmp(buffer, "Outer right eye corner") == 0) //10
		{
			lmks[27] = static_cast<double>(readValX);
			lmks[28] = static_cast<double>(readValY);
			lmks[29] = static_cast<double>(readValZ);
			loadedLmks[9] = true;
		}
		else if(strcmp(buffer, "Nose saddle left") == 0) //11
		{
			lmks[30] = static_cast<double>(readValX);
			lmks[31] = static_cast<double>(readValY);
			lmks[32] = static_cast<double>(readValZ);
			loadedLmks[10] = true;
		}
		else if(strcmp(buffer, "Nose saddle right") == 0) //12
		{
			lmks[33] = static_cast<double>(readValX);
			lmks[34] = static_cast<double>(readValY);
			lmks[35] = static_cast<double>(readValZ);
			loadedLmks[11] = true;
		}
		else if(strcmp(buffer, "Left nose peak") == 0) //13
		{
			lmks[36] = static_cast<double>(readValX);
			lmks[37] = static_cast<double>(readValY);
			lmks[38] = static_cast<double>(readValZ);
			loadedLmks[12] = true;
		}
		else if(strcmp(buffer, "Nose tip") == 0) //16
		{
			lmks[39] = static_cast<double>(readValX);
			lmks[40] = static_cast<double>(readValY);
			lmks[41] = static_cast<double>(readValZ);
			loadedLmks[13] = true;
		}
		else if(strcmp(buffer, "Right nose peak") == 0) //15
		{
			lmks[42] = static_cast<double>(readValX);
			lmks[43] = static_cast<double>(readValY);
			lmks[44] = static_cast<double>(readValZ);
			loadedLmks[14] = true;
		}
		else if(strcmp(buffer, "Left mouth corner") == 0) //17
		{
			lmks[45] = static_cast<double>(readValX);
			lmks[46] = static_cast<double>(readValY);
			lmks[47] = static_cast<double>(readValZ);
			loadedLmks[15] = true;
		}
		else if(strcmp(buffer, "Upper lip outer middle") == 0) //17
		{
			lmks[48] = static_cast<double>(readValX);
			lmks[49] = static_cast<double>(readValY);
			lmks[50] = static_cast<double>(readValZ);
			loadedLmks[16] = true;
		}
		else if(strcmp(buffer, "Right mouth corner") == 0) //18
		{
			lmks[51] = static_cast<double>(readValX);
			lmks[52] = static_cast<double>(readValY);
			lmks[53] = static_cast<double>(readValZ);
			loadedLmks[17] = true;
		}
		else if(strcmp(buffer, "Upper lip inner middle") == 0) //19
		{
			lmks[54] = static_cast<double>(readValX);
			lmks[55] = static_cast<double>(readValY);
			lmks[56] = static_cast<double>(readValZ);
			loadedLmks[18] = true;
		}

		else if(strcmp(buffer, "Lower lip inner middle") == 0) //20
		{
			lmks[57] = static_cast<double>(readValX);
			lmks[58] = static_cast<double>(readValY);
			lmks[59] = static_cast<double>(readValZ);
			loadedLmks[19] = true;
		}
		else if(strcmp(buffer, "Lower lip outer middle") == 0) //21
		{
			lmks[60] = static_cast<double>(readValX);
			lmks[61] = static_cast<double>(readValY);
			lmks[62] = static_cast<double>(readValZ);
			loadedLmks[20] = true;
		}
		else if(strcmp(buffer, "Chin middle") == 0) //22
		{
			lmks[63] = static_cast<double>(readValX);
			lmks[64] = static_cast<double>(readValY);
			lmks[65] = static_cast<double>(readValZ);
			loadedLmks[21] = true;
		}
	}

	return fclose(pFile)==0;
}

bool FileLoader::loadThinPlateSpline(const std::string& sstrFileName, std::vector<double>& vecC, std::vector<double>& matA, std::vector<double>& matW, std::vector<double>& sourcePoints)
{
	if(sstrFileName.empty())
	{
		return false;
	}

	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	char strOutput[1000];
	size_t sourceDim(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, sourceDim))
	{
		return false;
	}

	size_t targetDim(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, targetDim))
	{
		return false;
	}

	size_t numSourcePoints(0);
	if(!readNextNumber(pFile, INTEGER, strOutput, numSourcePoints))
	{
		return false;
	}

	vecC.reserve(targetDim);
	matA.reserve(targetDim*sourceDim);
	matW.reserve(numSourcePoints*targetDim);
	sourcePoints.reserve(numSourcePoints*sourceDim);

	bool bEnd(false);
	while(!bEnd)
	{
		while(!bEnd 
			&& !isEqual(strOutput, "c") 
			&& !isEqual(strOutput, "A")
			&& !isEqual(strOutput, "W")
			&& !isEqual(strOutput, "points"))
		{
			bEnd = readNextNode(pFile, strOutput);
		}

		if(!bEnd && isEqual(strOutput, "c"))
		{
			processBlockStructure(pFile, strOutput, vecC);
		}

		if(!bEnd && isEqual(strOutput, "A"))
		{
			std::vector<double> tmpAT;
			processBlockStructure(pFile, strOutput, tmpAT);

			if(tmpAT.size() != targetDim*sourceDim)
			{
				return false;
			}

			matA.resize(targetDim*sourceDim);
			for(size_t i = 0; i < targetDim; ++i)
			{
				for(size_t j = 0; j < sourceDim; ++j)
				{
					matA[j*targetDim+i] = tmpAT[i*sourceDim+j];
				}
			}
		}

		if(!bEnd && isEqual(strOutput, "W"))
		{
			std::vector<double> tmpWT;
			processBlockStructure(pFile, strOutput, tmpWT);

			if(tmpWT.size() != numSourcePoints*targetDim)
			{
				return false;
			}

			matW.resize(numSourcePoints*targetDim);
			for(size_t i = 0; i < numSourcePoints; ++i)
			{
				for(size_t j = 0; j < targetDim; ++j)
				{
					matW[j*numSourcePoints+i] = tmpWT[i*targetDim+j];
				}
			}
		}

		if(!bEnd && isEqual(strOutput, "points"))
		{
			processBlockStructure(pFile, strOutput, sourcePoints);
		}
	}

	bool bSuccess = (vecC.size() == targetDim) && (matA.size() == targetDim*sourceDim) && (matW.size() == numSourcePoints*targetDim) && (sourcePoints.size() == numSourcePoints*sourceDim);
	return fclose(pFile)==0 && bSuccess;
}

bool FileLoader::loadSimpleLandmarks(const std::string& sstrFileName, std::vector<double>& landmarks, std::vector<bool>& loaded)
{
	std::vector<double> tmpLandmarks;
	if(!loadVertexDataFile(sstrFileName, false, tmpLandmarks))
	{
		return false;
	}

	if(tmpLandmarks.size() != 60)
	{
		return false;
	}

	//1
	landmarks.push_back(tmpLandmarks[3]);
	landmarks.push_back(tmpLandmarks[4]);
	landmarks.push_back(tmpLandmarks[5]);
	loaded.push_back(true);

	//0
	landmarks.push_back(tmpLandmarks[0]);
	landmarks.push_back(tmpLandmarks[1]);
	landmarks.push_back(tmpLandmarks[2]);
	loaded.push_back(true);

	//2
	landmarks.push_back(tmpLandmarks[6]);
	landmarks.push_back(tmpLandmarks[7]);
	landmarks.push_back(tmpLandmarks[8]);
	loaded.push_back(true);

	//3
	landmarks.push_back(tmpLandmarks[9]);
	landmarks.push_back(tmpLandmarks[10]);
	landmarks.push_back(tmpLandmarks[11]);
	loaded.push_back(true);

	//6
	landmarks.push_back(tmpLandmarks[18]);
	landmarks.push_back(tmpLandmarks[19]);
	landmarks.push_back(tmpLandmarks[20]);
	loaded.push_back(true);

	//4
	landmarks.push_back(tmpLandmarks[12]);
	landmarks.push_back(tmpLandmarks[13]);
	landmarks.push_back(tmpLandmarks[14]);
	loaded.push_back(true);

	//5
	landmarks.push_back(tmpLandmarks[15]);
	landmarks.push_back(tmpLandmarks[16]);
	landmarks.push_back(tmpLandmarks[17]);
	loaded.push_back(true);

	//7
	landmarks.push_back(tmpLandmarks[21]);
	landmarks.push_back(tmpLandmarks[22]);
	landmarks.push_back(tmpLandmarks[23]);
	loaded.push_back(true);

	//8
	landmarks.push_back(tmpLandmarks[24]);
	landmarks.push_back(tmpLandmarks[25]);
	landmarks.push_back(tmpLandmarks[26]);
	loaded.push_back(true);

	//9
	landmarks.push_back(tmpLandmarks[27]);
	landmarks.push_back(tmpLandmarks[28]);
	landmarks.push_back(tmpLandmarks[29]);
	loaded.push_back(true);

	//10
	landmarks.push_back(tmpLandmarks[30]);
	landmarks.push_back(tmpLandmarks[31]);
	landmarks.push_back(tmpLandmarks[32]);
	loaded.push_back(true);

	//11
	landmarks.push_back(tmpLandmarks[33]);
	landmarks.push_back(tmpLandmarks[34]);
	landmarks.push_back(tmpLandmarks[35]);
	loaded.push_back(true);

	//-
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	landmarks.push_back(0.0);
	loaded.push_back(false);

	return true;
}

bool FileLoader::loadWeights(const std::string& sstrFileName, std::vector<double>& weights, size_t& dim, size_t& numVecs)
{
	const char* cstrFileName = sstrFileName.c_str();
	if(cstrFileName==NULL)
	{
		return false;
	}

	FILE* pFile = fopen(cstrFileName, "r");
	if(pFile==NULL)
	{
		return false;
	}

	dim = 0;
	numVecs = 0;
	fscanf(pFile, "%d %d ", &numVecs, &dim);

	for(int i = 0; i < numVecs; ++i)
	{
		for(int j = 0; j < dim; ++j)
		{
			double weightVal(0.0);
			int ret = fscanf(pFile, "%lf ", &weightVal);
			if(ret == EOF)
			{
				break;
			}

			weights.push_back(weightVal);
		}
	}

	return fclose(pFile)==0 && static_cast<int>(weights.size()) == numVecs*dim;
}

bool FileLoader::readNextNode(FILE* pFile, char* cstrOutput)
{
	memset(cstrOutput, 0, 1000);

	//Read next word till end of the line, space, or any bracket
	//Ignore comments marked by # at the beginning of the line
	size_t currSize(0);
	while(currSize < 1000) //One nod is probably not longer than 1000 characters
	{
		int currChar = fgetc(pFile);

		if(currChar == EOF)
		{
			return true;
		}
	  
		// #
		if(currChar == 35)
		{
			//Ignore rest of the line
			// 10 = new line
			do
			{
				currChar = fgetc(pFile);
			}
			while(currChar != EOF && currChar != 10);

			//If we already read something stop here
			if(currSize > 0)
			{
				return currChar == EOF;
			}
		}

		//If last element was a bracket treat it special
		//  91 = [		92 = ]
		// 123 = {    125 = }
		if(currChar == 91 || currChar == 92 || currChar == 123 || currChar == 125)
		{
			if(currSize == 0)
			{
				//Return the bracket if it is the first character
				cstrOutput[currSize] = (char)currChar;
				++currSize;

				return false;
			}
			else
			{
				//Push back bracket to the stream if it is not the first character
				currChar = ungetc(currChar, pFile);
				if(currChar == EOF)
				{
					//Failed pushing bracket back into stream
					std::cout << "Failed pushing back bracket into stream" << std::endl;
					return true;
				}
				else
				{
					return false;
				}
			}
		}
		
		//Special handling for space, tab and new line 
		// ( 32 = space, 9 = horizontal tab, 11 = vertical tab, 10 = new line, 12 = new page)
		// if first character: continue
		// if not first character: stop reading
		//if(currChar == 32 || currChar = 9 || currChar = 11 || currChar == 10 || currChar == 12)
		if(currChar >= 0 && currChar <= 32)
		{
			if(currSize == 0)
			{
				continue;
			}
			else
			{
				return false;
			}
		}

		cstrOutput[currSize] = (char)currChar;
		++currSize;
	}

	return false;
}

bool FileLoader::processShapeNode(FILE* pFile, char* cstrOutput, DataContainer& outPoly)
{
	bool bEnd(false);
	while(!bEnd)
	{
		bEnd = readNextNode(pFile, cstrOutput);

		if(bEnd)
		{
			break;
		}

		if(!bEnd && isEqual(cstrOutput, "ImageTexture"))
		{
			char cstrTextureName[1000] = "";
			bEnd &= processImageTexture(pFile, cstrOutput, cstrTextureName);
			
			std::string sstrTextureName(cstrTextureName);
			if(!sstrTextureName.empty())
			{
				outPoly.setTextureName(sstrTextureName);
			}
		}
		
		if(!bEnd && isEqual(cstrOutput, "coord"))
		{
			std::vector<double> vertexList;
			processCoordinates(pFile, cstrOutput, 3, vertexList);
			outPoly.setVertexList(vertexList);
		}
		
		if(!bEnd && isEqual(cstrOutput, "coordIndex"))
		{
			std::vector<std::vector<int>> vertexIndexList;
			processIndices(pFile, cstrOutput, vertexIndexList);
			outPoly.setVertexIndexList(vertexIndexList);
		}
		
		if(!bEnd && isEqual(cstrOutput, "TextureCoordinate"))
		{
			std::vector<double> textureList;
			processCoordinates(pFile, cstrOutput, 2, textureList);
			outPoly.setTextureList(textureList);
		}
		
		if(!bEnd && isEqual(cstrOutput, "texCoordIndex"))
		{
			std::vector<std::vector<int>> textureIndexList;
			processIndices(pFile, cstrOutput, textureIndexList);
			outPoly.setTextureIndexList(textureIndexList);
		}
	}

	return bEnd;
}

bool FileLoader::processImageTexture(FILE* pFile, char* cstrOutput, char* cstrTextureName)
{
	bool bEnd = readNodeBlocksUntil(pFile, "url", cstrOutput);

	assert(!bEnd);
	assert(!isEqual(cstrOutput, "}"));
	assert(isEqual(cstrOutput, "url"));

	bEnd = readNextNode(pFile, cstrOutput);
	
	if(isEqual(cstrOutput, "["))
	{
		bEnd = readNextNode(pFile, cstrOutput);
	}

	const size_t length = strlen(cstrOutput); //sizeof(cstrOutput)/sizeof(cstrOutput[0]);

	// Remove /, \ and " from the beginning of the name
	int nFirstLetter(0);
	for(size_t i = 0; i < length; ++i)
	{
		if(cstrOutput[i] != '\\' && cstrOutput[i] != '/'  && cstrOutput[i] != '\"')
		{
			break;
		}

		++nFirstLetter;
	}

	int nLastLetter = static_cast<int>(length)-1;
	for(size_t i = 0; i < length; ++i)
	{
		if(cstrOutput[length-i-1] != '\\' && cstrOutput[length-i-1] != '/'  && cstrOutput[length-i-1] != '\"')
		{
			break;
		}

		--nLastLetter;
	}

	strncpy(cstrTextureName, cstrOutput+nFirstLetter, nLastLetter-nFirstLetter+1);
	return bEnd;
}

void FileLoader::processBlockStructure(FILE* pFile, char* strOutput, std::vector<double>& values)
{
	bool bEnd = readNodeBlocksUntil(pFile, "{", strOutput);;
	while(!bEnd)
	{
		double number(0.0);
		if(!readNextNumber(pFile, FLOATING_POINT, strOutput, number))
		{
			break;
		}

		values.push_back(number);
	}
}