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

#include "MathHelper.h"

//#include "KDTree3.h"

#include <vector>
#include <map>
#include <set>

#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#include <sstream>

namespace clapack
{
	extern "C"
	{
		#include "blaswrap.h"
		#include "f2c.h"
		extern int dgemm_(char *transa, char *transb, integer *m, integer *n
								, integer *k, doublereal *alpha, doublereal *a, integer *lda
								, doublereal *b, integer *ldb, doublereal *beta, doublereal *c
								, integer *ldc);
		extern int dgels_(char *trans, integer *m, integer *n, integer *nrhs, doublereal *a
								, integer *lda, doublereal *b, integer *ldb, doublereal *work
								, integer *lwork, integer *info);
		extern int dgesdd_(char *jobz, integer *m, integer *n, doublereal *a
								, integer *lda, doublereal *s, doublereal *u, integer *ldu
								, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork
								, integer *iwork, integer *info);
		extern int dgetrf_(integer *m, integer *n, doublereal *a, integer *lda
								, integer *ipiv, integer *info);
		extern int dgetri_(integer *n, doublereal *a, integer *lda, integer *ipiv
								, doublereal *work, integer *lwork, integer *info);	
	}
}

bool MathHelper::matSet(std::vector<double>& matrix, const size_t numRows, const size_t row, const size_t col, const double value)
{
	//const size_t numCol = matrix.size()/numRows;
	//if(row >= numRows || col >= numCol)
	//{
	//	return false;
	//}

	matrix[row+col*numRows] = value;
	return true;
}

double MathHelper::matGet(const std::vector<double>& matrix, const size_t numRows, const size_t row, const size_t col)
{
	return matrix[row+col*numRows];
}

int MathHelper::getRandomInteger(const int min, const int max)
{
	//srand (time(NULL));
	return rand()%(max-min+1)+min;
}

double MathHelper::getRandomDouble(const double min, const double max, const bool bReachMax)
{
	double delta = 0.0;
	if(bReachMax)
	{
		delta = static_cast<double>(rand())/RAND_MAX;
	}
	else
	{
		delta = static_cast<double>(rand())/(RAND_MAX+1.0);
	}

	return min + (max-min)*delta;
}

double MathHelper::getGaussianRandomVariable(const double mean, const double variance)
{
	//Box-Muller transform

	////Random variables in the interval [0,1]
	//double rand1 = rand()/((double)RAND_MAX);
	//double rand2 = rand()/((double)RAND_MAX);

	//if(rand1 < DBL_MIN)
	//{
	//	rand1 = DBL_MIN;
	//}

	////Standard normal distributed variable with mean 0 and variance 1 (X ~ N(0,1))
	//const double standardNormalVariable = sqrt(-2.0*log(rand1))*cos(2.0*M_PI*rand2);
	//
	//// X ~ N(0,1) -> sigma*X + mean ~ N(mean, sigma^2)
	//return mean+sqrt(variance)*standardNormalVariable;

	double standardNormalVariable = 0.0;
	//By central limit theorem use 48 = 12 * 4 trials of uniform random variable:
	for(int i = 0; i < 48; i++)
	{
		//add a random number in (0, 1]
		standardNormalVariable += ((double)rand()/((double)(RAND_MAX)+(double)(1)));
	}
	//the variable result is now of a Gaussian distribution (24, 4).
	standardNormalVariable /= 2.0;
	//the variable result is now of a Gaussian distribution (12, 1).
	standardNormalVariable -= 12.0;
	//the variable result is now of a Gaussian distribution (0, 1), as wanted.

	return mean+sqrt(variance)*standardNormalVariable;
}

void MathHelper::getRandomlyPermutedInteger(const int min, const int max, std::vector<int>& permutedInteger)
{
	std::map<double, int> randomMap;

	for(int iVal = min; iVal <= max; ++iVal)
	{
		//Random value between 0 and 1
		const double randValue = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
		if(randomMap.find(randValue) == randomMap.end())
		{
			randomMap.insert(std::make_pair(randValue, iVal));
		}
		else
		{
			--iVal;
		}
	}

	std::map<double, int>::const_iterator currIter = randomMap.begin();
	std::map<double, int>::const_iterator endIter = randomMap.end();
	for(; currIter != endIter; ++currIter)
	{
		permutedInteger.push_back(currIter->second);
	}
}

int MathHelper::triangleObtuse(const Vec3d& A, const Vec3d& B, const Vec3d& C)
{
	const Vec3d a = C-B;
	const Vec3d b = A-C;
	const Vec3d c = B-A;
	const double aSqrLength = a.sqrLength();
	const double bSqrLength = b.sqrLength();
	const double cSqrLength = c.sqrLength();

	if(cSqrLength > bSqrLength)
	{
		if(cSqrLength > aSqrLength)
		{
			// cSqrLenght > bSqrLength && cSqrLength > aSqrLength
			if(aSqrLength+bSqrLength < cSqrLength)
			{
				return 2;
			}
		}
		else 
		{
			//aSqrLength >= cSqrLength > bSqrLength
			if(bSqrLength+cSqrLength < aSqrLength)
			{
				return 0;
			}
		}
	}
	else //cSqrLength <= bSqrLength
	{
		if(bSqrLength > aSqrLength)
		{
			//bSqrLength  >= cSqrLength && bSqrLength > aSqrLength
			if(aSqrLength+cSqrLength < bSqrLength)
			{
				return 1;
			}
		}
		else
		{
			//aSqrLength >= bSqrLength >= cSqrLength 
			if(bSqrLength+cSqrLength < aSqrLength)
			{
				return 0;
			}
		}
	}

	return -1;
}

void MathHelper::computeVoronoiAreas(const Vec3d& A, const Vec3d& B, const Vec3d& C, double& areaA, double& areaB, double& areaC)
{
	areaA = 0.0;
	areaB = 0.0;
	areaC = 0.0;

	const Vec3d vecAB = B-A;
	const Vec3d vecAC = C-A;
	const double area = 0.5*(vecAB.crossProduct(vecAC)).length();
	if(area < pow(math_eps, 2))
	{
		return;
	}

	int obtuseVertex = triangleObtuse(A, B, C);
	if(obtuseVertex == -1)
	{
		// cotAlpha = <AB, AC> / |AB x AC|
		const double cotAlpha = vecAB.dotProduct(vecAC) / (2.0*area);
		
		// cotBeta = <BA, BC> / |BA x BC|
		const Vec3d vecBA = A-B;
		const Vec3d vecBC = C-B;
		const double cotBeta = vecBC.dotProduct(vecBA) / (2.0*area);
		
		// cotGamma = <CA, CB> / |CA x CB|
		const Vec3d vecCA = A-C;
		const Vec3d vecCB = B-C;
		const double cotGamma = vecCA.dotProduct(vecCB) / (2.0*area);

		const double sqrABLength = vecAB.sqrLength();
		const double sqrBCLength = vecBC.sqrLength();
		const double sqrCALength = vecCA.sqrLength();

		const double factor = 1.0/8.0;
		areaA = factor*(sqrABLength*cotGamma + sqrCALength*cotBeta);
		areaB = factor*(sqrBCLength*cotAlpha + sqrABLength*cotGamma);
		areaC = factor*(sqrCALength*cotBeta + sqrBCLength*cotAlpha);
	}
	else
	{
		const double halfArea = 0.5*area;
		const double quaterArea = 0.25*area;

		areaA = quaterArea;
		areaB = quaterArea;
		areaC = quaterArea;

		if(obtuseVertex == 0)
		{
			areaA = halfArea;
		}
		else if(obtuseVertex == 1)
		{
			areaB = halfArea;
		}
		else if(obtuseVertex == 2)
		{
			areaC = halfArea;
		}
	}
}

//Compute normals, based on Max1999 - Weights for Computing Vertex Normals from Facet Normals
void MathHelper::computeVertexNormals(const DataContainer& poly, std::vector<double>& vertexNormals)
{
	const std::vector<double>& vertexList = poly.getVertexList();
	const std::vector<std::vector<int>>& vertexIndexList = poly.getVertexIndexList();

	std::map<int, std::vector<size_t>> polygonsForVertexIndexMap;
	{
		for(size_t i = 0; i < vertexIndexList.size(); ++i)
		{
			const std::vector<int>& currPolygonIndices = vertexIndexList[i];
			for(size_t j = 0; j < currPolygonIndices.size(); ++j)
			{
				const int currIndex = currPolygonIndices[j];

				std::map<int, std::vector<size_t>>::iterator mapIter = polygonsForVertexIndexMap.find(currIndex);
				if(mapIter!=polygonsForVertexIndexMap.end())
				{
					std::vector<size_t>& vec = mapIter->second;
					vec.push_back(i);
				}
				else
				{
					std::vector<size_t> vec;
					vec.push_back(i);
					polygonsForVertexIndexMap.insert(std::make_pair(currIndex, vec));
				}
			}
		}
	}

	const size_t vertexNormalDim = vertexList.size();
	vertexNormals.resize(vertexNormalDim, 0.0);
	
	std::map<int, std::vector<size_t>>::const_iterator currPointPolyIter = polygonsForVertexIndexMap.begin();
	std::map<int, std::vector<size_t>>::const_iterator endPointPolyIter = polygonsForVertexIndexMap.end();
	for(; currPointPolyIter != endPointPolyIter; ++currPointPolyIter)
	{
		const int currIndex = currPointPolyIter->first;
		const std::vector<size_t>& neighborPolygons = currPointPolyIter->second;

		const Vec3d currVertex(vertexList[3*currIndex], vertexList[3*currIndex+1], vertexList[3*currIndex+2]);
		Vec3d vertexNormal(0.0, 0.0, 0.0);

		for(size_t i = 0; i < neighborPolygons.size(); ++i)
		{
			const std::vector<int>& currPoly = vertexIndexList[neighborPolygons[i]];

			const size_t currPolySize = currPoly.size();

			int tmpPos(-1);
			for(size_t j = 0; j < currPolySize; ++j)
			{
				tmpPos = currPoly[j] == currIndex ? static_cast<int>(j) : tmpPos;
			}

			if(tmpPos == -1 || currPoly[tmpPos] != currIndex)
			{
#ifdef DEBUG_OUTPUT
				std::cout << "Wrong neighboring index while computing normal vector" << std::endl;
#endif
				continue;
			}

			const int prevIndex = currPoly[(tmpPos+(currPolySize-1))%currPolySize];
			const int nextIndex = currPoly[(tmpPos+1)%currPolySize];

			const Vec3d prevVertex(vertexList[3*prevIndex], vertexList[3*prevIndex+1], vertexList[3*prevIndex+2]);
			const Vec3d nextVertex(vertexList[3*nextIndex], vertexList[3*nextIndex+1], vertexList[3*nextIndex+2]);

			const Vec3d v1 = nextVertex-currVertex;
			const Vec3d v2 = prevVertex-currVertex;

			const double v1SqrLength = v1.sqrLength();
			const double v2SqrtLength = v2.sqrLength();

			if(v1SqrLength < DBL_EPSILON || v2SqrtLength < DBL_EPSILON)
			{
#ifdef DEBUG_OUTPUT
				std::cout << "Zero length edge while computing normal vector" << std::endl;
#endif
				continue;
			}

			const double factor = 1.0/(v1SqrLength*v2SqrtLength);

			Vec3d v1xv2;
			v1.crossProduct(v2, v1xv2);
			vertexNormal += v1xv2*factor;
		}
		
		if(!vertexNormal.normalize())
		{
#ifdef DEBUG_OUTPUT
			std::cout << "Zero length normal vector while computing normal vector" << std::endl;
#endif
			continue;
		}
		
		vertexNormals[3*currIndex] = vertexNormal[0];
		vertexNormals[3*currIndex+1] = vertexNormal[1];
		vertexNormals[3*currIndex+2] = vertexNormal[2];
	}
}

void MathHelper::computeVertexAreas(const DataContainer& mesh, std::vector<double>& vertexAreas, double& meshArea)
{
	const std::vector<double>& vertices = mesh.getVertexList();
	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();

	const size_t numVertices = mesh.getNumVertices();
	vertexAreas.resize(numVertices, 0.0);
	meshArea = 0.0;

	for(size_t i = 0; i < vertexIndexList.size(); ++i)
	{
		const std::vector<int>& currPolygonIndices = vertexIndexList[i];
		
		const int indexA = currPolygonIndices[0];
		const int aStartIndex = 3*indexA;
		const Vec3d vertexA(vertices[aStartIndex], vertices[aStartIndex+1], vertices[aStartIndex+2]);
		
		const int indexB = currPolygonIndices[1];
		const int bStartIndex = 3*indexB;
		const Vec3d vertexB(vertices[bStartIndex], vertices[bStartIndex+1], vertices[bStartIndex+2]);

		const int indexC = currPolygonIndices[2];
		const int cStartIndex = 3*indexC;
		const Vec3d vertexC(vertices[cStartIndex], vertices[cStartIndex+1], vertices[cStartIndex+2]);

		double areaA(0.0);
		double areaB(0.0);
		double areaC(0.0);
		MathHelper::computeVoronoiAreas(vertexA, vertexB, vertexC, areaA, areaB, areaC);

		vertexAreas[indexA] += areaA;
		vertexAreas[indexB] += areaB;
		vertexAreas[indexC] += areaC;

		meshArea += (areaA + areaB + areaC);
	}
}

void MathHelper::computeBoundaryVertices(const DataContainer& mesh, std::vector<size_t>& boundaryVertexIndices)
{
	const size_t numVertices = mesh.getNumVertices();

	std::vector<size_t> pointConnections;
	pointConnections.resize(numVertices*numVertices, 0);

	const std::vector<std::vector<int>>& vertexIndices = mesh.getVertexIndexList();
	for(size_t iTriangle = 0; iTriangle < vertexIndices.size(); ++iTriangle)
	{
		const std::vector<int>& currTriangle = vertexIndices[iTriangle];
		const int i1 = currTriangle[0];
		const int i2 = currTriangle[1];
		const int i3 = currTriangle[2];

		//i1i2
		pointConnections[i1*numVertices+i2] += 1;
		pointConnections[i2*numVertices+i1] += 1;

		//i2i3
		pointConnections[i2*numVertices+i3] += 1;
		pointConnections[i3*numVertices+i2] += 1;

		//i3i1
		pointConnections[i3*numVertices+i1] += 1;
		pointConnections[i1*numVertices+i3] += 1;
	}

	std::set<size_t> tmpBoundaryVertices;

	for(size_t i = 0; i < numVertices; ++i)
	{
		for(size_t j = i; j < numVertices; ++j)
		{
			const size_t index = i*numVertices + j;
			if(pointConnections[index] == 0)
			{
				//Points are not connected
				continue;
			}
			else if(pointConnections[index] == 1)
			{
				//Points are only connected by one edge -> bondary vertices
				tmpBoundaryVertices.insert(i);
				tmpBoundaryVertices.insert(j);
			}
			else if(pointConnections[index] == 2)
			{
				//Points are connected by two edges
				continue;
			}
			else
			{
				//Should never happen
				std::cout << "findBoundaryVertices error" << std::endl;
				continue;
			}
		}
	}

	std::set<size_t>::const_iterator currIter = tmpBoundaryVertices.begin();
	std::set<size_t>::const_iterator endIter = tmpBoundaryVertices.end();
	for(; currIter != endIter; ++currIter)
	{
		boundaryVertexIndices.push_back(*currIter);
	}
}

void MathHelper::getPlaneProjection(const Vec3d& p1, const Vec3d& p2, const Vec3d& n2, Vec3d& outPoint)
{
	//given: direction v, normal n
	//searched: projected direction w
	// w = v - <n,v>n

	Vec3d diff = p2-p1;
	double d = diff.dotProduct(n2);

	outPoint = n2;
	outPoint.scalarMult(d);

	outPoint = p1 + outPoint;
}

bool MathHelper::pointInPlane(const Vec3d& pPlane, const Vec3d& nPlane, const Vec3d& p)
{
	const Vec3d diff = p-pPlane;
	return fabs(diff.dotProduct(nPlane)) < DBL_EPSILON;
}

bool MathHelper::getBarycentricCoords(const Vec3d& p1, const Vec3d& v1, const Vec3d& v2, const Vec3d& p, double& u, double& v)
{
	//const Vec3d pn = v1.crossProduct(v2);

	//if(!MathHelper::pointInPlane(p1, pn, p))
	//{
	//	return false;
	//}

	const Vec3d vp = p-p1;
	const double vpv1 = vp.dotProduct(v1);
	const double vpv2 = vp.dotProduct(v2);
	const double v1v1 = v1.sqrLength();
	const double v2v2 = v2.sqrLength();
	const double v1v2 = v1.dotProduct(v2);

	v = (vpv2*v1v1-vpv1*v1v2)/(v2v2*v1v1-v1v2*v1v2);
	u = (vpv1-v*v1v2)/v1v1;
	return true;
}

bool MathHelper::pointInTriangle(const Vec3d& p1, const Vec3d& p2, const Vec3d& p3, const Vec3d& p)
{
	const Vec3d v1 = p2-p1;
	const Vec3d v2 = p3-p1;

	const Vec3d pn = v1.crossProduct(v2);

	if(!MathHelper::pointInPlane(p1, pn, p))
	{
		return false;
	}

	double u(0.0);
	double v(0.0);
	if(!getBarycentricCoords(p1, v1, v2, p, u, v))
	{
		return false;
	}

	return (u>=0.0 && v>=0.0 && (u+v <= 1.0));
}

void MathHelper::initTransformation(double& s, std::vector<double>& R, std::vector<double>& t)
{
	s = 1.0;

	R.clear();
	R.reserve(9);

	for(size_t i = 0; i < 9; ++i)
	{
		R.push_back(i%4 == 0 ? 1.0 : 0.0);
	}

	t.clear();
	t.reserve(3);

	for(size_t i = 0; i < 3; ++i)
	{
		t.push_back(0.0);
	}
}

void MathHelper::invertTransformation(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp
									, double& invs, std::vector<double>& invR, std::vector<double>& invt)
{
	const std::string sstrTmpTransOp = sstrTransOp == "+" ? "-" : "+";
	const std::string sstrTmpRotOp = sstrRotOp == "N" ? "T" : "N";

	MathHelper::initTransformation(invs, invR, invt);

	// s* = 1/s
	invs *= 1/s;
	
	// R* = R^T
	MathHelper::rotateData(R, sstrTmpRotOp, invR);

	// t* = - 1/s * R^T * t
	MathHelper::translateData(t, sstrTmpTransOp, invt);
	MathHelper::rotateData(R, sstrTmpRotOp, invt);
	MathHelper::scaleData(1/s, invt);
}

void MathHelper::transformMesh(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp, DataContainer& mesh)
{
	std::vector<double> vertexList = mesh.getVertexList();
	MathHelper::transformData(s, R, sstrRotOp, t, sstrTransOp, vertexList);

	mesh.setVertexList(vertexList);
}

void MathHelper::transformData(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp, std::vector<double>& data)
{
	MathHelper::rotateData(R, sstrRotOp, data);
	MathHelper::scaleData(s, data);
	MathHelper::translateData(t, sstrTransOp, data);
}

void MathHelper::transformThinPlateSpline(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp
														, std::vector<double>& vecC, std::vector<double>& matA, std::vector<double>& matW)
{
	// c' = sRc + t
	transformData(s, R, sstrRotOp, t, sstrTransOp, vecC);

	// A' = sRA
	MathHelper::rotateData(R, sstrRotOp, matA);
	MathHelper::scaleData(s, matA);

	// (W')^T = sRW^T
	std::vector<double> RT;
	if(sstrRotOp == "N")
	{
		transposeMatrix(R, 3, 3, RT);
	}
	else
	{
		RT = R;
	}

	const size_t numSamples = matW.size()/3;

	std::vector<double> tmpMatW;
	MathHelper::matrixMult(matW, numSamples, 3, "N", RT, 3, tmpMatW);
	MathHelper::scaleData(s, tmpMatW);

	matW = tmpMatW;
}

bool MathHelper::alignData(std::vector<double>& source, const std::vector<double>& target)
{
	const size_t dim = target.size();
	if(source.size() % dim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for not correct " << source.size() % dim << " != " << 0 << std::endl;
#endif
		return false;
	}

	std::vector<double> currSample;
	currSample.resize(dim);

	const size_t numSamples = source.size()/dim;
	for(size_t i = 0; i < numSamples; ++i)
	{
		const size_t startIndex = i*dim;

		for(size_t j = 0; j < dim; ++j)
		{
			currSample[j] = source[startIndex+j];
		}

		double s(0.0);
		std::vector<double> R; 
		std::vector<double> t;
		if(!MathHelper::computeAlignmentTrafo(currSample, target, s, R, t, false))
		{
			std::cout << "Unable to refine alignment of sample " << i << std::endl;
			continue;
		}

		MathHelper::transformData(s, R, "N", t, "+", currSample);

		for(size_t j = 0; j < dim; ++j)
		{
			source[startIndex+j] = currSample[j];
		}
	}

	return true;
}

bool MathHelper::alignData(std::vector<double>& source, const std::vector<double>& target, const std::vector<size_t>& vertexIndices)
{
	const size_t dim = target.size();
	if(source.size() % dim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for not correct " << source.size() % dim << " != " << 0 << std::endl;
#endif
		return false;
	}

	const size_t numReducedIndices = vertexIndices.size();
	const size_t reducedDim = 3*numReducedIndices;

	std::vector<double> currSample;
	currSample.resize(dim);

	std::vector<double> currReducedSample;
	currReducedSample.resize(reducedDim);

	std::vector<double> reducedTarget;
	reducedTarget.resize(reducedDim, 0.0);

	for(size_t i = 0; i < numReducedIndices; ++i)
	{
		const size_t currId = vertexIndices[i];
		
		reducedTarget[3*i] = target[3*currId];
		reducedTarget[3*i+1] = target[3*currId+1];
		reducedTarget[3*i+2] = target[3*currId+2];
	}

	const size_t numSamples = source.size()/dim;
	for(size_t i = 0; i < numSamples; ++i)
	{
		const size_t startIndex = i*dim;

		for(size_t j = 0; j < numReducedIndices; ++j)
		{
			const size_t currId = vertexIndices[j];
		
			currReducedSample[3*j] = source[startIndex+3*currId];
			currReducedSample[3*j+1] = source[startIndex+3*currId+1];
			currReducedSample[3*j+2] = source[startIndex+3*currId+2];
		}

		double s(0.0);
		std::vector<double> R; 
		std::vector<double> t;
		if(!MathHelper::computeAlignmentTrafo(currReducedSample, reducedTarget, s, R, t, false))
		{
			std::cout << "Unable to refine alignment of sample " << i << std::endl;
			continue;
		}

		for(size_t j = 0; j < dim; ++j)
		{
			currSample[j] = source[startIndex+j];
		}

		MathHelper::transformData(s, R, "N", t, "+", currSample);

		for(size_t j = 0; j < dim; ++j)
		{
			source[startIndex+j] = currSample[j];
		}
	}

	return true;
}

bool MathHelper::computeRigidLandmarkAlignment(const std::vector<double>& lmks1, const std::vector<bool>& lmks1Loaded, const std::vector<double>& lmks2, const std::vector<bool>& lmks2Loaded
															, double& s, std::vector<double>& R, std::vector<double>& t, bool bScaling)
{
	if(lmks1.size() != lmks2.size())
	{
		std::cout << "MathHelper::computeRigidLandmarkAlignment(...) - Number of landmarks does not match " << lmks1.size() << " != " << lmks2.size() << std::endl;
		return false;
	}

	std::vector<double> modelLandmarks;
	std::vector<double> targetLandmarks;

	//Use the first 8 landmarks for rigid alignment
	for(size_t i = 0; i < 8; ++i)
	{
		if(lmks1Loaded[i] && lmks2Loaded[i])
		{
			for(size_t j = 0; j < 3; ++j)
			{
				modelLandmarks.push_back(lmks1[3*i+j]);
				targetLandmarks.push_back(lmks2[3*i+j]);
			}
		}
	}

	return MathHelper::computeAlignmentTrafo(targetLandmarks, modelLandmarks, s, R, t, bScaling);
}

bool MathHelper::computeRigidProcrustesAlignment(std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& procrustesMean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return false;
	}

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Calc mean
		MathHelper::computeMean(data, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		MathHelper::alignData(data, procrustesMean);
	}

	// Calc mean
	MathHelper::computeMean(data, dataDim, procrustesMean);

	return true;
}

bool MathHelper::computeRigidProcrustesAlignment(const std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& alignedData, std::vector<double>& procrustesMean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return false;
	}

	alignedData = data;

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Calc mean
		MathHelper::computeMean(alignedData, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		MathHelper::alignData(alignedData, procrustesMean);
	}

	// Calc mean
	MathHelper::computeMean(alignedData, dataDim, procrustesMean);

	return true;
}

bool MathHelper::computeRigidProcrustesAlignment(const std::vector<double>& data, const std::vector<size_t>& vertexIndices, const size_t dataDim, const size_t numIter, std::vector<double>& alignedData, std::vector<double>& procrustesMean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return false;
	}

	alignedData = data;

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Calc mean
		MathHelper::computeMean(alignedData, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		MathHelper::alignData(alignedData, procrustesMean, vertexIndices);
	}

	// Calc mean
	MathHelper::computeMean(alignedData, dataDim, procrustesMean);

	return true;
}

bool MathHelper::computeAlignmentTrafo(const std::vector<double>& source, const std::vector<double>& target, double& s, std::vector<double>& R, std::vector<double>& t, bool bScaling)
{
	std::vector<double> tmpSourceData = source;
	std::vector<double> tmpTargetData = target;

	std::vector<double> sourceMean;
	MathHelper::centerData(tmpSourceData, sourceMean);

	std::vector<double> targetMean;
	MathHelper::centerData(tmpTargetData, targetMean);
	
	s = 1.0;

	if(bScaling)
	{
		if(!MathHelper::computeScaling(tmpSourceData, tmpTargetData, s))
		{
			return false;
		}

		MathHelper::scaleData(s, tmpSourceData);
	}

	if(!MathHelper::computeRotationAlignmentMatrix(tmpSourceData, tmpTargetData, R))
	{
		return false;
	}

	// t = -s*R*targetMean + sourceMean
	t.clear();
	t = sourceMean;

	MathHelper::transformData(-s, R, "N", targetMean, "+", t);

	return true;
}

bool MathHelper::computeScaling(const std::vector<double>& source, const std::vector<double>& target, double& s)
{
	if(source.size() != target.size() || source.size()%3 != 0)
	{
		s = 1.0;
		return false;
	}
		
	const size_t dataSize = source.size();
	const size_t numPoints = source.size()/3;

	double sourceDist(0.0);
	double targetDist(0.0);

	for(size_t i = 0; i < dataSize; ++i)
	{
		sourceDist += pow(source[i], 2);
		targetDist += pow(target[i], 2);
	}

	sourceDist = sqrt(sourceDist/static_cast<double>(numPoints));
	targetDist = sqrt(targetDist/static_cast<double>(numPoints));

	s = sourceDist > 0.0 ? targetDist / sourceDist : 1.0;
	return true;
}

//! Solve A*X = B
//! A in R^(m x n)
//! B in R^(m x o)
//! out: X in B^(n x o)
bool MathHelper::computeLeastSquaresSolution(const std::vector<double>& A, const size_t numRowsA, const size_t numColA, const std::vector<double>& B, const size_t numColB, std::vector<double>& X)
{
	char trans = 'N';
	long int m = static_cast<long int>(numRowsA);
	long int n = static_cast<long int>(numColA);
	long int nrhs = static_cast<long int>(numColB);

	double* tmpA = new double[m*n];
	double* tmpB = new double[m*nrhs];

	for(size_t row = 0; row < numRowsA; ++row)
	{
		for(size_t col = 0; col < numColA; ++col)
		{
			const size_t index = row*numColA+col;
			tmpA[index] = A[index];
		}

		for(size_t col = 0; col < numColB; ++col)
		{
			const size_t index = row*numColB+col;
			tmpB[index] = B[index];
		}
	}

	long int lda = std::max<long int>(m, 1);
	long int lbd = std::max<long int>(m, n);

	long int lwork = max(1, 2*m*n);
	double* work = new double[lwork];
	
	long int info = 0;

	//Estimate rotation matrix by A*X=B (points are rows of A and B)
	clapack::dgels_(&trans, &m, &n, &nrhs, tmpA, &lda, tmpB, &lbd, work, &lwork, &info);
	if(info != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout<<"Calculation of rotation matrix not successful "<< info << std::endl;
#endif		
		delete [] tmpA;
		delete [] tmpB;
		delete [] work;
		return false;
	}

	X.clear();
	X.resize(numColA*numColB);

	for(size_t col = 0; col < numColB; ++col)
	{
		for(size_t row = 0; row < numColA; ++row)
		{
			//numRowsA = numRowsB
			X[col*numColA+row] = tmpB[col*numRowsA+row];
		}
	}

	delete [] tmpA;
	delete [] tmpB;
	delete [] work;

	return true;
}

bool MathHelper::invertMatrix(const std::vector<double>& matrix, const size_t dim, std::vector<double>& invMatrix)
{
	const size_t matrixSize = matrix.size();
	
	double* A = new double[matrixSize];
	
	for(size_t i = 0; i < matrixSize; ++i)
	{
		A[i] = matrix[i];
	}

	long int n = static_cast<long int>(dim);
	long int* ipiv = new long int [n];
	double* work = new double[n];
	long int info;
	clapack::dgetrf_(&n, &n, A, &n, ipiv, &info);
	if(info != 0)
	{
		delete [] A;
		delete [] ipiv;
		delete [] work;
		return false;
	}

	clapack::dgetri_(&n, A, &n, ipiv, work, &n, &info);
	if(info != 0)
	{
		delete [] A;
		delete [] ipiv;
		delete [] work;
		return false;
	}

	invMatrix.clear();
	invMatrix.resize(matrixSize);
	for(size_t i = 0; i < matrixSize; ++i)
	{
		invMatrix[i] = A[i];
	}

	delete [] A;
	delete [] ipiv;
	delete [] work;
	return true;
}

bool MathHelper::transposeMatrix(const std::vector<double>& matrix, const size_t numRows, const size_t numCols, std::vector<double>& matrixT)
{
	if(!matrix.size() == numRows*numCols)
	{
		return false;
	}

	matrixT.clear();
	matrixT.resize(numRows*numCols, 0.0);

	for(size_t i = 0; i < numCols; ++i)
	{
		const size_t colOffset = i*numRows;
		for(size_t j = 0; j < numRows; ++j)
		{
			matrixT[j*numCols+i] = matrix[colOffset+j];
		}
	}

	return true;
}

void MathHelper::matrixMult(const std::vector<double>& MA, const size_t numARows, const size_t numACols, const std::string sstrTransA, const std::vector<double>& MB, const size_t numBCols, std::vector<double>& AB)
{
	char transA = sstrTransA == "N" ? 'N' : 'T';
	char transB = 'N';

	long int m = sstrTransA == "N" ? static_cast<long int>(numARows) : static_cast<long int>(numACols);
	//long int m = static_cast<long int>(numARows);
	long int n = static_cast<long int>(numBCols);
	long int k = sstrTransA == "N" ? static_cast<long int>(numACols) : static_cast<long int>(numARows);
	//long int k = static_cast<long int>(numACols);
	double alpha = 1.0;

	// MA: 
	// sstrTransA == "N": numARows x numACols
	// sstrTransA == "T": numACols x numARows

	// MB: 
	// sstrTransA == "N": numACols x numBCols
	// sstrTransA == "T": numARows x numBCols

	if(MA.size() != m*k)
	{
		std::cout << "Wrong MA input dimension: " << MA.size() << " != " << m*k << std::endl;
		return;
	}

	if(MB.size() != k*n)
	{
		std::cout << "Wrong MB input dimension: " << MB.size() << " != " << k*n << std::endl;
		return;
	}

	long int lda = sstrTransA == "N" ? m : k;
	long int ldb = k;
	long int ldc = m;

	const int mk = m*k;
	double* A = new double[mk];
#pragma omp parallel for
	for(int i = 0; i < mk; ++i)
	{
		A[i] = MA[i];
	}

	const int kn = k*n;
	double* B = new double[kn];
#pragma omp parallel for
	for(int i = 0; i < kn; ++i)
	{
		B[i] = MB[i];
	} 

	const int mn = m*n;
	double* C = new double[mn];

	double beta = 0.0;

	clapack::dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	AB.resize(mn);
#pragma omp parallel for
	for(int i = 0; i < mn; ++i)
	{
		AB[i] = C[i];
	} 

	delete [] A;
	delete [] B;
	delete [] C;
}

bool MathHelper::computeSingularValues(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& S)
{
	const size_t numElements = numRows*numColumns;

	double* data = new double[numElements];
	for(int i = 0; i < numElements; ++i)
	{
		data[i] = input[i];
	}

	char jobz = 'N';
	long int m = static_cast<long int>(numRows);
	long int n = static_cast<long int>(numColumns);

	long int lda = m;

	double* s = new double[min(m,n)];

	double* u(NULL);
	long int ldu = m;

	double* vt(NULL);
	long int ldvt = min(m,n);

	long int lwork = 3*min(m,n)*min(m,n)+max(max(m,n),4*min(m,n)*min(m,n)+4*min(m,n));
	double* work = new double[lwork];

	long int * iwork = new long int[8*min(m,n)];
	long int info = 0;

	clapack::dgesdd_(&jobz, &m, &n, data, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);

	if(info != 0)
	{
		delete [] data;
		delete [] s;
		delete [] work;
		delete [] iwork;
		return false;
	}

	const size_t sSize = static_cast<size_t>(min(m,n));

	S.clear();
	S.resize(sSize);

	for(size_t i = 0; i < sSize; ++i)
	{
		S[i] = s[i];
	}

	delete [] data;
	delete [] s;
	delete [] work;
	delete [] iwork;
	return true;
}

bool MathHelper::computeLeftSingularVectors(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& U, std::vector<double>& S)
{
	std::vector<double> VT;
	return MathHelper::computeSingularVectors(input, numRows, numColumns, U, S, VT);
}

bool MathHelper::computeSingularVectors(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& U, std::vector<double>& S, std::vector<double>& VT)
{
	const size_t numElements = numRows*numColumns;

	double* data = new double[numElements];
	for(int i = 0; i < numElements; ++i)
	{
		data[i] = input[i];
	}

	char jobz = 'S';
	long int m = static_cast<long int>(numRows);
	long int n = static_cast<long int>(numColumns);

	long int lda = m;

	double* s = new double[min(m,n)];

	double* u = new double[m*min(m,n)];
	long int ldu = m;

	double* vt = new double[n*min(m,n)];
	long int ldvt = min(m,n);

	long int lwork = 3*min(m,n)*min(m,n)+max(max(m,n),4*min(m,n)*min(m,n)+4*min(m,n));
	double* work = new double[lwork];

	long int * iwork = new long int[8*min(m,n)];
	long int info = 0;

	clapack::dgesdd_(&jobz, &m, &n, data, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);

	if(info != 0)
	{
		delete [] data;
		delete [] s;
		delete [] u;
		delete [] vt;
		delete [] work;
		delete [] iwork;
		return false;
	}

	const int uSize = m*min(m,n);
	const int sSize = min(m,n);
	const int vSize = n*min(m,n);

	U.clear();
	U.resize(uSize);
#pragma omp parallel for
	for(int i = 0; i < uSize; ++i)
	{
		U[i] = u[i];
	}

	S.clear();
	S.resize(sSize);
#pragma omp parallel for
	for(int i = 0; i < sSize; ++i)
	{
		S[i] = s[i];
	}

	VT.clear();
	VT.resize(vSize);
#pragma omp parallel for
	for(int i = 0; i < vSize; ++i)
	{
		VT[i] = vt[i];
	}

	delete [] data;
	delete [] s;
	delete [] u;
	delete [] vt;
	delete [] work;
	delete [] iwork;
	return true;
}

bool MathHelper::computeRotationAlignmentMatrix(const std::vector<double>& source, const std::vector<double>& target, std::vector<double>& R)
{
	const int dim = 3;
	const size_t minSampleSize = min(source.size(), target.size());
	const int minNumPoints = static_cast<int>(minSampleSize)/dim;

	char trans = 'N';
	long int m = static_cast<long int>(minNumPoints);
	long int n = static_cast<long int>(dim);

	double* X = new double[n*n];

	{
		long int nrhs = n;

		double* A = new double[m*n];
		double* B = new double[m*nrhs];

		for(int i = 0; i < m; i++)
		{
			A[i] = source[dim*i];
			A[m+i] = source[dim*i+1];
			A[2*m+i] = source[dim*i+2];

			B[i] = target[dim*i];
			B[m+i] = target[dim*i+1];
			B[2*m+i] = target[dim*i+2];
		}

		long int lda = m;
		long int lbd = m;

		long int lwork = max(1, 2*m*n);
		double* work = new double[lwork];
	
		long int info = 0;

		//Estimate rotation matrix by A*X=B (points are rows of A and B)
		clapack::dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &lbd, work, &lwork, &info);
		if(info != 0)
		{
#ifdef DEBUG_OUTPUT
			std::cout<<"Calculation of rotation matrix not successful "<< info << std::endl;
#endif		
			delete [] A;
			delete [] B;
			delete [] work;
			return false;
		}

	#pragma omp parallel for
		for(int i = 0; i < n; ++i)
		{
			for(int j = 0; j < n; ++j)
			{
				X[i*n+j] = B[i*m+j];
			}
		}

		delete [] A;
		delete [] B;
		delete [] work;
	}

	double* U = new double[n*n];
	double* VT = new double[n*n];
	
	{
		char jobz = 'S';

		double* S = new double[n];

		long int ldu = n;
		long int ldvt = n;

		long int lwork = 7*n*n+4*n;
		double* work = new double[lwork];

		long int * iwork = new long int[8*n];
		long int info = 0;

		//Singular value decomposition of estimated rotation matrix (X=U*S*VT)
		clapack::dgesdd_(&jobz, &n, &n, X, &n, S, U, &ldu, VT, &ldvt, work, &lwork, iwork, &info);
		if(info != 0)
		{
#ifdef DEBUG_OUTPUT
			std::cout<<"SVD decomposition of R not successful "<< info << std::endl;
#endif	

			delete [] U;
			delete [] VT;
			delete [] S;
			delete [] work;
			delete [] iwork;
			delete [] X;
			return false;
		}

		delete [] S;
		delete [] work;
		delete [] iwork;
	}

	{
		char transU = 'N';
		char transVT = 'N';

		long int ldu = n;
		long int ldvt = n;
		long int ldr = n;

		double alpha = 1.0;
		double beta = 0.0;
		
		//X=U*VT
		clapack::dgemm_(&transU, &transVT, &n, &n, &n, &alpha, U, &ldu, VT, &ldvt, &beta, X, &ldr);
	}

	//disallow reflections
   const double det = X[0]*X[4]*X[8]+X[3]*X[7]*X[2]+X[6]*X[1]*X[5]-X[2]*X[4]*X[6]-X[5]*X[7]*X[0]-X[8]*X[1]*X[3];
	if(det < 0)
	{
		std::cout<<"Determinant is "<< det << std::endl;
		delete [] X;
		delete [] U;
		delete [] VT;
		return false;
	}

	R.clear();
	R.resize(n*n);

	//Transpose tmpR matrix because we return rotation matrix which is multiplied from left to the data 
#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			R[i*n+j] = X[j*n+i];
		}
	}

	delete [] X;
	delete [] U;
	delete [] VT;

	return true;
}

void MathHelper::computeMean(const std::vector<double>& data, const size_t dataDim, std::vector<double>& mean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return;
	}

	mean.clear();
	mean.resize(dataDim, 0.0);
	
	const size_t numSamples = data.size()/dataDim;
	if(numSamples == 0)
	{
		return;
	}

	for(size_t i = 0; i < numSamples; ++i)
	{
		const size_t startIndex = i*dataDim;

		for(size_t j = 0; j < dataDim; ++j)
		{
			const size_t currIndex = startIndex+j;
			mean[j] += data[currIndex];
		}
	}

	const double factor = 1.0 / static_cast<double>(numSamples);
	for(size_t i = 0; i < dataDim; ++i)
	{
		mean[i] *= factor;
	}
}

void MathHelper::computeProcrustesMean(const std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& procrustesMean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return;
	}

	std::vector<double> tmpData = data;

	for(size_t iter = 0; iter < numIter; ++iter)
	{
		// Calc mean
		MathHelper::computeMean(tmpData, dataDim, procrustesMean);

		// Compute alignment to the mean shape
		MathHelper::alignData(tmpData, procrustesMean);
	}

	// Calc mean
	MathHelper::computeMean(tmpData, dataDim, procrustesMean);
}

void MathHelper::centerData(std::vector<double>& data, std::vector<double>& mean)
{
	mean.clear();
	mean.push_back(0.0);
	mean.push_back(0.0);
	mean.push_back(0.0);

	const size_t numPoints = data.size()/3;

	for(size_t i = 0; i < numPoints; ++i)
	{
		for(size_t j = 0; j < 3; ++j)
		{
			mean[j] += data[i*3+j];
		}
	}

	const double invNumPoints(1.0/static_cast<double>(numPoints));
	for(size_t i = 0; i < 3; ++i)
	{
		mean[i] *= invNumPoints;
	}

	MathHelper::translateData(mean, "-", data);
}

void MathHelper::centerData(std::vector<double>& data, const size_t dataDim, std::vector<double>& mean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return;
	}

	MathHelper::computeMean(data, dataDim, mean);

	const size_t numSamples = data.size()/dataDim;
	for(size_t i = 0; i < numSamples; ++i)
	{
		const size_t startIndex = i*dataDim;

		for(size_t j = 0; j < dataDim; ++j)
		{
			const size_t currIndex = startIndex+j;
			data[currIndex] -= mean[j];
		}
	}
}

void MathHelper::centerData(const std::vector<double>& data, const size_t dataDim, std::vector<double>& centeredData, std::vector<double>& mean)
{
	if(data.size() % dataDim != 0)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Data dimension for centering not correct" << std::endl;
#endif
		return;
	}

	MathHelper::computeMean(data, dataDim, mean);

	const size_t numSamples = data.size()/dataDim;
	centeredData.resize(numSamples*dataDim, 0.0);

	//TODO Run in parallel
	//for(size_t i = 0; i < numSamples; ++i)
#pragma omp parallel for	
	for(int i = 0; i < numSamples; ++i)
	{
		const size_t startIndex = i*dataDim;

		for(size_t j = 0; j < dataDim; ++j)
		{
			const size_t currIndex = startIndex+j;
			centeredData[currIndex] = data[currIndex]-mean[j];
		}
	}
}

void MathHelper::rotateMesh(const std::vector<double>& R, const std::string& sstrOp, DataContainer& mesh)
{
	std::vector<double> vertexList = mesh.getVertexList();
	MathHelper::rotateData(R, sstrOp, vertexList);
	mesh.setVertexList(vertexList);
}

void MathHelper::rotateData(const std::vector<double>& R, const std::string& sstrOp, std::vector<double>& data)
{
	if(R.size() != 9)
	{
		return;
	}

	const int numPoints = static_cast<int>(data.size()/3);

	char transA = sstrOp == "N" ? 'N' : 'T';
	char transB = 'N';

	long int m = static_cast<long int>(3);
	long int n = static_cast<long int>(numPoints);
	long int k = static_cast<long int>(3);	
	double alpha = 1.0;

	long int lda = m;
	long int ldb = k;
	long int ldc = m;

	const int mm = m*m;
	double* A = new double[mm];
#pragma omp parallel for	
	for(int i = 0; i < mm; ++i)
	{
		A[i] = R[i];
	}

	const int mn = m*n;
	double* B = new double[mn];
#pragma omp parallel for	
	for(int i = 0; i < mn; ++i)
	{
		B[i] = data[i];
	} 

	double* C = new double[mn];

	double beta = 0.0;

	clapack::dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	data.clear();
	data.resize(mn);
#pragma omp parallel for	
	for(int i = 0; i < mn; ++i)
	{
		data[i] = C[i];
	} 
	
	delete [] A;
	delete [] B;
	delete [] C;
}

void MathHelper::rotateData(const std::vector<double>& R, const std::string& sstrOp, std::vector<Vec3d*>& data)
{
	if(R.size() != 9)
	{
		return;
	}

	const int numPoints = static_cast<int>(data.size());

	std::vector<double> tmpData;
	tmpData.resize(numPoints*3);

#pragma omp parallel for	
	for(int i = 0; i < numPoints; ++i)
	{
		const int i3 = 3*i;

		const Vec3d* pCurrVec = data[i];
		if(pCurrVec!=NULL)
		{
			tmpData[i3] = (*pCurrVec)[0];
			tmpData[i3+1] = (*pCurrVec)[1];
			tmpData[i3+2] = (*pCurrVec)[2];
		}
		else
		{
			tmpData[i3] = 0.0;
			tmpData[i3+1] = 0.0;
			tmpData[i3+2] = 0.0;
		}
	}

	rotateData(R, sstrOp, tmpData);

#pragma omp parallel for	
	for(int i = 0; i < numPoints; ++i)
	{
		Vec3d* pCurrVec = data[i];
		if(pCurrVec!=NULL)
		{
			const size_t currIndex = 3*i;
			(*pCurrVec)[0] = tmpData[currIndex];
			(*pCurrVec)[1] = tmpData[currIndex+1];
			(*pCurrVec)[2] = tmpData[currIndex+2];
		}
	}
}

void MathHelper::translateMesh(const std::vector<double>& t, const std::string& sstrOp, DataContainer& mesh)
{
	std::vector<double> vertexList = mesh.getVertexList();
	MathHelper::translateData(t, sstrOp, vertexList);
	mesh.setVertexList(vertexList);
}

void MathHelper::translateData(const std::vector<double>& t, const std::string& sstrOp, std::vector<double>& data)
{
	if(t.size() != 3)
	{
		return;
	}

	const size_t numPoints = data.size() / 3;
	for(size_t i = 0; i < numPoints; ++i)
	{
		const size_t startIndex = 3*i;

		for(size_t j = 0; j < 3; ++j)
		{
			const double value = sstrOp == "+" ? data[startIndex+j]+t[j] : data[startIndex+j]-t[j];
			data[startIndex+j] = value;
		}
	}
}

void MathHelper::scaleMesh(const double factor, DataContainer& mesh)
{
	std::vector<double> vertexList = mesh.getVertexList();
	MathHelper::scaleData(factor, vertexList);
	mesh.setVertexList(vertexList);
}

void MathHelper::scaleData(const double factor, std::vector<double>& data)
{
	for(size_t i = 0; i < data.size(); ++i)
	{
		data[i] *= factor;
	}
}

std::string MathHelper::convertToStdString(const size_t num)
{
	std::stringstream out;
	out << num;
	return out.str();
}

bool MathHelper::interpolateNDimPoints(const std::vector<double>& p1, const std::vector<double>& p2, const double lambda, std::vector<double>& outP)
{
	if(lambda < 0.0 || lambda > 1.0)
	{
		return false;
	}

	if(p1.size() != p2.size())
	{
		return false;
	}

	const size_t dim = p1.size();
	
	outP.clear();
	outP.reserve(dim);

	for(size_t i = 0; i < dim; ++i)
	{
		const double currVal = p1[i]+lambda*(p2[i]-p1[i]);
		outP.push_back(currVal);
	}

	return true;
}

bool MathHelper::getWeightCurveMidpoint(const std::vector<double>& weights, const size_t dim, std::vector<double>& midpoint)
{
	const size_t numPoints = weights.size()/dim;
	if(numPoints%2 != 1)
	{
		std::cout << "Num weight samples even" << std::endl;
		return false;
	}

	const size_t index = numPoints/2;
	for(size_t i = 0; i < dim; ++i)
	{
		midpoint.push_back(weights[index*dim+i]);
	}

	return true;
}

bool MathHelper::interpolateExpressionWeightMidpoint(const std::vector<double>& weights1, const std::vector<double>& weights2, const size_t dim, const size_t numOutFrames, std::vector<double>& interpolatedWeights)
{
	const size_t sampleSize = weights1.size() / dim;

	if(weights1.size() != weights2.size() || sampleSize == 0)
	{
		return false;
	}

	std::vector<double> w1Midpoint;
	getWeightCurveMidpoint(weights1, dim, w1Midpoint);

	std::vector<double> w2Midpoint;
	getWeightCurveMidpoint(weights2, dim, w2Midpoint);


	std::vector<double> diffVec;
	diffVec.reserve(dim);

	for(size_t i = 0; i < dim; ++i)
	{
		diffVec.push_back(w1Midpoint[i] - w2Midpoint[i]);
	}

	for(size_t i = 0; i < numOutFrames; ++i)
	{
		const double lambda = static_cast<double>(i)/static_cast<double>(numOutFrames-1);

		std::vector<double> samplePoint;
		if(!interpolateNDimPoints(w1Midpoint, w2Midpoint, lambda, samplePoint))
		{
			std::cout << "sampleWeights() - unable to sample curve" << std::endl;
			return false;
		}

		for(size_t j = 0; j < dim; ++j)
		{
			interpolatedWeights.push_back(samplePoint[j]);
		}
	}

	return true;
}

double MathHelper::evaluateThinPlateSpline(const std::vector<double>& point)
{
	double value(0.0);

	const size_t pointDim = point.size();
	for(size_t i = 0; i < pointDim; ++i)
	{
		value += pow(point[i], 2);
	}

	if(value < pow(math_eps, 2))
	{
		return 0.0;
	}
	else 
	{
		return value*log(sqrt(value));
	}
}

//#define DEBUG_TEST_OUT

bool MathHelper::computeInterpolationBasis(const std::vector<double>& sourcePoints, const size_t sourcePointsDim, const std::vector<double>& targetPoints, const size_t targetPointsDim
												, std::vector<double>& vecC, std::vector<double>& matA, std::vector<double>& matW)
{
	const size_t numPoints = sourcePoints.size()/sourcePointsDim;
	if(numPoints != targetPoints.size()/targetPointsDim)
	{
		std::cout << "Different number of source and target points " << numPoints << " != " << targetPoints.size()/targetPointsDim << std::endl;
		return false;
	}

	const size_t numTmpARows = numPoints+1+sourcePointsDim;
	const size_t numTmpACols = numTmpARows;
	std::vector<double> tmpA;
	tmpA.resize(numTmpARows*numTmpACols, 0.0);

	for(size_t i = 0; i < numPoints; ++i)
	{
		const size_t matRow1 = i;
		const size_t matCol2 = matRow1;

		for(size_t j = i; j < numPoints; ++j)
		{
			const size_t matCol1 = j;
			const size_t matRow2 = matCol1;

			//Compute (s_i - s_j)
			std::vector<double> currDiffVec;
			currDiffVec.resize(sourcePointsDim, 0.0);

			for(size_t k = 0; k < sourcePointsDim; ++k)
			{
				currDiffVec[k] = sourcePoints[i*sourcePointsDim+k] - sourcePoints[j*sourcePointsDim+k];
			}

			//Evaluate radial basis function at (s_i - s_j)
			const double currRBFValue = evaluateThinPlateSpline(currDiffVec);

			MathHelper::matSet(tmpA, numTmpARows, matRow1, matCol1, currRBFValue);
			MathHelper::matSet(tmpA, numTmpARows, matRow2, matCol2, currRBFValue);
		}

		MathHelper::matSet(tmpA, numTmpARows, numPoints, i, 1.0);
		MathHelper::matSet(tmpA, numTmpARows, i, numPoints, 1.0);

		for(size_t j = 0; j < sourcePointsDim; ++j)
		{
			const double currValue = sourcePoints[i*sourcePointsDim+j];	
			MathHelper::matSet(tmpA, numTmpARows, numPoints+1+j, i, currValue);
			MathHelper::matSet(tmpA, numTmpARows, i, numPoints+1+j, currValue);
		}
	}

	const size_t numTmpBRows = numTmpARows;
	std::vector<double> tmpB;
	tmpB.resize(numTmpBRows*targetPointsDim, 0.0);
	
	for(size_t i = 0; i < numPoints; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			const double currValue = targetPoints[i*targetPointsDim+j];
			MathHelper::matSet(tmpB, numTmpBRows, i, j, currValue);
		}
	}

#ifdef DEBUG_TEST_OUT
	std::cout << std::endl;
	std::cout << "Matrix A" << std::endl;
	for(size_t i = 0; i < numTmpARows; ++i)
	{
		for(size_t j = 0; j < numTmpARows; ++j)
		{
			std::cout << MathHelper::matGet(tmpA, numTmpARows, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "Matrix B" << std::endl;
	for(size_t i = 0; i < numTmpBRows; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			std::cout << MathHelper::matGet(tmpB, numTmpBRows, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif

	std::vector<double> tmpX;
	if(!MathHelper::computeLeastSquaresSolution(tmpA, numTmpARows, numTmpACols, tmpB, targetPointsDim, tmpX))
	{
		return false;
	}

#ifdef DEBUG_TEST_OUT
	std::cout << std::endl;
	std::cout << "Matrix X" << std::endl;
	for(size_t i = 0; i < numTmpBRows; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			std::cout << MathHelper::matGet(tmpX, numTmpBRows, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::vector<double> AB;
	MathHelper::matrixMult(tmpA, numTmpARows, numTmpACols, "N", tmpX, targetPointsDim, AB);

	std::cout << std::endl;
	std::cout << "Matrix AB" << std::endl;
	for(size_t i = 0; i < numTmpARows; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			std::cout << MathHelper::matGet(AB, numTmpARows, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif

	const size_t numTmpXRows = numTmpACols;
	if(tmpX.size() != numTmpXRows*targetPointsDim)
	{
		return false;
	}

	matW.resize(numPoints*targetPointsDim);

	for(size_t i = 0; i < numPoints; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			const double currValue = MathHelper::matGet(tmpX, numTmpXRows, i, j);
			MathHelper::matSet(matW, numPoints, i, j, currValue);
		}
	}

	vecC.resize(targetPointsDim);
	for(size_t i = 0; i < targetPointsDim; ++i)
	{
		vecC[i] = MathHelper::matGet(tmpX, numTmpXRows, numPoints, i);
	}

	matA.resize(targetPointsDim*sourcePointsDim);
	for(size_t i = 0; i < targetPointsDim; ++i)
	{
		for(size_t j = 0; j < sourcePointsDim; ++j)
		{
			const double currValue = MathHelper::matGet(tmpX, numTmpXRows, numPoints+1+j, i);
			MathHelper::matSet(matA, targetPointsDim, i, j, currValue);
		}
	}

#ifdef DEBUG_TEST_OUT
	std::cout << std::endl;
	std::cout << "Matrix W" << std::endl;
	for(size_t i = 0; i < numPoints; ++i)
	{
		for(size_t j = 0; j < targetPointsDim; ++j)
		{
			std::cout << MathHelper::matGet(matW, numPoints, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "Vector c" << std::endl;
	for(size_t i = 0; i < targetPointsDim; ++i)
	{
		std::cout << vecC[i] << " " << std::endl;
	}
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "Matrix A" << std::endl;
	for(size_t i = 0; i < targetPointsDim; ++i)
	{
		for(size_t j = 0; j < sourcePointsDim; ++j)
		{
			std::cout << MathHelper::matGet(matA, targetPointsDim, i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif

	return true;
}

bool MathHelper::evaluateInterpolation(const std::vector<double>& vecC, const std::vector<double>& matA, const std::vector<double>& matW, const std::vector<double>& sourcePoints, const std::vector<double>& inSourcePoint, std::vector<double>& outTargetPoint)
{
	const size_t targetDim = vecC.size();
	const size_t sourceDim = matA.size()/targetDim;
	const size_t numSourcePoints = sourcePoints.size()/sourceDim;
	if(inSourcePoint.size() != sourceDim)
	{
		return false;
	}

	if(matW.size() != numSourcePoints*targetDim)
	{
		return false;
	}

	outTargetPoint = vecC;

	std::vector<double> trafoPoint1;
	matrixMult(matA, targetDim, sourceDim, "N", inSourcePoint, 1, trafoPoint1);

	std::vector<double> sourceBasisPoint;
	sourceBasisPoint.resize(numSourcePoints);

//#pragma omp parallel for
	for(int i = 0; i < numSourcePoints; ++i)
	{
		std::vector<double> currDiffVec;
		currDiffVec.resize(sourceDim, 0.0);

		for(size_t j = 0; j < sourceDim; ++j)
		{
			currDiffVec[j] = inSourcePoint[j]-sourcePoints[i*sourceDim+j];
		}

		const double currRBFValue = evaluateThinPlateSpline(currDiffVec);
		sourceBasisPoint[i] = currRBFValue;
	}

	std::vector<double> trafoPoint2;
	matrixMult(matW, numSourcePoints, targetDim, "T", sourceBasisPoint, 1, trafoPoint2);

	for(size_t i = 0; i < targetDim; ++i)
	{
		outTargetPoint[i] += trafoPoint1[i];
		outTargetPoint[i] += trafoPoint2[i];
	}

	return true;
}

void MathHelper::compute1dVariance(const std::vector<double>& samples, double& mean, double& variance)
{
	const size_t numSamples = samples.size();
	if(numSamples == 0)
	{
		return;
	}

	if(numSamples == 1)
	{
		mean = samples[0];
		variance = 0.0;
		return;
	}

	mean = 0.0;
	variance = 0.0;

	for(size_t i = 0; i < numSamples; ++i)
	{
		mean += samples[i];
	}

	mean /= static_cast<double>(numSamples);

	for(size_t i = 0; i < numSamples; ++i)
	{
		const double currDiff = samples[i]-mean;
		variance += currDiff*currDiff;
	}

	variance /= static_cast<double>(numSamples-1);
}

bool MathHelper::incrementMean(const std::vector<double>& newPoint, const size_t numPoints, std::vector<double>& currMean)
{
	if(numPoints == 0)
	{
		return true;
	}

	if(newPoint.size() != currMean.size())
	{
		std::cout << "MathHelper::incrementMean(...) - wrong vector dimensions " << newPoint.size() << " != " << currMean.size() << std::endl;
		return false;
	}

	const double factor = 1.0/static_cast<double>(numPoints);

	const size_t dim = newPoint.size();
	for(size_t i = 0; i < dim; ++i)
	{
		currMean[i] += factor*newPoint[i];
	}

	return true;
}

bool MathHelper::incrementCov(const std::vector<double>& newPoint, const size_t numPoints, const std::vector<double>& mean, std::vector<double>& currCov)
{
	if(numPoints <= 1)
	{
		return true;
	}

	if(newPoint.size() != mean.size())
	{
		std::cout << "MathHelper::incrementMean(...) - wrong vector dimensions (mean) " << newPoint.size() << " != " << mean.size() << std::endl;
		return false;
	}

	const size_t dim = newPoint.size();
	if(dim*dim != currCov.size())
	{
		std::cout << "MathHelper::incrementMean(...) - wrong vector dimensions (cov) " << dim*dim << " != " << currCov.size() << std::endl;
		return false;
	}

	const double factor = 1.0/static_cast<double>(numPoints-1);

	for(size_t col = 0; col < dim; ++col)
	{
		for(size_t row = col; row < dim; ++row)
		{
			const double tmpCovValue = factor*(newPoint[col]-mean[col])*(newPoint[row]-mean[row]);
			currCov[col*dim+row] += tmpCovValue;
			currCov[row*dim+col] = currCov[col*dim+row];
		}
	}

	return true;
}

void MathHelper::cleanMesh(DataContainer& mesh)
{
	std::set<size_t> validPointIndices;

	const std::vector<std::vector<int>>& vertexIndexList = mesh.getVertexIndexList();
	for(size_t i = 0; i < vertexIndexList.size(); ++i)
	{
		const std::vector<int>& currPolygonIndices = vertexIndexList[i];
		const size_t currPolygonSize = currPolygonIndices.size();

		size_t numDisjointVertices(0);
		for(size_t j = 0; j < currPolygonSize; ++j)
		{
			const size_t i1 = j;
			const size_t i2 = (j+1)%currPolygonSize;

			if(currPolygonIndices[i1] != currPolygonIndices[i2])
			{
				++numDisjointVertices;
			}
		}

		if(numDisjointVertices < 3)
		{
			continue;
		}

		for(size_t j = 0; j < currPolygonSize; ++j)
		{
			validPointIndices.insert(currPolygonIndices[j]);
		}
	}


	std::vector<double> meshVertices = mesh.getVertexList();
	std::vector<double> meshVertexColors = mesh.getVertexColorList();
	bool bHasVertexColors(meshVertices.size() == meshVertexColors.size());

	std::vector<std::pair<int, int>> oldNewMap;

	size_t newId = 0;
	for(size_t id = 0; id < mesh.getNumVertices(); ++id)
	{
		if(validPointIndices.find(id) == validPointIndices.end())
		{
			oldNewMap.push_back(std::make_pair(static_cast<int>(id), -1));
		}
		else
		{
			oldNewMap.push_back(std::make_pair(static_cast<int>(id), static_cast<int>(newId)));
			++newId;
		}
	}

	DataContainer cleanMesh = mesh;

	std::vector<double> cleanMeshVertices;
	std::vector<double> cleanMeshVertexColors;

	for(size_t oldId = 0; oldId < mesh.getNumVertices(); ++oldId)
	{
		const int newPointId = oldNewMap[oldId].second;
		if(newPointId == -1)
		{
			continue;
		}
		
		cleanMeshVertices.push_back(meshVertices[3*oldId]);
		cleanMeshVertices.push_back(meshVertices[3*oldId+1]);
		cleanMeshVertices.push_back(meshVertices[3*oldId+2]);

		if(bHasVertexColors)
		{
			cleanMeshVertexColors.push_back(meshVertexColors[3*oldId]);
			cleanMeshVertexColors.push_back(meshVertexColors[3*oldId+1]);
			cleanMeshVertexColors.push_back(meshVertexColors[3*oldId+2]);
		}
	}
	
	cleanMesh.setVertexList(cleanMeshVertices);
	cleanMesh.setVertexColorList(cleanMeshVertexColors);

	std::vector<std::vector<int>> cleanMeshPolygons;
	for(size_t i = 0; i < vertexIndexList.size(); ++i)
	{
		const std::vector<int>& currPolygonIndices = vertexIndexList[i];
		const size_t currPolygonSize = currPolygonIndices.size();

		size_t numDisjointVertices(0);
		for(size_t j = 0; j < currPolygonSize; ++j)
		{
			const size_t i1 = j;
			const size_t i2 = (j+1)%currPolygonSize;

			if(currPolygonIndices[i1] != currPolygonIndices[i2])
			{
				++numDisjointVertices;
			}
		}

		if(numDisjointVertices < 3)
		{
			continue;
		}

		std::vector<int> newPolygonIndices;
		for(size_t j = 0; j < currPolygonSize; ++j)
		{
			const int newId = oldNewMap[currPolygonIndices[j]].second;
			if(newId == -1)
			{
				std::cout << "Error" << std::endl;
				return;
			}

			newPolygonIndices.push_back(newId);
		}

		cleanMeshPolygons.push_back(newPolygonIndices);
	}

	cleanMesh.setVertexIndexList(cleanMeshPolygons);

	if(cleanMesh.getNumVertices() != mesh.getNumVertices())
	{
		std::cout << "Removed " << mesh.getNumVertices() - cleanMesh.getNumVertices() << " vertices" << std::endl;
		std::cout << "Removed " << mesh.getNumFaces() - cleanMesh.getNumFaces() << " faces" << std::endl;
	}

	mesh = cleanMesh;
}