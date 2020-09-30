/*************************************************************************************************************************/
// This source is provided for NON-COMMERCIAL RESEARCH PURPOSES only, and is provided “as is” WITHOUT ANY WARRANTY; 
// without even the implied warranty of fitness for a particular purpose. The redistribution of the code is not permitted.
//
// If you use the source or part of it in a publication, cite the following paper:
// 
// T. Bolkart, S. Wuhrer
// 3D Faces in Motion: Fully Automatic Registration and Statistical Analysis.
// Computer Vision and Image Understanding, 131:100-115, 2015
//
// Copyright (c) 2015 Timo Bolkart, Stefanie Wuhrer
/*************************************************************************************************************************/

#ifndef MathHelper_H
#define MathHelper_H

#include <stdlib.h>
#include "Definitions.h"
//#include "Mesh.h"
#include "DataContainer.h"
#include "VectorNX.h"

class MathHelper
{
public:
	static bool matSet(std::vector<double>& matrix, const size_t numRows, const size_t row, const size_t col, const double value);
	static double matGet(const std::vector<double>& matrix, const size_t numRows, const size_t row, const size_t col);

	static int getRandomInteger(const int min, const int max);

	static double getRandomDouble(const double min, const double max, bool bReachMax = true);

	static void getRandomlyPermutedInteger(const int min, const int max, std::vector<int>& permutedInteger);

	static double getGaussianRandomVariable(const double mean, const double variance);

	//!Checks for a triangle with corners A, B, and C if it is obtuse (any angle > 90°) or acute.
	//! \param A		corner vertex A
	//! \param B		corner vertex B
	//! \param C		corner vertex C
	//! return			output of obtuse vertex index (A = 0, B = 1, C = 2), 
	//						output of -1 if triangle is acute or right
	static int triangleObtuse(const Vec3d& A, const Vec3d& B, const Vec3d& C);

	static void computeVoronoiAreas(const Vec3d& A, const Vec3d& B, const Vec3d& C, double& areaA, double& areaB, double& areaC);

	//! Compute normals, based on Max1999 - Weights for Computing Vertex Normals from Facet Normals
	static void computeVertexNormals(const DataContainer& poly, std::vector<double>& vertexNormals);

	static void computeVertexAreas(const DataContainer& mesh, std::vector<double>& vertexAreas, double& meshArea);

	static void computeBoundaryVertices(const DataContainer& mesh, std::vector<size_t>& boundaryVertexIndices);

	/*static void computeBoundaryVertices(const DataContainer& mesh, bool bInner, bool bOuter, std::vector<size_t>& boundaryVertexIndices);*/

	//static void computeVertexNormalsOld(const DataContainer& poly, std::vector<Vec3d*>& vertexNormals);

	//Compute projection of p1 into tangential plane of p2
	static void getPlaneProjection(const Vec3d& p1, const Vec3d& p2, const Vec3d& n2, Vec3d& outPoint);

	static bool pointInPlane(const Vec3d& pPlane, const Vec3d& nPlane, const Vec3d& p);

	static bool getBarycentricCoords(const Vec3d& p1, const Vec3d& v1, const Vec3d& v2, const Vec3d& p, double& u, double& v);

	static bool pointInTriangle(const Vec3d& p1, const Vec3d& p2, const Vec3d& p3, const Vec3d& p);


	static void initTransformation(double& s, std::vector<double>& R, std::vector<double>& t);

	static void invertTransformation(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp, double& invs, std::vector<double>& invR, std::vector<double>& invt);

	static void transformMesh(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp, DataContainer& mesh);

	static void transformData(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp, std::vector<double>& data);

	static void transformThinPlateSpline(const double s, const std::vector<double>& R, const std::string& sstrRotOp, const std::vector<double>& t, const std::string& sstrTransOp
													, std::vector<double>& vecC, std::vector<double>& matA, std::vector<double>& matW);

	static bool alignData(std::vector<double>& source, const std::vector<double>& target);

	static bool alignData(std::vector<double>& source, const std::vector<double>& target, const std::vector<size_t>& vertexIndices);

	static bool computeRigidLandmarkAlignment(const std::vector<double>& lmks1, const std::vector<bool>& lmks1Loaded, const std::vector<double>& lmks2, const std::vector<bool>& lmks2Loaded
														, double& s, std::vector<double>& R, std::vector<double>& t, bool bScaling = true);

	static bool computeRigidProcrustesAlignment(std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& procrustesMean);

	static bool computeRigidProcrustesAlignment(const std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& alignedData, std::vector<double>& procrustesMean);

	static bool computeRigidProcrustesAlignment(const std::vector<double>& data, const std::vector<size_t>& vertexIndices, const size_t dataDim, const size_t numIter, std::vector<double>& alignedData, std::vector<double>& procrustesMean);

	//! Computes scaling, rotation and translation for modelData = s*R*sequenceData+t
	static bool computeAlignmentTrafo(const std::vector<double>& source, const std::vector<double>& target, double& s, std::vector<double>& R, std::vector<double>& t, bool bScaling = true);
	
	static bool computeScaling(const std::vector<double>& source, const std::vector<double>& target, double& s);

	//! Calculating best solution of R*source = target
	static bool computeRotationAlignmentMatrix(const std::vector<double>& source, const std::vector<double>& target, std::vector<double>& R);

	static void computeMean(const std::vector<double>& data, const size_t dataDim, std::vector<double>& mean);

	static void computeProcrustesMean(const std::vector<double>& data, const size_t dataDim, const size_t numIter, std::vector<double>& procrustesMean);

	static void centerData(std::vector<double>& data, std::vector<double>& mean);

	static void centerData(std::vector<double>& data, const size_t dataDim, std::vector<double>& mean);

	static void centerData(const std::vector<double>& data, const size_t dataDim, std::vector<double>& centeredData, std::vector<double>& mean);

	static void rotateMesh(const std::vector<double>& R, const std::string& sstrOp, DataContainer& mesh);

	static void rotateData(const std::vector<double>& R, const std::string& sstrOp, std::vector<double>& data);

	static void rotateData(const std::vector<double>& R, const std::string& sstrOp, std::vector<Vec3d*>& data);

	static void translateMesh(const std::vector<double>& t, const std::string& sstrOp, DataContainer& mesh);

	static void translateData(const std::vector<double>& t, const std::string& sstrOp, std::vector<double>& data);

	static void scaleMesh(const double factor, DataContainer& mesh);

	static void scaleData(const double factor, std::vector<double>& data);


	static std::string convertToStdString(const size_t num);

	static std::string convertToStdString(const double num);

	static bool interpolateNDimPoints(const std::vector<double>& p1, const std::vector<double>& p2, const double lambda, std::vector<double>& outP);

	static bool interpolateExpressionWeightMidpoint(const std::vector<double>& weights1, const std::vector<double>& weights2, const size_t dim, const size_t numOutFrames, std::vector<double>& interpolatedWeights);

	static double evaluateThinPlateSpline(const std::vector<double>& point);

	static bool computeInterpolationBasis(const std::vector<double>& sourcePoints, const size_t sourcePointsDim, const std::vector<double>& targetPoints, const size_t targetPointsDim
												, std::vector<double>& vecC, std::vector<double>& matA, std::vector<double>& matW);

	static bool evaluateInterpolation(const std::vector<double>& vecC, const std::vector<double>& matA, const std::vector<double>& matW, const std::vector<double>& sourcePoints, const std::vector<double>& inSourcePoint, std::vector<double>& outTargetPoint);



	static void compute1dVariance(const std::vector<double>& samples, double& mean, double& variance);

	static bool computeLeastSquaresSolution(const std::vector<double>& A, const size_t numRowsA, const size_t numColA, const std::vector<double>& B, const size_t numColB, std::vector<double>& X);

	//Invert squared matrix with dimension dim x dim
	static bool invertMatrix(const std::vector<double>& matrix, const size_t dim, std::vector<double>& invMatrix);

	static bool transposeMatrix(const std::vector<double>& matrix, const size_t numARows, const size_t numACols, std::vector<double>& matrixT);

	//! Matrix multiplication AB = MA*MB (sstrTransA = "N") or AB = MA^T*MB (sstrTransA = "T")
	//! \param MA				matrix of dimension numARows x numACols
	//! \param numARows		number of rows of matrix A
	//! \param numACols		number of columns of matrix A
	//! \param sstrTransA	identifier if matrix should be transposed for multiplication
	//! \param MB				matrix of dimension numBCols x numBCols (sstrTransA = "N") or numARows x numBCols (sstrTransA = "T")
	//! \param numBCols		number of columns of matrix B
	//! \param AB				matrix multiplication of dimension numARows x numBCols (sstrTransA = "N") or numACols x numBCols (sstrTransA = "T") 
	static void matrixMult(const std::vector<double>& MA, const size_t numARows, const size_t numACols, const std::string sstrTransA, const std::vector<double>& MB, const size_t numBCols, std::vector<double>& AB);


	static bool computeSingularValues(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& S);

	static bool computeLeftSingularVectors(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& U, std::vector<double>& S);

	static bool computeSingularVectors(const std::vector<double>& input, const size_t numRows, const size_t numColumns, std::vector<double>& U, std::vector<double>& S, std::vector<double>& VT);

	static bool incrementMean(const std::vector<double>& newPoint, const size_t numPoints, std::vector<double>& currMean);

	static bool incrementCov(const std::vector<double>& newPoint, const size_t numPoints, const std::vector<double>& mean, std::vector<double>& currCov);

	static void cleanMesh(DataContainer& mesh);

private:

	static bool getWeightCurveMidpoint(const std::vector<double>& weights, const size_t dim, std::vector<double>& midpoint);
};

#endif