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

#ifndef MDLSHAPECOSTFUNCTION_H
#define MDLSHAPECOSTFUNCTION_H

#include "Definitions.h"
#include <stdlib.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_cost_function.h>

#include <vector>

class MDLShapeCostFunction : public vnl_cost_function
{
public:
	MDLShapeCostFunction(std::vector<double>& data, const std::vector<double>& initialParam, const size_t numIdentities, const size_t numExpressions, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs
								, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec, const std::vector<double>& vertexAreas, const double meanShapeArea, const double identityWeight
								, const double expressionWeight, const double maxParameterVariation, const size_t iShape);

	~MDLShapeCostFunction();

	bool setLandmarks(const std::vector<size_t>& lmkIndices, std::vector<double>& lmks, const double lmkWeight);

	bool setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight);

	virtual void compute(const vnl_vector<double>& x, double* f, vnl_vector<double>* g);

	//void computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues);

	//void computeCovarianceMatrix(const size_t mode, std::vector<double>& covarianceMatrix);

#ifdef OUTPUT_GRADIENT_NORM
	double getGradientNorm() const { return m_gradientNorm; }
#endif

private:
	void init(double* f, vnl_vector<double>* g);

	void addModeCompactnessEnergy(const size_t mode, const std::vector<double>& modeEigenValues, double* f);

	//void addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t iIndex, const size_t jIndex, vnl_vector<double>* g);
	void addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, vnl_vector<double>* g);

	//void addShapeLandmarkEnergy(const size_t iIndex, const size_t jIndex, double* f, vnl_vector<double>* g);
	void addShapeLandmarkEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g);
		
	//void addShapeSmoothnessEnergy(const size_t iIndex, const size_t jIndex, double* f, vnl_vector<double>* g);
	void addShapeSmoothnessEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g);

	//OLD VERSION
	//void computeEigenValueDerivativeOld(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t eigenValueIndex
	//											, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij);

	void computeEigenValueDerivative(const size_t mode, const std::vector<double>& precomputedModeDataSum, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t eigenValueIndex
												, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij);

	void computeShapeParameterDerivative(const size_t iIndex, const size_t jIndex, std::vector<double>& dXij_dalpha);

	void updateShapeParametrization(const vnl_vector<double>& x, const size_t iIndex, const size_t jIndex);

	void updateShapeData(const size_t iIndex, const size_t jIndex);

	//Compute the sum over all identities or expressions of the centered data.
	//mode = 2: Computes sum over all expressions (x_i = sum_{m=1}^{d3} c_{im})
	//mode = 3: Computes sum over all identities (x_j = sum_{m=1}^{d2} c_{mj})
	void computeModeDataSum(const size_t mode, std::vector<double>& modeDataSum);

	void computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues);

	void computeCovarianceMatrix(const size_t mode, std::vector<double>& covarianceMatrix, size_t& matrixDim);

#ifdef OUTPUT_GRADIENT_NORM
	double computeGradientNorm(const vnl_vector<double>* g) const;
#endif


	MDLShapeCostFunction(const MDLShapeCostFunction& costFunction);

	MDLShapeCostFunction& operator=(const MDLShapeCostFunction& costFunction);

	//Vertices of all samples (d2 identities, d3 expressions)
	//Changed w.r.t. the current computed parametrization within each iteration step
	//Dimension d1*d2*d3
	std::vector<double>& m_data;
	//Centered data
	//Dimension d1*d2*d3
	std::vector<double> m_centeredData;
	//Initial parameterization of the data
	//Dimension 2n*d2*d3
	const std::vector<double>& m_initialParam;
	//Data parametrization of currently processed shape
	//Dimension d1
	std::vector<double> m_dataParam_ij;
	//Dimension d1
	std::vector<double> m_dataMean;

	const size_t m_dataSize;
	const size_t m_numIdentities;
	const size_t m_numExpressions;
	const size_t m_numSamples;
	const size_t m_sampleDataDim;
	const size_t m_sampleParamDim;
	const size_t m_numSampleVertices;

	const std::vector<std::vector<double>>& m_vecCs;
	const std::vector<std::vector<double>>& m_matAs;
	const std::vector<std::vector<double>>& m_matWs;
	const std::vector<std::vector<double>>& m_sourcePointsVec;

	const std::vector<double>& m_vertexAreas;
	const double m_meanShapeArea;

	std::vector<size_t> m_lmkIndices;
	std::vector<double> m_lmks;
	size_t m_numLmks;

	const	double m_identityWeight;
	const double m_expressionWeight;
	double m_lmkWeight;
	double m_smoothWeight;

	const double m_maxSqrParameterVariation;

	std::vector<std::vector<size_t>> m_precomputedSmoothnessIndices;
	std::vector<std::vector<double>> m_precomputedSmoothnessWeights;

	size_t m_iIndex;
	size_t m_jIndex;

	bool m_bUpdateData;

#ifdef OUTPUT_GRADIENT_NORM
	double m_gradientNorm;
#endif
};

#endif