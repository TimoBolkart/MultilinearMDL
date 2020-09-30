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

#include "MDLShapeCostFunction.h"
#include "MathHelper.h"
#include "Definitions.h"

#include <math.h>
#include <iostream>

//#define OUTPUT_MODE_COMPACTNESS
#ifdef OUTPUT_MODE_COMPACTNESS
#include <iomanip>  
#endif

//#define OUTPUT_FUNCTION_VALUE
//#define CHECK_COVARIANCE_EIGENVALUES
//#define OUTPUT_TIME

#ifdef CHECK_COVARIANCE_EIGENVALUES
#include "MultilinearModel.h"
#endif

//#undef DEBUG_OUTPUT

#ifdef OUTPUT_TIME
#include "PerformanceCounter.h"
#endif

const double MODE_2_REGULARIZER = 0.01;
const double MODE_3_REGULARIZER = 0.01;

MDLShapeCostFunction::MDLShapeCostFunction(std::vector<double>& data, const std::vector<double>& initialParam, const size_t numIdentities, const size_t numExpressions, const std::vector<std::vector<double>>& vecCs, const std::vector<std::vector<double>>& matAs
											, const std::vector<std::vector<double>>& matWs, const std::vector<std::vector<double>>& sourcePointsVec, const std::vector<double>& vertexAreas, const double meanShapeArea, const double identityWeight
											, const double expressionWeight, const double maxParameterVariation, const size_t iShape)
: vnl_cost_function(2*data.size()/(3*numIdentities*numExpressions))
, m_data(data)
, m_initialParam(initialParam)
, m_dataSize(data.size())
, m_numIdentities(numIdentities)
, m_numExpressions(numExpressions)
, m_numSamples(numIdentities*numExpressions)
, m_sampleDataDim(m_dataSize/m_numSamples)
, m_sampleParamDim(2*m_sampleDataDim/3)
, m_numSampleVertices(m_sampleDataDim/3)
, m_vecCs(vecCs)
, m_matAs(matAs)
, m_matWs(matWs)
, m_sourcePointsVec(sourcePointsVec)
, m_vertexAreas(vertexAreas)
, m_meanShapeArea(meanShapeArea)
, m_identityWeight(identityWeight)
, m_expressionWeight(expressionWeight)
, m_lmkWeight(0.0)
, m_smoothWeight(0.0)
, m_maxSqrParameterVariation(std::pow(maxParameterVariation, 2))
, m_bUpdateData(false)
, m_iIndex(iShape%numIdentities)
, m_jIndex((iShape-m_iIndex)/numIdentities)
#ifdef OUTPUT_GRADIENT_NORM
, m_gradientNorm(0.0)
#endif
{
	MathHelper::centerData(m_data, m_sampleDataDim, m_centeredData, m_dataMean);

	m_dataParam_ij.resize(m_sampleParamDim, 0.0);

	const size_t dataOffset = iShape*m_sampleParamDim;
	for(size_t i = 0; i < m_sampleParamDim; ++i)
	{
		m_dataParam_ij[i] = m_initialParam[dataOffset+i];
	}
}

MDLShapeCostFunction::~MDLShapeCostFunction()
{

}

bool MDLShapeCostFunction::setLandmarks(const std::vector<size_t>& lmkIndices, std::vector<double>& lmks, const double lmkWeight)
{
	const size_t numLmks = lmkIndices.size();
	if(lmks.size() != 3*numLmks*m_numSamples)
	{
#ifdef DEBUG_OUTPUT
		std::cout << "Unable to set landmarks due to wrong dimension " << lmks.size() << " != " << numLmks*m_numSamples << std::endl;
#endif

		return false;
	}

	m_lmkIndices = lmkIndices;
	m_lmks = lmks;
	m_numLmks = numLmks;
	m_lmkWeight = lmkWeight;

	return true;
}

bool MDLShapeCostFunction::setSmoothnessValues(const std::vector<std::vector<size_t>>& precomputedSmoothnessIndices, const std::vector<std::vector<double>>& precomputedSmoothnessWeights, const double smoothnessWeight)
{
	if(precomputedSmoothnessIndices.size() != precomputedSmoothnessWeights.size())
	{
		return false;
	}

	m_precomputedSmoothnessIndices = precomputedSmoothnessIndices;
	m_precomputedSmoothnessWeights = precomputedSmoothnessWeights;
	m_smoothWeight = smoothnessWeight;
	return true;
}

void MDLShapeCostFunction::compute(const vnl_vector<double>& x, double *f, vnl_vector<double>* g)
{
#ifdef OUTPUT_TIME
	const double t1 = PerformanceCounter::getTime();
#endif

	updateShapeParametrization(x, m_iIndex, m_jIndex);

#ifdef OUTPUT_TIME
	const double t2 = PerformanceCounter::getTime();
#endif
	
	if(m_bUpdateData)
	{
		updateShapeData(m_iIndex, m_jIndex);

		MathHelper::centerData(m_data, m_sampleDataDim, m_centeredData, m_dataMean);
		m_bUpdateData = false;
	}

#ifdef OUTPUT_TIME
	const double t3 = PerformanceCounter::getTime();
#endif

	init(f, g);

#ifdef OUTPUT_TIME
	const double t4 = PerformanceCounter::getTime();
#endif

	std::vector<double> mode2EigenVectors; 
	std::vector<double> mode2EigenValues;
	computeCovarianceEigenVectors(2, mode2EigenVectors, mode2EigenValues);
	
#ifdef OUTPUT_TIME
	const double t5 = PerformanceCounter::getTime();
#endif

	std::vector<double> mode3EigenVectors; 
	std::vector<double> mode3EigenValues;
	computeCovarianceEigenVectors(3, mode3EigenVectors, mode3EigenValues);

/**
	{
		std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
		std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

		for(int a = 0; a < mode2EigenValues.size(); ++a)
		{
			//Version 1
			std::vector<double> dLambda_dXij_1;
			computeEigenValueDerivative(2, mode2EigenVectors, mode2EigenValues, a, m_iIndex, m_jIndex, dLambda_dXij_1);

			////Version 2
			//std::vector<double> dLambda_dXij_2;
			//dLambda_dXij_2.resize(m_sampleDataDim, 0.0);

			//const double factor1 = 2.0*mode2EigenVectors[a*16+m_iIndex];
			//for(size_t j = 0; j < 16; ++j)
			//{
			//	const double factor2 = mode2EigenVectors[a*16+j];

			//	for(size_t k = 0; k < m_sampleDataDim; ++k)
			//	{
			//		dLambda_dXij_2[k] += factor1*factor2*m_centeredData[j*m_sampleDataDim+k];
			//	}
			//}

			//Version 3
			std::vector<double> dLambda_dXij_3;
			dLambda_dXij_3.resize(m_sampleDataDim, 0.0);

			for(size_t j = 0; j < m_numIdentities; ++j)
			{
				const double factor1 = (2.0/(m_numIdentities*m_numExpressions*m_numExpressions))*mode2EigenVectors[a*m_numIdentities+j];

				for(size_t k = 0; k < m_numIdentities; ++k)
				{
					const double factor2 = mode2EigenVectors[a*m_numIdentities+k];
					
					for(size_t m = 0; m < m_numExpressions; ++m)
					{
						const double factor3 = (k==m_iIndex && m==m_jIndex) ? (15.0) : (-1.0);

						const size_t currDataOffset = j*m_sampleDataDim+m*m_sampleDataDim*m_numIdentities;
						for(size_t l = 0; l < m_sampleDataDim; ++l)
						{
							dLambda_dXij_3[l] += factor1*factor2*factor3*m_centeredData[currDataOffset+l];
						}
					}
				}
			}

			double diff(0.0);

			for(size_t i = 0; i < m_sampleDataDim; ++i)
			{
				//const double currDiff = dLambda_dXij_1[i] - dLambda_dXij_2[i];
				const double currDiff = dLambda_dXij_1[i] - dLambda_dXij_3[i];
				//std::cout << "                " << dLambda_dXij_1[i] << "   " << dLambda_dXij_3[i] << std::endl;
				//std::cout << "                " << dLambda_dXij_3[i] << "   " << dLambda_dXij_2[i] << std::endl;
				diff += std::pow(currDiff, 2.0);
			}

			diff = sqrt(diff);

			std::cout << "MODE 2 GRADIENT DIFF: " << a << " - " << diff << std::endl;
		}

		for(int a = 0; a < mode3EigenValues.size(); ++a)
		{
			//Version 1
			std::vector<double> dLambda_dXij_1;
			computeEigenValueDerivative(3, mode3EigenVectors, mode3EigenValues, a, m_iIndex, m_jIndex, dLambda_dXij_1);

			////Version 2
			//std::vector<double> dLambda_dXij_2;
			//dLambda_dXij_2.resize(m_sampleDataDim, 0.0);

			//for(size_t k = 0; k < m_sampleDataDim; ++k)
			//{
			//	dLambda_dXij_2[k] = 0.125*m_centeredData[m_iIndex*m_sampleDataDim+k];
			//}

			//Version 3
			std::vector<double> dLambda_dXij_3;
			dLambda_dXij_3.resize(m_sampleDataDim, 0.0);

			for(size_t j = 0; j < m_numExpressions; ++j)
			{
				const double factor1 = (2.0/(m_numIdentities*m_numExpressions*m_numIdentities))*mode3EigenVectors[a*m_numExpressions+j];

				for(size_t k = 0; k < m_numExpressions; ++k)
				{
					const double factor2 = mode3EigenVectors[a*m_numExpressions+k];
					
					for(size_t m = 0; m < m_numIdentities; ++m)
					{
						const double factor3 = (m==m_iIndex && k==m_jIndex) ? (15.0) : (-1.0);

						const size_t currDataOffset = m*m_sampleDataDim+j*m_sampleDataDim*m_numIdentities;
						for(size_t l = 0; l < m_sampleDataDim; ++l)
						{
							dLambda_dXij_3[l] += factor1*factor2*factor3*m_centeredData[currDataOffset+l];
						}
					}
				}
			}


			double diff(0.0);

			for(size_t i = 0; i < m_sampleDataDim; ++i)
			{
				const double currDiff = dLambda_dXij_1[i] - dLambda_dXij_3[i];
				//std::cout << "                " << dLambda_dXij_1[i] << "   " << dLambda_dXij_2[i] << std::endl;
				diff += std::pow(currDiff, 2.0);
			}

			diff = sqrt(diff);

			std::cout << "MODE 3 GRADIENT DIFF: " << a << " - " << diff << std::endl;
		}

		std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
		std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
	}

**/

#ifdef OUTPUT_TIME
	const double t6 = PerformanceCounter::getTime();
#endif

	addModeCompactnessEnergy(2, mode2EigenValues, f);

#ifdef OUTPUT_TIME
	const double t7 = PerformanceCounter::getTime();
#endif

	addModeCompactnessEnergy(3, mode3EigenValues, f);

#ifdef OUTPUT_TIME
	const double t8 = PerformanceCounter::getTime();
#endif


#ifdef OUTPUT_FUNCTION_VALUE
	const double compactnessValue = *f;
#endif

	std::vector<double> dXij_dalpha;
	computeShapeParameterDerivative(m_iIndex, m_jIndex, dXij_dalpha);

#ifdef OUTPUT_TIME
	const double t9 = PerformanceCounter::getTime();
#endif

	addModeShapeCompactnessGradient(2, mode2EigenVectors, mode2EigenValues, m_iIndex, m_jIndex, dXij_dalpha, g);

#ifdef OUTPUT_TIME
	const double t10 = PerformanceCounter::getTime();
#endif

	addModeShapeCompactnessGradient(3, mode3EigenVectors, mode3EigenValues, m_iIndex, m_jIndex, dXij_dalpha, g);

#ifdef OUTPUT_TIME
	const double t11 = PerformanceCounter::getTime();
#endif
	
	addShapeLandmarkEnergy(m_iIndex, m_jIndex, dXij_dalpha, f, g);

#ifdef OUTPUT_TIME
	const double t12 = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_FUNCTION_VALUE
	const double landmarkValue = *f-compactnessValue;
#endif

	addShapeSmoothnessEnergy(m_iIndex, m_jIndex, dXij_dalpha, f, g);

#ifdef OUTPUT_TIME
	const double t13 = PerformanceCounter::getTime();
#endif

#ifdef OUTPUT_GRADIENT_NORM
	m_gradientNorm = computeGradientNorm(g);
#endif

#ifdef OUTPUT_FUNCTION_VALUE
	std::cout << "Function evaluation" << std::endl;
	std::cout << "Compactness energy: " << compactnessValue << std::endl;
	std::cout << "Landmark energy: " << landmarkValue << std::endl;
	std::cout << "Smoothness energy: " << *f-landmarkValue-compactnessValue << std::endl;
	std::cout << "Function value: " << *f << std::endl;
#ifdef OUTPUT_GRADIENT_NORM
	std::cout << "Gradient norm: " << m_gradientNorm << std::endl;
#endif
	std::cout << std::endl;
#endif

#ifdef OUTPUT_TIME
	const double paramUpdateDiff = t2-t1; 
	const double dataUpdateDiff = t3-t2; 
	const double initParamDiff = t4-t3; 
	const double mode2EVsDiff = t5-t4; 
	const double mode3EvsDiff = t6-t5; 
	const double mode2CompEnergyDiff = t7-t6; 
	const double mode3CompEnergyDiff = t8-t7; 
	const double shapeParamDerivDiff = t9-t8; 
	const double mode2CompGradDiff = t10-t9; 
	const double mode3CompGradDiff = t11-t10; 
	const double lmkEnergyDiff = t12-t11; 
	const double smoothnessEnergyDiff = t13-t12; 
	const double overallTime = t13-t1; 

	std::cout << std::endl;
	std::cout << "Function evaluation timing: " << std::endl;
	std::cout << "Parameter update: " << paramUpdateDiff << " s  " << paramUpdateDiff/60 << " min" << std::endl;
	std::cout << "Data update: " << dataUpdateDiff << " s  " << dataUpdateDiff/60 << " min" << std::endl;
	std::cout << "Parameter init: " << initParamDiff << " s  " << initParamDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 EVs: " << mode2EVsDiff << " s  " << mode2EVsDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 EVs: " << mode3EvsDiff << " s  " << mode3EvsDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 Comp Energy: " << mode2CompEnergyDiff << " s  " << mode2CompEnergyDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 Comp Energy: " << mode3CompEnergyDiff << " s  " << mode3CompEnergyDiff/60 << " min" << std::endl;
	std::cout << "Shape param deriv: " << shapeParamDerivDiff << " s  " << shapeParamDerivDiff/60 << " min" << std::endl;
	std::cout << "Mode 2 Comp Grad: " << mode2CompGradDiff << " s  " << mode2CompGradDiff/60 << " min" << std::endl;
	std::cout << "Mode 3 Comp Grad: " << mode3CompGradDiff << " s  " << mode3CompGradDiff/60 << " min" << std::endl;
	std::cout << "Lmk Energy: " << lmkEnergyDiff << " s  " << lmkEnergyDiff/60 << " min" << std::endl;
	std::cout << "Smooth Energy: " << smoothnessEnergyDiff << " s  " << smoothnessEnergyDiff/60 << " min" << std::endl;
	std::cout << "Overall time: " << overallTime << " s  " << overallTime/60 << " min" << std::endl;
	std::cout << std::endl;
#endif


	m_bUpdateData = true;
/**
	{
		std::fstream outStream("E:\\Computations\\MDL\\TestData\\TestOutput\\DataComputeEnd.txt", std::ios::out);

		for(size_t iSample = 0; iSample < m_numSamples; ++iSample)
		{
			const size_t dataOffset = iSample*m_sampleDataDim;

			for(size_t j = 0; j < m_sampleDataDim; ++j)
			{
				outStream << m_data[dataOffset+j] << " ";
			}
			outStream << std::endl;
		}

		outStream.close();
	}
**/
}

void MDLShapeCostFunction::init(double* f, vnl_vector<double>* g)
{
	*f = 0.0;

	const size_t dim = m_sampleParamDim;
		
	//TODO Run in parallel
	//for(size_t i = 0; i < dim; ++i)
#pragma omp parallel for
	for(int i = 0; i < dim; ++i)
	{
		(*g)[i] = 0.0;
	}
}

void MDLShapeCostFunction::addModeCompactnessEnergy(const size_t mode, const std::vector<double>& modeEigenValues, double* f)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t numModeEigenValues = modeEigenValues.size();

	bool bMode2 = (mode == 2);
	const double modeWeight = (bMode2 ? m_identityWeight : m_expressionWeight)/static_cast<double>(numModeEigenValues);
	const double modeRegularizer = bMode2 ? MODE_2_REGULARIZER : MODE_3_REGULARIZER;

	if(modeWeight < math_eps)
	{
		return;
	}

	//Compute energy
	double value(0.0);
	for(size_t i = 0; i < numModeEigenValues; ++i)
	{
		value += log(modeEigenValues[i] + modeRegularizer);
		//value -= log(modeRegularizer);
	}

	value *= modeWeight;
	(*f) += value;
}

//void MDLShapeCostFunction::addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t iIndex, const size_t jIndex, vnl_vector<double>* g)
void MDLShapeCostFunction::addModeShapeCompactnessGradient(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha
																			, vnl_vector<double>* g)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t numModeEigenValues = modeEigenValues.size();

	bool bMode2 = (mode == 2);
	const double modeWeight = (bMode2 ? m_identityWeight : m_expressionWeight)/static_cast<double>(numModeEigenValues);
	const double modeRegularizer = bMode2 ? MODE_2_REGULARIZER : MODE_3_REGULARIZER;

	if(modeWeight < math_eps)
	{
		return;
	}

	//Optimized version
	std::vector<double> tmpMemoryMatrix;
	tmpMemoryMatrix.resize(m_sampleParamDim * numModeEigenValues, 0.0);

	//Declared outside tue to performance reasons
	//std::vector<double> tmpSum;
	//tmpSum.resize(2*m_numSampleVertices, 0.0);

	//Pre-compute the sum over all expressions (mode == 2) or identities (mode == 3)
	std::vector<double> modeDataSum;
	computeModeDataSum(mode, modeDataSum);

#pragma omp parallel for
	for(int a = 0; a < numModeEigenValues; ++a)
	{
		const size_t eigenValueOffset = a*m_sampleParamDim;

		// Dimension 2*n x 3
		// n matrices of dimension 2 x 3
		//std::vector<double> dXij_dalpha;
		//computeShapeParameterDerivative(iIndex, jIndex, dXij_dalpha);

		// Dimension 3*n
		std::vector<double> dLambda_dXij;
		computeEigenValueDerivative(mode, modeDataSum, modeEigenVectors, modeEigenValues, a, iIndex, jIndex, dLambda_dXij);

/**
		std::vector<double> dLambda_dXij_Old;
		computeEigenValueDerivativeOld(mode, modeEigenVectors, modeEigenValues, a, iIndex, jIndex, dLambda_dXij_Old);

		double old_new_diff(0.0);
		for(size_t i = 0; i < dLambda_dXij.size(); ++i)
		{
			const double currDiff = dLambda_dXij_Old[i]-dLambda_dXij[i];
			old_new_diff += std::pow(currDiff, 2);
		}

		old_new_diff = sqrt(old_new_diff);
		if(old_new_diff > math_eps)
		{
			for(int i = 0; i < 1; ++i)
			{
				for(int j = 0; j < 1; ++j)
				{
					std::cout << "XXX Difference of old and new eigenvalue derivative: " << old_new_diff << std::endl;
					std::cout << "Mode: " << mode << std::endl;
					std::cout << "iIndex: " << iIndex << " jIndex: " << jIndex << std::endl;
					std::cout << "Eigenvalue: " << a << std::endl;
				}
			}

			for(int b = 0; b < m_sampleParamDim; ++b)
			{
				(*g)[b] = 0.0;
			}
		}
		else
		{
			std::cout << "Ok " << " Mode: " << mode << " iIndex: " << iIndex << " jIndex: " << jIndex << " Eigenvalue: " << a << std::endl;
		}
**/

		//std::vector<double> tmpSum;
		//tmpSum.resize(2*m_numSampleVertices, 0.0);

		for(size_t b = 0; b < m_numSampleVertices; ++b)
		{
			const size_t tmp2b = 2*b;
			const size_t tmp3b = 3*b;

			// (dXij[v_b] / dalpha[v_b]) * (dLambda / dXij[v_b])
			const double val1 = dXij_dalpha[tmp2b]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+2];
			const double val2 = dXij_dalpha[tmp2b+1]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+2];

			tmpMemoryMatrix[eigenValueOffset+tmp2b] = val1;
			tmpMemoryMatrix[eigenValueOffset+tmp2b+1] = val2;
		}
	}

#pragma omp parallel for
	for(int b = 0; b < m_sampleParamDim; ++b)
	{
		for(int a = 0; a < numModeEigenValues; ++a)
		{
			const size_t eigenValueOffset = a*m_sampleParamDim;

			// d E / d lambda
			const double dE_dlambda = 1.0/(modeEigenValues[a]+modeRegularizer);
			const double factor = modeWeight*dE_dlambda;
		
			(*g)[b] += factor*tmpMemoryMatrix[eigenValueOffset+b];
		}
	}

	//Not optimized version
	//for(size_t a = 0; a < numModeEigenValues; ++a)
	//{
	//	// d E / d lambda
	//	const double dE_dlambda = 1.0/(modeEigenValues[a]+modeRegularizer);

	//	// Dimension 2*n x 3
	//	// n matrices of dimension 2 x 3
	//	//std::vector<double> dXij_dalpha;
	//	//computeShapeParameterDerivative(iIndex, jIndex, dXij_dalpha);

	//	// Dimension 3*n
	//	std::vector<double> dLambda_dXij;
	//	computeEigenValueDerivative(mode, modeEigenVectors, modeEigenValues, a, iIndex, jIndex, dLambda_dXij);

	//	std::vector<double> tmpSum;
	//	tmpSum.resize(2*m_numSampleVertices, 0.0);

	//	for(size_t b = 0; b < m_numSampleVertices; ++b)
	//	{
	//		const size_t tmp2b = 2*b;
	//		const size_t tmp3b = 3*b;

	//		// (dXij[v_b] / dalpha[v_b]) * (dLambda / dXij[v_b])
	//		const double val1 = dXij_dalpha[tmp2b]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b]*dLambda_dXij[tmp3b+2];
	//		const double val2 = dXij_dalpha[tmp2b+1]*dLambda_dXij[tmp3b] + dXij_dalpha[m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+1] + dXij_dalpha[2*m_sampleParamDim+tmp2b+1]*dLambda_dXij[tmp3b+2];

	//		tmpSum[tmp2b] = val1;
	//		tmpSum[tmp2b+1] = val2;
	//	}

	//	//Mode 2: w_2 * (dE / dLambda)
	//	//Mode 3: w_3 * (dE / dLambda)
	//	const double factor = modeWeight*dE_dlambda;
	//	for(size_t b = 0; b < m_sampleParamDim; ++b)
	//	{
	//		(*g)[b] += factor*tmpSum[b];
	//	}
	//}
}

//void MDLShapeCostFunction::addShapeLandmarkEnergy(const size_t iIndex, const size_t jIndex, double* f, vnl_vector<double>* g)
void MDLShapeCostFunction::addShapeLandmarkEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g)
{
	if(m_numLmks == 0 || m_lmkWeight < math_eps)
	{
		return;
	}

	//const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	const double regularizerFactor = m_lmkWeight / static_cast<double>(m_numLmks*d2*d3);

	const size_t iSample = (jIndex*d2 + iIndex);
	const size_t dataOffset = iSample*m_sampleDataDim;
	//const size_t dataParamOffset = (jIndex*d2 + iIndex)*m_sampleParamDim;
	const size_t lmkOffset = iSample*3*m_numLmks;

	//TODO compute derivative only of landmark points
	//std::vector<double> dXij_dalpha;
	//computeShapeParameterDerivative(iIndex, jIndex, dXij_dalpha);

	double value(0.0);

	for(size_t iLmk = 0; iLmk < m_numLmks; ++iLmk)
	{
		const size_t currLmkIndex = m_lmkIndices[iLmk];

		const size_t tmp2currLmkIndex = 2*currLmkIndex;
		const size_t currLmkDataOffset = dataOffset+3*currLmkIndex;
		const size_t currLmkOffset = lmkOffset+3*iLmk;

		const double tmpX = m_data[currLmkDataOffset]-m_lmks[currLmkOffset];
		const double tmpY = m_data[currLmkDataOffset+1]-m_lmks[currLmkOffset+1];
		const double tmpZ = m_data[currLmkDataOffset+2]-m_lmks[currLmkOffset+2];
		value += pow(tmpX, 2) + pow(tmpY, 2), + pow(tmpZ, 2);

		const double val1 = dXij_dalpha[tmp2currLmkIndex]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currLmkIndex]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currLmkIndex]*tmpZ;
		const double val2 = dXij_dalpha[tmp2currLmkIndex+1]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currLmkIndex+1]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currLmkIndex+1]*tmpZ;

		(*g)[tmp2currLmkIndex] += 2.0*regularizerFactor*val1;
		(*g)[tmp2currLmkIndex+1] += 2.0*regularizerFactor*val2;
	}

	value *= regularizerFactor; 
	(*f) += value;
}

//void MDLShapeCostFunction::addShapeSmoothnessEnergy(const size_t iIndex, const size_t jIndex, double* f, vnl_vector<double>* g)
void MDLShapeCostFunction::addShapeSmoothnessEnergy(const size_t iIndex, const size_t jIndex, const std::vector<double>& dXij_dalpha, double* f, vnl_vector<double>* g)
{
	if(m_precomputedSmoothnessIndices.empty() || m_smoothWeight < math_eps)
	{
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	//const size_t d3 = m_numExpressions;

	//Since we only optimize for one shape and therefore only the smoothness of current shape is considered for optimization,
	//there is no regularization w.r.t. d2 and d3
	//const double regularizerFactor = m_smoothWeight / static_cast<double>(m_numSampleVertices*d2*d3);
	const double regularizerFactor = m_smoothWeight / static_cast<double>(m_numSampleVertices);

	//std::vector<double> dXij_dalpha;
	//computeShapeParameterDerivative(iIndex, jIndex, dXij_dalpha);

	const size_t dataOffset = (jIndex*d2 + iIndex)*d1;
	//const size_t dataParamOffset = (jIndex*d2 + iIndex)*m_sampleParamDim;

	const size_t numVertices = m_precomputedSmoothnessIndices.size();
	for(size_t iVertex = 0; iVertex < numVertices; ++iVertex)
	{
		const std::vector<size_t>& currSmoothnessIndices = m_precomputedSmoothnessIndices[iVertex];
		const std::vector<double>& currSmoothnessWeights = m_precomputedSmoothnessWeights[iVertex];

		if(currSmoothnessIndices.empty())
		{
			continue;
		}

		double tmpX(0.0);
		double tmpY(0.0);
		double tmpZ(0.0);

		const size_t currNumSmoothnessIndices = currSmoothnessIndices.size();
		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			const size_t currVertexStartIndex = 3*currSmoothnessIndices[j];
			const size_t currSmoothStartIndex = dataOffset + currVertexStartIndex;
			const double currSmoothWeight = currSmoothnessWeights[j];

			tmpX += currSmoothWeight*(m_data[currSmoothStartIndex]);
			tmpY += currSmoothWeight*(m_data[currSmoothStartIndex+1]);
			tmpZ += currSmoothWeight*(m_data[currSmoothStartIndex+2]);

			//tmpX += currSmoothWeight*(m_data[currSmoothStartIndex]);
			//tmpY += currSmoothWeight*(m_data[currSmoothStartIndex+1]);
			//tmpZ += currSmoothWeight*(m_data[currSmoothStartIndex+2]);
		}

		(*f) += regularizerFactor*(std::pow(tmpX, 2) + std::pow(tmpY, 2) + std::pow(tmpZ, 2));

		for(size_t j = 0; j < currNumSmoothnessIndices; ++j)
		{
			const size_t currVertexIndex = currSmoothnessIndices[j];

			const size_t tmp2currVertexIndex = 2*currVertexIndex;
								
			const double val1 = dXij_dalpha[tmp2currVertexIndex]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currVertexIndex]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currVertexIndex]*tmpZ;
			const double val2 = dXij_dalpha[tmp2currVertexIndex+1]*tmpX + dXij_dalpha[m_sampleParamDim+tmp2currVertexIndex+1]*tmpY + dXij_dalpha[2*m_sampleParamDim+tmp2currVertexIndex+1]*tmpZ;

			const double currSmoothWeight = 2.0*regularizerFactor*currSmoothnessWeights[j];
			(*g)[tmp2currVertexIndex] += currSmoothWeight*val1;
			(*g)[tmp2currVertexIndex+1] += currSmoothWeight*val2;
		}
	}
}

//OLD VERSION
//void MDLShapeCostFunction::computeEigenValueDerivativeOld(const size_t mode, const std::vector<double>& modeEigenVectors, const std::vector<double>& modeEigenValues, const size_t eigenValueIndex
//																		, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij)
//{
//	if(mode != 2 && mode != 3)
//	{
//		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
//		return;
//	}
//
//	const size_t d1 = m_sampleDataDim;
//	const size_t d2 = m_numIdentities;
//	const size_t d3 = m_numExpressions;
//
//	const size_t e2 = (mode == 2) ? d2 : d3;
//	const size_t e3 = (mode == 2) ? d3 : d2;
//
//#ifdef DEBUG_OUTPUT
//	const size_t numEigenValues = modeEigenValues.size();
//	const size_t eigenVectorDim = modeEigenVectors.size()/numEigenValues;
//
//	if(eigenVectorDim != e2)
//	{
//		std::cout << "Eigenvector of wrong dimension " << eigenVectorDim << " != " << e2 << std::endl;
//	}
//#endif
//
//	//Dimension 3n
//	dLambda_dXij.resize(d1, 0.0);
//
//	const size_t d1d2 = d1*d2;
//	const size_t d2d3 = d2*d3;
//	const size_t aEigenvectorIndex = eigenValueIndex*e2;
//
//	//Optimized version
//	//Declared outside tue to performance reasons
//	std::vector<double> tmpFactors;
//	tmpFactors.resize(e2, 0.0);
//
//	const bool bMode2 = (mode == 2);
//	for(int r = 0; r < e2; ++r)
//	{
//		// e_a[r]   r-th entry of a-th eigenvector
//		const double ev_a_r = modeEigenVectors[aEigenvectorIndex+r];
//
//		for(size_t l = 0; l < e3; ++l)
//		{
//			//*Mode 2: sum_(l=1)^(d3) ev_a[s]*(d2*d3*delta_is*delta_jl - 1)
//			//*Mode 3: sum_(l=1)^(d2) ev_a[s]*(d2*d3*delta_il*delta_js - 1)
//			//*double tmpFactor(0.0);
//		#pragma omp parallel for
//			for(int s = 0; s < e2; ++s)
//			{
//				//Kronecker deltas
//				//Mode 2: delta_is
//				//Mode 3: delta_il
//				const size_t p = bMode2 ? s : l;
//				const double delta_ip = (iIndex == p) ? 1.0 : 0.0;
//
//				//Mode 2: delta_jl
//				//Mode 3: delta_js
//				const size_t q = bMode2 ? l : s;
//				const double delta_jq = (jIndex == q) ? 1.0 : 0.0;
//
//				//Mode 2: d2*d3*delta_is*delta_jl
//				//Mode 3: d2*d3*delta_il*delta_js
//				const double d2d3_delta_ip_delta_jq = d2d3*delta_ip*delta_jq;
//
//				// e_a[s]   r-th entry of a-th eigenvector
//				const double ev_a_s = modeEigenVectors[aEigenvectorIndex+s];
//
//				//*tmpFactor += ev_a_s*(d2d3_delta_ip_delta_jq-1.0);
//
//				//Mode 2: ev_a[r]*ev_a[s]*(d2*d3*delta_is*delta_jl - 1.0)
//				//Mode 3: ev_a[r]*ev_a[s]*(d2*d3*delta_il*delta_js - 1.0)
//				tmpFactors[s] = ev_a_r*ev_a_s*(d2d3_delta_ip_delta_jq-1.0);
//			}
//
//			//tmpFactor *= ev_a_r;
//
//			const size_t r2l3 = bMode2 ? r : l;
//			const size_t l2r3 = bMode2 ? l : r;
//			const size_t dataStartIndex = r2l3*d1+l2r3*d1d2;
//
//		#pragma omp parallel for
//			for(int k = 0; k < d1; ++k)
//			{
//				//Mode 2: dLambda_dXij[k] += tmpFactor*c_rl[k]
//				//			c_jl[k] -> r*d1 + l*d1*d2 + k
//				//Mode 3: dLambda_dXij[k] += tmpFactor*c_lr[k]
//				//			c_lj[k] -> l*d1 + r*d1*d2 + k
//				//*dLambda_dXij[k] += tmpFactor*m_centeredData[dataStartIndex+k];
//				for(int s = 0; s < e2; ++s)
//				{
//					dLambda_dXij[k] += tmpFactors[s]*m_centeredData[dataStartIndex+k];
//				}
//			}			
//		}
//	}
//
///*
//	const bool bMode2 = (mode == 2);
//	for(int r = 0; r < e2; ++r)
//	{
//		// e_a[r]   r-th entry of a-th eigenvector
//		const double ev_a_r = modeEigenVectors[aEigenvectorIndex+r];
//
//		for(size_t s = 0; s < e2; ++s)
//		{
//			// e_a[s]   r-th entry of a-th eigenvector
//			const double ev_a_s = modeEigenVectors[aEigenvectorIndex+s];
//
//			for(size_t l = 0; l < e3; ++l)
//			{
//				//Kronecker deltas
//				//Mode 2: delta_is
//				//Mode 3: delta_il
//				const size_t p = bMode2 ? s : l;
//				const double delta_ip = (iIndex == p) ? 1.0 : 0.0;
//
//				//Mode 2: delta_jl
//				//Mode 3: delta_js
//				const size_t q = bMode2 ? l : s;
//				const double delta_jq = (jIndex == q) ? 1.0 : 0.0;
//
//				//Mode 2: d2*d3*delta_is*delta_jl
//				//Mode 3: d2*d3*delta_il*delta_js
//				const double d2d3_delta_ip_delta_jq = d2d3*delta_ip*delta_jq;
//				
//				//Mode 2: ev_a[r]*ev_a[s]*d2*d3*delta_is*delta_jl
//				//Mode 3: ev_a[r]*ev_a[s]*d2*d3*delta_il*delta_js
//				const double tmpFactor = ev_a_r*ev_a_s*(d2d3_delta_ip_delta_jq-1.0);
//
//				const size_t r2l3 = bMode2 ? r : l;
//				const size_t l2r3 = bMode2 ? l : r;
//				const size_t dataStartIndex = r2l3*d1+l2r3*d1d2;
//
//				for(size_t k = 0; k < d1; ++k)
//				{
//					//Mode 2: dLambda_dXij[k] += tmpFactor*c_rl[k]
//					//			c_jl[k] -> r*d1 + l*d1*d2 + k
//					//Mode 3: dLambda_dXij[k] += tmpFactor*c_lr[k]
//					//			c_lj[k] -> l*d1 + r*d1*d2 + k
//					dLambda_dXij[k] += tmpFactor*m_centeredData[dataStartIndex+k];
//				}
//			}
//		}
//	}
//*/
//
//	const double d2d3e3A = d2d3*e3*m_meanShapeArea;
//	
//	//TODO Run in parallel
//	//for(size_t k = 0; k < m_numSampleVertices; ++k)
//#pragma omp parallel for		
//	for(int k = 0; k < m_numSampleVertices; ++k)
//	{
//		//Mode 2: 2*deltaA(v_k) / (d2*d3*d3*A)
//		//Mode 3: 2*deltaA(v_k) / (d2*d3*d2*A)
//		const double factor = (2.0*m_vertexAreas[k])/d2d3e3A;
//
//		const size_t tmp3k = 3*k;
//		dLambda_dXij[tmp3k] *= factor;
//		dLambda_dXij[tmp3k+1] *= factor;
//		dLambda_dXij[tmp3k+2] *= factor;
//	}
//}

void MDLShapeCostFunction::computeEigenValueDerivative(const size_t mode, const std::vector<double>& precomputedModeDataSum, const std::vector<double>& modeEigenVectors, const std::vector<double>& /*modeEigenValues*/, const size_t eigenValueIndex
																		, const size_t iIndex, const size_t jIndex, std::vector<double>& dLambda_dXij)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "Mode specification incorrect " << mode << " != " << 2 << " and " << mode << " != " << 3 << std::endl;
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;


	const bool bMode2 = (mode == 2);
	const size_t e2 = bMode2 ? d2 : d3;
	const size_t e3 = bMode2 ? d3 : d2;

//#ifdef DEBUG_OUTPUT
//	const size_t numEigenValues = modeEigenValues.size();
//	const size_t eigenVectorDim = modeEigenVectors.size()/numEigenValues;
//
//	if(eigenVectorDim != e2)
//	{
//		std::cout << "Eigenvector of wrong dimension " << eigenVectorDim << " != " << e2 << std::endl;
//	}
//#endif

	//Dimension 3n
	dLambda_dXij.resize(d1, 0.0);

	const size_t d1d2 = d1*d2;
	const size_t d2d3 = d2*d3;
	const size_t aEigenvectorIndex = eigenValueIndex*e2;

	// e_a[i]   i-th entry of a-th eigenvector
	// Mode2: ev_a_e = e_a[iIndex]
	// Mode3: ev_a_e = e_a[jIndex]
	const double ev_a_e = bMode2 ? modeEigenVectors[aEigenvectorIndex+iIndex] : modeEigenVectors[aEigenvectorIndex+jIndex];

	double ev_a_sum(0.0);
	for(size_t s = 0; s < e2; ++s)
	{
		// e_a[s]   r-th entry of a-th eigenvector
		const double ev_a_s = modeEigenVectors[aEigenvectorIndex+s];
		ev_a_sum += ev_a_s;
	}

	for(int r = 0; r < e2; ++r)
	{
		// e_a[r]   r-th entry of a-th eigenvector
		const double ev_a_r = modeEigenVectors[aEigenvectorIndex+r];

		const size_t r2i3 = bMode2 ? r : iIndex;
		const size_t j2r3 = bMode2 ? jIndex : r;
		//const size_t dataStartIndex = (jIndex*d2+r)*d1;
		//const size_t dataStartIndex = (r*d2+iIndex)*d1;
		const size_t dataStartIndex = r2i3*d1 + j2r3*d1d2;

		const double d2d3ev_a_e = d2d3*ev_a_e;

#pragma omp parallel for
		for(int k = 0; k < d1; ++k)
		{
			//Mode2: e_ad2*d3
			const double currValue = d2d3ev_a_e*m_centeredData[dataStartIndex+k]-ev_a_sum*precomputedModeDataSum[r*d1+k];
			dLambda_dXij[k] += ev_a_r*currValue;
		}
	}
	
/**
	std::vector<double> dLambda_dXij_TEST;
	dLambda_dXij_TEST.resize(d1, 0.0);

	for(int r = 0; r < e2; ++r)
	{
		// e_a[r]   r-th entry of a-th eigenvector
		const double ev_a_r = modeEigenVectors[aEigenvectorIndex+r];

		for(size_t s = 0; s < e2; ++s)
		{
			// e_a[s]   r-th entry of a-th eigenvector
			const double ev_a_s = modeEigenVectors[aEigenvectorIndex+s];

			for(size_t l = 0; l < e3; ++l)
			{
				//Kronecker deltas
				//Mode 2: delta_is
				//Mode 3: delta_il
				const size_t p = bMode2 ? s : l;
				const double delta_ip = (iIndex == p) ? 1.0 : 0.0;

				//Mode 2: delta_jl
				//Mode 3: delta_js
				const size_t q = bMode2 ? l : s;
				const double delta_jq = (jIndex == q) ? 1.0 : 0.0;

				//Mode 2: d2*d3*delta_is*delta_jl
				//Mode 3: d2*d3*delta_il*delta_js
				const double d2d3_delta_ip_delta_jq = d2d3*delta_ip*delta_jq;
				
				//Mode 2: ev_a[r]*ev_a[s]*d2*d3*delta_is*delta_jl
				//Mode 3: ev_a[r]*ev_a[s]*d2*d3*delta_il*delta_js
				const double tmpFactor = ev_a_r*ev_a_s*(d2d3_delta_ip_delta_jq-1.0);

				const size_t r2l3 = bMode2 ? r : l;
				const size_t l2r3 = bMode2 ? l : r;
				const size_t dataStartIndex = r2l3*d1+l2r3*d1d2;

				for(size_t k = 0; k < d1; ++k)
				{
					//Mode 2: dLambda_dXij[k] += tmpFactor*c_rl[k]
					//			c_jl[k] -> r*d1 + l*d1*d2 + k
					//Mode 3: dLambda_dXij[k] += tmpFactor*c_lr[k]
					//			c_lj[k] -> l*d1 + r*d1*d2 + k
					dLambda_dXij_TEST[k] += tmpFactor*m_centeredData[dataStartIndex+k];
				}
			}
		}
	}

	double diff(0.0);
	for(size_t i = 0; i < d1; ++i)
	{
		const double currDiff = dLambda_dXij_TEST[i]-dLambda_dXij[i];
		diff += std::pow(currDiff, 2);
	}

	diff = sqrt(diff);
	std::cout << "DIFFERENCE: " << diff << std::endl;
**/

	const double d2d3e3A = d2d3*e3*m_meanShapeArea;
	
	//TODO Run in parallel
	//for(size_t k = 0; k < m_numSampleVertices; ++k)
#pragma omp parallel for		
	for(int k = 0; k < m_numSampleVertices; ++k)
	{
		//Mode 2: 2*deltaA(v_k) / (d2*d3*d3*A)
		//Mode 3: 2*deltaA(v_k) / (d2*d3*d2*A)
		const double factor = (2.0*m_vertexAreas[k])/d2d3e3A;

		const size_t tmp3k = 3*k;
		dLambda_dXij[tmp3k] *= factor;
		dLambda_dXij[tmp3k+1] *= factor;
		dLambda_dXij[tmp3k+2] *= factor;
	}
}

void MDLShapeCostFunction::computeShapeParameterDerivative(const size_t iIndex, const size_t jIndex, std::vector<double>& dXij_dalpha)
{
	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;

	//Dimension 2n x 3
	dXij_dalpha.resize(d1*m_sampleParamDim, 0.0);

	const size_t ijIndex = jIndex*d2+iIndex;
	const std::vector<double>& matAij = m_matAs[ijIndex];
	const std::vector<double>& matWij = m_matWs[ijIndex];
	const std::vector<double>& currSourcePoints = m_sourcePointsVec[ijIndex];

	const double sqr_math_eps = std::pow(math_eps, 2);

	for(size_t k = 0; k < m_numSampleVertices; ++k)
	{
		// d sigma / d alpha 
		//Dimension 2 x n
		std::vector<double> tmpDerivMatrix;
		tmpDerivMatrix.resize(2*m_numSampleVertices, 0.0);

		const double alphaij_xk_1 = m_dataParam_ij[2*k];
		const double alphaij_xk_2 = m_dataParam_ij[2*k+1];

	#pragma omp parallel for
		for(int i = 0; i < m_numSampleVertices; ++i)
		{
			const size_t tmp2i = 2*i;

			const double uDiff = alphaij_xk_1 - currSourcePoints[2*i];
			const double vDiff = alphaij_xk_2 - currSourcePoints[2*i+1];
			const double tmpSqrLength = std::pow(uDiff, 2) + std::pow(vDiff, 2);
			if(tmpSqrLength > sqr_math_eps)
			{
				const double factor = 2.0*log(sqrt(tmpSqrLength))+1.0;
				tmpDerivMatrix[tmp2i] = factor*uDiff;
				tmpDerivMatrix[tmp2i+1] = factor*vDiff;
			}
		}

		//Dimension 2 x 3
		std::vector<double> outMat;
		MathHelper::matrixMult(tmpDerivMatrix, 2, m_numSampleVertices, "N", matWij, 3, outMat);

		//dXij_dalpha[v_k] / dalpha[v_k] = A_ij^T + tmpDerivMatrix*W_ij
		const size_t tmp2k = 2*k;

	//#pragma omp parallel for
		for(int col = 0; col < 3; ++col)
		{
			const size_t col2 = col*2;
			const size_t col2n = col*m_sampleParamDim;
			for(size_t row = 0; row < 2; ++row)
			{
				dXij_dalpha[col2n+tmp2k+row] = matAij[row*3+col] + outMat[col2+row];
			}
		}
	}
}

void MDLShapeCostFunction::updateShapeParametrization(const vnl_vector<double>& x, const size_t iIndex, const size_t jIndex)
{
	const size_t d2 = m_numIdentities;

	const size_t iSample = jIndex*d2+iIndex;
	const size_t paramOffset = iSample*m_sampleParamDim;

#pragma omp parallel for
	for(int iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
	{
		const size_t tmpParamStartIndex = 2*iVertex;

		double deltaAlpha1 = x[tmpParamStartIndex];
		double deltaAlpha2 = x[tmpParamStartIndex+1];
		const double tmpSqrLength = std::pow(deltaAlpha1,2) + std::pow(deltaAlpha2, 2);
		if(tmpSqrLength > m_maxSqrParameterVariation)
		{
			const double factor = sqrt(m_maxSqrParameterVariation/tmpSqrLength);
			deltaAlpha1 *= factor;
			deltaAlpha2 *= factor;
		}

		const double initAlpha1 = m_initialParam[paramOffset+tmpParamStartIndex];
		const double initAlpha2 = m_initialParam[paramOffset+tmpParamStartIndex+1];

		m_dataParam_ij[tmpParamStartIndex] = initAlpha1+deltaAlpha1;
		m_dataParam_ij[tmpParamStartIndex+1] = initAlpha2+deltaAlpha2; 
	}
}

void MDLShapeCostFunction::updateShapeData(const size_t iIndex, const size_t jIndex)
{
	//const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;

	const size_t iSample = jIndex*d2+iIndex;
	const size_t dataOffset = iSample*m_sampleDataDim;

	const std::vector<double>& currVecC = m_vecCs[iSample]; 
	const std::vector<double>& currMatA = m_matAs[iSample];
	const std::vector<double>& currMatW = m_matWs[iSample];
	const std::vector<double>& currSourcePoints = m_sourcePointsVec[iSample];

	//TODO Run in parallel
	//for(size_t iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
#pragma omp parallel for
	for(int iVertex = 0; iVertex < m_numSampleVertices; ++iVertex)
	{
		const size_t tmpDataStartIndex = 3*iVertex;
		const size_t tmpParamStartIndex = 2*iVertex;
		
		std::vector<double> paramVertex;
		paramVertex.push_back(m_dataParam_ij[tmpParamStartIndex]);
		paramVertex.push_back(m_dataParam_ij[tmpParamStartIndex+1]);

		std::vector<double> outDataVertex;
		if(!MathHelper::evaluateInterpolation(currVecC, currMatA, currMatW, currSourcePoints, paramVertex, outDataVertex))
		{
			std::cout << "MDLShapeCostFunction::updateData(...) - unable to compute tps point" << std::endl;
			continue;
		}

		m_data[dataOffset+tmpDataStartIndex] = outDataVertex[0];
		m_data[dataOffset+tmpDataStartIndex+1] = outDataVertex[1];
		m_data[dataOffset+tmpDataStartIndex+2] = outDataVertex[2];
	}
}

void MDLShapeCostFunction::computeModeDataSum(const size_t mode, std::vector<double>& modeDataSum)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "MDLShapeCostFunction::computeModeDataSum(...) - wrong mode input " << mode << std::endl;
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	modeDataSum.clear();

	bool bMode2 = (mode == 2);

	const size_t e2 = bMode2 ? d2 : d3;
	modeDataSum.resize(e2*d1, 0.0);

	if(mode == 2)
	{
		//Computes sum over all expressions (x_i = sum_{m=1}^{d3} c_{im})
#pragma omp parallel for	
		for(int r = 0; r < d2; ++r)
		{
			const size_t rd1 = r*d1;
			for(int m = 0; m < d3; ++m)
			{
				const size_t startDataIndex = (m*d2+r)*d1;

				for(int k = 0; k < d1; ++k)
				{
					modeDataSum[rd1+k] += m_centeredData[startDataIndex+k];
				}
			}
		}
	}
	else
	{
		//Computes sum over all identities (x_j = sum_{m=1}^{d2} c_{mj})
#pragma omp parallel for	
		for(int r = 0; r < d3; ++r)
		{
			const size_t rd1 = r*d1;
			for(int m = 0; m < d2; ++m)
			{
				const size_t startDataIndex = (r*d2+m)*d1;

				for(int k = 0; k < d1; ++k)
				{
					modeDataSum[rd1+k] += m_centeredData[startDataIndex+k];
				}
			}
		}
	}
}

void MDLShapeCostFunction::computeCovarianceEigenVectors(const size_t mode, std::vector<double>& eigenVectors, std::vector<double>& eigenValues)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "MDLShapeCostFunction::computeCovarianceEigenVectors(...) - wrong mode input " << mode << std::endl;
		return;
	}

	size_t dim(0);
	std::vector<double> covarianceMatrix;
	computeCovarianceMatrix(mode, covarianceMatrix, dim);

	std::vector<double> tmpEigenVectors;
	std::vector<double> tmpEigenValues;
	MathHelper::computeLeftSingularVectors(covarianceMatrix, dim, dim, tmpEigenVectors, tmpEigenValues);

	size_t numEigenValues(0);
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		if(fabs(tmpEigenValues[i]) > math_eps)
		{
			++numEigenValues;
		}
	}

/**
	numEigenValues = tmpEigenValues.size();
**/

	eigenVectors.resize(dim*numEigenValues);
	eigenValues.resize(numEigenValues);

	//TODO Run in parallel
	//for(size_t i = 0; i < numEigenValues; ++i)
#pragma omp parallel for
	for(int i = 0; i < numEigenValues; ++i)
	{
		const size_t startIndex = i*dim;
		for(size_t j = 0; j < dim; ++j)
		{
			const size_t index = startIndex+j;
			eigenVectors[index] = tmpEigenVectors[index];
		}

		eigenValues[i] = tmpEigenValues[i];
	}

#ifdef CHECK_COVARIANCE_EIGENVALUES
	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	Tensor dataTensor;
	dataTensor.init(m_data, d1, d2, d3);

	std::cout << "MM Covariance eigenvalues" << std::endl;

	if(mode == 2)
	{
		//Multiplication mode 2
		std::vector<double> A2; // d2 x d1*d3
		dataTensor.unfold(2, A2);

		std::vector<double> U2; // d2 x d2	(d2 x min(d2, d1*d3))
		std::vector<double> S2; // d2 x d1*d3 (vector of size min(d2, d1*d3))
		//std::vector<double> V2; // d1*d3 x d1*d3
		MathHelper::computeLeftSingularVectors(A2, d2, d1*d3, U2, S2);

		std::cout << "Num eigenvalues: " << S2.size() << std::endl;
		std::cout << "EV: ";
		for(size_t i = 0; i < S2.size(); ++i)
		{
			const double currEV = std::pow(S2[i], 2.0)/static_cast<double>(d3);
			std::cout << currEV << " ";
		}
		std::cout << std::endl;
	}
	else
	{
		//Multiplication mode 3
		std::vector<double> A3; // d3 x d1*d2
		dataTensor.unfold(3, A3);

		std::vector<double> U3;	// d3 x d3	(d3 x min(d3, d1*d2))
		std::vector<double> S3; // d3 x d1*d2 (vector of size d3 x min(d3, d1*d2))
		//std::vector<double> V3; // d1*d2 x d1*d2
		MathHelper::computeLeftSingularVectors(A3, d3, d1*d2, U3, S3);

		std::cout << "Num eigenvalues: " << S3.size() << std::endl;
		std::cout << "EV: ";
		for(size_t i = 0; i < S3.size(); ++i)
		{
			const double currEV = std::pow(S3[i], 2.0)/static_cast<double>(d2);
			std::cout << currEV << " ";
		}
		std::cout << std::endl;
	}
#endif

#ifdef OUTPUT_MODE_COMPACTNESS
	double sum(0.0);
	std::cout << "EigenValues: ";
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		std::cout << tmpEigenValues[i] << " ";
		sum += tmpEigenValues[i];
	}

	std::cout << std::endl;
	std::cout << "Compactness mode " << mode << std::endl;
	
	double tmpSum(0.0);
	for(size_t i = 0; i < tmpEigenValues.size(); ++i)
	{
		tmpSum += tmpEigenValues[i];
		const double currValue = 100.0*tmpSum/sum;
		std::cout << std::setprecision(3) <<  tmpSum << " (" << currValue << ") " << std::endl;
	}
#endif
}

void MDLShapeCostFunction::computeCovarianceMatrix(const size_t mode, std::vector<double>& covarianceMatrix, size_t& matrixDim)
{
	if(mode != 2 && mode != 3)
	{
		std::cout << "MDLShapeCostFunction::computeCovarianceMatrix(...) - wrong mode input " << mode << std::endl;
		return;
	}

	const size_t d1 = m_sampleDataDim;
	const size_t d2 = m_numIdentities;
	const size_t d3 = m_numExpressions;

	const size_t e2 = (mode == 2) ? d2 : d3;
	const size_t e3 = (mode == 2) ? d3 : d2;

	matrixDim = e2;
	covarianceMatrix.resize(e2*e2, 0.0);

	//Mode 2: 1/(d3*A)
	//Mode 3: 1/(d2*A)
	const double e3AreaReg = 1.0/(static_cast<double>(e3*m_meanShapeArea));

	//Mode 2: d2
	//Mode 3: 1
	const size_t x2_2 = (mode == 2) ? d2 : 1;
	
	//Mode 2: 1
	//Mode 3: d2
	const size_t x2_3 = (mode == 3) ? d2 : 1;

	//TODO Run in parallel
	//for(size_t i = 0; i < e2; ++i)
#pragma omp parallel for
	for(int i = 0; i < e2; ++i)
	{
		//Mode 2: i*d1
		//Mode 3: i*d1*d2
		const size_t id1x = i*d1*x2_3;

		for(size_t j = i; j < e2; ++j)
		{
			//Mode 2: j*d1
			//Mode 3: j*d1*d2
			const size_t jd1x = j*d1*x2_3;

			double value(0.0);

			for(size_t r = 0; r < e3; ++r)
			{
				//Mode 2: r*d1*d2
				//Mode 3: r*d1
				const size_t rd1x = r*d1*x2_2;

				for(size_t k = 0; k < m_numSampleVertices; ++k)
				{
					//Mode 2
					// c_ir[v_k] -> i*d1 + r*d1*d2 + 3*k
					// c_jr[v_k] -> j*d1 + r*d1*d2 + 3*k

					//Mode 3
					// c_ri[v_k] -> i*d1*d2 + r*d1 + 3*k
					// c_rj[v_k] -> j*d1*d2 + r*d1 + 3*k

					const size_t iStartIndex = id1x + rd1x + 3*k;
					const size_t jStartIndex = jd1x + rd1x + 3*k;

					//Mode 2: c_ir[v_k] * c_jr[v_k]
					//Mode 3: c_ri[v_k] * c_rj[v_k]
					double tmpValue(0.0);
					//tmpValue += m_data[iStartIndex]*m_data[jStartIndex];
					//tmpValue += m_data[iStartIndex+1]*m_data[jStartIndex+1];
					//tmpValue += m_data[iStartIndex+2]*m_data[jStartIndex+2];
					tmpValue += m_centeredData[iStartIndex]*m_centeredData[jStartIndex];
					tmpValue += m_centeredData[iStartIndex+1]*m_centeredData[jStartIndex+1];
					tmpValue += m_centeredData[iStartIndex+2]*m_centeredData[jStartIndex+2];
					
					tmpValue *= m_vertexAreas[k];
					value += tmpValue;
				}
			}

			value *= e3AreaReg;

			//The covariance matrix is symmetric
			covarianceMatrix[i*e2+j] = value;
			covarianceMatrix[j*e2+i] = value;
		}
	}
}

#ifdef OUTPUT_GRADIENT_NORM
double MDLShapeCostFunction::computeGradientNorm(const vnl_vector<double>* g) const
{
	double norm(0.0);

	for(size_t i = 0; i < m_sampleParamDim; ++i)
	{
		norm += std::pow((*g)[i], 2);
	}

	return sqrt(norm);
}
#endif

