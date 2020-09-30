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

#include "PerformanceCounter.h"

#ifdef _WIN32
#include <windows.h> 
#else
#include <sys/time.h>
#include <cstddef>
#endif


double PerformanceCounter::getTime()
{
#ifdef _WIN32
	LONGLONG f1, t1;
	if(!QueryPerformanceFrequency((LARGE_INTEGER*)&f1))
	{
		return 0.0;
	}

	if(!QueryPerformanceCounter((LARGE_INTEGER*)&t1))
	{
		return 0.0;
	}

	return static_cast<double>(t1)/static_cast<double>(f1);
#else
	 struct timeval t1;
    if (gettimeofday(&t1, NULL) != 0)
	 {
		 return 0.0;
    }

	 return static_cast<double>(t1.tv_sec) + 0.000001*static_cast<double>(t1.tv_usec); 
#endif
}
