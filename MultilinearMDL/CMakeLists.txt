SET( MM_MDL_Files
	FileLoader.cpp
	FileWriter.cpp
	MathHelper.cpp
	MDLHelper.cpp
	MDLShapeCostFunction.cpp
	MDLShapeCostFunctionPCA.cpp
	MultilinearModel.cpp
	PerformanceCounter.cpp
	Main.cpp
)

INCLUDE_DIRECTORIES(${CLAPACK_INCLUDE_DIR}) 
INCLUDE_DIRECTORIES(${ITK_INCLUDES}) 
   
ADD_EXECUTABLE(MM_MDL ${MM_MDL_Files})
TARGET_LINK_LIBRARIES(MM_MDL ${ITK_LIBRARIES} ${CLAPACK_LIBRARIES})   