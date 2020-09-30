::Adjust program path
call ..\VS2010\MultilinearMDL\Release\MM_MDL.exe -tps ..\Example FileCollection.txt tpsFileCollection.txt ..\Example\TextureCoordinates.txt
call ..\VS2010\MultilinearMDL\Release\MM_MDL.exe -opt ..\Example FileCollection.txt tpsFileCollection.txt ..\Example\OuterBoundaryIndices.txt ..\Example\InnerBoundaryIndices.txt ..\Example\Result
pause