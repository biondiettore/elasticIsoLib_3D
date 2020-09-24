cmake_minimum_required(VERSION 2.8)
project(ELASTIC_ISO_FLOAT_3D-lib)


#Adding include files
include_directories(include)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS ${INC_DIR}/spaceInterpGpu_3D.h
						${INC_DIR}/stagger_3D.h
						${INC_DIR}/fdParamElastic_3D.h
						${INC_DIR}/nonlinearPropElasticShotsGpu_3D.h
						${INC_DIR}/nonlinearPropElasticGpu_3D.h
						${INC_DIR}/nonlinearPropElasticGpuFunctions_3D.h
						${INC_DIR}/BornElasticShotsGpu_3D.h
)

#Files to be compiled
set(SRC_DIR src)
#Files for nonlinear operator library
set(SOURCE_NL ${SRC_DIR}/nonlinearPropElasticShotsGpu_3D.cpp
							${SRC_DIR}/spaceInterpGpu_3D.cpp
							${SRC_DIR}/nonlinearPropElasticGpu_3D.cpp
							${SRC_DIR}/stagger_3D.cpp
							${SRC_DIR}/fdParamElastic_3D.cpp
							${SRC_DIR}/nonlinearPropElasticGpuFunctions_3D.cu
)

#Files for Born operator library
set(SOURCE_BORN ${SRC_DIR}/BornElasticShotsGpu_3D.cpp
								${SRC_DIR}/spaceInterpGpu_3D.cpp
								${SRC_DIR}/BornElasticGpu_3D.cpp
								${SRC_DIR}/stagger_3D.cpp
								${SRC_DIR}/fdParamElastic_3D.cpp
								${SRC_DIR}/BornElasticGpuFunctions_3D.cu
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
								 ${hypercube_BINARY_DIR}
								 ${buffers-lib_BINARY_DIR}
								 ${json_BINARY_DIR}
								 ${sepVector-lib_BINARY_DIR}
								 ${SEPlib_LIBDIR}
)


#Setting io libraries
set(LIBS genericCpp sepVector sep3d sep)

#Compiling the project library
add_library(ELASTIC_ISO_FLOAT_NL_3D ${SOURCE_NL})
target_link_libraries(ELASTIC_ISO_FLOAT_NL_3D ${LIBS} OpenMP::OpenMP_CXX)
add_library(ELASTIC_ISO_FLOAT_BORN_3D ${SOURCE_BORN})
target_link_libraries(ELASTIC_ISO_FLOAT_BORN_3D ${LIBS} OpenMP::OpenMP_CXX)

# Compiling the project executables
# Model padding program for GPU
add_executable(padElasticFileGpuMain_3D ${SRC_DIR}/padElasticFileGpuMain_3D.cpp)
target_link_libraries(padElasticFileGpuMain_3D ${LIBS} OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS ELASTIC_ISO_FLOAT_NL_3D DESTINATION lib)
install(TARGETS ELASTIC_ISO_FLOAT_BORN_3D DESTINATION lib)
install(TARGETS padElasticFileGpuMain_3D DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
