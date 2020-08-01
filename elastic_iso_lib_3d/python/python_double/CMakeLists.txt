cmake_minimum_required(VERSION 2.8)
project(pyElastic_iso_double_3D)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ELASTIC_ISO_DOUBLE library
include_directories(../../elastic_iso_double/include)

# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
  #PYBIND11 library headers
  include_directories(${PYTHON_INCLUDE_DIR})
  include_directories(${pybind11_INCLUDE_DIR})
  include_directories(${PYTHON_INCLUDE_PATH})
  #library headers
  # include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../elastic_isotropic_operators")

  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_PATH})

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${GENERIC_EXPORT_INC_DIRS})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  #SEP LIBRARY DIRECTORIES
  link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
  )

  set(LIBS genericCpp buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

  # Compiling and adding module to library
  pybind11_add_module(pyElastic_iso_double_nl_3D MODULE pyElastic_iso_double_nl_3D.cpp)
  target_link_libraries(pyElastic_iso_double_nl_3D PRIVATE ELASTIC_ISO_DOUBLE_NL_3D ${LIBS})

  # Installing library
	set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  install( TARGETS pyElastic_iso_double_nl_3D DESTINATION ${PYTHON_DIR}) # Module for nonlinear operator
  # install( FILES Elastic_iso_double_3D.py DESTINATION ${PYTHON_DIR})

  # Installing Python executables
  # install( FILES nonlinearPythonDoubleMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)
