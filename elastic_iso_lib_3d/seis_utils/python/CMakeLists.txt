project(python-seis_utils3D)

set(PythonMods elasticParamConvertModule_3D.py dataCompModule_3D.py)
set(PythonMains generateGeom.py dataCompMain_3D.py)

set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install(FILES ${PythonMods}  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ DESTINATION ${PYTHON_DIR})
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ DESTINATION bin)
