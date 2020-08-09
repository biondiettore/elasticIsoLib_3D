#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "nonlinearPropElasticShotsGpu_3D.h"
// #include "spaceInterpGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyElastic_iso_double_nl_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

	py::class_<spaceInterpGpu_3D, std::shared_ptr<spaceInterpGpu_3D>>(clsGeneric, "spaceInterpGpu_3D")

      .def(py::init<const std::shared_ptr<double1DReg>, const std::shared_ptr<double1DReg>, const std::shared_ptr<double1DReg>, const std::shared_ptr<SEP::hypercube>, int &, std::shared_ptr<paramObj>, int, double, double, double, std::string, int>(), "Initialize a spaceInterpGpu_3D object using zcoord, xcoord, ycoord, elastic model, and nt")

      ;

  py::class_<nonlinearPropElasticShotsGpu_3D, std::shared_ptr<nonlinearPropElasticShotsGpu_3D>>(clsGeneric,"nonlinearPropElasticShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double4DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>>(), "Initialize a nonlinearPropElasticShotsGpu_3D")

      .def("forward", (void (nonlinearPropElasticShotsGpu_3D::*)(const bool, const std::shared_ptr<double4DReg>, std::shared_ptr<double4DReg>)) &nonlinearPropElasticShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (nonlinearPropElasticShotsGpu_3D::*)(const bool, const std::shared_ptr<double4DReg>, std::shared_ptr<double4DReg>)) &nonlinearPropElasticShotsGpu_3D::adjoint, "Adjoint")

      .def("setBackground",(void (nonlinearPropElasticShotsGpu_3D::*)(std::shared_ptr<double4DReg>)) &nonlinearPropElasticShotsGpu_3D::setBackground,"Function to set background elastic model")

      .def("dotTest",(bool (nonlinearPropElasticShotsGpu_3D::*)(const bool, const double)) &nonlinearPropElasticShotsGpu_3D::dotTest,"Dot-Product Test")


;

}
