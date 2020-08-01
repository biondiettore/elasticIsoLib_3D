#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// #include "nonlinearPropShotsGpu_3D.h"
#include "spaceInterpGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyElastic_iso_double_nl_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

	py::class_<spaceInterpGpu_3D, std::shared_ptr<spaceInterpGpu_3D>>(clsGeneric, "spaceInterpGpu_3D")

      .def(py::init<const std::shared_ptr<double1DReg>, const std::shared_ptr<double1DReg>, const std::shared_ptr<double1DReg>, const std::shared_ptr<SEP::hypercube>, int &, std::shared_ptr<paramObj>, int, double, double, double, std::string, int>(), "Initialize a spaceInterpGpu_3D object using zcoord, xcoord, ycoord, elastic model, and nt")

      ;

//   py::class_<nonlinearPropShotsGpu_3D, std::shared_ptr<nonlinearPropShotsGpu_3D>>(clsGeneric,"nonlinearPropShotsGpu_3D")
//
//       .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<deviceGpu_3D>>>(), "Initialize a nonlinearPropShotsGpu_3D")
//
//       // Constructor for Ginsu
//       .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, std::vector<int>, std::vector<int>>(), "Initialize a nonlinearPropShotsGpu_3D with Ginsu")
//
//       .def("forward", (void (nonlinearPropShotsGpu_3D::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::forward, "Forward")
//
//       .def("adjoint", (void (nonlinearPropShotsGpu_3D::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::adjoint, "Adjoint")
//
//       .def("setVel_3D",(void (nonlinearPropShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::setVel_3D,"Function to set background velocity")
//
//       .def("dotTest",(bool (nonlinearPropShotsGpu_3D::*)(const bool, const double)) &nonlinearPropShotsGpu_3D::dotTest,"Dot-Product Test")
//
//       .def("getDampVolumeShots_3D",(std::shared_ptr<double3DReg> (nonlinearPropShotsGpu_3D::*)()) &nonlinearPropShotsGpu_3D::getDampVolumeShots_3D,"Function to get the damping volume computed on the CPU for debugging")
//
// ;

}
