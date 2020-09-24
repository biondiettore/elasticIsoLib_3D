#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "BornElasticShotsGpu_3D.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyElastic_iso_float_born_3D, clsGeneric) {

	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

	py::class_<BornElasticShotsGpu_3D, std::shared_ptr<BornElasticShotsGpu_3D>>(clsGeneric,"BornElasticShotsGpu_3D")
			.def(py::init<std::shared_ptr<SEP::float4DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<SEP::float3DReg>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>, std::vector<std::shared_ptr<spaceInterpGpu_3D>>>(), "Initialize a BornElasticShotsGpu_3D")

			.def("forward", (void (BornElasticShotsGpu_3D::*)(const bool, const std::shared_ptr<SEP::float4DReg>, std::shared_ptr<SEP::float4DReg>)) &BornElasticShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (BornElasticShotsGpu_3D::*)(const bool, const std::shared_ptr<SEP::float4DReg>, std::shared_ptr<SEP::float4DReg>)) &BornElasticShotsGpu_3D::adjoint, "Adjoint")

			.def("setBackground", (void (BornElasticShotsGpu_3D::*)(std::shared_ptr<SEP::float4DReg>)) &BornElasticShotsGpu_3D::setBackground, "setBackground")

			.def("dotTest",(bool (BornElasticShotsGpu_3D::*)(const bool, const float)) &BornElasticShotsGpu_3D::dotTest,"Dot-Product Test")

			;

}
