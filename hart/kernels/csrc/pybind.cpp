#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "layernorm/layernorm.h"
#include "rope/fused_rope.h"
#include "rope/fused_rope_with_pos.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rope_forward_func", &fused_rope_forward_func,
        "Fused rope forward function");
  m.def("fused_rope_with_pos_forward_func", &fused_rope_with_pos_forward_func,
        "Fused rope forward function with B,S,D embedding");
  m.def("fused_rope_backward_func", &fused_rope_backward_func,
        "Fused rope backward function");
  m.def("rms_norm", &rms_norm, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("epsilon"), py::arg("use_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}
