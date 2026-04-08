// lib/models/transverse_ising.hpp
#pragma once

#include "tensor/tensor.hpp"

#include <vector>

namespace ds_tn
{

[[nodiscard]] auto transverse_ising_mpo(usize num_sites, f64 J, f64 h) -> std::vector<Tensor>;

}  // namespace ds_tn
