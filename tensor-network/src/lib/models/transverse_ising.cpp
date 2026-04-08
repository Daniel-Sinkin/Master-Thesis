// lib/models/transverse_ising.cpp
#include "models/transverse_ising.hpp"

#include "ndarray/ndarray.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto mpo_leg_names(usize site, usize num_sites) -> std::array<std::string, 4>
{
    const auto left_leg = [&]
    {
        if (site == 0) return std::string{"edge_left"};
        return std::format("bond_{}{}", site - 1, site);
    };
    const auto right_leg = [&]
    {
        if (site + 1 == num_sites) return std::string{"edge_right"};
        return std::format("bond_{}{}", site, site + 1);
    };
    return {
        left_leg(),
        std::format("physical_out_{}", site),
        std::format("physical_in_{}", site),
        right_leg(),
    };
}

auto set_block(Tensor& tensor, usize left_bond, usize right_bond, const NDArray& op) -> void
{
    for (auto physical_out = 0zu; physical_out < op.shape(0); ++physical_out)
    {
        for (auto physical_in = 0zu; physical_in < op.shape(1); ++physical_in)
        {
            const auto val = op(physical_out, physical_in);
            tensor(left_bond, physical_out, physical_in, right_bond) = val;
        }
    }
}

}  // namespace

auto transverse_ising_mpo(usize num_sites, f64 J, f64 h) -> std::vector<Tensor>
{
    if (num_sites == 0)
    {
        throw std::invalid_argument("transverse_ising_mpo requires num_sites >= 1.");
    }

    const auto I = NDArray::matrix({
        {1.0, 0.0},
        {0.0, 1.0},
    });
    auto X = NDArray::matrix({
        {0.0, 1.0},
        {1.0, 0.0},
    });
    auto Z = NDArray::matrix({
        {1.0, 0.0},
        {0.0, -1.0},
    });

    auto minus_h_X = X;
    minus_h_X *= -h;

    auto minus_J_Z = Z;
    minus_J_Z *= -J;

    auto out = std::vector<Tensor>{};
    out.reserve(num_sites);

    if (num_sites == 1)
    {
        auto site = Tensor({1, 2, 2, 1}, mpo_leg_names(0, num_sites));
        set_block(site, 0, 0, minus_h_X);
        out.push_back(std::move(site));
        return out;
    }

    auto first = Tensor({1, 2, 2, 3}, mpo_leg_names(0, num_sites));
    set_block(first, 0, 0, minus_h_X);
    set_block(first, 0, 1, minus_J_Z);
    set_block(first, 0, 2, I);
    out.push_back(std::move(first));

    for (auto site_idx = 1zu; site_idx + 1 < num_sites; ++site_idx)
    {
        auto site = Tensor({3, 2, 2, 3}, mpo_leg_names(site_idx, num_sites));
        set_block(site, 0, 0, I);
        set_block(site, 1, 0, Z);
        set_block(site, 2, 0, minus_h_X);
        set_block(site, 2, 1, minus_J_Z);
        set_block(site, 2, 2, I);
        out.push_back(std::move(site));
    }

    auto last = Tensor({3, 2, 2, 1}, mpo_leg_names(num_sites - 1, num_sites));
    set_block(last, 0, 0, I);
    set_block(last, 1, 0, Z);
    set_block(last, 2, 0, minus_h_X);
    out.push_back(std::move(last));

    return out;
}

}  // namespace ds_tn
