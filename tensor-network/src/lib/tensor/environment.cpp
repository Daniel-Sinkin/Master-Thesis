// lib/tensor/environment.cpp
#include "tensor/environment.hpp"

#include "tensor/contraction.hpp"

#include <array>
#include <format>
#include <stdexcept>
#include <string>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto prefixed_leg_name(std::string_view prefix, const std::string& leg_name)
    -> std::string
{
    return std::format("{}_{}", prefix, leg_name);
}

auto require_valid_mps_tensor(const Tensor& tensor, const char* function_name, const char* argument_name)
    -> void
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            std::format("{} requires {} to be a valid Tensor.", function_name, argument_name)
        );
    }
    if (!tensor.is_tensor3())
    {
        throw std::invalid_argument(
            std::format("{} requires {} to be rank-3.", function_name, argument_name)
        );
    }
}

auto require_valid_mpo_tensor(const Tensor& tensor, const char* function_name, const char* argument_name)
    -> void
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            std::format("{} requires {} to be a valid Tensor.", function_name, argument_name)
        );
    }
    if (tensor.rank() != 4)
    {
        throw std::invalid_argument(
            std::format("{} requires {} to be rank-4.", function_name, argument_name)
        );
    }
}

auto require_compatible_site_tensors(
    const Tensor& mps_tensor, const Tensor& mpo_tensor, const char* function_name
) -> void
{
    require_valid_mps_tensor(mps_tensor, function_name, "mps_tensor");
    require_valid_mpo_tensor(mpo_tensor, function_name, "mpo_tensor");

    if (mps_tensor.shape(1) != mpo_tensor.shape(1) or mps_tensor.shape(1) != mpo_tensor.shape(2))
    {
        throw std::invalid_argument(
            std::format(
                "{} requires the MPS physical extent to match both MPO physical extents.",
                function_name
            )
        );
    }
}

[[nodiscard]] auto expected_right_environment_leg_names(
    const Tensor& mps_tensor, const Tensor& mpo_tensor
) -> std::array<std::string, 3>
{
    return {
        prefixed_leg_name("bra", mps_tensor.leg_name(2)),
        mpo_tensor.leg_name(3),
        prefixed_leg_name("ket", mps_tensor.leg_name(2)),
    };
}

[[nodiscard]] auto expected_right_environment_shape(
    const Tensor& mps_tensor, const Tensor& mpo_tensor
) -> std::array<usize, 3>
{
    return {
        mps_tensor.shape(2),
        mpo_tensor.shape(3),
        mps_tensor.shape(2),
    };
}

auto require_compatible_right_environment(
    const Tensor& right_environment, const Tensor& mps_tensor, const Tensor& mpo_tensor
) -> void
{
    if (right_environment.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            "update_right_environment requires right_environment to be a valid Tensor."
        );
    }
    if (!right_environment.is_tensor3())
    {
        throw std::invalid_argument(
            "update_right_environment requires right_environment to be rank-3."
        );
    }

    const auto expected_shape = expected_right_environment_shape(mps_tensor, mpo_tensor);
    if (!std::ranges::equal(right_environment.shape(), std::span<const usize>{expected_shape}))
    {
        throw std::invalid_argument(
            "update_right_environment requires right_environment to have compatible bond extents."
        );
    }

    const auto expected_leg_names = expected_right_environment_leg_names(mps_tensor, mpo_tensor);
    if (!std::ranges::equal(
            right_environment.leg_names(), std::span<const std::string>{expected_leg_names}
        ))
    {
        throw std::invalid_argument(
            "update_right_environment requires right_environment to have compatible leg names."
        );
    }
}

[[nodiscard]] auto environment_ket_tensor(const Tensor& mps_tensor, const Tensor& mpo_tensor)
    -> Tensor
{
    auto ket = mps_tensor;
    ket.rename_leg(mps_tensor.leg_name(0), prefixed_leg_name("ket", mps_tensor.leg_name(0)));
    ket.rename_leg(mps_tensor.leg_name(1), mpo_tensor.leg_name(2));
    ket.rename_leg(mps_tensor.leg_name(2), prefixed_leg_name("ket", mps_tensor.leg_name(2)));
    return ket;
}

[[nodiscard]] auto environment_bra_tensor(const Tensor& mps_tensor, const Tensor& mpo_tensor)
    -> Tensor
{
    auto bra = mps_tensor;
    bra.rename_leg(mps_tensor.leg_name(0), prefixed_leg_name("bra", mps_tensor.leg_name(0)));
    bra.rename_leg(mps_tensor.leg_name(1), mpo_tensor.leg_name(1));
    bra.rename_leg(mps_tensor.leg_name(2), prefixed_leg_name("bra", mps_tensor.leg_name(2)));
    return bra;
}

auto require_compatible_mps_mpo(const MPS& mps, std::span<const Tensor> mpo, const char* function_name)
    -> void
{
    if (mps.size() == 0 or mpo.empty())
    {
        throw std::invalid_argument(std::format("{} requires non-empty MPS and MPO chains.", function_name));
    }
    if (mps.size() != mpo.size())
    {
        throw std::invalid_argument(std::format("{} requires mps.size() == mpo.size().", function_name));
    }

    for (auto site = 0zu; site < mps.size(); ++site)
    {
        require_compatible_site_tensors(mps[site], mpo[site], function_name);
    }

    for (auto site = 0zu; site + 1 < mps.size(); ++site)
    {
        if (mps[site].leg_name(2) != mps[site + 1].leg_name(0))
        {
            throw std::invalid_argument(
                std::format(
                    "{} requires adjacent MPS tensors to share matching bond leg names.",
                    function_name
                )
            );
        }
        if (mpo[site].leg_name(3) != mpo[site + 1].leg_name(0))
        {
            throw std::invalid_argument(
                std::format(
                    "{} requires adjacent MPO tensors to share matching bond leg names.",
                    function_name
                )
            );
        }
    }
}

}  // namespace

auto right_boundary_environment(const Tensor& mps_tensor, const Tensor& mpo_tensor) -> Tensor
{
    require_compatible_site_tensors(mps_tensor, mpo_tensor, "right_boundary_environment");

    if (mps_tensor.shape(2) != 1 or mpo_tensor.shape(3) != 1)
    {
        throw std::invalid_argument(
            "right_boundary_environment requires trivial right boundary bond dimensions."
        );
    }

    auto out = Tensor{
        std::vector<usize>{1, 1, 1},
        expected_right_environment_leg_names(mps_tensor, mpo_tensor),
    };
    out.data()[0] = 1.0;
    return out;
}

auto
update_right_environment(const Tensor& right_environment, const Tensor& mps_tensor, const Tensor& mpo_tensor)
    -> Tensor
{
    require_compatible_site_tensors(mps_tensor, mpo_tensor, "update_right_environment");
    require_compatible_right_environment(right_environment, mps_tensor, mpo_tensor);

    const auto tmp = contract(mpo_tensor, environment_ket_tensor(mps_tensor, mpo_tensor));
    const auto local = contract(environment_bra_tensor(mps_tensor, mpo_tensor), tmp);
    return contract(local, right_environment);
}

auto right_environments(const MPS& mps, std::span<const Tensor> mpo) -> std::vector<Tensor>
{
    require_compatible_mps_mpo(mps, mpo, "right_environments");

    auto out = std::vector<Tensor>(mps.size());
    out.back() = right_boundary_environment(mps[mps.size() - 1], mpo[mpo.size() - 1]);

    for (auto site = mps.size() - 1; site > 0; --site)
    {
        out[site - 1] = update_right_environment(out[site], mps[site], mpo[site]);
    }

    return out;
}

}  // namespace ds_tn
