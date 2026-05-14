// lib/tensor/peps.hpp
#pragma once

#include "tensor/tensor.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ds_tn
{

class Peps
{
  public:
    enum class ConfigValidity : u8
    {
        valid = 0,
        too_few_rows,
        too_few_cols,
        bond_dim_too_small,
        physical_dim_too_small
    };

    struct Config
    {
        usize n_rows{3};
        usize n_cols{3};
        usize bond_dim{2};
        usize physical_dim{2};
        bool fully_padded{false};

        [[nodiscard]] auto check_validity() const -> ConfigValidity;
        [[nodiscard]] auto to_string() const -> std::string;
    };

    struct MetadataConfig
    {
        bool include_classname{true};
        bool include_memory{false};
        usize memory_digits{2};
    };

    using Scalar = double;

    static constexpr usize k_dummy_dim{1zu};

    static constexpr usize k_leg_right{0zu};
    static constexpr usize k_leg_top{1zu};
    static constexpr usize k_leg_left{2zu};
    static constexpr usize k_leg_bottom{3zu};
    static constexpr usize k_leg_physical{4zu};

    explicit Peps(Config cfg);

    [[nodiscard]] static auto to_string(ConfigValidity validity) -> std::string_view;

    [[nodiscard]] auto config() const noexcept -> const Config&;
    [[nodiscard]] auto n_rows() const noexcept -> usize;
    [[nodiscard]] auto n_cols() const noexcept -> usize;
    [[nodiscard]] auto bond_dim() const noexcept -> usize;
    [[nodiscard]] auto physical_dim() const noexcept -> usize;
    [[nodiscard]] auto fully_padded() const noexcept -> bool;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto tensors() noexcept -> std::span<Tensor>;
    [[nodiscard]] auto tensors() const noexcept -> std::span<const Tensor>;
    [[nodiscard]] auto operator()(usize row, usize col) -> Tensor&;
    [[nodiscard]] auto operator()(usize row, usize col) const -> const Tensor&;
    [[nodiscard]] auto at(usize row, usize col) -> Tensor&;
    [[nodiscard]] auto at(usize row, usize col) const -> const Tensor&;
    [[nodiscard]] auto total_entries() const noexcept -> usize;
    auto print_metadata(MetadataConfig cfg) const -> void;
    auto print_metadata() const -> void;

  private:
    [[nodiscard]] static auto is_valid(ConfigValidity validity) -> bool;
    [[nodiscard]] auto storage_index(usize row, usize col) const -> usize;

    Config cfg_{};
    std::vector<Tensor> nodes_{};
};

struct RandomPepsConfig
{
    usize physical_dim{2};
    usize bond_dim{2};
    bool fully_padded{false};

    RandomOptions random_options{RandomNormalOptions{.mu = 0.0, .sigma = 0.1}};
    std::optional<TensorSeed> seed{};
    bool apply_algebraic_power_law_suppression{false};
};

[[nodiscard]] auto random_peps(usize n_rows, usize n_cols, RandomPepsConfig cfg = {}) -> Peps;

}  // namespace ds_tn
