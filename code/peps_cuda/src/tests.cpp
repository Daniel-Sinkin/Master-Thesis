#include "peps_cuda/memory.hpp"
#include "peps_cuda/peps.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

peps_cuda::Complex cx(double real, double imag = 0.0) {
    return {static_cast<peps_cuda::Real>(real),
            static_cast<peps_cuda::Real>(imag)};
}

double tol(double fp64_tol, double fp32_tol) {
    return (sizeof(peps_cuda::Real) == sizeof(float)) ? fp32_tol : fp64_tol;
}

peps_cuda::PEPS make_all_zero_product_peps(int lx, int ly) {
    peps_cuda::PEPS peps;
    peps.lx = lx;
    peps.ly = ly;
    peps.local_dim = 2;
    peps.max_bond_dim = 1;
    peps.sites.resize(static_cast<std::size_t>(lx * ly));
    for (int x = 0; x < lx; ++x) {
        for (int y = 0; y < ly; ++y) {
            peps_cuda::SiteTensor site;
            site.phys = 2;
            site.north = 1;
            site.east = 1;
            site.south = 1;
            site.west = 1;
            site.data = {{1.0, 0.0}, {0.0, 0.0}};
            peps.at(x, y) = std::move(site);
        }
    }
    return peps;
}

std::vector<int> decode_julia_column_major_index(std::size_t linear,
                                                 const std::vector<int> &dims) {
    std::vector<int> indices(dims.size(), 0);
    for (std::size_t axis = 0; axis < dims.size(); ++axis) {
        indices[axis] = static_cast<int>(linear % static_cast<std::size_t>(dims[axis]));
        linear /= static_cast<std::size_t>(dims[axis]);
    }
    return indices;
}

peps_cuda::SiteTensor
make_site_from_julia_theta(const std::vector<peps_cuda::Complex> &theta,
                           std::size_t &cursor,
                           const std::vector<char> &link_dirs) {
    peps_cuda::SiteTensor tensor;
    tensor.phys = 2;
    tensor.north = 1;
    tensor.east = 1;
    tensor.south = 1;
    tensor.west = 1;
    for (const char dir : link_dirs) {
        if (dir == 'n') {
            tensor.north = 2;
        } else if (dir == 'e') {
            tensor.east = 2;
        } else if (dir == 's') {
            tensor.south = 2;
        } else if (dir == 'w') {
            tensor.west = 2;
        } else {
            throw std::runtime_error("unknown Julia fixture link direction");
        }
    }
    tensor.data.assign(tensor.parameter_count(), cx(0.0));

    std::vector<int> julia_dims = {tensor.phys};
    for (const char dir : link_dirs) {
        if (dir == 'n') {
            julia_dims.push_back(tensor.north);
        } else if (dir == 'e') {
            julia_dims.push_back(tensor.east);
        } else if (dir == 's') {
            julia_dims.push_back(tensor.south);
        } else {
            julia_dims.push_back(tensor.west);
        }
    }

    const std::size_t count = tensor.parameter_count();
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::vector<int> indices =
            decode_julia_column_major_index(linear, julia_dims);
        int n = 0;
        int e = 0;
        int s = 0;
        int w = 0;
        for (std::size_t axis = 0; axis < link_dirs.size(); ++axis) {
            const int value = indices[axis + 1];
            const char dir = link_dirs[axis];
            if (dir == 'n') {
                n = value;
            } else if (dir == 'e') {
                e = value;
            } else if (dir == 's') {
                s = value;
            } else {
                w = value;
            }
        }
        tensor(indices[0], n, e, s, w) = theta[cursor + linear];
    }
    cursor += count;
    return tensor;
}

void append_site_values_in_julia_order(
    const peps_cuda::SiteTensor &tensor,
    const std::vector<peps_cuda::Complex> &site_major_values,
    std::size_t site_offset, const std::vector<char> &link_dirs,
    std::vector<peps_cuda::Complex> &out) {
    std::vector<int> julia_dims = {tensor.phys};
    for (const char dir : link_dirs) {
        if (dir == 'n') {
            julia_dims.push_back(tensor.north);
        } else if (dir == 'e') {
            julia_dims.push_back(tensor.east);
        } else if (dir == 's') {
            julia_dims.push_back(tensor.south);
        } else if (dir == 'w') {
            julia_dims.push_back(tensor.west);
        } else {
            throw std::runtime_error("unknown Julia fixture link direction");
        }
    }

    const std::size_t count = tensor.parameter_count();
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::vector<int> indices =
            decode_julia_column_major_index(linear, julia_dims);
        int n = 0;
        int e = 0;
        int s = 0;
        int w = 0;
        for (std::size_t axis = 0; axis < link_dirs.size(); ++axis) {
            const int value = indices[axis + 1];
            const char dir = link_dirs[axis];
            if (dir == 'n') {
                n = value;
            } else if (dir == 'e') {
                e = value;
            } else if (dir == 's') {
                s = value;
            } else {
                w = value;
            }
        }
        out.push_back(site_major_values[site_offset +
                                        tensor.index(indices[0], n, e, s, w)]);
    }
}

void test_product_state_amplitude_energy_and_gradient() {
    const peps_cuda::PEPS peps = make_all_zero_product_peps(2, 2);
    peps_cuda::Sample sample = peps_cuda::make_zero_sample(2, 2, 2);

    const peps_cuda::Complex psi =
        peps_cuda::contract_amplitude_exact(peps, sample);
    require(std::abs(psi - peps_cuda::Complex{1.0, 0.0}) < 1.0e-12,
            "all-zero product amplitude should be one");

    sample.at(1, 1) = 1;
    const peps_cuda::Complex psi_flipped =
        peps_cuda::contract_amplitude_exact(peps, sample);
    require(std::abs(psi_flipped) < 1.0e-12,
            "product amplitude with one wrong spin should be zero");
    sample.at(1, 1) = 0;

    const std::vector<peps_cuda::LocalOperatorTerm> ham =
        peps_cuda::make_nearest_neighbor_heisenberg(2, 2, 1.0);
    const peps_cuda::Complex e =
        peps_cuda::local_energy_exact(peps, sample, ham, psi);
    require(std::abs(e - peps_cuda::Complex{1.0, 0.0}) < 1.0e-12,
            "2x2 all-up Heisenberg energy should be four bonds times 1/4");

    const std::vector<peps_cuda::Complex> grad =
        peps_cuda::log_gradients_exact(peps, sample, psi);
    require(grad.size() == 8, "2x2 D=1 spin-1/2 product PEPS has 8 parameters");
    for (std::size_t i = 0; i < grad.size(); ++i) {
        const peps_cuda::Complex expected =
            (i % 2 == 0) ? peps_cuda::Complex{1.0, 0.0}
                         : peps_cuda::Complex{0.0, 0.0};
        require(std::abs(grad[i] - expected) < 1.0e-12,
                "product-state gradient should select sampled physical slices");
    }
}

void test_flip_classification() {
    require(peps_cuda::classify_flip_sites({}, 4) ==
                peps_cuda::FlipKind::Diagonal,
            "empty flip should be diagonal");
    require(peps_cuda::classify_flip_sites({5}, 4) ==
                peps_cuda::FlipKind::SingleSite,
            "single flip should be single-site");
    require(peps_cuda::classify_flip_sites({5, 6}, 4) ==
                peps_cuda::FlipKind::HorizontalNearest,
            "same-row adjacent flips should be horizontal nearest");
    require(peps_cuda::classify_flip_sites({5, 9}, 4) ==
                peps_cuda::FlipKind::VerticalNearest,
            "same-column adjacent flips should be vertical nearest");
    require(peps_cuda::classify_flip_sites({5, 6, 9, 10}, 4) ==
                peps_cuda::FlipKind::Plaquette2x2,
            "2x2 support should be plaquette bucket");
    require(peps_cuda::classify_flip_sites({4, 7}, 4) ==
                peps_cuda::FlipKind::HorizontalLong,
            "same-row non-adjacent support should be horizontal-long");
    require(peps_cuda::flip_kind_index(peps_cuda::FlipKind::Other) == 6,
            "flip-kind indices should be stable for bucket arrays");
}

void test_sampling_and_minsr_shapes() {
    const peps_cuda::PEPS peps = make_all_zero_product_peps(2, 2);
    const std::vector<peps_cuda::LocalOperatorTerm> ham =
        peps_cuda::make_nearest_neighbor_heisenberg(2, 2, 1.0);
    const peps_cuda::SampleBatch batch =
        peps_cuda::generate_Oks_and_Eks_exact(peps, ham, 3, 1234);

    require(batch.samples.size() == 3, "sampler should produce requested samples");
    for (const peps_cuda::Sample &sample : batch.samples) {
        for (int spin : sample.spin) {
            require(spin == 0, "exact product-state sampler should always draw zero");
        }
    }

    const std::vector<peps_cuda::Complex> direction =
        peps_cuda::minsr_direction(batch.O, batch.local_energy, 1.0e-6);
    require(direction.size() == 8, "minSR direction should have parameter count");
    require(std::isfinite(peps_cuda::squared_norm(direction)),
            "minSR direction norm should be finite");
    const std::vector<peps_cuda::Complex> weighted_direction =
        peps_cuda::minsr_direction_weighted(
            batch.O, batch.local_energy, std::vector<double>(3, 1.0), 1.0e-6);
    require(weighted_direction == direction,
            "unit weights should match unweighted minSR");

    const std::vector<int> spins =
        peps_cuda::flatten_samples_sample_major(batch.samples);
    require(spins.size() == 12, "3 samples on 2x2 should flatten to 12 spins");

    const std::vector<int> bonds = peps_cuda::make_nearest_neighbor_bond_pairs(2, 2);
    require(bonds.size() == 8, "2x2 lattice should have four nearest bonds");
    const auto summary = peps_cuda::summarize_flip_buckets(batch.samples.front(), ham);
    require(summary[0] == 4, "all-up 2x2 Heisenberg sample has four diagonal terms");
    const std::vector<peps_cuda::LocalOperatorTerm> tfi =
        peps_cuda::make_transverse_field_ising(2, 2, 1.0, 0.5);
    const auto tfi_summary =
        peps_cuda::summarize_flip_buckets(batch.samples.front(), tfi);
    require(tfi_summary[1] == 4,
            "transverse field should produce one single-site flip per site");
    const std::vector<peps_cuda::LocalOperatorTerm> rydberg =
        peps_cuda::make_square_rydberg_hamiltonian(2, 2, 1.0, 2.0, 3.0, -1.0);
    require(rydberg.size() == 10,
            "2x2 all-to-all Rydberg helper should emit four site and six pair terms");
    require(peps_cuda::dense_o_bytes(3, peps.parameter_count()) ==
                3 * peps.parameter_count() * sizeof(peps_cuda::Complex),
            "dense O byte helper should match sample*parameter complex storage");
    require(peps_cuda::sampled_sector_parameter_count(peps) == 4,
            "D=1 sampled-sector rows keep one parameter per site");
    require(peps_cuda::sampled_sector_o_bytes(
                3, peps_cuda::sampled_sector_parameter_count(peps)) ==
                3 * 4 * sizeof(peps_cuda::Complex),
            "sampled-sector byte helper should match compact storage");

    peps_cuda::Sample flipped_sample = batch.samples.front();
    flipped_sample.at(0, 0) = 1;
    const std::vector<std::vector<peps_cuda::Complex>> dense_rows = {
        {{1.0, 0.0},
         {0.0, 0.0},
         {2.0, 0.0},
         {0.0, 0.0},
         {3.0, 0.0},
         {0.0, 0.0},
         {4.0, 0.0},
         {0.0, 0.0}},
        {{0.0, 0.0},
         {10.0, 0.0},
         {20.0, 0.0},
         {0.0, 0.0},
         {30.0, 0.0},
         {0.0, 0.0},
         {40.0, 0.0},
         {0.0, 0.0}},
    };
    const std::vector<peps_cuda::Sample> gram_samples = {batch.samples.front(),
                                                         flipped_sample};
    const std::vector<std::vector<peps_cuda::Complex>> compact_rows =
        peps_cuda::compact_sampled_sector_log_gradients(peps, gram_samples,
                                                        dense_rows);
    require(compact_rows[0].size() == 4 && compact_rows[1].size() == 4,
            "compact sampled-sector rows should drop unsampled physical sectors");
    const std::vector<std::vector<peps_cuda::Complex>> compact_gram =
        peps_cuda::sampled_sector_gram(peps, gram_samples, compact_rows, 0.5);
    require(std::abs(compact_gram[0][0] - peps_cuda::Complex{30.5, 0.0}) <
                1.0e-12,
            "sampled-sector Gram diagonal should include shift");
    require(std::abs(compact_gram[0][1] - peps_cuda::Complex{290.0, 0.0}) <
                1.0e-12,
            "sampled-sector Gram should skip sites with different spins");
    const std::vector<peps_cuda::Complex> dense_direction =
        peps_cuda::minsr_direction(dense_rows,
                                   {{1.0, 0.0}, {2.0, 0.0}}, 0.5);
    const std::vector<peps_cuda::Complex> parameter_space_direction =
        peps_cuda::sr_direction_parameter_space(
            dense_rows, {{1.0, 0.0}, {2.0, 0.0}}, 0.5);
    for (std::size_t i = 0; i < dense_direction.size(); ++i) {
        require(std::abs(parameter_space_direction[i] - dense_direction[i]) <
                    tol(1.0e-10, 1.0e-4),
                "parameter-space SR should match dual minSR with the same ridge");
    }
    const std::vector<peps_cuda::Complex> compact_direction =
        peps_cuda::minsr_direction_sampled_sector(
            peps, gram_samples, compact_rows, {{1.0, 0.0}, {2.0, 0.0}}, 0.5);
    require(compact_direction.size() == dense_direction.size(),
            "sampled-sector minSR should scatter back to dense parameters");
    for (std::size_t i = 0; i < dense_direction.size(); ++i) {
        require(std::abs(compact_direction[i] - dense_direction[i]) <
                    tol(1.0e-10, 1.0e-4),
                "sampled-sector minSR should match dense minSR for sparse rows");
    }
    const std::vector<double> nontrivial_weights = {0.5, 2.0};
    const std::vector<peps_cuda::Complex> dense_weighted_direction =
        peps_cuda::minsr_direction_weighted(
            dense_rows, {{1.0, 0.0}, {2.0, 0.0}}, nontrivial_weights, 0.5);
    const std::vector<peps_cuda::Complex> compact_weighted_direction =
        peps_cuda::minsr_direction_sampled_sector_weighted(
            peps, gram_samples, compact_rows, {{1.0, 0.0}, {2.0, 0.0}},
            nontrivial_weights, 0.5);
    for (std::size_t i = 0; i < dense_weighted_direction.size(); ++i) {
        require(std::abs(compact_weighted_direction[i] -
                         dense_weighted_direction[i]) < tol(1.0e-10, 1.0e-4),
                "weighted sampled-sector minSR should match dense weighted minSR");
    }

    const std::vector<peps_cuda::Complex> log_psi = {
        cx(0.0), cx(std::log(2.0))};
    const std::vector<double> log_pc = {0.0, std::log(4.0)};
    const std::vector<double> weights =
        peps_cuda::compute_importance_weights(log_psi, log_pc);
    require(weights.size() == 2, "importance helper should preserve sample count");
    require(std::abs(weights[0] - 1.0) < tol(1.0e-12, 1.0e-6) &&
                std::abs(weights[1] - 1.0) < tol(1.0e-12, 1.0e-6),
            "importance weights should be normalized to mean one");

    const peps_cuda::PackedPEPS packed =
        peps_cuda::pack_site_tensors_physical_major(peps, 4);
    require(packed.sites.size() == 4, "packed PEPS should keep one entry per site");
    require(packed.projected_sample_stride == 16,
            "aligned D=1 product state projected stride should include padding");
    require(packed.data[packed.sites[0].offset] == peps_cuda::Complex{1.0, 0.0},
            "packed data should preserve physical-major tensor values");
    const peps_cuda::ParameterLayout layout = peps_cuda::make_parameter_layout(peps);
    require(layout.blocks.size() == 4, "parameter layout should keep one block per site");
    require(layout.total_parameters == peps.parameter_count(),
            "parameter layout should sum to PEPS parameter count");
    require(layout.blocks[1].offset == 2,
            "D=1 spin-1/2 product state has two parameters per site");

    const peps_cuda::MemorySnapshot snapshot =
        peps_cuda::get_process_memory_snapshot();
    require(snapshot.peak_rss_bytes > 0,
            "process memory helper should report peak RSS on supported hosts");
    require(!peps_cuda::format_bytes(snapshot.peak_rss_bytes).empty(),
            "memory byte formatter should produce text");
}

void test_random_complex_sampled_sector_invariants() {
    const peps_cuda::PEPS peps =
        peps_cuda::make_random_open_peps(2, 2, 2, 2, 2026, 0.2, 0.1);
    std::vector<peps_cuda::Sample> samples;
    samples.push_back(peps_cuda::make_zero_sample(2, 2, 2));
    samples.push_back(peps_cuda::make_zero_sample(2, 2, 2));
    samples.back().at(0, 0) = 1;
    samples.push_back(peps_cuda::make_zero_sample(2, 2, 2));
    samples.back().at(0, 1) = 1;

    std::vector<std::vector<peps_cuda::Complex>> dense_rows;
    for (const peps_cuda::Sample &sample : samples) {
        const peps_cuda::Complex psi =
            peps_cuda::contract_amplitude_exact(peps, sample);
        require(std::abs(psi) > 1.0e-14,
                "random invariant fixture should avoid zero amplitudes");
        dense_rows.push_back(peps_cuda::log_gradients_exact(peps, sample, psi));
    }

    const std::vector<std::vector<peps_cuda::Complex>> compact_rows =
        peps_cuda::compact_sampled_sector_log_gradients(peps, samples,
                                                        dense_rows);
    const std::vector<std::vector<peps_cuda::Complex>> gram =
        peps_cuda::sampled_sector_gram(peps, samples, compact_rows, 1.0e-4);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        for (std::size_t j = 0; j < samples.size(); ++j) {
            peps_cuda::Complex expected{0.0, 0.0};
            for (std::size_t p = 0; p < dense_rows[i].size(); ++p) {
                expected += dense_rows[i][p] * std::conj(dense_rows[j][p]);
            }
            if (i == j) {
                expected += peps_cuda::Complex{1.0e-4, 0.0};
            }
            require(std::abs(gram[i][j] - expected) < tol(1.0e-10, 1.0e-4),
                    "sampled-sector Gram should match dense sparse-row Gram");
            require(std::abs(gram[i][j] - std::conj(gram[j][i])) <
                        tol(1.0e-10, 1.0e-4),
                    "sampled-sector Gram should be Hermitian");
        }
        require(real(gram[i][i]) > 0.0 &&
                    std::abs(imag(gram[i][i])) < tol(1.0e-10, 1.0e-4),
                "sampled-sector Gram diagonal should be positive real");
    }

    const std::vector<peps_cuda::Complex> energies = {
        {0.25, -0.1}, {-0.2, 0.05}, {0.4, 0.3}};
    const std::vector<peps_cuda::Complex> dense_direction =
        peps_cuda::minsr_direction(dense_rows, energies, 1.0e-4);
    const std::vector<peps_cuda::Complex> compact_direction =
        peps_cuda::minsr_direction_sampled_sector(peps, samples, compact_rows,
                                                  energies, 1.0e-4);
    require(dense_direction.size() == compact_direction.size(),
            "compact and dense minSR directions should have the same length");
    for (std::size_t i = 0; i < dense_direction.size(); ++i) {
        require(std::abs(dense_direction[i] - compact_direction[i]) <
                    tol(1.0e-10, 1.0e-4),
                "compact sampled-sector minSR should equal dense minSR");
    }
}

void test_julia_d1_reference_fixture_alignment() {
    // From julia_reference/fixtures/reference_fixtures.jsonl,
    // case real_3x2_D1_zero_sample. For D=1 the ITensor link ordering is
    // immaterial, so the C++ open-PEPS oracle should match the Julia fixture.
    const std::vector<peps_cuda::Complex> theta = {
        cx(-0.13164979128118226),
        cx(0.99129628893465582),
        cx(-0.31199803724281328),
        cx(0.95008274626825628),
        cx(0.97462032337897764),
        cx(0.2238643009875792),
        cx(0.90431583621426492),
        cx(-0.42686399282686643),
        cx(-0.95362372610222823),
        cx(-0.30100131065313007),
        cx(-0.37769818353620965),
        cx(0.9259287673214649),
    };
    peps_cuda::PEPS peps;
    peps.lx = 3;
    peps.ly = 2;
    peps.local_dim = 2;
    peps.max_bond_dim = 1;
    peps.sites.resize(6);
    for (int site = 0; site < 6; ++site) {
        peps_cuda::SiteTensor tensor;
        tensor.phys = 2;
        tensor.north = 1;
        tensor.east = 1;
        tensor.south = 1;
        tensor.west = 1;
        tensor.data = {theta[2 * site], theta[2 * site + 1]};
        peps.sites[static_cast<std::size_t>(site)] = std::move(tensor);
    }

    const peps_cuda::Sample sample = peps_cuda::make_zero_sample(3, 2, 2);
    const peps_cuda::Complex psi =
        peps_cuda::contract_amplitude_exact(peps, sample);
    const peps_cuda::Complex log_psi = std::log(psi);
    require(std::abs(log_psi - cx(-4.3397982205093646, 0.0)) <
                tol(1.0e-12, 1.0e-5),
            "C++ logpsi should match Julia D=1 fixture");

    const std::vector<peps_cuda::Complex> ok =
        peps_cuda::log_gradients_exact(peps, sample, psi);
    require(ok.size() == 12, "Julia D=1 fixture should have 12 parameters");
    const std::vector<peps_cuda::Complex> ok_first8 = {
        cx(-7.5959102575724158),
        cx(0.0),
        cx(-3.2051483683589566),
        cx(0.0),
        cx(1.0260405780715018),
        cx(0.0),
        cx(1.1058083469889211),
        cx(0.0),
    };
    for (std::size_t i = 0; i < ok_first8.size(); ++i) {
        require(std::abs(ok[i] - ok_first8[i]) < tol(1.0e-12, 1.0e-5),
                "C++ Ok prefix should match Julia D=1 fixture");
    }
    require(std::abs(peps_cuda::squared_norm(ok) - 78.355902296461238) <
                tol(1.0e-10, 1.0e-4),
            "C++ Ok norm should match Julia D=1 fixture");

    // The Julia fixture uses ITensor "X,Y,Z" Pauli operators. The C++ helper is
    // written in spin-operator normalization, so J=4 gives the same diagonal
    // all-up bond energy.
    const std::vector<peps_cuda::LocalOperatorTerm> ham =
        peps_cuda::make_nearest_neighbor_heisenberg(3, 2, 4.0);
    const peps_cuda::Complex energy =
        peps_cuda::local_energy_exact(peps, sample, ham, psi);
    require(std::abs(energy - cx(7.0, 0.0)) < tol(1.0e-12, 1.0e-5),
            "C++ Heisenberg energy should match Julia Pauli-normalized fixture");
}

void test_julia_d2_reference_fixture_alignment() {
    // From julia_reference/fixtures/reference_fixtures.jsonl,
    // case real_3x2_D2_zero_sample. This checks both Julia column-major tensor
    // flattening and the h_link/v_link to east/west/north/south mapping.
    const std::vector<peps_cuda::Complex> theta = {
        cx(-0.72392323926331592, 0),
        cx(-0.28562022804384146, 0),
        cx(-0.61501134092321552, 0),
        cx(0.39428884267174724, 0),
        cx(0.30670123337358102, 0),
        cx(-0.054782117709993659, 0),
        cx(-0.060254025902506836, 0),
        cx(-0.8717547323966568, 0),
        cx(-0.47931039407152709, 0),
        cx(-0.61196069023480149, 0),
        cx(0.20881973684148253, 0),
        cx(-0.5934306844508439, 0),
        cx(0.8541782020508214, 0),
        cx(-0.16898074928093143, 0),
        cx(0.093307633703942544, 0),
        cx(-0.48282376806268745, 0),
        cx(0.044731050106331827, 0),
        cx(-0.36212767612034624, 0),
        cx(0.3377086550960926, 0),
        cx(-0.86764943589980548, 0),
        cx(0.38646612151063414, 0),
        cx(-0.6595485855013572, 0),
        cx(0.44357520760708397, 0),
        cx(0.46784680770955284, 0),
        cx(-0.92048440597900405, 0),
        cx(-0.31755742060482256, 0),
        cx(0.17019428265266368, 0),
        cx(0.15132630016172685, 0),
        cx(0.036763139561149585, 0),
        cx(-0.57707581319064893, 0),
        cx(-0.81254403303566614, 0),
        cx(-0.073513072157902037, 0),
        cx(0.15067630559773781, 0),
        cx(0.70115011100290414, 0),
        cx(0.29144990405274479, 0),
        cx(-0.33781218367475069, 0),
        cx(-0.33121177594890644, 0),
        cx(-0.19829449543375324, 0),
        cx(0.3615349633576348, 0),
        cx(0.083039614546389404, 0),
        cx(-0.8885570461741672, 0),
        cx(-0.039609920496871509, 0),
        cx(0.12676067493488566, 0),
        cx(-0.41050356623599715, 0),
        cx(0.060480455688435367, 0),
        cx(-0.12851072869286168, 0),
        cx(-0.042611620511394638, 0),
        cx(-0.048242514217158083, 0),
        cx(0.18818344658382391, 0),
        cx(0.35612042258657145, 0),
        cx(0.6746471078685834, 0),
        cx(0.618560033378451, 0),
        cx(-0.94507542935810984, 0),
        cx(-0.037915697487823863, 0),
        cx(0.3219448756648311, 0),
        cx(-0.04178911031299791, 0),
        cx(0.19478839928442349, 0),
        cx(0.40389075616603526, 0),
        cx(-0.2506473949100157, 0),
        cx(-0.49579054094100183, 0),
        cx(0.4594603508532149, 0),
        cx(0.46479288581812606, 0),
        cx(-0.14269056692437423, 0),
        cx(-0.20688286764155187, 0),
    };

    const std::vector<std::vector<char>> site_dirs = {
        {'e', 's'},
        {'w', 's'},
        {'n', 'e', 's'},
        {'w', 'n', 's'},
        {'n', 'e'},
        {'w', 'n'},
    };

    std::size_t cursor = 0;
    peps_cuda::PEPS peps;
    peps.lx = 3;
    peps.ly = 2;
    peps.local_dim = 2;
    peps.max_bond_dim = 2;
    for (const std::vector<char> &dirs : site_dirs) {
        peps.sites.push_back(make_site_from_julia_theta(theta, cursor, dirs));
    }
    require(cursor == theta.size(), "Julia D=2 fixture import should consume theta");

    const peps_cuda::Sample sample = peps_cuda::make_zero_sample(3, 2, 2);
    const peps_cuda::Complex psi =
        peps_cuda::contract_amplitude_exact(peps, sample);
    const peps_cuda::Complex log_psi = std::log(psi);
    require(std::abs(log_psi - cx(-3.0564057669214719, 0.0)) <
                tol(1.0e-12, 1.0e-5),
            "C++ logpsi should match Julia D=2 fixture");

    const std::vector<peps_cuda::Complex> ok =
        peps_cuda::log_gradients_exact(peps, sample, psi);
    require(ok.size() == 64, "Julia D=2 fixture should have 64 parameters");
    std::vector<peps_cuda::Complex> ok_julia_order;
    std::size_t site_offset = 0;
    for (std::size_t site = 0; site < peps.sites.size(); ++site) {
        append_site_values_in_julia_order(peps.sites[site], ok, site_offset,
                                          site_dirs[site], ok_julia_order);
        site_offset += peps.sites[site].parameter_count();
    }
    require(ok_julia_order.size() == ok.size(),
            "Julia-order Ok transpose should preserve parameter count");
    const std::vector<peps_cuda::Complex> ok_first8 = {
        cx(-1.9160467277314042, 0.0),
        cx(0.0),
        cx(1.1527064468246799, 0.0),
        cx(0.0),
        cx(0.98047575720953295, 0.0),
        cx(0.0),
        cx(-0.35090866523073011, 0.0),
        cx(0.0),
    };
    for (std::size_t i = 0; i < ok_first8.size(); ++i) {
        require(std::abs(ok_julia_order[i] - ok_first8[i]) <
                    tol(1.0e-12, 1.0e-5),
                "C++ Ok prefix should match Julia D=2 fixture");
    }
    require(std::abs(peps_cuda::squared_norm(ok) - 104.0693327703223) <
                tol(1.0e-10, 1.0e-4),
            "C++ Ok norm should match Julia D=2 fixture");

    const std::vector<peps_cuda::LocalOperatorTerm> ham =
        peps_cuda::make_nearest_neighbor_heisenberg(3, 2, 4.0);
    const peps_cuda::Complex energy =
        peps_cuda::local_energy_exact(peps, sample, ham, psi);
    require(std::abs(energy - cx(7.0, 0.0)) < tol(1.0e-12, 1.0e-5),
            "C++ Heisenberg energy should match Julia D=2 fixture");

    peps_cuda::Sample checker = peps_cuda::make_zero_sample(3, 2, 2);
    checker.spin = {0, 1, 1, 0, 0, 1};
    const peps_cuda::Complex checker_psi =
        peps_cuda::contract_amplitude_exact(peps, checker);
    const peps_cuda::Complex checker_log_psi = std::log(checker_psi);
    require(std::abs(checker_log_psi - cx(-2.3163526110730195,
                                          3.141592653589793)) <
                tol(1.0e-12, 1.0e-5),
            "C++ logpsi should match Julia real D=2 checker fixture");
    const std::vector<peps_cuda::Complex> checker_ok =
        peps_cuda::log_gradients_exact(peps, checker, checker_psi);
    std::vector<peps_cuda::Complex> checker_ok_julia_order;
    site_offset = 0;
    for (std::size_t site = 0; site < peps.sites.size(); ++site) {
        append_site_values_in_julia_order(peps.sites[site], checker_ok,
                                          site_offset, site_dirs[site],
                                          checker_ok_julia_order);
        site_offset += peps.sites[site].parameter_count();
    }
    const std::vector<peps_cuda::Complex> checker_ok_first8 = {
        cx(-0.37196168454679485, 0.0),
        cx(0.0),
        cx(-0.59717898003166747, 0.0),
        cx(0.0),
        cx(1.6385624650941026, 0.0),
        cx(0.0),
        cx(2.3084379800357775, 0.0),
        cx(0.0),
    };
    for (std::size_t i = 0; i < checker_ok_first8.size(); ++i) {
        require(std::abs(checker_ok_julia_order[i] - checker_ok_first8[i]) <
                    tol(1.0e-12, 1.0e-5),
                "C++ Ok prefix should match Julia real D=2 checker fixture");
    }
    require(std::abs(peps_cuda::squared_norm(checker_ok) -
                     69.09352416768316) < tol(1.0e-10, 1.0e-4),
            "C++ Ok norm should match Julia real D=2 checker fixture");
    const peps_cuda::Complex checker_energy =
        peps_cuda::local_energy_exact(peps, checker, ham, checker_psi);
    require(std::abs(checker_energy - cx(-12.39419879515855, 0.0)) <
                tol(1.0e-12, 1.0e-5),
            "C++ energy should match Julia real D=2 checker fixture");
}

void test_julia_complex_d2_reference_fixture_alignment() {
    // From julia_reference/fixtures/reference_fixtures.jsonl,
    // case complex_3x2_D2_zero_sample. This catches phase conventions and the
    // holomorphic log-gradient ordering used by the Julia reference.
    const std::vector<peps_cuda::Complex> theta = {
        cx(-0.21274452686707623, -0.43692537524664665),
        cx(-0.4534110079578324, 0.53790225584300566),
        cx(0.22993907418577753, -0.21754564760488002),
        cx(-0.26846608695896279, 0.13136193383723507),
        cx(0.49117223310233293, 0.52409272923110939),
        cx(-0.28036021934115868, 0.37470304123792925),
        cx(0.29108536490330855, 0.25096584440896358),
        cx(0.31880187711424113, 0.30840033229491309),
        cx(-0.36058890913201136, -0.13896564955392091),
        cx(-0.011272909398229105, 0.23728357207852224),
        cx(-0.39497490125235318, 0.20851395920681753),
        cx(0.65883336258109004, 0.40086028904979454),
        cx(-0.35287644424513331, 0.35777772627704352),
        cx(0.16390972803243797, 0.64932374078315602),
        cx(0.3672247107697118, -0.34347370524355147),
        cx(-0.12582077728434163, 0.17414520377711445),
        cx(0.36736956740851556, 0.10374896950530663),
        cx(0.8198853194677016, 0.14219986889873792),
        cx(0.035323666033129857, -0.14439247762129712),
        cx(-0.24831026833309924, 0.27944243439374256),
        cx(-0.029367209467027644, 0.6101056495627345),
        cx(0.074250602303868263, -0.29639704638878761),
        cx(-0.64716446513280634, 0.043384362330698811),
        cx(-0.16173123698140315, -0.294420212431589),
        cx(0.47595595394598461, -0.17368882678365252),
        cx(-0.069151693998431019, -0.16035714004432716),
        cx(0.20703247053547291, -0.41748685922066131),
        cx(-0.18176927756894076, -0.68014995439384485),
        cx(0.45218907635726141, 0.14144813970312192),
        cx(-0.3569957291972502, 0.23719312668143933),
        cx(-0.32241773493736026, -0.4858889645987024),
        cx(0.33859480915322582, 0.37030150753305491),
        cx(0.45845101574879332, 0.020491242158566948),
        cx(0.12236324158256115, 0.06405669193704315),
        cx(-0.25964985535630714, 0.1899803120294512),
        cx(0.4963314131548775, 0.13015053727856513),
        cx(-0.37790803111550791, 0.28675219153227577),
        cx(-0.042446690744517832, 0.15102083713766873),
        cx(0.22075574290905761, -0.14328969746836992),
        cx(-0.20487318674686766, 0.2065037042931778),
        cx(-0.37275467323282524, 0.13106487059173116),
        cx(0.0081059907526790376, -0.064928122025022766),
        cx(-0.29946539193187155, -0.023488484894310359),
        cx(0.0081861013631102254, -0.3016594858681878),
        cx(0.1727910829046452, 0.38812537891769383),
        cx(-0.21718771683416541, 0.33395743715969806),
        cx(0.24230728844144514, 0.41416510273959983),
        cx(-0.27626520515854069, -0.11195417543371652),
        cx(0.36565143818554513, 0.76989328924618894),
        cx(0.017437948229108876, -0.41687426669767219),
        cx(-0.040467157642626927, -0.14618654247746563),
        cx(-0.22268696436072752, 0.16394368700251472),
        cx(0.14220586269937674, 0.14425665981631369),
        cx(-0.034040588877831013, -0.056508770306232088),
        cx(0.76668485496026828, 0.53366249352976791),
        cx(0.2640083177046047, -0.1109675189270318),
        cx(0.056968741533532929, -0.25182611023553869),
        cx(0.21705107056359696, 0.36565985889644642),
        cx(-0.12293000669453928, 0.18877631752309035),
        cx(-0.17913677367710279, -0.020986347297623976),
        cx(-0.29736413505415615, -0.24738402519346914),
        cx(0.30694263210213113, -0.20901170546576553),
        cx(0.35792350222711083, 0.22783247389690389),
        cx(-0.25246570220126413, -0.37143158243950153),
    };

    const std::vector<std::vector<char>> site_dirs = {
        {'e', 's'},
        {'w', 's'},
        {'n', 'e', 's'},
        {'w', 'n', 's'},
        {'n', 'e'},
        {'w', 'n'},
    };

    std::size_t cursor = 0;
    peps_cuda::PEPS peps;
    peps.lx = 3;
    peps.ly = 2;
    peps.local_dim = 2;
    peps.max_bond_dim = 2;
    for (const std::vector<char> &dirs : site_dirs) {
        peps.sites.push_back(make_site_from_julia_theta(theta, cursor, dirs));
    }
    require(cursor == theta.size(),
            "Julia complex D=2 fixture import should consume theta");

    const peps_cuda::Sample sample = peps_cuda::make_zero_sample(3, 2, 2);
    const peps_cuda::Complex psi =
        peps_cuda::contract_amplitude_exact(peps, sample);
    const peps_cuda::Complex log_psi = std::log(psi);
    require(std::abs(log_psi - cx(-2.5805203294143384, 0.45231141154023086)) <
                tol(1.0e-12, 1.0e-5),
            "C++ logpsi should match Julia complex D=2 fixture");

    const std::vector<peps_cuda::Complex> ok =
        peps_cuda::log_gradients_exact(peps, sample, psi);
    require(ok.size() == 64,
            "Julia complex D=2 fixture should have 64 parameters");
    std::vector<peps_cuda::Complex> ok_julia_order;
    std::size_t site_offset = 0;
    for (std::size_t site = 0; site < peps.sites.size(); ++site) {
        append_site_values_in_julia_order(peps.sites[site], ok, site_offset,
                                          site_dirs[site], ok_julia_order);
        site_offset += peps.sites[site].parameter_count();
    }
    const std::vector<peps_cuda::Complex> ok_first8 = {
        cx(-0.41687361315866595, -0.62252330908667775),
        cx(0.0),
        cx(2.0623599230548133, 0.9044352641885397),
        cx(0.0),
        cx(-0.06405120369295568, -0.023064497781227086),
        cx(0.0),
        cx(0.99852417469855315, -0.96049734651760743),
        cx(0.0),
    };
    for (std::size_t i = 0; i < ok_first8.size(); ++i) {
        require(std::abs(ok_julia_order[i] - ok_first8[i]) <
                    tol(1.0e-12, 1.0e-5),
                "C++ Ok prefix should match Julia complex D=2 fixture");
    }
    require(std::abs(peps_cuda::squared_norm(ok) - 34.213225294470924) <
                tol(1.0e-10, 1.0e-4),
            "C++ Ok norm should match Julia complex D=2 fixture");

    const std::vector<peps_cuda::LocalOperatorTerm> ham =
        peps_cuda::make_nearest_neighbor_heisenberg(3, 2, 4.0);
    const peps_cuda::Complex energy =
        peps_cuda::local_energy_exact(peps, sample, ham, psi);
    require(std::abs(energy - cx(7.0, 0.0)) < tol(1.0e-12, 1.0e-5),
            "C++ Heisenberg energy should match Julia complex D=2 fixture");

    peps_cuda::Sample checker = peps_cuda::make_zero_sample(3, 2, 2);
    checker.spin = {0, 1, 1, 0, 0, 1};
    const peps_cuda::Complex checker_psi =
        peps_cuda::contract_amplitude_exact(peps, checker);
    const peps_cuda::Complex checker_log_psi = std::log(checker_psi);
    require(std::abs(checker_log_psi -
                     cx(-1.962853745666855, 1.7408781402625095)) <
                tol(1.0e-12, 1.0e-5),
            "C++ logpsi should match Julia complex D=2 checker fixture");
    const std::vector<peps_cuda::Complex> checker_ok =
        peps_cuda::log_gradients_exact(peps, checker, checker_psi);
    std::vector<peps_cuda::Complex> checker_ok_julia_order;
    site_offset = 0;
    for (std::size_t site = 0; site < peps.sites.size(); ++site) {
        append_site_values_in_julia_order(peps.sites[site], checker_ok,
                                          site_offset, site_dirs[site],
                                          checker_ok_julia_order);
        site_offset += peps.sites[site].parameter_count();
    }
    const std::vector<peps_cuda::Complex> checker_ok_first8 = {
        cx(-0.17454559877172651, 0.11967000065003602),
        cx(0.0),
        cx(-0.38200593647489689, 1.6593229120548927),
        cx(0.0),
        cx(0.018190487677563566, -0.99982703419434427),
        cx(0.0),
        cx(0.14834036321606239, -0.2443467274788674),
        cx(0.0),
    };
    for (std::size_t i = 0; i < checker_ok_first8.size(); ++i) {
        require(std::abs(checker_ok_julia_order[i] - checker_ok_first8[i]) <
                    tol(1.0e-12, 1.0e-5),
                "C++ Ok prefix should match Julia complex D=2 checker fixture");
    }
    require(std::abs(peps_cuda::squared_norm(checker_ok) -
                     26.45126958611757) < tol(1.0e-10, 1.0e-4),
            "C++ Ok norm should match Julia complex D=2 checker fixture");
    const peps_cuda::Complex checker_energy =
        peps_cuda::local_energy_exact(peps, checker, ham, checker_psi);
    require(std::abs(checker_energy -
                     cx(-3.3694756809376236, 4.91737242081064)) <
                tol(1.0e-12, 1.0e-5),
            "C++ energy should match Julia complex D=2 checker fixture");
}

} // namespace

int main() {
    try {
        test_product_state_amplitude_energy_and_gradient();
        test_flip_classification();
        test_sampling_and_minsr_shapes();
        test_random_complex_sampled_sector_invariants();
        test_julia_d1_reference_fixture_alignment();
        test_julia_d2_reference_fixture_alignment();
        test_julia_complex_d2_reference_fixture_alignment();
    } catch (const std::exception &err) {
        std::cerr << "test failure: " << err.what() << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "peps_cuda_unit_tests passed\n";
    return EXIT_SUCCESS;
}
