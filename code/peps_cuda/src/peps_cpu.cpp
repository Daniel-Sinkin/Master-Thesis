#include "peps_cuda/peps.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace peps_cuda {
namespace {

std::size_t checked_pow(std::size_t base, int exp) {
    std::size_t out = 1;
    for (int i = 0; i < exp; ++i) {
        if (base != 0 && out > std::numeric_limits<std::size_t>::max() / base) {
            throw std::overflow_error("integer power overflow");
        }
        out *= base;
    }
    return out;
}

std::uint64_t encode_append(std::uint64_t code, int value, int dim) {
    return code * static_cast<std::uint64_t>(dim) +
           static_cast<std::uint64_t>(value);
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment <= 1) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

Real as_real(double value) { return static_cast<Real>(value); }

Complex cx(double real, double imag = 0.0) {
    return {as_real(real), as_real(imag)};
}

std::vector<int> decode_tuple(std::uint64_t code, const std::vector<int> &dims) {
    std::vector<int> out(dims.size(), 0);
    for (std::size_t i = dims.size(); i-- > 0;) {
        out[i] = static_cast<int>(code % static_cast<std::uint64_t>(dims[i]));
        code /= static_cast<std::uint64_t>(dims[i]);
    }
    return out;
}

Sample decode_sample(std::uint64_t code, int lx, int ly, int local_dim) {
    Sample sample = make_zero_sample(lx, ly, local_dim);
    for (std::size_t i = sample.spin.size(); i-- > 0;) {
        sample.spin[i] = static_cast<int>(code % local_dim);
        code /= static_cast<std::uint64_t>(local_dim);
    }
    return sample;
}

int local_state_code(const Sample &sample, const std::vector<int> &sites) {
    int code = 0;
    for (int site : sites) {
        code = code * sample.local_dim + sample.spin[static_cast<std::size_t>(site)];
    }
    return code;
}

std::vector<int> decode_local_values(int out_code, int count, int local_dim) {
    std::vector<int> values(static_cast<std::size_t>(count), 0);
    for (std::size_t i = values.size(); i-- > 0;) {
        values[i] = out_code % local_dim;
        out_code /= local_dim;
    }
    return values;
}

Sample apply_local_values(const Sample &sample, const std::vector<int> &sites,
                          const std::vector<int> &values) {
    Sample flipped = sample;
    for (std::size_t i = 0; i < sites.size(); ++i) {
        flipped.spin[static_cast<std::size_t>(sites[i])] = values[i];
    }
    return flipped;
}

std::vector<Complex> solve_dense(std::vector<std::vector<Complex>> a,
                                 std::vector<Complex> b) {
    const int n = static_cast<int>(a.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs(a[static_cast<std::size_t>(col)]
                                [static_cast<std::size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            const double candidate =
                std::abs(a[static_cast<std::size_t>(row)]
                          [static_cast<std::size_t>(col)]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best == 0.0) {
            throw std::runtime_error("singular dense system in minSR solve");
        }
        if (pivot != col) {
            std::swap(a[static_cast<std::size_t>(pivot)],
                      a[static_cast<std::size_t>(col)]);
            std::swap(b[static_cast<std::size_t>(pivot)],
                      b[static_cast<std::size_t>(col)]);
        }
        const Complex diag = a[static_cast<std::size_t>(col)]
                              [static_cast<std::size_t>(col)];
        for (int j = col; j < n; ++j) {
            a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= diag;
        }
        b[static_cast<std::size_t>(col)] /= diag;

        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const Complex factor =
                a[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (factor == Complex{0.0, 0.0}) {
                continue;
            }
            for (int j = col; j < n; ++j) {
                a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * a[static_cast<std::size_t>(col)]
                              [static_cast<std::size_t>(j)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
        }
    }
    return b;
}

struct ExactDistribution {
    std::vector<double> cumulative_weights;
    double norm = 0.0;
    int lx = 0;
    int ly = 0;
    int local_dim = 0;
};

} // namespace

std::size_t SiteTensor::parameter_count() const {
    return static_cast<std::size_t>(phys) * static_cast<std::size_t>(north) *
           static_cast<std::size_t>(east) * static_cast<std::size_t>(south) *
           static_cast<std::size_t>(west);
}

std::size_t SiteTensor::index(int p, int n, int e, int s, int w) const {
    return (((static_cast<std::size_t>(p) * north + static_cast<std::size_t>(n)) *
                 east +
             static_cast<std::size_t>(e)) *
                south +
            static_cast<std::size_t>(s)) *
               west +
           static_cast<std::size_t>(w);
}

Complex &SiteTensor::operator()(int p, int n, int e, int s, int w) {
    return data[index(p, n, e, s, w)];
}

const Complex &SiteTensor::operator()(int p, int n, int e, int s, int w) const {
    return data[index(p, n, e, s, w)];
}

SiteTensor &PEPS::at(int x, int y) { return sites[site_index(x, y)]; }

const SiteTensor &PEPS::at(int x, int y) const {
    return sites[site_index(x, y)];
}

std::size_t PEPS::site_index(int x, int y) const {
    return static_cast<std::size_t>(x) * static_cast<std::size_t>(ly) +
           static_cast<std::size_t>(y);
}

std::size_t PEPS::parameter_count() const {
    std::size_t total = 0;
    for (const SiteTensor &site : sites) {
        total += site.parameter_count();
    }
    return total;
}

int &Sample::at(int x, int y) { return spin[site_index(x, y)]; }

const int &Sample::at(int x, int y) const { return spin[site_index(x, y)]; }

std::size_t Sample::site_index(int x, int y) const {
    return static_cast<std::size_t>(x) * static_cast<std::size_t>(ly) +
           static_cast<std::size_t>(y);
}

PEPS make_random_open_peps(int lx, int ly, int local_dim, int bond_dim,
                           std::uint64_t seed, double mean_shift,
                           double scale) {
    if (lx <= 0 || ly <= 0 || local_dim <= 0 || bond_dim <= 0) {
        throw std::invalid_argument("invalid PEPS dimensions");
    }
    PEPS peps;
    peps.lx = lx;
    peps.ly = ly;
    peps.local_dim = local_dim;
    peps.max_bond_dim = bond_dim;
    peps.sites.resize(static_cast<std::size_t>(lx * ly));

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, scale);
    for (int x = 0; x < lx; ++x) {
        for (int y = 0; y < ly; ++y) {
            SiteTensor site;
            site.phys = local_dim;
            site.north = (x == 0) ? 1 : bond_dim;
            site.south = (x == lx - 1) ? 1 : bond_dim;
            site.west = (y == 0) ? 1 : bond_dim;
            site.east = (y == ly - 1) ? 1 : bond_dim;
            site.data.resize(site.parameter_count());
            for (Complex &value : site.data) {
                value = cx(mean_shift + normal(rng), normal(rng));
            }
            peps.at(x, y) = std::move(site);
        }
    }
    return peps;
}

Sample make_zero_sample(int lx, int ly, int local_dim) {
    Sample sample;
    sample.lx = lx;
    sample.ly = ly;
    sample.local_dim = local_dim;
    sample.spin.assign(static_cast<std::size_t>(lx * ly), 0);
    return sample;
}

Complex contract_amplitude_exact(const PEPS &peps, const Sample &sample) {
    if (peps.lx != sample.lx || peps.ly != sample.ly ||
        peps.local_dim != sample.local_dim) {
        throw std::invalid_argument("PEPS and sample dimensions do not match");
    }

    std::unordered_map<std::uint64_t, Complex> boundary;
    boundary.emplace(0, Complex{1.0, 0.0});
    const std::uint64_t key_stride =
        static_cast<std::uint64_t>(std::max(peps.max_bond_dim + 1, 2));

    for (int x = 0; x < peps.lx; ++x) {
        std::vector<int> north_dims(static_cast<std::size_t>(peps.ly), 1);
        std::vector<int> south_dims(static_cast<std::size_t>(peps.ly), 1);
        for (int y = 0; y < peps.ly; ++y) {
            north_dims[static_cast<std::size_t>(y)] = peps.at(x, y).north;
            south_dims[static_cast<std::size_t>(y)] = peps.at(x, y).south;
        }

        std::unordered_map<std::uint64_t, Complex> next_boundary;
        for (const auto &[north_code, boundary_value] : boundary) {
            const std::vector<int> north = decode_tuple(north_code, north_dims);
            std::unordered_map<std::uint64_t, Complex> partial;
            partial.emplace(0, boundary_value);

            for (int y = 0; y < peps.ly; ++y) {
                const SiteTensor &site = peps.at(x, y);
                std::unordered_map<std::uint64_t, Complex> updated;
                for (const auto &[packed_key, value] : partial) {
                    const int west = static_cast<int>(packed_key % key_stride);
                    const std::uint64_t south_code = packed_key / key_stride;
                    if (west >= site.west) {
                        continue;
                    }
                    const int p = sample.at(x, y);
                    const int n = north[static_cast<std::size_t>(y)];
                    for (int e = 0; e < site.east; ++e) {
                        for (int s = 0; s < site.south; ++s) {
                            const Complex coeff = site(p, n, e, s, west);
                            if (coeff == Complex{0.0, 0.0}) {
                                continue;
                            }
                            const std::uint64_t next_south =
                                encode_append(south_code, s,
                                              south_dims[static_cast<std::size_t>(y)]);
                            updated[next_south * key_stride + static_cast<std::uint64_t>(e)] +=
                                value * coeff;
                        }
                    }
                }
                partial.swap(updated);
            }

            for (const auto &[packed_key, value] : partial) {
                const int open_east = static_cast<int>(packed_key % key_stride);
                if (open_east == 0) {
                    next_boundary[packed_key / key_stride] += value;
                }
            }
        }
        boundary.swap(next_boundary);
    }

    const auto it = boundary.find(0);
    return it == boundary.end() ? Complex{0.0, 0.0} : it->second;
}

namespace {

ExactDistribution build_exact_distribution(const PEPS &peps,
                                           std::size_t max_enumerated_states) {
    const int nsites = peps.lx * peps.ly;
    const std::size_t states =
        checked_pow(static_cast<std::size_t>(peps.local_dim), nsites);
    if (states > max_enumerated_states) {
        throw std::runtime_error(
            "exact sampler would enumerate too many states; use direct sampling");
    }

    ExactDistribution distribution;
    distribution.cumulative_weights.resize(states, 0.0);
    distribution.lx = peps.lx;
    distribution.ly = peps.ly;
    distribution.local_dim = peps.local_dim;

    double cumulative = 0.0;
    for (std::size_t code = 0; code < states; ++code) {
        const Sample sample =
            decode_sample(static_cast<std::uint64_t>(code), peps.lx, peps.ly,
                          peps.local_dim);
        cumulative += std::norm(contract_amplitude_exact(peps, sample));
        distribution.cumulative_weights[code] = cumulative;
    }
    distribution.norm = cumulative;
    if (distribution.norm == 0.0) {
        throw std::runtime_error("cannot sample from zero-norm PEPS");
    }
    return distribution;
}

Sample draw_exact_sample(const ExactDistribution &distribution,
                         std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> uniform(0.0, distribution.norm);
    const double r = uniform(rng);
    const auto it = std::lower_bound(distribution.cumulative_weights.begin(),
                                     distribution.cumulative_weights.end(), r);
    const std::size_t code =
        static_cast<std::size_t>(std::distance(
            distribution.cumulative_weights.begin(), it));
    const std::size_t clamped =
        std::min(code, distribution.cumulative_weights.size() - 1);
    return decode_sample(static_cast<std::uint64_t>(clamped), distribution.lx,
                         distribution.ly, distribution.local_dim);
}

} // namespace

FlipKind classify_flip_sites(const std::vector<int> &sites, int ly) {
    if (sites.empty()) {
        return FlipKind::Diagonal;
    }
    if (sites.size() == 1) {
        return FlipKind::SingleSite;
    }

    std::vector<int> xs;
    std::vector<int> ys;
    xs.reserve(sites.size());
    ys.reserve(sites.size());
    for (int site : sites) {
        xs.push_back(site / ly);
        ys.push_back(site % ly);
    }
    const auto [xmin_it, xmax_it] = std::minmax_element(xs.begin(), xs.end());
    const auto [ymin_it, ymax_it] = std::minmax_element(ys.begin(), ys.end());
    const int dx = *xmax_it - *xmin_it;
    const int dy = *ymax_it - *ymin_it;

    if (dx == 0 && dy == 1 && sites.size() <= 2) {
        return FlipKind::HorizontalNearest;
    }
    if (dx == 1 && dy == 0 && sites.size() <= 2) {
        return FlipKind::VerticalNearest;
    }
    if (dx <= 1 && dy <= 1 && sites.size() <= 4) {
        return FlipKind::Plaquette2x2;
    }
    if (dx == 0) {
        return FlipKind::HorizontalLong;
    }
    return FlipKind::Other;
}

std::string to_string(FlipKind kind) {
    switch (kind) {
    case FlipKind::Diagonal:
        return "diagonal";
    case FlipKind::SingleSite:
        return "single_site";
    case FlipKind::HorizontalNearest:
        return "horizontal_nearest";
    case FlipKind::VerticalNearest:
        return "vertical_nearest";
    case FlipKind::Plaquette2x2:
        return "plaquette_2x2";
    case FlipKind::HorizontalLong:
        return "horizontal_long";
    case FlipKind::Other:
        return "other";
    }
    return "unknown";
}

std::size_t flip_kind_index(FlipKind kind) {
    switch (kind) {
    case FlipKind::Diagonal:
        return 0;
    case FlipKind::SingleSite:
        return 1;
    case FlipKind::HorizontalNearest:
        return 2;
    case FlipKind::VerticalNearest:
        return 3;
    case FlipKind::Plaquette2x2:
        return 4;
    case FlipKind::HorizontalLong:
        return 5;
    case FlipKind::Other:
        return 6;
    }
    return 6;
}

std::vector<FlipContribution>
enumerate_flip_contributions(const Sample &sample,
                             const LocalOperatorTerm &term) {
    const int support = static_cast<int>(term.sites.size());
    const int dim =
        static_cast<int>(checked_pow(static_cast<std::size_t>(term.local_dim),
                                     support));
    if (static_cast<int>(term.matrix.size()) != dim * dim) {
        throw std::invalid_argument("operator matrix has wrong size");
    }
    const int in_code = local_state_code(sample, term.sites);
    std::vector<FlipContribution> out;
    for (int out_code = 0; out_code < dim; ++out_code) {
        const Complex elem =
            term.coefficient *
            term.matrix[static_cast<std::size_t>(out_code * dim + in_code)];
        if (std::abs(elem) == 0.0) {
            continue;
        }
        std::vector<int> values =
            decode_local_values(out_code, support, term.local_dim);
        std::vector<int> changed_sites;
        std::vector<int> changed_values;
        for (int i = 0; i < support; ++i) {
            const int site = term.sites[static_cast<std::size_t>(i)];
            if (sample.spin[static_cast<std::size_t>(site)] !=
                values[static_cast<std::size_t>(i)]) {
                changed_sites.push_back(site);
                changed_values.push_back(values[static_cast<std::size_t>(i)]);
            }
        }
        FlipContribution flip;
        flip.sites = std::move(changed_sites);
        flip.output_values = std::move(changed_values);
        flip.matrix_element = elem;
        flip.kind = classify_flip_sites(flip.sites, sample.ly);
        out.push_back(std::move(flip));
    }
    return out;
}

std::array<std::size_t, flip_kind_count>
summarize_flip_buckets(const Sample &sample,
                       const std::vector<LocalOperatorTerm> &hamiltonian) {
    std::array<std::size_t, flip_kind_count> out{};
    for (const LocalOperatorTerm &term : hamiltonian) {
        for (const FlipContribution &flip :
             enumerate_flip_contributions(sample, term)) {
            ++out[flip_kind_index(flip.kind)];
        }
    }
    return out;
}

Complex local_energy_exact(const PEPS &peps, const Sample &sample,
                           const std::vector<LocalOperatorTerm> &hamiltonian,
                           Complex psi) {
    if (std::abs(psi) == 0.0) {
        throw std::runtime_error("cannot compute local energy for zero amplitude");
    }
    Complex energy = {0.0, 0.0};
    for (const LocalOperatorTerm &term : hamiltonian) {
        for (const FlipContribution &flip :
             enumerate_flip_contributions(sample, term)) {
            Sample flipped = apply_local_values(sample, flip.sites, flip.output_values);
            const Complex psi_flipped = contract_amplitude_exact(peps, flipped);
            energy += flip.matrix_element * psi_flipped / psi;
        }
    }
    return energy;
}

std::vector<Complex> log_gradients_exact(const PEPS &peps,
                                         const Sample &sample, Complex psi,
                                         std::size_t max_parameters) {
    const std::size_t parameters = peps.parameter_count();
    if (parameters > max_parameters) {
        throw std::runtime_error("gradient guard exceeded; lower D/L or raise cap");
    }
    if (std::abs(psi) == 0.0) {
        throw std::runtime_error("cannot compute log gradients for zero amplitude");
    }

    std::vector<Complex> out(parameters);
    std::size_t offset = 0;
    for (int x = 0; x < peps.lx; ++x) {
        for (int y = 0; y < peps.ly; ++y) {
            PEPS basis = peps;
            SiteTensor &site = basis.at(x, y);
            const std::size_t count = site.parameter_count();
            std::fill(site.data.begin(), site.data.end(), Complex{0.0, 0.0});
            for (std::size_t k = 0; k < count; ++k) {
                site.data[k] = Complex{1.0, 0.0};
                out[offset + k] = contract_amplitude_exact(basis, sample) / psi;
                site.data[k] = Complex{0.0, 0.0};
            }
            offset += count;
        }
    }
    return out;
}

Sample sample_exact_small(const PEPS &peps, std::mt19937_64 &rng,
                          std::size_t max_enumerated_states) {
    return draw_exact_sample(
        build_exact_distribution(peps, max_enumerated_states), rng);
}

SampleEvaluation evaluate_sample_exact(
    const PEPS &peps, const Sample &sample,
    const std::vector<LocalOperatorTerm> &hamiltonian,
    std::size_t max_gradient_parameters) {
    SampleEvaluation eval;
    eval.sample = sample;
    eval.psi = contract_amplitude_exact(peps, sample);
    eval.log_psi = std::log(eval.psi);
    eval.local_energy = local_energy_exact(peps, sample, hamiltonian, eval.psi);
    eval.log_gradients =
        log_gradients_exact(peps, sample, eval.psi, max_gradient_parameters);
    return eval;
}

SampleBatch generate_Oks_and_Eks_exact(
    const PEPS &peps, const std::vector<LocalOperatorTerm> &hamiltonian,
    int sample_count, std::uint64_t seed, std::size_t max_enumerated_states,
    std::size_t max_gradient_parameters) {
    if (sample_count <= 0) {
        throw std::invalid_argument("sample_count must be positive");
    }

    std::mt19937_64 rng(seed);
    const ExactDistribution distribution =
        build_exact_distribution(peps, max_enumerated_states);
    SampleBatch batch;
    batch.samples.reserve(static_cast<std::size_t>(sample_count));
    batch.log_psi.reserve(static_cast<std::size_t>(sample_count));
    batch.local_energy.reserve(static_cast<std::size_t>(sample_count));
    batch.importance_weights.assign(static_cast<std::size_t>(sample_count), 1.0);
    batch.O.reserve(static_cast<std::size_t>(sample_count));

    for (int i = 0; i < sample_count; ++i) {
        Sample sample = draw_exact_sample(distribution, rng);
        SampleEvaluation eval =
            evaluate_sample_exact(peps, sample, hamiltonian, max_gradient_parameters);
        batch.samples.push_back(std::move(eval.sample));
        batch.log_psi.push_back(eval.log_psi);
        batch.local_energy.push_back(eval.local_energy);
        batch.O.push_back(std::move(eval.log_gradients));
    }
    return batch;
}

std::vector<Complex> minsr_direction(const std::vector<std::vector<Complex>> &O,
                                     const std::vector<Complex> &E,
                                     double diagonal_shift) {
    return minsr_direction_weighted(
        O, E, std::vector<double>(O.size(), 1.0), diagonal_shift);
}

std::vector<Complex>
sr_direction_parameter_space(const std::vector<std::vector<Complex>> &O,
                             const std::vector<Complex> &E,
                             double diagonal_shift) {
    const int ns = static_cast<int>(O.size());
    if (ns == 0 || static_cast<int>(E.size()) != ns) {
        throw std::invalid_argument("O/E sample dimensions do not match");
    }
    const int np = static_cast<int>(O.front().size());
    for (const auto &row : O) {
        if (static_cast<int>(row.size()) != np) {
            throw std::invalid_argument("ragged O matrix");
        }
    }

    std::vector<std::vector<Complex>> g(static_cast<std::size_t>(np),
                                        std::vector<Complex>(
                                            static_cast<std::size_t>(np)));
    std::vector<Complex> rhs(static_cast<std::size_t>(np), Complex{0.0, 0.0});
    for (int p = 0; p < np; ++p) {
        for (int q = 0; q < np; ++q) {
            Complex dot = {0.0, 0.0};
            for (int s = 0; s < ns; ++s) {
                dot += std::conj(O[static_cast<std::size_t>(s)]
                                  [static_cast<std::size_t>(p)]) *
                       O[static_cast<std::size_t>(s)][static_cast<std::size_t>(q)];
            }
            g[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] = dot;
        }
        g[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)] +=
            cx(diagonal_shift);
        for (int s = 0; s < ns; ++s) {
            rhs[static_cast<std::size_t>(p)] -=
                std::conj(O[static_cast<std::size_t>(s)]
                           [static_cast<std::size_t>(p)]) *
                E[static_cast<std::size_t>(s)];
        }
    }
    return solve_dense(std::move(g), std::move(rhs));
}

std::vector<Complex>
minsr_direction_weighted(const std::vector<std::vector<Complex>> &O,
                         const std::vector<Complex> &E,
                         const std::vector<double> &weights,
                         double diagonal_shift) {
    const int ns = static_cast<int>(O.size());
    if (ns == 0 || static_cast<int>(E.size()) != ns ||
        static_cast<int>(weights.size()) != ns) {
        throw std::invalid_argument("O/E/weight sample dimensions do not match");
    }
    const int np = static_cast<int>(O.front().size());
    for (const auto &row : O) {
        if (static_cast<int>(row.size()) != np) {
            throw std::invalid_argument("ragged O matrix");
        }
    }
    for (double weight : weights) {
        if (!(weight >= 0.0) || !std::isfinite(weight)) {
            throw std::invalid_argument("minSR weights must be finite and nonnegative");
        }
    }

    std::vector<std::vector<Complex>> t(static_cast<std::size_t>(ns),
                                        std::vector<Complex>(
                                            static_cast<std::size_t>(ns)));
    for (int s = 0; s < ns; ++s) {
        for (int sp = 0; sp < ns; ++sp) {
            Complex dot = {0.0, 0.0};
            const Real scale = as_real(std::sqrt(
                weights[static_cast<std::size_t>(s)] *
                weights[static_cast<std::size_t>(sp)]));
            for (int p = 0; p < np; ++p) {
                dot += scale *
                       O[static_cast<std::size_t>(s)][static_cast<std::size_t>(p)] *
                       std::conj(O[static_cast<std::size_t>(sp)]
                                  [static_cast<std::size_t>(p)]);
            }
            t[static_cast<std::size_t>(s)][static_cast<std::size_t>(sp)] = dot;
        }
        t[static_cast<std::size_t>(s)][static_cast<std::size_t>(s)] +=
            cx(diagonal_shift);
    }

    std::vector<Complex> weighted_e(E.size());
    for (int s = 0; s < ns; ++s) {
        weighted_e[static_cast<std::size_t>(s)] =
            as_real(std::sqrt(weights[static_cast<std::size_t>(s)])) *
            E[static_cast<std::size_t>(s)];
    }

    const std::vector<Complex> x = solve_dense(std::move(t), std::move(weighted_e));
    std::vector<Complex> direction(static_cast<std::size_t>(np), Complex{0.0, 0.0});
    for (int p = 0; p < np; ++p) {
        for (int s = 0; s < ns; ++s) {
            direction[static_cast<std::size_t>(p)] -=
                as_real(std::sqrt(weights[static_cast<std::size_t>(s)])) *
                std::conj(O[static_cast<std::size_t>(s)][static_cast<std::size_t>(p)]) *
                x[static_cast<std::size_t>(s)];
        }
    }
    return direction;
}

std::vector<Complex> minsr_direction_sampled_sector(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    const std::vector<Complex> &E, double diagonal_shift) {
    return minsr_direction_sampled_sector_weighted(
        peps, samples, sampled_sector_rows, E,
        std::vector<double>(samples.size(), 1.0), diagonal_shift);
}

std::vector<Complex> minsr_direction_sampled_sector_weighted(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    const std::vector<Complex> &E, const std::vector<double> &weights,
    double diagonal_shift) {
    const std::size_t ns = samples.size();
    if (ns == 0 || E.size() != ns || weights.size() != ns ||
        sampled_sector_rows.size() != ns) {
        throw std::invalid_argument(
            "sampled-sector minSR sample dimensions do not match");
    }

    const ParameterLayout layout = make_parameter_layout(peps);
    std::vector<std::size_t> sampled_offsets;
    std::vector<std::size_t> slice_sizes;
    sampled_offsets.reserve(peps.sites.size());
    slice_sizes.reserve(peps.sites.size());
    std::size_t sampled_parameter_count = 0;
    for (const SiteTensor &site : peps.sites) {
        const std::size_t slice_size = static_cast<std::size_t>(site.north) *
                                       static_cast<std::size_t>(site.east) *
                                       static_cast<std::size_t>(site.south) *
                                       static_cast<std::size_t>(site.west);
        sampled_offsets.push_back(sampled_parameter_count);
        slice_sizes.push_back(slice_size);
        sampled_parameter_count += slice_size;
    }

    for (std::size_t s = 0; s < ns; ++s) {
        if (samples[s].spin.size() != peps.sites.size()) {
            throw std::invalid_argument("sample and PEPS site count mismatch");
        }
        if (sampled_sector_rows[s].size() != sampled_parameter_count) {
            throw std::invalid_argument("ragged sampled-sector O rows");
        }
        if (!(weights[s] >= 0.0) || !std::isfinite(weights[s])) {
            throw std::invalid_argument(
                "sampled-sector minSR weights must be finite and nonnegative");
        }
    }

    std::vector<std::vector<Complex>> t(
        ns, std::vector<Complex>(ns, Complex{0.0, 0.0}));
    for (std::size_t row = 0; row < ns; ++row) {
        for (std::size_t col = 0; col < ns; ++col) {
            Complex dot = {0.0, 0.0};
            const Real scale = as_real(std::sqrt(weights[row] * weights[col]));
            for (std::size_t site = 0; site < peps.sites.size(); ++site) {
                if (samples[row].spin[site] != samples[col].spin[site]) {
                    continue;
                }
                const std::size_t begin = sampled_offsets[site];
                const std::size_t end = begin + slice_sizes[site];
                for (std::size_t p = begin; p < end; ++p) {
                    dot += scale * sampled_sector_rows[row][p] *
                           std::conj(sampled_sector_rows[col][p]);
                }
            }
            if (row == col) {
                dot += cx(diagonal_shift);
            }
            t[row][col] = dot;
        }
    }

    std::vector<Complex> weighted_e(E.size());
    for (std::size_t s = 0; s < ns; ++s) {
        weighted_e[s] = as_real(std::sqrt(weights[s])) * E[s];
    }
    const std::vector<Complex> x = solve_dense(std::move(t), std::move(weighted_e));

    std::vector<Complex> direction(layout.total_parameters, Complex{0.0, 0.0});
    for (std::size_t s = 0; s < ns; ++s) {
        const Real scale = as_real(std::sqrt(weights[s]));
        for (std::size_t site = 0; site < peps.sites.size(); ++site) {
            const int spin = samples[s].spin[site];
            const SiteTensor &site_tensor = peps.sites[site];
            if (spin < 0 || spin >= site_tensor.phys) {
                throw std::invalid_argument("sample spin outside local dimension");
            }
            const ParameterBlock &block = layout.blocks[site];
            const std::size_t src = sampled_offsets[site];
            const std::size_t dst =
                block.offset + static_cast<std::size_t>(spin) *
                                   block.physical_stride;
            for (std::size_t p = 0; p < slice_sizes[site]; ++p) {
                direction[dst + p] -=
                    scale * std::conj(sampled_sector_rows[s][src + p]) * x[s];
            }
        }
    }
    return direction;
}

std::vector<LocalOperatorTerm> make_nearest_neighbor_heisenberg(int lx, int ly,
                                                                double j1) {
    std::vector<LocalOperatorTerm> terms;
    auto add_pair = [&](int a, int b) {
        LocalOperatorTerm term;
        term.sites = {a, b};
        term.local_dim = 2;
        term.coefficient = cx(j1);
        term.matrix.assign(16, Complex{0.0, 0.0});
        for (int in = 0; in < 4; ++in) {
            const int s0 = in / 2;
            const int s1 = in % 2;
            const double z0 = (s0 == 0) ? 0.5 : -0.5;
            const double z1 = (s1 == 0) ? 0.5 : -0.5;
            term.matrix[static_cast<std::size_t>(in * 4 + in)] +=
                cx(z0 * z1);
            if (s0 != s1) {
                const int out = s1 * 2 + s0;
                term.matrix[static_cast<std::size_t>(out * 4 + in)] +=
                    Complex{0.5, 0.0};
            }
        }
        terms.push_back(std::move(term));
    };

    for (int x = 0; x < lx; ++x) {
        for (int y = 0; y < ly; ++y) {
            const int site = x * ly + y;
            if (x + 1 < lx) {
                add_pair(site, (x + 1) * ly + y);
            }
            if (y + 1 < ly) {
                add_pair(site, x * ly + (y + 1));
            }
        }
    }
    return terms;
}

std::vector<LocalOperatorTerm>
make_transverse_field_ising(int lx, int ly, double jz, double hx) {
    std::vector<LocalOperatorTerm> terms;
    terms.reserve(static_cast<std::size_t>(lx * ly * 3));

    for (int site = 0; site < lx * ly; ++site) {
        LocalOperatorTerm x_term;
        x_term.sites = {site};
        x_term.local_dim = 2;
        x_term.coefficient = cx(-hx);
        x_term.matrix = {Complex{0.0, 0.0}, Complex{1.0, 0.0},
                         Complex{1.0, 0.0}, Complex{0.0, 0.0}};
        terms.push_back(std::move(x_term));
    }

    auto add_pair = [&](int a, int b) {
        LocalOperatorTerm term;
        term.sites = {a, b};
        term.local_dim = 2;
        term.coefficient = cx(jz);
        term.matrix.assign(16, Complex{0.0, 0.0});
        for (int in = 0; in < 4; ++in) {
            const int s0 = in / 2;
            const int s1 = in % 2;
            const double z0 = (s0 == 0) ? 1.0 : -1.0;
            const double z1 = (s1 == 0) ? 1.0 : -1.0;
            term.matrix[static_cast<std::size_t>(in * 4 + in)] =
                cx(z0 * z1);
        }
        terms.push_back(std::move(term));
    };

    for (int x = 0; x < lx; ++x) {
        for (int y = 0; y < ly; ++y) {
            const int site = x * ly + y;
            if (x + 1 < lx) {
                add_pair(site, (x + 1) * ly + y);
            }
            if (y + 1 < ly) {
                add_pair(site, x * ly + (y + 1));
            }
        }
    }
    return terms;
}

std::vector<LocalOperatorTerm>
make_square_rydberg_hamiltonian(int lx, int ly, double omega, double detuning,
                                double c6, double cutoff_radius) {
    if (lx <= 0 || ly <= 0) {
        throw std::invalid_argument("invalid lattice dimensions");
    }

    std::vector<LocalOperatorTerm> terms;
    terms.reserve(static_cast<std::size_t>(lx * ly * 2));

    for (int site = 0; site < lx * ly; ++site) {
        LocalOperatorTerm term;
        term.sites = {site};
        term.local_dim = 2;
        term.coefficient = cx(1.0);
        term.matrix = {cx(0.0), cx(0.5 * omega), cx(0.5 * omega),
                       cx(-detuning)};
        terms.push_back(std::move(term));
    }

    for (int a = 0; a < lx * ly; ++a) {
        const int ax = a / ly;
        const int ay = a % ly;
        for (int b = a + 1; b < lx * ly; ++b) {
            const int bx = b / ly;
            const int by = b % ly;
            const double dx = static_cast<double>(ax - bx);
            const double dy = static_cast<double>(ay - by);
            const double r = std::sqrt(dx * dx + dy * dy);
            if (cutoff_radius > 0.0 && r > cutoff_radius) {
                continue;
            }
            const double v = c6 / std::pow(r, 6.0);
            LocalOperatorTerm term;
            term.sites = {a, b};
            term.local_dim = 2;
            term.coefficient = cx(v);
            term.matrix.assign(16, Complex{0.0, 0.0});
            term.matrix[15] = Complex{1.0, 0.0};
            terms.push_back(std::move(term));
        }
    }
    return terms;
}

std::vector<int> make_nearest_neighbor_bond_pairs(int lx, int ly) {
    if (lx <= 0 || ly <= 0) {
        throw std::invalid_argument("invalid lattice dimensions");
    }
    std::vector<int> pairs;
    pairs.reserve(static_cast<std::size_t>(2 * (lx * (ly - 1) + ly * (lx - 1))));
    for (int x = 0; x < lx; ++x) {
        for (int y = 0; y < ly; ++y) {
            const int site = x * ly + y;
            if (x + 1 < lx) {
                pairs.push_back(site);
                pairs.push_back((x + 1) * ly + y);
            }
            if (y + 1 < ly) {
                pairs.push_back(site);
                pairs.push_back(x * ly + (y + 1));
            }
        }
    }
    return pairs;
}

std::vector<int> flatten_samples_sample_major(
    const std::vector<Sample> &samples) {
    if (samples.empty()) {
        return {};
    }
    const int lx = samples.front().lx;
    const int ly = samples.front().ly;
    const int local_dim = samples.front().local_dim;
    const std::size_t site_count = samples.front().spin.size();
    std::vector<int> out;
    out.reserve(samples.size() * site_count);
    for (const Sample &sample : samples) {
        if (sample.lx != lx || sample.ly != ly || sample.local_dim != local_dim ||
            sample.spin.size() != site_count) {
            throw std::invalid_argument("ragged sample batch cannot be flattened");
        }
        out.insert(out.end(), sample.spin.begin(), sample.spin.end());
    }
    return out;
}

std::size_t dense_o_bytes(int sample_count, std::size_t parameter_count) {
    if (sample_count < 0) {
        throw std::invalid_argument("negative sample_count");
    }
    const std::size_t samples = static_cast<std::size_t>(sample_count);
    if (parameter_count != 0 &&
        samples > std::numeric_limits<std::size_t>::max() / parameter_count) {
        throw std::overflow_error("dense O element count overflow");
    }
    const std::size_t elems = samples * parameter_count;
    if (elems > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
        throw std::overflow_error("dense O byte count overflow");
    }
    return elems * sizeof(Complex);
}

std::size_t sampled_sector_parameter_count(const PEPS &peps) {
    std::size_t count = 0;
    for (const SiteTensor &site : peps.sites) {
        count += static_cast<std::size_t>(site.north) *
                 static_cast<std::size_t>(site.east) *
                 static_cast<std::size_t>(site.south) *
                 static_cast<std::size_t>(site.west);
    }
    return count;
}

std::size_t sampled_sector_o_bytes(int sample_count,
                                   std::size_t sampled_parameter_count) {
    return dense_o_bytes(sample_count, sampled_parameter_count);
}

std::vector<Complex>
compact_sampled_sector_log_gradients(const PEPS &peps, const Sample &sample,
                                     const std::vector<Complex> &dense_row) {
    if (sample.spin.size() != peps.sites.size()) {
        throw std::invalid_argument("sample and PEPS site count mismatch");
    }
    const ParameterLayout layout = make_parameter_layout(peps);
    if (dense_row.size() != layout.total_parameters) {
        throw std::invalid_argument("dense O row length does not match PEPS");
    }

    std::vector<Complex> compact;
    compact.reserve(sampled_sector_parameter_count(peps));
    for (const ParameterBlock &block : layout.blocks) {
        const std::size_t site_index = block.site;
        const SiteTensor &site = peps.sites[site_index];
        const int spin = sample.spin[site_index];
        if (spin < 0 || spin >= site.phys) {
            throw std::invalid_argument("sample spin outside local dimension");
        }
        const std::size_t begin = block.offset +
                                  static_cast<std::size_t>(spin) *
                                      block.physical_stride;
        compact.insert(compact.end(), dense_row.begin() + begin,
                       dense_row.begin() + begin + block.physical_stride);
    }
    return compact;
}

std::vector<std::vector<Complex>> compact_sampled_sector_log_gradients(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &dense_rows) {
    if (samples.size() != dense_rows.size()) {
        throw std::invalid_argument("sample and O row counts do not match");
    }
    std::vector<std::vector<Complex>> compact;
    compact.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        compact.push_back(
            compact_sampled_sector_log_gradients(peps, samples[i], dense_rows[i]));
    }
    return compact;
}

std::vector<std::vector<Complex>> sampled_sector_gram(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    double diagonal_shift) {
    const std::size_t ns = samples.size();
    if (sampled_sector_rows.size() != ns) {
        throw std::invalid_argument("sample and sampled-sector row counts differ");
    }

    std::vector<std::size_t> sampled_offsets;
    std::vector<std::size_t> slice_sizes;
    sampled_offsets.reserve(peps.sites.size());
    slice_sizes.reserve(peps.sites.size());
    std::size_t cursor = 0;
    for (const SiteTensor &site : peps.sites) {
        const std::size_t slice_size = static_cast<std::size_t>(site.north) *
                                       static_cast<std::size_t>(site.east) *
                                       static_cast<std::size_t>(site.south) *
                                       static_cast<std::size_t>(site.west);
        sampled_offsets.push_back(cursor);
        slice_sizes.push_back(slice_size);
        cursor += slice_size;
    }

    for (std::size_t s = 0; s < ns; ++s) {
        if (samples[s].spin.size() != peps.sites.size()) {
            throw std::invalid_argument("sample and PEPS site count mismatch");
        }
        if (sampled_sector_rows[s].size() != cursor) {
            throw std::invalid_argument("ragged sampled-sector O rows");
        }
    }

    std::vector<std::vector<Complex>> gram(
        ns, std::vector<Complex>(ns, Complex{0.0, 0.0}));
    for (std::size_t row = 0; row < ns; ++row) {
        for (std::size_t col = 0; col < ns; ++col) {
            Complex dot = {0.0, 0.0};
            for (std::size_t site = 0; site < peps.sites.size(); ++site) {
                if (samples[row].spin[site] != samples[col].spin[site]) {
                    continue;
                }
                const std::size_t begin = sampled_offsets[site];
                const std::size_t end = begin + slice_sizes[site];
                for (std::size_t p = begin; p < end; ++p) {
                    dot += sampled_sector_rows[row][p] *
                           std::conj(sampled_sector_rows[col][p]);
                }
            }
            if (row == col) {
                dot += cx(diagonal_shift);
            }
            gram[row][col] = dot;
        }
    }
    return gram;
}

std::vector<double>
compute_importance_weights(const std::vector<Complex> &log_psi,
                           const std::vector<double> &log_sampling_probability) {
    if (log_psi.size() != log_sampling_probability.size()) {
        throw std::invalid_argument("importance-weight inputs have different sizes");
    }
    if (log_psi.empty()) {
        return {};
    }

    std::vector<double> log_ratios(log_psi.size(), 0.0);
    double max_ratio = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < log_psi.size(); ++i) {
        log_ratios[i] = 2.0 * log_psi[i].real() - log_sampling_probability[i];
        max_ratio = std::max(max_ratio, log_ratios[i]);
    }

    double shifted_sum = 0.0;
    for (double value : log_ratios) {
        shifted_sum += std::exp(value - max_ratio);
    }
    const double log_z =
        max_ratio + std::log(shifted_sum) -
        std::log(static_cast<double>(log_sampling_probability.size()));

    std::vector<double> weights(log_psi.size(), 0.0);
    for (std::size_t i = 0; i < log_psi.size(); ++i) {
        weights[i] = std::exp(log_ratios[i] - log_z);
    }
    return weights;
}

PackedPEPS pack_site_tensors_physical_major(const PEPS &peps,
                                            std::size_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument("alignment must be at least one");
    }
    PackedPEPS packed;
    packed.lx = peps.lx;
    packed.ly = peps.ly;
    packed.local_dim = peps.local_dim;
    packed.sites.reserve(peps.sites.size());
    packed.projected_offsets.reserve(peps.sites.size());

    std::size_t data_cursor = 0;
    std::size_t projected_cursor = 0;
    for (const SiteTensor &site : peps.sites) {
        PackedSiteMetadata meta;
        meta.phys = site.phys;
        meta.north = site.north;
        meta.east = site.east;
        meta.south = site.south;
        meta.west = site.west;
        meta.physical_slice_size = static_cast<std::size_t>(site.north) *
                                   static_cast<std::size_t>(site.east) *
                                   static_cast<std::size_t>(site.south) *
                                   static_cast<std::size_t>(site.west);
        data_cursor = align_up(data_cursor, alignment);
        projected_cursor = align_up(projected_cursor, alignment);
        meta.offset = data_cursor;
        packed.projected_offsets.push_back(projected_cursor);
        packed.sites.push_back(meta);
        data_cursor += static_cast<std::size_t>(site.phys) *
                       meta.physical_slice_size;
        projected_cursor += meta.physical_slice_size;
    }

    packed.projected_sample_stride = align_up(projected_cursor, alignment);
    packed.data.assign(data_cursor, Complex{0.0, 0.0});
    for (std::size_t site_index = 0; site_index < peps.sites.size();
         ++site_index) {
        const SiteTensor &site = peps.sites[site_index];
        const PackedSiteMetadata &meta = packed.sites[site_index];
        for (int p = 0; p < site.phys; ++p) {
            for (int n = 0; n < site.north; ++n) {
                for (int e = 0; e < site.east; ++e) {
                    for (int s = 0; s < site.south; ++s) {
                        for (int w = 0; w < site.west; ++w) {
                            const std::size_t packed_index =
                                meta.offset +
                                static_cast<std::size_t>(p) *
                                    meta.physical_slice_size +
                                (((static_cast<std::size_t>(n) * site.east +
                                   static_cast<std::size_t>(e)) *
                                      site.south +
                                  static_cast<std::size_t>(s)) *
                                     site.west +
                                 static_cast<std::size_t>(w));
                            packed.data[packed_index] = site(p, n, e, s, w);
                        }
                    }
                }
            }
        }
    }
    return packed;
}

ParameterLayout make_parameter_layout(const PEPS &peps) {
    ParameterLayout layout;
    layout.blocks.reserve(peps.sites.size());
    std::size_t offset = 0;
    for (std::size_t site_index = 0; site_index < peps.sites.size();
         ++site_index) {
        const SiteTensor &site = peps.sites[site_index];
        ParameterBlock block;
        block.site = site_index;
        block.offset = offset;
        block.count = site.parameter_count();
        block.physical_stride = static_cast<std::size_t>(site.north) *
                                static_cast<std::size_t>(site.east) *
                                static_cast<std::size_t>(site.south) *
                                static_cast<std::size_t>(site.west);
        layout.blocks.push_back(block);
        offset += block.count;
    }
    layout.total_parameters = offset;
    return layout;
}

double squared_norm(const std::vector<Complex> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0,
                           [](double acc, Complex z) { return acc + std::norm(z); });
}

} // namespace peps_cuda
