#include "peps_cuda/memory.hpp"
#include "peps_cuda/peps.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const int lx = argc > 1 ? std::atoi(argv[1]) : 2;
    const int ly = argc > 2 ? std::atoi(argv[2]) : 2;
    const int bond_dim = argc > 3 ? std::atoi(argv[3]) : 2;
    const int samples = argc > 4 ? std::atoi(argv[4]) : 4;
    const std::string model = argc > 5 ? argv[5] : "heisenberg";

    try {
        std::vector<std::pair<std::string, peps_cuda::MemorySnapshot>>
            memory_trace;
        const auto trace_memory = [&](std::string stage) {
            memory_trace.emplace_back(std::move(stage),
                                      peps_cuda::get_process_memory_snapshot());
        };

        trace_memory("start");
        const peps_cuda::PEPS peps =
            peps_cuda::make_random_open_peps(lx, ly, 2, bond_dim, 1234, 0.1, 0.25);
        trace_memory("after_peps_init");
        std::vector<peps_cuda::LocalOperatorTerm> ham;
        if (model == "heisenberg") {
            ham = peps_cuda::make_nearest_neighbor_heisenberg(lx, ly, 1.0);
        } else if (model == "tfi") {
            ham = peps_cuda::make_transverse_field_ising(lx, ly, 1.0, 0.5);
        } else if (model == "rydberg") {
            ham = peps_cuda::make_square_rydberg_hamiltonian(lx, ly, 1.0, 2.0,
                                                             1.0, -1.0);
        } else {
            throw std::invalid_argument("model must be heisenberg, tfi, or rydberg");
        }
        trace_memory("after_hamiltonian");
        peps_cuda::SampleBatch batch =
            peps_cuda::generate_Oks_and_Eks_exact(peps, ham, samples, 99);
        const double minsr_shift =
            (sizeof(peps_cuda::Real) == sizeof(float)) ? 1.0e-4 : 1.0e-6;
        trace_memory("after_generate_O_E");
        const std::vector<peps_cuda::Complex> direction =
            peps_cuda::minsr_direction(batch.O, batch.local_energy, minsr_shift);
        trace_memory("after_minsr");
        const std::vector<int> flattened_spins =
            peps_cuda::flatten_samples_sample_major(batch.samples);

        std::cout << "PEPS CUDA scaffold CPU exact demo\n";
        std::cout << "lattice=" << lx << "x" << ly << " D=" << bond_dim
                  << " samples=" << samples << " model=" << model << "\n";
        std::cout << "parameters=" << peps.parameter_count() << "\n";
        std::cout << "sampled-sector parameters="
                  << peps_cuda::sampled_sector_parameter_count(peps) << "\n";
        std::cout << "dense O bytes="
                  << peps_cuda::dense_o_bytes(samples, peps.parameter_count())
                  << "\n";
        std::cout << "sampled-sector O bytes="
                  << peps_cuda::sampled_sector_o_bytes(
                         samples, peps_cuda::sampled_sector_parameter_count(peps))
                  << "\n";
        std::cout << "flattened sample spins=" << flattened_spins.size() << "\n";
        std::cout << "first log(psi)=" << batch.log_psi.front() << "\n";
        std::cout << "first E_loc=" << batch.local_energy.front() << "\n";
        const auto flips =
            peps_cuda::enumerate_flip_contributions(batch.samples.front(), ham.front());
        const auto flip_summary =
            peps_cuda::summarize_flip_buckets(batch.samples.front(), ham);
        std::cout << "first Hamiltonian term emits " << flips.size()
                  << " flip contributions; first kind="
                  << peps_cuda::to_string(flips.front().kind) << "\n";
        std::cout << "flip bucket counts: diagonal=" << flip_summary[0]
                  << " single=" << flip_summary[1]
                  << " horizontal=" << flip_summary[2]
                  << " vertical=" << flip_summary[3]
                  << " plaquette=" << flip_summary[4]
                  << " horizontal_long=" << flip_summary[5]
                  << " other=" << flip_summary[6] << "\n";
        std::cout << "minSR direction squared norm="
                  << peps_cuda::squared_norm(direction) << "\n";
        std::cout << "memory trace:\n";
        for (const auto &[stage, snapshot] : memory_trace) {
            std::cout << "  " << stage
                      << " current_rss="
                      << peps_cuda::format_bytes(snapshot.current_rss_bytes)
                      << " peak_rss="
                      << peps_cuda::format_bytes(snapshot.peak_rss_bytes) << "\n";
        }
    } catch (const std::exception &err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }
    return 0;
}
