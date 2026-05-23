#pragma once

#include <cstddef>
#include <string>

namespace peps_cuda {

struct MemorySnapshot {
    std::size_t current_rss_bytes = 0;
    std::size_t peak_rss_bytes = 0;
};

MemorySnapshot get_process_memory_snapshot();

std::string format_bytes(std::size_t bytes);

} // namespace peps_cuda
