#include "peps_cuda/memory.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace peps_cuda {
namespace {

std::size_t current_rss() {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t result =
        task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count);
    if (result == KERN_SUCCESS) {
        return static_cast<std::size_t>(info.resident_size);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::size_t size_pages = 0;
    std::size_t resident_pages = 0;
    statm >> size_pages >> resident_pages;
    return resident_pages * static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

std::size_t peak_rss() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
#endif
}

} // namespace

MemorySnapshot get_process_memory_snapshot() {
    return MemorySnapshot{current_rss(), peak_rss()};
}

std::string format_bytes(std::size_t bytes) {
    static constexpr std::array<const char *, 5> units = {"B", "KiB", "MiB",
                                                          "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit_index = 0;
    while (value >= 1024.0 && unit_index + 1 < units.size()) {
        value /= 1024.0;
        ++unit_index;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, units[unit_index]);
    return std::string(buffer);
}

} // namespace peps_cuda
