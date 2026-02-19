#!/usr/bin/env python3
"""Interactive CUDA transfer timeline visualizer.

This script synthesizes H2D, kernel, and D2H activity across CUDA streams and
shows how overlap changes end-to-end runtime.

Examples:
    python3 scripts/cuda_transfer_timeline_demo.py
    python3 scripts/cuda_transfer_timeline_demo.py --no-gui \
        --output images/cuda_transfer_timeline_demo.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    stream: int
    kind: str
    start_us: float
    duration_us: float

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us


def transfer_duration_us(megabytes: float, gb_per_second: float) -> float:
    bytes_count = megabytes * 1024.0 * 1024.0
    return (bytes_count / (gb_per_second * 1e9)) * 1e6


def simulate_timeline(
    *,
    stream_count: int,
    chunks_per_stream: int,
    h2d_mb: float,
    d2h_mb: float,
    h2d_bw_gbps: float,
    d2h_bw_gbps: float,
    base_kernel_us: float,
    kernel_scale: float,
    overlap: float,
    launch_gap_us: float,
) -> tuple[list[Event], float]:
    h2d_us = transfer_duration_us(h2d_mb, h2d_bw_gbps)
    d2h_us = transfer_duration_us(d2h_mb, d2h_bw_gbps)
    kernel_us = max(1.0, base_kernel_us * kernel_scale)

    per_chunk_serial = h2d_us + launch_gap_us + kernel_us + launch_gap_us + d2h_us
    serial_total = per_chunk_serial * stream_count * chunks_per_stream

    period = max(launch_gap_us, per_chunk_serial * (1.0 - overlap))
    stream_offset = period / max(1, stream_count)

    events: list[Event] = []
    for chunk in range(chunks_per_stream):
        for stream in range(stream_count):
            base_start = chunk * period + stream * stream_offset
            h2d_start = base_start
            kernel_start = h2d_start + h2d_us + launch_gap_us
            d2h_start = kernel_start + kernel_us + launch_gap_us

            events.append(Event(stream=stream, kind="H2D", start_us=h2d_start, duration_us=h2d_us))
            events.append(
                Event(
                    stream=stream,
                    kind="Kernel",
                    start_us=kernel_start,
                    duration_us=kernel_us,
                )
            )
            events.append(Event(stream=stream, kind="D2H", start_us=d2h_start, duration_us=d2h_us))

    return events, serial_total


def timeline_makespan(events: list[Event]) -> float:
    if not events:
        return 0.0
    start = min(event.start_us for event in events)
    end = max(event.end_us for event in events)
    return max(0.0, end - start)


def draw_timeline(
    ax,
    *,
    events: list[Event],
    stream_count: int,
    serial_total: float,
    h2d_bw_gbps: float,
    d2h_bw_gbps: float,
    kernel_scale: float,
    overlap: float,
) -> None:
    colors = {"H2D": "#2563EB", "Kernel": "#DC2626", "D2H": "#059669"}
    ax.clear()

    handles: dict[str, object] = {}
    lane_height = 0.76
    for event in events:
        y_bottom = event.stream - (lane_height / 2.0)
        bar = ax.broken_barh(
            [(event.start_us, event.duration_us)],
            (y_bottom, lane_height),
            facecolors=colors[event.kind],
            edgecolors="#111827",
            linewidth=0.55,
            alpha=0.9,
        )
        handles.setdefault(event.kind, bar)

    makespan_us = timeline_makespan(events)
    overlap_factor = (serial_total / makespan_us) if makespan_us > 0 else 0.0

    ax.set_ylim(-1.0, float(stream_count))
    ax.set_xlim(0.0, makespan_us * 1.08 if makespan_us > 0 else 1.0)
    ax.set_yticks(list(range(stream_count)))
    ax.set_yticklabels([f"Stream {idx}" for idx in range(stream_count)])
    ax.set_xlabel("Time [microseconds]")
    ax.set_title(
        "CUDA Pipeline Timeline\n"
        f"makespan={makespan_us:,.1f} us | overlap x{overlap_factor:.2f} | "
        f"H2D={h2d_bw_gbps:.0f} GB/s D2H={d2h_bw_gbps:.0f} GB/s "
        f"kernel_scale={kernel_scale:.2f} overlap={overlap:.2f}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.27)

    ordered = ["H2D", "Kernel", "D2H"]
    ax.legend([handles[kind] for kind in ordered], ordered, loc="upper right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=4, help="Number of CUDA streams.")
    parser.add_argument("--chunks", type=int, default=6, help="Work chunks per stream.")
    parser.add_argument("--h2d-mb", type=float, default=64.0, help="H2D bytes per chunk [MiB].")
    parser.add_argument("--d2h-mb", type=float, default=32.0, help="D2H bytes per chunk [MiB].")
    parser.add_argument("--kernel-us", type=float, default=210.0, help="Base kernel runtime [us].")
    parser.add_argument("--h2d-bw", type=float, default=32.0, help="Initial H2D bandwidth [GB/s].")
    parser.add_argument("--d2h-bw", type=float, default=26.0, help="Initial D2H bandwidth [GB/s].")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path for a snapshot.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Render one snapshot without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import matplotlib

        if args.no_gui:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with "
            "`python3 -m pip install matplotlib`."
        ) from exc

    if args.no_gui:
        fig, ax = plt.subplots(figsize=(13.0, 7.3))
        events, serial_total = simulate_timeline(
            stream_count=max(1, args.streams),
            chunks_per_stream=max(1, args.chunks),
            h2d_mb=max(1e-3, args.h2d_mb),
            d2h_mb=max(1e-3, args.d2h_mb),
            h2d_bw_gbps=max(1e-3, args.h2d_bw),
            d2h_bw_gbps=max(1e-3, args.d2h_bw),
            base_kernel_us=max(1.0, args.kernel_us),
            kernel_scale=1.0,
            overlap=0.55,
            launch_gap_us=9.0,
        )
        draw_timeline(
            ax,
            events=events,
            stream_count=max(1, args.streams),
            serial_total=serial_total,
            h2d_bw_gbps=max(1e-3, args.h2d_bw),
            d2h_bw_gbps=max(1e-3, args.d2h_bw),
            kernel_scale=1.0,
            overlap=0.55,
        )
        output_path = args.output or Path("images/cuda_transfer_timeline_demo.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated {output_path}")
        return

    plt.rcParams.update(
        {
            "figure.figsize": (13.0, 7.3),
            "font.size": 10.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)

    slider_h2d_ax = fig.add_axes((0.11, 0.20, 0.76, 0.03))
    slider_d2h_ax = fig.add_axes((0.11, 0.15, 0.76, 0.03))
    slider_kernel_ax = fig.add_axes((0.11, 0.10, 0.76, 0.03))
    slider_overlap_ax = fig.add_axes((0.11, 0.05, 0.76, 0.03))

    slider_h2d = Slider(slider_h2d_ax, "H2D BW [GB/s]", 8.0, 120.0, valinit=max(8.0, args.h2d_bw), valstep=1.0)
    slider_d2h = Slider(slider_d2h_ax, "D2H BW [GB/s]", 8.0, 120.0, valinit=max(8.0, args.d2h_bw), valstep=1.0)
    slider_kernel = Slider(slider_kernel_ax, "Kernel Scale", 0.25, 3.0, valinit=1.0, valstep=0.01)
    slider_overlap = Slider(slider_overlap_ax, "Overlap", 0.0, 0.9, valinit=0.55, valstep=0.01)

    def redraw(_) -> None:
        events, serial_total = simulate_timeline(
            stream_count=max(1, args.streams),
            chunks_per_stream=max(1, args.chunks),
            h2d_mb=max(1e-3, args.h2d_mb),
            d2h_mb=max(1e-3, args.d2h_mb),
            h2d_bw_gbps=max(1e-3, slider_h2d.val),
            d2h_bw_gbps=max(1e-3, slider_d2h.val),
            base_kernel_us=max(1.0, args.kernel_us),
            kernel_scale=max(0.01, slider_kernel.val),
            overlap=max(0.0, min(0.9, slider_overlap.val)),
            launch_gap_us=9.0,
        )
        draw_timeline(
            ax,
            events=events,
            stream_count=max(1, args.streams),
            serial_total=serial_total,
            h2d_bw_gbps=slider_h2d.val,
            d2h_bw_gbps=slider_d2h.val,
            kernel_scale=slider_kernel.val,
            overlap=slider_overlap.val,
        )
        fig.canvas.draw_idle()

    slider_h2d.on_changed(redraw)
    slider_d2h.on_changed(redraw)
    slider_kernel.on_changed(redraw)
    slider_overlap.on_changed(redraw)

    redraw(None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160, bbox_inches="tight")
        print(f"Generated {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
