#!/usr/bin/env python3
"""CUDA performance visualization page (Dash).

Sections:
1) Memory access patterns: coalescing + bank conflicts.
2) Arithmetic intensity: ideal/effective heatmaps, block-size sensitivity, roofline.
3) Transfer load: H2D/kernel/D2H timeline and runtime breakdown.

Run:
    python scripts/cuda_warp_access_playground.py
    python scripts/cuda_warp_access_playground.py --port 8060
    python scripts/cuda_warp_access_playground.py --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


SWEEP_POINTS = [
    (1, 0.333333, 0.151654, 0.442512, 1327.535463),
    (2, 0.666667, 0.151808, 0.884128, 1326.192246),
    (4, 1.333333, 0.151910, 1.767064, 1325.298278),
    (8, 2.666667, 0.152832, 3.512818, 1317.306572),
    (16, 5.333333, 0.164659, 6.520995, 1222.686552),
    (32, 10.666667, 0.222464, 9.653174, 904.985084),
    (64, 21.333333, 0.396749, 10.825407, 507.440954),
    (128, 42.666667, 0.757658, 11.337489, 265.722387),
    (256, 85.333333, 1.479014, 11.615755, 136.122131),
]

BLOCK_OPTIONS = [8, 12, 16, 24, 32, 48, 64]
TRANSFER_FIXED_WINDOW_US = 60_000.0


@dataclass(frozen=True)
class AccessMetrics:
    requested_bytes: int
    transferred_bytes: int
    efficiency_pct: float
    sectors_touched: int
    lines_128b: int
    max_conflict: int
    replay_factor: float


@dataclass(frozen=True)
class Event:
    stream: int
    kind: str
    start_us: float
    duration_us: float

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


# -----------------------------
# Memory-access model
# -----------------------------
def lane_ids(mode: str, active_lanes: int, np):
    base = np.arange(32, dtype=int)
    if mode == "first":
        return base[:active_lanes]
    if mode == "even":
        return base[base % 2 == 0][:active_lanes]
    if mode == "odd":
        return base[base % 2 == 1][:active_lanes]
    if mode == "interleaved4":
        return np.concatenate([base[i::4] for i in range(4)])[:active_lanes]
    return base[:active_lanes]


def global_addresses_bytes(*, lanes, pattern: str, word_size: int, base_offset: int, stride: int, np):
    if pattern == "contiguous":
        element_index = lanes
    elif pattern == "strided":
        element_index = lanes * stride
    elif pattern == "broadcast":
        element_index = np.zeros_like(lanes)
    elif pattern == "gather":
        element_index = ((lanes * 9) % 32) * stride
    elif pattern == "split":
        half = max(1, lanes.shape[0] // 2)
        first = lanes[:half]
        second = lanes[half:]
        element_index = np.concatenate([first, second + (stride * 16)])
    else:
        element_index = lanes
    return base_offset + (element_index * word_size)


def sector_occupancy(addresses, word_size: int) -> dict[int, int]:
    occupancy: dict[int, int] = {}
    for address in addresses:
        first_sector = int(address // 32)
        last_sector = int((address + word_size - 1) // 32)
        for sector in range(first_sector, last_sector + 1):
            occupancy[sector] = occupancy.get(sector, 0) + 1
    return occupancy


def shared_banks(lanes, pattern: str, base_word: int, stride: int, np):
    if pattern == "contiguous":
        word_indices = base_word + lanes
    elif pattern == "strided":
        word_indices = base_word + lanes * stride
    elif pattern == "broadcast":
        word_indices = np.full(lanes.shape, base_word, dtype=int)
    elif pattern == "xor":
        word_indices = base_word + (lanes ^ stride)
    else:
        word_indices = base_word + lanes
    banks = word_indices % 32
    histogram = np.bincount(banks, minlength=32)
    return banks, histogram


def compute_access_metrics(sectors, bank_hist, active_lanes: int, word_size: int, np) -> AccessMetrics:
    requested_bytes = int(active_lanes * word_size)
    transferred_bytes = int(len(sectors) * 32)
    efficiency_pct = 100.0 * requested_bytes / max(1, transferred_bytes)
    lines_128b = len({int(sector // 4) for sector in sectors.tolist()}) if sectors.size else 0
    max_conflict = int(bank_hist.max()) if bank_hist.size else 0
    replay_factor = float(max_conflict)
    return AccessMetrics(
        requested_bytes=requested_bytes,
        transferred_bytes=transferred_bytes,
        efficiency_pct=efficiency_pct,
        sectors_touched=len(sectors),
        lines_128b=lines_128b,
        max_conflict=max_conflict,
        replay_factor=replay_factor,
    )


# -----------------------------
# Arithmetic-intensity model
# -----------------------------
def pad_to(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile


def ai_ideal(m: int, n: int, k: int, elem_bytes: int) -> float:
    flops = 2.0 * m * n * k
    bytes_moved = elem_bytes * (m * k + k * n + 2 * m * n)
    return flops / max(1.0, bytes_moved)


def ai_effective_with_padding(m: int, n: int, k: int, tile: int, elem_bytes: int) -> float:
    mp = pad_to(m, tile)
    np_ = pad_to(n, tile)
    kp = pad_to(k, tile)
    useful_flops = 2.0 * m * n * k
    padded_bytes = elem_bytes * (mp * kp + kp * np_ + 2 * mp * np_)
    return useful_flops / max(1.0, padded_bytes)


def build_dim_values(min_dim: int, max_dim: int, step: int, np):
    min_dim = clamp_int(min_dim, 4, 4096)
    max_dim = clamp_int(max_dim, min_dim, 4096)
    step = clamp_int(step, 1, 1024)
    return np.arange(min_dim, max_dim + 1, step, dtype=int)


def build_ai_surfaces(ms, ns, k: int, tile: int, elem_bytes: int, np):
    ideal = np.zeros((len(ms), len(ns)), dtype=float)
    effective = np.zeros((len(ms), len(ns)), dtype=float)
    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            ideal[i, j] = ai_ideal(int(m), int(n), k, elem_bytes)
            effective[i, j] = ai_effective_with_padding(int(m), int(n), k, tile, elem_bytes)
    ratio = np.divide(effective, np.maximum(ideal, 1e-12))
    return ideal, effective, ratio


def build_block_sensitivity(square_dims, k: int, elem_bytes: int, np):
    z = np.zeros((len(square_dims), len(BLOCK_OPTIONS)), dtype=float)
    for i, dim in enumerate(square_dims):
        base = ai_ideal(int(dim), int(dim), k, elem_bytes)
        for j, tile in enumerate(BLOCK_OPTIONS):
            eff = ai_effective_with_padding(int(dim), int(dim), k, int(tile), elem_bytes)
            z[i, j] = eff / max(base, 1e-12)
    return z


# -----------------------------
# Transfer-load model
# -----------------------------
def transfer_duration_us(megabytes: float, gb_per_second: float) -> float:
    bytes_count = megabytes * 1024.0 * 1024.0
    return (bytes_count / max(1e-9, gb_per_second * 1e9)) * 1e6


def simulate_timeline(
    *,
    stream_count: int,
    chunks_per_stream: int,
    h2d_mb: float,
    d2h_mb: float,
    h2d_bw_gbps: float,
    d2h_bw_gbps: float,
    base_kernel_us: float,
    overlap: float,
    launch_gap_us: float,
):
    h2d_us = transfer_duration_us(h2d_mb, h2d_bw_gbps)
    d2h_us = transfer_duration_us(d2h_mb, d2h_bw_gbps)
    kernel_us = max(1.0, base_kernel_us)

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
            events.append(Event(stream=stream, kind="Kernel", start_us=kernel_start, duration_us=kernel_us))
            events.append(Event(stream=stream, kind="D2H", start_us=d2h_start, duration_us=d2h_us))

    return events, serial_total


def timeline_makespan(events: list[Event]) -> float:
    if not events:
        return 0.0
    start = min(event.start_us for event in events)
    end = max(event.end_us for event in events)
    return max(0.0, end - start)


def clip_duration_to_window(event: Event, window_us: float) -> float:
    if event.start_us >= window_us:
        return 0.0
    clipped_end = min(event.end_us, window_us)
    return max(0.0, clipped_end - event.start_us)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for Dash.")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash server.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    parser.add_argument("--check", action="store_true", help="Run model checks and exit.")
    return parser.parse_args()


# -----------------------------
# Dash app
# -----------------------------
def main() -> None:
    args = parse_args()

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: numpy. Install with `python3 -m pip install numpy`.") from exc

    if args.check:
        lanes = lane_ids("first", 32, np)
        addresses = global_addresses_bytes(
            lanes=lanes,
            pattern="contiguous",
            word_size=4,
            base_offset=0,
            stride=1,
            np=np,
        )
        sectors = np.array(sorted(sector_occupancy(addresses, word_size=4)), dtype=int)
        _, bank_hist = shared_banks(lanes, "contiguous", 0, 1, np)
        access_metrics = compute_access_metrics(sectors, bank_hist, 32, 4, np)

        ms = build_dim_values(16, 64, 8, np)
        ideal, effective, ratio = build_ai_surfaces(ms, ms, 64, 16, 4, np)

        events, serial_total = simulate_timeline(
            stream_count=4,
            chunks_per_stream=6,
            h2d_mb=64.0,
            d2h_mb=32.0,
            h2d_bw_gbps=32.0,
            d2h_bw_gbps=26.0,
            base_kernel_us=210.0,
            overlap=0.55,
            launch_gap_us=9.0,
        )
        makespan = timeline_makespan(events)

        print("Dashboard model check passed")
        print(f"access_efficiency={access_metrics.efficiency_pct:.1f}%")
        print(f"ai_mean_ratio={float(ratio.mean()):.3f}")
        print(f"timeline_overlap_factor={serial_total / max(1e-9, makespan):.2f}")
        return

    try:
        from dash import Dash, Input, Output, dcc, html
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: dash/plotly. Install with `python3 -m pip install dash plotly`."
        ) from exc

    app = Dash(__name__)
    app.title = "CUDA Visualisation Page"

    page_style = {
        "minHeight": "100vh",
        "padding": "18px 20px 24px 20px",
        "background": "linear-gradient(180deg, #EDF8FF 0%, #F9F5E8 44%, #EEF9F3 100%)",
        "fontFamily": "'IBM Plex Sans', 'Avenir Next', 'Segoe UI', sans-serif",
        "color": "#0B1324",
    }
    panel_style = {
        "background": "#FFFFFF",
        "border": "1px solid #E5E7EB",
        "borderRadius": "12px",
        "padding": "14px",
        "boxShadow": "0 2px 10px rgba(15, 23, 42, 0.04)",
    }
    card_style = {
        "background": "#FFFFFF",
        "border": "1px solid #E5E7EB",
        "borderRadius": "10px",
        "padding": "10px 12px",
    }
    metric_value = {"fontSize": "22px", "fontWeight": "700", "lineHeight": "1.1"}
    metric_label = {"fontSize": "12px", "letterSpacing": "0.03em", "color": "#4B5563"}
    control_title = {"fontWeight": 700, "fontSize": "13px", "marginBottom": "8px", "color": "#10223E"}

    def metric(id_: str, title: str):
        return html.Div(
            [html.Div(title, style=metric_label), html.Div(id=id_, style=metric_value)],
            style=card_style,
        )

    app.layout = html.Div(
        style=page_style,
        children=[
            html.Div(
                [
                    html.H1(
                        "CUDA Visualisation Page",
                        style={"margin": "0 0 6px 0", "fontSize": "31px", "fontWeight": 800},
                    ),
                    html.Div(
                        "Memory access patterns, arithmetic intensity, transfer load, and block-size heatmaps in one interactive dashboard.",
                        style={"fontSize": "14px", "color": "#334155"},
                    ),
                ],
                style={
                    "padding": "14px 16px",
                    "borderRadius": "13px",
                    "marginBottom": "14px",
                    "background": "linear-gradient(90deg, #D8F3FF 0%, #F6F0C6 52%, #DAF7E9 100%)",
                    "border": "1px solid #E7E2C9",
                },
            ),
            dcc.Tabs(
                id="main-tabs",
                value="tab-access",
                children=[
                    dcc.Tab(
                        label="Memory Access Patterns",
                        value="tab-access",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Global Access", style=control_title),
                                            html.Label("Pattern", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Dropdown(
                                                id="access-global-pattern",
                                                options=[
                                                    {"label": "Contiguous", "value": "contiguous"},
                                                    {"label": "Strided", "value": "strided"},
                                                    {"label": "Broadcast", "value": "broadcast"},
                                                    {"label": "Gather (permute)", "value": "gather"},
                                                    {"label": "Split warp", "value": "split"},
                                                ],
                                                value="contiguous",
                                                clearable=False,
                                                style={"marginBottom": "8px"},
                                            ),
                                            html.Label("Lane activation", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Dropdown(
                                                id="access-lane-mode",
                                                options=[
                                                    {"label": "First N lanes", "value": "first"},
                                                    {"label": "Even lanes first", "value": "even"},
                                                    {"label": "Odd lanes first", "value": "odd"},
                                                    {"label": "4-way interleaved", "value": "interleaved4"},
                                                ],
                                                value="first",
                                                clearable=False,
                                                style={"marginBottom": "8px"},
                                            ),
                                            html.Label("Word size", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.RadioItems(
                                                id="access-word-size",
                                                options=[
                                                    {"label": "4B", "value": 4},
                                                    {"label": "8B", "value": 8},
                                                    {"label": "16B", "value": 16},
                                                ],
                                                value=4,
                                                inline=True,
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                            html.Div(style={"height": "10px"}),
                                            html.Label("Active lanes", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="access-active-lanes",
                                                min=1,
                                                max=32,
                                                step=1,
                                                value=32,
                                                marks={1: "1", 8: "8", 16: "16", 24: "24", 32: "32"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Global stride (elements)", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="access-global-stride",
                                                min=1,
                                                max=32,
                                                step=1,
                                                value=1,
                                                marks={1: "1", 8: "8", 16: "16", 24: "24", 32: "32"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Global base offset (bytes)", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="access-global-offset",
                                                min=0,
                                                max=255,
                                                step=1,
                                                value=0,
                                                marks={0: "0", 64: "64", 128: "128", 192: "192", 255: "255"},
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Shared-Memory Access", style=control_title),
                                            html.Label("Pattern", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Dropdown(
                                                id="access-shared-pattern",
                                                options=[
                                                    {"label": "Contiguous", "value": "contiguous"},
                                                    {"label": "Strided", "value": "strided"},
                                                    {"label": "Broadcast", "value": "broadcast"},
                                                    {"label": "XOR remap", "value": "xor"},
                                                ],
                                                value="contiguous",
                                                clearable=False,
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Label("Shared stride (words)", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="access-shared-stride",
                                                min=1,
                                                max=32,
                                                step=1,
                                                value=1,
                                                marks={1: "1", 8: "8", 16: "16", 24: "24", 32: "32"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Shared base word", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="access-shared-base",
                                                min=0,
                                                max=31,
                                                step=1,
                                                value=0,
                                                marks={0: "0", 8: "8", 16: "16", 24: "24", 31: "31"},
                                            ),
                                            html.Div(style={"height": "16px"}),
                                            html.Div(
                                                [
                                                    html.Div("Notes", style={"fontWeight": 700, "marginBottom": "8px"}),
                                                    html.Ul(
                                                        [
                                                            html.Li("Coalescing is modeled with 32-byte sectors."),
                                                            html.Li("128-byte lines are grouped as 4 adjacent sectors."),
                                                            html.Li("Bank conflict replay estimate uses max bank occupancy."),
                                                        ],
                                                        style={"marginTop": "0", "paddingLeft": "18px", "fontSize": "12px", "color": "#334155"},
                                                    ),
                                                ],
                                                style={
                                                    "background": "#F8FAFC",
                                                    "border": "1px solid #E2E8F0",
                                                    "borderRadius": "10px",
                                                    "padding": "10px",
                                                },
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(330px, 1fr))",
                                    "gap": "12px",
                                    "padding": "14px 0",
                                },
                            ),
                            html.Div(
                                [
                                    metric("access-m-eff", "Global Load Efficiency"),
                                    metric("access-m-xfer", "Transferred Bytes"),
                                    metric("access-m-sectors", "Sectors Touched"),
                                    metric("access-m-lines", "128B Lines"),
                                    metric("access-m-conflict", "Max Bank Conflict"),
                                    metric("access-m-replay", "Replay Estimate"),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                                    "gap": "10px",
                                    "marginBottom": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="access-fig", style={"height": "350px"}),
                                    dcc.Graph(id="access-sector-fig", style={"height": "350px"}),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))",
                                    "gap": "12px",
                                },
                            ),
                            dcc.Graph(id="access-bank-fig", style={"height": "330px", "marginTop": "10px"}),
                        ],
                    ),
                    dcc.Tab(
                        label="Arithmetic Intensity",
                        value="tab-ai",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Heatmap Controls", style=control_title),
                                            html.Label("Precision", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.RadioItems(
                                                id="ai-precision",
                                                options=[
                                                    {"label": "FP32", "value": "fp32"},
                                                    {"label": "FP64", "value": "fp64"},
                                                ],
                                                value="fp32",
                                                inline=True,
                                                labelStyle={"marginRight": "14px"},
                                            ),
                                            html.Div(style={"height": "10px"}),
                                            html.Label("Fixed K dimension", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-k",
                                                min=16,
                                                max=256,
                                                step=8,
                                                value=64,
                                                marks={16: "16", 64: "64", 128: "128", 192: "192", 256: "256"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Min dimension", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-min-dim",
                                                min=8,
                                                max=256,
                                                step=8,
                                                value=16,
                                                marks={8: "8", 64: "64", 128: "128", 192: "192", 256: "256"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Max dimension", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-max-dim",
                                                min=32,
                                                max=512,
                                                step=16,
                                                value=256,
                                                marks={32: "32", 128: "128", 256: "256", 384: "384", 512: "512"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Dimension step", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-step",
                                                min=4,
                                                max=32,
                                                step=4,
                                                value=8,
                                                marks={4: "4", 8: "8", 16: "16", 24: "24", 32: "32"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Block/tile size", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-tile",
                                                min=8,
                                                max=64,
                                                step=4,
                                                value=16,
                                                marks={8: "8", 16: "16", 32: "32", 48: "48", 64: "64"},
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Roofline Controls", style=control_title),
                                            html.Label("Peak compute [TFLOP/s]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-peak-tflops",
                                                min=4.0,
                                                max=30.0,
                                                step=0.1,
                                                value=11.6,
                                                marks={4: "4", 8: "8", 12: "12", 16: "16", 24: "24", 30: "30"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Peak bandwidth [GB/s]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="ai-peak-bw",
                                                min=300.0,
                                                max=2000.0,
                                                step=5.0,
                                                value=1327.0,
                                                marks={300: "300", 800: "800", 1200: "1200", 1600: "1600", 2000: "2000"},
                                            ),
                                            html.Div(style={"height": "16px"}),
                                            html.Div(
                                                [
                                                    html.Div("Integrated Ideas", style={"fontWeight": 700, "marginBottom": "8px"}),
                                                    html.Ul(
                                                        [
                                                            html.Li("Ideal vs padded AI directly mirrors your heatmap script."),
                                                            html.Li("Block-size sensitivity map compares multiple tile sizes."),
                                                            html.Li("Roofline uses your measured intensity sweep points."),
                                                        ],
                                                        style={"marginTop": "0", "paddingLeft": "18px", "fontSize": "12px", "color": "#334155"},
                                                    ),
                                                ],
                                                style={
                                                    "background": "#F8FAFC",
                                                    "border": "1px solid #E2E8F0",
                                                    "borderRadius": "10px",
                                                    "padding": "10px",
                                                },
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(330px, 1fr))",
                                    "gap": "12px",
                                    "padding": "14px 0",
                                },
                            ),
                            html.Div(
                                [
                                    metric("ai-m-mean", "Mean Effective AI"),
                                    metric("ai-m-retention", "Mean Retention"),
                                    metric("ai-m-knee", "Roofline Knee (AI)"),
                                    metric("ai-m-bound", "Sweep Regime"),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                                    "gap": "10px",
                                    "marginBottom": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="ai-ideal-fig", style={"height": "360px"}),
                                    dcc.Graph(id="ai-effective-fig", style={"height": "360px"}),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))",
                                    "gap": "12px",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="ai-block-fig", style={"height": "360px"}),
                                    dcc.Graph(id="ai-roofline-fig", style={"height": "360px"}),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(350px, 1fr))",
                                    "gap": "12px",
                                    "marginTop": "10px",
                                },
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Memory Transfer Load",
                        value="tab-transfer",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Transfer Timeline Controls", style=control_title),
                                            html.Label("Streams", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-streams",
                                                min=1,
                                                max=8,
                                                step=1,
                                                value=4,
                                                marks={1: "1", 2: "2", 4: "4", 6: "6", 8: "8"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Chunks per stream", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-chunks",
                                                min=1,
                                                max=16,
                                                step=1,
                                                value=6,
                                                marks={1: "1", 4: "4", 8: "8", 12: "12", 16: "16"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("H2D size per chunk [MiB]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-h2d-mb",
                                                min=4,
                                                max=256,
                                                step=4,
                                                value=64,
                                                marks={4: "4", 64: "64", 128: "128", 192: "192", 256: "256"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("D2H size per chunk [MiB]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-d2h-mb",
                                                min=4,
                                                max=256,
                                                step=4,
                                                value=32,
                                                marks={4: "4", 64: "64", 128: "128", 192: "192", 256: "256"},
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Runtime Controls", style=control_title),
                                            html.Label("Kernel runtime [us]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-kernel-us",
                                                min=20,
                                                max=2000,
                                                step=10,
                                                value=210,
                                                marks={20: "20", 200: "200", 600: "600", 1200: "1200", 2000: "2000"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("H2D bandwidth [GB/s]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-h2d-bw",
                                                min=8,
                                                max=120,
                                                step=1,
                                                value=32,
                                                marks={8: "8", 32: "32", 64: "64", 96: "96", 120: "120"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("D2H bandwidth [GB/s]", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-d2h-bw",
                                                min=8,
                                                max=120,
                                                step=1,
                                                value=26,
                                                marks={8: "8", 32: "32", 64: "64", 96: "96", 120: "120"},
                                            ),
                                            html.Div(style={"height": "12px"}),
                                            html.Label("Overlap factor", style={"fontSize": "12px", "color": "#475569"}),
                                            dcc.Slider(
                                                id="xfer-overlap",
                                                min=0.0,
                                                max=0.9,
                                                step=0.01,
                                                value=0.55,
                                                marks={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 0.9: "0.9"},
                                            ),
                                        ],
                                        style=panel_style,
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(330px, 1fr))",
                                    "gap": "12px",
                                    "padding": "14px 0",
                                },
                            ),
                            html.Div(
                                [
                                    metric("xfer-m-makespan", "Fixed Time Window"),
                                    metric("xfer-m-overlap", "Overlap Speedup"),
                                    metric("xfer-m-bytes", "Transferred Data"),
                                    metric("xfer-m-effbw", "Effective Link Throughput"),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                                    "gap": "10px",
                                    "marginBottom": "10px",
                                },
                            ),
                            dcc.Graph(id="xfer-timeline-fig", style={"height": "430px"}),
                            dcc.Graph(id="xfer-breakdown-fig", style={"height": "300px", "marginTop": "6px"}),
                        ],
                    ),
                ],
            ),
        ],
    )

    def apply_base_layout(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title={"text": title, "x": 0.02},
            margin={"l": 52, "r": 18, "t": 42, "b": 44},
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.96)",
            plot_bgcolor="#FCFCFD",
            legend={"orientation": "h", "y": 1.12, "x": 0.0},
        )
        fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False)
        fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
        return fig

    @app.callback(
        [
            Output("access-fig", "figure"),
            Output("access-sector-fig", "figure"),
            Output("access-bank-fig", "figure"),
            Output("access-m-eff", "children"),
            Output("access-m-xfer", "children"),
            Output("access-m-sectors", "children"),
            Output("access-m-lines", "children"),
            Output("access-m-conflict", "children"),
            Output("access-m-replay", "children"),
        ],
        [
            Input("access-global-pattern", "value"),
            Input("access-lane-mode", "value"),
            Input("access-word-size", "value"),
            Input("access-active-lanes", "value"),
            Input("access-global-stride", "value"),
            Input("access-global-offset", "value"),
            Input("access-shared-pattern", "value"),
            Input("access-shared-stride", "value"),
            Input("access-shared-base", "value"),
        ],
    )
    def update_access(
        global_pattern: str,
        lane_mode: str,
        word_size: int,
        active_lanes: int,
        global_stride: int,
        global_offset: int,
        shared_pattern: str,
        shared_stride: int,
        shared_base: int,
    ):
        active_lanes = clamp_int(active_lanes, 1, 32)
        word_size = clamp_int(word_size, 4, 16)
        global_stride = clamp_int(global_stride, 1, 32)
        global_offset = clamp_int(global_offset, 0, 255)
        shared_stride = clamp_int(shared_stride, 1, 32)
        shared_base = clamp_int(shared_base, 0, 31)

        lanes = lane_ids(lane_mode, active_lanes, np)
        addresses = global_addresses_bytes(
            lanes=lanes,
            pattern=global_pattern,
            word_size=word_size,
            base_offset=global_offset,
            stride=global_stride,
            np=np,
        )
        sector_map = sector_occupancy(addresses, word_size)
        sectors = np.array(sorted(sector_map), dtype=int)
        sector_counts = np.array([sector_map[s] for s in sectors], dtype=int) if sectors.size else np.array([], dtype=int)

        banks, bank_hist = shared_banks(
            lanes=lanes,
            pattern=shared_pattern,
            base_word=shared_base,
            stride=shared_stride,
            np=np,
        )
        metrics = compute_access_metrics(sectors, bank_hist, active_lanes, word_size, np)

        line_ids = addresses // 128

        access_fig = go.Figure()
        access_fig.add_trace(
            go.Scatter(
                x=lanes.tolist(),
                y=addresses.tolist(),
                mode="markers+lines",
                marker={
                    "size": 11,
                    "color": line_ids.tolist(),
                    "colorscale": "Turbo",
                    "line": {"width": 1, "color": "#0F172A"},
                    "showscale": True,
                    "colorbar": {"title": "128B line"},
                },
                line={"width": 2, "color": "#94A3B8"},
                customdata=np.stack([line_ids, banks], axis=1),
                hovertemplate=(
                    "Lane %{x}<br>"
                    "Address %{y} B<br>"
                    "128B line %{customdata[0]}<br>"
                    "Shared bank %{customdata[1]}<extra></extra>"
                ),
                name="Lane accesses",
            )
        )
        access_fig.update_xaxes(title_text="Lane ID", dtick=1)
        access_fig.update_yaxes(title_text="Global address [bytes]")
        access_fig = apply_base_layout(access_fig, "Warp Global Access Map")

        sector_fig = go.Figure()
        if sectors.size:
            sector_fig.add_trace(
                go.Bar(
                    x=(sectors * 32).tolist(),
                    y=sector_counts.tolist(),
                    marker={"color": "#F59E0B", "line": {"width": 1, "color": "#78350F"}},
                    hovertemplate="Sector @ %{x} B<br>Lanes touching %{y}<extra></extra>",
                    name="Sector occupancy",
                )
            )
        sector_fig.update_xaxes(title_text="32-byte sector base address [bytes]")
        sector_fig.update_yaxes(title_text="Lanes touching sector")
        sector_fig = apply_base_layout(sector_fig, "Coalescing Sector Occupancy")

        bank_fig = go.Figure()
        bank_fig.add_trace(
            go.Bar(
                x=list(range(32)),
                y=bank_hist.tolist(),
                marker={"color": "#0EA5A4", "line": {"width": 1, "color": "#134E4A"}},
                hovertemplate="Bank %{x}<br>Lanes %{y}<extra></extra>",
                name="Bank load",
            )
        )
        bank_fig.update_xaxes(title_text="Shared-memory bank", dtick=1)
        bank_fig.update_yaxes(title_text="Lanes mapped to bank")
        bank_fig = apply_base_layout(bank_fig, "Shared-Memory Bank Pressure")

        return (
            access_fig,
            sector_fig,
            bank_fig,
            f"{metrics.efficiency_pct:.1f}%",
            f"{metrics.transferred_bytes} B",
            str(metrics.sectors_touched),
            str(metrics.lines_128b),
            f"{metrics.max_conflict}x",
            f"{metrics.replay_factor:.1f}x",
        )

    @app.callback(
        [
            Output("ai-ideal-fig", "figure"),
            Output("ai-effective-fig", "figure"),
            Output("ai-block-fig", "figure"),
            Output("ai-roofline-fig", "figure"),
            Output("ai-m-mean", "children"),
            Output("ai-m-retention", "children"),
            Output("ai-m-knee", "children"),
            Output("ai-m-bound", "children"),
        ],
        [
            Input("ai-precision", "value"),
            Input("ai-k", "value"),
            Input("ai-min-dim", "value"),
            Input("ai-max-dim", "value"),
            Input("ai-step", "value"),
            Input("ai-tile", "value"),
            Input("ai-peak-tflops", "value"),
            Input("ai-peak-bw", "value"),
        ],
    )
    def update_ai(
        precision: str,
        k: int,
        min_dim: int,
        max_dim: int,
        step: int,
        tile: int,
        peak_tflops: float,
        peak_bw_gbps: float,
    ):
        elem_bytes = 8 if precision == "fp64" else 4
        k = clamp_int(k, 4, 1024)
        tile = clamp_int(tile, 4, 256)
        min_dim = clamp_int(min_dim, 4, 4096)
        max_dim = clamp_int(max_dim, min_dim, 4096)
        step = clamp_int(step, 1, 1024)

        ms = build_dim_values(min_dim, max_dim, step, np)
        ns = ms
        ideal, effective, ratio = build_ai_surfaces(ms, ns, k, tile, elem_bytes, np)

        ideal_fig = go.Figure(
            data=go.Heatmap(
                z=ideal,
                x=ns.tolist(),
                y=ms.tolist(),
                colorscale="Magma",
                colorbar={"title": "FLOP/byte"},
                hovertemplate="M=%{y}<br>N=%{x}<br>Ideal AI=%{z:.3f}<extra></extra>",
            )
        )
        ideal_fig.update_xaxes(title_text="N dimension")
        ideal_fig.update_yaxes(title_text="M dimension")
        ideal_fig = apply_base_layout(ideal_fig, f"Ideal AI (K={k}, {'FP64' if elem_bytes == 8 else 'FP32'})")

        effective_fig = go.Figure(
            data=go.Heatmap(
                z=effective,
                x=ns.tolist(),
                y=ms.tolist(),
                colorscale="Magma",
                colorbar={"title": "FLOP/byte"},
                hovertemplate=(
                    "M=%{y}<br>N=%{x}<br>"
                    f"Effective AI (tile={tile})=%{{z:.3f}}<extra></extra>"
                ),
            )
        )
        effective_fig.update_xaxes(title_text="N dimension")
        effective_fig.update_yaxes(title_text="M dimension")
        effective_fig = apply_base_layout(effective_fig, f"Effective AI With Padding (tile={tile})")

        block_matrix = build_block_sensitivity(ms, k, elem_bytes, np)
        block_fig = go.Figure(
            data=go.Heatmap(
                z=block_matrix,
                x=BLOCK_OPTIONS,
                y=ms.tolist(),
                zmin=0.0,
                zmax=1.0,
                colorscale="Viridis",
                colorbar={"title": "Retention"},
                hovertemplate=(
                    "M=N=%{y}<br>"
                    "tile=%{x}<br>"
                    "AI retention=%{z:.3f}<extra></extra>"
                ),
            )
        )
        block_fig.update_xaxes(title_text="Block/tile size")
        block_fig.update_yaxes(title_text="Square size M=N")
        block_fig = apply_base_layout(block_fig, "Block-Size Sensitivity (effective/ideal AI)")

        ai_values = np.array([row[1] for row in SWEEP_POINTS], dtype=float)
        avg_ms_values = np.array([row[2] for row in SWEEP_POINTS], dtype=float)
        tflops_values = np.array([row[3] for row in SWEEP_POINTS], dtype=float)
        bw_values = np.array([row[4] for row in SWEEP_POINTS], dtype=float)
        iter_values = np.array([row[0] for row in SWEEP_POINTS], dtype=int)

        peak_tflops = clamp_float(peak_tflops, 0.1, 1_000.0)
        peak_bw_tbps = clamp_float(peak_bw_gbps, 0.1, 10_000.0) / 1000.0
        knee_ai = peak_tflops / max(1e-9, peak_bw_tbps)

        ai_axis = np.logspace(np.log10(max(0.15, float(ai_values.min()) / 2.0)), np.log10(float(ai_values.max()) * 1.4), 180)
        roofline_curve = np.minimum(peak_tflops, peak_bw_tbps * ai_axis)
        bound_labels = np.where(ai_values < knee_ai, "memory-bound", "compute-bound")
        memory_count = int(np.count_nonzero(ai_values < knee_ai))

        roofline_fig = go.Figure()
        roofline_fig.add_trace(
            go.Scatter(
                x=ai_axis.tolist(),
                y=roofline_curve.tolist(),
                mode="lines",
                line={"width": 3, "color": "#334155"},
                name="Roofline",
                hovertemplate="AI %{x:.3f}<br>Roofline %{y:.3f} TFLOP/s<extra></extra>",
            )
        )
        roofline_fig.add_trace(
            go.Scatter(
                x=ai_values.tolist(),
                y=tflops_values.tolist(),
                mode="markers+lines+text",
                text=[str(x) for x in iter_values.tolist()],
                textposition="top center",
                marker={
                    "size": 12,
                    "color": bw_values.tolist(),
                    "colorscale": "Turbo",
                    "line": {"width": 1, "color": "#0F172A"},
                    "showscale": True,
                    "colorbar": {"title": "GB/s"},
                },
                customdata=np.stack([avg_ms_values, bw_values, bound_labels], axis=1),
                hovertemplate=(
                    "compute_iters=%{text}<br>"
                    "AI=%{x:.3f}<br>"
                    "TFLOP/s=%{y:.3f}<br>"
                    "avg_ms=%{customdata[0]:.4f}<br>"
                    "BW=%{customdata[1]:.1f} GB/s<br>"
                    "regime=%{customdata[2]}<extra></extra>"
                ),
                name="Measured sweep",
            )
        )
        roofline_fig.add_vline(
            x=knee_ai,
            line_width=2,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"knee={knee_ai:.2f}",
            annotation_position="top",
        )
        roofline_fig.update_xaxes(type="log", title_text="Arithmetic intensity [FLOP/byte]")
        roofline_fig.update_yaxes(title_text="Effective performance [TFLOP/s]")
        roofline_fig = apply_base_layout(roofline_fig, "Sweep vs Roofline")

        mean_effective_ai = float(effective.mean()) if effective.size else 0.0
        mean_retention = float(ratio.mean()) if ratio.size else 0.0
        regime_text = f"{memory_count}/{ai_values.size} memory-bound"

        return (
            ideal_fig,
            effective_fig,
            block_fig,
            roofline_fig,
            f"{mean_effective_ai:.3f}",
            f"{100.0 * mean_retention:.1f}%",
            f"{knee_ai:.2f}",
            regime_text,
        )

    @app.callback(
        [
            Output("xfer-timeline-fig", "figure"),
            Output("xfer-breakdown-fig", "figure"),
            Output("xfer-m-makespan", "children"),
            Output("xfer-m-overlap", "children"),
            Output("xfer-m-bytes", "children"),
            Output("xfer-m-effbw", "children"),
        ],
        [
            Input("xfer-streams", "value"),
            Input("xfer-chunks", "value"),
            Input("xfer-h2d-mb", "value"),
            Input("xfer-d2h-mb", "value"),
            Input("xfer-kernel-us", "value"),
            Input("xfer-h2d-bw", "value"),
            Input("xfer-d2h-bw", "value"),
            Input("xfer-overlap", "value"),
        ],
    )
    def update_transfer(
        streams: int,
        chunks: int,
        h2d_mb: float,
        d2h_mb: float,
        kernel_us: float,
        h2d_bw: float,
        d2h_bw: float,
        overlap: float,
    ):
        streams = clamp_int(streams, 1, 32)
        chunks = clamp_int(chunks, 1, 100)
        h2d_mb = clamp_float(h2d_mb, 0.1, 10000.0)
        d2h_mb = clamp_float(d2h_mb, 0.1, 10000.0)
        kernel_us = clamp_float(kernel_us, 1.0, 10_000_000.0)
        h2d_bw = clamp_float(h2d_bw, 0.1, 10_000.0)
        d2h_bw = clamp_float(d2h_bw, 0.1, 10_000.0)
        overlap = clamp_float(overlap, 0.0, 0.95)

        events, serial_total = simulate_timeline(
            stream_count=streams,
            chunks_per_stream=chunks,
            h2d_mb=h2d_mb,
            d2h_mb=d2h_mb,
            h2d_bw_gbps=h2d_bw,
            d2h_bw_gbps=d2h_bw,
            base_kernel_us=kernel_us,
            overlap=overlap,
            launch_gap_us=9.0,
        )
        makespan = timeline_makespan(events)
        overlap_speedup = serial_total / max(1e-9, makespan)
        window_us = TRANSFER_FIXED_WINDOW_US

        colors = {"H2D": "#2563EB", "Kernel": "#DC2626", "D2H": "#059669"}
        clipped = {"H2D": [], "Kernel": [], "D2H": []}
        clipped_stage_totals = {"H2D": 0.0, "Kernel": 0.0, "D2H": 0.0}
        mib = 1024.0 * 1024.0
        h2d_bytes_per_event = h2d_mb * mib
        d2h_bytes_per_event = d2h_mb * mib
        visible_transfer_bytes = 0.0

        for event in events:
            visible_duration = clip_duration_to_window(event, window_us)
            if visible_duration <= 0.0:
                continue
            clipped[event.kind].append((event.stream, event.start_us, visible_duration, event.duration_us))
            clipped_stage_totals[event.kind] += visible_duration
            visible_fraction = visible_duration / max(1e-9, event.duration_us)
            if event.kind == "H2D":
                visible_transfer_bytes += h2d_bytes_per_event * visible_fraction
            elif event.kind == "D2H":
                visible_transfer_bytes += d2h_bytes_per_event * visible_fraction

        timeline_fig = go.Figure()
        for kind in ("H2D", "Kernel", "D2H"):
            kind_events = clipped[kind]
            timeline_fig.add_trace(
                go.Bar(
                    y=[f"Stream {event[0]}" for event in kind_events],
                    x=[event[2] for event in kind_events],
                    base=[event[1] for event in kind_events],
                    orientation="h",
                    marker={"color": colors[kind], "line": {"width": 0.7, "color": "#0F172A"}},
                    name=kind,
                    customdata=[[event[3]] for event in kind_events],
                    hovertemplate=(
                        "Stream %{y}<br>"
                        "Start %{base:.1f} us<br>"
                        "Visible duration %{x:.1f} us<br>"
                        "Original duration %{customdata[0]:.1f} us<extra></extra>"
                    ),
                )
            )
        timeline_fig.update_layout(barmode="overlay")
        timeline_fig.update_xaxes(
            title_text="Time [microseconds]",
            range=[0.0, window_us],
        )
        timeline_fig.update_yaxes(
            title_text="Stream",
            categoryorder="array",
            categoryarray=[f"Stream {i}" for i in reversed(range(streams))],
        )
        timeline_fig = apply_base_layout(
            timeline_fig,
            (
                "Transfer Timeline "
                f"(clipped to fixed window={window_us:,.0f} us, "
                f"actual makespan={makespan:,.1f} us, overlap x{overlap_speedup:.2f})"
            ),
        )

        breakdown_fig = go.Figure()
        breakdown_fig.add_trace(
            go.Bar(
                x=list(clipped_stage_totals.keys()),
                y=[clipped_stage_totals[k] for k in clipped_stage_totals],
                marker={
                    "color": [colors["H2D"], colors["Kernel"], colors["D2H"]],
                    "line": {"width": 1, "color": "#0F172A"},
                },
                hovertemplate="%{x}<br>Visible busy time %{y:.1f} us<extra></extra>",
            )
        )
        breakdown_fig.update_xaxes(title_text="Pipeline stage")
        breakdown_fig.update_yaxes(title_text="Visible stage time in fixed window [us]")
        breakdown_fig = apply_base_layout(breakdown_fig, "Per-Stage Busy-Time (Window-Clipped)")

        total_gb = visible_transfer_bytes / 1e9
        effective_bw = visible_transfer_bytes / max(1e-9, window_us * 1e-6) / 1e9

        return (
            timeline_fig,
            breakdown_fig,
            f"{window_us:,.0f} us",
            f"x{overlap_speedup:.2f}",
            f"{total_gb:.3f} GB",
            f"{effective_bw:.2f} GB/s",
        )

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
