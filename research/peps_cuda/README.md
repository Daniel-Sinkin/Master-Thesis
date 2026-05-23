# PEPS CUDA Research Bundle

This directory collects sources and working notes for the PEPS CUDA thesis
implementation in `code/peps_cuda/`.

## Start Here

- `research_log.md`: chronological work log and timer record.
- `annotated_sources.md`: source-by-source notes with viability/usefulness.
- `source_inventory.md`: local PDFs/archives and external links.
- `reference_algorithm_anchors.md`: compact algorithm facts from the paper,
  local PDF, Julia repo, and Wu/Nys small-o trick.

## Implementation Design

- `cuda_design.md`: bridge from the Julia/paper pipeline to C++/CUDA.
- `implementation_map.md`: Julia reference stage to current scaffold mapping.
- `program_lifetime_trace.md`: stage-by-stage object lifetime and memory
  pressure map for the Julia reference loop.
- `reference_alignment_plan.md`: Julia fixture/regression strategy and blockers.
- `julia_fixture_axis_mapping.md`: theta-order/link-label conventions needed to
  import Julia fixtures into C++.
- `julia_code_review.md`: reference implementation critique and ROI map.
- `julia_cpu_profile_report.md`: local CPU profiling results for the Julia
  reference examples and synthetic fixtures.
- `testing_infrastructure.md`: invariant/regression test plan.
- `boundary_mps_lowering.md`: index formulas and GEMM views for row absorption.
- `direct_gram_accumulation.md`: sampled-sector/direct minSR Gram plan that
  avoids persistent dense or compact `O` at thesis scale.
- `memory_hierarchy_notes.md`: HBM/L2/shared/register placement guidance.
- `multi_gpu_strategy.md`: first JUPITER/GH200 decomposition plan.
- `next_implementation_steps.md`: prioritized continuation plan after the
  scaffold.

## Profiling And Hardware

- `hardware_notes.md`: A100/H100/H200/JUPITER GH200 facts.
- `performance_plan.md`: staged correctness and profiling plan.
- `performance_targets_and_size_constraints.md`: outcome ladder and input-size
  memory limits.
- `benchmark_triage_snapshot.md`: GH200 one-GPU dense/compact/direct-Gram
  transition points for the first benchmark matrix.
- `precision_decision_matrix.md`: FP64/FP32/TF32/lower-precision measurement
  policy and thesis argument.
- `profiling_kpis.md`: Nsight metrics, NVTX names, roofline/CSV schema.
- `cluster_first_run_checklist.md`: first A100/GH200 run checklist.
- `cublas_grouped_gemm_plan.md`: library benchmark plan for boundary buckets.
- `ozaki_precision_notes.md`: Ozaki/Ozaki-II FP64 emulation assessment.
- `tensor_network_library_survey.md`: reusable CUDA/TN backend survey.

## Bibliography

- `references.bib`: working BibTeX entries for the core algorithm, hardware, and
  tensor-network GPU sources.
- `sources/`: local PDF/archive/source bundle. This includes the main arXiv
  paper, the local PEPS structure PDF, the Julia repo snapshot, NVIDIA docs, and
  selected tensor-network GPU papers.
