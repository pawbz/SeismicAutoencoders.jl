# After-Training APC CLI Runbook

This runbook documents the five acausal+causal post-training CLIs copied into this folder. They are intended to be run after VQ-VAE training has produced `source_state_averages.jld2` artifacts, and they cover transfer, candidate ranking, phase-correction sweeps, final all-triplet evaluation, and export for tomography.

## Prerequisites

Run commands from the repository root:

```bash
cd /home/sanket/Desktop/SeismicAutoencoders.jl
julia --project=. --startup-file=no <cli>.jl ...
```

Most commands need:

- `--saved-root DIR`: root containing trained or transferred source-state artifacts.
- `--raw-data-dir DIR`: raw pair JLD2 directory, with files named like `SN43_SN57-*.jld2`.
- `--triplets-csv FILE`: unfiltered station triplet CSV, usually `station_triplet_csvs/all_station_triplets_unfiltered.csv`.
- `--output-dir DIR`: destination for generated CSVs, plots, artifacts, and metadata.
- `--config FILE`: optional TOML defaults for MFT and triplet-analysis options.

Important dependency note: these CLIs use relative includes such as `include(joinpath(@__DIR__, "..."))`. Before running them, make sure the helper files are in the same folder as the CLIs:

- `triplet_analysis_cli.jl`
- `phase_candidate_sweep_cli.jl`
- `phase_final_all_triplets_cli.jl`
- `pair_export_selected_candidate_cli.jl`
- `acausal_plus_causal_downstream_helpers.jl`

## Shared Concepts

Pair labels accept `SN43-SN63` or `SN43_SN63` where pair input is supported. Triplets use `A-B-C`, for example `SN43-SN53-SN61`.

Candidate labels use:

```text
"seed... | state label | branch"
```

Examples:

```text
"seed1235 | S1 s1=3 | causal"
"seed1237 | S2 s2=1 | acausal"
"seed1236 | (5,3) | acausal_plus_causal"
```

The acausal+causal branch accepts these aliases:

- `mean`
- `causal+acausal`
- `acausal_plus_causal`
- `selected_state_acausal_plus_causal`

Shared split-filter options:

```bash
--apply-acausal-plus-causal-split-filter true|false
--split-period X
--split-overlap X
--split-filter-order N
```

When enabled, the split filter is applied only to `selected_state_acausal_plus_causal`. `--split-period` is required, `--split-overlap` defaults to `1.0`, and `--split-filter-order` defaults to `2`.

Useful flags:

- `--dry-run`: print discovered work and expected outputs without running the expensive analysis or writing final output files.
- `--overwrite`: replace existing outputs. Without it, the CLIs refuse to overwrite existing files.
- `--show-plot`: where supported, display generated PNGs in the terminal if `chafa` or `viu` is installed.

## Recommended Workflow

1. Transfer a trained reference pair to the remaining raw pairs with `transfer_all_states_cli.jl`.
2. Rank candidates on a reference triplet with `group_velocity_acausal_plus_causal_cli.jl`.
3. Use the selected candidate labels in `phase_candidate_sweep_acausal_plus_causal_cli.jl` to find good phase-correction denominators.
4. Evaluate the final candidate and chosen denominator values over all passing triplets with `phase_final_all_triplets_acausal_plus_causal_cli.jl`.
5. Export waveforms and dispersion products for downstream tomography with `pair_export_acausal_plus_causal_candidate_cli.jl`.

## 1. `transfer_all_states_cli.jl`

### What It Does

Transfers state assignments and averaged state artifacts from one trained reference pair to other raw station pairs by matching timestamp labels. This is useful when one reference pair has trained state labels that should be reused on other pairs sharing overlapping time windows.

### Typical Command

```bash
julia --project=. --startup-file=no transfer_all_states_cli.jl \
  --saved-root /path/to/saved_root \
  --reference-pair SN43-SN63 \
  --raw-data-dir /path/to/raw_data \
  --output-dir /path/to/output/transferred_states \
  --min-overlap 1 \
  --overwrite
```

To check what would happen first:

```bash
julia --project=. --startup-file=no transfer_all_states_cli.jl \
  --saved-root /path/to/saved_root \
  --reference-pair SN43-SN63 \
  --raw-data-dir /path/to/raw_data \
  --output-dir /path/to/output/transferred_states \
  --dry-run
```

### Important Options

- `--reference-pair PAIR`: trained reference pair, for example `SN43-SN63` or `SN43_SN63`.
- `--target-pairs LIST`: comma- or space-separated target pairs. Defaults to all raw pairs except the reference pair.
- `--min-overlap N`: minimum matched timestamps required per target. Default: `1`.
- `--overwrite`: replace existing transferred artifacts and recreate audit CSVs.

### Outputs

Transferred artifacts are written under:

```text
<output-dir>/<target_pair>/<reference_run_tag>/transferred_source_state_averages.jld2
```

Audit CSVs are written directly under `<output-dir>`:

- `transfer_summary.csv`: per reference-run and target-pair status, counts, output path, and error text.
- `matched_windows.csv`: matched timestamp rows with reference and target indices plus causal/acausal assignments.
- `missing_reference_windows.csv`: reference timestamps that were not found in the target pair.

## 2. `group_velocity_acausal_plus_causal_cli.jl`

### What It Does

Ranks selected-state candidates on a single reference triplet using group-velocity consistency. It compares selected causal, selected acausal, selected acausal+causal, and global reference curves, then reports the best candidate ranks by RMSE.

### Typical Command

```bash
julia --project=. --startup-file=no group_velocity_acausal_plus_causal_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \
  --candidate-triplet SN43-SN53-SN61 \
  --output-dir /path/to/output/group_velocity_apc \
  --top-ranks 5 \
  --max-candidates 25 \
  --bandpass 10:30 \
  --overwrite
```

With acausal+causal split filtering:

```bash
julia --project=. --startup-file=no group_velocity_acausal_plus_causal_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \
  --candidate-triplet SN43-SN53-SN61 \
  --output-dir /path/to/output/group_velocity_apc \
  --apply-acausal-plus-causal-split-filter true \
  --split-period 18 \
  --split-overlap 1 \
  --split-filter-order 2 \
  --overwrite
```

### Important Options

- `--candidate-triplet A-B-C`: reference triplet used to choose candidates. Default: `SN43-SN53-SN61`.
- `--top-ranks N`: candidate ranks retained per selected family. Default: `5`.
- `--max-candidates N`: candidate states considered per pair before triplet ranking. Default: `25`.
- Geometry filters: `--max-delta-az`, `--max-delta-d`, `--min-segment-distance`.
- MFT options: `--period-min`, `--period-max`, `--nperiods`, `--velocity-min`, `--velocity-max`, `--bandwidth-factor`, `--zero-pad-factor`, `--upsample-factor`, `--precision`, `--use-phtovel`.
- Phase correction denominator options used internally for scoring: `--selected-phvel-correction-denominator`, `--causal-phvel-correction-denominator`, `--selected-mean-phvel-correction-denominator`.

### Outputs

CSV files in `<output-dir>`:

- `reference_triplet_top_candidates.csv`: ranked candidate states for the reference triplet.
- `reference_triplet_group_velocity_rows.csv`: period-level group-velocity consistency rows.
- `reference_triplet_group_velocity_rmse.csv`: RMSE summary/ranking.
- `reference_triplet_selected_state_waveforms.csv`: waveform rows for selected candidates.

Plots in `<output-dir>/plots`:

- `reference_triplet_group_velocity.html`
- `reference_triplet_group_velocity.png`
- `reference_triplet_rmse.html`
- `reference_triplet_rmse.png`
- `reference_triplet_selected_state_waveforms.html`
- `reference_triplet_selected_state_waveforms.png`

At the end, this CLI also prints a suggested follow-up command for the phase candidate sweep.

## 3. `phase_candidate_sweep_acausal_plus_causal_cli.jl`

### What It Does

Sweeps phase-correction denominators `d` for one or more candidate labels across the passing triplets. It includes global mean, global causal, global acausal, and the supplied selected candidates, then reports the best `d` values and reference-triplet error behavior.

### Typical Command

```bash
julia --project=. --startup-file=no phase_candidate_sweep_acausal_plus_causal_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \
  --reference-triplet SN43-SN53-SN61 \
  --candidates "seed1235 | S1 s1=3 | causal; seed1237 | S2 s2=1 | acausal; seed1236 | (5,3) | acausal_plus_causal" \
  --d-values "0,2:0.2:12" \
  --output-dir /path/to/output/phase_sweep_apc \
  --overwrite
```

### Important Options

- `--candidates TEXT`: semicolon-separated candidate labels in `seed | state | branch` form.
- `--reference-triplet A-B-C`: triplet used for reference reporting and plots. Default: `SN43-SN53-SN61`.
- `--d-values LIST`: denominators for phase correction `-pi/d`. Accepts comma lists and `start:step:stop` ranges, for example `0,2:0.2:12`.
- `--max-triplets N`: optional cap for quick checks. Default: all passing triplets.
- Geometry, bandpass, MFT, split-filter, `--dry-run`, `--overwrite`, and `--show-plot` options are supported.

### Outputs

CSV files in `<output-dir>`:

- `phase_reference_sweep_rows.csv`: period-level sweep rows.
- `phase_reference_sweep_summary.csv`: summary rows by curve/candidate and denominator.
- `phase_reference_sweep_argmin.csv`: per-curve argmin results.
- `phase_reference_sweep_argmin_mean.csv`: aggregated argmin means.
- `phase_reference_sweep_reference_errorbars.csv`: reference-triplet errorbar data.
- `reference_pair_averaged_crosscorrelations.csv`: averaged cross-correlation waveform rows for the reference allocation pair.
- `reference_triplet_phase_period_plot_rows.csv`: period-level rows for the reference triplet plot.

Plots in `<output-dir>/plots`:

- `reference_triplet_phase_abs_vabc_minus_vac.html`
- `reference_triplet_phase_abs_vabc_minus_vac.png`
- `reference_phase_correction_sweep.html`
- `reference_phase_correction_sweep.png`
- `reference_phase_correction_errorbars.html`
- `reference_phase_correction_errorbars.png`
- `reference_pair_averaged_crosscorrelations.html`
- `reference_pair_averaged_crosscorrelations.png`

## 4. `phase_final_all_triplets_acausal_plus_causal_cli.jl`

### What It Does

Evaluates one final candidate and chosen denominator values or schedules over all passing triplets. It is the production-style phase-consistency pass after the sweep has identified good correction settings.

### Typical Command

```bash
julia --project=. --startup-file=no phase_final_all_triplets_acausal_plus_causal_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \
  --final-candidate "seed1236 | (5,3) | acausal_plus_causal" \
  --global-mean-d 4.6 \
  --global-causal-d 5.8 \
  --global-acausal-d 4.2 \
  --candidate-d 6.0 \
  --output-dir /path/to/output/final_phase_apc \
  --overwrite
```

Using multiple candidate denominators or a period-dependent schedule:

```bash
julia --project=. --startup-file=no phase_final_all_triplets_acausal_plus_causal_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --final-candidate "seed1236 | (5,3) | acausal_plus_causal" \
  --global-mean-d 4.6 \
  --global-causal-d 5.8 \
  --global-acausal-d 4.2 \
  --candidate-d "6.2,6.5,7,7.5,8,8.5,9" \
  --candidate-d "short_mid_long:{5=4,10=5.5,18=8}" \
  --output-dir /path/to/output/final_phase_apc \
  --overwrite
```

### Important Options

- `--final-candidate TEXT`: exactly one candidate label in `seed | state | branch` form.
- `--global-mean-d X`: denominator for global mean, causal+acausal.
- `--global-causal-d X`: denominator for global causal.
- `--global-acausal-d X`: denominator for global acausal.
- `--candidate-d X`: denominator or denominators for the final selected candidate. Accepts scalar, comma list, range, or inline schedules such as `name:{5=4,10=5.5,18=8}`.
- `--candidate-d-schedules FILE`: TOML file containing `[[candidate_d_schedules]]` entries.
- `--d-values LIST`: fallback denominator list applied to all four curve families when the curve-specific options are not provided.
- Geometry, bandpass, MFT, split-filter, `--dry-run`, `--overwrite`, and `--show-plot` options are supported.

### Outputs

CSV files in `<output-dir>`:

- `final_all_triplet_phase_rows.csv`: period-level final phase-consistency rows.
- `final_all_triplet_phase_mean_abs_error.csv`: mean absolute error grouped for plotting/comparison.
- `final_all_triplet_phase_summary.csv`: summary statistics for final curves and denominator settings.

Plots in `<output-dir>/plots`:

- `final_all_triplet_phase_mean_abs_error.html`
- `final_all_triplet_phase_mean_abs_error.png`

## 5. `pair_export_acausal_plus_causal_candidate_cli.jl`

### What It Does

Exports the final selected candidate into per-pair waveform CSVs and dispersion products for downstream tomography. It writes both machine-readable CSVs and DSURFTOMO-style group/phase velocity TXT files.

### Typical Command

```bash
julia --project=. --startup-file=no pair_export_acausal_plus_causal_candidate_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \
  --final-candidate "seed1236 | (5,3) | acausal_plus_causal" \
  --global-mean-d 4.6 \
  --global-causal-d 5.8 \
  --global-acausal-d 4.2 \
  --candidate-d 6.0 \
  --pairs all \
  --output-dir /path/to/output/pair_export_apc \
  --overwrite
```

With per-period thresholding before writing TXT/plot values:

```bash
julia --project=. --startup-file=no pair_export_acausal_plus_causal_candidate_cli.jl \
  --config triplet_mft_defaults.toml \
  --saved-root /path/to/saved_root \
  --raw-data-dir /path/to/raw_data \
  --final-candidate "seed1236 | (5,3) | acausal_plus_causal" \
  --global-mean-d 4.6 \
  --global-causal-d 5.8 \
  --global-acausal-d 4.2 \
  --candidate-d 6.0 \
  --threshold-by-period \
  --threshold-mode median \
  --threshold-fraction 10 \
  --output-dir /path/to/output/pair_export_apc \
  --overwrite
```

### Important Options

- `--final-candidate TEXT`: exactly one candidate label in `seed | state | branch` form.
- `--global-mean-d`, `--global-causal-d`, `--global-acausal-d`, `--candidate-d`, `--candidate-d-schedules`, and `--d-values`: same denominator controls as the final all-triplets CLI.
- `--pairs all|triplets`: export all artifact pairs or only pairs needed by the filtered triplets. Default: `all`.
- `--threshold-by-period`: remove dispersion TXT/plot values outside per-period, per-curve bounds.
- `--threshold-mode mean|median`: thresholding mode. Default: `median`.
- `--threshold-fraction X`: percentile or percent range parameter. Default: `10`.
- Bandpass, wavelength filtering, split-filter, `--dry-run`, and `--overwrite` options are supported.

### Outputs

Folders and files under `<output-dir>`:

- `waveform_csvs/`: per-pair waveform CSVs. Each includes global acausal, global causal, global mean, best selected state, selected causal, selected acausal, selected acausal+causal, and filtered versions of those columns.
- `dispersion_csvs/all_pair_dispersion_phase_group.csv`: all-pair group and phase dispersion rows.
- `dispersion_txts/`: DSURFTOMO-style group/phase velocity TXT files, one per curve and denominator schedule.
- `dispersion_labeled_csvs/`: labeled group/phase dispersion CSVs matching the TXT exports.
- `plots/dispersion_txts/`: plots generated from the dispersion TXT outputs.
- `metadata/export_metadata.csv`: run settings, selected candidate, denominator choices, threshold settings, and split-filter settings.
- `metadata/dispersion_txt_manifest.csv`: manifest of generated TXT files and row counts.

## Quick Safety Checklist

Before running the expensive CLIs:

```bash
julia --project=. --startup-file=no <cli>.jl ... --dry-run
```

Check that:

- The number of discovered raw pairs is reasonable.
- Needed triplet pairs have artifacts under `--saved-root`.
- The intended `--output-dir` is correct.
- Existing outputs should really be replaced before adding `--overwrite`.

