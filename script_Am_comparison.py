import csv
from pathlib import Path

import dbetto
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

version = "v1.2.1"
measure = "am_HS1_top_dlt"
campaign = "c1"

metadata_path = "/global/u2/r/ritaferi/HADES_DATA/hades-metadata"
base_dir = Path(metadata_path).parent

reference_detector = "V03421A"

detectors = [
    "V03422A",
    "V06643A",
    "V06659A",
    "V07298B",
    "V07302A",
    "V07302B",
    "V07647B",
    "V10437B",
    "V10447B",
    "V10784A",
    "V11924A",
    "V11925A",
    "V11947B",
    "V13044A",
    "V13049A",
    "V14618A",
    "V14654A",
    "V14673A",
]

energy_ranges = [
    (20, 150),
    (200, 350),
    (600, 700),
    (1350, 1600),
    (2550, 2700),
]

heatmap_range = (20, 150)

position_tolerance = 5.0  # mm: applied independently to r and z
phi_tolerance = 5.0       # degrees

# Original 0.5 keV bins become 2 keV bins.
rebin_factor = 4

# Mask bins with insufficient reference counts.
min_reference_counts = 25

heatmap_limit_percent = 50
dpi = 200

reference_color = "navy"
detector_color = "deepskyblue"

spectrum_dir = base_dir / "raw_energy_spectrum"

output_dir = (
    base_dir
    / "spectral_differences"
    / version
    / campaign
    / measure
    / f"reference_{reference_detector}"
)


# ============================================================
# METADATA AND RUN MATCHING
# ============================================================

def get_run_names(configuration):
    return sorted(
        key
        for key in configuration
        if key.startswith("run") and key[3:].isdigit()
    )


def same_setup(
    meta_ref,
    meta_other,
    position_tolerance,
    phi_tolerance,
):
    card_ref = (
        meta_ref["daq_settings"]["flashcam"]["card_interface"]
    )
    card_other = (
        meta_other["daq_settings"]["flashcam"]["card_interface"]
    )

    if card_ref != card_other:
        return False

    position_ref = meta_ref["source_position"]
    position_other = meta_other["source_position"]

    for key in ("r_in_mm", "z_in_mm"):
        value_ref = float(position_ref[key])
        value_other = float(position_other[key])

        if not np.isfinite(value_ref) or not np.isfinite(value_other):
            return False

        if abs(value_ref - value_other) > position_tolerance:
            return False

    phi_ref = float(position_ref["phi_in_deg"])
    phi_other = float(position_other["phi_in_deg"])

    if not np.isfinite(phi_ref) or not np.isfinite(phi_other):
        return False

    delta_phi = (phi_ref - phi_other + 180) % 360 - 180

    return abs(delta_phi) <= phi_tolerance


def find_pairs(db, reference, detector, measurement, campaign):
    config_ref = db.hardware.configuration[reference][campaign]
    config_other = db.hardware.configuration[detector][campaign]

    if (
        measurement not in config_ref
        or measurement not in config_other
    ):
        return []

    pairs = []

    for run_ref in get_run_names(config_ref[measurement]):
        meta_ref = (
            db.hardware.configuration[reference]
            [campaign][measurement][run_ref]
        )

        for run_other in get_run_names(config_other[measurement]):
            meta_other = (
                db.hardware.configuration[detector]
                [campaign][measurement][run_other]
            )

            if same_setup(
                meta_ref,
                meta_other,
                position_tolerance,
                phi_tolerance,
            ):
                pairs.append(
                    (run_ref, meta_ref, run_other, meta_other)
                )

    return pairs


# ============================================================
# CACHED HISTOGRAMS AND REBINNING
# ============================================================

def load_cached_histogram(detector, run):
    path = (
        spectrum_dir
        / version
        / campaign
        / detector
        / f"{detector}_{measure}_{run}_spectrum.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cached histogram not found: {path}"
        )

    with np.load(path, allow_pickle=False) as spectrum:
        counts = np.asarray(
            spectrum["counts"], dtype=float
        ).copy()

        bins = np.asarray(
            spectrum["bins"], dtype=float
        ).copy()

    if (
        counts.ndim != 1
        or bins.ndim != 1
        or len(bins) != len(counts) + 1
        or not np.all(np.isfinite(bins))
        or not np.all(np.diff(bins) > 0)
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0)
    ):
        raise ValueError(f"Invalid histogram: {path}")

    return counts, bins


def rebin_histogram(counts, bins, factor):
    if not isinstance(factor, int) or factor < 1:
        raise ValueError(
            "rebin_factor must be a positive integer."
        )

    if len(counts) % factor != 0:
        raise ValueError(
            "The number of bins must be divisible by rebin_factor."
        )

    rebinned_counts = counts.reshape(-1, factor).sum(axis=1)
    rebinned_bins = bins[::factor].copy()

    return rebinned_counts, rebinned_bins


def load_pair_histograms(
    reference,
    run_ref,
    detector,
    run_other,
    histogram_cache,
):
    key_ref = (reference, run_ref)
    key_other = (detector, run_other)

    if key_ref not in histogram_cache:
        counts, bins = load_cached_histogram(reference, run_ref)
        histogram_cache[key_ref] = rebin_histogram(
            counts, bins, 1
        )

    if key_other not in histogram_cache:
        counts, bins = load_cached_histogram(detector, run_other)
        histogram_cache[key_other] = rebin_histogram(
            counts, bins, 1
        )

    counts_ref, bins_ref = histogram_cache[key_ref]
    counts_other, bins_other = histogram_cache[key_other]

    if not np.array_equal(bins_ref, bins_other):
        raise ValueError(
            f"Different bin edges: {reference}/{run_ref} "
            f"and {detector}/{run_other}"
        )

    return counts_ref, counts_other, bins_ref


def select_window(bins, xmin, xmax):
    centers = (bins[:-1] + bins[1:]) / 2

    # Preserve the original selection rule: use bin centers.
    return (centers >= xmin) & (centers < xmax)


# ============================================================
# RATE DIFFERENCES AND APPROXIMATE UNCERTAINTIES
# ============================================================

def calculate_percent_difference(
    counts_ref,
    counts_other,
    time_ref,
    time_other,
    minimum_counts,
):
    rate_ref = counts_ref / time_ref
    rate_other = counts_other / time_other

    difference = np.full(rate_ref.shape, np.nan)
    uncertainty = np.full(rate_ref.shape, np.nan)

    valid = (
        (counts_ref > 0)
        & (counts_ref >= minimum_counts)
    )

    ratio = rate_other[valid] / rate_ref[valid]
    difference[valid] = 100 * (ratio - 1)

    # Approximate propagation of independent Poisson variances.
    scale = time_ref / time_other

    variance_ratio = scale**2 * (
        counts_other[valid] / counts_ref[valid]**2
        + counts_other[valid]**2 / counts_ref[valid]**3
    )

    uncertainty[valid] = 100 * np.sqrt(variance_ratio)

    return rate_ref, rate_other, difference, uncertainty


# ============================================================
# SHAPE DIFFERENCES FOR THE HEATMAP
# ============================================================

def calculate_shape_difference(
    counts_ref,
    counts_other,
    bins,
    energy_range,
    minimum_counts,
):
    xmin, xmax = energy_range
    window = select_window(bins, xmin, xmax)

    if not np.any(window):
        raise ValueError("No bins in the shape-normalization window.")

    total_ref = counts_ref[window].sum()
    total_other = counts_other[window].sum()

    if total_ref <= 0 or total_other <= 0:
        raise ValueError(
            "Cannot normalize spectral shapes: "
            "zero counts in the heatmap window."
        )

    # Normalize using ALL bins in the window BEFORE masking.
    shape_ref = counts_ref / total_ref
    shape_other = counts_other / total_other

    difference = np.full(counts_ref.shape, np.nan)

    valid = (
        window
        & (counts_ref > 0)
        & (counts_ref >= minimum_counts)
    )

    difference[valid] = 100 * (
        shape_other[valid] / shape_ref[valid] - 1
    )

    return difference


# ============================================================
# METADATA LABELS
# ============================================================

def make_metadata_label(detector, run, meta):
    position = meta["source_position"]
    card = meta["daq_settings"]["flashcam"]["card_interface"]

    voltage = meta.get("high_voltage_in_V")
    voltage_text = (
        f"{float(voltage):g} V"
        if voltage is not None
        else "not available"
    )

    return (
        f"{detector} — {run}\n"
        f"r = {float(position['r_in_mm']):g} mm, "
        f"z = {float(position['z_in_mm']):g} mm, "
        f"φ = {float(position['phi_in_deg']):g}°\n"
        f"Live time = "
        f"{float(meta['run_live_time_in_s']) / 3600:g} h\n"
        f"Voltage = {voltage_text}\n"
        f"Card interface: {card}"
    )


# ============================================================
# RATE SPECTRA WITH PERCENTAGE DIFFERENCES
# ============================================================

def plot_percent_comparison(
    detector,
    run_ref,
    meta_ref,
    run_other,
    meta_other,
    counts_ref,
    counts_other,
    bins,
):
    time_ref = float(meta_ref["run_live_time_in_s"])
    time_other = float(meta_other["run_live_time_in_s"])

    rate_ref, rate_other, difference, uncertainty = (
        calculate_percent_difference(
            counts_ref,
            counts_other,
            time_ref,
            time_other,
            min_reference_counts,
        )
    )

    fig = plt.figure(
        figsize=(16, 13),
        layout="constrained",
    )
    outer_grid = fig.add_gridspec(3, 2)

    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    ylabel = (
        f"Counts / ({widths[0]:g} keV s)"
        if np.allclose(widths, widths[0])
        else "Counts / (bin s)"
    )

    for index, (xmin, xmax) in enumerate(energy_ranges):
        inner_grid = outer_grid[index // 2, index % 2].subgridspec(
            2,
            1,
            height_ratios=[3, 1.3],
            hspace=0,
        )

        ax_spectrum = fig.add_subplot(inner_grid[0])
        ax_difference = fig.add_subplot(
            inner_grid[1],
            sharex=ax_spectrum,
        )

        ax_spectrum.stairs(
            rate_ref,
            bins,
            color=reference_color,
            label=reference_detector,
            linewidth=1.2,
        )

        ax_spectrum.stairs(
            rate_other,
            bins,
            color=detector_color,
            label=detector,
            linewidth=1.2,
        )

        ax_spectrum.set_yscale("log")
        ax_spectrum.set_ylabel(ylabel)
        ax_spectrum.set_title(f"{xmin}–{xmax} keV")
        ax_spectrum.tick_params(axis="x", labelbottom=False)

        ax_spectrum.legend(
            fontsize=9,
            labelcolor="linecolor",
        )

        visible = select_window(bins, xmin, xmax)

        positive = np.concatenate([
            rate_ref[visible & (rate_ref > 0)],
            rate_other[visible & (rate_other > 0)],
        ])

        if positive.size:
            ax_spectrum.set_ylim(
                positive.min() / 1.5,
                positive.max() * 2,
            )

        ax_difference.stairs(
            difference,
            bins,
            color=reference_color,
            linewidth=1,
        )

        valid = visible & np.isfinite(difference)

        ax_difference.errorbar(
            centers[valid],
            difference[valid],
            yerr=uncertainty[valid],
            fmt="none",
            ecolor=reference_color,
            alpha=0.3,
            linewidth=0.6,
        )

        ax_difference.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )

        ax_difference.set_ylabel("Difference (%)")
        ax_difference.set_xlabel("Energy (keV)")
        ax_difference.set_xlim(xmin, xmax)
        ax_difference.grid(axis="y", alpha=0.2)

    ax_info = fig.add_subplot(outer_grid[2, 1])
    ax_info.axis("off")

    ax_info.text(
        0.03,
        0.97,
        make_metadata_label(reference_detector, run_ref, meta_ref),
        color=reference_color,
        fontsize=11,
        va="top",
    )

    ax_info.text(
        0.03,
        0.60,
        make_metadata_label(detector, run_other, meta_other),
        color=detector_color,
        fontsize=11,
        va="top",
    )

    bin_description = (
        f"Rebinned width: {widths[0]:g} keV"
        if np.allclose(widths, widths[0])
        else "Rebinned variable-width bins"
    )

    ax_info.text(
        0.03,
        0.22,
        "Difference = 100 × (detector rate / reference rate − 1)\n"
        f"{bin_description}\n"
        f"Reference count threshold: {min_reference_counts}\n"
        "Error bars: approximate 1σ statistical uncertainty",
        fontsize=9,
        va="top",
    )

    fig.suptitle(
        f"{detector} vs {reference_detector} — {measure}",
        fontsize=15,
    )

    return fig


# ============================================================
# SHAPE COMPARISON: NORMALIZE EACH ENERGY WINDOW
# ============================================================

def plot_shape_comparison(
    detector,
    run_ref,
    run_other,
    counts_ref,
    counts_other,
    bins,
):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 11),
        layout="constrained",
    )

    for ax, (xmin, xmax) in zip(axes.flat, energy_ranges):
        mask = select_window(bins, xmin, xmax)

        total_ref = counts_ref[mask].sum()
        total_other = counts_other[mask].sum()

        if total_ref <= 0 or total_other <= 0:
            ax.text(
                0.5,
                0.5,
                "Insufficient counts",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        else:
            shape_ref = counts_ref / total_ref
            shape_other = counts_other / total_other

            ax.stairs(
                shape_ref,
                bins,
                color=reference_color,
                label=reference_detector,
                linewidth=1.2,
            )

            ax.stairs(
                shape_other,
                bins,
                color=detector_color,
                label=detector,
                linewidth=1.2,
            )

            ax.set_yscale("log")

            positive = np.concatenate([
                shape_ref[mask & (shape_ref > 0)],
                shape_other[mask & (shape_other > 0)],
            ])

            if positive.size:
                ax.set_ylim(
                    positive.min() / 1.5,
                    positive.max() * 2,
                )

            ax.legend(
                fontsize=9,
                labelcolor="linecolor",
            )

        ax.set_xlim(xmin, xmax)
        ax.set_title(f"Normalized within {xmin}–{xmax} keV")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Fraction of window counts / bin")
        ax.grid(axis="y", alpha=0.2)

    axes[2, 1].axis("off")

    fig.suptitle(
        f"Shape comparison — {measure}\n"
        f"{reference_detector}/{run_ref} "
        f"vs {detector}/{run_other}",
        fontsize=14,
    )

    return fig


# ============================================================
# INTEGRATED WINDOW VALUES
# ============================================================

def calculate_window_summary(
    detector,
    run_ref,
    meta_ref,
    run_other,
    meta_other,
    counts_ref,
    counts_other,
    bins,
):
    rows = []

    time_ref = float(meta_ref["run_live_time_in_s"])
    time_other = float(meta_other["run_live_time_in_s"])

    for xmin, xmax in energy_ranges:
        mask = select_window(bins, xmin, xmax)

        n_ref = float(counts_ref[mask].sum())
        n_other = float(counts_other[mask].sum())

        rate_ref = n_ref / time_ref
        rate_other = n_other / time_other

        difference = np.nan
        difference_error = np.nan

        if n_ref > 0:
            difference = 100 * (rate_other / rate_ref - 1)

            scale = time_ref / time_other

            variance = scale**2 * (
                n_other / n_ref**2
                + n_other**2 / n_ref**3
            )

            difference_error = 100 * np.sqrt(variance)

        rows.append({
            "reference": reference_detector,
            "reference_run": run_ref,
            "detector": detector,
            "detector_run": run_other,
            "xmin_keV": xmin,
            "xmax_keV": xmax,
            "counts_reference": n_ref,
            "counts_detector": n_other,
            "rate_reference_per_s": rate_ref,
            "rate_detector_per_s": rate_other,
            "rate_reference_error_per_s": np.sqrt(n_ref) / time_ref,
            "rate_detector_error_per_s": np.sqrt(n_other) / time_other,
            "difference_percent": difference,
            "difference_error_percent": difference_error,
        })

    return rows


# ============================================================
# SHAPE HEATMAP: 20–150 keV ONLY
# ============================================================

def plot_shape_heatmap(entries, bins, run_ref, meta_ref):
    xmin, xmax = heatmap_range

    mask = select_window(bins, xmin, xmax)
    indices = np.flatnonzero(mask)

    if not indices.size:
        raise ValueError("No bins in the heatmap window.")

    first = indices[0]
    last = indices[-1] + 1

    matrix = np.vstack([
        entry["shape_difference"][mask]
        for entry in entries
    ])

    labels = []

    for entry in entries:
        position = entry["position"]

        labels.append(
            f"{entry['detector']} — {entry['run']} | "
            f"r={float(position['r_in_mm']):g}, "
            f"z={float(position['z_in_mm']):g} mm | "
            f"φ={float(position['phi_in_deg']):g}°"
        )

    fig, ax = plt.subplots(
        figsize=(15, max(5, 0.40 * len(entries) + 2.5)),
        layout="constrained",
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")

    norm = TwoSlopeNorm(
        vmin=-heatmap_limit_percent,
        vcenter=0,
        vmax=heatmap_limit_percent,
    )

    image = ax.pcolormesh(
        bins[first:last + 1],
        np.arange(len(entries) + 1),
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_yticks(np.arange(len(entries)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Compatible detector runs")

    position_ref = meta_ref["source_position"]

    ax.set_title(
        f"Spectral shape differences — {measure}\n"
        f"Reference: {reference_detector}/{run_ref} | "
        f"r={float(position_ref['r_in_mm']):g} mm, "
        f"z={float(position_ref['z_in_mm']):g} mm, "
        f"φ={float(position_ref['phi_in_deg']):g}°\n"
        f"Each spectrum normalized independently within "
        f"{xmin}–{xmax} keV",
        fontsize=12,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        extend="both",
    )

    colorbar.set_label(
        "100 × (detector shape / reference shape − 1) [%]"
    )

    fig.supxlabel(
        f"Gray: reference bin has fewer than "
        f"{min_reference_counts} counts. "
        f"Colors saturate at ±{heatmap_limit_percent}%.\n"
        f"Matching: same card, Δr and Δz ≤ "
        f"{position_tolerance:g} mm, Δφ ≤ {phi_tolerance:g}°.",
        fontsize=9,
    )

    return fig


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_figure(fig, path):
    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(fig)


def save_csv(rows, path, fields):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summaries(window_rows, errors):
    save_csv(
        window_rows,
        output_dir / "window_summary.csv",
        fields=[
            "reference",
            "reference_run",
            "detector",
            "detector_run",
            "xmin_keV",
            "xmax_keV",
            "counts_reference",
            "counts_detector",
            "rate_reference_per_s",
            "rate_detector_per_s",
            "rate_reference_error_per_s",
            "rate_detector_error_per_s",
            "difference_percent",
            "difference_error_percent",
        ],
    )

    save_csv(
        errors,
        output_dir / "errors.csv",
        fields=[
            "detector",
            "reference_run",
            "detector_run",
            "error",
        ],
    )


# ============================================================
# EXECUTION
# ============================================================

def main():
    print("Starting detector comparisons...", flush=True)
    print(f"Measurement: {measure}", flush=True)
    print(f"Reference: {reference_detector}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    db = dbetto.TextDB(metadata_path)

    percent_dir = output_dir / "percent_differences"
    shape_dir = output_dir / "shape_comparisons"
    heatmap_dir = output_dir / "shape_heatmaps_20_150keV"

    for directory in (
        percent_dir,
        shape_dir,
        heatmap_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    window_rows = []
    errors = []
    heatmap_entries = {}
    reference_metadata = {}
    heatmap_bins = None

    # Keep reference histograms in memory across detector comparisons.
    reference_cache = {}

    for detector in tqdm(
        detectors,
        desc="Detectors",
        unit="detector",
    ):
        try:
            pairs = find_pairs(
                db,
                reference_detector,
                detector,
                measure,
                campaign,
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            tqdm.write(f"ERROR {detector}: {error}")

            errors.append({
                "detector": detector,
                "reference_run": "",
                "detector_run": "",
                "error": error,
            })

            save_summaries(window_rows, errors)
            continue

        tqdm.write(
            f"{detector}: {len(pairs)} compatible run pairs."
        )

        if not pairs:
            continue

        # Release other-detector histograms after each detector.
        histogram_cache = dict(reference_cache)

        for run_ref, meta_ref, run_other, meta_other in tqdm(
            pairs,
            desc=f"Runs: {detector}",
            unit="pair",
            leave=False,
        ):
            figures_before = set(plt.get_fignums())

            try:
                time_ref = float(
                    meta_ref["run_live_time_in_s"]
                )
                time_other = float(
                    meta_other["run_live_time_in_s"]
                )

                if (
                    not np.isfinite(time_ref)
                    or not np.isfinite(time_other)
                    or time_ref <= 0
                    or time_other <= 0
                ):
                    raise ValueError("Invalid live time.")

                counts_ref, counts_other, bins = (
                    load_pair_histograms(
                        reference_detector,
                        run_ref,
                        detector,
                        run_other,
                        histogram_cache,
                    )
                )

                reference_cache[(reference_detector, run_ref)] = (
                    counts_ref,
                    bins,
                )

                if heatmap_bins is None:
                    heatmap_bins = bins.copy()

                elif not np.array_equal(heatmap_bins, bins):
                    raise ValueError(
                        "Bin edges are incompatible with "
                        "the existing heatmap data."
                    )

                filename = (
                    f"{reference_detector}_{run_ref}"
                    f"_vs_{detector}_{run_other}.png"
                )

                fig = plot_percent_comparison(
                    detector,
                    run_ref,
                    meta_ref,
                    run_other,
                    meta_other,
                    counts_ref,
                    counts_other,
                    bins,
                )

                save_figure(fig, percent_dir / filename)

                fig = plot_shape_comparison(
                    detector,
                    run_ref,
                    run_other,
                    counts_ref,
                    counts_other,
                    bins,
                )

                save_figure(fig, shape_dir / filename)

                window_rows.extend(
                    calculate_window_summary(
                        detector,
                        run_ref,
                        meta_ref,
                        run_other,
                        meta_other,
                        counts_ref,
                        counts_other,
                        bins,
                    )
                )

                # Heatmaps use SHAPE differences, not rate differences.
                shape_difference = calculate_shape_difference(
                    counts_ref,
                    counts_other,
                    bins,
                    heatmap_range,
                    min_reference_counts,
                )

                heatmap_entries.setdefault(run_ref, []).append({
                    "detector": detector,
                    "run": run_other,
                    "position": {
                        key: float(meta_other["source_position"][key])
                        for key in (
                            "r_in_mm",
                            "z_in_mm",
                            "phi_in_deg",
                        )
                    },
                    "shape_difference": shape_difference,
                })

                reference_metadata[run_ref] = meta_ref

                tqdm.write(
                    f"Saved: {detector}/{run_other} "
                    f"vs {reference_detector}/{run_ref}"
                )

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                tqdm.write(
                    f"ERROR {detector}/{run_other}: {error}"
                )

                errors.append({
                    "detector": detector,
                    "reference_run": run_ref,
                    "detector_run": run_other,
                    "error": error,
                })

            finally:
                for number in (
                    set(plt.get_fignums()) - figures_before
                ):
                    plt.close(number)

            save_summaries(window_rows, errors)

    # One SHAPE heatmap per reference run.
    if heatmap_bins is not None:
        xmin, xmax = heatmap_range

        for run_ref, entries in tqdm(
            sorted(heatmap_entries.items()),
            desc="Shape heatmaps",
            unit="heatmap",
        ):
            figures_before = set(plt.get_fignums())

            try:
                fig = plot_shape_heatmap(
                    entries,
                    heatmap_bins,
                    run_ref,
                    reference_metadata[run_ref],
                )

                save_figure(
                    fig,
                    heatmap_dir / (
                        f"shape_heatmap_{run_ref}_"
                        f"{xmin}_{xmax}_keV.png"
                    ),
                )

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                tqdm.write(
                    f"ERROR heatmap {run_ref}: {error}"
                )

                errors.append({
                    "detector": "HEATMAP",
                    "reference_run": run_ref,
                    "detector_run": "",
                    "error": error,
                })

            finally:
                for number in (
                    set(plt.get_fignums()) - figures_before
                ):
                    plt.close(number)

    save_summaries(window_rows, errors)

    tqdm.write(f"\nResults saved in: {output_dir}")
    tqdm.write(f"Shape heatmaps saved in: {heatmap_dir}")
    tqdm.write(f"Errors recorded: {len(errors)}")


if __name__ == "__main__":
    main()