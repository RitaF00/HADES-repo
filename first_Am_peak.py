from pathlib import Path
import csv
import json

import dbetto
import lh5
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.special import erf, erfc, xlogy
from scipy.stats import exponnorm


# ============================================================
# CONFIGURATION
# ============================================================

version = "v1.2.1"
#version = "v1.1.0"
campaign = "c1"
measure = "am_HS1_top_dlt"

metadata_path = "/global/u2/r/ritaferi/HADES_DATA/hades-metadata"
base_dir = Path(metadata_path).parent


detectors = [
    "V03421A",
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

'''
detectors = ["V03421A",
            "V03422A",
            "V04199A",
            "V04545A",
            "V08682A",
            "V08682B",
            "V09372A",
            "V14673A"]
'''

runs = [
    "run0001",
    "run0002",
    "run0003",
]

if version == "v1.2.1":
    fit_range = (54, 63)
    energy_range = (0, 65)
else:
    energy_range = (0, 150)
    fit_range = (100, 150)

# 35000 bins of 0.1 keV.
histogram_bins = np.linspace(0, 3500, 35001)

dpi = 300
show_plots = True
force_rebuild = False

spectrum_dir = base_dir / "raw_energy_spectrum"

output_dir = (
    base_dir
    / "spectral_differences"
    / version
    / campaign
    / measure
    / "all_detectors_peak_area_normalized"
)

# Background calibrati prodotti con background_efb3_single_file.py.
background_dir = (
    base_dir / "background_calibration_single_file" / version / campaign
)

parameter_names = [
    "area_peak",
    "mu",
    "sigma",
    "area_tail",
    "tau",
    "background",
    "step",
]


# ============================================================
# METADATA
# ============================================================

def metadata_label(detector, run, meta):
    """Complete metadata for terminal output."""
    pos = meta["source_position"]
    card = meta["daq_settings"]["flashcam"]["card_interface"]

    voltage = meta.get("high_voltage_in_V")
    voltage_text = (
        f"{float(voltage):g} V"
        if voltage is not None else "N/A"
    )

    live_time = float(meta["run_live_time_in_s"])

    return (
        f"{detector}/{run}\n"
        f"r={float(pos['r_in_mm']):g} mm, "
        f"z={float(pos['z_in_mm']):g} mm, "
        f"φ={float(pos['phi_in_deg']):g}°\n"
        f"Live time={live_time / 3600:g} h | "
        f"HV={voltage_text} | Card={card}"
    )


def spectrum_label(detector, meta):
    """Compact legend: one line per detector."""
    pos = meta["source_position"]

    return (
        f"{detector} | "
        f"r={float(pos['r_in_mm']):g}, "
        f"z={float(pos['z_in_mm']):g} mm | "
        f"φ={float(pos['phi_in_deg']):g}°"
    )


# ============================================================
# LOAD OR REBUILD HISTOGRAM
# ============================================================

def load_histogram(detector, run):
    path = (
        spectrum_dir
        / version
        / campaign
        / detector
        / f"{detector}_{measure}_{run}_spectrum.npz"
    )

    if path.exists() and not force_rebuild:
        try:
            with np.load(path, allow_pickle=False) as spectrum:
                counts = np.asarray(
                    spectrum["counts"], dtype=float
                ).copy()

                bins = np.asarray(
                    spectrum["bins"], dtype=float
                ).copy()

            valid = (
                counts.ndim == 1
                and bins.ndim == 1
                and counts.size == histogram_bins.size - 1
                and bins.shape == histogram_bins.shape
                and np.all(np.isfinite(counts))
                and np.all(counts >= 0)
                and np.allclose(
                    bins,
                    histogram_bins,
                    rtol=0,
                    atol=1e-8,
                )
            )

            if valid:
                print(f"Carico: {path}", flush=True)
                return counts, bins

            print(
                f"Rigenero istogramma incompatibile: {path}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"Rigenero: errore nel caricamento: {exc}",
                flush=True,
            )

    # run0001 -> r001.
    run_tag = f"r{int(run[3:]):03d}"

    hit_dir = (
        Path("/global/cfs/cdirs/m2676/data/teststands/hades")
        / "prodenv"
        / "ref"
        / version
        / "generated"
        / "tier"
        / "hit"
        / detector
        / campaign
        / measure
    )

    files = sorted(
        hit_dir.glob(
            f"char_data-{detector}-{measure}-"
            f"{run_tag}-*-tier_hit.lh5"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"Nessun file LH5 per {detector}/{run}: {hit_dir}"
        )

    counts = np.zeros(
        histogram_bins.size - 1,
        dtype=np.int64,
    )

    # Read one file at a time.
    for index, filename in enumerate(files, start=1):
        print(
            f"{detector}/{run}: file {index}/{len(files)}",
            flush=True,
        )

        energy = lh5.read(
            "hit/cuspEmax_ctc_cal",
            str(filename),
        ).nda

        partial_counts, _ = np.histogram(
            energy,
            bins=histogram_bins,
        )

        counts += partial_counts
        del energy, partial_counts

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")

    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                counts=counts,
                bins=histogram_bins,
            )

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)

    print(f"Salvato: {path}", flush=True)

    return counts.astype(float), histogram_bins.copy()


# ============================================================
# GAUSSIAN + LEFT TAIL + BACKGROUND + STEP
#
# Areas: total component counts.
# Background and step: counts / keV.
# ============================================================

def gaussian_bin_counts(edges, area, mu, sigma):
    z = (edges - mu) / (np.sqrt(2) * sigma)
    return area * 0.5 * np.diff(erf(z))


def left_tail_bin_counts(edges, area, mu, sigma, tau):
    cdf = exponnorm.sf(
        mu - edges,
        tau / sigma,
        loc=0,
        scale=sigma,
    )

    return area * np.maximum(np.diff(cdf), 0)


def step_primitive(x, mu, sigma):
    u = (x - mu) / (np.sqrt(2) * sigma)

    return 0.5 * (
        (x - mu) * erfc(u)
        - sigma * np.sqrt(2 / np.pi) * np.exp(-u**2)
    )


def am60_components(edges, parameters):
    (
        area, mu, sigma,
        tail_area, tau,
        background, step,
    ) = parameters

    peak = gaussian_bin_counts(
        edges, area, mu, sigma
    )

    tail = left_tail_bin_counts(
        edges, tail_area, mu, sigma, tau
    )

    continuum = (
        background * np.diff(edges)
        + step * np.diff(
            step_primitive(edges, mu, sigma)
        )
    )

    return {
        "gaussian": peak,
        "tail": tail,
        "continuum": continuum,
        "total": peak + tail + continuum,
    }


def am60_density(x, parameters):
    """Continuous model for smooth plotting, in counts / keV."""
    (
        area, mu, sigma,
        tail_area, tau,
        background, step,
    ) = parameters

    peak = (
        area / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-0.5 * ((x - mu) / sigma)**2)
    )

    tail = tail_area * exponnorm.pdf(
        mu - x,
        tau / sigma,
        loc=0,
        scale=sigma,
    )

    continuum = (
        background
        + step * 0.5 * erfc(
            (x - mu) / (np.sqrt(2) * sigma)
        )
    )

    return {
        "gaussian": peak,
        "tail": tail,
        "continuum": continuum,
        "total": peak + tail + continuum,
    }


def poisson_residuals(parameters, edges, counts):
    expected = np.maximum(
        am60_components(edges, parameters)["total"],
        1e-12,
    )

    deviance = 2 * (
        expected - counts
        + xlogy(counts, counts / expected)
    )

    return (
        np.sign(expected - counts)
        * np.sqrt(np.maximum(deviance, 0))
    )


# ============================================================
# FIT RAW COUNTS
# ============================================================

def fit_am60(counts, bins):
    counts = np.asarray(counts, dtype=float)
    bins = np.asarray(bins, dtype=float)

    if counts.ndim != 1 or bins.ndim != 1:
        raise ValueError("counts e bins devono essere 1D.")

    if counts.size == 0 or bins.size != counts.size + 1:
        raise ValueError(
            "Dimensioni istogramma incompatibili."
        )

    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Conteggi non validi.")

    if (
        not np.isfinite(bins).all()
        or np.any(np.diff(bins) <= 0)
    ):
        raise ValueError("Bordi non validi.")

    xmin, xmax = fit_range

    indices = np.flatnonzero(
        (bins[:-1] >= xmin)
        & (bins[1:] <= xmax)
    )

    n_parameters = 7

    if indices.size <= n_parameters:
        raise ValueError(
            "Troppi pochi bin nella finestra di fit."
        )

    first = indices[0]
    last = indices[-1] + 1

    edges = bins[first:last + 1]
    y = counts[first:last]

    if y.sum() <= 0:
        raise ValueError(
            "Nessun conteggio nella finestra di fit."
        )

    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    density = y / widths

    background = max(
        float(np.percentile(density, 10)), 1e-3
    )


    if version == "v1.2.1":
        height = max(
            density[np.argmin(np.abs(centers - 59.54))]
            - background,
            1,
        )
    
        area_guess = height * np.sqrt(2 * np.pi) * 0.4
    
        p0 = [
            area_guess, 59.54, 0.4,
            area_guess * 0.1, 1.0,
            background, background * 0.5,
        ]
    
        lower = [
            0, 58.8, 0.05,
            0, 0.05,
            0, 0,
        ]
    
        upper = [
            np.inf, 60.3, 1.2,
            np.inf, 10,
            np.inf, np.inf,
        ]
    else:
        height = max(
            density[np.argmin(np.abs(centers - 110.8))]
            - background,
            1,
        )
    
        area_guess = height * np.sqrt(2 * np.pi) * 0.4
    
        p0 = [
            area_guess,  110.8, 0.4,
            area_guess * 0.1, 1.0,
            background, background * 0.5,
        ]
    
        lower = [
            0,100, 0.05,
            0, 0.05,
            0, 0,
        ]
    
        upper = [
            np.inf,150, 1.2,
            np.inf, 10,
            np.inf, np.inf,
        ]


    fit = least_squares(
        poisson_residuals,
        p0,
        bounds=(lower, upper),
        args=(edges, y),
        x_scale="jac",
        max_nfev=20000,
    )

    if not fit.success:
        raise RuntimeError(
            f"Fit fallito: {fit.message}"
        )

    rank = np.linalg.matrix_rank(fit.jac)
    full_rank = rank == n_parameters

    covariance = (
        np.linalg.pinv(fit.jac.T @ fit.jac)
        if full_rank
        else np.full(
            (n_parameters, n_parameters), np.nan
        )
    )

    errors = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    area = float(fit.x[0])

    if not np.isfinite(area) or area <= 0:
        raise ValueError("Area gaussiana non valida.")

    return {
        "parameters": fit.x,
        "errors": errors,
        "covariance": covariance,
        "area_peak": area,
        "fwhm": 2.35482 * fit.x[2],
        "fwhm_error": 2.35482 * errors[2],
        "edges": edges,
        "counts": y,
        "components": am60_components(edges, fit.x),
        "residuals": fit.fun,
        "deviance": float(np.sum(fit.fun**2)),
        "ndof": y.size - n_parameters,
        "full_rank": full_rank,
        "at_bounds": bool(
            np.any(fit.active_mask != 0)
        ),
        "active_mask": fit.active_mask,
        "nfev": fit.nfev,
        "message": fit.message,
    }


# ============================================================
# FIT FIGURE: (5, 3)
# ============================================================

def plot_am60_fit(result, detector, run, meta):
    fig, (ax, ax_res) = plt.subplots(
        2, 1,
        figsize=(5, 3),
        sharex=True,
        height_ratios=[3, 1],
        layout="constrained",
        dpi=dpi,
    )

    edges = result["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    x = np.linspace(edges[0], edges[-1], 3000)
    smooth = am60_density(
        x, result["parameters"]
    )

    ax.stairs(
        result["counts"],
        edges,
        color="navy",
        linewidth=0.8,
        label="Data",
    )

    for key, label, color, linestyle in [
        ("total", "Total fit", "darkorange", "-"),
        ("gaussian", "Gaussian", "crimson", "--"),
        ("tail", "Left tail", "green", "--"),
        ("continuum", "Continuum", "gray", "--"),
    ]:
        ax.plot(
            x,
            smooth[key] * width,
            color=color,
            linestyle=linestyle,
            linewidth=0.9,
            label=label,
        )

    ax.set_yscale("log")

    positive = result["counts"][
        result["counts"] > 0
    ]

    ax.set_ylim(
        min(0.5, positive.min() / 2),
        max(
            result["counts"].max(),
            smooth["total"].max() * width,
        ) * 2,
    )

    p = result["parameters"]
    errors = result["errors"]

    text = (
        rf"$A_G={p[0]:.0f}\pm{errors[0]:.0f}$ counts" "\n"
        rf"$\mu={p[1]:.4f}\pm{errors[1]:.4f}$ keV" "\n"
        rf"$\sigma={p[2]:.4f}\pm{errors[2]:.4f}$ keV" "\n"
        rf"$A_T={p[3]:.0f}\pm{errors[3]:.0f}$ counts" "\n"
        rf"$\tau={p[4]:.3f}\pm{errors[4]:.3f}$ keV" "\n"
        rf"$B={p[5]:.1f}\pm{errors[5]:.1f}$ counts/keV" "\n"
        rf"$S={p[6]:.1f}\pm{errors[6]:.1f}$ counts/keV" "\n"
        f"FWHM={result['fwhm']:.4f}"
        f" ± {result['fwhm_error']:.4f} keV\n"
        f"D/ndof={result['deviance']:.1f}"
        f"/{result['ndof']}"
    )

    ax.text(
        0.98, 0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5,
        color="navy",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 2,
        },
    )

    pos = meta["source_position"]

    ax.set_title(
        f"{detector}/{run} | "
        f"r={float(pos['r_in_mm']):g}, "
        f"z={float(pos['z_in_mm']):g} mm | "
        f"φ={float(pos['phi_in_deg']):g}°",
        fontsize=7,
    )

    ax.set_ylabel(
        f"Counts / {width:g} keV",
        fontsize=7,
    )

    ax.legend(
        fontsize=5.5,
        loc="upper left",
        frameon=False,
        labelspacing=0.25,
        handlelength=1.5,
    )

    ax_res.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=0.6,
    )

    ax_res.plot(
        centers,
        result["residuals"],
        ".",
        color="navy",
        markersize=2,
    )

    ax_res.set_ylabel(
        "Deviance\nresidual",
        fontsize=6,
    )

    ax_res.set_xlabel(
        "Energy (keV)",
        fontsize=7,
    )

    ax_res.set_xlim(
        edges[0], edges[-1]
    )

    for axis in (ax, ax_res):
        axis.tick_params(labelsize=6)

    return fig


# ============================================================
# SAVE RESULTS
# ============================================================

def save_csv(rows, path):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_fit(detector, run, meta, result):
    directory = output_dir / "fits" / detector
    directory.mkdir(
        parents=True, exist_ok=True
    )

    stem = f"{detector}_{measure}_{run}_fit"
    pos = meta["source_position"]

    row = {
        "detector": detector,
        "run": run,
        "version": version,
        "campaign": campaign,
        "measurement": measure,
        "r_mm": float(pos["r_in_mm"]),
        "z_mm": float(pos["z_in_mm"]),
        "phi_deg": float(pos["phi_in_deg"]),
        "live_time_s": float(
            meta["run_live_time_in_s"]
        ),
        "high_voltage_V": meta.get(
            "high_voltage_in_V", ""
        ),
        "card_interface": (
            meta["daq_settings"]["flashcam"]
            ["card_interface"]
        ),
        "fit_xmin_keV": result["edges"][0],
        "fit_xmax_keV": result["edges"][-1],
    }

    for name, value, error in zip(
        parameter_names,
        result["parameters"],
        result["errors"],
    ):
        row[name] = value
        row[name + "_error"] = error

    row.update({
        "fwhm_keV": result["fwhm"],
        "fwhm_error_keV": result["fwhm_error"],
        "deviance": result["deviance"],
        "ndof": result["ndof"],
        "full_rank": result["full_rank"],
        "at_bounds": result["at_bounds"],
        "normalization_factor": (
            1 / result["area_peak"]
        ),
        "nfev": result["nfev"],
        "message": result["message"],
    })

    save_csv(
        [row],
        directory / f"{stem}.csv",
    )

    np.savez_compressed(
        directory / f"{stem}.npz",
        parameter_names=np.array(parameter_names),
        parameters=result["parameters"],
        errors=result["errors"],
        covariance=result["covariance"],
        edges=result["edges"],
        counts=result["counts"],
        residuals=result["residuals"],
        active_mask=result["active_mask"],
        **{
            f"model_{key}": values
            for key, values
            in result["components"].items()
        },
    )

    fig = plot_am60_fit(
        result, detector, run, meta
    )

    try:
        # Preserve the exact 5 x 3 inch canvas.
        fig.savefig(
            directory / f"{stem}.png",
            dpi=dpi,
            facecolor="white",
        )
    finally:
        plt.close(fig)

    return row


# ============================================================
# ONE OVERLAY PER RUN — CLEAN LEGEND
# ============================================================

def plot_run(entries, run, colors, legend_mode="full"):
    if legend_mode not in ("full", "detectors"):
        raise ValueError("legend_mode must be 'full' or 'detectors'.")

    # The two versions have exactly the same canvas and spectrum-axis area.
    # A separate legend axis prevents longer labels from shrinking the plot.
    height = max(4, 0.60 * len(entries) + 1.5)
    fig = plt.figure(figsize=(12, 5), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    ax_legend = fig.add_subplot(grid[0, 1])
    ax_legend.axis("off")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92)

    positives = []
    for entry in entries:
        detector = entry["detector"]
        label = (metadata_label(detector, run, entry["meta"])
                 if legend_mode == "full" else detector)
        ax.stairs(entry["values"], entry["edges"],
                  color=colors[detector], linewidth=1, label=label)
        positive = entry["values"][entry["values"] > 0]
        if positive.size:
            positives.append(positive)

    ax.set_yscale("log")
    ax.set_xlim(*energy_range)
    if version == "v1.2.1":
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 65])
        peak_position = 59.5409
    else:
        ax.set_xticks([0, 20, 40, 60, 80, 100, 115])
        peak_position = 110.8

    if positives:
        positive = np.concatenate(positives)
        ax.set_ylim(positive.min()/1.5, positive.max()*2)

    ax.axvline(peak_position, color="gray", linestyle="--",
               linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts / Gaussian peak area / bin")
    ax.set_title(f"{measure} — {run}", fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="upper left",
                     borderaxespad=0, fontsize=7,
                     labelcolor="linecolor", frameon=False,
                     labelspacing=0.8 if legend_mode == "full" else 0.4,
                     handlelength=1.5, handletextpad=0.5)
    return fig


# ============================================================
# CALIBRATED BACKGROUND: ALL AVAILABLE efb3 RUNS PER DETECTOR
# ============================================================

_background_cache = {}


def get_calibrated_background(db, detector, target_bins):
    # target_bins is the fixed histogram_bins used throughout this script.
    if detector in _background_cache:
        return _background_cache[detector]

    counts_total = np.zeros(len(target_bins) - 1, dtype=np.int64)
    time_total = 0.0
    included_runs = []
    try:
        config = db.hardware.configuration[detector][campaign]["bkg"]
    except (KeyError, FileNotFoundError):
        print(f"{detector}: no efb3 background", flush=True)
        _background_cache[detector] = None
        return None

    for bkg_run in sorted(config):
        if not (bkg_run.startswith("run") and bkg_run[3:].isdigit()):
            continue
        try:
            meta = config[bkg_run]
            card = meta["daq_settings"]["flashcam"]["card_interface"]
            if card != "efb3":
                continue
            folder = background_dir / detector / bkg_run
            stem = f"{detector}_bkg_{bkg_run}"
            energy_path = folder / f"{stem}_e_cal.npy"
            calibration_path = folder / f"{stem}_calibration.json"
            if not energy_path.exists() or not calibration_path.exists():
                continue
            with calibration_path.open(encoding="utf-8") as stream:
                calibration = json.load(stream)
            if (calibration.get("detector") != detector
                    or calibration.get("run") != bkg_run
                    or calibration.get("card_interface") != "efb3"):
                raise ValueError("Inconsistent background calibration metadata.")
            live_time = float(meta["run_live_time_in_s"])
            if not np.isfinite(live_time) or live_time <= 0:
                raise ValueError("Invalid background live time.")
            saved_time = calibration.get("run_live_time_in_s")
            if saved_time is not None and not np.isclose(float(saved_time), live_time):
                raise ValueError("Background live times disagree.")
            energy = np.load(energy_path, mmap_mode="r", allow_pickle=False)
            if energy.ndim != 1:
                raise ValueError("Background energy array must be 1D.")
            # Rehistogram events: do not split the saved 1 keV counts
            # into artificial 0.1 keV bins.
            run_counts = np.zeros_like(counts_total)
            for first in range(0, energy.size, 1000000):
                values = np.asarray(energy[first:first + 1000000])
                run_counts += np.histogram(
                    values[np.isfinite(values)], bins=target_bins
                )[0]
            counts_total += run_counts
            time_total += live_time
            included_runs.append(bkg_run)
            print(f"{detector}: calibrated bkg {bkg_run}, t={live_time:g} s", flush=True)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"Skipping background {detector}/{bkg_run}: {exc}", flush=True)

    if not included_runs:
        print(f"{detector}: no efb3 background", flush=True)
        _background_cache[detector] = None
        return None

    result = {"counts": counts_total, "live_time_s": time_total,
              "runs": included_runs}
    _background_cache[detector] = result
    return result


def plot_background_subtracted(entries, run, colors, legend_mode):
    # Reuse exactly the same layout and legend as the original overlay.
    fig = plot_run(entries, run, colors, legend_mode=legend_mode)
    ax = fig.axes[0]
    visible_values = np.concatenate([entry["values"] for entry in entries])
    absolute = np.abs(visible_values[np.isfinite(visible_values)])
    maximum = float(absolute.max()) if absolute.size else 0.0
    threshold = max(maximum * 1e-4, 1e-12)
    ax.set_yscale("symlog", linthresh=threshold)
    ymin = min(float(visible_values.min()), 0.0)
    ymax = max(float(visible_values.max()), threshold)
    #ax.set_ylim(ymin * 1.2 if ymin < 0 else -threshold, ymax * 1.5)
    ax.set_ylim(1e-5, 1)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_ylabel("Background-subtracted counts / Gaussian peak area / bin")
    ax.set_title(f"{measure} — {run} — background subtracted", fontsize=10)
    return fig


# ============================================================
# EXECUTION
# ============================================================
def save_source_background_plot(
    detector,
    run,
    meta,
    counts,
    bins,
    bkg,
):
    source_time = float(meta["run_live_time_in_s"])
    background_time = float(bkg["live_time_s"])

    if (
        not np.isfinite([source_time, background_time]).all()
        or source_time <= 0
        or background_time <= 0
    ):
        raise ValueError("Invalid source or background live time.")

    alpha = source_time / background_time
    scaled_background = alpha * bkg["counts"]

    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    ylabel = (
        f"Counts / {widths[0]:g} keV"
        if np.allclose(widths, widths[0])
        else "Counts / bin"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 3.5),
        dpi=dpi,
        layout="constrained",
    )

    for ax, (xmin, xmax) in zip(
        axes,
        [(0, 3500), (0, 500)],
    ):
        ax.stairs(
            counts,
            bins,
            color="navy",
            linewidth=0.8,
            label="Source data",
        )

        ax.stairs(
            scaled_background,
            bins,
            color="darkorange",
            linewidth=0.8,
            label="Background scaled to source live time",
        )

        visible = (
            (centers >= xmin)
            & (centers < xmax)
        )

        positive = np.concatenate([
            counts[visible & (counts > 0)],
            scaled_background[
                visible & (scaled_background > 0)
            ],
        ])

        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)

        if positive.size:
            ax.set_ylim(
                positive.min() / 2,
                positive.max() * 2,
            )

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xmin}–{xmax} keV", fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
    )

    axes[1].text(
        0.03,
        0.04,
        f"Source live time: {source_time / 3600:g} h\n"
        f"Background live time: {background_time / 3600:g} h\n"
        f"Background runs: {', '.join(bkg['runs'])}\n"
        f"Scale: {alpha:.5g}",
        transform=axes[1].transAxes,
        fontsize=7,
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    fig.suptitle(
        f"{detector}/{run} — {measure}",
        fontsize=10,
    )

    directory = (
        output_dir
        / "source_background_comparison"
        / detector
    )
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / (
        f"{detector}_{measure}_{run}"
        "_source_background_full_and_zoom.png"
    )

    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}", flush=True)

        if show_plots:
            plt.show()

    finally:
        plt.close(fig)



def main():
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    db = dbetto.TextDB(metadata_path)

    detector_list = list(
        dict.fromkeys(detectors)
    )

    cmap = plt.get_cmap("tab20")

    colors = {
        detector: cmap(index % 20)
        for index, detector
        in enumerate(detector_list)
    }

    fit_rows = []
    errors = []

    for run in runs:
        print(
            f"\n{'=' * 70}\nRUN: {run}",
            flush=True,
        )

        entries = []
        subtracted_entries = []

        for detector in detector_list:
            try:
                meta = (
                    db.hardware.configuration[detector]
                    [campaign][measure][run]
                )

                print(
                    "\n" + metadata_label(
                        detector, run, meta
                    ),
                    flush=True,
                )

                counts, bins = load_histogram(
                    detector, run
                )

                # Fit original raw counts.
                result = fit_am60(counts, bins)

                for name, value, error in zip(
                    parameter_names,
                    result["parameters"],
                    result["errors"],
                ):
                    print(
                        f"{name} = {value:.6g}"
                        f" ± {error:.3g}",
                        flush=True,
                    )

                if (
                    result["at_bounds"]
                    or not result["full_rank"]
                ):
                    print(
                        "CHECK FIT: boundary parameter "
                        "or unidentified uncertainties.",
                        flush=True,
                    )

                # Divide only by the Gaussian area.
                normalized = (
                    counts / result["area_peak"]
                )

                centers = (
                    bins[:-1] + bins[1:]
                ) / 2

                indices = np.flatnonzero(
                    (centers >= energy_range[0])
                    & (centers < energy_range[1])
                )

                if not indices.size:
                    raise ValueError(
                        "No bins in plotted energy range."
                    )

                first = indices[0]
                last = indices[-1] + 1

                row = save_fit(
                    detector, run, meta, result
                )

                fit_rows.append(row)

                entries.append({
                    "detector": detector,
                    "meta": meta,
                    "values": normalized[first:last],
                    "edges": bins[first:last + 1],
                })

                # Additional spectrum: time-scaled background subtraction.
                # Fit remains on raw source data; Poisson fit is NOT applied
                # to signed background-subtracted counts.
                try:
                    bkg = get_calibrated_background(db, detector, bins)
                    if bkg is not None:
                        # Il confronto mostra i conteggi originali della sorgente
                        # e il background scalato al suo live time, senza /area.
                        # Un errore grafico non deve bloccare la sottrazione.
                        try:
                            save_source_background_plot(
                                detector=detector,
                                run=run,
                                meta=meta,
                                counts=counts,
                                bins=bins,
                                bkg=bkg,
                            )
                        except Exception as plot_exc:
                            print(
                                f"Source/background plot failed {detector}/{run}: {plot_exc}",
                                flush=True,
                            )
                            errors.append({
                                "detector": detector,
                                "run": run,
                                "error": f"Source/background plot: {plot_exc}",
                            })

                        source_time = float(meta["run_live_time_in_s"])
                        if not np.isfinite(source_time) or source_time <= 0:
                            raise ValueError("Invalid source live time.")
                        alpha = source_time / bkg["live_time_s"]
                        scaled_background = alpha * bkg["counts"]
                        net = counts - scaled_background
                        variance = counts + alpha**2 * bkg["counts"]
                        area = result["area_peak"]
                        subtracted_entries.append({
                            "detector": detector, "meta": meta,
                            "values": net[first:last] / area,
                            "edges": bins[first:last + 1],
                        })
                        net_dir = output_dir / "background_subtracted" / detector
                        net_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            net_dir / f"{detector}_{measure}_{run}_net.npz",
                            bins=bins, source_counts=counts,
                            background_counts=bkg["counts"],
                            scaled_background=scaled_background,
                            net_counts=net,
                            statistical_error=np.sqrt(variance),
                            normalized_counts=net / area,
                            normalized_statistical_error=np.sqrt(variance) / area,
                            gaussian_area=area,
                            source_live_time_s=source_time,
                            background_live_time_s=bkg["live_time_s"],
                            scale=alpha,
                            background_runs=np.array(bkg["runs"]),
                        )
                        print(f"{detector}/{run}: background scale={alpha:.6g}", flush=True)
                except Exception as exc:
                    print(f"Background subtraction failed {detector}/{run}: {exc}", flush=True)
                    errors.append({"detector": detector, "run": run,
                                   "error": f"Background subtraction: {exc}"})

                print(
                    "Normalization: divide by "
                    f"{result['area_peak']:.6g} counts",
                    flush=True,
                )

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"Skipping {detector}/{run}: "
                    f"{message}",
                    flush=True,
                )

                errors.append({
                    "detector": detector,
                    "run": run,
                    "error": message,
                })

            save_csv(
                fit_rows,
                output_dir / "fit_parameters.csv",
            )

            save_csv(
                errors,
                output_dir / "errors.csv",
            )

        if not entries:
            print(
                f"No usable spectra for {run}: "
                "skipping overlay.",
                flush=True,
            )
            continue

        xmin, xmax = energy_range
        for legend_mode in ("full", "detectors"):
            fig = plot_run(entries, run, colors, legend_mode=legend_mode)
            try:
                path = output_dir / (
                    f"overlay_{run}_{xmin:g}_{xmax:g}keV_"
                    f"legend_{legend_mode}.png"
                )
                # Same canvas and axis positions for both legend versions.
                fig.savefig(path, dpi=dpi, facecolor="white")
                print(f"Saved: {path}", flush=True)
                if show_plots:
                    plt.show()
            finally:
                plt.close(fig)

        if subtracted_entries:
            for legend_mode in ("full", "detectors"):
                fig = plot_background_subtracted(
                    subtracted_entries, run, colors, legend_mode
                )
                try:
                    path = output_dir / (
                        f"overlay_background_subtracted_{run}_"
                        f"{xmin:g}_{xmax:g}keV_legend_{legend_mode}.png"
                    )
                    fig.savefig(path, dpi=dpi, facecolor="white")
                    print(f"Saved: {path}", flush=True)
                    if show_plots:
                        plt.show()
                finally:
                    plt.close(fig)
        else:
            print(f"{run}: no detectors with usable calibrated efb3 background.", flush=True)

    print(
        f"\nResults saved in: {output_dir}\n"
        f"Successful fits: {len(fit_rows)}\n"
        f"Skipped detector/runs: {len(errors)}",
        flush=True,
    )



if __name__ == "__main__":
    main()