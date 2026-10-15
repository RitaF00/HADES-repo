from pathlib import Path
import json

import dbetto
import lh5
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from pygama.pargen import energy_cal as pgc
from pygama.math import distributions as pgf


# ============================================================
# CONFIGURATION
# ============================================================

version = "v1.2.1"
campaign = "c1"
measure = "bkg"
required_card = "efb3"

metadata_path = (
    "/global/u2/r/ritaferi/HADES_DATA/hades-metadata"
)

# Mantengo il campo del tuo esempio.
# Per partire dall'estimatore NON calibrato:
# energy_field = "hit/cuspEmax_ctc"
energy_field = "hit/cuspEmax_ctc_cal"

energy_parameter = energy_field.rsplit("/", 1)[-1]

input_upper_cut = 8000

# Rilegge i file LH5 anche se esiste la cache.
force_rebuild = True

# Mostra tutti i plot oltre a salvarli.
show_plots = True

dpi = 150

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

base = Path(
    "/global/cfs/cdirs/m2676/data/teststands/hades/"
    f"prodenv/ref/{version}/generated/tier/hit"
)

cache_dir = (
    Path("/global/u2/r/ritaferi/HADES_DATA")
    / "energy_cache"
    / version
    / campaign
)

output_dir = (
    Path("/global/u2/r/ritaferi/HADES_DATA")
    / "background_recalibration_pygama"
    / version
    / campaign
)

output_dir.mkdir(parents=True, exist_ok=True)

# Spettro finale: bin da 1 keV.
bins_cal = np.linspace(0, 3500, 3501)

peak_pars_selected = [
    (583.191, (20, 20), pgf.hpge_peak),
    (727.330, (30, 30), pgf.hpge_peak),
    (1460.8,  (30, 30), pgf.hpge_peak),
    (2614.50, (60, 60), pgf.hpge_peak),
]

# Eventuali stime iniziali specifiche.
# Unità: keV / unità dell'energia di input.
guess_overrides = {
    # ("V03421A", "run0003"): 2614.553 / 6607,
}

db = dbetto.TextDB(metadata_path)

# Verifica che siano stati importati gli oggetti distribuzione.
assert callable(getattr(pgf.hpge_peak, "required_args", None))
assert callable(getattr(pgf.gauss_on_step, "required_args", None))


# ============================================================
# OUTPUT HELPERS
# ============================================================

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Cannot serialize {type(value).__name__}"
    )


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            indent=4,
            default=json_default,
            ensure_ascii=False,
        )


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved plot: {path}", flush=True)

        if show_plots:
            # Notebook: visualizzazione della figura.
            # Terminale: utilizza il backend matplotlib disponibile.
            try:
                from IPython import get_ipython
                in_notebook = get_ipython() is not None
            except ImportError:
                in_notebook = False

            if in_notebook:
                from IPython.display import display
                display(fig)
            else:
                # La figura può essere già stata chiusa da pygama.
                # Il file PNG è comunque salvato.
                if plt.fignum_exists(fig.number):
                    plt.show()

    finally:
        plt.close(fig)


def save_pygama_plot(method, path, *args, **kwargs):
    figures_before = set(plt.get_fignums())

    try:
        result = method(*args, **kwargs)
        fig = None

        if isinstance(result, Figure):
            fig = result

        elif isinstance(getattr(result, "figure", None), Figure):
            fig = result.figure

        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, Figure):
                    fig = item
                    break

                if isinstance(
                    getattr(item, "figure", None),
                    Figure,
                ):
                    fig = item.figure
                    break

        if fig is None:
            new_figures = sorted(
                set(plt.get_fignums()) - figures_before
            )

            if new_figures:
                fig = plt.figure(new_figures[-1])

        if fig is None:
            raise RuntimeError(
                "No accessible figure returned by pygama."
            )

        save_figure(fig, path)

    finally:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)


# ============================================================
# LOAD OR BUILD EVENT-ENERGY CACHE
#
# Salva le energie finite prima del taglio <8000.
# ============================================================

def load_or_build_energies(detector, run, files):
    folder = cache_dir / detector / energy_parameter
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / (
        f"{detector}_{measure}_{run}_input_energies.npy"
    )

    manifest_path = path.with_suffix(".json")

    manifest = {
        "energy_field": energy_field,
        "files": [
            {
                "path": str(filename),
                "size": filename.stat().st_size,
                "mtime_ns": filename.stat().st_mtime_ns,
            }
            for filename in files
        ],
    }

    cache_matches = False

    if (
        path.exists()
        and manifest_path.exists()
        and not force_rebuild
    ):
        try:
            with manifest_path.open(
                encoding="utf-8"
            ) as stream:
                cache_matches = json.load(stream) == manifest

        except (OSError, ValueError):
            cache_matches = False

    if cache_matches:
        print(
            f"Loading energy cache: {path}",
            flush=True,
        )

        energies = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if energies.ndim != 1:
            raise ValueError(
                "Cached energies must be 1D."
            )

        return energies

    if not files:
        raise FileNotFoundError(
            "No LH5 files available to build the energy cache."
        )

    arrays = []

    for index, filename in enumerate(files, start=1):
        print(
            f"  Reading file {index}/{len(files)}",
            flush=True,
        )

        values = np.asarray(
            lh5.read(
                energy_field,
                str(filename),
            ).nda,
            dtype=float,
        ).reshape(-1)

        values = values[np.isfinite(values)]

        if values.size:
            arrays.append(values)

    if not arrays:
        raise ValueError("No finite input energies.")

    energies = np.concatenate(arrays)
    del arrays

    print(
        f"Saving input-energy cache: "
        f"{energies.size:,} events "
        f"({energies.nbytes / 1e6:.1f} MB)",
        flush=True,
    )

    np.save(
        path,
        energies,
        allow_pickle=False,
    )

    save_json(manifest, manifest_path)

    return energies


# ============================================================
# PROCESS ONE DETECTOR / RUN
# ============================================================

def process_run(detector, run, meta, files):
    run_dir = output_dir / detector / run
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{detector}_{measure}_{run}"

    all_energies = load_or_build_energies(
        detector,
        run,
        files,
    )

    mask = (
        np.isfinite(all_energies)
        & (all_energies < input_upper_cut)
    )

    e_uncal = np.asarray(
        all_energies[mask],
        dtype=float,
    )

    del all_energies, mask

    if e_uncal.size == 0:
        raise ValueError(
            "No energies remain after selection."
        )

    print(
        f"Selected events: {e_uncal.size:,}",
        flush=True,
    )

    percentile = float(
        np.percentile(e_uncal, 99.9)
    )

    if not np.isfinite(percentile) or percentile <= 0:
        raise ValueError(
            "Invalid 99.9 percentile."
        )

    guess_kev = guess_overrides.get(
        (detector, run),
        2614.553 / percentile,
    )

    # --------------------------------------------------------
    # INPUT SPECTRUM
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 3),
        layout="constrained",
    )

    ax.hist(
        e_uncal,
        bins=1000,
        histtype="step",
        color="navy",
        linewidth=0.8,
    )

    ax.axvline(
        percentile,
        color="orange",
        linestyle="--",
        linewidth=0.8,
        label=f"99.9 percentile = {percentile:.2f}",
    )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)

    ax.set_xlabel(energy_parameter)
    ax.set_ylabel("Counts / bin")

    ax.set_title(
        f"{detector}/{run} — input spectrum"
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    save_figure(
        fig,
        run_dir / f"{stem}_input_spectrum.png",
    )

    # --------------------------------------------------------
    # PYGAMA CALIBRATION
    # --------------------------------------------------------

    hpgecal = pgc.HPGeCalibration(
        energy_parameter,
        [item[0] for item in peak_pars_selected],
        guess_kev=guess_kev,
        deg=1,
        debug_mode=True,
    )

    print("Finding peaks...", flush=True)

    hpgecal.hpge_find_energy_peaks(
        e_uncal
    )

    print("Getting peaks...", flush=True)

    hpgecal.hpge_get_energy_peaks(
        e_uncal
    )

    print("Fitting peaks...", flush=True)

    hpgecal.hpge_fit_energy_peaks(
        e_uncal,
        peak_pars=peak_pars_selected,
    )

    if not hpgecal.results:
        raise RuntimeError(
            "Pygama returned no calibration results."
        )

    latest_result = hpgecal.results[
        list(hpgecal.results)[-1]
    ]

    peak_parameters = latest_result.get(
        "peak_parameters", {}
    )

    if not peak_parameters:
        raise RuntimeError(
            "Pygama returned no fitted peak parameters."
        )

    coefficients = np.asarray(
        hpgecal.pars,
        dtype=float,
    )

    if (
        coefficients.shape != (2,)
        or not np.isfinite(coefficients).all()
        or coefficients[1] <= 0
    ):
        raise ValueError(
            f"Invalid calibration coefficients: {coefficients}"
        )

    print(
        f"Coefficients for {detector}/{run}:\n"
        f"  a0 = {coefficients[0]:.10g}\n"
        f"  a1 = {coefficients[1]:.10g}",
        flush=True,
    )

    # --------------------------------------------------------
    # APPLY CALIBRATION
    #
    # E_cal = a0 + a1 * E_input
    # --------------------------------------------------------

    e_cal = np.polynomial.polynomial.polyval(
        e_uncal,
        coefficients,
    )

    if not np.isfinite(e_cal).all():
        raise ValueError(
            "Non-finite calibrated energies."
        )

    counts_cal, _ = np.histogram(
        e_cal,
        bins=bins_cal,
    )

    # --------------------------------------------------------
    # SAVE CALIBRATED ENERGIES AND HISTOGRAM
    # --------------------------------------------------------

    energy_path = run_dir / f"{stem}_e_cal.npy"

    print(
        f"Saving calibrated energies "
        f"({e_cal.nbytes / 1e6:.1f} MB)...",
        flush=True,
    )

    np.save(
        energy_path,
        e_cal,
        allow_pickle=False,
    )

    print(
        f"Saved energies: {energy_path}",
        flush=True,
    )

    np.savez(
        run_dir / f"{stem}_calibrated_spectrum.npz",
        counts=counts_cal,
        bins=bins_cal,
        calibration_coefficients=coefficients,
    )

    save_json(
        {
            "detector": detector,
            "run": run,
            "version": version,
            "campaign": campaign,
            "measurement": measure,
            "card_interface": required_card,
            "energy_field": energy_field,
            "input_upper_cut": input_upper_cut,
            "selected_events": e_uncal.size,
            "guess_kev": guess_kev,
            "degree": 1,
            "coefficient_order": [
                "constant",
                "linear",
            ],
            "coefficients": coefficients,
            "peak_parameter_keys": [
                str(key) for key in peak_parameters
            ],
            "run_live_time_in_s": meta.get(
                "run_live_time_in_s"
            ),
            "input_files": [
                str(filename) for filename in files
            ],
            "e_cal_file": str(energy_path),
        },
        run_dir / f"{stem}_calibration.json",
    )

    # --------------------------------------------------------
    # PYGAMA DIAGNOSTIC PLOTS
    # --------------------------------------------------------

    plot_errors = []

    plot_settings = [
        (
            "peak_fits",
            hpgecal.plot_fits,
            {
                "figsize": (9, 6),
                "ncols": 2,
                "nrows": 2,
                "fontsize": 9,
            },
        ),
        (
            "calibration_fit",
            hpgecal.plot_cal_fit,
            {
                "figsize": (8, 6),
            },
        ),
    ]

    for plot_name, method, kwargs in plot_settings:
        try:
            save_pygama_plot(
                method,
                run_dir / f"{stem}_{plot_name}.png",
                e_uncal,
                **kwargs,
            )

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"Plot error ({plot_name}): {message}",
                flush=True,
            )

            plot_errors.append({
                "plot": plot_name,
                "error": message,
            })

    save_json(
        plot_errors,
        run_dir / f"{stem}_plot_errors.json",
    )

    # --------------------------------------------------------
    # CALIBRATED SPECTRUM
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 3),
        layout="constrained",
    )

    ax.stairs(
        counts_cal,
        bins_cal,
        color="navy",
        linewidth=0.8,
    )

    for energy, _, _ in peak_pars_selected:
        ax.axvline(
            energy,
            color="orange",
            linestyle="--",
            linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_xlim(0, 3500)

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts / 1 keV")

    ax.set_title(
        f"{detector}/{run} — {required_card}"
    )

    save_figure(
        fig,
        run_dir / f"{stem}_calibrated_spectrum.png",
    )

    return {
        "detector": detector,
        "run": run,
        "counts": counts_cal,
        "coefficients": coefficients,
        "energy_path": str(energy_path),
    }


# ============================================================
# ALL DETECTORS / ALL efb3 BACKGROUND RUNS
# ============================================================

entries = []
errors = []

for detector in detectors:
    print(
        f"\n{'=' * 70}\n"
        f"DETECTOR: {detector}",
        flush=True,
    )

    try:
        config = (
            db.hardware.configuration[detector]
            [campaign][measure]
        )

    except (KeyError, FileNotFoundError) as exc:
        message = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"Skipping {detector}: {message}",
            flush=True,
        )

        errors.append({
            "detector": detector,
            "run": "",
            "error": message,
        })

        save_json(
            errors,
            output_dir / "errors.json",
        )
        continue

    runs = sorted(
        key
        for key in config
        if key.startswith("run")
        and key[3:].isdigit()
    )

    for run in runs:
        try:
            meta = config[run]

            card = (
                meta["daq_settings"]["flashcam"]
                ["card_interface"]
            )

            if card != required_card:
                print(
                    f"Skipping {detector}/{run}: "
                    f"card={card}",
                    flush=True,
                )
                continue

            run_tag = f"r{int(run[3:]):03d}"

            folder = (
                base
                / detector
                / campaign
                / measure
            )

            files = sorted(
                folder.glob(
                    f"char_data-{detector}-{measure}-"
                    f"{run_tag}-*-tier_hit.lh5"
                )
            )

            if not files:
                raise FileNotFoundError(
                    f"No LH5 files in {folder} "
                    f"for {run_tag}."
                )

            print(
                f"Processing {detector}/{run} | "
                f"card={card} | files={len(files)}",
                flush=True,
            )

            entry = process_run(
                detector,
                run,
                meta,
                files,
            )

            entries.append(entry)

            print(
                f"Completed: {detector}/{run}",
                flush=True,
            )

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"ERROR {detector}/{run}: {message}",
                flush=True,
            )

            errors.append({
                "detector": detector,
                "run": run,
                "error": message,
            })

        finally:
            save_json(
                errors,
                output_dir / "errors.json",
            )


# ============================================================
# FINAL OVERLAY
# ============================================================

if entries:
    fig, ax = plt.subplots(
        figsize=(10, 5),
        layout="constrained",
    )

    cmap = plt.get_cmap("tab20")

    colors = {
        detector: cmap(index % 20)
        for index, detector in enumerate(detectors)
    }

    available_runs = sorted({
        entry["run"]
        for entry in entries
    })

    line_styles = ["-", "--", ":", "-."]

    styles = {
        run: line_styles[
            index % len(line_styles)
        ]
        for index, run in enumerate(available_runs)
    }

    for entry in entries:
        ax.stairs(
            entry["counts"],
            bins_cal,
            color=colors[entry["detector"]],
            linestyle=styles[entry["run"]],
            linewidth=0.8,
            alpha=0.85,
            label=(
                f"{entry['detector']} / "
                f"{entry['run']}"
            ),
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_xlim(0, 3500)

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts / 1 keV")

    ax.set_title(
        f"Background — {required_card} — {version}"
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=6,
        frameon=False,
        labelcolor="linecolor",
    )

    ax.grid(axis="y", alpha=0.2)

    save_figure(
        fig,
        output_dir / "all_backgrounds_calibrated.png",
    )

else:
    print(
        "No background spectra were successfully calibrated.",
        flush=True,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

save_json(
    [
        {
            "detector": entry["detector"],
            "run": entry["run"],
            "coefficients": entry["coefficients"],
            "energy_path": entry["energy_path"],
        }
        for entry in entries
    ],
    output_dir / "completed_calibrations.json",
)

print(
    f"\n{'=' * 70}\n"
    f"Completed: {len(entries)} detector/run calibrations.\n"
    f"Errors: {len(errors)}\n"
    f"Results: {output_dir}",
    flush=True,
)