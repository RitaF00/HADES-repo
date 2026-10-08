import glob
from pathlib import Path

import dbetto
import lh5
import numpy as np
import matplotlib

# Salvataggio senza aprire finestre sul login node.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm


# ============================================================
# CONFIGURAZIONE
# ============================================================

version = "v1.2.1"
measure = "am_HS1_top_dlt"
campaign = "c1"

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
    "V11924A",
    "V11925A",
    "V11947B",
    "V13044A",
    "V13049A",
    "V14618A",
    "V14654A",
    "V14673A",
]

bins = np.linspace(0, 7000, 14000 + 1)

energy_ranges = [
    (20, 150),
    (200, 350),
    (600, 700),
    (1350, 1600),
    (2550, 2700),
]

peaks = {
    "gamma_59p9keV": 59.9,
    "gamma_99keV": 99.0,
    "gamma_103keV": 103.0,
    "gamma_123p1keV": 123.1,
    "gamma_125p3keV": 125.3,
    "gamma_208p1keV": 208.1,
    "gamma_335p4keV": 335.4,
    "gamma_662p4keV": 662.4,
    "gamma_2614keV": 2615,
    "gamma_1460p8keV": 1460.8,
}

spectrum_colors = ["navy", "deepskyblue", "lightseagreen"]
peak_color = "darkorange"

spectrum_dir = base_dir / "raw_energy_spectrum"
plot_dir = (
    base_dir
    / "detector_run_plots"
    / version
    / campaign
    / measure
)

force_rebuild = False
overwrite_plots = True
dpi = 200


# ============================================================
# METADATI E FILE
# ============================================================

def get_run_names(configuration):
    return sorted(
        key
        for key in configuration
        if key.startswith("run") and key[3:].isdigit()
    )


def get_run_files(detector, measurement, run, version, campaign):
    run_number = int(run[3:])

    pattern = (
        "/global/cfs/cdirs/m2676/data/teststands/hades/"
        f"prodenv/ref/{version}/generated/tier/hit/"
        f"{detector}/{campaign}/{measurement}/"
        f"char_data-{detector}-{measurement}"
        f"-r{run_number:03d}-*-tier_hit.lh5"
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Nessun file per {detector}, {measurement}, {run}"
        )

    return files


# ============================================================
# LETTURA O CARICAMENTO DELL'ISTOGRAMMA
# ============================================================

def read_run_histogram(
    detector,
    measurement,
    run,
    meta,
    bins,
    version,
    campaign,
    spectrum_dir,
    force_rebuild=False,
):
    time_s = float(meta["run_live_time_in_s"])

    if not np.isfinite(time_s) or time_s <= 0:
        raise ValueError(
            f"Live time non valido per {detector}, {run}: {time_s}"
        )

    cache_dir = (
        Path(spectrum_dir) / version / campaign / detector
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    spectrum_path = cache_dir / (
        f"{detector}_{measurement}_{run}_spectrum.npz"
    )

    counts = None

    if spectrum_path.exists() and not force_rebuild:
        try:
            with np.load(
                spectrum_path, allow_pickle=False
            ) as spectrum:
                saved_bins = spectrum["bins"]
                saved_counts = spectrum["counts"]

                if (
                    np.array_equal(saved_bins, bins)
                    and saved_counts.shape == (len(bins) - 1,)
                ):
                    counts = saved_counts.copy()

        except (OSError, ValueError, KeyError, EOFError):
            tqdm.write(f"Cache non leggibile: {spectrum_path}")

        if counts is not None:
            tqdm.write(f"Carico: {spectrum_path}")

    if counts is None:
        files = get_run_files(
            detector, measurement, run, version, campaign
        )

        tqdm.write(f"Creo: {spectrum_path}")

        counts = np.zeros(len(bins) - 1, dtype=np.int64)

        for filename in tqdm(
            files,
            desc=f"{detector} | {run}",
            unit="file",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            obj = lh5.read("hit/cuspEmax_ctc_cal", filename)

            if isinstance(obj, tuple):
                obj = obj[0]

            energy = np.asarray(obj.nda)
            finite_energy = energy[np.isfinite(energy)]

            counts += np.histogram(
                finite_energy, bins=bins
            )[0]

            del obj, energy, finite_energy

        temporary_path = spectrum_path.with_suffix(".tmp.npz")

        np.savez_compressed(
            temporary_path,
            counts=counts,
            bins=bins,
        )
        temporary_path.replace(spectrum_path)

    return {
        "counts": counts,
        "time_s": time_s,
        "source_position": dict(meta["source_position"]),
        "card_interface": (
            meta["daq_settings"]["flashcam"]["card_interface"]
        ),
    }


# ============================================================
# PREPARAZIONE DI TUTTI I RUN DEL DETECTOR
# ============================================================

def prepare_detector_runs(
    db,
    detector,
    measurement,
    bins,
    version,
    campaign,
    spectrum_dir,
    force_rebuild=False,
):
    configuration = (
        db.hardware.configuration[detector]
        [campaign][measurement]
    )

    runs = get_run_names(configuration)

    if not runs:
        raise ValueError(
            f"Nessun run per {detector}, {measurement}"
        )

    run_data = {}

    for run in runs:
        meta = (
            db.hardware.configuration[detector]
            [campaign][measurement][run]
        )

        run_data[run] = read_run_histogram(
            detector=detector,
            measurement=measurement,
            run=run,
            meta=meta,
            bins=bins,
            version=version,
            campaign=campaign,
            spectrum_dir=spectrum_dir,
            force_rebuild=force_rebuild,
        )

    return run_data


# ============================================================
# LABEL DELLA LEGENDA
# ============================================================

def make_run_label(run, data):
    position = data["source_position"]

    return (
        f"{run}\n"
        f"r = {float(position['r_in_mm']):g} mm, "
        f"z = {float(position['z_in_mm']):g} mm, "
        f"φ = {float(position['phi_in_deg']):g}°\n"
        f"Live time = {data['time_s']:g} s\n"
        f"Card = {data['card_interface']}"
    )


# ============================================================
# FIGURA CON QUATTRO SUBPLOT
# ============================================================
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot_detector_runs(
    detector,
    measurement,
    run_data,
    bins,
    energy_ranges,
    peaks,
    spectrum_colors,
    peak_color,
):
    if len(energy_ranges) != 5:
        raise ValueError("Servono cinque intervalli energetici.")

    if not run_data:
        raise ValueError("Nessun run da rappresentare.")

    figure_height = max(12, 1.15 * len(run_data) + 1.5)

    fig = plt.figure(
        figsize=(16, figure_height),
        layout="constrained",
    )

    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1, 1, 0.65],
        wspace=0.12,
        hspace=0.12,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0:2]),
    ]

    # Legenda esterna ai cinque grafici.
    ax_legend = fig.add_subplot(grid[:, 2])
    ax_legend.axis("off")

    fig.suptitle(
        f"{detector} — {measurement}",
        fontsize=16,
        fontweight="semibold",
    )

    widths = np.diff(bins)
    ylabel = (
        f"Counts / ({widths[0]:.1f} keV s)"
        if np.allclose(widths, widths[0])
        else "Counts / (bin s)"
    )

    legend_handles = []
    legend_labels = []
    legend_colors = []
    rates = []

    for index, (run, data) in enumerate(sorted(run_data.items())):
        color = spectrum_colors[index % len(spectrum_colors)]

        linestyle = ["-", "--", "-.", ":"][
            (index // len(spectrum_colors)) % 4
        ]

        rate = data["counts"] / data["time_s"]
        rates.append(rate)

        for ax in axes:
            ax.stairs(
                rate,
                bins,
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                zorder=3,
            )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
        )
        legend_labels.append(make_run_label(run, data))
        legend_colors.append(color)

    for ax, (xmin, xmax) in zip(axes, energy_ranges):
        ax.set_xlim(xmin, xmax)
        ax.set_yscale("log")

        ax.set_xlabel("Energy (keV)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{xmin:g}–{xmax:g} keV",
            fontsize=12,
            pad=12,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            direction="out",
            length=5,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3,
        )

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            which="major",
            color="0.88",
            linewidth=0.6,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Bin che intersecano il range visibile.
        visible = (
            (bins[:-1] < xmax)
            & (bins[1:] > xmin)
        )

        positive_parts = [
            rate[visible & (rate > 0)]
            for rate in rates
        ]
        positive_parts = [
            values for values in positive_parts
            if values.size
        ]

        if positive_parts:
            positive = np.concatenate(positive_parts)

            # Margine superiore per le etichette dei picchi.
            ax.set_ylim(
                positive.min() / 1.5,
                positive.max() * 5,
            )

        # Un solo ciclo sui picchi.
        for energy in sorted(set(peaks.values())):
            if not xmin <= energy <= xmax:
                continue

            ax.axvline(
                energy,
                color=peak_color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
                zorder=2,
            )

            # Offset in punti: non sposta l'energia della linea.
            ax.annotate(
                f"{energy:g} keV",
                xy=(energy, 0.98),
                xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=8,
                color=peak_color,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=1,
                ),
                zorder=5,
            )

    legend_handles.append(
        Line2D(
            [0], [0],
            color=peak_color,
            linestyle="--",
            linewidth=1.2,
        )
    )
    legend_labels.append("Picchi gamma")
    legend_colors.append(peak_color)

    legend = ax_legend.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        frameon=False,
        fontsize=10,
        labelspacing=1.8,
        handlelength=2.5,
        handletextpad=0.8,
        borderaxespad=0,
    )

    for text, color in zip(legend.get_texts(), legend_colors):
        text.set_color(color)

    return fig


# ============================================================
# ESECUZIONE E SALVATAGGIO
# ============================================================

def main():
    db = dbetto.TextDB(metadata_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for detector in tqdm(
        detectors,
        desc="Detector",
        unit="detector",
        position=0,
        dynamic_ncols=True,
    ):
        plot_path = plot_dir / (
            f"{detector}_{measure}.png"
        )

        if (
            plot_path.exists()
            and not overwrite_plots
            and not force_rebuild
        ):
            tqdm.write(f"Già presente: {plot_path}")
            results.append((detector, "already_saved", ""))
            continue

        figures_before = set(plt.get_fignums())
        run_data = None
        fig = None

        try:
            run_data = prepare_detector_runs(
                db=db,
                detector=detector,
                measurement=measure,
                bins=bins,
                version=version,
                campaign=campaign,
                spectrum_dir=spectrum_dir,
                force_rebuild=force_rebuild,
            )

            fig = plot_detector_runs(
                detector=detector,
                measurement=measure,
                run_data=run_data,
                bins=bins,
                energy_ranges=energy_ranges,
                peaks=peaks,
                spectrum_colors=spectrum_colors,
                peak_color=peak_color,
            )

            fig.savefig(
                plot_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )

            tqdm.write(f"Salvato: {plot_path}")
            results.append((detector, "saved", ""))

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"ERRORE {detector}: {error}")
            results.append((detector, "error", error))

        finally:
            for number in (
                set(plt.get_fignums()) - figures_before
            ):
                plt.close(number)

            run_data = None
            fig = None

    # Riepilogo su file.
    import csv

    summary_path = plot_dir / "summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["detector", "status", "error"])
        writer.writerows(results)

    tqdm.write(f"\nPlot salvati in: {plot_dir}")

    for status in ("saved", "already_saved", "error"):
        number = sum(row[1] == status for row in results)
        tqdm.write(f"{status}: {number}")


if __name__ == "__main__":
    main()