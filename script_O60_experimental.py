"""Per-run Am fits and O60 = N60 / (N99 + N103).
Run on NERSC: pixi run python script_O60_experimental.py
N60 is the Gaussian area only; the fitted left tail is saved separately.
Poisson fits require raw, unweighted, non-background-subtracted counts.
"""
from pathlib import Path
import csv
import json
import re
import dbetto
import lh5
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf, erfc, xlogy
from scipy.stats import exponnorm

version = "v1.2.1"
campaign = "c1"
measure = "am_HS1_top_dlt"
metadata_path = "/global/u2/r/ritaferi/HADES_DATA/hades-metadata"
base_dir = Path(metadata_path).parent
# None: discover every detector with data or cached spectra.
#detectors = None
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
    #"V10784A",
    "V11924A",
    "V11925A",
    "V11947B",
    "V13044A",
    "V13049A",
    "V14618A",
    "V14654A",
    "V14673A",
]



fit_range60 = (54.0, 63.0)
fit_range100 = (96.0, 106.0)
histogram_bins = np.linspace(0, 3500, 35001)
dpi = 300
force_rebuild = False
spectrum_dir = base_dir / "raw_energy_spectrum"
output_dir = base_dir / "O60_experimental" / version / campaign / measure
hit_root = (Path("/global/cfs/cdirs/m2676/data/teststands/hades/prodenv/ref")
            / version / "generated/tier/hit")
names60 = ["N60", "mu60", "sigma60", "N60_tail", "tau60", "B60", "S60"]
names100 = ["N99", "mu99", "N101", "mu101", "N103", "mu103",
            "sigma100", "S99", "S103", "B100"]

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

    xmin, xmax = fit_range60

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


    height = max(density[np.argmin(np.abs(centers - 59.54))] - background, 1)
    area_guess = height * np.sqrt(2 * np.pi) * 0.4
    p0 = [area_guess, 59.54, 0.4, area_guess * 0.1, 1.0, background, background * 0.5]
    lower = [0, 58.8, 0.05, 0, 0.05, 0, 0]
    upper = [np.inf, 60.3, 1.2, np.inf, 10, np.inf, np.inf]

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



# Three Gaussians with shared sigma; two low-energy steps at mu99 and mu103.
def am100_components(edges, p):
    n1, m1, n2, m2, n3, m3, sigma, s1, s2, background = p
    components = {
        "gaussian_99": gaussian_bin_counts(edges, n1, m1, sigma),
        "gaussian_101": gaussian_bin_counts(edges, n2, m2, sigma),
        "gaussian_103": gaussian_bin_counts(edges, n3, m3, sigma),
        "step_99": s1 * np.maximum(np.diff(step_primitive(edges, m1, sigma)), 0),
        "step_103": s2 * np.maximum(np.diff(step_primitive(edges, m3, sigma)), 0),
        "background": background * np.diff(edges),
    }
    components["total"] = sum(components.values())
    return components


def fit_am100(counts, bins):
    counts, bins = np.asarray(counts, float), np.asarray(bins, float)
    if (counts.ndim != 1 or bins.ndim != 1 or bins.size != counts.size + 1
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0)):
        raise ValueError("Istogramma non valido")
    idx = np.flatnonzero((bins[:-1] >= fit_range100[0]) &
                         (bins[1:] <= fit_range100[1]))
    if idx.size <= 10:
        raise ValueError("Troppi pochi bin nella finestra 100 keV")
    edges = bins[idx[0]:idx[-1]+2]
    y = counts[idx[0]:idx[-1]+1]
    if y.sum() <= 0:
        raise ValueError("Nessun conteggio nella finestra 100 keV")
    centers = (edges[:-1] + edges[1:]) / 2
    density = y / np.diff(edges)
    background = max(float(np.percentile(density, 10)), 1e-3)
    def guess(mu):
        return max(density[np.argmin(abs(centers-mu))]-background, 1) * np.sqrt(2*np.pi)*0.4
    p0 = [guess(99), 99, guess(101.1), 101.1, guess(103), 103,
          0.4, background/2, background/2, background]
    lower = [0, 98.5, 0, 100.6, 0, 102.5, 0.05, 0, 0, 0]
    upper = [np.inf, 99.5, np.inf, 101.6, np.inf, 103.5, 1.2,
             np.inf, np.inf, np.inf]
    def residuals(p):
        expected = np.maximum(am100_components(edges, p)["total"], 1e-12)
        d = 2 * (expected-y+xlogy(y, y/expected))
        return np.sign(expected-y)*np.sqrt(np.maximum(d, 0))
    fit = least_squares(residuals, p0, bounds=(lower, upper),
                        x_scale="jac", max_nfev=20000)
    if not fit.success:
        raise RuntimeError(f"Fit 100 keV fallito: {fit.message}")
    full_rank = np.linalg.matrix_rank(fit.jac) == 10
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) if full_rank else np.full((10, 10), np.nan)
    errors = np.sqrt(np.maximum(np.diag(cov), 0))
    return dict(parameters=fit.x, errors=errors, covariance=cov, edges=edges,
                counts=y, components=am100_components(edges, fit.x),
                residuals=fit.fun, deviance=float(sum(fit.fun**2)), ndof=y.size-10,
                full_rank=full_rank, at_bounds=bool(np.any(fit.active_mask)),
                active_mask=fit.active_mask, nfev=fit.nfev, message=fit.message,
                fwhm=2.35482*fit.x[6], fwhm_error=2.35482*errors[6])


def discover_detectors():
    found = set()
    if hit_root.exists():
        found.update(d.name for d in hit_root.iterdir()
                     if d.is_dir() and (d/campaign/measure).is_dir())
    cached = spectrum_dir/version/campaign
    if cached.exists():
        found.update(d.name for d in cached.iterdir()
                     if d.is_dir() and any(d.glob(f"*_{measure}_run*_spectrum.npz")))
    return sorted(found) if detectors is None else list(dict.fromkeys(detectors))


def discover_runs(detector, db):
    available = set()
    folder = hit_root/detector/campaign/measure
    for path in folder.glob(f"char_data-{detector}-{measure}-r*-*-tier_hit.lh5"):
        match = re.search(r"-r(\d+)-", path.name)
        if match:
            available.add(f"run{int(match[1]):04d}")
    for path in (spectrum_dir/version/campaign/detector).glob(f"*_{measure}_run*_spectrum.npz"):
        match = re.search(r"_(run\d+)_spectrum", path.name)
        if match:
            available.add(match[1])
    # Also attempt runs declared in metadata, even if the data are unavailable.
    try:
        available.update(k for k in db.hardware.configuration[detector][campaign][measure]
                         if re.fullmatch(r"run\d+", k))
    except (KeyError, AttributeError, TypeError, FileNotFoundError):
        pass
    return sorted(available, key=lambda r: int(r[3:]))


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fit_summary(result, names):
    return {
        "parameters": {name: {"value": float(v), "error": float(e)}
                       for name, v, e in zip(names, result["parameters"], result["errors"])},
        "parameter_order": names,
        "covariance": result["covariance"],
        "fit_range_keV": [result["edges"][0], result["edges"][-1]],
        **{key: result[key] for key in ["fwhm", "fwhm_error", "deviance", "ndof",
           "full_rank", "at_bounds", "active_mask", "nfev", "message"]},
    }


def save_fit_npz(path, result, names, detector, run):
    np.savez_compressed(path, detector=detector, run=run, parameter_names=np.array(names),
                        **{k: result[k] for k in ["parameters", "errors", "covariance",
                          "edges", "counts", "residuals", "active_mask"]},
                        **{f"model_{k}": v for k, v in result["components"].items()})


def plot_two_regions(results, detector, run, meta, ratio):
    # Exactly two subplots, one for each energy window.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained", dpi=dpi)
    colors = {"total": "darkorange", "gaussian": "crimson", "tail": "green",
              "continuum": "gray", "gaussian_99": "crimson", "gaussian_101": "deepskyblue",
              "gaussian_103": "green", "step_99": "purple", "step_103": "brown", "background": "gray"}
    for ax, (region, result) in zip(axes, results.items()):
        ax.set_title(f"{region} keV", fontsize=9)
        ax.set_xlabel("Energy (keV)", fontsize=8)
        ax.tick_params(labelsize=7)
        if result is None:
            ax.text(.5, .5, "Fit unavailable: see results.json", ha="center", transform=ax.transAxes)
            continue
        edges = result["edges"]
        widths = np.diff(edges)
        ax.stairs(result["counts"], edges, color="navy", linewidth=.8, label="Data")
        # Plot integrated bin counts, matching the fitted model exactly.
        for key, values in result["components"].items():
            ax.stairs(np.where(values > 0, values, np.nan), edges, color=colors[key],
                      linestyle="-" if key == "total" else "--", linewidth=.9,
                      label=key.replace("_", " "))
        ax.set_yscale("log")
        positive = result["counts"][result["counts"] > 0]
        ax.set_ylim(max(.1, min(.5, positive.min()/2)),
                    max(result["counts"].max(), result["components"]["total"].max())*3)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel(f"Counts / {widths[0]:g} keV", fontsize=8)
        indices = [0] if region == "60" else [0, 2, 4]
        labels = ["N60"] if region == "60" else ["N99", "N101", "N103"]
        text = "\n".join(f"{name} = {result['parameters'][i]:.0f} ± {result['errors'][i]:.0f}"
                         for name, i in zip(labels, indices))
        text += f"\nFWHM = {result['fwhm']:.3f} ± {result['fwhm_error']:.3f} keV"
        text += f"\nD/ndof = {result['deviance']:.1f}/{result['ndof']}"
        ax.text(.98, .97, text, transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.legend(fontsize=6, loc="upper left", frameon=False)
    title = f"{detector}/{run}"
    pos = meta.get("source_position", {})
    if pos:
        title += f" | r={pos.get('r_in_mm')} mm, z={pos.get('z_in_mm')} mm, φ={pos.get('phi_in_deg')}°"
    if ratio["value"] is not None:
        error = ratio["error"]
        title += f" | N60/(N99+N103)={ratio['value']:.4f}"
        if error is not None:
            title += f" ± {error:.4f}"
    fig.suptitle(title, fontsize=9)
    return fig


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    db = dbetto.TextDB(metadata_path)
    all_results, rows = {}, []
    detector_list = discover_detectors()
    if not detector_list:
        raise RuntimeError("Nessun detector trovato: controllare version e percorsi")
    for detector in detector_list:
        all_results[detector] = {}
        for run in discover_runs(detector, db):
            print(f"\n{detector}/{run}", flush=True)
            directory = output_dir/detector/run
            directory.mkdir(parents=True, exist_ok=True)
            record = dict(detector=detector, run=run, version=version,
                          campaign=campaign, measurement=measure, errors=[])
            fits = {"60": None, "100": None}
            try:
                meta = db.hardware.configuration[detector][campaign][measure][run]
            except (KeyError, AttributeError, TypeError, FileNotFoundError) as exc:
                meta = {}
                record["errors"].append(f"Metadata unavailable: {exc}")
            record["metadata"] = {
                "source_position": dict(meta.get("source_position", {})),
                "live_time_s": meta.get("run_live_time_in_s"),
                "high_voltage_V": meta.get("high_voltage_in_V"),
            }
            try:
                counts, bins = load_histogram(detector, run)
                for region, fit_function, names in [("60", fit_am60, names60), ("100", fit_am100, names100)]:
                    try:
                        result = fit_function(counts, bins)
                        fits[region] = result
                        record[f"fit_{region}"] = fit_summary(result, names)
                        save_fit_npz(directory/f"fit_{region}.npz", result, names, detector, run)
                    except Exception as exc:
                        record[f"fit_{region}"] = None
                        record["errors"].append(f"Fit {region}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                record["errors"].append(f"Histogram: {type(exc).__name__}: {exc}")
            ratio = dict(value=None, error=None, definition="N60 / (N99 + N103)",
                         uncertainty_method="Delta method; includes Cov(N99,N103); disjoint fit windows")
            if all(v is not None for v in fits.values()):
                r60, r100 = fits["60"], fits["100"]
                n60 = r60["parameters"][0]
                denominator = r100["parameters"][0]+r100["parameters"][4]
                if denominator > 0:
                    ratio["value"] = float(n60/denominator)
                    cov = r100["covariance"]
                    var_d = cov[0, 0]+cov[4, 4]+2*cov[0, 4]
                    variance = r60["covariance"][0, 0]/denominator**2 + n60**2*var_d/denominator**4
                    if np.isfinite(variance) and variance >= 0:
                        ratio["error"] = float(np.sqrt(variance))
                ratio["check_fit"] = any(r["at_bounds"] or not r["full_rank"] for r in fits.values())
            record["ratio"] = ratio
            record["status"] = "ok" if all(v is not None for v in fits.values()) else "incomplete"
            all_results[detector][run] = record
            save_json(directory/"results.json", record)
            row = dict(detector=detector, run=run, status=record["status"],
                       N60_over_N99_plus_N103=ratio["value"], ratio_error=ratio["error"])
            for region, names in [("60", names60), ("100", names100)]:
                result = fits[region]
                for i, name in enumerate(names):
                    row[name] = float(result["parameters"][i]) if result is not None else None
                    row[name+"_error"] = float(result["errors"][i]) if result is not None else None
                for key in ["fwhm", "fwhm_error", "deviance", "ndof", "full_rank", "at_bounds"]:
                    row[f"{key}_{region}"] = result[key] if result is not None else None
            row["errors"] = " | ".join(record["errors"])
            rows.append(row)
            if any(v is not None for v in fits.values()):
                fig = plot_two_regions(fits, detector, run, meta, ratio)
                try:
                    fig.savefig(directory/"fits_60_100.png", dpi=dpi, facecolor="white")
                    fig.savefig(directory/"fits_60_100.pdf", facecolor="white")
                finally:
                    plt.close(fig)
            # Checkpoint after every run.
            save_json(output_dir/"all_results.json", all_results)
            with (output_dir/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"ratio={ratio['value']} ± {ratio['error']} | {record['status']}", flush=True)
            for error in record["errors"]:
                print(error, flush=True)
    print(f"\nResults saved in {output_dir}", flush=True)
    return all_results


if __name__ == "__main__":
    main()
