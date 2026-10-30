"""Per-run Am fits and O60 = N60 / (N99 + N103).
Run on NERSC: pixi run python script_O60_pygama_QC_counting_DLF_yaml_sigma.py
N60 is the Gaussian area only; the fitted left tail is saved separately.
Modern pygama least-squares fits use QC-selected unweighted counts.
Adaptation of legacy model; NOT an exact reproduction of legacy gauss_cdf.
Gaussian and low-tail PDFs come from pygama.math.distributions.
Areas are total Gaussian counts, not peak heights.
Least Squares evaluates models at bin centers and excludes empty bins.
Independent Gaussian sigmas; shared relative step S in 95–107 keV.
B is the right continuum density; each step height is B*S.
Bounds/initial values are adapted to the area parametrization.
"""
from pathlib import Path
import csv
import json
import yaml
import re
import dbetto
import lh5
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygama.math import binned_fitting as pgb
from pygama.math import distributions as pgf
from scipy.stats import chi2
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



fit_range60 = (57.0, 62.0)
fit_range100 = (97.0, 105.0)
histogram_bins = np.linspace(0, 3500, 35001)
dpi = 300
force_rebuild = False
spectrum_dir = base_dir / "raw_energy_spectrum"
output_dir = base_dir / "O60_experimental_pygama" / version / campaign / measure
hit_root = (Path("/global/cfs/cdirs/m2676/data/teststands/hades/prodenv/ref")
            / version / "generated/tier/hit")
names60 = ["N60", "mu60", "sigma60", "N60_tail", "tau60", "B60", "S60"]
names100 = ["N99", "mu99", "sigma99", "N103", "mu103", "sigma103",
            "N101", "mu101", "sigma101", "B100", "S100"]

QC_SELECTION = '~is_discharge & ((is_positive_polarity_candidate == 127) | (is_positive_polarity_candidate == 63)); is_low_cuspEmax ignored; no additional energy selection'

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
                    spectrum["counts_clean"], dtype=float
                ).copy()

                bins = np.asarray(
                    spectrum["bins"], dtype=float
                ).copy()
                saved_selection = (
                    str(spectrum["qc_selection"].item())
                    if "qc_selection" in spectrum.files else None
                )

            valid = (
                saved_selection == QC_SELECTION
                and
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
                print(f"Carico counts_clean (QC): {path}", flush=True)
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

    counts_clean = np.zeros_like(counts)

    total_events = 0
    selected_events = 0

    # Read one file at a time; sum all files of the run.
    for index, filename in enumerate(files, start=1):
        print(
            f"{detector}/{run}: file {index}/{len(files)}",
            flush=True,
        )

        hit = lh5.read("hit", str(filename))
        energy = np.asarray(hit["cuspEmax_ctc_cal"].nda, dtype=float)
        discharge_raw = np.asarray(hit["is_discharge"].nda)
        positive_bits = np.asarray(hit["is_positive_polarity_candidate"].nda)
        if (energy.ndim != 1
                or discharge_raw.shape != energy.shape
                or positive_bits.shape != energy.shape):
            raise ValueError(f"Energy/QC flag shapes do not match: {filename}")
        if not np.isin(discharge_raw, [0, 1]).all():
            raise ValueError(f"is_discharge is not boolean: {filename}")
        if (not np.issubdtype(positive_bits.dtype, np.integer)
                or np.any(positive_bits < 0) or np.any(positive_bits > 255)):
            raise ValueError(f"Invalid packed 8-bit QC field: {filename}")

        is_discharge = discharge_raw.astype(bool)
        mask = (~is_discharge) & ((positive_bits == 127) | (positive_bits == 63))
        partial_counts = np.histogram(energy, bins=histogram_bins)[0]
        partial_clean = np.histogram(energy[mask], bins=histogram_bins)[0]
        counts += partial_counts
        counts_clean += partial_clean
        total_events += energy.size
        selected_events += int(mask.sum())
        print(f"  QC selected (all energies): {mask.sum():,}/{energy.size:,}", flush=True)
        del hit, energy, discharge_raw, positive_bits, is_discharge, mask
        del partial_counts, partial_clean

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")

    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                counts=counts,
                counts_clean=counts_clean,
                qc_selection=QC_SELECTION,
                total_events=total_events,
                selected_events=selected_events,
                bins=histogram_bins,
            )

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"Salvato: {path}\n"
        f"Eventi totali: {total_events:,}\n"
        f"Eventi selezionati (tutte le energie): {selected_events:,}\n"
        f"Conteggi QC nei bin: {counts_clean.sum():,}",
        flush=True,
    )

    return counts_clean.astype(float), histogram_bins.copy()


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


# Resolution metadata used ONLY by the counting method.
# This is intentionally the calibration measurement requested by the user,
# even when the source spectrum belongs to am_HS1_top_dlt.
resolution_measure = "am_HS1_lat_dlt"
resolution_energy_parameter = "cuspEmax_ctc_cal"
resolution_cache = {}
counting_nsigma60 = 3.0
counting_nsigma100 = 3.0


def load_counting_resolution(detector):
    if detector in resolution_cache:
        return resolution_cache[detector]
    folder = (Path("/global/cfs/cdirs/m2676/data/teststands/hades/prodenv/ref")
              / version / "generated/par/hit" / detector / campaign / resolution_measure)
    files = sorted(folder.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No resolution YAML found: {folder}/*.yaml")
    # Same selection as the supplied notebook: first file in sorted order.
    selected = files[0]
    if len(files) > 1:
        print(f"{detector}: {len(files)} resolution YAMLs; using first sorted file: {selected}", flush=True)
    with selected.open(encoding="utf-8") as stream:
        par = yaml.safe_load(stream)
    try:
        resolution = par["results"]["ecal"][resolution_energy_parameter]["eres_quadratic"]
        pars = resolution["parameters"]
        coefficients = {name: float(pars[name]) for name in ("a", "b", "c")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid eres_quadratic parameters in {selected}: {exc}") from exc
    if not np.isfinite(list(coefficients.values())).all():
        raise ValueError(f"Non-finite resolution coefficients: {selected}")
    result = dict(file=str(selected), available_files=[str(f) for f in files],
                  file_selection="first file in sorted order", measurement=resolution_measure,
                  energy_parameter=resolution_energy_parameter,
                  parameters=coefficients, fwhm_to_sigma=2.355,
                  expression="sqrt(a + b*E + c*E**2) / 2.355",
                  uncertainty_note="Resolution parameters and window boundaries treated as fixed")
    # Validate all relevant energies before adding the entry to the cache.
    for energy in (59.5, 99.0, 103.0):
        extract_sigma(energy, result)
    resolution_cache[detector] = result
    print(f"{detector}: resolution YAML: {selected}", flush=True)
    return result


def extract_sigma(energy_keV, resolution):
    energy = float(energy_keV)
    a, b, c = (resolution["parameters"][name] for name in ("a", "b", "c"))
    variance_fwhm = a + b*energy + c*energy**2
    if not np.isfinite(variance_fwhm) or variance_fwhm <= 0:
        raise ValueError(f"Invalid FWHM squared at {energy:g} keV: {variance_fwhm}")
    return float(np.sqrt(variance_fwhm)/2.355)


def counting_window(bins, low, high):
    """Include every bin intersecting the requested interval; use actual edges."""
    tol = 1e-9
    mask = (bins[:-1] < high - tol) & (bins[1:] > low + tol)
    indices = np.flatnonzero(mask)
    if not indices.size or low < bins[0] or high > bins[-1]:
        raise ValueError(f"Window outside histogram: {low}, {high}")
    return mask, {
        "requested_range_keV": [float(low), float(high)],
        "actual_range_keV": [float(bins[indices[0]]), float(bins[indices[-1]+1])],
        "width_keV": float(np.diff(bins)[mask].sum()),
        "n_bins": int(mask.sum()),
    }


def count_region(counts, bins, peaks, sidebands, sigmas, nsigma):
    counts, bins = np.asarray(counts, float), np.asarray(bins, float)
    if (counts.ndim != 1 or bins.ndim != 1 or bins.size != counts.size + 1
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0)
            or not np.isfinite(nsigma) or nsigma <= 0):
        raise ValueError("Invalid counting histogram or sigma")
    masks, windows = {}, {}
    for name, interval in sidebands.items():
        masks[name], windows[name] = counting_window(bins, *interval)
    for name, center in peaks.items():
        sigma = float(sigmas[name])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid sigma for {name}: {sigma}")
        masks[name], windows[name] = counting_window(
            bins, center-nsigma*sigma, center+nsigma*sigma)
        windows[name]["energy_keV"] = float(center)
        windows[name]["sigma_keV"] = sigma
    # Disjoint windows prevent peak counts being reused as background.
    occupancy = np.sum(list(masks.values()), axis=0)
    if np.any(occupancy > 1):
        raise ValueError("Counting windows overlap: reduce sigma or change windows")
    for name in windows:
        windows[name]["counts"] = float(counts[masks[name]].sum())
    side_mask = masks["left"] | masks["right"]
    side_width = windows["left"]["width_keV"] + windows["right"]["width_keV"]
    side_counts = float(counts[side_mask].sum())
    weights, net = [], {}
    for name in peaks:
        scale = windows[name]["width_keV"] / side_width
        w = masks[name].astype(float) - scale*side_mask
        weights.append(w)
        net[name] = {
            "value": float(w @ counts),
            "error": float(np.sqrt((w*w) @ counts)),
            "gross_counts": windows[name]["counts"],
            "background_counts": float(scale*side_counts),
            "background_scale": float(scale),
        }
    weights = np.asarray(weights)
    covariance = (weights*counts) @ weights.T
    sum_weights = weights.sum(axis=0)
    result = {
        "sigma_keV": {name: float(sigmas[name]) for name in peaks},
        "sigma_source": "detector YAML eres_quadratic",
        "n_sigma": float(nsigma), "windows": windows,
        "background_density_counts_per_keV": side_counts/side_width,
        "peaks": net, "peak_order": list(peaks), "covariance": covariance,
        "sum": {"value": float(sum_weights @ counts),
                "error": float(np.sqrt((sum_weights**2) @ counts))},
        "assumption": "Locally constant continuum; independent Poisson bins; fixed windows",
        "gaussian_acceptance_corrected": False,
    }
    return result


def counting_method60(counts, bins, resolution):
    sigmas = {"N60": extract_sigma(59.5, resolution)}
    result = count_region(counts, bins, {"N60": 59.5},
                          {"left": (56.5, 57.5), "right": (61.5, 62.5)},
                          sigmas, counting_nsigma60)
    result["resolution_metadata"] = resolution
    return result


def counting_method100(counts, bins, resolution):
    sigmas = {"N99": extract_sigma(99.0, resolution),
              "N103": extract_sigma(103.0, resolution)}
    result = count_region(counts, bins, {"N99": 99.0, "N103": 103.0},
                          {"left": (94.0, 96.0), "right": (105.0, 107.0)},
                          sigmas, counting_nsigma100)
    result["resolution_metadata"] = resolution
    return result


def counting_ratio(r60, r100):
    n, en = r60["peaks"]["N60"]["value"], r60["peaks"]["N60"]["error"]
    d, ed = r100["sum"]["value"], r100["sum"]["error"]
    result = dict(value=None, error=None, definition="N60 / (N99 + N103)",
                  uncertainty_method="Poisson delta method; shared-sideband covariance included; fixed windows",
                  denominator=d, denominator_error=ed,
                  check_counts=bool(n <= 0 or d <= 0))
    if d > 0:
        result.update(value=float(n/d), error=float(np.sqrt(en**2/d**2+n**2*ed**2/d**4)))
    return result


def plot_counting_regions(counts, bins, counting, detector, run, meta):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=dpi, layout="constrained")
    colors = {"left": "lightgreen", "right": "green", "N60": "lightcoral",
              "N99": "lightcoral", "N103": "salmon"}
    labels = {"left": "Left sideband", "right": "Right sideband",
              "N60": "59.5 keV window", "N99": "99 keV window", "N103": "103 keV window"}
    for ax, region, limits in zip(axes, ("60", "100"), ((56, 64), (93, 108))):
        r = counting[region]
        ax.stairs(counts, bins, color="navy", linewidth=.8, label="Data (QC)", zorder=10)
        for name, window in r["windows"].items():
            low, high = window["actual_range_keV"]
            mask = (bins[:-1] >= low-1e-9) & (bins[1:] <= high+1e-9)
            idx = np.flatnonzero(mask)
            ax.stairs(counts[idx], bins[idx[0]:idx[-1]+2], fill=True,
                      color=colors[name], alpha=.5,
                      label=f"{labels[name]}: {window['counts']:.0f}")
        visible = (bins[:-1] < limits[1]) & (bins[1:] > limits[0])
        positive = counts[visible & (counts > 0)]
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(.5, positive.min()/2) if positive.size else .5,
                    top=max(2., positive.max()*5) if positive.size else 10.)
        ax.set_xlim(*limits)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(f"Counts / {bins[1]-bins[0]:g} keV")
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        text = "\n".join(f"{name} net = {p['value']:.0f} ± {p['error']:.0f}"
                         for name, p in r["peaks"].items())
        ax.text(.98, .97, text, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    title = f"{detector}/{run} — QC — counting method"
    pos = meta.get("source_position", {})
    if pos:
        title += f"\nr={pos.get('r_in_mm')} mm, z={pos.get('z_in_mm')} mm, φ={pos.get('phi_in_deg')}°"
    ratio = counting["ratio"]
    if ratio["value"] is not None:
        title += f" | N60/(N99+N103)={ratio['value']:.4f} ± {ratio['error']:.4f}"
    fig.suptitle(title, fontsize=8)
    return fig


def step_density(x, mu, sigma, background, relative_step, xlo, xhi):
    """Low-energy step B*S*Phi((mu-x)/sigma), in counts/keV.

    B is the right continuum density and S is dimensionless.
    xlo/xhi are retained for compatibility with the density interface.
    """
    x = np.asarray(x, dtype=float)
    return background * relative_step * 0.5 * erfc(
        (x - mu) / (np.sqrt(2) * sigma)
    )


def density60(x, p, xlo, xhi):
    n, mu, sigma, nt, tau, background, step = p
    out = {
        "gaussian": n * pgf.gaussian.get_pdf(x, mu, sigma),
        # Positive tau in pygama.exgauss gives the LOW-energy tail.
        "tail": nt * pgf.exgauss.get_pdf(x, mu, sigma, tau),
        "continuum": background + step_density(x, mu, sigma, background, step, xlo, xhi),
    }
    out["total"] = sum(out.values())
    return out


def density100(x, p, xlo, xhi):
    n99, m99, s99, n103, m103, s103, n101, m101, s101, bg, step = p
    out = {
        "gaussian_99": n99 * pgf.gaussian.get_pdf(x, m99, s99),
        "gaussian_103": n103 * pgf.gaussian.get_pdf(x, m103, s103),
        "gaussian_101": n101 * pgf.gaussian.get_pdf(x, m101, s101),
        "step_99": step_density(x, m99, s99, bg, step, xlo, xhi),
        "step_103": step_density(x, m103, s103, bg, step, xlo, xhi),
        "background": np.full_like(x, bg),
    }
    out["total"] = sum(out.values())
    return out


def fit_region_pygama(counts, bins, region):
    """Modern pygama least squares with Poisson variances at bin centers.

    This is an adaptation of the legacy fit, not a bitwise reproduction.
    Areas N are total Gaussian counts; B is counts/keV and S is dimensionless.
    Each step has height B*S; right continuum B, far-left B*(1+n_steps*S).
    Empty bins are excluded by pygama's Least Squares implementation.
    """
    counts, bins = np.asarray(counts, float), np.asarray(bins, float)
    if (counts.ndim != 1 or bins.ndim != 1 or bins.size != counts.size+1
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0)):
        raise ValueError("Invalid histogram")
    fit_range = fit_range60 if region == "60" else fit_range100
    idx = np.flatnonzero((bins[:-1] >= fit_range[0]-1e-9) & (bins[1:] <= fit_range[1]+1e-9))
    if not idx.size:
        raise ValueError("No bins in fit range")
    edges = bins[idx[0]:idx[-1]+2]
    y = counts[idx[0]:idx[-1]+1]
    widths = np.diff(edges)
    if not np.allclose(widths, widths[0]):
        raise ValueError("Uniform bins are required for center-evaluated least squares")
    width = float(widths[0])
    xlo, xhi = float(edges[0]), float(edges[-1])
    centers = (edges[:-1]+edges[1:])/2
    background = max(float(np.percentile(y/width, 10)), 1e-3)
    def area_guess(mu, sigma):
        h = max(float(y[np.argmin(abs(centers-mu))]/width-background), 1.)
        return h*np.sqrt(2*np.pi)*sigma
    if region == "60":
        names = names60
        density = density60
        p0 = [area_guess(59.5, .3), 59.5, .3,
              area_guess(59.5, .3)*.1, 1., background, .5]
        limits = [(0., None), (58.8, 60.3), (.02, 1.5),
                  (0., None), (.02, 10.), (0., None), (0., None)]
        def model(x, N60, mu60, sigma60, N60_tail, tau60, B60, S60):
            return width*density60(x, (N60, mu60, sigma60, N60_tail, tau60, B60, S60), xlo, xhi)["total"]
        sigma_indices = {"60": 2}
    else:
        names = names100
        density = density100
        small_mu = 100.5 if current_fit_detector == "V08682A" else 101.
        p0 = [area_guess(99., .5), 99., .5,
              area_guess(103., .5), 103., .5,
              area_guess(small_mu, .5)*.1, small_mu, .5,
              background, .5]
        # Bounds keep the components identifiable. Shared relative step S.
        limits = [(0., None), (98.5, 99.5), (.02, 1.5),
                  (0., None), (102.5, 103.5), (.02, 1.5),
                  (0., None), (100., 101.6), (.02, 1.5),
                  (0., None), (0., None)]
        def model(x, N99, mu99, sigma99, N103, mu103, sigma103,
                  N101, mu101, sigma101, B100, S100):
            return width*density100(x, (N99, mu99, sigma99, N103, mu103, sigma103,
                                       N101, mu101, sigma101, B100, S100), xlo, xhi)["total"]
        sigma_indices = {"99": 2, "103": 5, "101": 8}
    if np.count_nonzero(y > 0) <= len(p0):
        raise ValueError("Too few nonempty bins for least-squares fit")
    coeff, reported_errors, cov = pgb.fit_binned(
        func=model, hist=y, bins=edges, var=y, guess=np.asarray(p0),
        cost_func="Least Squares", extended=True,
        bounds={name: bound for name, bound in zip(names, limits)},
    )
    p = np.asarray(list(coeff), float)
    errors = np.asarray(list(reported_errors), float)
    if not np.isfinite(p).all() or not np.isfinite(model(centers, *p)).all():
        raise RuntimeError("Non-finite fit parameters or predictions")
    covariance = np.full((len(p), len(p)), np.nan) if cov is None else np.asarray(cov, float)
    expected = model(centers, *p)
    if np.any(expected < 0):
        raise RuntimeError("Negative expected counts")
    used = y > 0
    residuals = np.full(y.shape, np.nan)
    residuals[used] = (y[used]-expected[used])/np.sqrt(y[used])
    chisquare = float(np.sum(residuals[used]**2))
    ndof = int(used.sum()-len(p))
    active = np.zeros(len(p), int)
    for i, (lo, hi) in enumerate(limits):
        if lo is not None and np.isclose(p[i], lo, atol=1e-7, rtol=1e-5): active[i] = -1
        if hi is not None and np.isclose(p[i], hi, atol=1e-7, rtol=1e-5): active[i] = 1
    full_rank = bool(np.isfinite(covariance).all() and np.linalg.matrix_rank(covariance) == len(p))
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise RuntimeError("Pygama returned invalid parameter errors")
    if not full_rank:
        errors[:] = np.nan
    # Continuous plotting; expected counts in the fit use the same midpoint model.
    components = {k: v*width for k, v in density(centers, p, xlo, xhi).items()}
    safe_expected = np.maximum(expected, 1e-12)
    deviance = float(2*np.sum(safe_expected-y+xlogy(y, y/safe_expected)))
    fwhms = {label: dict(value=float(2.35482*p[i]), error=float(2.35482*errors[i]))
             for label, i in sigma_indices.items()}
    first = "60" if region == "60" else "99"
    return dict(parameters=p, errors=errors, covariance=covariance,
                edges=edges, counts=y, components=components,
                residuals=residuals, deviance=deviance, ndof=ndof,
                chi_square=chisquare, p_value=float(chi2.sf(chisquare, ndof)),
                full_rank=full_rank, at_bounds=bool(np.any(active)), active_mask=active,
                nfev=None, message="pygama fit_binned: Least Squares; wrapper does not expose Minuit convergence status",
                fwhm=fwhms[first]["value"], fwhm_error=fwhms[first]["error"],
                fwhm_reference=first, fwhm_peaks=fwhms,
                area_peak=float(p[0]), fit_method="pygama.math.binned_fitting.fit_binned / Least Squares")


current_fit_detector = None


def fit_am60(counts, bins):
    return fit_region_pygama(counts, bins, "60")


def fit_am100(counts, bins):
    return fit_region_pygama(counts, bins, "100")


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
        "continuum_parameterization": "B + B*S*sum(Phi((mu-E)/sigma)); B in counts/keV, S dimensionless",
        "fit_method": result["fit_method"],
        "chi_square": result["chi_square"],
        "p_value": result["p_value"],
        "fwhm_reference": result["fwhm_reference"],
        "fwhm_peaks": result["fwhm_peaks"],
        "covariance": result["covariance"],
        "fit_range_keV": [result["edges"][0], result["edges"][-1]],
        **{key: result[key] for key in ["fwhm", "fwhm_error", "deviance", "ndof",
           "full_rank", "at_bounds", "active_mask", "nfev", "message"]},
    }


def save_fit_npz(path, result, names, detector, run):
    np.savez_compressed(path, detector=detector, run=run, qc_applied=True, parameter_names=np.array(names),
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
        ax.stairs(result["counts"], edges, color="navy", linewidth=.8, label="Data (QC)")
        # Smooth curves evaluated with the SAME midpoint convention as the fit.
        smooth_x = np.linspace(edges[0], edges[-1], 3000)
        density = density60 if region == "60" else density100
        smooth = density(smooth_x, result["parameters"], edges[0], edges[-1])
        for key, values in result["components"].items():
            curve = smooth[key] * widths[0]
            ax.plot(smooth_x, np.where(curve > 0, curve, np.nan), color=colors[key],
                      linestyle="-" if key == "total" else "--", linewidth=.9,
                      label=key.replace("_", " "))
        ax.set_yscale("log")
        positive = result["counts"][result["counts"] > 0]
        ax.set_ylim(max(.1, min(.5, positive.min()/2)),
                    max(result["counts"].max(), result["components"]["total"].max())*3)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(100, 5e5)
        ax.set_ylabel(f"Counts / {widths[0]:g} keV", fontsize=8)
        indices = [0] if region == "60" else [0, 6, 3]
        labels = ["N60"] if region == "60" else ["N99", "N101", "N103"]
        text = "\n".join(f"{name} = {result['parameters'][i]:.0f} ± {result['errors'][i]:.0f}"
                         for name, i in zip(labels, indices))
        
        for peak, width_info in result["fwhm_peaks"].items():
            text += f"\nFWHM {peak} = {width_info['value']:.3f} ± {width_info['error']:.3f} keV"
        text += f"\nχ²/ndof = {result['chi_square']:.1f}/{result['ndof']}"
        ax.text(.98, .97, text, transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.legend(fontsize=6, loc="upper left", frameon=False)
    title = f"{detector}/{run} — QC"
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


# ============================================================
# SLOW PULSE OBSERVABLE: (35–40 minus scaled 80–85) / N60
# ============================================================

# ============================================================
# Actual DLF signal window remains 30–45 keV; 35–40 is shown for reference.
# ============================================================

def calculate_odlf(counts, bins, peak60):
    counts, bins = np.asarray(counts, float), np.asarray(bins, float)
    if (counts.ndim != 1 or bins.ndim != 1 or bins.size != counts.size+1
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0)):
        raise ValueError("Invalid histogram for O_DLF")
    centers = (bins[:-1]+bins[1:])/2
    widths = np.diff(bins)
    windows = {}
    for name, low, high in [
        ("slow_pulse", 35., 40.), ("background", 80., 85.),
        ("slow_pulse_inner", 35., 40.),
        ("background_alternative", 80., 85.),
    ]:
        mask = (centers >= low) & (centers < high)
        idx = np.flatnonzero(mask)
        if not idx.size or low < bins[0] or high > bins[-1]:
            raise ValueError(f"Window unavailable: {low}–{high} keV")
        number = float(counts[mask].sum())
        windows[name] = dict(
            requested_range_keV=[low, high],
            actual_range_keV=[float(bins[idx[0]]), float(bins[idx[-1]+1])],
            width_keV=float(widths[mask].sum()), counts=number,
            error=float(np.sqrt(number)),
        )
    signal, background = windows["slow_pulse"], windows["background"]
    scale = signal["width_keV"]/background["width_keV"]
    sp = signal["counts"]-scale*background["counts"]
    variance = signal["counts"]+scale**2*background["counts"]
    result = dict(
        definition="[N(35–40 keV) − scale*N(80–85 keV)] / N60_net_counting",
        windows=windows, background_scale=float(scale),
        SP_net=dict(value=float(sp), error=float(np.sqrt(variance))),
        N60_net=None, ratio=dict(value=None, error=None),
        uncertainty_method="Independent Poisson windows; delta method; fixed windows",
        check_counts=bool(sp <= 0),
    )
    if not peak60 or peak60.get("value") is None:
        return result
    n = float(peak60["value"])
    e = peak60.get("error")
    if not np.isfinite(n):
        return result
    valid_error = e is not None and np.isfinite(float(e)) and float(e) >= 0
    result["N60_net"] = dict(value=n, error=float(e) if valid_error else None)
    result["check_counts"] = bool(sp <= 0 or n <= 0)
    if n > 0:
        result["ratio"]["value"] = float(sp/n)
        if valid_error:
            result["ratio"]["error"] = float(np.sqrt(variance/n**2+sp**2*float(e)**2/n**4))
    return result


def plot_odlf_windows(counts, bins, result, detector, run, meta):
    counts, bins = np.asarray(counts), np.asarray(bins)
    centers = (bins[:-1]+bins[1:])/2
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi, layout="constrained")
    ax.stairs(counts, bins, color="gray", fill=True, alpha=.4, linewidth=.8, label="Data (QC)")
    for name, color, alpha in [
        ("background_alternative", "royalblue", .5),
        ("background", "deepskyblue", .5),
        ("slow_pulse", "lightgreen", .5),
        ("slow_pulse_inner", "green", .6),
    ]:
        w = result["windows"][name]
        low, high = w["requested_range_keV"]
        idx = np.flatnonzero((centers >= low) & (centers < high))
        ax.stairs(counts[idx], bins[idx[0]:idx[-1]+2], fill=True,
                  color=color, alpha=alpha,
                  label=f"{low:g}–{high:g} keV: {w['counts']:,.0f} counts")
    ax.stairs(counts, bins, color="gray", linewidth=.6, zorder=10)
    sp, ratio = result["SP_net"], result["ratio"]
    text = f"SP net = {sp['value']:,.0f} ± {sp['error']:,.0f}"
    if ratio["value"] is not None:
        text += f"\nO_DLF = {ratio['value']:.4g}"
        if ratio["error"] is not None:
            text += f" ± {ratio['error']:.2g}"
    ax.text(.98, .97, text, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    visible = (centers >= 5) & (centers < 150)
    positive = counts[visible & (counts > 0)]
    ax.set_yscale("log")
    ax.set_xlim(5, 150)
    ax.set_ylim(max(.5, positive.min()/2) if positive.size else .5,
                max(2., positive.max()*3) if positive.size else 10.)
    ax.set_xlabel("E (keV)")
    widths = np.diff(bins)
    ax.set_ylabel(f"Counts / {widths[0]:g} keV" if np.allclose(widths, widths[0]) else "Counts / bin")
    title = f"{detector}/{run} — {measure} — QC"
    pos = meta.get("source_position", {})
    if pos:
        title += f"\nr={pos.get('r_in_mm')} mm, z={pos.get('z_in_mm')} mm, φ={pos.get('phi_in_deg')}°"
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    return fig


def main():
    global current_fit_detector
    output_dir.mkdir(parents=True, exist_ok=True)
    db = dbetto.TextDB(metadata_path)
    all_results, rows = {}, []
    detector_list = discover_detectors()
    if not detector_list:
        raise RuntimeError("Nessun detector trovato: controllare version e percorsi")
    for detector in detector_list:
        current_fit_detector = detector
        all_results[detector] = {}
        for run in discover_runs(detector, db):
            print(f"\n{detector}/{run}", flush=True)
            directory = output_dir/detector/run
            directory.mkdir(parents=True, exist_ok=True)
            record = dict(detector=detector, run=run, version=version,
                          campaign=campaign, measurement=measure, errors=[],
                          qc_applied=True,
                          qc_selection=QC_SELECTION)
            fits = {"60": None, "100": None}
            counts, bins = None, None
            record["counting_method"] = None
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
                denominator = r100["parameters"][0]+r100["parameters"][3]
                if denominator > 0:
                    ratio["value"] = float(n60/denominator)
                    cov = r100["covariance"]
                    var_d = cov[0, 0]+cov[3, 3]+2*cov[0, 3]
                    variance = r60["covariance"][0, 0]/denominator**2 + n60**2*var_d/denominator**4
                    if np.isfinite(variance) and variance >= 0:
                        ratio["error"] = float(np.sqrt(variance))
                ratio["check_fit"] = any(r["at_bounds"] or not r["full_rank"] for r in fits.values())
            # Counting remains available even if a shape fit failed.
            if counts is not None:
                try:
                    resolution = load_counting_resolution(detector)
                    record["counting_resolution"] = resolution
                    counting = {"60": counting_method60(counts, bins, resolution),
                                "100": counting_method100(counts, bins, resolution)}
                    counting["ratio"] = counting_ratio(counting["60"], counting["100"])
                    record["counting_method"] = counting
                except Exception as exc:
                    record["errors"].append(f"Counting: {type(exc).__name__}: {exc}")
            record["ratio_counting"] = (
                record["counting_method"]["ratio"] if record["counting_method"] else None)
            record["ratio"] = ratio
            record["DLF"] = None
            if counts is not None:
                try:
                    cm = record.get("counting_method") or {}
                    peak60 = ((cm.get("60") or {}).get("peaks") or {}).get("N60")
                    record["DLF"] = calculate_odlf(counts, bins, peak60)
                    fig = plot_odlf_windows(counts, bins, record["DLF"], detector, run, meta)
                    try:
                        fig.savefig(directory/"slow_pulse_windows.png", dpi=dpi, facecolor="white")
                        fig.savefig(directory/"slow_pulse_windows.pdf", facecolor="white")
                    finally:
                        plt.close(fig)
                except Exception as exc:
                    record["errors"].append(f"DLF: {type(exc).__name__}: {exc}")
            record["status"] = "ok" if all(v is not None for v in fits.values()) else "incomplete"
            all_results[detector][run] = record
            save_json(directory/"results.json", record)
            row = dict(detector=detector, run=run, status=record["status"], qc_applied=True,
                       N60_over_N99_plus_N103=ratio["value"], ratio_error=ratio["error"])
            for region, names in [("60", names60), ("100", names100)]:
                result = fits[region]
                for i, name in enumerate(names):
                    row[name] = float(result["parameters"][i]) if result is not None else None
                    row[name+"_error"] = float(result["errors"][i]) if result is not None else None
                for key in ["fwhm", "fwhm_error", "deviance", "ndof", "full_rank", "at_bounds"]:
                    row[f"{key}_{region}"] = result[key] if result is not None else None
            cm = record["counting_method"]
            for name, region in (("N60", "60"), ("N99", "100"), ("N103", "100")):
                p = cm[region]["peaks"][name] if cm else {}
                row[name+"_counting"] = p.get("value")
                row[name+"_counting_error"] = p.get("error")
            row["ratio_counting"] = cm["ratio"]["value"] if cm else None
            row["ratio_counting_error"] = cm["ratio"]["error"] if cm else None
            resolution_info = record.get("counting_resolution") or {}
            row["counting_resolution_file"] = resolution_info.get("file")
            for name, region in (("N60", "60"), ("N99", "100"), ("N103", "100")):
                row[name+"_counting_sigma_keV"] = cm[region]["sigma_keV"][name] if cm else None
            dlf = record.get("DLF") or {}
            sp, odlf = dlf.get("SP_net") or {}, dlf.get("ratio") or {}
            row["SP_net"] = sp.get("value")
            row["SP_net_error"] = sp.get("error")
            row["O_DLF"] = odlf.get("value")
            row["O_DLF_error"] = odlf.get("error")
            row["errors"] = " | ".join(record["errors"])
            rows.append(row)
            if any(v is not None for v in fits.values()):
                fig = plot_two_regions(fits, detector, run, meta, ratio)
                try:
                    fig.savefig(directory/"fits_60_100.png", dpi=dpi, facecolor="white")
                    fig.savefig(directory/"fits_60_100.pdf", facecolor="white")
                finally:
                    plt.close(fig)
            if record["counting_method"] is not None:
                try:
                    fig = plot_counting_regions(counts, bins, record["counting_method"], detector, run, meta)
                    try:
                        fig.savefig(directory/"counting_60_100.png", dpi=dpi, facecolor="white")
                        fig.savefig(directory/"counting_60_100.pdf", facecolor="white")
                    finally:
                        plt.close(fig)
                except Exception as exc:
                    record["errors"].append(f"Counting plot: {type(exc).__name__}: {exc}")
                    row["errors"] = " | ".join(record["errors"])
                    save_json(directory/"results.json", record)
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
