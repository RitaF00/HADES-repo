# hades-metadata [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Metadata database for the LEGEND HPGe characterization test stand at HADES.

> [!NOTE]
>
> Metadata format specification is documented at
> [legend-exp.github.io/legend-data-format-specs/metadata](https://legend-exp.github.io/legend-data-format-specs/dev/metadata).

## Structure

- `dataprod/`
  - `config` →
    [hades-dataflow-config](https://github.com/legend-exp/hades-dataflow-config)
- `hardware/`
  - `configuration` → metadata from the DAQ system (see
    [legend-exp/hades-l200tests-daq](https://github.com/legend-exp/hades-l200tests-daq))
  - `detectors` →
    [legend-detectors](https://github.com/legend-exp/legend-detectors)
  - `holder_wrap` → dimensions of the detector holder and mylar wrap.

## Basic usage

### 1. Loading the database

```python
from dbetto import TextDB
db = TextDB("hades-metadata/")
hw = db.hardware
```

### 2. Extracting the source position for a given measurement

```python
pos = db.hardware.configuration[det].c1.am_HS1_top_dlt.run0001.source_position
```

### 3. Extracting the holder and wrap dimensions

```python
holder = db.hardware.holder_wrap.[det].holder
wrap = db.hardware.holder_wrap.[det].wrap
```

## Contributing

[pre-commit](https://pre-commit.com) is configured to automatically run on pull
requests. The auto-fixing [pre-commit.ci](https://pre-commit.ci/) bot is
unfortunately not available, since the repository is private. Before submitting
a PR, you _must_ run pre-commit locally and address all the reported issues.
Install instructions are found [here](https://pre-commit.com/#installation).

```console
$ pre-commit install # highly recommended
$ pre-commit run --all-files
```

## Related repositories

- [legend-exp/legend-pygeom-hades](https://github.com/legend-exp/legend-pygeom-hades)
- [legend-exp/hades-dataflow](https://github.com/legend-exp/hades-dataflow)
- [legend-exp/hades-l200tests-daq](https://github.com/legend-exp/hades-l200tests-daq)
