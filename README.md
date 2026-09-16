# Helios Moments

**Helios Moments** is a Python package for calculating proton plasma moments from three-dimensional ion velocity distribution functions (VDFs) measured by the Helios E1 plasma experiment.

The package processes Helios 1 and Helios 2 ion measurements, separates the proton and alpha-particle contributions using either a regular or machine-learning-based speed-cut method, associates each VDF with the corresponding magnetic field measurements, and numerically calculates proton density, bulk velocity, parallel and perpendicular temperatures, and heat fluxes.

## Overview

For each selected Helios VDF, the package:

1. Reads and validates the ion VDF.
2. Applies VDF quality-control filters.
3. Determines the proton–alpha separation speed using either the regular or
   machine-learning speed-cut method.
4. Associates the VDF with E2 or E3 magnetic field measurements and evaluates
   their quality.
5. Numerically calculates the proton plasma moments.
6. Repeats the calculation using the velocity bins neighboring the nominal
   speed cut to estimate sensitivity to the cut.
7. Writes the results and quality flags to CSV.

The package processes observations sequentially and writes results to disk
during the calculation, allowing large datasets to be analyzed without
storing all calculated moments in memory. Interrupted calculations can also
be resumed from the last successfully saved observation.


## Installation

### Requirements

Helios Moments requires **Python 3.9 or later**.

The main Python dependencies are:

- NumPy
- SciPy
- pandas
- Matplotlib
- scikit-learn
- numba

These dependencies are installed automatically when the package is installed.

### Install from source

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd helios_moments
```

Install the package using:

```bash
python -m pip install .
```

### Editable installation

If you plan to modify or develop the code, install the package in editable mode:

```bash
python -m pip install -e .
```

With an editable installation, changes made to the source code under
`src/helios_moments/` are available without reinstalling the package.

### Verify the installation

To verify that the package can be imported successfully, run:

```bash
python -c "import helios_moments; print('Helios Moments installed successfully')"
```

## Data Requirements and Configuration

### Data requirements

Helios Moments does not include the original Helios spacecraft data. The
required data must be downloaded separately.

The calculation uses data from the following Helios instruments:

- **E1 plasma experiment:** three-dimensional ion velocity distribution
  functions (VDFs).
- **E2 magnetic field experiment:** 4 Hz magnetic field measurements.
- **E3 magnetic field experiment:** 6 s magnetic field measurements.

The Helios data are publicly available from the UC Berkeley Space Sciences
Laboratory Helios data archive:

- E1 ion distribution functions:
  https://helios-data.ssl.berkeley.edu/data/E1_experiment/E1_original_data/
- E2 magnetic-field data:
  https://helios-data.ssl.berkeley.edu/data/E2_experiment/
- E3 magnetic-field data:
  https://helios-data.ssl.berkeley.edu/data/E3_experiment/

Data for either **Helios 1**, **Helios 2**, or both spacecraft can be used.

### Configuration

Before running the package, edit `data_paths.txt` to specify the locations
of the Helios data on your system:

```text
# VDF directories
H1_VDF=/path/to/helios1/vdf/data
H2_VDF=/path/to/helios2/vdf/data

# Magnetic-field directories
H1_6S=/path/to/helios1/e3/data
H1_4HZ=/path/to/helios1/e2/data

H2_6S=/path/to/helios2/e3/data
H2_4HZ=/path/to/helios2/e2/data
```

`H1` and `H2` refer to Helios 1 and Helios 2, while `6S` and `4HZ` refer to the E3 and E2 magnetic field datasets, respectively. The paths should point to the top-level directories containing each dataset.

## Usage

Moment calculations are run from the command line using
`scripts/run_moments.py`.

Before running the calculation, make sure that the Helios data directories
have been specified in `data_paths.txt` as described in the
[Data Requirements and Configuration](#data-requirements-and-configuration)
section.

### Basic command

The main command-line options are:

- `--probe`: Helios spacecraft (`1` or `2`)
- `--start`, `--end`: observation interval in `YYYY-DDD HH:MM:SS` format
- `--method`: `regular`, `ml`, or `both`
- `--output`: output CSV filename

Here, `DDD` is the day of year; for example, `1976-100 00:00:00`
corresponds to day 100 of 1976 at 00:00:00.

For the `ml` method, the machine-learning model is trained using the labeled
training data included with the package before the requested observations
are processed.

When `both` is selected, two output files are created automatically.

### Start a new run

```bash
python scripts/run_moments.py \
    --probe 2 \
    --start "1976-100 00:00:00" \
    --end "1976-120 23:59:59" \
    --method ml \
    --output moments_ml.csv
```

### Resuming an interrupted calculation

For long calculations, the `--resume` option can be used to continue an
existing calculation:

```bash
python scripts/run_moments.py \
    --probe 2 \
    --start "1976-100 00:00:00" \
    --end "1976-120 23:59:59" \
    --method ml \
    --output moments_ml.csv \
    --resume
```

When `--resume` is used, existing results are preserved and measurements
that have already been saved are skipped. New results are appended to the
existing output file.

Without `--resume`, an existing output file with the same name is
overwritten.

### Command-line help

A summary of the available command-line options can be displayed with:

```bash
python scripts/run_moments.py --help
```

### Magnetic field tools

The package also provides `get_magnetic_field` and `plot_magnetic_field`
for retrieving and plotting the Helios E2 and E3 magnetic field measurements.

Example usage is provided in
[`notebooks/magnetic_field_examples.ipynb`](notebooks/magnetic_field_examples.ipynb).

## Output

The calculated moments are saved as CSV files, with one row corresponding
to each processed Helios VDF observation.

Each output row contains:

- Observation time, spacecraft, and instrument information.
- Heliocentric distance.
- Magnetic field components and magnetic field magnitude.
- VDF and magnetic field quality/status flags.
- Proton number density.
- Radial, tangential, and normal proton bulk velocities.
- Parallel and perpendicular proton temperatures.
- Parallel and perpendicular proton heat fluxes.
- Proton–alpha speed-cut values.

For the calculated plasma moments, the output includes three values labeled
`min`, `mid`, and `max`. The `mid` value corresponds to the nominal speed
cut, while `min` and `max` are obtained using the neighboring velocity bins
around the nominal cut and provide estimates of the sensitivity of the
calculated moments to the choice of speed cut.

For example:

```text
n_min [cm^-3]
n_mid [cm^-3]
n_max [cm^-3]

```

The same convention is used for the bulk velocity components,
temperatures, heat fluxes, and speed cut.

For the machine-learning method, the output additionally contains:

```text
v_cut_std [km/s]
```

which gives the standard deviation of the speed-cut predictions from the
individual trees in the Random Forest model.

A complete description of the output parameters, units, status codes, and calculation procedure is provided in the Helios Numerical Moments User Guide.


## Data Product

The complete Helios numerical proton moments dataset produced with this
software is publicly available at:

[Helios Archive/link]

Users who wish to work directly with the calculated moments can download
the dataset without running the processing pipeline.

## Documentation

The **Helios Numerical Moments User Guide**, available in the
[`docs/`](docs/) directory, provides detailed descriptions of:

- VDF preprocessing and quality control.
- Magnetic field selection and quality assessment.
- Numerical moment calculations.
- Status flags, output parameters, and units.

For details of the regular and machine-learning proton–alpha speed-cut
methods and the scientific interpretation of the calculated moments,
see the associated publication listed in the [Citation](#citation) section.

## Citation
If you use Helios Moments or the numerical moments produced with this
software in scientific work, please cite the associated publication.

Citation information will be added here upon publication.

