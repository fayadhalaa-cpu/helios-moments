import os
import re
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from .helios_vdfs_data import estimate_start_end_time


def extract_probe(filename):
    """
    Extract probe number as a string ('1' or '2')
    from filenames like 'h2y80d001h00m04s43_hdm.0'
    """
    basename = os.path.basename(filename).lower()
    if basename.startswith("h1"):
        return "1"
    elif basename.startswith("h2"):
        return "2"
    return None


def extract_datetime(filename):
    """
    Extract datetime from filenames like:
    'h2y80d001h00m04s43_hdm.0'
    """
    match = re.search(r"y(\d{2})d(\d{3})h(\d{2})m(\d{2})s(\d{2})", filename)
    if match:
        year = 1900 + int(match.group(1))
        doy = int(match.group(2))
        hour = int(match.group(3))
        minute = int(match.group(4))
        second = int(match.group(5))

        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)
    return None



def import_field_files(datafile, paths, DEBUG_MF=True):
    """
    Given a datafile path, returns paths to associated 6-second and 4Hz magnetic field data files.
    """
    timestamp = extract_datetime(datafile)
    if timestamp is None:
        raise ValueError(f"Could not extract datetime from filename: {datafile}")

    year_full = timestamp.year
    year_short = str(year_full)[2:]
    day_of_year = f"{timestamp.timetuple().tm_yday:03d}"

    probe = extract_probe(datafile)
    if probe not in {"1", "2"}:
        raise ValueError(f"Could not extract probe from filename: {datafile}")

    if probe == "1":
        mag_6s_base_path = paths["H1_6S"]
        mag_4Hz_base_path = paths["H1_4HZ"]
    else:
        mag_6s_base_path = paths["H2_6S"]
        mag_4Hz_base_path = paths["H2_4HZ"]

    # --- Search for 6s file ---
    year_folder = os.path.join(mag_6s_base_path, str(year_full))
    mag_6s_file = None
    if os.path.isdir(year_folder):
        for file in os.listdir(year_folder):
            if file.startswith(f"h{probe}{year_short}{day_of_year}") and file.endswith(".asc"):
                mag_6s_file = os.path.join(year_folder, file)
                break

    # --- Search for 4Hz file ---
    mag_4Hz_file = None
    if os.path.isdir(mag_4Hz_base_path):
        for file in os.listdir(mag_4Hz_base_path):
            if (
                file.startswith(f"he{probe}")
                and file[5:7] == year_short
                and file[7:10] == day_of_year
                and file.endswith(".asc")
            ):
                mag_4Hz_file = os.path.join(mag_4Hz_base_path, file)
                break

    if DEBUG_MF:
        print("\n--- import_field_files DEBUG ---")
        print("datafile:", datafile)
        print("probe:", probe, " year:", year_full, " doy:", day_of_year)
        print("mag_6s_base_path:", mag_6s_base_path)
        print("mag_4Hz_base_path:", mag_4Hz_base_path)
        print("year_folder exists:", os.path.isdir(year_folder))
        print("FOUND 6s:", mag_6s_file)
        print("FOUND 4Hz:", mag_4Hz_file)

    return mag_6s_file, mag_4Hz_file


def _get_time_window(datafile, Proton_VDF=None, fallback_seconds=40):

    # Use VDF-derived window when Proton_VDF is available
    if Proton_VDF is not None:
        try:
            dist_starttime, dist_endtime = estimate_start_end_time(datafile, Proton_VDF)
            return dist_starttime, dist_endtime, "VDF"

        except Exception:
            pass

    # Otherwise use filename + fallback window
    t0 = extract_datetime(datafile)

    t1 = t0 + timedelta(seconds=fallback_seconds)

    return t0, t1, "filename_40s"


def load_magnetic_field_day(datafile, paths, filter_e2=True, window=10,
                            prominence=15.0, DEBUG_MF=False):
    """
    Load the full daily E3 (6-second) and E2 (4-Hz) magnetic field
    datasets associated with a VDF file.

    For E2, outliers are detected over the full daily
    time series and stored as flag columns in the cached dataframe.

    Returns
    -------
    day_key : tuple
        (probe, year, day_of_year)

    mag6s : pandas.DataFrame
        Full E3 daily magnetic field dataframe.

    mag4hz : pandas.DataFrame
        Full E2 daily magnetic field dataframe, including:
        Bx_flag, By_flag, Bz_flag, artifact_flag.
    """

    timestamp = extract_datetime(datafile)
    if timestamp is None:
        raise ValueError(f"Could not extract datetime from filename: {datafile}")

    probe = extract_probe(datafile)
    if probe not in {"1", "2"}:
        raise ValueError(f"Could not extract probe from filename: {datafile}")

    year = timestamp.year
    day_of_year = timestamp.timetuple().tm_yday
    day_key = (probe, year, day_of_year)

    mag6s_file, mag4hz_file = import_field_files(datafile, paths, DEBUG_MF=DEBUG_MF)

    mag6s = pd.DataFrame()
    mag4hz = pd.DataFrame()

    # ==================================================
    # Load complete E3 6-second daily file
    # ==================================================

    if mag6s_file:
        try:
            colspecs = [
                (1, 2), (2, 4), (4, 7), (7, 9), (9, 11), (11, 13),
                (13, 15), (15, 22), (22, 29), (29, 36), (36, 42),
                (42, 48), (48, 54), (54, 60),
            ]

            column_names = [
                "probe", "year", "doy", "hour", "minute", "second",
                "n_avg", "Bx", "By", "Bz", "B_mag",
                "std_Bx", "std_By", "std_Bz",
            ]

            mag6s = pd.read_fwf(mag6s_file, colspecs=colspecs, names=column_names)

            for col in ["year", "doy", "hour", "minute", "second"]:
                mag6s = mag6s[pd.to_numeric(mag6s[col], errors="coerce").notnull()]

            mag6s["year"] = mag6s["year"].apply(lambda y: 1900 + int(float(y)))

            mag6s["datetime"] = mag6s.apply(
                lambda row: datetime.strptime(
                    f"{int(row['year'])} {int(row['doy'])} "
                    f"{int(row['hour'])} {int(row['minute'])} {int(row['second'])}",
                    "%Y %j %H %M %S"
                ),
                axis=1,
            )

            mag6s.set_index("datetime", inplace=True)
            mag6s = mag6s[
                ["Bx", "By", "Bz", "B_mag", "std_Bx", "std_By", "std_Bz"]
            ]
            mag6s.sort_index(inplace=True)

        except Exception as e:
            mag6s = pd.DataFrame()
            if DEBUG_MF:
                print(f"Error loading daily 6s file {mag6s_file}: {e}")

    # ==================================================
    # Load complete E2 4-Hz daily file
    # ==================================================

    if mag4hz_file:
        try:
            mag4hz = pd.read_csv(
                mag4hz_file, sep=r"\s+", header=None, usecols=[0, 4, 5, 6],
                names=["datetime", "Bx", "By", "Bz"], dtype={"datetime": str}
            )

            mag4hz["datetime"] = pd.to_datetime(mag4hz["datetime"], errors="coerce")
            mag4hz.dropna(subset=["datetime"], inplace=True)
            mag4hz.set_index("datetime", inplace=True)
            mag4hz.sort_index(inplace=True)

            # Detect suspicious E2 measurements over the full daily file
            if filter_e2 and not mag4hz.empty:
                Bx_flag = _moving_average_peak_flag(
                    mag4hz["Bx"], window=window, prominence=prominence
                )
                By_flag = _moving_average_peak_flag(
                    mag4hz["By"], window=window, prominence=prominence
                )
                Bz_flag = _moving_average_peak_flag(
                    mag4hz["Bz"], window=window, prominence=prominence
                )

                mag4hz["Bx_flag"] = Bx_flag
                mag4hz["By_flag"] = By_flag
                mag4hz["Bz_flag"] = Bz_flag
                mag4hz["artifact_flag"] = Bx_flag | By_flag | Bz_flag

                if DEBUG_MF:
                    print(
                        f"E2 daily filter detected {Bx_flag.sum()} Bx, "
                        f"{By_flag.sum()} By, {Bz_flag.sum()} Bz suspicious measurements."
                    )
                    print(
                        f"Total unique suspicious E2 timestamps: "
                        f"{mag4hz['artifact_flag'].sum()}"
                    )

            else:
                mag4hz["Bx_flag"] = False
                mag4hz["By_flag"] = False
                mag4hz["Bz_flag"] = False
                mag4hz["artifact_flag"] = False

        except Exception as e:
            mag4hz = pd.DataFrame()
            if DEBUG_MF:
                print(f"Error loading daily 4Hz file {mag4hz_file}: {e}")

    # ==================================================
    # Debug information
    # ==================================================

    if DEBUG_MF:
        print("\n--- load_magnetic_field_day DEBUG ---")
        print(f"Cached day: Helios {probe}, {year}-{day_of_year:03d}")
        print(f"E3 measurements loaded: {len(mag6s)}")
        print(f"E2 measurements loaded: {len(mag4hz)}")

        if not mag4hz.empty and "artifact_flag" in mag4hz.columns:
            print(f"E2 suspicious timestamps cached: {mag4hz['artifact_flag'].sum()}")

    return day_key, mag6s, mag4hz


def _moving_average_peak_flag(series,window=10,prominence=15.0,):

    if len(series) < 3:
        return pd.Series(False, index=series.index)

    actual_window = min(window, len(series))

    background = series.rolling(window=actual_window,center=True, min_periods=1,).mean()

    residual = series - background

    peaks_up, _ = find_peaks(residual.to_numpy(),prominence=prominence,)

    peaks_down, _ = find_peaks(-residual.to_numpy(),prominence=prominence,)

    flag = pd.Series(False,index=series.index,)

    flag.iloc[peaks_up] = True
    flag.iloc[peaks_down] = True

    return flag

def extract_magnetic_field_values(datafile, paths, Proton_VDF=None, fallback_seconds=40,mag6s=None, mag4hz=None, DEBUG_MF=False):
    """
    Return:
        Bx, By, Bz, B, Bstd, B_instrument, B_status, filter_status

    B_status:
        '1' = magnetic field available
        '2' = wobbly magnetic field
        '3' = no magnetic field data available
        '4' = one magnetic-field component is too low (<10^-14 nT)

    filter_status:
        0 = no suspicious E2 measurements occur within the VDF interval
        1 = one or more E2 outliers were removed

    mag6s : pandas.DataFrame or None
        Cached full-day E3 magnetic-field dataframe.

    mag4hz : pandas.DataFrame or None
        Cached full-day E2 magnetic-field dataframe containing the artifact
        flags calculated by load_magnetic_field_day().

    Notes
    -----
    E2 outlier detection is performed once over the full daily magnetic-field
    file in load_magnetic_field_day(). If any component of an E2 measurement
    is flagged, the entire row is removed before calculating the VDF field.
    """

    dist_starttime, dist_endtime, window_mode = _get_time_window(
        datafile,Proton_VDF=Proton_VDF, fallback_seconds=fallback_seconds)

    # ---- Load daily magnetic-field data if not supplied ----
    if mag6s is None and mag4hz is None:
        _, mag6s, mag4hz = load_magnetic_field_day(datafile, paths, DEBUG_MF=DEBUG_MF)

    if mag6s is None:
        mag6s = pd.DataFrame()
    if mag4hz is None:
        mag4hz = pd.DataFrame()

    filtered_6s_B_df = pd.DataFrame()
    filtered_4hz_B_df = pd.DataFrame()
    filter_status = 0

    # ==================================================
    # Select E3 6-second data for this VDF
    # ==================================================

    if not mag6s.empty:
        try:
            filtered_6s_B_df = mag6s.loc[
                (mag6s.index > dist_starttime) & (mag6s.index < dist_endtime),
                ["Bx", "By", "Bz", "B_mag", "std_Bx", "std_By", "std_Bz"]
            ].copy()
        except Exception as e:
            if DEBUG_MF:
                print(f"Error selecting cached 6s data: {e}")

    # ==================================================
    # Select E2 4-Hz data for this VDF and remove artifacts
    # ==================================================

    if not mag4hz.empty:
        try:
            vdf_4hz = mag4hz.loc[
                (mag4hz.index > dist_starttime) & (mag4hz.index < dist_endtime)
            ].copy()

            if not vdf_4hz.empty:
                if "artifact_flag" not in vdf_4hz.columns:
                    raise ValueError(
                        "Cached E2 data do not contain artifact flags. "
                        "Reload using load_magnetic_field_day()."
                    )

                any_flag = vdf_4hz["artifact_flag"].astype(bool)
                filter_status = int(any_flag.any())

                if DEBUG_MF and filter_status == 1:
                    print(
                        f"Removing {any_flag.sum()} suspicious E2 measurements "
                        "from this VDF interval."
                    )

                vdf_4hz = vdf_4hz.loc[~any_flag].copy()
                filtered_4hz_B_df = vdf_4hz[["Bx", "By", "Bz"]].copy()

        except Exception as e:
            if DEBUG_MF:
                print(f"Error selecting cached 4Hz data: {e}")

    # ==================================================
    # Check whether magnetic field is wobbly
    # ==================================================

    def check_wobbly_field(mag):
        mag_vecs = mag[["Bx", "By", "Bz"]].values

        if len(mag_vecs) < 2:
            return True

        norms = np.linalg.norm(mag_vecs, axis=1)
        good = norms > 0

        if np.count_nonzero(good) < 2:
            return True

        mag_normed = mag_vecs[good] / norms[good][:, None]
        dotprods = np.einsum("ji,ki->jk", mag_normed, mag_normed)

        return np.any(dotprods < np.cos(np.deg2rad(90)))

    # ==================================================
    # Check very-low field components
    # ==================================================

    def check_low_field(Bx, By, Bz):
        if abs(Bx) < 1e-14 or abs(By) < 1e-14 or abs(Bz) < 1e-14:
            return "4"
        return "1"

    # ==================================================
    # Try E3 first
    # ==================================================

    if (
        not filtered_6s_B_df.empty
        and not filtered_6s_B_df[["Bx", "By", "Bz"]].isnull().values.any()
    ):
        B_instrument = "E3"

        if check_wobbly_field(filtered_6s_B_df):
            if DEBUG_MF:
                print(f"Using 6s field ({window_mode} window), but field is wobbly.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, B_instrument, "2", 0

        B_avg = filtered_6s_B_df[["Bx", "By", "Bz"]].mean().values
        B_std = np.linalg.norm(
            filtered_6s_B_df[["Bx", "By", "Bz"]].std().values
        )

        if DEBUG_MF:
            print(f"Using 6s field ({window_mode} window).")

        B_status = check_low_field(B_avg[0], B_avg[1], B_avg[2])

        return (
            B_avg[0], B_avg[1], B_avg[2], np.linalg.norm(B_avg),
            B_std, B_instrument, B_status, 0
        )

    # ==================================================
    # Then try E2
    # ==================================================

    if (
        not filtered_4hz_B_df.empty
        and not filtered_4hz_B_df[["Bx", "By", "Bz"]].isnull().values.any()
    ):
        B_instrument = "E2"

        if check_wobbly_field(filtered_4hz_B_df):
            if DEBUG_MF:
                print(f"Using 4Hz field ({window_mode} window), but field is wobbly.")
            return (
                np.nan, np.nan, np.nan, np.nan, np.nan,
                B_instrument, "2", filter_status
            )

        B_avg = filtered_4hz_B_df[["Bx", "By", "Bz"]].mean().values
        B_std = np.linalg.norm(
            filtered_4hz_B_df[["Bx", "By", "Bz"]].std().values
        )

        if DEBUG_MF:
            print(
                f"Using 4Hz field ({window_mode} window). "
                f"Filter status = {filter_status}"
            )

        B_status = check_low_field(B_avg[0], B_avg[1], B_avg[2])

        return (B_avg[0], B_avg[1], B_avg[2], np.linalg.norm(B_avg),
            B_std, B_instrument, B_status, filter_status)

    # ==================================================
    # No usable magnetic field
    # ==================================================

    if DEBUG_MF:
        print(f"No usable magnetic field found for {datafile} "
            f"with {window_mode} window.")

    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "3", filter_status