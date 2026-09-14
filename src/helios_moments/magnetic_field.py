
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .calculate_moments import _transform_B_to_RTN
from scipy.signal import find_peaks


def import_MF_files(start_time, end_time, paths, probe,DEBUG_MF=False):
    """
    Find magnetic field files between start_time and end_time.

    Returns
    -------
    mag_6s_files : list
        Paths to E3 6-second magnetic field files.

    mag_4Hz_files : list
        Paths to E2 4-Hz magnetic field files.
    """

    if end_time < start_time:
        raise ValueError(
            "end_time must be greater than or equal to start_time"
        )

    probe = str(probe)

    if probe == "1":
        mag_6s_base_path = paths["H1_6S"]
        mag_4Hz_base_path = paths["H1_4HZ"]

    elif probe == "2":
        mag_6s_base_path = paths["H2_6S"]
        mag_4Hz_base_path = paths["H2_4HZ"]

    else:
        raise ValueError("probe must be '1' or '2'")

    mag_6s_files = []
    mag_4Hz_files = []

    current_date = start_time.date()
    end_date = end_time.date()

    while current_date <= end_date:

        year_full = current_date.year
        year_short = str(year_full)[2:]
        day_of_year = f"{current_date.timetuple().tm_yday:03d}"

        # --------------------------------------------------
        # Search 6-second E3 data
        # --------------------------------------------------

        year_folder = os.path.join(
            mag_6s_base_path,
            str(year_full)
        )

        if os.path.isdir(year_folder):

            for filename in os.listdir(year_folder):

                if (
                    filename.startswith(
                        f"h{probe}{year_short}{day_of_year}"
                    )
                    and filename.endswith(".asc")
                ):
                    filepath = os.path.join(
                        year_folder,
                        filename
                    )

                    mag_6s_files.append(filepath)

        # --------------------------------------------------
        # Search 4-Hz E2 data
        # --------------------------------------------------

        if os.path.isdir(mag_4Hz_base_path):

            for filename in os.listdir(mag_4Hz_base_path):

                if (
                    filename.startswith(f"he{probe}")
                    and filename[5:7] == year_short
                    and filename[7:10] == day_of_year
                    and filename.endswith(".asc")
                ):
                    filepath = os.path.join(
                        mag_4Hz_base_path,
                        filename
                    )

                    mag_4Hz_files.append(filepath)

        current_date += timedelta(days=1)

    # --------------------------------------------------
    # Sort chronologically by filename
    # --------------------------------------------------

    mag_6s_files = sorted(mag_6s_files)
    mag_4Hz_files = sorted(mag_4Hz_files)

    if DEBUG_MF:

        print("\n--- import_MF_files DEBUG ---")
        print("probe:", probe)
        print("start_time:", start_time)
        print("end_time:", end_time)

        print("\n6-second files:")
        for filepath in mag_6s_files:
            print(filepath)

        print("\n4-Hz files:")
        for filepath in mag_4Hz_files:
            print(filepath)

    return mag_6s_files, mag_4Hz_files

def _moving_average_peak_flag(series, window=10, prominence=15.0):
    """
    Flag outliers measurements using a moving-average
    background and scipy.signal.find_peaks.
    
    """
    background = series.rolling(window=window, center=True, min_periods=1).mean()
   
    # Difference from the local background
    residual = series - background
    values = residual.to_numpy()
    
    
    peaks_up, _ = find_peaks(residual.to_numpy(), prominence=prominence)
    peaks_down, _ = find_peaks(-residual.to_numpy(), prominence=prominence)

    flag = pd.Series(False, index=series.index)
    flag.iloc[peaks_up] = True
    flag.iloc[peaks_down] = True

    return flag

def get_magnetic_field(start_time,end_time,paths,probe,instrument="E3",
    filter_outliers=False,DEBUG_MF=False,):
    """
    Return magnetic field measurements between start_time and end_time.

    Parameters
    ----------
    start_time : datetime
        Beginning of the requested interval.
    end_time : datetime
        End of the requested interval.
    paths : dict
        Dictionary containing the Helios data paths.
    probe : str or int
        Helios spacecraft, 1 or 2.
    instrument : {"auto", "E2", "E3"}
        "E3"   : use 6-second magnetic field data.
        "E2"   : use 4-Hz magnetic field data.
        "auto" : use E3 if available, otherwise E2.
    filter_outliers : bool
        If True, apply the moving-average peak filter to E2 measurements.
        E3 measurements are never filtered.
    DEBUG_MF : bool
        Print debugging information.

    Returns
    -------
    B_df : pandas.DataFrame
        Magnetic field measurements indexed by datetime.
    B_instrument : str or np.nan
        Instrument used ("E2" or "E3").
    """
    valid_instruments = ("auto", "E2", "E3")
    if instrument not in valid_instruments:
        raise ValueError("instrument must be 'auto', 'E2', or 'E3'")
    if not isinstance(filter_outliers, bool):
        raise ValueError("filter_outliers must be True or False")
    probe = str(probe)

    mag_6s_files, mag_4Hz_files = import_MF_files(
        start_time,
        end_time,
        paths,
        probe,
        DEBUG_MF=False,
    )

    # Read E3 6-second data
    filtered_6s_B_df = pd.DataFrame()
    if instrument in ("auto", "E3"):
        dfs = []
        for mag6s_file in mag_6s_files:
            try:
                colspecs = [
                    (1, 2), (2, 4), (4, 7), (7, 9),
                    (9, 11), (11, 13), (13, 15),
                    (15, 22), (22, 29), (29, 36),
                    (36, 42), (42, 48), (48, 54),
                    (54, 60),
                ]
                column_names = [
                    "probe", "year", "doy", "hour", "minute", "second",
                    "n_avg", "Bx", "By", "Bz", "B_mag",
                    "std_Bx", "std_By", "std_Bz",
                ]
                df = pd.read_fwf(
                    mag6s_file,
                    colspecs=colspecs,
                    names=column_names,
                )
                time_columns = ["year", "doy", "hour", "minute", "second"]
                for col in time_columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df.dropna(subset=time_columns, inplace=True)
                df["year"] = 1900 + df["year"].astype(int)
                df["datetime"] = pd.to_datetime(
                    df["year"].astype(str)
                    + "-"
                    + df["doy"].astype(int).astype(str),
                    format="%Y-%j",
                )
                df["datetime"] += pd.to_timedelta(df["hour"], unit="h")
                df["datetime"] += pd.to_timedelta(df["minute"], unit="m")
                df["datetime"] += pd.to_timedelta(df["second"], unit="s")
                df.set_index("datetime", inplace=True)
                dfs.append(df)
            except Exception as e:
                if DEBUG_MF:
                    print(f"Error reading 6s file {mag6s_file}: {e}")
        if dfs:
            mag6s = pd.concat(dfs)
            mag6s.sort_index(inplace=True)
            filtered_6s_B_df = mag6s.loc[
                (mag6s.index >= start_time) & (mag6s.index <= end_time),
                ["Bx", "By", "Bz"],
            ].copy()

    # Read E2 4-Hz data
    filtered_4hz_B_df = pd.DataFrame()
    if instrument in ("auto", "E2"):
        dfs = []
        for mag4hz_file in mag_4Hz_files:
            try:
                df = pd.read_csv(
                    mag4hz_file,
                    sep=r"\s+",
                    header=None,
                    usecols=[0, 4, 5, 6],
                    names=["datetime", "Bx", "By", "Bz"],
                    dtype={"datetime": str},
                )
                df["datetime"] = pd.to_datetime(
                    df["datetime"],
                    errors="coerce",
                )
                df.dropna(subset=["datetime"], inplace=True)
                df.set_index("datetime", inplace=True)
                dfs.append(df)
            except Exception as e:
                if DEBUG_MF:
                    print(f"Error reading 4Hz file {mag4hz_file}: {e}")
        if dfs:
            mag4hz = pd.concat(dfs)
            mag4hz.sort_index(inplace=True)
            filtered_4hz_B_df = mag4hz.loc[
                (mag4hz.index >= start_time) & (mag4hz.index <= end_time),
                ["Bx", "By", "Bz"],
            ].copy()

    # Prepare magnetic field
    def prepare_field(B_df, B_instrument, probe):
        B_df = B_df.copy()

        # Convert SSE -> RTN
        Br, Bt, Bn = _transform_B_to_RTN(
            B_instrument,
            probe,
            B_df["Bx"],
            B_df["By"],
            B_df["Bz"],
        )
        B_df["Br"] = Br
        B_df["Bt"] = Bt
        B_df["Bn"] = Bn

        # Magnetic field magnitude
        B_df["B"] = np.sqrt(
            B_df["Br"]**2 + B_df["Bt"]**2 + B_df["Bn"]**2
        )

        # Magnetic field quality status
        B_df["B Status"] = "1"
        low_field = (
            (B_df["Br"].abs() < 1e-14)
            | (B_df["Bt"].abs() < 1e-14)
            | (B_df["Bn"].abs() < 1e-14)
        )
        B_df.loc[low_field, "B Status"] = "2"

        # Remove moving-average peak outliers from E2 only
        if filter_outliers and B_instrument == "E2":
            Br_flag = _moving_average_peak_flag(B_df["Br"])
            Bt_flag = _moving_average_peak_flag(B_df["Bt"])
            Bn_flag = _moving_average_peak_flag(B_df["Bn"])
            flagged = Br_flag | Bt_flag | Bn_flag
            if DEBUG_MF:
                print(
                    f"Outlier filtering ({B_instrument}): "
                    f"{flagged.sum()} of {len(B_df)} measurements removed"
                )
            B_df = B_df.loc[~flagged].copy()
            
        return B_df

    # Select instrument
    if instrument in ("auto", "E3") and not filtered_6s_B_df.empty:
        B_df = prepare_field(filtered_6s_B_df, "E3", probe)
        return B_df, "E3"
    if instrument in ("auto", "E2") and not filtered_4hz_B_df.empty:
        B_df = prepare_field(filtered_4hz_B_df, "E2", probe)
        return B_df, "E2"
    return pd.DataFrame(), np.nan


def plot_magnetic_field(start_time,end_time,paths,probe,instrument="both",filter_outliers=False):
    """Plot E2 only, E3 only, or both magnetic field instruments in RTN coordinates."""
    if instrument not in ("E2","E3","both"):
        raise ValueError("instrument must be 'E2', 'E3', or 'both'")
    E2_df = pd.DataFrame()
    E3_df = pd.DataFrame()
    if instrument in ("E2","both"):
        E2_df, _ = get_magnetic_field(
            start_time=start_time,
            end_time=end_time,
            paths=paths,
            probe=probe,
            instrument="E2",
            filter_outliers=filter_outliers
        )
    if instrument in ("E3","both"):
        E3_df, _ = get_magnetic_field(
            start_time=start_time,
            end_time=end_time,
            paths=paths,
            probe=probe,
            instrument="E3",
            filter_outliers=False
        )
    if instrument == "E2" and E2_df.empty:
        raise ValueError("No E2 magnetic field data were found for the requested interval.")
    if instrument == "E3" and E3_df.empty:
        raise ValueError("No E3 magnetic field data were found for the requested interval.")
    if instrument == "both" and E2_df.empty and E3_df.empty:
        raise ValueError("No E2 or E3 magnetic field data were found for the requested interval.")

    fig, axes = plt.subplots(4,1,figsize=(10,10),sharex=True)

    # BR
    if not E2_df.empty:
        axes[0].plot(E2_df.index,E2_df["Br"],color="crimson",label="E2")
    if not E3_df.empty:
        axes[0].plot(E3_df.index,E3_df["Br"],color="black",label="E3")
    axes[0].set_ylabel(r"$B_R$ [nT]",fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # BT
    if not E2_df.empty:
        axes[1].plot(E2_df.index,E2_df["Bt"],color="green",label="E2")
    if not E3_df.empty:
        axes[1].plot(E3_df.index,E3_df["Bt"],color="black",label="E3")
    axes[1].set_ylabel(r"$B_T$ [nT]",fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # BN
    if not E2_df.empty:
        axes[2].plot(E2_df.index,E2_df["Bn"],color="blue",label="E2")
    if not E3_df.empty:
        axes[2].plot(E3_df.index,E3_df["Bn"],color="black",label="E3")
    axes[2].set_ylabel(r"$B_N$ [nT]",fontsize=14)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # |B|
    if not E2_df.empty:
        axes[3].plot(E2_df.index,E2_df["B"],color="gray",label="E2")
    if not E3_df.empty:
        axes[3].plot(E3_df.index,E3_df["B"],color="black",label="E3")
    axes[3].set_ylabel(r"$|B|$ [nT]",fontsize=14)
    axes[3].set_xlabel("Time",fontsize=14)
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    if instrument == "both":
        fig.suptitle(f"Helios {probe} Magnetic Field: E2 vs E3")
    else:
        fig.suptitle(f"Helios {probe} Magnetic Field: {instrument}")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes


# def plot_magnetic_field_moving_average_peak_flags(
#     start_time, end_time, paths, probe, instrument="auto", window=10, prominence=15.0
# ):
#     B_df, B_instrument = get_magnetic_field(
#         start_time=start_time, end_time=end_time, paths=paths,
#         probe=probe, instrument=instrument
#     )

#     if B_df.empty:
#         raise ValueError("No magnetic field data were found for the requested interval.")

#     Br_flag = _moving_average_peak_flag(B_df["Br"], window=window, prominence=prominence)
#     Bt_flag = _moving_average_peak_flag(B_df["Bt"], window=window, prominence=prominence)
#     Bn_flag = _moving_average_peak_flag(B_df["Bn"], window=window, prominence=prominence)

#     fig, ax = plt.subplots(figsize=(10, 5))

#     ax.plot(B_df.index, B_df["Br"], color="red", label=r"$B_R$")
#     ax.plot(B_df.index, B_df["Bt"], color="green", label=r"$B_T$")
#     ax.plot(B_df.index, B_df["Bn"], color="blue", label=r"$B_N$")

#     ax.scatter(B_df.index[Br_flag], B_df.loc[Br_flag, "Br"], color="black", marker="x", s=35)
#     ax.scatter(B_df.index[Bt_flag], B_df.loc[Bt_flag, "Bt"], color="black", marker="x", s=35)
#     ax.scatter(B_df.index[Bn_flag], B_df.loc[Bn_flag, "Bn"], color="black", marker="x", s=35)

#     ax.set_xlabel("Time")
#     ax.set_ylabel("Magnetic Field [nT]")
#     ax.set_title(f"Helios {probe} Magnetic Field ({B_instrument}) — Moving Average + Peak Flags")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     fig.autofmt_xdate()
#     fig.tight_layout()

#     return fig, ax