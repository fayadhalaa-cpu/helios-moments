import os
import glob
from datetime import timedelta

from .helios_mf import extract_datetime, extract_probe


def get_bad_files(day_dir):
    """
    Read baddata.txt in one day directory and return
    the files listed there as a set of normalized paths.
    """

    bad_files = set()
    baddata_file = os.path.join(day_dir, "baddata.txt")

    if not os.path.isfile(baddata_file):
        return bad_files

    with open(baddata_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if parts:
                bad_path = os.path.normpath(
                    os.path.join(
                        day_dir,
                        os.path.basename(parts[-1])
                    )
                )
                bad_files.add(bad_path)

    return bad_files


def find_files(
    base_dir,
    probe,
    start_time,
    end_time,
):
    """
    Yield valid Helios files between start_time and end_time.

    Files are searched day by day so that all matching
    filenames do not need to be stored in memory.

    Parameters
    ----------
    base_dir : str
        Root directory containing the Helios year/day directories.

    probe : int or str
        Helios probe number, e.g. 1 or 2.

    start_time : datetime
        Beginning of requested interval.

    end_time : datetime
        End of requested interval.

    Yields
    ------
    str
        File path satisfying the requested time interval.
    """

    if end_time < start_time:
        raise ValueError(
            "end_time must be greater than or equal to start_time"
        )

    probe = str(probe)

    current_date = start_time.date()
    end_date = end_time.date()

    while current_date <= end_date:

        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday

        day_dir = os.path.join(
            base_dir,
            str(year),
            str(day_of_year),
        )

        if os.path.isdir(day_dir):

            files = glob.glob(os.path.join(day_dir, "*.0"))
            files += glob.glob(os.path.join(day_dir, "*.1"))

            bad_files = get_bad_files(day_dir)

            # Sort only the files from this day.
            files = sorted(
                files,
                key=lambda path: (
                    extract_datetime(os.path.basename(path))
                    or start_time
                ),
            )

            for filepath in files:

                filepath = os.path.normpath(filepath)

                if filepath in bad_files:
                    continue

                filename = os.path.basename(filepath)

                timestamp = extract_datetime(filename)
                file_probe = extract_probe(filename)

                if timestamp is None:
                    continue

                if file_probe is None:
                    continue

                if str(file_probe) != probe:
                    continue

                if start_time <= timestamp <= end_time:
                    yield filepath

        current_date += timedelta(days=1)