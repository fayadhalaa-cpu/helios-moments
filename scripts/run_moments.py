import argparse
import csv
import os
from datetime import datetime

from helios_moments.config import load_paths
from helios_moments.file_search import find_files
from helios_moments.calculate_moments import Calculate_moments, keys, keys_ml
from helios_moments.ml_speed_cut import train_vcut_model
from helios_moments.helios_mf import extract_datetime, load_magnetic_field_day


def parse_time(time_string):
    """Parse time in YYYY-DOY HH:MM:SS format."""
    return datetime.strptime(time_string, "%Y-%j %H:%M:%S")


def get_vdf_base_dir(paths, probe):
    """Return VDF directory for the selected Helios probe."""
    probe = str(probe)
    if probe == "1":
        return paths["H1_VDF"]
    if probe == "2":
        return paths["H2_VDF"]
    raise ValueError("Probe must be 1 or 2.")


def get_last_saved_datetime(output_file):
    """Return datetime from the last data row of an existing CSV."""
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        return None

    with open(output_file, "rb") as f:
        f.seek(0, os.SEEK_END)
        position = f.tell() - 1

        while position >= 0:
            f.seek(position)
            if f.read(1) not in (b"\n", b"\r"):
                break
            position -= 1

        if position < 0:
            return None

        while position >= 0:
            f.seek(position)
            if f.read(1) == b"\n":
                position += 1
                break
            position -= 1

        f.seek(max(position, 0))
        last_line = f.readline().decode("utf-8").strip()

    if not last_line or last_line.startswith("Datetime,"):
        return None

    row = next(csv.reader([last_line]))
    try:
        return datetime.fromisoformat(row[0])
    except (ValueError, IndexError):
        return None


def open_output_file(output_file, fieldnames, resume=False):
    """Open output CSV and return file and DictWriter."""
    file_has_data = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    mode = "a" if resume and file_has_data else "w"

    f = open(output_file, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if mode == "w":
        writer.writeheader()

    return f, writer


def main():
    parser = argparse.ArgumentParser(description="Calculate proton moments from Helios VDF observations.")
    parser.add_argument("--probe", required=True, choices=["1", "2"], help="Helios probe number.")
    parser.add_argument("--start", required=True, help='Start time: "YYYY-DOY HH:MM:SS".')
    parser.add_argument("--end", required=True, help='End time: "YYYY-DOY HH:MM:SS".')
    parser.add_argument("--method", required=True, choices=["regular", "ml", "both"])
    parser.add_argument("--paths", default="data_paths.txt", help="Data-path configuration file.")
    parser.add_argument("--output", required=True, help="Output CSV filename.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted calculation.")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    start_time, end_time = parse_time(args.start), parse_time(args.end)

    if end_time < start_time:
        raise ValueError("End time must be later than or equal to start time.")

    base_dir = get_vdf_base_dir(paths, args.probe)
    do_regular = args.method in ("regular", "both")
    do_ml = args.method in ("ml", "both")

    if args.method == "both":
        root, ext = os.path.splitext(args.output)
        ext = ext or ".csv"
        regular_output, ml_output = f"{root}_regular{ext}", f"{root}_ml{ext}"
    elif args.method == "regular":
        regular_output, ml_output = args.output, None
    else:
        regular_output, ml_output = None, args.output

    last_regular_time = get_last_saved_datetime(regular_output) if args.resume and regular_output else None
    last_ml_time = get_last_saved_datetime(ml_output) if args.resume and ml_output else None

    if last_regular_time:
        print(f"Regular output will resume after: {last_regular_time}")
    if last_ml_time:
        print(f"ML output will resume after: {last_ml_time}")

    model = None
    if do_ml:
        print("Training ML speed-cut model...")
        model, *_ = train_vcut_model()
        print("ML model trained.")

    regular_file = ml_file = regular_writer = ml_writer = None
    current_day = mag6s_cache = mag4hz_cache = None
    count = skipped_count = error_count = 0

    try:
        if do_regular:
            regular_file, regular_writer = open_output_file(
                regular_output, keys, resume=args.resume
            )

        if do_ml:
            ml_file, ml_writer = open_output_file(
                ml_output, keys_ml, resume=args.resume
            )

        file_iterator = find_files(
            base_dir=base_dir, probe=args.probe,
            start_time=start_time, end_time=end_time
        )

        for datafile in file_iterator:
            file_time = extract_datetime(os.path.basename(datafile))

            do_regular_this_file = do_regular
            do_ml_this_file = do_ml

            if args.resume and last_regular_time is not None and file_time <= last_regular_time:
                do_regular_this_file = False

            if args.resume and last_ml_time is not None and file_time <= last_ml_time:
                do_ml_this_file = False

            if not do_regular_this_file and not do_ml_this_file:
                skipped_count += 1
                continue

            file_day = (args.probe, file_time.year, file_time.timetuple().tm_yday)

            if file_day != current_day:
                print("Loading MF cache for:", file_day)

                current_day, mag6s_cache, mag4hz_cache = load_magnetic_field_day(
                    datafile, paths, DEBUG_MF=False
                )

                print(
                    "MF cache loaded:",
                    len(mag6s_cache) if mag6s_cache is not None else None,
                    len(mag4hz_cache) if mag4hz_cache is not None else None
                )

            count += 1
            if count == 1 or count % 100 == 0:
                print(f"Processed {count} new files...")

            try:
                out_reg, out_ml = Calculate_moments(
                    datafile=datafile,
                    paths=paths,
                    Reg_cut=do_regular_this_file,
                    ML_cut=do_ml_this_file,
                    model=model,
                    mag6s=mag6s_cache,
                    mag4hz=mag4hz_cache
                )

                if out_reg is not None and regular_writer is not None:
                    regular_writer.writerow(out_reg)

                if out_ml is not None and ml_writer is not None:
                    ml_writer.writerow(out_ml)

            except Exception as exc:
                error_count += 1
                print(f"ERROR processing {datafile}")
                print(exc)

            if count % 10 == 0:
                if regular_file:
                    regular_file.flush()
                if ml_file:
                    ml_file.flush()

        if regular_file:
            regular_file.flush()
        if ml_file:
            ml_file.flush()

    finally:
        if regular_file:
            regular_file.close()
        if ml_file:
            ml_file.close()

    print("\nFinished.")
    print(f"New files processed: {count}")
    print(f"Already processed and skipped: {skipped_count}")
    print(f"Errors: {error_count}")

    if regular_output:
        print(f"Regular results: {regular_output}")

    if ml_output:
        print(f"ML results: {ml_output}")


if __name__ == "__main__":
    main()