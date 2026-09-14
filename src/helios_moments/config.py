def load_paths(path_file):
    """
    Read a simple KEY=VALUE path configuration file.

    Parameters
    ----------
    path_file : str
        Path to the configuration file.

    Returns
    -------
    dict
        Dictionary of path names and values.
    """

    paths = {}

    with open(path_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            paths[key.strip()] = value.strip()

    return paths