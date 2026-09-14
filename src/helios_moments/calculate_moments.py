import numpy as np
import pandas as pd
from dataclasses import dataclass

from .moments_calculations import Distribution_function, Moments, Density_and_velocity
from .helios_vdfs_data import (
    get_ion_instrument,
    get_radial_distance,
    extract_reduced_vdf,
    read_proton_vdf,
    filter_3D_VDF,
)
from .regular_speed_cut import goodcut, clean3D
from .ml_speed_cut import train_vcut_model, predict_v_cut
from .helios_mf import extract_magnetic_field_values,extract_probe,extract_datetime

# -----------------------------
# Status codes for moment calculation
# -----------------------------
statusdict = {
    1: "Moments are calculated successfully",
    2: "No magnetic field data available",
    3: "One of the magnetic field components is too low (<10^-14 nT)",
    4: "Empty 3D VDF",
    5: "No non-zero I1a data available",
    6: "No non-zero I1b data available",
    7: "Corrupted I1a/I1b: I1a & I1b peaks do not overlap",
    8: "Corrupted I1a/I1b: zero peak",
    9: "Less than 35 points available for numerical integration",
    10: "Corrupted distribution file: Negative Counts",
    11: "Corrupted distribution file",
    12: "Less than 3 angular bins available in either direction",
    13: "Proton peak not present in 3D distribution",
    14: "Cut not found",
}

# -----------------------------
# Magnetic-field status codes
# -----------------------------
bstatusdict = {
    1: "Magnetic field is available",
    2: "Wobbly magnetic field",
    3: "No magnetic field data available",
    4: "One of the magnetic field components is too low (<10^-14 nT)",
}


# -----------------------------
# Output keys (Regular)
# -----------------------------
keys = [
    "Datetime", "Status", "B Status", "Ion instrument", "B instrument", "Probe", "r [au]",
    "Bx [nT]", "By [nT]", "Bz [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]",
    "B [nT]", "sigma B [nT]",
    "n_min [cm^-3]", "n_mid [cm^-3]", "n_max [cm^-3]",
    "vr_min [km/s]", "vr_mid [km/s]", "vr_max [km/s]",
    "vt_min [km/s]", "vt_mid [km/s]", "vt_max [km/s]",
    "vn_min [km/s]", "vn_mid [km/s]", "vn_max [km/s]",
    "Tpar_min [K]", "Tpar_mid [K]", "Tpar_max [K]",
    "Tperp_min [K]", "Tperp_mid [K]", "Tperp_max [K]",
    "qpar_min [W/m^2]", "qpar_mid [W/m^2]", "qpar_max [W/m^2]",
    "qperp_min [W/m^2]", "qperp_mid [W/m^2]", "qperp_max [W/m^2]",
    "v_cut_min [km/s]", "v_cut_mid [km/s]", "v_cut_max [km/s]",
]

# ML output = regular keys + one extra
keys_ml = keys + ["v_cut_std [km/s]"]


# -----------------------------
# Context container
# -----------------------------
@dataclass
class VDFContext:
    starttime: object
    probe: str
    ion_instrument: str
    B_instrument: str
    radial_distance: float
    Bx: float
    By: float
    Bz: float
    Br: float
    Bt: float
    Bn: float
    B: float
    B_std: float


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _safe_int_status(x, default=3):
    """
    Convert magnetic-field status to a plain Python int.
    Handles pandas NA / NaN / strings safely.
    """
    try:
        if pd.isna(x):
            return default
        return int(x)
    except (TypeError, ValueError):
        return default

def return_nans(status, B_status, ctx: VDFContext, keys_list):
    """
    Return a dict with all keys in keys_list.
    Fill everything with NaN except metadata + B components.
    """
    status = _safe_int_status(status, default=14)
    B_status = _safe_int_status(B_status, default=3)

    out = {k: np.nan for k in keys_list}

    out["Datetime"] = ctx.starttime
    out["Status"] = status
    out["B Status"] = B_status
    out["Ion instrument"] = ctx.ion_instrument
    out["B instrument"] = ctx.B_instrument
    out["Probe"] = ctx.probe
    out["r [au]"] = ctx.radial_distance

    out["Bx [nT]"] = ctx.Bx
    out["By [nT]"] = ctx.By
    out["Bz [nT]"] = ctx.Bz
    out["Br [nT]"] = ctx.Br
    out["Bt [nT]"] = ctx.Bt
    out["Bn [nT]"] = ctx.Bn
    out["B [nT]"] = ctx.B
    out["sigma B [nT]"] = ctx.B_std

    return out


def _transform_B_to_RTN(B_instrument, probe, Bx_raw, By_raw, Bz_raw):
    """
    Convert field from SSE to RTN.
    Returns (Br, Bt, Bn).
    """
    Br, Bt, Bn = Bx_raw, By_raw, Bz_raw

    if B_instrument == "E2":
        Br, Bt, Bn = -Br, -Bt, -Bn
        if str(probe) == "2":
            Bt, Bn = -Bt, -Bn
    elif B_instrument == "E3":
        Br, Bt, Bn = -Br, -Bt, -Bn

    return Br, Bt, Bn


def _pack_moments_into(out, moments_tuple):
    """
    moments_tuple = (densities, vx_all, vy_all, vz_all, Tpar, Tperp, qpar, qperp)
    """
    densities, vx_all, vy_all, vz_all, Tpar, Tperp, qpar, qperp = moments_tuple

    out["n_min [cm^-3]"] = densities[0]
    out["n_mid [cm^-3]"] = densities[1]
    out["n_max [cm^-3]"] = densities[2]

    out["vr_min [km/s]"] = vx_all[0]
    out["vr_mid [km/s]"] = vx_all[1]
    out["vr_max [km/s]"] = vx_all[2]

    out["vt_min [km/s]"] = vy_all[0]
    out["vt_mid [km/s]"] = vy_all[1]
    out["vt_max [km/s]"] = vy_all[2]

    out["vn_min [km/s]"] = vz_all[0]
    out["vn_mid [km/s]"] = vz_all[1]
    out["vn_max [km/s]"] = vz_all[2]

    out["Tpar_min [K]"] = Tpar[0]
    out["Tpar_mid [K]"] = Tpar[1]
    out["Tpar_max [K]"] = Tpar[2]

    out["Tperp_min [K]"] = Tperp[0]
    out["Tperp_mid [K]"] = Tperp[1]
    out["Tperp_max [K]"] = Tperp[2]

    out["qpar_min [W/m^2]"] = qpar[0]
    out["qpar_mid [W/m^2]"] = qpar[1]
    out["qpar_max [W/m^2]"] = qpar[2]

    out["qperp_min [W/m^2]"] = qperp[0]
    out["qperp_mid [W/m^2]"] = qperp[1]
    out["qperp_max [W/m^2]"] = qperp[2]

    return out


def Calculate_moments(datafile,paths, Reg_cut=False, ML_cut=False, model=None, labeled_pkl=None, mag6s=None, mag4hz=None):
    """
    Calculate proton velocity-distribution-function (VDF) moments for one
    measurement.

    The function reads the ions VDF and corresponding magnetic field data,
    applies the requested speed-cut method, and calculates the proton moments.
    The regular and machine-learning speed-cut methods can be used separately
    or together.

    Parameters
    ----------
    datafile : str or path-like
        Path to the Helios ion VDF datafile.
    paths : dict
        Dictionary containing the paths to the Helios VDF and magnetic field
        data directories.
    Reg_cut : bool, optional
        If True, calculate moments using the regular speed-cut method.
        Default is False.
    ML_cut : bool, optional
        If True, calculate moments using the machine-learning speed-cut method.
        Default is False.
    model : sklearn.ensemble.RandomForestRegressor, optional
        Pre-trained model used by the machine-learning speed-cut method.
        If None and ML_cut is True, a model is trained using the supplied or
        packaged training data. Default is None.
    labeled_pkl : str or path-like, optional
        Path to a labeled training-data file for the
        machine-learning speed-cut model. If None, the packaged training data
        are used. Default is None.

    Returns
    -------
    out_reg : dict or None
        Dictionary containing the moments and associated quantities calculated
        using the regular speed-cut method. None if Reg_cut is False.
    out_ml : dict or None
        Dictionary containing the moments and associated quantities calculated
        using the machine-learning speed-cut method. None if ML_cut is False.
    """
    # -----------------------------
    # import Datetime 
    # -----------------------------    
    starttime=extract_datetime(datafile)
    
    # -----------------------------
    # import Probe 
    # -----------------------------
    probe=extract_probe(datafile)
    
    # -----------------------------
    # Import the VDF file 
    # -----------------------------
    with open(datafile, "r") as f:
        lines = f.readlines()
        
    Proton_VDF = read_proton_vdf(lines)

    # -----------------------------
    # import magnetic field 
    # ----------------------------- 
    Bx_raw, By_raw, Bz_raw, B, B_std, B_instrument, B_status,_ = extract_magnetic_field_values(datafile, paths, Proton_VDF=Proton_VDF,mag6s=mag6s, mag4hz=mag4hz)
 
    #Bx_raw, By_raw, Bz_raw, B, B_std, B_instrument, B_status=extract_magnetic_field_values(datafile,paths,Proton_VDF=Proton_VDF)
    
    # -----------------------------
    # Cnovert magnetic field from nT to Gauss
    # ----------------------------- 
    Bx_raw =Bx_raw* 1e-5 
    By_raw =By_raw* 1e-5
    Bz_raw =Bz_raw* 1e-5
    B =B* 1e-5
    B_std= B_std* 1e-5
    
    
    # -----------------------------
    # import radial distance, ion instrument, transform B to RTN  
    # ----------------------------- 
    
    radial_distance = get_radial_distance(lines)
    ion_instrument = get_ion_instrument(lines)
    Br, Bt, Bn = _transform_B_to_RTN(B_instrument, probe, Bx_raw, By_raw, Bz_raw)


    ctx = VDFContext(
        starttime=starttime,
        probe=probe,
        ion_instrument=ion_instrument,
        B_instrument=B_instrument,
        radial_distance=radial_distance,
        Bx=np.nan if pd.isna(Bx_raw) else Bx_raw * 1e5,
        By=np.nan if pd.isna(By_raw) else By_raw * 1e5,
        Bz=np.nan if pd.isna(Bz_raw) else Bz_raw * 1e5,
        Br=np.nan if pd.isna(Br) else Br * 1e5,
        Bt=np.nan if pd.isna(Bt) else Bt * 1e5,
        Bn=np.nan if pd.isna(Bn) else Bn * 1e5,
        B=np.nan if pd.isna(B) else B * 1e5,
        B_std=np.nan if pd.isna(B_std) else B_std * 1e5,
    )

    # -----------------------------
    # Step 1: load reduced + 3D VDF
    # -----------------------------
    I1a, I1b, status = extract_reduced_vdf(lines, ion_instrument)
    if status is not None:
        out_reg = return_nans(status, B_status, ctx, keys) if Reg_cut else None
        out_ml = return_nans(status, B_status, ctx, keys_ml) if ML_cut else None
        if out_ml is not None:
            out_ml["v_cut_std [km/s]"] = np.nan
        return out_reg, out_ml

    if Proton_VDF is None:
        out_reg = return_nans(9, B_status, ctx, keys) if Reg_cut else None
        out_ml = return_nans(9, B_status, ctx, keys_ml) if ML_cut else None
        if out_ml is not None:
            out_ml["v_cut_std [km/s]"] = np.nan
        return out_reg, out_ml

    # -----------------------------
    # Step 2: filter 3D VDF
    # -----------------------------
    Proton_VDF, status = filter_3D_VDF(I1a, Proton_VDF)
    if status is not None:
        out_reg = return_nans(status, B_status, ctx, keys) if Reg_cut else None
        out_ml = return_nans(status, B_status, ctx, keys_ml) if ML_cut else None
        if out_ml is not None:
            out_ml["v_cut_std [km/s]"] = np.nan
        return out_reg, out_ml

    # ==================================================
    # Regular cut branch
    # ==================================================
    out_reg = None

    if Reg_cut:
        v_cut_reg, _delta = goodcut(I1a, I1b)

        if not np.isfinite(v_cut_reg):
            out_reg = return_nans(14, B_status, ctx, keys)
        else:
            min_cut, mid_cut, max_cut, Vmin, Vmid, Vmax, status = clean3D(I1a, v_cut_reg, Proton_VDF)

            if status is not None:
                out_reg = return_nans(status, B_status, ctx, keys)

            elif B_status in ("2", "3"): 
                # field is discarded either because it is wobbly or not measued (calculate n and v)
                dist = Distribution_function(Vmin, Vmid, Vmax)
                moments_tuple = Density_and_velocity(dist)

                out_reg = return_nans(2, B_status, ctx, keys)
                out_reg["v_cut_min [km/s]"] = min_cut
                out_reg["v_cut_mid [km/s]"] = mid_cut
                out_reg["v_cut_max [km/s]"] = max_cut
                out_reg = _pack_moments_into(out_reg, moments_tuple)

            elif B_status=='4':
                # one of the field components is too low, but calculate moments anyway
                dist = Distribution_function(Vmin, Vmid, Vmax)
                moments_tuple = Moments(dist, Br, Bt, Bn, B)

                out_reg = return_nans(3, B_status, ctx, keys)
                out_reg["v_cut_min [km/s]"] = min_cut
                out_reg["v_cut_mid [km/s]"] = mid_cut
                out_reg["v_cut_max [km/s]"] = max_cut
                out_reg = _pack_moments_into(out_reg, moments_tuple)

            else:
                dist = Distribution_function(Vmin, Vmid, Vmax)
                moments_tuple = Moments(dist, Br, Bt, Bn, B)

                out_reg = return_nans(1, B_status, ctx, keys)
                out_reg["v_cut_min [km/s]"] = min_cut
                out_reg["v_cut_mid [km/s]"] = mid_cut
                out_reg["v_cut_max [km/s]"] = max_cut
                out_reg = _pack_moments_into(out_reg, moments_tuple)

    if not ML_cut:
        return out_reg, None

    # ==================================================
    # ML cut branch
    # ==================================================
    if model is None:
        model, *_ = train_vcut_model(labeled_pkl)

    v_cut_ml, v_cut_std = predict_v_cut(I1a, I1b, model)

    if not np.isfinite(v_cut_ml):
        out_ml = return_nans(14, B_status, ctx, keys_ml)
        out_ml["v_cut_std [km/s]"] = np.nan
        return out_reg, out_ml

    min_cut, mid_cut, max_cut, Vmin, Vmid, Vmax, status = clean3D(I1a, v_cut_ml, Proton_VDF)

    if status is not None:
        out_ml = return_nans(status, B_status, ctx, keys_ml)
        out_ml["v_cut_std [km/s]"] = v_cut_std
        return out_reg, out_ml

    elif B_status in ("2", "3"):
        dist_ml = Distribution_function(Vmin, Vmid, Vmax)
        moments_tuple_ml = Density_and_velocity(dist_ml)

        out_ml = return_nans(2, B_status, ctx, keys_ml)
        out_ml["v_cut_std [km/s]"] = v_cut_std
        out_ml["v_cut_min [km/s]"] = min_cut
        out_ml["v_cut_mid [km/s]"] = mid_cut
        out_ml["v_cut_max [km/s]"] = max_cut
        out_ml = _pack_moments_into(out_ml, moments_tuple_ml)

    elif B_status=='4':
        dist_ml = Distribution_function(Vmin, Vmid, Vmax)
        moments_tuple_ml = Moments(dist_ml, Br, Bt, Bn, B)

        out_ml = return_nans(3, B_status, ctx, keys_ml)
        out_ml["v_cut_std [km/s]"] = v_cut_std
        out_ml["v_cut_min [km/s]"] = min_cut
        out_ml["v_cut_mid [km/s]"] = mid_cut
        out_ml["v_cut_max [km/s]"] = max_cut
        out_ml = _pack_moments_into(out_ml, moments_tuple_ml)

    else:
        dist_ml = Distribution_function(Vmin, Vmid, Vmax)
        moments_tuple_ml = Moments(dist_ml, Br, Bt, Bn, B)

        out_ml = return_nans(1, B_status, ctx, keys_ml)
        out_ml["v_cut_std [km/s]"] = v_cut_std
        out_ml["v_cut_min [km/s]"] = min_cut
        out_ml["v_cut_mid [km/s]"] = mid_cut
        out_ml["v_cut_max [km/s]"] = max_cut
        out_ml = _pack_moments_into(out_ml, moments_tuple_ml)

    return out_reg, out_ml