import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Dictionary mapping status codes to messages explaining the codes
statusdict = {
          # Success
              1: 'Moments are calculated successfully',
          # Magnetic field problems 
              2: 'No magnetic field data available', # Proton density and velocity compenents are still computed.
              3: 'One of the magnetic field components is too low (<10^-14 nT)', # Moments are computed. 
          # Data problems 
              4: 'Empty 3D VDF', 
              5: 'No non-zero I1a data available',
              6: 'No non-zero I1b data available',
              7: 'Corrupted I1a/I1b: I1a & I1b peaks do not overlap',
              8: 'Corrupted I1a/I1b: zero peak',
              9: 'Less than 35 points available for numerical integration',
             10: 'Corrupted distribution file: Negative Counts',
             11: 'Corrupted distribution file',
             12: 'Less than 3 angular bins available in either direction',
             13: 'Proton peak not present in 3D distribution',
          # Cut not found 
             14: 'Cut not found'}

def get_ion_instrument(lines):
    # line 4
    values1 = np.fromstring(lines[3], sep=" ")
    instrument = int(values1[4])

    if instrument == 1:
        return 'I1a'
    else:
        return 'I3'

def get_radial_distance(lines):
    values = np.fromstring(lines[4], sep=" ")
    r = values[0]   # distance Helios S/C to Sun
    return r
    
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
        year = 1900 + int(match.group(1))  # 80 → 1980
        doy = int(match.group(2))
        hour = int(match.group(3))
        minute = int(match.group(4))
        second = int(match.group(5))

        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)
    return None

def read_proton_vdf(lines):
    """
    Reads proton VDF data from already-loaded file lines.
    Returns a DataFrame or None if data is missing/invalid.
    """
    start_marker = 'Maximum of distribution'
    end_marker = '2-D electron'
    data = []
    recording = False

    try:
        for line in lines:
            if start_marker in line:
                recording = True
                continue

            if end_marker in line:
                break

            if recording:
                try:
                    values = [float(v) for v in line.strip().split()]
                    data.append(values)
                except ValueError:
                    continue

        if not data:
            return None

        All_Data = pd.DataFrame(
            data,
            columns=[
                'i', 'j', 'k', 'zfp', 'counts',
                'vx', 'vy', 'vz', 'energy',
                'elevation angle', 'azimuthal angle'])

        required_cols = {'i', 'j', 'k', 'counts'}
        if not required_cols.issubset(All_Data.columns):
            return None

        All_Data[['i', 'j', 'k', 'counts']] = \
            All_Data[['i', 'j', 'k', 'counts']].astype(int)

        Proton_VDF = All_Data.iloc[:, :8].copy()

        Proton_VDF["|v|"] = np.sqrt(Proton_VDF["vx"]**2 +Proton_VDF["vy"]**2 +Proton_VDF["vz"]**2)

        return Proton_VDF

    except Exception:
        return None
    

def extract_reduced_vdf(lines,ion_instrument): 
        
    # 32 onboard integrated energy channels of protons instrument I1a
    i1aint_list = [float(v) for v in lines[16].strip().split()]
    i1aint = np.array(i1aint_list)
    i1aint_DF = pd.DataFrame(i1aint, columns=['df'])
    
    i1av_list = [float(v) for v in lines[18].strip().split()]
    i1av = np.array(i1av_list)
    i1av_DF = pd.DataFrame(i1av, columns=['v'])

    # combine v + df (so I1a has both columns)
    I1a = pd.concat([i1av_DF, i1aint_DF], axis=1)

    # 32 onboard integrated energy channels of protons instrument I1b   
    i1bint_list = [float(v) for v in lines[20].strip().split()]
    i1bint = np.array(i1bint_list)
    i1bint_DF = pd.DataFrame(i1bint, columns=['df'])
    
    i1bv_list = [float(v) for v in lines[22].strip().split()]
    i1bv = np.array(i1bv_list)
    i1bv_DF = pd.DataFrame(i1bv, columns=['v'])

    # combine v + df (so I1b has both columns)
    I1b = pd.concat([i1bv_DF, i1bint_DF], axis=1)
    
    # --- If any arrays are empty, return None ---
    if I1a.empty:
        return  None,None,5 
    if I1b.empty:
        return  None,None,6
    # if peak is zero in either I1a to I1b 
    if np.max(I1a['df'])==0 or np.max(I1b['df'])==0: 
        return  None,None,8 
    
    if ion_instrument == 'I1a':
        # If the difference between the maximum values of I1a and I1b is too
        # large, assume we have a garbled I1a distribution function
        if I1b['df'].max() / I1a['df'].max() > 10:
            return None,None,7

        #Get the peak velocity:
        v_peak_b = I1b.loc[I1b['df'].idxmax(), 'v']

        # Find nearest I1a velocity to I1b peak velocity, then compare df ratio
        idx_near = (I1a['v'] - v_peak_b).abs().idxmin()
        ratio = I1a.loc[idx_near, 'df'] / I1b['df'].max()

        if ratio < 0.01:
            return None,None,7
    else:
        if I1a['df'].max() / I1b['df'].max() > 5:
            return None,None,7
        
    return I1a, I1b,None


def estimate_start_end_time(datafile, Proton_VDF):
    """
    Estimate the start and end time of a VDF measurement based on energy bins.

    Returns:
        (datetime, datetime): Estimated start and end time of the measurement.
    """
    starttime = extract_datetime(datafile)  
    
    if starttime is None:
        raise ValueError("Could not parse datetime from filename")

    if Proton_VDF is None or Proton_VDF.empty or 'k' not in Proton_VDF.columns:
        raise ValueError("VDF data is empty or missing 'k' column")
    
    E_bin=Proton_VDF['k']
    min_dist_Ebin=int(np.min(E_bin))
    max_dist_Ebin=int(np.max(E_bin))
    dist_starttime = starttime + timedelta(seconds=min_dist_Ebin)
    dist_endtime = starttime + timedelta(seconds=max_dist_Ebin + 1)
    
    return dist_starttime, dist_endtime

def filter_3D_VDF(I1a,Proton_VDF):  

    if Proton_VDF is None:
        return None,11
    
    if Proton_VDF.empty: 
        return  None,4
    
    if Proton_VDF.shape[0]<35:
        return None,9

    # Return if any of the counts are less than zero (indicates corruped file)
    if (Proton_VDF['counts'] < 0).any():
        return None,10
    # Get rid of velocities higher than 100*velocities in the I1a 1D distribution
    Proton_VDF = Proton_VDF[Proton_VDF['|v|'] <= I1a['v'].max() * 100]
    
    # after the previous step it may become empty or less than 35 points     
    if Proton_VDF.empty: 
        return  None,4

    if Proton_VDF.shape[0]<35:
        return None,9

    # Work out number of angular bins in each direction
    n_phi_bins = Proton_VDF['i'].nunique()
    n_theta_bins = Proton_VDF['j'].nunique()
    
    # If less than a 3x3 grid, return
    if n_phi_bins < 3 or n_theta_bins < 3:
        return None,12

    # Return if the minimum velocity bin in the 3D distribution is not below
    # the velocity of the peak in I1a (assumed to be the proton peak)
    vs_3D = Proton_VDF['|v|']
    v_peak = I1a.loc[I1a['df'].idxmax(), 'v']
    if vs_3D.min() > v_peak:
        return None,13
    return Proton_VDF,None 



