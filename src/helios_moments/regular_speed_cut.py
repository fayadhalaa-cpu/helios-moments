import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def check_depression(vr, I1a_arr, I1b_arr, min_depression_ratio=0.001):
    
    #Remove noise from the data   
    vr=vr[3:-1]
    I1a_arr=I1a_arr[3:-1]
    I1b_arr=I1b_arr[3:-1]
    
    smoothed_l1a = gaussian_filter1d(I1a_arr, sigma=0.05)
    smoothed_l1b = gaussian_filter1d(I1b_arr, sigma=0.05)
    
    peaks, _ = find_peaks(I1a_arr) # this returns the indecis of the local maxima 
    if len(peaks) < 2:
        return None, None, None
    sorted_peaks = sorted(peaks, key=lambda p: smoothed_l1a[p], reverse=True) #lambda is an anonymous function used as a key for sorting 
    #Reverse=True: descending sorting, so the highest peak in I1a comes first 
    proton_peak = sorted_peaks[0] 

    # Find alpha peak (selects only peaks that are right to the protons peak)
    alpha_peak_candidates = [p for p in sorted_peaks[1:]
                             if vr[p] > vr[proton_peak] and smoothed_l1b[p] > smoothed_l1a[p]]
 
    if not alpha_peak_candidates:
        return None,None,None

    alpha_peak = alpha_peak_candidates[0]

    #Regect if the apha peak is too far in velocity

    if vr[alpha_peak] >= 1.45 * vr[proton_peak]:
        return None,None,None    

    #Reject isolated alpha peaks  
    if alpha_peak > 0 and alpha_peak < len(smoothed_l1b) - 1:
        if smoothed_l1b[alpha_peak - 1] == 0 and smoothed_l1b[alpha_peak + 1] == 0:
            return None,None,None
    else:
        return None,None,None
    
# Find the depression between the two peaks 
    start = min(proton_peak, alpha_peak)
    end = max(proton_peak, alpha_peak)
    valley_index = np.argmin(I1a_arr[start:end+1]) + start
    valley_value = I1a_arr[valley_index]
  # Calculate the ratio of the depression 
    depression = (I1a_arr[alpha_peak] - valley_value) / I1a_arr[alpha_peak]
    #depression_to_alpha = (I1a_arr[alpha_peak] - valley_value) / I1a_arr[alpha_peak]

    if depression >= min_depression_ratio:
        return proton_peak+3, alpha_peak+3, valley_index+3  #adjust index 
    else:
        return None,None,None
    
def speed_cut(I1a,I1b):
    v = I1a['v'].values
    I1a_arr = I1a['df'].values
    I1b_arr = I1a_I1b = np.interp(I1a.index.values, I1b.index.values, I1b['df'].values) 

    proton_peak, alpha_peak, valley_index = check_depression(v, I1a_arr, I1b_arr)
    if valley_index is None:
        return np.nan
    else:
        return v[valley_index]

def find_speed_cut1(I1a,I1b, min_speed=0): #The cut defined by |δf|/f

    I1a_I1b = np.interp(I1a.index.values, I1b.index.values, I1b['df'].values) 
    i1av=I1a['v']
    I1a_df=np.array(I1a['df'])
    
    # Find max f
    f_max = np.max(I1a_df)

    # Find min speed index
    peak_1D_v_idx = np.nanargmax(I1a_df)
    min_speed = max(peak_1D_v_idx, min_speed)
    min_speed_idx = np.nanargmin(np.abs(np.arange(len(I1a_df)) - min_speed))
    
    # Compute maxima for normalization
    f_max = np.max(I1a['df'].values)
    f_max_b = np.max(I1a_I1b)
    
    # Smooth the data with Gaussian filter
    smoothed_l1a = gaussian_filter1d(I1a['df'], sigma=0.6)
    smoothed_l1b = gaussian_filter1d(I1a_I1b, sigma=0.6)
    
    f_max_I1a_sm = np.max(smoothed_l1a)  
    
    if not np.isfinite(f_max_I1a_sm) or f_max_I1a_sm == 0:
        return np.nan, np.nan
    # Calculate delta_f/f
    deltaf_f = abs(f_max - I1a_df[min_speed_idx:]) / f_max 
    deltaf_f_sm= abs(f_max_I1a_sm - smoothed_l1a[min_speed_idx:]) /  f_max_I1a_sm
    
    first_drop_index = None
    for i in range(1, len(deltaf_f_sm)):
        if deltaf_f_sm[i] < deltaf_f_sm[i - 1]:  # Look for first decrease
            first_drop_index = i - 1
            break

    # If no decrease was found, return NaN values
    if first_drop_index is None:
        return np.nan, np.nan

    # Make sure first_drop_index is valid before using it
    if first_drop_index + min_speed_idx >= len(i1av):
        return np.nan, np.nan

    # Compute speed cut
    deltaf_f1 = deltaf_f_sm[first_drop_index]
    speed_cut1 = i1av[first_drop_index + min_speed_idx]
    
    # Check if Speed cut good 
    cut1_ind = I1a.loc[I1a['v'] == speed_cut1].index[0]
    f_cut1a = I1a['df'][cut1_ind]
    f_array = np.array([I1a['df'][cut1_ind - 1],I1a['df'][cut1_ind], I1a['df'][cut1_ind + 1]])
    if (0.75 <= deltaf_f1 < 1  and (f_array[0] != 0 and f_array[1] != 0 and f_array[2] != 0)  and (cut1_ind < len(i1av) - 3)):
        return speed_cut1, deltaf_f1
    else:
        return np.nan,np.nan

def find_speed_cut2(I1a,I1b, min_speed=0):
    I1a_df = I1a['df'].values  
    i1av   =I1a['v']
# Find Peak Velocity index and value 
    peak_1D_v_idx = np.nanargmax(I1a_df) #index of the maximum value of DF 
    peak_1D_v = I1a.index.values[peak_1D_v_idx] # also returns the index. 

    min_speed = max(peak_1D_v, min_speed) # We adjust the minimum speed to be the 'index' of the peak speed extracted from i1a 
    min_speed_idx = np.nanargmin(np.abs(I1a.index.values - min_speed)) 
    I1a_I1b = np.interp(I1a.index.values, I1b.index.values, I1b['df'].values) 
    
    with np.errstate(divide='ignore', invalid='ignore'): # np.errstate is used to ignore divide-by-zero and invalid operation warnings.
        ratios = I1a_df / I1a_I1b # the ratio of the original DF of i1a over the interpolated (here the same as I1b)
        ratios = ratios / ratios[peak_1D_v_idx] # the ratios normalized to the DF of the peak. 
        threshold = 0.6
        good_ratio_idx = ((ratios[min_speed_idx:] < threshold) &
                          (ratios[min_speed_idx:] > 0.2))
        
    if not np.any(good_ratio_idx):
        return np.nan,np.nan

    ratio_idx = min_speed_idx + np.nanargmax(good_ratio_idx)
    speed_cut_ind = I1a['v'].index.values[ratio_idx]
    speed_cut2=i1av[speed_cut_ind]
    
    f_max=np.max(I1a['df'].values)
    cut_ind = I1a.index[I1a['v'] == speed_cut2]
    f_cut=I1a_df[cut_ind]
    
    #delta_f/f 
    
    deltaf_f2=abs(f_max-f_cut)/(f_max)
    
    if deltaf_f2 >= 0.75:
        if cut_ind < len(i1av) - 3:
            return speed_cut2, deltaf_f2
        else:
            return np.nan, np.nan

    if deltaf_f2 < 0.75:
        deltaf_f_arr = abs(f_max - I1a_df[min_speed_idx:]) / f_max
        valid_cuts = np.where(deltaf_f_arr >= 0.75)[0]

        if len(valid_cuts) > 0:
            new_cut_idx = min_speed_idx + valid_cuts[0]
            speed_cut2 = i1av[new_cut_idx]
            deltaf_f2 = deltaf_f_arr[valid_cuts[0]]

            if new_cut_idx >= len(i1av) - 3:
                return np.nan, np.nan
        else:
            return np.nan, np.nan

    return speed_cut2, deltaf_f2
def find_speed_cut3(I1a,I1b):
    min_speed=0
    
    # Interpolate I1b onto I1a's velocity grid
    I1a_I1b = np.interp(I1a.index.values, I1b.index.values, I1b['df'].values) 
    I1a_df  = I1a['df'].values
    i1av    =  I1a['v']
    
    peak_1D_v_idx = np.nanargmax(I1a_df) #index of the maximum value of DF 
    peak_1D_v = I1a.index.values[peak_1D_v_idx] # also returns the index. 

    min_speed = max(peak_1D_v, min_speed) # We adjust the minimum speed to be the 'index' of the peak speed extracted from i1a 
    min_speed_idx = np.nanargmin(np.abs(I1a.index.values - min_speed)) 
    
    # Compute maxima for normalization
    f_max = np.max(I1a['df'].values)
    f_max_b = np.max(I1a_I1b)
    
    # Smooth the data with Gaussian filter
    smoothed_l1a = gaussian_filter1d(I1a['df'], sigma=0.6)
    smoothed_l1b = gaussian_filter1d(I1a_I1b, sigma=0.6)
    
    # Normalize the smoothed data
    
    if not np.isfinite(f_max) or f_max == 0:
        norm_I1a = np.full_like(smoothed_l1a, np.nan)
    else:
        norm_I1a = smoothed_l1a / f_max 
    if not np.isfinite(f_max_b) or f_max_b == 0:
        norm_I1b = np.full_like(smoothed_l1b, np.nan)
    else:
        norm_I1b = smoothed_l1b / f_max_b    

    # Step 1: Find the index of the maximum point in smoothed_l1a
    max_idx = np.argmax(smoothed_l1a)
    
    # Step 2: Find the first index where norm_I1b > norm_I1a, only after the maximum
    crossover_indices = np.where(norm_I1b[max_idx:] > (norm_I1a[max_idx:]*1.41))[0]
    if len(crossover_indices) == 0:
        return np.nan, np.nan  # No crossover found after the maximum
    
    crossover_idx = crossover_indices[0] + max_idx  # Adjust index to original array
    
    # Step 3: Map the crossover index to the velocity grid using I1a['v']
    i1av_indecies = I1a['v'].index.values
    cut_ind = i1av_indecies[crossover_idx]
    speed_cut3 = I1a['v'][cut_ind]
    f_cut=I1a_df[cut_ind]
    
    #delta_f/f 
    deltaf_f3=abs(f_max-f_cut)/(f_max)

    if deltaf_f3 >= 0.75:
        if cut_ind < len(i1av) - 3:
            return speed_cut3, deltaf_f3
        else:
            return np.nan, np.nan

    if deltaf_f3 < 0.75:
        deltaf_f_arr = abs(f_max - I1a_df[min_speed_idx:]) / f_max
        valid_cuts = np.where(deltaf_f_arr >= 0.75)[0]

        if len(valid_cuts) > 0:
            new_cut_idx = min_speed_idx + valid_cuts[0]
            speed_cut3 = i1av[new_cut_idx]
            deltaf_f3 = deltaf_f_arr[valid_cuts[0]]

            if new_cut_idx >= len(i1av) - 3:
                return np.nan, np.nan
        else:
            return np.nan, np.nan

    return speed_cut3, deltaf_f3

def goodcut(I1a,I1b): 
    speed_cut1, deltaf_f1 = find_speed_cut1(I1a,I1b, min_speed=0)
    speed_cut2, deltaf_f2 = find_speed_cut2(I1a,I1b, min_speed=0)
    speed_cut3, deltaf_f3 = find_speed_cut3(I1a,I1b)
    speed_cut4= speed_cut(I1a,I1b)
    deltaf_f4= 'Valley cut'
    # ----------------
    # Decision logic
    # ----------------
    if not np.isnan(speed_cut4):
        return speed_cut4, deltaf_f4 
    elif not np.isnan(speed_cut1):
        return speed_cut1, deltaf_f1
    elif not np.isnan(speed_cut2):
        return speed_cut2, deltaf_f2
    elif not np.isnan(speed_cut3):
        return speed_cut3, deltaf_f3
    else:
        return np.nan,np.nan

        
        
def clean3D(I1a,v_cut,Proton_VDF):   
    # Find the index of the closest value to v_cut
    i1av=I1a['v']
    cut_ind = (I1a['v'] - v_cut).abs().idxmin()
    min_cut_ind=cut_ind-1
    max_cut_ind=cut_ind+1 
    min_cut=i1av[min_cut_ind]
    max_cut=i1av[max_cut_ind]
      
    Proton_VDF =Proton_VDF[Proton_VDF['counts'] != 1]

    # A handful of files seem to have some garbage counts in them
    Proton_VDF= Proton_VDF[Proton_VDF['counts'] < 32768]

    # Throw away high and low energy bins
    # (which contain just noise)
    Proton_VDF= Proton_VDF[Proton_VDF ['k'] > 3]
    Proton_VDF= Proton_VDF[Proton_VDF ['k'] < 32]
    
    #remove alpha particles 
    Proton_VDF_min = Proton_VDF[Proton_VDF['|v|'] <= min_cut]
    Proton_VDF_mid = Proton_VDF[Proton_VDF['|v|'] <= v_cut]
    Proton_VDF_max = Proton_VDF[Proton_VDF['|v|'] <= max_cut]
    
    if any(v is None or v.empty or v.shape[0] < 35
       for v in [Proton_VDF_min, Proton_VDF_mid, Proton_VDF_max]):
        return None,None,None, None, None,None,9
    
    return min_cut,v_cut,max_cut, Proton_VDF_min, Proton_VDF_mid, Proton_VDF_max,None