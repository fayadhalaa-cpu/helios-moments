import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from numba import njit

# Fixed seed for reproducible Monte Carlo sampling
np.random.seed(0)

def Distribution_function(Proton_VDF_min, Proton_VDF_mid, Proton_VDF_max): #Monte Carlo no 

    def process_vdf_monte_carlo(Proton_VDF):
        # Distribution function in s**3 / cm**6  
        Proton_VDF['zfp'] = Proton_VDF['zfp'] * 10**30 # Convert it to be in s**3 / km**6  
        
        # Extract velocity components and VDF values
        vx = Proton_VDF.iloc[:, 5].to_numpy(dtype=np.float64)
        vy = Proton_VDF.iloc[:, 6].to_numpy(dtype=np.float64)
        vz = Proton_VDF.iloc[:, 7].to_numpy(dtype=np.float64)
        zfp = Proton_VDF.iloc[:, 3].to_numpy(dtype=np.float64)
        # combine velocities into a 3D array, each row is a velocity vector 
        points = np.vstack((vx, vy, vz)).T
        
        # Define the full velocity box
        Vx_min, Vx_max = min(vx), max(vx)
        Vy_min, Vy_max = min(vy), max(vy)
        Vz_min, Vz_max = min(vz), max(vz)
        box_volume = (Vx_max - Vx_min) * (Vy_max - Vy_min) * (Vz_max - Vz_min)
        
        tree = KDTree(points)
        # This is a spatial data structure that organizes the 3D velocity points for efficient nearest-neighbor searches.
        
        # Monte Carlo sampling
        n_samples = 100000
        samples = np.random.uniform([Vx_min, Vy_min, Vz_min], [Vx_max, Vy_max, Vz_max], (n_samples, 3)) #each row is a random velocity vector
        vx_samples, vy_samples, vz_samples = samples[:, 0], samples[:, 1], samples[:, 2]
        _, indices = tree.query(samples) # finds the index of the nearest known velocity point from the original dataset 
        # query returns two outputs, distances (between the original data point and the random sample)
        # and indices of the closest original velocity point for that random sample.
        distribution_fn = zfp[indices]  # Sampled zfp values
        
        
        return distribution_fn, box_volume, vx_samples, vy_samples, vz_samples
    
  
    distribution_min, volume_min, vx_min, vy_min, vz_min = process_vdf_monte_carlo(Proton_VDF_min)
    distribution_mid, volume_mid, vx_mid, vy_mid, vz_mid = process_vdf_monte_carlo(Proton_VDF_mid)
    distribution_max, volume_max, vx_max, vy_max, vz_max = process_vdf_monte_carlo(Proton_VDF_max)
    
   
    return (
        (distribution_min, volume_min, vx_min, vy_min, vz_min),
        (distribution_mid, volume_mid, vx_mid, vy_mid, vz_mid),
        (distribution_max, volume_max, vx_max, vy_max, vz_max)
    )
@njit
def _moment_sums(f, cx, cy, cz):
    s_xx = s_xy = s_xz = s_yy = s_yz = s_zz = 0.0

    s_xxx = s_xxy = s_xxz = 0.0
    s_xyy = s_xyz = s_xzz = 0.0
    s_yyy = s_yyz = s_yzz = s_zzz = 0.0

    for i in range(len(f)):
        fi = f[i]
        x = cx[i]
        y = cy[i]
        z = cz[i]

        xx = x*x
        yy = y*y
        zz = z*z

        s_xx += fi*xx
        s_xy += fi*x*y
        s_xz += fi*x*z
        s_yy += fi*yy
        s_yz += fi*y*z
        s_zz += fi*zz

        s_xxx += fi*xx*x
        s_xxy += fi*xx*y
        s_xxz += fi*xx*z
        s_xyy += fi*x*yy
        s_xyz += fi*x*y*z
        s_xzz += fi*x*zz
        s_yyy += fi*yy*y
        s_yyz += fi*yy*z
        s_yzz += fi*y*zz
        s_zzz += fi*zz*z

    return (
        s_xx, s_xy, s_xz, s_yy, s_yz, s_zz,
        s_xxx, s_xxy, s_xxz, s_xyy, s_xyz,
        s_xzz, s_yyy, s_yyz, s_yzz, s_zzz,
    )
def Moments(distributions, Bx, By, Bz, B):

    proton_mass=1.6726*10**(-24) # proton mass in grams
    
    densities = []
    vx_all, vy_all, vz_all = [], [], []
    parallel_Temps, perp_Temps = [], []
    parallel_heat_flux, perp_heat_flux = [], []
    
    for distribution_fn, volume, vx_samples, vy_samples, vz_samples in distributions:
        
        density = np.mean(distribution_fn) * volume * (10**-15)

        N = len(distribution_fn)

        ux = (np.dot(distribution_fn, vx_samples) / N * volume) / (density * 1e15)
        uy = (np.dot(distribution_fn, vy_samples) / N * volume) / (density * 1e15)
        uz = (np.dot(distribution_fn, vz_samples) / N * volume) / (density * 1e15)
        bx=Bx/B
        by=By/B
        bz=Bz/B
    
        cx = vx_samples-ux
        cy = vy_samples-uy
        cz = vz_samples-uz

        (s_xx, s_xy, s_xz, s_yy, s_yz, s_zz,
            s_xxx, s_xxy, s_xxz, s_xyy, s_xyz,
            s_xzz, s_yyy, s_yyz, s_yzz, s_zzz,) = _moment_sums(distribution_fn, cx, cy, cz)
        # Pressure tensor
        # The second-order tensor <f c_i c_j> is symmetric,
        # so only 6 unique components need to be calculated.
        # multiplying by 10^-30 is to convert vdf from km^-6 to cm^-6 
        # and multiplying by 10^10  is to convert velocities from km/s to cm/s, so all units are consistent.
        C2_scale = 1e-20
        P_scale = volume * 1e15 * proton_mass  # multiplying volume by 10^15 is to convert the volume from km^3 -> cm^3

        m_xx = (s_xx / N) * C2_scale
        m_xy = (s_xy / N) * C2_scale
        m_xz = (s_xz / N) * C2_scale
        m_yy = (s_yy / N) * C2_scale
        m_yz = (s_yz / N) * C2_scale
        m_zz = (s_zz / N) * C2_scale

        pxx = m_xx * P_scale
        pxy = m_xy * P_scale
        pxz = m_xz * P_scale
        pyx = pxy
        pyy = m_yy * P_scale
        pyz = m_yz * P_scale
        pzx = pxz
        pzy = pyz
        pzz = m_zz * P_scale        
        
        kb=1.3807*(10**-16) #erg/k 

        p_parallel= (pxx*bx*bx)+(pxy*bx*by)+(pxz*bx*bz)+(pyx*by*bx)+(pyy*by*by)+(pyz*by*bz)+(pzx*bz*bx)+(pzy*bz*by)+(pzz*bz*bz)

        T_parallel=p_parallel/(density*kb)

        p_perp=0.5*((pxx*(1-(bx*bx)))-(pxy*bx*by)-(pxz*bx*bz)-(pyx*by*bx)+(pyy*(1-(by*by)))-(pyz*by*bz)-(pzx*bz*bx)-(pzy*bz*by)+(pzz*(1-(bz*bz))))

        T_perp=p_perp/(density*kb)

        # Heat flux calculations
        # The third-order tensor <f c_i c_j c_k> is symmetric because
        # multiplication of cx, cy, cz is commutative.
        # Therefore, only 10 unique components need to be calculated
        # instead of all 27 components.

        # N was already defined above for the bulk velocity calculation:
        # N = len(distribution_fn)
        
      # volume * 1e9 converts velocity-space volume from km^3 -> m^3.
      # 1e-18 converts the VDF from km^-6 to m^-6.
      # proton_mass is defined in grams, so *0.001 converts g -> kg.
        C3_scale = 1e9 * 1e-18

        m_xxx = (s_xxx / N) * C3_scale
        m_xxy = (s_xxy / N) * C3_scale
        m_xxz = (s_xxz / N) * C3_scale
        m_xyy = (s_xyy / N) * C3_scale
        m_xyz = (s_xyz / N) * C3_scale
        m_xzz = (s_xzz / N) * C3_scale
        m_yyy = (s_yyy / N) * C3_scale
        m_yyz = (s_yyz / N) * C3_scale
        m_yzz = (s_yzz / N) * C3_scale
        m_zzz = (s_zzz / N) * C3_scale

        q_scale = volume * 1e9 * (proton_mass * 0.001)

        qxxx = m_xxx * q_scale
        qxxy = m_xxy * q_scale
        qxxz = m_xxz * q_scale
        qxyy = m_xyy * q_scale
        qxyz = m_xyz * q_scale
        qxzz = m_xzz * q_scale
        qyyy = m_yyy * q_scale
        qyyz = m_yyz * q_scale
        qyzz = m_yzz * q_scale
        qzzz = m_zzz * q_scale        
       
        # Use symmetry of q_ijk to obtain the remaining components

        # xxy permutations
        qxyx = qxxy
        qyxx = qxxy

        # xxz permutations
        qxzx = qxxz
        qzxx = qxxz

        # xyy permutations
        qyxy = qxyy
        qyyx = qxyy

        # xyz permutations
        qxzy = qxyz
        qyxz = qxyz
        qyzx = qxyz
        qzxy = qxyz
        qzyx = qxyz

        # xzz permutations
        qzxz = qxzz
        qzzx = qxzz

        # yyz permutations
        qyzy = qyyz
        qzyy = qyyz

        # yzz permutations
        qzyz = qyzz
        qzzy = qyzz


        # Define q_parallel

        # (q:bb)_x
        qbbx = (
            (qxxx*bx*bx) + (qxyx*bx*by) + (qxzx*bx*bz)
            + (qyxx*by*bx) + (qyyx*by*by) + (qyzx*by*bz)
            + (qzxx*bz*bx) + (qzyx*bz*by) + (qzzx*bz*bz)
        )

        # (q:bb)_y
        qbby = (
            (qxxy*bx*bx) + (qxyy*bx*by) + (qxzy*bx*bz)
            + (qyxy*by*bx) + (qyyy*by*by) + (qyzy*by*bz)
            + (qzxy*bz*bx) + (qzyy*bz*by) + (qzzy*bz*bz)
        )

        # (q:bb)_z
        qbbz = (
            (qxxz*bx*bx) + (qxyz*bx*by) + (qxzz*bx*bz)
            + (qyxz*by*bx) + (qyyz*by*by) + (qyzz*by*bz)
            + (qzxz*bz*bx) + (qzyz*bz*by) + (qzzz*bz*bz)
        )

        q_parallel = (qbbx*bx) + (qbby*by) + (qbbz*bz)


        # Define q_perp

        # (q:(1-bb)/2)_x
        q1_bbx = 0.5 * (
            (qxxx*(1-bx*bx)) + (qxyx*(1-bx*by)) + (qxzx*(1-bx*bz))
            + (qyxx*(1-by*bx)) + (qyyx*(1-by*by)) + (qyzx*(1-by*bz))
            + (qzxx*(1-bz*bx)) + (qzyx*(1-bz*by)) + (qzzx*(1-bz*bz))
        )

        # (q:(1-bb)/2)_y
        q1_bby = 0.5 * (
            (qxxy*(1-bx*bx)) + (qxyy*(1-bx*by)) + (qxzy*(1-bx*bz))
            + (qyxy*(1-by*bx)) + (qyyy*(1-by*by)) + (qyzy*(1-by*bz))
            + (qzxy*(1-bz*bx)) + (qzyy*(1-bz*by)) + (qzzy*(1-bz*bz))
        )

        # (q:(1-bb)/2)_z
        q1_bbz = 0.5 * (
            (qxxz*(1-bx*bx)) + (qxyz*(1-bx*by)) + (qxzz*(1-bx*bz))
            + (qyxz*(1-by*bx)) + (qyyz*(1-by*by)) + (qyzz*(1-by*bz))
            + (qzxz*(1-bz*bx)) + (qzyz*(1-bz*by)) + (qzzz*(1-bz*bz))
        )

        q_perp = (q1_bbx*bx) + (q1_bby*by) + (q1_bbz*bz)        
        densities.append(density)
        vx_all.append(ux)
        vy_all.append(uy)
        vz_all.append(uz)
        parallel_Temps.append(T_parallel)
        perp_Temps.append(T_perp)
        parallel_heat_flux.append(q_parallel)
        perp_heat_flux.append(q_perp)

    return densities, vx_all, vy_all, vz_all, parallel_Temps, perp_Temps, parallel_heat_flux, perp_heat_flux


def Density_and_velocity(distributions):
    """
    Calculates only density and velocity components when magnetic field data
    are unavailable. Temperature and heat flux outputs are returned as
    3-element NaN lists so the output shape matches Moments().
    """

    densities = []
    vx_all, vy_all, vz_all = [], [], []

    for distribution_fn, volume, vx_samples, vy_samples, vz_samples in distributions:

        density = np.mean(distribution_fn) * volume * 1e-15
        
        N = len(distribution_fn)

        ux = (np.dot(distribution_fn, vx_samples) / N * volume) / (density * 1e15)
        uy = (np.dot(distribution_fn, vy_samples) / N * volume) / (density * 1e15)
        uz = (np.dot(distribution_fn, vz_samples) / N * volume) / (density * 1e15)
        densities.append(density)
        vx_all.append(ux)
        vy_all.append(uy)
        vz_all.append(uz)

    # keep same return structure as Moments()
    parallel_Temps = [np.nan, np.nan, np.nan]
    perp_Temps = [np.nan, np.nan, np.nan]
    parallel_heat_flux = [np.nan, np.nan, np.nan]
    perp_heat_flux = [np.nan, np.nan, np.nan]

    return (
        densities,
        vx_all,
        vy_all,
        vz_all,
        parallel_Temps,
        perp_Temps,
        parallel_heat_flux,
        perp_heat_flux,
    )