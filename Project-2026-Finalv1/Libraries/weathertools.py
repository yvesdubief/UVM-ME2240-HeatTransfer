import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import scipy.constants as csts
from scipy.integrate import solve_ivp
from Libraries import thermodynamics as thermo
from Libraries import HT_external_convection as extconv
from Libraries import HT_natural_convection as natconv
from Libraries import HT_thermal_resistance as res
class WeatherData(object):
    """
    How to
    """
    def __init__(self,weather):
        interp_method = 'linear' 
        weather['Solar Radiation'] = weather['Solar Radiation'].fillna(0.0)
        weather['Cloud Cover'] = weather['Cloud Cover'].fillna(0.0)
        weather['Precipitation'] = weather['Precipitation'].fillna(0.0)
        weather['Wind Speed'] = weather['Wind Speed'].interpolate(interp_method)
        weather['Temperature'] = weather['Temperature'].interpolate(interp_method)
        weather['Solar Radiation'] = weather['Solar Radiation'].interpolate(interp_method)
        weather['Cloud Cover'] = weather['Cloud Cover'].interpolate(interp_method)
        weather['Dew Point'] = weather['Dew Point'].interpolate(interp_method)
        weather['Precipitation'] = weather['Precipitation'].interpolate(interp_method)
        weather['Relative Humidity'] = weather['Relative Humidity'].interpolate(interp_method)

        self.spreadsheet = weather
        t_data = np.arange(0,weather.shape[0]*15*60,15*60)
        self.t = t_data
        U_atm = np.clip(np.abs(weather['Wind Speed'][:].to_numpy()/3.6),0.,None) #converted from km/h to m/s
        T_atm = weather['Temperature'][:].to_numpy()
        q_sun = weather['Solar Radiation'][:].to_numpy()
        cc = np.clip(weather['Cloud Cover'][:].to_numpy()/100.,0.,1.) # converted from % to fraction
        rh = np.clip(weather['Relative Humidity'][:].to_numpy(),0.,1.) # left as %
        p_r = weather['Precipitation'][:].to_numpy()*1e-3/(15*60) #converted to mm to m/s 
        T_dp = weather['Dew Point'][:].to_numpy()

        interp_method = 'linear'


        self.U_atmospheric = interp1d(t_data,U_atm,kind=interp_method)

        self.T_atmospheric = interp1d(t_data,T_atm,kind=interp_method)

        self.sun_irradiation = interp1d(t_data,q_sun,kind=interp_method)

        self.cloud_cover = interp1d(t_data,cc,kind=interp_method)

        self.dew_point = interp1d(t_data,T_dp,kind=interp_method)

        self.relative_humidity = interp1d(t_data,rh,kind=interp_method)

        self.rain_rate = interp1d(t_data,p_r,kind=interp_method)
    def qpp_outsideconvection(self,t,Ts,Lplate):
        Uinf = self.U_atmospheric(t)
        Tinf = self.T_atmospheric(t)
        T_f = 0.5*(Tinf + Ts)/2
        air_f = thermo.Fluid('air',T_f,'C')
        Re = np.abs(Uinf)*Lplate/air_f.nu
        Gr = natconv.Gr(beta=air_f.beta,DT=np.abs(Ts-Tinf),D=Lplate,nu=air_f.nu)
        Ra = natconv.Ra(beta=air_f.beta,DT=np.abs(Ts-Tinf),D=Lplate,nu=air_f.nu,alpha=air_f.alpha)
        if (Uinf < 0.15):
            ForcedConvection = False
            NaturalConvection = True
        else:
            Ri = Gr / Re**2
            if Ri < 0.1:
                ForcedConvection = True
                NaturalConvection = False
            elif Ri > 10:
                ForcedConvection = False
                NaturalConvection = True
            else:
                ForcedConvection = True
                NaturalConvection = True
        if ForcedConvection:
            if (Re <= 5e5):
                airflow = extconv.FlatPlate('laminar','isothermal',U_infty=Uinf,nu=air_f.nu,alpha=air_f.alpha, L=Lplate,xi=0,Re_xc= 5e5)
                airflow.average(Lplate)
                hconv_f = airflow.Nu_ave*air_f.k/Lplate
            elif Re > 5e5:
                airflow = extconv.FlatPlate('mixed','isothermal',U_infty=Uinf,nu=air_f.nu,alpha=air_f.alpha, L=Lplate,xi=0,Re_xc= 5e5)
                airflow.average(Lplate)
                hconv_f = airflow.Nu_ave*air_f.k/Lplate
        else:
            hconv_f = 0
        #Natural convection flux
        if NaturalConvection and Ra > 1e4:
            if Ts >= Tinf:
                airflow = natconv.FlatPlate(Ra,air_f.Pr,'upper','hot')
            else:
                airflow = natconv.FlatPlate(Ra,air_f.Pr,'upper','cold')
            hconv_n = airflow.Nu*air_f.k/Lplate
        else:
            hconv_n = 0
        #Total convection flux (here not a function of Ri)
        h = hconv_n + hconv_f
        qpp = h*(Ts - Tinf)

        return qpp

    def T_sky_hr(self,t,Ts):
        # Ts must be in Celsius
        Tdp = self.dew_point(t)
        cc = self.cloud_cover(t)
        Tinf = self.T_atmospheric(t)
        eps_sky = 1.
        eps_clear = 0.711 + 0.56*(Tdp/100.) + 0.73*(Tdp/100.)**2
        Ca = 1. + 0.02224*cc + 0.0035*cc**2 + 0.00028*cc**3
        Tsky  = (Ca*eps_clear)**0.25*thermo.C2K(Tinf)
        hr = eps_sky*csts.sigma*(Tsky+thermo.C2K(Ts))* \
            (Tsky**2+thermo.C2K(Ts)**2)
        return Tsky,hr
    def qpp_skyradiation(self,t,Ts):
        # Ts must be in Celsius
        Tdp = self.dew_point(t)
        cc = self.cloud_cover(t)
        Tinf = self.T_atmospheric(t)
        Tsky,hr = self.T_sky_hr(t,Ts)
        qsky = hr*(thermo.C2K(Ts) - Tsky)
        return qsky  
    def T_wet_bulb(self,T,RH):
        return T * np.arctan(0.1515977*(RH + 8.313659)**0.5) + np.arctan(T + RH) \
                - np.arctan(RH - 1.676331) + 0.00391838*RH**1.5*np.arctan(0.023101*RH) \
                - 4.686035
    def qpp_rain(self,t,Ts):
        pr = self.rain_rate(t)
        Tinf = self.T_atmospheric(t)
        RH = self.relative_humidity(t)
        Twb = self.T_wet_bulb(Tinf,RH)
    #     print("rain",pr,Twb,Tinf,RH,Ts)
        rho = 1000.
        Cp = 4.19e3
        return rho*Cp*pr*(Ts - Twb)

