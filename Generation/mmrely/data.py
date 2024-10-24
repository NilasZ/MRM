import os
import re
from scipy.interpolate import CubicSpline
from numpy.fft import fft,ifft,fftshift,ifftshift
import numpy as np
import pandas as pd

def load_target_datasets(base_path):
    # return dic: keys target_1 target_2 target_3 target4 and there narrowband wideband ffe dic.
    target_data = {}

    # 列出目录中的所有文件
    for file in os.listdir(base_path):
        if file.endswith(".ffe"):
            # 从文件名提取目标编号和带类型
            target_num = file.split('_')[0]
            target_key = f"target_{int(target_num) // 10}"

            # 将文件路径分配给相应的目标和带类型
            if target_key not in target_data:
                target_data[target_key] = {'narrow': None, 'wide': None}
            
            if target_num.endswith('0'):  # 表示窄带
                target_data[target_key]['narrow'] = os.path.join(base_path, file)
            else:  # 表示宽带
                target_data[target_key]['wide'] = os.path.join(base_path, file)

    return target_data

class RCS:
    def __init__(self, target_idx, los_angles):
        self.target_idx = target_idx
        self.los_angles = los_angles
        self.static_rcs ,self.theta = self.extract_rcs_data()

    def read_csv(self, file_path):
        return pd.read_csv(file_path)

    def extract_rcs_data(self):
        base_filename = f"{self.target_idx + 1}0_FarField1"
        h_csv = f"{base_filename}_H.csv"
        v_csv = f"{base_filename}_V.csv"

        h_data = self.read_csv(f'./EM_data/{self.target_idx}/' + h_csv)
        v_data = self.read_csv(f'./EM_data/{self.target_idx}/' + v_csv)

        rcs_data = {
            "total": 20 * np.log10(h_data["RCS(Total)"].values),
            "HH": 20 * np.log10(h_data["RCS(Phi)"].values),
            "HV": 20 * np.log10(h_data["RCS(Theta)"].values),
            "VV": 20 * np.log10(v_data["RCS(Theta)"].values),
            "VH": 20 * np.log10(v_data["RCS(Phi)"].values),
        }

        theta_values = h_data["Theta"].values
        return rcs_data, theta_values

    def compute_rcs(self, rcs_data):
        theta_range = np.arange(0, 180.1, 0.1)
        index = [np.abs(theta_range - gamma).argmin() for gamma in self.los_angles]

        rcs_total_db = rcs_data["total"][index]
        rcs_HH_db = rcs_data["HH"][index]
        rcs_HV_db = rcs_data["HV"][index]
        rcs_VV_db = rcs_data["VV"][index]
        rcs_VH_db = rcs_data["VH"][index]

        return {
            "total": rcs_total_db,
            "HH": rcs_HH_db,
            "HV": rcs_HV_db,
            "VV": rcs_VV_db,
            "VH": rcs_VH_db
        }

    def get_rcs(self):
        rcs_total_db = self.compute_rcs(self.static_rcs)
        return rcs_total_db
    
class Echo:
    def __init__(self, target_idx, los_angles):
        self.target_idx = target_idx
        self.los_angles = los_angles
        self.electric_field_data, self.theta = self.extract_electric_field_data()

    def read_csv(self, file_path):
        return pd.read_csv(file_path)

    def extract_electric_field_data(self):
        base_filename = f"{self.target_idx + 1}0_FarField1"
        h_csv = f"{base_filename}_H.csv"
        v_csv = f"{base_filename}_V.csv"

        h_data = self.read_csv(f'./EM_data/{self.target_idx}/' + h_csv)
        v_data = self.read_csv(f'./EM_data/{self.target_idx}/' + v_csv)

        Etheta_H = h_data["Re(Etheta)"].values + 1j * h_data["Im(Etheta)"].values
        Ephi_H = h_data["Re(Ephi)"].values + 1j * h_data["Im(Ephi)"].values
        E_H = Etheta_H + Ephi_H

        Etheta_V = v_data["Re(Etheta)"].values + 1j * v_data["Im(Etheta)"].values
        Ephi_V = v_data["Re(Ephi)"].values + 1j * v_data["Im(Ephi)"].values
        # E_V = Etheta_V + Ephi_V

        electric_field_data = {
            "total": E_H,
            "HH": Ephi_H,
            "HV": Etheta_H,
            "VV": Etheta_V,
            "VH": Ephi_V,
        }

        theta_values = h_data["Theta"].values
        return electric_field_data, theta_values

    def compute_echo(self, electric_field_data):
        theta_range = np.linspace(0, 180, 1801)

        cs_real = {}
        cs_imag = {}

        for key, field in electric_field_data.items():
            cs_real[key] = CubicSpline(theta_range, np.real(field.T))
            cs_imag[key] = CubicSpline(theta_range, np.imag(field.T))

        echoes = {}
        for key in electric_field_data.keys():
            ys_real = cs_real[key](self.los_angles)
            ys_imag = cs_imag[key](self.los_angles)
            echoes[key] = ys_real + 1j * ys_imag

        return echoes

    def get_echo(self):
        echoes = self.compute_echo(self.electric_field_data)
        return echoes
    

class HRRP:
    def __init__(self, target_idx, los_angles):
        self.target_idx = target_idx
        self.los_angles = los_angles
        self.electric_field_data, self.theta, self.frequency = self.extract_electric_field_data()
        self.static_hrrp = self.extract_static()

    def read_csv(self, file_path):
        return pd.read_csv(file_path)

    def extract_electric_field_data(self):
        base_filename = f"{self.target_idx + 1}1_FarField1"
        h_csv = f"{base_filename}_H.csv"
        v_csv = f"{base_filename}_V.csv"

        h_data = self.read_csv(f'./EM_data/{self.target_idx}/' + h_csv)
        v_data = self.read_csv(f'./EM_data/{self.target_idx}/' + v_csv)

        theta = h_data["Theta"]
        frequency = h_data["Frequency"]

        Etheta_H = h_data["Re(Etheta)"].values + 1j * h_data["Im(Etheta)"].values
        Ephi_H = h_data["Re(Ephi)"].values + 1j * h_data["Im(Ephi)"].values
        E_H = Etheta_H + Ephi_H   
        
        Etheta_V = v_data["Re(Etheta)"].values + 1j * v_data["Im(Etheta)"].values
        Ephi_V = v_data["Re(Ephi)"].values + 1j * v_data["Im(Ephi)"].values

        electric_field_data = {
            "total": E_H,
            "HH": Ephi_H,
            "HV": Etheta_H,
            "VV": Etheta_V,
            "VH": Ephi_V,
        }       
        return electric_field_data, theta, frequency
    
    def extract_static(self):
        theta_unique = self.theta.unique()
        freq_unique = self.frequency.unique()

        static_hrrp = {}
        for key, field in self.electric_field_data.items():
            static_field = np.zeros((len(freq_unique), len(theta_unique)), dtype=complex)
            for i, theta_val in enumerate(theta_unique):
                idx = np.where(self.theta == theta_val)
                static_field[:, i] = field[idx]
            static_hrrp[key] = static_field.squeeze()
        return static_hrrp

    def extract_hrrp(self):
        theta_range = np.arange(0, 180.1, 0.1)
        pol_hrrp = {}

        for key, field in self.static_hrrp.items():
            index = [np.abs(theta_range - gamma).argmin() for gamma in self.los_angles]
            hrrp = fftshift(ifft(field[:, index], axis=0), axes=0)
            pol_hrrp[key] = hrrp
        
        return pol_hrrp

