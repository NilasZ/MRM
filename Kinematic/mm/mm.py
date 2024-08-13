import numpy as np
import re
from scipy.interpolate import CubicSpline
from numpy.fft import fft,ifft,fftshift,ifftshift
from scipy.signal import stft, windows

def info():
    print('This module include micro motion related functions and classes.')

def micro_motion_transform(ang_x_int, ang_y_int, ang_z_int, alpha, belta, freq, amp_n, t):
    
    # Initial rotation matrix R0
    Rint_x = np.array([[1, 0, 0],
                       [0, np.cos(ang_x_int), -np.sin(ang_x_int)],
                       [0, np.sin(ang_x_int), np.cos(ang_x_int)]])
    
    Rint_y = np.array([[np.cos(ang_y_int), 0, np.sin(ang_y_int)],
                       [0, 1, 0],
                       [-np.sin(ang_y_int), 0, np.cos(ang_y_int)]])
    
    Rint_z = np.array([[np.cos(ang_z_int), -np.sin(ang_z_int), 0],
                       [np.sin(ang_z_int), np.cos(ang_z_int), 0],
                       [0, 0, 1]])
    
    Rint = np.dot(Rint_z, np.dot(Rint_y, Rint_x))
    
    # Spin transformation matrix
    symmetric_spin_axis_int = np.array([0, 0, 1])
    symmetric_spin_axis = np.dot(Rint, symmetric_spin_axis_int)
    Es = np.array([[0, -symmetric_spin_axis[2], symmetric_spin_axis[1]],
                   [symmetric_spin_axis[2], 0, -symmetric_spin_axis[0]],
                   [-symmetric_spin_axis[1], symmetric_spin_axis[0], 0]])
    Rs = np.eye(3) + np.dot(Es, np.sin(freq[0]*t)) + np.dot(np.dot(Es, Es), (1 - np.cos(freq[0]*t)))
    
    # Precession transformation matrix
    Ec = np.array([[0, -np.sin(belta), np.sin(alpha)*np.cos(belta)],
                   [np.sin(belta), 0, -np.cos(alpha)*np.cos(belta)],
                   [-np.sin(alpha)*np.cos(belta), np.cos(alpha)*np.cos(belta), 0]])
    Rc = np.eye(3) + np.dot(Ec, np.sin(freq[1]*t)) + np.dot(np.dot(Ec, Ec), (1 - np.cos(freq[1]*t)))
    
    # Nutation transformation matrix
    symmetric_cone_axis = np.array([np.cos(belta)*np.cos(alpha), np.cos(belta)*np.sin(alpha), np.sin(belta)])
    xn = symmetric_cone_axis / np.linalg.norm(symmetric_cone_axis)
    zt = np.dot(Rc, np.dot(Rint, symmetric_spin_axis_int))
    zn = np.cross(xn, zt) / np.linalg.norm(np.cross(xn, zt))
    yn = np.cross(xn, zn) / np.linalg.norm(np.cross(xn, zn))
    An = np.column_stack((xn, yn, zn))
    delt_belta = amp_n*np.sin(freq[2]*t)
    Bn = np.array([[np.cos(delt_belta), -np.sin(delt_belta), 0],
                   [np.sin(delt_belta), np.cos(delt_belta), 0],
                   [0, 0, 1]])
    Rn = np.dot(An, np.dot(Bn, np.linalg.inv(An)))

    micro_motion_T = np.dot(Rn, np.dot(Rc, np.dot(Rs, Rint)))
    
    return micro_motion_T

def view_warhead(ax, ang_x_int, ang_y_int, ang_z_int, alpha, belta, freq, amp_n, t, elev, azim):
    # ---------------------------------------------------------- view angle test here --------------------------------------------------------------
    # Define the grid
    ro = np.linspace(0.001, 0.3, 50)
    phi = np.linspace(-np.pi, np.pi, 50)

    # Create the meshgrid for (ro, phi)
    Ro, Phi = np.meshgrid(ro, phi)
    X = Ro * np.cos(Phi)
    Y = Ro * np.sin(Phi)
    Z = -3 * np.sqrt(X**2 + Y**2) + 0.6

    # Flatten the matrices
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()

    micro_motion_T = micro_motion_transform(ang_x_int, ang_y_int, ang_z_int, alpha, belta, freq, amp_n, t)

    cor = micro_motion_T @ np.vstack((x, y, z))

    # Reshaping the results back to their original shape
    xx = cor[0, :].reshape(X.shape)
    yy = cor[1, :].reshape(Y.shape)
    zz = cor[2, :].reshape(Z.shape)

    # Plotting the transformed mesh
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)

    # Plot limits and view angle
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 1])
    ax.set_zlim([-0.38, 1.38])
    ax.set_xticklabels([])
    ax.set_xlabel('X (m)')
    ax.set_xticks([])
    ax.set_ylabel('Y (m)')
    ax.set_yticks([])  # Hide y ticks
    ax.set_zlabel('Z (m)')
    ax.set_zticks([])

    # Plotting a line representing the cone axis in reference coordinate

    cone_axis = np.array([np.cos(alpha) * np.cos(belta), np.sin(alpha) * np.cos(belta), np.sin(belta)])
    ax.plot([0, cone_axis[0]], [0, cone_axis[1]], [0, cone_axis[2]], linestyle='--', color='r')

    # Plotting a line representing the central axis after rotate
    central_axis = np.array([0, 0, 0.6])
    ref_central_axis = micro_motion_T @ central_axis
    ax.plot([0, ref_central_axis[0]], [0, ref_central_axis[1]], [0, ref_central_axis[2]], linestyle='--', color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(elev, azim)

def angle_between_vectors(a, b):

    dot_product = np.dot(a, b)
    
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    cos_theta = dot_product / (a_norm * b_norm)
    

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)

    angle_deg = np.degrees(angle_rad)

    return np.pi-angle_rad, 180-angle_deg

def solve_kepler(M, e, tol=1e-10):
    E = M
    while True:
        E_new = E + (M + e * np.sin(E) - E) / (1 - e * np.cos(E))
        if np.abs(E_new - E) < tol:
            break
        E = E_new
    return E

def get_los(ang_x_int, ang_y_int, ang_z_int, alpha, belta, freq, amp_n , T, point_num, start_position, a, e, Omega, i, omega, radar_position, random_phase = False):
    t_range = np.linspace(0,T,point_num)
    G = 6.67430e-11  # gravity constants
    M = 5.972e24     # earth mass
    mu = G * M       # earth standrad gravity constants
    central_axis = np.array([0,0,0.6])
    r_interval = []
    c_interval = []

    if random_phase:
        initial_time = np.abs(np.random.randn(1)*5)
    

    for t in t_range:
        M = start_position + np.sqrt(mu / a**3) * t
        if random_phase:
            t = initial_time[0] + t
            micro_motion_T = micro_motion_transform(ang_x_int= ang_x_int, ang_y_int= ang_y_int, ang_z_int= ang_z_int, alpha= alpha, belta= belta, freq = freq, amp_n = amp_n , t= t)
        else:
            micro_motion_T = micro_motion_transform(ang_x_int= ang_x_int, ang_y_int= ang_y_int, ang_z_int= ang_z_int, alpha= alpha, belta= belta, freq = freq, amp_n = amp_n , t= t)

        ref_central_axis = micro_motion_T @ central_axis

        E = solve_kepler(M, e)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e * np.cos(E))
        r_orbital = r * np.array([np.cos(nu), np.sin(nu), 0])

        cos_Omega, sin_Omega = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_omega, sin_omega = np.cos(omega), np.sin(omega)

        rotation_matrix = np.array([
            [cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i, -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i, sin_Omega * sin_i],
            [sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i, -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i, -cos_Omega * sin_i],
            [sin_omega * sin_i, cos_omega * sin_i, cos_i]
        ])

        r_inertial = np.dot(rotation_matrix, r_orbital)
        r_interval.append(r_inertial)
        c_interval.append(ref_central_axis)

    los_rad = []
    los_deg = []
    for i in range(len(r_interval)):
        vector_a = radar_position - r_interval[i]
        angle_rad, angle_deg = angle_between_vectors(vector_a, c_interval[i] )
        los_rad.append(angle_rad)
        los_deg.append(angle_deg)
    
    return np.array(los_rad), np.array(los_deg)

def awgn(signal, snr, seed=0):

    np.random.seed(seed)
    
    # Calculate signal power and convert SNR to linear scale
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr / 10)
    
    # Calculate noise power and generate complex noise
    noise_power = signal_power / snr_linear
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise = noise_real + 1j * noise_imag
    
    # Add noise to the signal
    signal_with_noise = signal + noise
    
    return signal_with_noise



