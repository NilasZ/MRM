import numpy as np
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from mpl_toolkits.mplot3d import Axes3D

def orbit_info():
    print('include all orbit stuff.')

def plot_earth(bad, R = 6371e3, simplify = True):
    # ------------------------------------ info -----------------------------------------------
    # Need to create a figure bad like: 
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # Usage:
    # plot_earth(bad = ax, R = 6371e3)
    # -----------------------------------------------------------------------------------------
    # create sphere data.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v)) 
    # plot white 
    if simplify:
        ax.plot_surface(x, y, z, color='white', edgecolor='black', linewidth=0.1, alpha=0.3)
    else:
        bm = Image.open('./earth_texture.jpg')
        bm = np.array(bm.resize([500,250]))/256
        lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
        lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

        x = R*np.outer(np.cos(lons), np.cos(lats)).T
        y = R*np.outer(np.sin(lons), np.cos(lats)).T
        z = R*np.outer(np.ones(np.size(lons)), np.sin(lats)).T

        





def orbital_elements(r0, v0, printall):
    # ------------------------------------ info -----------------------------------------------
    # r0: initial position (meters).
    # v0: initial velocity (meters/sec).
    # printall(bool): if print orbit elements.
    # return: orbit basic elements.
    # ---------------------------------- Parameters -------------------------------------------
    G = 6.67430e-11  # gravity constants
    M = 5.972e24     # earth mass
    mu = G * M       # earth standrad gravity constants
    # -----------------------------------------------------------------------------------------

    # 计算轨道能量和角动量
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)
    energy = 0.5 * v0_norm**2 - G * M / r0_norm

    # 计算轨道根数
    a = -G * M / (2 * energy)  # 半长轴
    e_vec = (np.cross(v0, h_vec) / G / M) - (r0 / r0_norm)  # 离心率矢量
    e = np.linalg.norm(e_vec)
    i = np.arccos(h_vec[2] / h)  # 倾角
    Omega = np.arctan2(h_vec[0], -h_vec[1])  # 升交点赤经
    omega = np.arctan2(e_vec[2] / np.sin(i), e_vec[0] * np.cos(Omega) + e_vec[1] * np.sin(Omega))  # 近地点幅角
    nu = np.arccos(np.dot(e_vec, r0) / (e * r0_norm))
    if np.dot(r0, v0) < 0:
        nu = 2 * np.pi - nu
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    
    # 计算平近点角
    M0 = E - e * np.sin(E)
    if printall:
        print(f"半长轴 (a): {a} m")
        print(f"离心率 (e): {e}")
        print(f"轨道倾角 (i): {np.rad2deg(i)} degrees")
        print(f"升交点赤经 (Ω): {np.rad2deg(Omega)} degrees")
        print(f"近地点幅角 (ω): {np.rad2deg(omega)} degrees")
        print(f"平近点角 (M0): {np.rad2deg(M0)} degrees")

    return a, e, i, Omega, omega, M0

def solve_kepler(M, e, tol=1e-10):
    E = M
    while True:
        E_new = E + (M + e * np.sin(E) - E) / (1 - e * np.cos(E))
        if np.abs(E_new - E) < tol:
            break
        E = E_new
    return E

def view_orbit(ax, r0, v0, radar, elev, azim ,title, orbit_points=None):
    
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = 6371e3 * np.cos(u) * np.sin(v)
    y = 6371e3 * np.sin(u) * np.sin(v)
    z = 6371e3 * np.cos(v)
    # ax.plot_surface(x, y, z, rstride=5, cstride=5, color='blue', alpha=0.5, linewidth=0.3, edgecolors='w')
    # ax.plot_wireframe(x, y, z, color='k', linewidth=0.1)
    ax.plot_surface(x, y, z, color='b', alpha=0.3, linewidth=0.5)

    # set label and title
    ax.set_xlabel('X (m)')
    ax.set_xticks([])
    ax.set_ylabel('Y (m)')
    ax.set_yticks([]) # hide y ticks
    ax.set_zlabel('Z (m)')
    ax.set_zticks([])
    ax.set_title(title)

    # plot scatter
    if radar is None:
        pass
    else:
        ax.scatter(r0[0], r0[1], r0[2], color='black', s = 5)
        ax.scatter(radar[0], radar[1], radar[2], color='red', s = 5)
        
    ax.quiver(r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], color='k', length=200, arrow_length_ratio=0.3, linewidth=1)

    # view
    ax.view_init(elev = elev, azim = azim) # view from y axis.

    # orbit
    if orbit_points is None:
        pass
    else:    
        ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], label='Orbit-6e', color='r', alpha=1)
    
    ax.set_box_aspect([1,1,1])


def calculate_orbit_points(a, e, i, Omega, omega, M0, num_points):
    # num_points: How many points will simple
    # ------------------------------------- Parameters -----------------------------------------------
    G = 6.67430e-11  # gravity constants
    M = 5.972e24     # earth mass
    mu = G * M       # earth standrad gravity constants
    # -----------------------------------------------------------------------------------------
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # 轨道周期
    dt = T / num_points
    
    r_points = []
    M_points = []
    for step in range(num_points):
        M = M0 + np.sqrt(mu / a**3) * step * dt
        E = solve_kepler(M, e)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e * np.cos(E))
        r_orbital = r * np.array([np.cos(nu), np.sin(nu), 0])

        # 轨道平面到惯性空间的旋转矩阵
        cos_Omega, sin_Omega = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_omega, sin_omega = np.cos(omega), np.sin(omega)

        rotation_matrix = np.array([
            [cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i, -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i, sin_Omega * sin_i],
            [sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i, -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i, -cos_Omega * sin_i],
            [sin_omega * sin_i, cos_omega * sin_i, cos_i]
        ])

        r_inertial = np.dot(rotation_matrix, r_orbital)
        r_points.append(r_inertial)
        M_points.append(M)

        if np.linalg.norm(r_inertial)<6371e3:
            break

    return np.array(r_points),M_points


def get_model(pdf_path):
    # path: it should be a pdf file export by FreeCAD
    pdf_document = fitz.open(pdf_path)
    # Extract the first page
    page = pdf_document.load_page(0)
    # Extract the image list
    image_list = page.get_images(full=True)
    
    if not image_list:
        print("No images found in the PDF file.")
        return None
    
    # Extract the first image (assuming there's only one image in the PDF)
    xref = image_list[0][0]
    base_image = pdf_document.extract_image(xref)
    image_bytes = base_image["image"]
    
    # Create a PIL image
    image = Image.open(io.BytesIO(image_bytes))

    image_array = np.array(image)

    return image_array
