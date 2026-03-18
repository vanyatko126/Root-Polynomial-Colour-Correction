import numpy as np
import pandas as pd
import colour
from sklearn.model_selection import train_test_split


def solve(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]

def delta_e2000(xyz1, xyz2, white_xyz):

    xyz1 = xyz1 / white_xyz[1]
    xyz2 = xyz2 / white_xyz[1]

    white_xy = colour.XYZ_to_xy(white_xyz)

    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=white_xy)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=white_xy)

    return colour.delta_E(lab1, lab2, method="CIE 2000")

def polynomial_exponents(degree):
    exps = []
    for a in range(degree + 1):
        for b in range(degree + 1):
            for c in range(degree + 1):
                if 1 <= a + b + c <= degree:
                    exps.append((a, b, c))
    return exps

def lcc_features(rgb):
    return rgb

def pcc_features(rgb, degree):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    feats = []
    for a, b_, c in polynomial_exponents(degree):
        feats.append((r ** a) * (g ** b_) * (b ** c))
    return np.stack(feats, axis=1)

def rpcc_features(rgb, degree):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    feats = []
    for a, b_, c in polynomial_exponents(degree):
        k = a + b_ + c
        feats.append((r ** a * g ** b_ * b ** c) ** (1 / k))
    return np.stack(feats, axis=1)

def build_features(rgb, method, degree):
    if method == "LCC":
        return lcc_features(rgb)
    if method == "PCC":
        return pcc_features(rgb, degree)
    return rpcc_features(rgb, degree)

illum = pd.read_csv("illum.csv")
munsell = pd.read_csv("munsell.csv")
xyz_fun = pd.read_csv("xyz_matching_fun.csv")
camera = pd.read_csv("canon600d.csv")

wl = np.arange(400, 721, 10)
d_lambda = 10

illum_i = np.interp(wl, illum["wavelength"], illum["D65"])
xbar = np.interp(wl, xyz_fun["wavelength"], xyz_fun["X"])
ybar = np.interp(wl, xyz_fun["wavelength"], xyz_fun["Y"])
zbar = np.interp(wl, xyz_fun["wavelength"], xyz_fun["Z"])

cam_r = np.interp(wl, camera["wavelength"], camera["red"])
cam_g = np.interp(wl, camera["wavelength"], camera["green"])
cam_b = np.interp(wl, camera["wavelength"], camera["blue"])

munsell = munsell[munsell["wavelength"] <= 720]
refl = munsell.drop(columns=["wavelength"]).values.T
n = refl.shape[0]

xyz = np.zeros((n, 3))
rgb = np.zeros((n, 3))

k = 100 / np.sum(illum_i * ybar * d_lambda)

for i in range(n):

    s = illum_i * refl[i]

    xyz[i,0] = k * np.sum(s * xbar) * d_lambda
    xyz[i,1] = k * np.sum(s * ybar) * d_lambda
    xyz[i,2] = k * np.sum(s * zbar) * d_lambda

    rgb[i,0] = np.sum(s * cam_r) * d_lambda
    rgb[i,1] = np.sum(s * cam_g) * d_lambda
    rgb[i,2] = np.sum(s * cam_b) * d_lambda

rgb_train, rgb_test_orig, xyz_train, xyz_test_orig = train_test_split(
    rgb,
    xyz,
    test_size=0.2,
    random_state=42,
)

white_rgb = np.array(
    [
        np.sum(illum_i * cam_r) * d_lambda,
        np.sum(illum_i * cam_g) * d_lambda,
        np.sum(illum_i * cam_b) * d_lambda,
    ]
)

rgb_half = rgb_test_orig * 0.5
xyz_half = xyz_test_orig * 0.5

black_level = 0.01 * white_rgb
mask_half = np.all(rgb_half >= black_level, axis=1)

removed_half = len(rgb_half) - np.sum(mask_half)

rgb_half = rgb_half[mask_half]
xyz_half = xyz_half[mask_half]

rgb_x2 = rgb_test_orig * 2
xyz_x2 = xyz_test_orig * 2

mask_x2 = np.all(rgb_x2 <= white_rgb, axis=1)

removed_x2 = len(rgb_x2) - np.sum(mask_x2)

rgb_x2 = rgb_x2[mask_x2]
xyz_x2 = xyz_x2[mask_x2]

datasets = {
    "original": (rgb_test_orig, xyz_test_orig),
    "x0.5": (rgb_half, xyz_half),
    "x2": (rgb_x2, xyz_x2),
}

methods = [("LCC", 1)]

for d in range(2, 6):
    methods.append(("PCC", d))

for d in range(2, 6):
    methods.append(("RPCC", d))

results = []

for dataset_name in ["original", "x0.5", "x2"]:
    rgb_test, xyz_test = datasets[dataset_name]

    for method, degree in methods:
        a_train = build_features(rgb_train, method, degree)
        m = solve(a_train, xyz_train)

        a_test = build_features(rgb_test, method, degree)
        xyz_pred = a_test @ m

        d_e = delta_e2000(xyz_test, xyz_pred, white_rgb)

        results.append(
            [
                dataset_name,
                method,
                degree,
                np.mean(d_e),
                np.median(d_e),
                np.percentile(d_e, 95),
            ]
        )

df = pd.DataFrame(
    results,
    columns=[
        "dataset",
        "method",
        "degree",
        "mean_dE",
        "median_dE",
        "p95_dE",
    ],
)

df.to_csv("metrics.csv", index=False)
