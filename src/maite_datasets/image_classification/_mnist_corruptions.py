# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Pulled from https://github.com/google-research/mnist-c
# Modified: - typing
#           - removed corruptions that required external file
#           - rewrote corruptions that required wand dependency
#           - specific functions to handle batches of images
#
# Notes: OpenCV is required for some corruptions. To install, use:
#    pip install opencv-python-headless
# or use the extra when install maite-datasets:
#    pip install maite-datasets[opencv]

import warnings
from io import BytesIO
from typing import TYPE_CHECKING, Callable

import numpy as np
import skimage as sk
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage import feature, transform
from skimage.filters import gaussian

warnings.simplefilter("ignore", UserWarning)

if TYPE_CHECKING:
    from cv2.typing import MatLike
else:
    MatLike = ArrayLike

# /////////////// Corruption Helpers ///////////////


def disk(radius: int, alias_blur: float = 0.1) -> MatLike:
    from cv2 import GaussianBlur

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array(radius**2 >= (X**2 + Y**2), dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# rewrite of wand library motion_blur using opencv and numpy
def motion_kernel(radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0) -> MatLike:
    from cv2 import getGaussianKernel, getRotationMatrix2D, warpAffine

    ksize = int(2 * radius + 1)
    if ksize % 2 == 0:
        ksize += 1

    kernel_1d = getGaussianKernel(ksize, sigma)
    kernel_2d = np.zeros((ksize, ksize))

    pad = ksize // 2
    kernel_2d[:, pad] = kernel_1d[:, 0]

    center = (pad, pad)
    rotation_matrix = getRotationMatrix2D(center, angle, 1.0)
    kernel_rotated = warpAffine(kernel_2d, rotation_matrix, (ksize, ksize))
    kernel_rotated /= kernel_rotated.sum()

    return kernel_rotated


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize: int = 256, wibbledecay: int = 3) -> NDArray[np.float64]:
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    if not (mapsize & (mapsize - 1) == 0):  # Power of two check
        raise ValueError("mapsize must be a power of 2.")
    maparray: NDArray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array: NDArray) -> NDArray:
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares() -> None:
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(squareaccum)

    def filldiamonds() -> None:
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


# modified to handle a batch of images
def clipped_zoom(x: NDArray[np.float32], zoom_factor: float) -> NDArray[np.float32]:
    h = x.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    x = np.array(scizoom(x[:, top : top + ch, top : top + ch], (1, zoom_factor, zoom_factor), order=1))
    # trim off any extra pixels
    trim_top = (x.shape[1] - h) // 2

    return x[:, trim_top : trim_top + h, trim_top : trim_top + h]


def line_from_points(c0: int, r0: int, c1: int, r1: int) -> NDArray[np.float32]:
    if c1 == c0:
        return np.zeros((28, 28), dtype=np.float32)

    # Decay function defined as log(1 - d/2) + 1
    cc, rr = np.meshgrid(np.linspace(0, 27, 28), np.linspace(0, 27, 28), sparse=True)

    m = (r1 - r0) / (c1 - c0)

    def f(c: NDArray[np.float64]) -> NDArray[np.float64]:
        return m * (c - c0) + r0

    dist = np.clip(np.abs(rr - f(cc)), 0, 2.3 - 1e-10)
    corruption = np.log(1 - dist / 2.3) + 1
    corruption = np.clip(corruption, 0, 1)

    left = int(np.floor(c0))
    right = int(np.ceil(c1))

    corruption[:, :left] = 0
    corruption[:, right:] = 0

    return np.clip(corruption, 0, 1)


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////


def identity(x: NDArray[np.number]) -> NDArray[np.float32]:
    return x.astype(np.float32)


def gaussian_noise(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

    x = x / 255.0
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x.astype(np.float32)


def shot_noise(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [60, 25, 12, 5, 3][severity - 1]

    x = x.astype(np.float64) / 255.0
    x = np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255
    return x.astype(np.float32)


def impulse_noise(x: NDArray[np.number], severity: int = 4) -> NDArray[np.float32]:
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(x / 255.0, mode="s&p", amount=c)
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


def speckle_noise(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]

    x = x / 255.0
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def gaussian_blur(x: NDArray[np.number], severity: int = 2) -> NDArray[np.float32]:
    c = [1, 2, 3, 4, 6][severity - 1]

    x = np.array([gaussian(img / 255.0, sigma=c) for img in x])
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def glass_blur(x: NDArray[np.number], severity: int = 1) -> NDArray[np.float32]:
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.array([gaussian(img / 255.0, sigma=c[0]) for img in x]) * 255
    x = x.astype(np.uint8)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(28 - c[1], c[1], -1):
            for w in range(28 - c[1], c[1], -1):
                if np.random.choice([True, False], 1)[0]:
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[:, h, w], x[:, h_prime, w_prime] = x[:, h_prime, w_prime], x[:, h, w]

    x = np.clip(np.array([gaussian(img / 255.0, sigma=c[0]) for img in x]), 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def defocus_blur(x: NDArray[np.number], severity: int = 1) -> NDArray[np.float32]:
    from cv2 import filter2D

    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = x.astype(np.float32) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    out = np.array([filter2D(x[i], -1, kernel) for i in range(x.shape[0])])

    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


# modified to remove wand dependency
def motion_blur(x: NDArray[np.number], severity: int = 1) -> NDArray[np.float32]:
    from cv2 import filter2D

    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    x = x.astype(np.float32) / 255.0
    kernel = motion_kernel(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    out = np.array([filter2D(x[i], -1, kernel) for i in range(x.shape[0])])

    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


def zoom_blur(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = x.astype(np.float32) / 255.0
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, float(zoom_factor))

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1, dtype=np.float32) * 255


def fog(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]

    x = x / 255.0
    max_val = x.max()
    x = x + c[0] * plasma_fractal(wibbledecay=c[1])[:28, :28]
    x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    return x.astype(np.float32)


# modified to remove wand dependency
def snow(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    from cv2 import filter2D

    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = x.astype(np.float32) / 255.0
    snow_layer = np.random.normal(size=(x.shape[-2], x.shape[-1]), loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[np.newaxis, :, :].astype(np.float32), c[2])
    snow_layer[snow_layer < c[3]] = 0

    kernel = motion_kernel(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    out = filter2D(snow_layer[0], -1, kernel)

    x = c[6] * x + (1 - c[6]) * np.maximum(x, x * 1.5 + 0.5)
    x = np.clip(x + out + np.rot90(out, k=2), 0, 1) * 255
    return x.astype(np.float32)


def spatter(x: NDArray[np.number], severity: int = 4) -> NDArray[np.float32]:
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]

    x = x.astype(np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255.0 * np.ones_like(x) * m
    x *= 1 - m
    x = np.clip(x + color, 0, 1) * 255
    return x.astype(np.float32)


def contrast(x: NDArray[np.number], severity: int = 4) -> NDArray[np.float32]:
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

    x = x / 255.0
    means = np.mean(x, axis=(1, 2), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def brightness(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

    x = x / 255.0
    x = np.clip(x + c, 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def saturate(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = x / 255.0
    x = np.clip(x * c[0] + c[1], 0, 1) * 255
    return x.astype(np.float32)


# modified to handle a batch of images
def jpeg_compression(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    c = [25, 18, 15, 10, 7][severity - 1]

    def compress_one(img_arr: NDArray[np.uint8], quality: int) -> NDArray[np.number]:
        output = BytesIO()
        img = Image.fromarray(img_arr, mode="L")
        img.save(output, "JPEG", quality=quality)
        jpeg = Image.open(output)
        return np.array(jpeg)

    out = [compress_one(x[i].astype(np.uint8), c) for i in range(x.shape[0])]
    out_arr = np.array(out)

    return out_arr.astype(np.float32)


# modified to handle a batch of images
def pixelate(x: NDArray[np.number], severity: int = 3) -> NDArray[np.float32]:
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    def pixelate_one(img_arr: NDArray[np.uint8], size: float) -> NDArray[np.number]:
        img = Image.fromarray(img_arr, mode="L")
        img = img.resize((int(28 * size), int(28 * size)), Image.Resampling.BOX)
        img = img.resize((28, 28), Image.Resampling.BOX)
        return np.array(img)

    out = [pixelate_one(x[i].astype(np.uint8), c) for i in range(x.shape[0])]
    out_arr = np.array(out)

    return out_arr.astype(np.float32)


# modified to handle a batch of images
# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(x_arr: NDArray[np.number], severity: int = 1) -> NDArray[np.float32]:
    from cv2 import BORDER_CONSTANT, getAffineTransform, warpAffine

    c = [
        (28 * 2, 28 * 0.7, 28 * 0.1),
        (28 * 2, 28 * 0.08, 28 * 0.2),
        (28 * 0.05, 28 * 0.01, 28 * 0.02),
        (28 * 0.07, 28 * 0.01, 28 * 0.02),
        (28 * 0.12, 28 * 0.01, 28 * 0.02),
    ][severity - 1]

    x_arr = x_arr.astype(np.float32) / 255.0
    shape = np.array([x_arr.shape[-2], x_arr.shape[-1]])

    # random affine
    center_square = shape // 2
    square_size = min(shape) // 3
    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = getAffineTransform(pts1, pts2)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    def elastic_one(img: NDArray, M: NDArray, shape: NDArray, indices: tuple[NDArray, NDArray]) -> NDArray[np.number]:
        img = warpAffine(img, M, [int(shape[0]), int(shape[1])], borderMode=BORDER_CONSTANT)
        return np.clip(map_coordinates(img, indices, order=1, mode="constant").reshape(shape), 0, 1)  # type: ignore

    out = np.array([elastic_one(x_arr[i], M, shape, indices) for i in range(x_arr.shape[0])]) * 255
    return out.astype(np.float32)


def quantize(x: NDArray[np.number], severity: int = 5) -> NDArray[np.float32]:
    bits = [5, 4, 3, 2, 1][severity - 1]

    x = x.astype(np.float32)
    x *= (2**bits - 1) / 255.0
    x = x.round()
    x *= 255.0 / (2**bits - 1)

    return x.astype(np.float32)


# modified to handle a batch of images
def shear(x: NDArray[np.number], severity: int = 2) -> NDArray[np.float32]:
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]

    # Randomly switch directions
    bit = np.random.choice([-1, 1], 1)[0]
    c *= bit
    aff = transform.AffineTransform(shear=c)

    # Calculate translation in order to keep image center (13.5, 13.5) fixed
    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]
    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(shear=c, translation=[a3, b3])

    x = x / 255.0
    out = np.array([transform.warp(img, inverse_map=aff) for img in x])
    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


# modified to handle a batch of images
def rotate(x: NDArray[np.number], severity: int = 2) -> NDArray[np.float32]:
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]

    # Randomly switch directions
    bit = np.random.choice([-1, 1], 1)[0]
    c *= bit
    aff = transform.AffineTransform(rotation=c)

    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]
    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(rotation=c, translation=[a3, b3])

    x = x / 255.0
    out = np.array([transform.warp(img, inverse_map=aff) for img in x])
    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


# modified to handle a batch of images
def scale(x: NDArray[np.number], severity: int = 3) -> NDArray[np.float32]:
    c = [(1 / 0.9, 1 / 0.9), (1 / 0.8, 1 / 0.8), (1 / 0.7, 1 / 0.7), (1 / 0.6, 1 / 0.6), (1 / 0.5, 1 / 0.5)][
        severity - 1
    ]

    aff = transform.AffineTransform(scale=c)

    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]
    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)
    aff = transform.AffineTransform(scale=c, translation=[a3, b3])

    x = x / 255.0
    out = np.array([transform.warp(img, inverse_map=aff) for img in x])
    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


# modified to handle a batch of images
def translate(x: NDArray[np.number], severity: int = 3) -> NDArray[np.float32]:
    c = [1, 2, 3, 4, 5][severity - 1]
    bit = np.random.choice([-1, 1], 2)
    dx = c * bit[0]
    dy = c * bit[1]
    aff = transform.AffineTransform(translation=[dx, dy])

    x = x / 255.0
    out = np.array([transform.warp(img, inverse_map=aff) for img in x])
    out = np.clip(out, 0, 1) * 255
    return out.astype(np.float32)


def line(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x / 255.0
    c0 = np.random.randint(low=0, high=5)
    c1 = np.random.randint(low=22, high=27)
    r0, r1 = np.random.randint(low=0, high=27, size=2)
    corruption = line_from_points(c0, int(r0), c1, int(r1))

    x = np.clip(x + corruption, 0, 1) * 255
    return x.astype(np.float32)


def dotted_line(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x / 255.0
    r0, r1 = np.random.randint(low=0, high=27, size=2)
    corruption = line_from_points(0, int(r0), 27, int(r1))

    idx = np.arange(0, 30, 2)
    off = True
    for i in range(1, len(idx)):
        if off:
            corruption[:, idx[i - 1] : idx[i]] = 0
        off = not off

    x = np.clip(x + corruption, 0, 1) * 255
    return x.astype(np.float32)


def zigzag(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x / 255.0
    # a, b are length and width of zigzags
    a = 2.0
    b = 2.0

    c0, c1 = 2, 25
    r0 = np.random.randint(low=0, high=27)
    r1 = r0 + np.random.randint(low=-5, high=5)

    theta = np.arctan((r1 - r0) / (c1 - c0))

    # Calculate length of straight line
    d = (c1 - c0) / np.cos(theta)
    endpoints = [(0.0, 0.0)]

    r_i = 0
    for i in range(int((d - a) // (2 * a)) + 1):
        c_i = (2 * i + 1) * a
        r_i = (-1) ** i * b
        endpoints.append((c_i, r_i))

    max_c = (2 * a) * (d // (2 * a))
    if d != max_c:
        endpoints.append((d, r_i / (2 * (d - max_c))))
    endpoints = np.array(endpoints).T

    # Rotate by theta
    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    endpoints = M.dot(endpoints)

    cs, rs = endpoints
    cs += c0
    rs += r0

    for i in range(1, endpoints.shape[1]):
        x += line_from_points(cs[i - 1], rs[i - 1], cs[i], rs[i])
        x = np.clip(x, 0, 1)

    x = x * 255
    return x.astype(np.float32)


def inverse(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x.astype(np.float32)
    return 255.0 - x


def stripe(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x.astype(np.float32)
    x[:, :7] = 255.0 - x[:, :7]
    x[:, 21:] = 255.0 - x[:, 21:]
    return x


# modified to handle a batch of images
def canny_edges(x: NDArray[np.number]) -> NDArray[np.float32]:
    x = x / 255.0

    out = np.array([feature.canny(img) for img in x]) * 255
    return out.astype(np.float32)


ALL_CORRUPTIONS: dict[str, Callable] = {
    "identity": identity,
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur,
    "glass_blur": glass_blur,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "fog": fog,
    "snow": snow,
    "spatter": spatter,
    "contrast": contrast,
    "brightness": brightness,
    "saturate": saturate,
    "jpeg_compression": jpeg_compression,
    "pixelate": pixelate,
    "elastic_transform": elastic_transform,
    "quantize": quantize,
    "shear": shear,
    "rotate": rotate,
    "scale": scale,
    "translate": translate,
    "line": line,
    "dotted_line": dotted_line,
    "zigzag": zigzag,
    "inverse": inverse,
    "stripe": stripe,
    "canny_edges": canny_edges,
}
