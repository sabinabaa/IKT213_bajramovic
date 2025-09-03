import cv2
import numpy as np
import os

# --- helpers ---
def load_image(fname: str) -> np.ndarray:
    return cv2.imread(os.path.join("picture", fname))

def save_image(img: np.ndarray, fname: str):
    cv2.imwrite(os.path.join("outupts", fname), img)


# --- Task 1 ---
def padding(img: np.ndarray, border: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REFLECT)

# --- Task 2 ---
# Numpy slicing → img[y0:y1, x0:x1]
#   - First index = rows (y axis = height)
#   - Second index = cols (x axis = width)
# It keeps rows in [y0, y1) and cols in [x0, x1).
def crop(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    return img[y0:y1, x0:x1]

# --- Task 3 ---
# cv2.resize(src, dsize, interpolation)
#   - dsize is (width, height)
#   - interpolation controls quality, cv2.INTER_LINEAR is standard for down/upsample.
def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

# --- Task 4 ---
def copy_lib(img: np.ndarray) -> np.ndarray:
    return img.copy()

# --- Task 5 ---
def grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Task 6 ---
def hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Task 7 ---
# cv2.cvtColor(img, cv2.COLOR_BGR2HSV) → convert from BGR to HSV
# Hue channel in OpenCV is [0..179]. if we add +hue and mod 180, colors rotate.
# cv2.merge + cv2.split are used to separate and rejoin channels.
def hue_shifted(img: np.ndarray, hue: int) -> np.ndarray:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h = ((h.astype(np.int16) + hue) % 180).astype(np.uint8)
    merged = cv2.merge([h, s, v])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

# --- Task 8 ---
def smoothing(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (15, 15), sigmaX=0)

# --- Task 9 ---
# cv2.rotate(src, code)
#   - cv2.ROTATE_90_CLOCKWISE → rotate 90°
#   - cv2.ROTATE_180          → rotate 180°
# handles shape + pixels automatically.
def rotation(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


if __name__ == "__main__":
    inp = "lena-2.png"
    img = load_image(inp)

    save_image(padding(img, 100), "lena_pad.png")
    save_image(crop(img, 80, img.shape[1]-130, 80, img.shape[0]-130), "lena_crop.png")
    save_image(resize(img, 200, 200), "lena_resize.png")
    save_image(copy_lib(img), "lena_copy.png")
    save_image(grayscale(img), "lena_gray.png")
    save_image(hsv(img), "lena_hsv.png")
    save_image(hue_shifted(img, 50), "lena_hue.png")
    save_image(smoothing(img), "lena_blur.png")
    save_image(rotation(img, 90), "lena_rot90.png")
    save_image(rotation(img, 180), "lena_rot180.png")