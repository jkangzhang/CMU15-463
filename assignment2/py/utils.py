import numpy as np

yiq_from_bgr = np.array([[ 0.114, 0.587, 0.299],
                         [-0.32134392, -0.27455667, 0.59590059],
                         [ 0.31119955, -0.52273617, 0.21153661]])
bgr_from_yiq = np.linalg.inv(yiq_from_bgr)

def _convert(array, img):
    return img @ array.T.copy()

def bgr2yiq(image):
    return _convert(yiq_from_bgr, image)

def yiq2bgr(image):
    return _convert(bgr_from_yiq, image)

def uint2float(imagein, min, max):
    out = imagein.copy()
    out = (out - min) * 1.0 / max
    out[out > 1] = 1.0
    out[out < 0] = 0.0
    return np.float32(out)

def float2uint(imagein, min, max):
    out = imagein.copy()
    out = out * max + min
    out[out > max] = max
    out[out < min] = min
    return out.astype(np.uint8)

def bgr2rgb(image):
    return image[...,::-1]



