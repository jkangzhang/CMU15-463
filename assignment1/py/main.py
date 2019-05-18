import os
import sys
import cv2
import numpy as np

class InCameraPipeline:
    def __init__(self, path):
        self.raw = cv2.imread(path, -1)
        self.raw = self.raw.astype(np.float64)
        print(self.raw.shape)
        self.rgb = 0

    def linearization(self, min_val, max_val):
        self.raw = (self.raw - min_val) / (max_val - min_val + 1)
        self.raw[self.raw < 0] = 0
        self.raw[self.raw > 1] = 1
        return self.raw

    def AWB(self, output_file, mode=0):
        r = self.raw[::2, ::2]
        b = self.raw[1::2, 1::2]
        g1 = self.raw[::2, 1::2]
        g2 = self.raw[1::2, ::2]
        g = np.concatenate((g1, g2), axis=0)
        if mode == 0:
            r_r = r.mean()
            r_g = g.mean()
            r_b = b.mean()
        else:
            r_r = r.max()
            r_g = g.max()
            r_b = b.max()
        self.raw[::2, ::2] *= r_g / r_r
        self.raw[1::2, 1::2] *= r_g / r_b

        img = self.raw.copy()
        img *= 255
        img = img.astype(np.uint8)
        cv2.imwrite(output_file, img)
        return self.raw

    def demosaic(self, output_file):
        g = np.zeros(shape=self.raw.shape, dtype=self.raw.dtype)
        r = np.zeros(shape=self.raw.shape, dtype=self.raw.dtype)
        b = np.zeros(shape=self.raw.shape, dtype=self.raw.dtype)
        r[::2, ::2] = self.raw[::2, ::2]
        b[1::2, 1::2] = self.raw[1::2, 1::2]
        g[::2, 1::2] = self.raw[::2, 1::2]
        g[1::2, ::2] = self.raw[1::2, ::2]
        kernel_g = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])
        kernel_br = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
        b_new = cv2.filter2D(b, -1, kernel_br)
        g_new = cv2.filter2D(g, -1, kernel_g)
        r_new = cv2.filter2D(r, -1, kernel_br)
        self.rgb = cv2.merge((b_new, g_new, r_new))
        img2 = self.rgb.copy()
        img2 *= 255
        img2 = img2.astype(np.uint8)
        cv2.imwrite(output_file, img2)
        return self.rgb

    def brightness(self, scale, output_file):
        self.rgb *= scale
        self.rgb[self.rgb > 1] = 1
        img2 = self.rgb.copy()
        img2 *= 255
        img2 = img2.astype(np.uint8)
        cv2.imwrite(output_file, img2)
        return self.rgb

    def gamma(self):
        dark_pixel = (self.rgb <= 0.0031308)
        bright_pixel = (self.rgb > 0.0031308)
        self.rgb[dark_pixel] *= 12.92
        self.rgb[bright_pixel] = (1 + 0.055) * (self.rgb[bright_pixel]**(1 / 2.4)) - 0.055
        img2 = self.rgb.copy()
        img2 *= 255
        img2 = img2.astype(np.uint8)
        cv2.imwrite("gamma.png", img2)
        return self.rgb

    def compression(self, output_file):
        self.rgb = (self.rgb * 255).astype(np.uint8)
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, bgr, (cv2.IMWRITE_JPEG_QUALITY, 95))

def main():
    raw_img_path = sys.argv[1]
    raw_pipeline = InCameraPipeline(raw_img_path)
    raw_pipeline.linearization(2047, 15000)
    # raw_pipeline.AWB('grey_world.png', 0)
    raw_pipeline.AWB('white_world.png', 1)
    raw_pipeline.demosaic('demosaic.png')
    raw_pipeline.brightness(1.24, 'brightness.png')
    raw_pipeline.gamma()
    raw_pipeline.compression("result.jpg")

if __name__ == '__main__':
    main()
