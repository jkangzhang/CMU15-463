import cv2
import numpy as np
import sys
from utils import bgr2yiq, yiq2bgr, uint2float, float2uint, bgr2rgb
import matplotlib.pyplot as plt

class EulerianVideoMagnification:

    def __init__(self):
        self.fps = 0
        self.framesCount = 0

    """
    Convert the given video to frames by using cv2 method

    Parameters
    ------------
    videoPath: the path of the video

    Returns
    ------------
    frames
    """
    def _convertVideoToFrames(self, videoPath, limit):
        videoCapture = cv2.VideoCapture(videoPath)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("video count:{}".format(count))
        assert(count > 0)
        limit = count if limit <= 0 or limit > count else limit
        frames = []
        while limit != 0:
            retval, img = videoCapture.read()
            limit -= 1
            if not retval:
                print("[Warning]Skip a bad frame with return value:{}".format(retval))
                continue
            img = np.array(img)
            frames.append(img)
        frames = np.array(frames)
        return frames, fps

    """
    Construct video with given frames using cv2 method

    Parameters
    ------------
    frames: frames
    outPath: video will be writen in the given path

    Returns
    ------------
    None
    """
    def _constructVideoFromFrames(self, frames, outPath):
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        h, w=frames[0].shape[0:2]
        writer = cv2.VideoWriter(outPath, fourcc, self.fps, (w, h), 1)
        for frame in frames:
            writer.write(frame)
        writer.release()

    """
    Construct image's pyramids

    Parameters
    ------------
    image: target image
    depth: levels of the pyramid

    Returns
    ------------
    the pyramid of the image
    """
    def _constructPyramid(self, image, depth):
        pyramid = []
        cur = image.copy()
        while depth > 1:
            depth -= 1
            h, w = cur.shape[0:2]
            if h < 4 or w < 4:
                break
            curBlur = cv2.GaussianBlur(cur, ksize=(3, 3), sigmaX=2)
            down = cv2.resize(curBlur, ((w + 1) // 2, (h + 1) // 2))
            up = cv2.resize(down, (w, h))
            blurUp = cv2.GaussianBlur(up, ksize=(3, 3), sigmaX=2)
            h = cur - blurUp
            pyramid.append(h)
            cur = down
        pyramid.append(cur)
        return pyramid

    """
    Reconstruct image from its pyramids

    Parameters
    ------------
    pyramid: image's pyramid

    Returns
    ------------
    the reconstructed image
    """
    def _reconstructImageFromPyramid(self, pyramid):
        cur = pyramid[-1]
        i = len(pyramid) - 2
        while i >= 0:
            h, w = pyramid[i].shape[0:2]
            up = cv2.resize(cur, (w, h))
            upBlur = cv2.GaussianBlur(up, ksize=(3, 3),sigmaX=2)
            cur = upBlur + pyramid[i]
            i -= 1
        return cur

    """
    Construct pyramids of each frame of the given path.

    Parameters
    ------------
    videoPath: the path of the video
    depth: the max depth of the pyramid
    limit: optional, the count of frames need to deal with, the default would
            every frame.

    Returns
    ------------
    Pyramids of the echo frames of the video (Each pyramid is actually a set of images)
    """
    def constructVideoPyramids(self, videoPath, depth, limit = -1):
        # get frames of the video
        frames, self.fps = self._convertVideoToFrames(videoPath, limit)
        # print(self.fps)
        self.framesCount = len(frames)
        # convert frames to float
        # convert float to yiq
        images = [bgr2yiq(uint2float(image, 0, 255)) for image in frames]
        # print(images.shape)
        # images2 = [float2uint(yiq2bgr(image), 0, 255) for image in images]
        # construct pyramid from images

        pyr0 = self._constructPyramid(images[0], depth)
        resultList=[]
        for i in range(depth):
            curPyr = np.zeros([len(images)]+list(pyr0[i].shape))
            resultList.append(curPyr)

        for fn in range(len(images)):
            pyrOfFrame = self._constructPyramid(images[fn], depth)
            for i in range(depth):
                resultList[i][fn]=pyrOfFrame[i]
        image_s = float2uint(yiq2bgr(resultList[3][0]), 0, 255)
        return resultList

    """
    Construct video with given frames using cv2 method

    Parameters
    ------------
    pyramids: pyramids of the each frame
    outPath: video will be writen in the given path

    Returns
    ------------
    None
    """
    def reconstructVideoFromPyramids(self, pyramids, outPath):
        images = []
        for i in range(len(pyramids[0])):
            imagePyramid = []
            for j in range(len(pyramids)):
                imagePyramid.append(pyramids[j][i])
            images.append(self._reconstructImageFromPyramid(imagePyramid))
        frames = [float2uint(yiq2bgr(image), 0, 255) for image in images]
        self._constructVideoFromFrames(frames, outPath)

    def _getFilteredImage(self, frames, fps, low, high):
        import scipy.fftpack as fftpack
        import scipy.signal as signal

        fft=fftpack.fft(frames, axis=0)
        freq = fftpack.fftfreq(frames.shape[0], d=1.0 / fps)
        # print(freq.shape, fft.shape)
        # print(freq)
        bound_low = (np.abs(freq - low)).argmin()
        bound_high = (np.abs(freq - high)).argmin()
        if (bound_low==bound_high) and (bound_high<len(fft)-1):
            bound_high+=1
        print(bound_low, bound_high)
        # bound_low = 10
        # bound_high = 20
        # print(bound_low, bound_high)
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        # ifft = np.fft.ifft2(fft)
        # ifft = np.abs(iffkt)
        # print(ifft.shape)
        iff=fftpack.ifft(fft, axis=0)
        # fshift = np.fft.fftshift(fft)
        # magnitude_spectrum = np.log(np.abs(fshift))
        # return magnitude_spectrum
        return abs(iff)

        # order = 52
        # omega = 0.5 * fps
        # low = low / omega
        # high = high / omega
        # b, a = signal.butter(order, [low, high], btype='band')
        # y = signal.lfilter(b, a, image, axis=0)
        # return y

        # freq2 = np.fft.fftfreq(image.shape[2], d=1.0 / fps)
        # ff = cv2.merge(freq, freq1)
        # return ff
        # return [np.abs(freq), np.abs(freq1)]
        # return magnitude_spectrum
        # fft = np.fft.fft2(image)
        # print(fft.shape)
        # print(image.shape)
        # fft = cv2.dft(image, flags = cv2.DFT_COMPLEX_OUTPUT)
        # print(np.max(fft), np.min(fft))
        # print(fft[0][0][0:10])
        # print(frequencies)
        # bound_low = (np.abs(frequencies - low)).argmin()
        # print(bound_low)
        # bound_high = (np.abs(frequencies - high)).argmin()
        # if (bound_low==bound_high) and (bound_high<len(fft)-1):
            # bound_high+=1
        # fft[:bound_low] = 0
        # fft[bound_high:-bound_high] = 0
        # fft[-bound_low:] = 0
        # iff=fftpack.ifft(fft, axis=0)
        # return np.abs(iff)

    def amplifyPyramids(self, pyramids, low, high, alpha):
        filteredPyramids = []
        aPyramids = []
        for i in range(len(pyramids)):
            # alpha >>= i
            tt = pyramids[i].copy()
            filteredPyramid = self._getFilteredImage(tt, self.fps, low, high)
            # tt[:,:,:,0] += filteredPyramid[:,:,:, 0] * alpha
            # tt[:,:,:,1] += filteredPyramid[:,:,:, 1] * alpha
            # tt[:,:,:,2] += filteredPyramid[:,:,:, 2] * alpha
            if i == len(pyramids) - 1:
                filteredPyramid = self._getFilteredImage(tt, self.fps, low, high)
                tt[:,:,:,0] += filteredPyramid[:,:,:, 0]
                tt[:,:,:,1] += filteredPyramid[:,:,:, 1] * alpha * 1.2
                tt[:,:,:,2] += filteredPyramid[:,:,:, 2] * alpha * 1.5
                # print(tt[:,:,:,1])
                # filteredPyramids.append(filteredPyramid)
            aPyramids.append(tt)

        # new_i = self._getFilteredImage(pyramids[0][3], self.fps, low, high, self.frames)
        # cv2.imwrite("1.png", float2uint(yiq2bgr(filteredPyramid[0][3]), 0, 255))
        # pp = pyramids[0][3].copy()
        # pp[:,:,0] += new_i[:,:,:] * 100
        # cv2.imwrite("2.png", float2uint(yiq2bgr(pp), 0, 255))
        # print(new_i.shape)
        # self.drawTemporal(filteredPyramid, 240, 200)
        return aPyramids

    """
    Draw temporal curve for pyramids

    Parameters
    ------------
    pyramids: pyramids of the each frame
    outPath: video will be writen in the given path

    Returns
    ------------
    None
    """
    def drawTemporal(self, pyramids, x, y):
        def _get_all_pixels(pyramids, level, x, y, channel):
            pixels = []
            for pyramid in pyramids:
                image = pyramid[level]
                pixels.append(image[x >> level][y >> level][channel])
            return pixels
        newPyramids = []
        for i in range(len(pyramids[0])):
            imagePyramid = []
            for j in range(len(pyramids)):
                imagePyramid.append(pyramids[j][i])
            newPyramids.append(imagePyramid)

        images_frame0 = [float2uint(yiq2bgr(image), 0, 255) for image in newPyramids[0]]
        plt.figure(figsize=(20, 8))
        plt.title("Temproal")
        radius=10
        rowcount = len(images_frame0)
        plt.subplots_adjust(hspace=0.6, wspace=0.6)
        # plt.tight_layout()
        x_axis = [t*1000/self.fps for t in range(len(newPyramids))]
        channels = ['Y', 'I', 'Q']
        for i in range(rowcount):
            # show image
            image_to_show = bgr2rgb(images_frame0[i])
            ax = plt.subplot(rowcount, 4, i * rowcount + 1)
            rect = plt.Rectangle([(x-radius)>>i, (y-radius)>>i], radius*2>>i, radius*2>>i, edgecolor='Red', facecolor='red')
            plt.title("frame0,{}".format(i), fontsize=10)
            plt.imshow(image_to_show)
            ax.add_patch(rect)
            # show channel
            for j in range(len(channels)):
                plt.subplot(rowcount, 4, i * rowcount + j + 2)
                plt.title(channels[j], fontsize=10)
                y_axis = _get_all_pixels(newPyramids, i, x, y, j)
                plt.plot(x_axis, y_axis)
                plt.xlabel('Time(ms)', fontsize=6)
                plt.ylabel('Pixel Value', fontsize=6)
        plt.show()

if __name__ == "__main__":
    video = sys.argv[1]
    outpath = sys.argv[2]
    euler = EulerianVideoMagnification()
    pyramids = euler.constructVideoPyramids(video, 7, 30)
    # euler.drawTemporal(pyramids, 240, 200)
    pyramids = euler.amplifyPyramids(pyramids, 0.83 / 1.4, 0.8 / 1.4, 130)
    # euler.drawTemporal(pyramids, 240, 200)
    euler.reconstructVideoFromPyramids(pyramids, outpath)
