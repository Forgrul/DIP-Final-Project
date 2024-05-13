import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()
    return args

def get_luminance(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return ycrcb[:, :, 0]

def MSR(img, w, sigmas):
    k = len(w)
    res = np.zeros_like(img, dtype='double')
    logI = np.log(img.astype('double') + 1)

    for i in range(k):
        g = cv2.GaussianBlur(img, (0,0), sigmas[i])
        logG = np.log(g.astype('double') + 1)
        res += (w[i] * (logI - logG))

    # res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    # res = res.astype('uint8')
    # print(np.min(res), np.max(res))
    # abs res: 0 ~ 1.67
    # res: -1.67~0.53
    return res

def MSRCR(img, w, sigmas):
    # MSR
    Lmsr = MSR(img, w, sigmas)

    # CR
    filter_size = 15 # not sure how to set this
    h, w = img.shape
    M = h * w
    extended = np.pad(img, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), 'reflect')
    Lmsrcr = np.zeros((h, w), dtype=np.double)
    for i in range(h):
        for j in range(w):
            window = extended[i:i+filter_size, j:j+filter_size]
            _sum = np.sum(window)
            C = np.log((img[i][j] + 1) / (1 / M * _sum + 1)) # filter_size=15, C ~= 6 / filter_size=50, C ~= 4.36
            Lmsrcr[i][j] = C * Lmsr[i][j]

    delta = 128
    kappa = 128
    Lmsrcr = delta * Lmsrcr + kappa
    # Lmsrcr = np.abs(Lmsrcr)

    # sort_L = np.sort(Lmsrcr, None)
    # N = Lmsrcr.size
    # Vmin = sort_L[int(N * 0.01)]
    # Vmax = sort_L[int(N * 0.99) - 1]
    # Lmsrcr[Lmsrcr < Vmin] = Vmin
    # Lmsrcr[Lmsrcr > Vmax] = Vmax

    # return cv2.normalize(Lmsrcr, None, 0, 255, cv2.NORM_MINMAX)
    # return np.abs(Lmsrcr)
    return Lmsrcr

def max_filter(img, size):
    extended = np.pad(img, ((size//2, size//2), (size//2, size//2)), 'reflect')
    res = np.zeros_like(img)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            window = extended[i:i+size, j:j+size]
            res[i][j] = np.max(window)
    
    return res

def blur_filter(img, size):
    extended = np.pad(img, ((size//2, size//2), (size//2, size//2)), 'reflect')
    res = np.zeros_like(img)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            window = extended[i:i+size, j:j+size]
            res[i][j] = np.mean(window)
    
    return res

def optimize_transmission(img, rough_transmission):
    miu = 5
    T = miu * rough_transmission
    T = T - img
    # T = max_filter(T, 15)
    # T = blur_filter(T, 20)

    # sort_T = np.sort(T, None)
    # N = T.size
    # Vmin = sort_T[int(N * 0.001)]
    # Vmax = sort_T[int(N * 0.999) - 1]
    # T[T < Vmin] = Vmin
    # T[T > Vmax] = Vmax

    return cv2.normalize(T, None, 0, 1, cv2.NORM_MINMAX)

def main():
    args = parse_args()
    img = cv2.imread(args.input_path)

    img = get_luminance(img)
    rough_transmission = MSRCR(img, (0.1, 0.1, 0.8), (1.5, 40, 150))

    optimized_transmission = optimize_transmission(img, rough_transmission)
    # print(optimized_transmission)
    
    plt.imshow(optimized_transmission, cmap='grey')
    plt.show()
    # cv2.imwrite(args.output_path, optimized_transmission)


if __name__ == "__main__":
    main()