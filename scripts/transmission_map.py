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
    res = np.zeros_like(img, dtype=np.double)
    logI = np.log(img.astype(np.double))

    for i in range(k):
        g = cv2.GaussianBlur(img, (0, 0), sigmas[i])
        logG = np.log(g.astype(np.double))
        res += w[i] * (logI - logG)

    # return cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    return res

def MSRCR(img, w, sigmas):
    # MSR
    Lmsr = MSR(img, w, sigmas)
    
    # CR
    # filter_size = 15 # not sure how to set this
    # h, w = img.shape
    # M = filter_size * filter_size
    # extended = np.pad(img, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), 'reflect')
    # Lmsrcr = np.zeros((h, w), dtype=np.double)
    # for i in range(h):
    #     for j in range(w):
    #         window = extended[i:i+filter_size, j:j+filter_size]
    #         _sum = np.sum(window)
    #         C = np.log(img[i][j] / (1 / M * _sum))
    #         Lmsrcr[i][j] = C * Lmsr[i][j]

    # delta = 128
    # kappa = 128
    # Lmsrcr = delta * Lmsrcr + kappa

    # return cv2.normalize(Lmsrcr, None, 0, 255, cv2.NORM_MINMAX)
    return Lmsr

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
    T = max_filter(T, 15)
    T = blur_filter(T, 20)

    return T

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