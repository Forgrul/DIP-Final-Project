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

def get_dark_channel(img):
    filter_size = 15
    extended = np.pad(img, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2), (0, 0)), 'reflect')
    
    h, w, _ = img.shape
    res = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            subarray = extended[i:i+filter_size, j:j+filter_size]
            res[i][j] = np.min(subarray)

    # plt.imshow(res, cmap='grey')
    # plt.show()
    return res

def get_decision_image(img):
    h, w, _ = img.shape
    res = np.zeros((h, w), dtype='double')

    for i in range(h):
        for j in range(w):
            # openCV use BGR order
            b = int(img[i][j][0])
            g = int(img[i][j][1])
            r = int(img[i][j][2])
            # val = (b**2 + g**2 + r**2 - (b + g + r)**2 / 3) ** 0.5
            res[i][j] = ((b**2 + g**2 + r**2) - (b + g + r)**2 / 3) ** 0.5
    
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    # plt.imshow(res, cmap='grey')
    # plt.show()
    return res

def get_atmospheric_value(img, threshold):
    dark_channel = get_dark_channel(img)
    decision_image = get_decision_image(img)
    dark_channel = dark_channel.flatten()
    decision_image = decision_image.flatten()

    # get the top 0.1%
    # num_pixels = img.shape[0] * img.shape[1]
    # sorted_indices = np.argsort(dark_channel)
    # candidates_indices = sorted_indices[(999 * num_pixels // 1000):]
    # candidates = dark_channel[candidates_indices]
    # candidates_value = decision_image[candidates_indices]

    # only use pixels whose f(x) > threshold
    # filtered = candidates[candidates_value > threshold]

    # method 2, filter then sample
    mask = decision_image > threshold
    candidates = dark_channel[mask]
    winners = np.sort(candidates)[(999 * len(candidates) // 1000):]
    
    return np.mean(winners)
    # return np.mean(filtered)

def main():
    args = parse_args()
    img = cv2.imread(args.input_path)

    A = get_atmospheric_value(img, 6)
    print(A)
    
    # plt.imshow(img)
    # plt.show()
    # cv2.imwrite(args.output_path, res)


if __name__ == "__main__":
    main()