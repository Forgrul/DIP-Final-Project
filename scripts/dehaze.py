from atmospheric_value import get_atmospheric_value
from transmission_map import get_luminance, MSRCR, optimize_transmission
import argparse
import cv2
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()
    return args

def dehaze(img, t, A):
    t[t < 0.1] = 0.1
    b, g, r = cv2.split(img)
    b = (b - A) / t + A
    g = (g - A) / t + A
    r = (r - A) / t + A

    return cv2.merge([b,g,r])

def main():
    args = parse_args()
    img = cv2.imread(args.input_path)

    A = get_atmospheric_value(img, 2)

    luminance = get_luminance(img)
    rough_transmission = MSRCR(luminance, (0.1, 0.1, 0.8), (1.5, 40, 150))
    optimized_transmission = optimize_transmission(luminance, rough_transmission)

    res = dehaze(img, optimized_transmission, A)
    
    # plt.imshow(optimized_transmission, cmap='grey')
    # plt.show()
    cv2.imwrite(args.output_path, res)

if __name__ == "__main__":
    main()