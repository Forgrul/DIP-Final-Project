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

def main():
    args = parse_args()
    img = cv2.imread(args.input_path)
    fig = plt.figure(figsize=(15, 5))
    for i in range(2, -1, -1):
        channel = img[:,:,i]
        fig.add_subplot(1, 3, 3-i)
        counts, bins = np.histogram(channel, range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.title("BGR"[i])
        # plt.show()
    plt.savefig(args.output_path)

if __name__ == "__main__":
    main()