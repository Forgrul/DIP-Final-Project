from atmospheric_value import get_atmospheric_value
from transmission_map import get_luminance, MSRCR, optimize_transmission
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-t", "--threshold")
    args = parser.parse_args()
    return args

def simplest_color_balance(channel,s1=0.01,s2=0.01):
    '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb). 
    Only suitable for 1-channel image'''
    sort_img=np.sort(channel,None)
    N=channel.size
    Vmin=sort_img[int(N*s1)]
    Vmax=sort_img[int(N*(1-s2))-1]
    channel[channel<Vmin]=Vmin
    channel[channel>Vmax]=Vmax
    return (channel-Vmin)*255/(Vmax-Vmin)

def dehaze(luminance, t, A):
    t[t < 0.1] = 0.1
    luminance = (luminance - A) / t + A
    luminance[luminance < 0] = 0
    luminance[luminance > 255] = 255

    return luminance.astype('uint8')

def adjust_color(img):
    b, g, r = cv2.split(img)
    
    b = b.astype('double')
    g = g.astype('double')
    r = r.astype('double')
    # b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    # g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    # r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    b = simplest_color_balance(b)
    g = simplest_color_balance(g)
    r = simplest_color_balance(r)
    b = b.astype('uint8')
    g = g.astype('uint8')
    r = r.astype('uint8')
    return cv2.merge([b,g,r])

def main():
    args = parse_args()
    img = cv2.imread(args.input_path)

    A = get_atmospheric_value(img, threshold=int(args.threshold))
    print("Atmospheric value A =", A)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    luminance = ycrcb[:,:,0]
    cr = ycrcb[:,:,1]
    cb = ycrcb[:,:,2]
    rough_transmission = MSRCR(luminance, (0.1, 0.1, 0.8), (1.5, 40, 150))
    optimized_transmission = optimize_transmission(luminance, rough_transmission)

    res = dehaze(luminance, optimized_transmission, A)

    res_ycrcb = cv2.merge([res,cr,cb])
    
    res = cv2.cvtColor(res_ycrcb, cv2.COLOR_YCR_CB2BGR)
    res = adjust_color(res)
    # show = cv2.cvtColor(res_ycrcb, cv2.COLOR_YCR_CB2RGB)
    # plt.imshow(show)
    # plt.show()
    cv2.imwrite(args.output_path, res)

if __name__ == "__main__":
    main()