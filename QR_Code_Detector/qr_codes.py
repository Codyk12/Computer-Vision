"""
Cody Kesker
QR Code Reader
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import decoder
from scipy import stats


def find_positioning_squares(file, plot):
    """
    Finds the corners of the Squares of QR Codes
    :param filename:
    :return:
    """
    # Load in img, set stuff up and suppress noise
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    img = img / 255
    img[img <= .5] = 0
    img[img > .5] = 1

    img = cv.resize(img, (200,200))

    plt.imshow(img, cmap='gray')
    plt.title('Original for ' + file)
    plt.show()


    # set up dictionary for checking other orientations
    d = {}
    d[0] = img
    d[1] = img.T
    centers = np.zeros_like(img)
    lengths = []

    tot_unit_len = 0
    for key in d:
        # get the segement lengths for each row
        length = get_segment_counts(d[key])
        lengths.append(length)

        # get the center candidates for the points
        center, unit_len = get_candiates(d[key], length)
        tot_unit_len += unit_len

        # If the img is the transposed one, flip it back
        if(key == 1):
            centers += center.T
        else:
            centers += center

    if (plot):
        plt.imshow(centers, cmap='gray')
        plt.title('Centers for ' + file)
        plt.show()

    # Transpose and swap columns to get points in (x, y) format
    points = np.array(np.where(centers > 1)).T
    points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()

    return img, points, lengths[0], tot_unit_len // len(d)

def reorder_points(img, points):
    """
    This function handles rotation in a QR Code
    :param img:
    :param points:
    :return:
    """
    y_half = img.shape[0] // 2
    x_half = img.shape[1] // 2
    quadrants = []
    for point in points:
        if(point[0] <= x_half and point[1] <= y_half):
            quadrants.append(1)
        elif (point[0] >= x_half and point[1] <= y_half):
            quadrants.append(2)
        elif (point[0] <= x_half and point[1] >= y_half):
            quadrants.append(3)
        elif (point[0] >= x_half and point[1] >= y_half):
            quadrants.append(4)

    # First Rotation
    if np.equal(quadrants, [1, 2, 4]).all():
        return np.array([points[1], points[2], points[0]])

    # Second Rotation
    if np.equal(quadrants, [2, 3, 4]).all():
        return np.array([points[2], points[1], points[0]])

    # Third rotation
    if np.equal(quadrants, [1, 3, 4]).all():
        return np.array([points[1], points[0], points[2]])

    return points


def get_segment_counts(img):
    """
    Gets the zeros and ones and counts for eah row in image
    :param img:
    :return:
    """
    row_segments = []
    lengths = []
    for i, row in enumerate(img):
        segments = []
        length = []
        i = 0

        while i < len(row) - 1:
            start = row[i]
            pixel = row[i]

            segment = []
            leng = 0
            while pixel == start:
                segment.append(pixel)
                i += 1
                leng += 1
                if (i == len(row)):
                    break
                pixel = row[i]
            length.append(leng)
            segments.append(segment)
        row_segments.append(segments)
        lengths.append(length)

    return np.array(lengths)


def get_candiates(img, lengths):
    """
    Marks the lines that could be along the center of an alignment part of a QR Code
    :param img:
    :param lengths:
    :return:
    """
    tol = int(img.shape[0]*.02)
    unit_lens = []
    centers = np.zeros_like(img)
    for i, row in enumerate(lengths):
        # Only if there are at least 5 segments for each row
        if (len(row) >= 5):
            for j in range(len(row) - 4):
                # Check for the 1-1-3-1-1 ratio
                if (row[j] > 5 and
                    np.isclose(row[j], row[j + 1], atol=tol) and
                    np.isclose(row[j + 1] * 3, row[j + 2], atol=tol*3) and
                    np.isclose(row[j + 1], row[j + 3], atol=tol) and
                        np.isclose(row[j + 1], row[j + 4], atol=tol)):
                    j_indx = int(np.sum(row[:j]) + row[j]*3.5)
                    if(j_indx < len(centers[i])):
                        centers[i, j_indx] = 1
                        unit_lens.append(row[j])

    return centers, np.average(unit_lens).astype('int')

def maximal_suppression(centers):
    """
    Suppresses points that didnt get very high votes
    :param centers:
    :return:
    """
    mask = centers < 2
    centers[mask] = 0

    return centers

def find_fourth_point(img, centers, unit_len):
    """
    Gets the fourth point for the homography
    :param centers:
    :return:
    """
    x_diff = centers[1][1] - centers[0][1]
    y_diff = centers[1][0] - centers[0][0]
    fourth = centers[2][0] + y_diff, centers[2][1] + x_diff

    points = np.append(centers, fourth).reshape((4,1,2))

    # Get relative squared points
    indent = int(unit_len*4)
    length = 200
    corrected_points = np.array([[indent, indent],
                                [length-indent, indent],
                                [indent, length-indent],
                                [length-indent, length-indent]])

    return points, corrected_points

def calculate_homography(points, corrected_points):
    """
    Calculates the homography for the given points
    :param points:
    :return:
    """
    h, status = cv.findHomography(points, corrected_points, cv.RANSAC, 5.0)
    return h

def warp_trim_img(img, h, r, indent, plot):
    """
    Warp the img given the homography and then trim it down
    :param img:
    :param h:
    :return:
    """

    warpped_img = cv.warpPerspective(img, h, (200, 200), borderValue=1, borderMode=cv.BORDER_CONSTANT)

    if (plot):
        plt.imshow(warpped_img, cmap='gray')
        plt.title('Warpped for ' + file)
        plt.show()

    return warpped_img


def discretize(img, cnt):
    """
     The simplest way to do this is to count the number of black and white pixels in a region and assign the block to the highest count.
      However, to do this, the size of the QR code needs to be determined. All QR codes have an odd number of bits per row and column,
       starting at size 21 and incrementing by 4 (i.e. 21x21, 25x25, 29x29, ...). For this lab, you will only need to check for sizes from 21 to 33.

    To check if a QR code matches a given size, discretize the QR code asumming the given size. Then, determine if the timing
    information alternates in the appropriate manner (see the Understanding QR Codes for more information). If the timing information
    is valid, then you can assume that the QR code is the given size.

    Once you have the correct size, discretize the QR code accordingly and return a Numpy array of True/False values.
    :param img:
    :return:
    """

    sizes = np.append(np.array([cnt]), 200 // np.array([21, 25, 29, 33, 37]))
    url = "Couldn't Read QR Code"
    for size in sizes:
        cnt = size
        descrete = []
        for i in range(cnt//2, len(img), cnt):
            row = []
            for j in range(cnt//2, len(img[i]), cnt):
                mode = stats.mode(img[i-cnt//2:i+cnt//2, j-cnt//2:j+cnt//2], axis=None)
                row.append(mode[0][0])

            descrete.append(row)

        descrete = np.array(descrete)
        try:
            url = decoder.decode(descrete)
            plt.imshow(descrete, cmap='gray')
            plt.title("Discretized for " + file)
            plt.show()
            break
        except Exception as e:
            print(e)
            pass

    return url


def main(file, plot):
    r = 1

    # Find the positions of the corners
    img, points, segments, unit_len = find_positioning_squares(file, plot)

    points = reorder_points(img, points)

    # Find the fourth point and the alligned corners
    four_points, corrected_points = find_fourth_point(img, points, unit_len)

    # Create the homography
    h = calculate_homography(four_points, corrected_points)

    # Warp and trim img
    warp_img = warp_trim_img(img, h, r, unit_len, plot)

    # Discretize the cropped qr code img
    url = discretize(warp_img, unit_len)
    print('**********************************')
    print('URL for', file, "is:", url)
    print('**********************************')
    return


def one_d(file):
    """
    Reads a one dimentional barcode
    :param file:
    :return:
    """
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    img = img / 255
    img[img <= .5] = 0
    img[img > .5] = 1

    # count the pixel lengths of strips
    counts = get_segment_counts(img)[0]

    counts = counts[7:]

    dct = {'122231': '.', '121124': 'a', '121421': 'b', '141122': 'c',
     '141221': 'd', '112214': 'e', '112412': 'f', '122114': 'g',
     '122411': 'h', '142112': 'i', '142211': 'j', '241211': 'k',
     '221114': 'l', '413111': 'm', '241112': 'n', '134111': 'o',
     '111242': 'p', '121142': 'q', '121241': 'r', '114212': 's',
     '124112': 't', '124211': 'u', '411212': 'v', '421112': 'w',
     '421211': 'x', '212141': 'y', '214121': 'z', '233111': ''}

    characters = []
    output = ""
    for i in range(len(counts))[::6]:

        character = np.array(counts[i:i+6])
        character = np.ceil(character / 2).astype('int')
        char = ""
        for c in character:
            char += str(c)

        if(char in dct.keys()):
            output += dct[char]

    print(output)

    return output

if __name__ == "__main__":
    plots = [True, True, True, True, True]
    # plots = [False, False, False, False, False]
    filenames = ['QR_codes/test1.png', 'QR_codes/test2.png', 'QR_codes/test5.png', 'QR_codes/test4.png',
                 'QR_codes/test3.png']

    # for i, file in enumerate(filenames):
    #     print(file)
    #     main(file, plots[i])
    #
    # challenge = ['QR_codes/challenge4.png']#, 'QR_codes/challenge3.png']
    #
    # for i, file in enumerate(challenge):
    #     print(file)
    #     main(file, plots[i])

    challenge = 'QR_codes/challenge5.png'
    one_d(challenge)
