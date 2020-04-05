"""
Cody Kesler
Lab 3 Image Stiching

"""
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from numpy import linalg as la
from itertools import combinations

def get_intrest_points(filenames, scale=3, plot=False):
    """
    Finds the interest points for the given image names
    :param filenames:
    :param resize:
    :return:
    """
    display_imgs = []
    imgs = []
    descriptors = []
    kps = []
    color_imgs = []
    color_copies = []
    # Load in the images
    for filename in filenames:
        color = cv.imread("./Images/" + filename)
        img = cv.cvtColor(color, cv.COLOR_BGR2GRAY)

        # Resize
        if (scale > 0):
            img = cv.resize(img, (0,0), fx=scale, fy=scale)
            color = cv.resize(color, (0,0), fx=scale, fy=scale)
        imgs.append(img)

        # Get feature points
        sift = cv.xfeatures2d.SIFT_create()

        # Get feature points
        kp, des = sift.detectAndCompute(img, None)

        # Draw feature points
        color_copy = np.copy(color)
        display_img = cv.drawKeypoints(color, kp, color_copy, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        display_imgs.append(display_img)

        # Appends stuff
        descriptors.append(des)
        kps.append(kp)
        color_imgs.append(color)
        color_copies.append(color_copy)

    if(plot):
        for img in color_copies:
            plt.imshow(img, cmap='gray')
            plt.show()

    return imgs, display_imgs, descriptors, kps, color_imgs


def match_features(base, imgs, color_imgs, descs, kps, plot=False):
    """
    Matches the interest points
    :param imgs:
    :param descs:
    :param kps:
    :param plot:
    :return:
    """
    matches = []
    n = len(imgs)-1
    # Get the forward matches for the images up to the base
    for i in range(base):
        matches.append(get_matches(descs[i], descs[i+1]))

    # Get the backward matches for the images down to the base
    backward_matches = []
    for i in range(n, base, -1):
        backward_matches.append(get_matches(descs[i], descs[i-1]))

    matches += backward_matches[::-1]

    if(plot):
        # Show the forward matches for the images up to the base
        for i in range(base):
            img3 = cv.drawMatches(color_imgs[i], kps[i], color_imgs[i+1], kps[i+1], matches[i], None,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(img3)
            plt.show()

        # Show the backward matches for the images down to the base
        for i in range(n, base, -1):
            img3 = cv.drawMatches(color_imgs[i], kps[i], color_imgs[i-1], kps[i-1], matches[i-1], None,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(img3)
            plt.show()

    return matches


def get_matches(desc1, desc2):
    """
    Filters Matches to get the best ones
    :param kp1:
    :param kp2:
    :param matches:
    :return:
    """
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


def get_homographies(base, imgs, kps, matches, tol):
    """
    Gets the correct Homographies for all pairs of consecutive images
    :param imgs:
    :param kps:
    :param matches:
    :return:
    """
    n = len(imgs)-1
    homographies = []

    # Calcualte homograhpies forward up to the base image
    for i in range(base):
        # Get homography from ransac
        homography = ransac(kps[i], kps[i+1], matches[i], tol)
        homographies.append(homography)

    homographies.append(np.eye(3).astype("float"))

    backward_matches = []
    # Calculate the homographies backward, down to the base image
    for i in range(n, base, -1):
        # Get homography from ransac
        homography = ransac(kps[i], kps[i-1], matches[i-1], tol)
        backward_matches.append(homography)

    homographies += backward_matches[::-1]

    return homographies



def ransac(kp1, kp2, matches, tol):
    """
    Computes the pairwise homography for 2 given images
    :param img1:
    :param imgs2:
    :param kp1:
    :param kp2:
    :param matches:
    :param tol:
    :return:
    """
    m = 200
    best_consensus_count = 0
    consensus_matches = []
    # Loop for m times given by (1 - (1 - p^m)^k
    for n in range(m):
        c_matches = []
        # Calculate the homography
        H = calculate_homography(kp1, kp2, matches)

        # loop though matches to calculate amount of consensus for this homography
        consensus_points = 0
        for match in matches:
            # Get first point and calculate transformed point
            pnt1 = kp1[match.queryIdx].pt
            b = np.hstack((pnt1, np.array([1])))
            # H = la.inv(H)
            pnt2_guess = H@b
            pnt2_guess /= pnt2_guess[-1]

            # Get real second point
            pnt2 = np.hstack((kp2[match.trainIdx].pt, np.array([1])))

            # Calculate difference and increment the consensus count
            if (la.norm(pnt2_guess - pnt2) < tol):
                c_matches.append(match)
                consensus_points += 1

        # If the homography is better, accept the new one
        if (consensus_points > best_consensus_count):
            consensus_matches = c_matches
            best_consensus_count = consensus_points

    return get_least_square_fit(kp1, kp2, consensus_matches)

def get_least_square_fit(kp1, kp2, consensus_matches):
    """
    Calculates the least squares fit
    :param kp1:
    :param kp2:
    :param consensus_matches:
    :return:
    """
    A = []
    b = []
    # Loop through matches
    for match in consensus_matches:
        x, y = kp1[match.queryIdx].pt
        xp, yp = kp2[match.trainIdx].pt

        # Create A matrix
        A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y])
        A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y])

        # Create b vector
        b.append(xp)
        b.append(yp)

    A = np.array(A)
    b = np.array(b)

    # Solve using least squares solver
    return np.append(la.lstsq(A, b, rcond=-1)[0], 1).reshape(3,3)


def calculate_homography(kp1, kp2, matches):
    """
    Create the homography for the two given images
    :param kp1:
    :param kp2:
    :param matches:
    :return:
    """
    # Gets random points from matches
    pts1, pts2 = get_points(kp1, kp2, matches)

    # Loops until all the points are not collinear
    while are_collinear(pts1):
        pts1, pts2 = get_points(kp1, kp2, matches)

    # Calculate Homography
    A = np.array([[pts1[0][0], pts1[0][1], 1, 0, 0, 0, -pts2[0][0] * pts1[0][0], -pts2[0][0] * pts1[0][1]],
                  [0, 0, 0, pts1[0][0], pts1[0][1], 1, -pts2[0][1] * pts1[0][0], -pts2[0][1] * pts1[0][1]],
                  [pts1[1][0], pts1[1][1], 1, 0, 0, 0, -pts2[1][0] * pts1[1][0], -pts2[1][0] * pts1[1][1]],
                  [0, 0, 0, pts1[1][0], pts1[1][1], 1, -pts2[1][1] * pts1[1][0], -pts2[1][1] * pts1[1][1]],
                  [pts1[2][0], pts1[2][1], 1, 0, 0, 0, -pts2[2][0] * pts1[2][0], -pts2[2][0] * pts1[2][1]],
                  [0, 0, 0, pts1[2][0], pts1[2][1], 1, -pts2[2][1] * pts1[2][0], -pts2[2][1] * pts1[2][1]],
                  [pts1[3][0], pts1[3][1], 1, 0, 0, 0, -pts2[3][0] * pts1[3][0], -pts2[3][0] * pts1[3][1]],
                  [0, 0, 0, pts1[3][0], pts1[3][1], 1, -pts2[3][1] * pts1[3][0], -pts2[3][1] * pts1[3][1]],
                  ])
    b = np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1], pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]])

    return np.reshape(np.hstack((la.solve(A, b), np.array([1]))), (3, 3))


def get_points(kp1, kp2, matches):
    """
    Gets random points for a homography
    :param kp1:
    :param kp2:
    :param match:
    :return:
    """
    points1 = []
    points2 = []
    for _ in range(5):

        # Get Random Points
        n = np.random.randint(0, len(matches)-1)
        points1.append(kp1[matches[n].queryIdx].pt)
        points2.append(kp2[matches[n].trainIdx].pt)

    return points1, points2


def are_collinear(points):
    """
    Checks the list of 4 points for any collinear set of 3 points
    :param pt1: tuple
    :param pt2: tuple
    :param pt3: tuple
    :return: bool: Are collinear or not
    """
    for comb in combinations(points, 3):
        pt1, pt2, pt3 = comb[0], comb[1], comb[2]
        if (pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1]) == 0):
            return True
    return False

def create_mosaic(base, imgs, color_imgs, homographies):
    """
    Stiches images together given their homographies
    :param imgs:
    :param homographies:
    :return:
    """
    n = len(imgs)
    base = n // 2
    n -= 1
    # base= 0
    ih, iw = imgs[0].shape

    heights = []
    widths = []
    xs = []
    ys = []

    for i in range(base):
        homographies[i] = homographies[i]@homographies[i+1]

    for i in range(n, base, -1):
        homographies[i] = homographies[i] @ homographies[i-1]

    for H in homographies:
        x, y, w, h = get_bounding_rectangle(H, iw, ih)
        xs.append(x)
        ys.append(y)
        heights.append(h)
        widths.append(w)

    height = np.max(heights)
    width = (np.max(xs) + np.max(widths)) - np.min(xs)

    T = np.eye(3)
    T[0,-1] = -np.min(xs)
    T[1,-1] = -np.min(ys)

    h1s = T@homographies

    morphs = []

    # Loops through forward to get warp the images
    for i in range(base+1):
        img = color_imgs[i]
        morph = cv.warpPerspective(img, h1s[i], (width, height))
        morphs.append(morph)

    # Loops through backward to get warp the images
    for i in range(n, base, -1):
        img = color_imgs[i]
        morph = cv.warpPerspective(img, h1s[i], (width, height))
        morphs.append(morph)

    mosaic = correct_blend(base, imgs, morphs)

    return cv.cvtColor(mosaic.astype('uint8'), cv.COLOR_BGR2RGB)


def correct_blend(base, imgs, morphs):
    n = len(imgs) - 1

    mosaic = morphs[base]
    for i in range(base, n):
        mosaic = np.where(mosaic == np.array([0,0,0]), morphs[i+1], mosaic*.9 + .1*morphs[i+1])

    for i in range(base, 0, -1):
        mosaic = np.where(mosaic == np.array([0, 0, 0]), morphs[i - 1], mosaic * .9 + .1*morphs[i - 1])

    return mosaic


def get_bounding_rectangle(H, w, h):
    """
    gets the bounding rectangle for the homogrphied image
    :param H:
    :param w:
    :param h:
    :return:
    """
    c1 = np.array([0, 0, 1])
    c2 = np.array([w, 0, 1])
    c3 = np.array([0, h, 1])
    c4 = np.array([w, h, 1])

    c1 = H @ c1
    c1 /= c1[-1]
    c1 = c1.astype('int')
    c2 = H @ c2
    c2 /= c2[-1]
    c2 = c2.astype('int')
    c3 = H @ c3
    c3 /= c3[-1]
    c3 = c3.astype('int')
    c4 = H @ c4
    c4 /= c4[-1]
    c4 = c4.astype('int')

    return cv.boundingRect(np.array([c1[:-1], c2[:-1], c3[:-1], c4[:-1]]))


def stitch_images(filenames, tol, plot, scale):
    """
    Stiches Images together
    :param filesnames:
    :param plot:
    :return:
    """

    # Run Sift and get decriptors and interest points
    imgs, display_imgs, descriptors, kps, color_imgs = get_intrest_points(filenames, scale=scale, plot=plot)

    base = len(imgs) // 2
    # Gets matches for interest points
    matches = match_features(base, imgs, color_imgs, descriptors, kps, plot=plot)

    # Get the pairwise homographies
    homographies = get_homographies(base, imgs, kps, matches, tol)

    # Create the mosaic of the images
    mosaic = create_mosaic(base, imgs, color_imgs, homographies)

    if(plot):
        plt.imshow(mosaic)
        plt.show()

    return mosaic


if __name__ == "__main__":
    files = ["campus1.jpg", "campus3.jpg", "campus2.jpg"] #, "./Images/campus3.jpg"]
    # files = ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"]
    # files = ["11.jpg", "12.jpg", "13.jpg"]
    # files = ["squaw1.png","squaw2.png","squaw3.png"]
    # files = ["templev1.png", "templev2.png", "templev3.png"]
    plot = True
    scale = .2
    tol = 20
    mosaic = stitch_images(files, tol, plot, scale)
