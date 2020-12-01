import numpy as np
import cv2
from PIL import Image


class HomographicAlignment:
    """
    Apply homographic alignment on background to match with the source image.
    """
    
    def __init__(self):
        self.detector = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    def __call__(self, src, bgr):
        src = np.asarray(src)
        bgr = np.asarray(bgr)

        keypoints_src, descriptors_src = self.detector.detectAndCompute(src, None)
        keypoints_bgr, descriptors_bgr = self.detector.detectAndCompute(bgr, None)

        matches = self.matcher.match(descriptors_bgr, descriptors_src, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        num_good_matches = int(len(matches) * 0.15)
        matches = matches[:num_good_matches]

        points_src = np.zeros((len(matches), 2), dtype=np.float32)
        points_bgr = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points_src[i, :] = keypoints_src[match.trainIdx].pt
            points_bgr[i, :] = keypoints_bgr[match.queryIdx].pt

        H, _ = cv2.findHomography(points_bgr, points_src, cv2.RANSAC)

        h, w = src.shape[:2]
        bgr = cv2.warpPerspective(bgr, H, (w, h))
        msk = cv2.warpPerspective(np.ones((h, w)), H, (w, h))

        # For areas that is outside of the background, 
        # We just copy pixels from the source.
        bgr[msk != 1] = src[msk != 1]

        src = Image.fromarray(src)
        bgr = Image.fromarray(bgr)
        
        return src, bgr
