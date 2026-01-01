"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography estimation demo. Note that scene has to be planar or just rotate the camera for the estimation to work properly.
"""

import cv2
import numpy as np
import torch

from time import time, sleep
import argparse, sys, tqdm
import threading

from modules.xfeat import XFeat
from modules.hotspot_loader import parse_svg_hotspots, create_hotspot_mask, blend_hotspot_overlay, warp_hotspot_mask, point_in_hotspot
from modules.hand_tracker import HandTracker, DwellDetector
from modules.audio_player import AudioPlayer
import os

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
    parser.add_argument('--max_kpts', type=int, default=3_000, help='Maximum number of keypoints.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat', help='Local feature detection method to use.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    parser.add_argument('--img', type=str, default=None, help='Path to an image to use as reference.')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of frames to skip between processing (0 = process every frame).')
    parser.add_argument('--infer_scale', type=float, default=1.0, help='Scale factor for inference resolution (e.g. 0.5 for half resolution).')
    parser.add_argument('--max_draw_matches', type=int, default=200, help='Maximum number of matches to draw (reduces CPU load).')
    parser.add_argument('--hotspot_dir', type=str, default='assets/hotspot', help='Directory containing SVG hotspot files.')
    parser.add_argument('--show_hotspots', action='store_true', default=True, help='Show hotspot overlay on reference frame.')
    parser.add_argument('--enable_hand_tracking', action='store_true', default=False, help='Enable hand tracking for gesture audio.')
    parser.add_argument('--sounds_dir', type=str, default='assets/sounds', help='Directory containing audio files.')
    parser.add_argument('--spine_position', type=float, default=0.5, help='Book spine position as fraction of width (0.0-1.0). Set to 0 or 1 for single page mode.')
    parser.add_argument('--single_page', action='store_true', default=False, help='Single page mode (cover/back cover), disables dual homography.')
    return parser.parse_args()


class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
            self.frame = frame
            sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()

    def get_last_frame(self):
        return self.frame

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        # Send tensor to the same device as the model (MPS/CUDA/CPU)
        tensor = torch.tensor(x).permute(2,0,1).float()[None].to(self.mtd.dev)
        return self.mtd.detectAndCompute(tensor)[0]


class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts):
    if method == "ORB":
        return Method(descriptor=cv2.ORB_create(max_kpts, fastThreshold=10), matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    elif method == "SIFT":
        return Method(descriptor=cv2.SIFT_create(max_kpts, contrastThreshold=-1, edgeThreshold=1000), matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True))
    elif method == "XFeat":
        return Method(descriptor=CVWrapper(XFeat(top_k = max_kpts)), matcher=XFeat())
    else:
        raise RuntimeError("Invalid Method.")


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.cam)
        self.width = args.width
        self.height = args.height
        self.ref_frame = None
        self.ref_precomp = [[],[]]
        self.corners = [[50, 50], [640-50, 50], [640-50, 480-50], [50, 480-50]]
        self.current_frame = None
        self.H = None
        self.setup_camera()

        #Init frame grabber thread
        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 #avg FPS over this number of frames

        # Frame skipping
        self.skip_frames = args.skip_frames
        self.frame_count = 0

        # Inference scaling
        self.infer_scale = args.infer_scale
        self.infer_width = int(self.width * self.infer_scale)
        self.infer_height = int(self.height * self.infer_scale)

        # Cached match results for frame skipping
        self.cached_kp1 = []
        self.cached_kp2 = []
        self.cached_good_matches = []

        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = init_method(args.method, max_kpts=args.max_kpts)
        
        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 3

        self.window_name = "Real-time matching - Press 's' to set the reference frame."

        if args.img is not None:
            self.ref_frame = cv2.imread(args.img)
            if self.ref_frame is None:
                print(f"Error: Could not load image {args.img}")
                exit()
            self.ref_frame = cv2.resize(self.ref_frame, (self.width, self.height))
            # Use scaled reference for feature extraction if inference scaling is enabled
            if self.infer_scale != 1.0 and args.method == 'XFeat':
                ref_for_inference = cv2.resize(self.ref_frame, (self.infer_width, self.infer_height))
                self.ref_precomp = self.method.descriptor.detectAndCompute(ref_for_inference, None)
            else:
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)
            print(f"Loaded reference image: {args.img}")

        # Load hotspots
        self.hotspot_mask = None
        self.hotspot_mask_display = None
        self.hotspots = {}
        if args.show_hotspots and os.path.isdir(args.hotspot_dir):
            self.hotspots = parse_svg_hotspots(args.hotspot_dir, self.width, self.height)
            if self.hotspots:
                self.hotspot_mask = create_hotspot_mask(self.hotspots, self.width, self.height)
                self.hotspot_mask_display = self.hotspot_mask.copy()
                print(f"Loaded {len(self.hotspots)} hotspots from {args.hotspot_dir}")
            else:
                print(f"No hotspots found in {args.hotspot_dir}")

        # Hand tracking and audio
        self.hand_tracker = None
        self.dwell_detector = None
        self.audio_player = None
        self.finger_tip_ref = None  # Finger tip in reference coordinates
        
        if args.enable_hand_tracking:
            try:
                self.hand_tracker = HandTracker()
                self.dwell_detector = DwellDetector(dwell_time=0.3)
                self.audio_player = AudioPlayer(sounds_dir=args.sounds_dir)
                print("Hand tracking enabled")
            except Exception as e:
                print(f"Warning: Could not initialize hand tracking: {e}")

        # Dual Homography for folded books
        self.spine_position = args.spine_position
        self.single_page_mode = args.single_page or args.spine_position <= 0.0 or args.spine_position >= 1.0
        self.spine_x = int(self.width * self.spine_position)  # Spine position in pixels
        self.H_left = None  # Left page homography
        self.H_right = None  # Right page homography
        self.min_partition_matches = 15  # Minimum matches per partition
        
        # Homography caching for occlusion robustness
        self.H_cache = None  # Last valid homography
        self.H_left_cache = None
        self.H_right_cache = None
        self.cache_frames = 0  # Number of frames since last valid homography
        self.max_cache_frames = 90  # Use cached homography for up to N frames 跟踪目标丢失时，使用缓存的有效homography
        
        if not self.single_page_mode:
            print(f"Dual homography enabled (spine at x={self.spine_x})")
        else:
            print("Single page mode (single homography)")


        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        #Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        
        # Apply hotspot overlay to reference frame
        ref_display = self.ref_frame.copy()
        if self.hotspot_mask is not None:
            ref_display = blend_hotspot_overlay(ref_display, self.hotspot_mask)
        
        # Draw page boundary quads on reference frame
        if not self.single_page_mode:
            # Left page outline
            cv2.rectangle(ref_display, (0, 0), (self.spine_x, self.height - 1), self.line_color, 4)
            # Right page outline
            cv2.rectangle(ref_display, (self.spine_x, 0), (self.width - 1, self.height - 1), self.line_color, 4)
        else:
            # Single page outline
            cv2.rectangle(ref_display, (0, 0), (self.width - 1, self.height - 1), self.line_color, 4)
        
        # Apply warped hotspot overlay to current frame if homography is valid
        current_display = self.current_frame.copy()
        if self.hotspot_mask is not None:
            # Dual homography mode: warp left and right halves separately
            if not self.single_page_mode and (self.H_left is not None or self.H_right is not None):
                # Split mask into left and right
                left_mask = self.hotspot_mask[:, :self.spine_x].copy()
                right_mask = self.hotspot_mask[:, self.spine_x:].copy()
                
                warped_combined = np.zeros_like(self.hotspot_mask)
                
                # Warp left half
                if self.H_left is not None:
                    warped_left = cv2.warpPerspective(left_mask, self.H_left, (self.width, self.height))
                    warped_combined = cv2.add(warped_combined, warped_left)
                
                # Warp right half (need to adjust for offset)
                if self.H_right is not None:
                    # Create full-size mask with right half in correct position
                    right_full = np.zeros_like(self.hotspot_mask)
                    right_full[:, self.spine_x:] = right_mask
                    warped_right = cv2.warpPerspective(right_full, self.H_right, (self.width, self.height))
                    warped_combined = cv2.add(warped_combined, warped_right)
                
                if warped_combined.any():
                    current_display = blend_hotspot_overlay(current_display, warped_combined)
            elif self.H is not None:
                # Single homography fallback
                warped_mask = warp_hotspot_mask(self.hotspot_mask, self.H, (self.width, self.height))
                if warped_mask is not None:
                    current_display = blend_hotspot_overlay(current_display, warped_mask)
        
        top_frame = np.hstack((ref_display, current_display))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame

        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        return top_frame_canvas

    def process(self):
        # Create a blank canvas for the top frame
        top_frame_canvas = self.create_top_frame()

        # Match features and draw matches on the bottom frame
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # Draw warped corners - dual quads in dual homography mode
        if not self.single_page_mode and (self.H_left is not None or self.H_right is not None):
            # Define left page corners (from reference frame)
            left_corners = [(0, 0), (self.spine_x, 0), (self.spine_x, self.height), (0, self.height)]
            right_corners = [(self.spine_x, 0), (self.width, 0), (self.width, self.height), (self.spine_x, self.height)]
            
            if self.H_left is not None:
                warped_left = self.warp_points(left_corners, self.H_left, self.width)
                self.draw_quad(top_frame_canvas, warped_left)
            
            if self.H_right is not None:
                warped_right = self.warp_points(right_corners, self.H_right, self.width)
                self.draw_quad(top_frame_canvas, warped_right)
        elif self.H is not None and len(self.corners) > 1:
            # Single homography mode - draw original corners
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        # Stack top and bottom frames vertically on the final canvas
        canvas = np.vstack((top_frame_canvas, bottom_frame))

        # Hand tracking and gesture audio
        if self.hand_tracker is not None:
            finger_tip_cam = self.hand_tracker.get_index_finger_tip(self.current_frame)
            
            if finger_tip_cam is not None:
                # Draw finger indicator on camera frame (right side of top frame)
                cam_x = finger_tip_cam[0] + self.width
                cam_y = finger_tip_cam[1]
                cv2.circle(canvas, (cam_x, cam_y), 10, (0, 0, 255), -1)
                
                # Transform to reference space if homography is valid
                # Choose correct homography based on dual mode
                H_to_use = None
                if not self.single_page_mode and (self.H_left is not None or self.H_right is not None):
                    # In dual mode, try both homographies and pick the one where
                    # the transformed point lands in the correct half of reference frame
                    best_match = None
                    
                    # Try left homography - result should be in left half (x < spine_x)
                    if self.H_left is not None:
                        try:
                            H_inv = np.linalg.inv(self.H_left)
                            test_ref = self.hand_tracker.transform_to_reference(finger_tip_cam, H_inv)
                            if test_ref is not None:
                                if 0 <= test_ref[0] < self.spine_x and 0 <= test_ref[1] < self.height:
                                    best_match = (self.H_left, test_ref)
                        except np.linalg.LinAlgError:
                            pass
                    
                    # Try right homography - result should be in right half (x >= spine_x)
                    if self.H_right is not None and best_match is None:
                        try:
                            H_inv = np.linalg.inv(self.H_right)
                            test_ref = self.hand_tracker.transform_to_reference(finger_tip_cam, H_inv)
                            if test_ref is not None:
                                if self.spine_x <= test_ref[0] < self.width and 0 <= test_ref[1] < self.height:
                                    best_match = (self.H_right, test_ref)
                        except np.linalg.LinAlgError:
                            pass
                    
                    if best_match is not None:
                        H_to_use = best_match[0]
                elif self.H is not None:
                    H_to_use = self.H
                
                if H_to_use is not None:
                    try:
                        H_inv = np.linalg.inv(H_to_use)
                        self.finger_tip_ref = self.hand_tracker.transform_to_reference(
                            finger_tip_cam, H_inv
                        )
                        
                        if self.finger_tip_ref is not None:
                            # Draw on reference frame (left side)
                            cv2.circle(canvas, self.finger_tip_ref, 10, (255, 0, 0), -1)
                            
                            # Check hotspot collision
                            hotspot_name = point_in_hotspot(self.finger_tip_ref, self.hotspots)
                            
                            # Dwell detection
                            if self.dwell_detector.update(self.finger_tip_ref, hotspot_name, time()):
                                if self.audio_player and hotspot_name:
                                    self.audio_player.play(hotspot_name)
                                    print(f"Playing: {hotspot_name}")
                    except np.linalg.LinAlgError:
                        pass

        cv2.imshow(self.window_name, canvas)

    def match_and_draw(self, ref_frame, current_frame):
        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Frame skipping logic: only run inference on selected frames
        self.frame_count += 1
        should_process = (self.skip_frames == 0) or (self.frame_count % (self.skip_frames + 1) == 0)

        if should_process:
            # Apply inference scaling if enabled
            if self.infer_scale != 1.0 and self.args.method == 'XFeat':
                infer_frame = cv2.resize(current_frame, (self.infer_width, self.infer_height))
            else:
                infer_frame = current_frame

            # Detect and compute features
            if self.args.method in ['SIFT', 'ORB']:
                kp1, des1 = self.ref_precomp
                kp2, des2 = self.method.descriptor.detectAndCompute(infer_frame, None)
            else:
                current = self.method.descriptor.detectAndCompute(infer_frame)
                kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
                kpts2, descs2 = current['keypoints'], current['descriptors']
                idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.60)
                points1 = kpts1[idx0].cpu().numpy()
                points2 = kpts2[idx1].cpu().numpy()

                # Scale keypoints back to display resolution if inference scaling is used
                if self.infer_scale != 1.0:
                    scale_factor = 1.0 / self.infer_scale
                    points1 = points1 * scale_factor
                    points2 = points2 * scale_factor


            if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
                # Match descriptors
                matches = self.method.matcher.match(des1, des2)

                if len(matches) > 10:
                    points1 = np.zeros((len(matches), 2), dtype=np.float32)
                    points2 = np.zeros((len(matches), 2), dtype=np.float32)

                    for i, match in enumerate(matches):
                        points1[i, :] = kp1[match.queryIdx].pt
                        points2[i, :] = kp2[match.trainIdx].pt

            if len(points1) > 10 and len(points2) > 10:
                # Dual Homography mode: partition points by spine position
                if not self.single_page_mode:
                    # Partition points into left and right based on spine_x (reference frame coordinates)
                    left_mask = points1[:, 0] < self.spine_x
                    right_mask = ~left_mask
                    
                    pts1_left, pts2_left = points1[left_mask], points2[left_mask]
                    pts1_right, pts2_right = points1[right_mask], points2[right_mask]
                    
                    # Compute left page homography
                    if len(pts1_left) >= self.min_partition_matches:
                        H_left, inliers_left = cv2.findHomography(pts1_left, pts2_left, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=500, confidence=0.99)
                        if inliers_left is not None and inliers_left.sum() >= self.min_inliers // 2:
                            self.H_left = H_left
                        else:
                            self.H_left = None
                    else:
                        self.H_left = None
                    
                    # Compute right page homography
                    if len(pts1_right) >= self.min_partition_matches:
                        H_right, inliers_right = cv2.findHomography(pts1_right, pts2_right, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=500, confidence=0.99)
                        if inliers_right is not None and inliers_right.sum() >= self.min_inliers // 2:
                            self.H_right = H_right
                        else:
                            self.H_right = None
                    else:
                        self.H_right = None
                    
                    # Fallback: if both partitions fail, use single homography
                    if self.H_left is None and self.H_right is None:
                        self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
                        inliers = inliers.flatten() > 0 if inliers is not None else np.array([])
                        if inliers.sum() < self.min_inliers:
                            self.H = None
                    else:
                        # Use left or right as the main H for compatibility
                        self.H = self.H_left if self.H_left is not None else self.H_right
                    
                    # Combine inliers for visualization (simplified)
                    inliers = np.ones(len(points1), dtype=bool)
                else:
                    # Single page mode: standard single homography
                    self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
                    inliers = inliers.flatten() > 0 if inliers is not None else np.array([])
                    if inliers.sum() < self.min_inliers:
                        self.H = None
                    self.H_left = None
                    self.H_right = None
                
                # Cache valid homographies for occlusion robustness
                if self.H is not None or self.H_left is not None or self.H_right is not None:
                    self.H_cache = self.H
                    self.H_left_cache = self.H_left
                    self.H_right_cache = self.H_right
                    self.cache_frames = 0
            else:
                # Not enough matches - try to use cached homography
                self.cache_frames += 1
                if self.cache_frames <= self.max_cache_frames and self.H_cache is not None:
                    self.H = self.H_cache
                    self.H_left = self.H_left_cache
                    self.H_right = self.H_right_cache
                else:
                    self.H = None
                    self.H_left = None
                    self.H_right = None
                inliers = np.array([])
                good_matches = []
                kp1 = []
                kp2 = []

            if len(inliers) > 0:
                if self.args.method in ["SIFT", "ORB"]:
                    good_matches = [m for i,m in enumerate(matches) if inliers[i]]
                else:
                    kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                    kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                    good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Cache keypoints and matches for frame skipping
            self.cached_kp1 = kp1
            self.cached_kp2 = kp2
            self.cached_good_matches = good_matches

            # # Limit matches for drawing to improve performance
            # draw_kp1 = kp1[:self.args.max_draw_matches]
            # draw_kp2 = kp2[:self.args.max_draw_matches]
            # draw_matches = good_matches[:self.args.max_draw_matches]

            # # Draw matches
            # matched_frame = cv2.drawMatches(ref_frame, draw_kp1, current_frame, draw_kp2, draw_matches, None, matchColor=(0, 200, 0), flags=2)
            
            # Skip drawing matches for performance testing
            matched_frame = np.hstack([ref_frame, current_frame])
        else:
            # Use cached results when skipping frames
            if self.cached_kp1 and self.cached_kp2 and self.cached_good_matches:
                # # Limit matches for drawing to improve performance
                # draw_kp1 = self.cached_kp1[:self.args.max_draw_matches]
                # draw_kp2 = self.cached_kp2[:self.args.max_draw_matches]
                # draw_matches = self.cached_good_matches[:self.args.max_draw_matches]
                # matched_frame = cv2.drawMatches(ref_frame, draw_kp1, current_frame, draw_kp2, 
                #                                 draw_matches, None, matchColor=(0, 200, 0), flags=2)
                good_matches = self.cached_good_matches
                
                # Skip drawing matches for performance testing
                matched_frame = np.hstack([ref_frame, current_frame])

            else:
                matched_frame = np.hstack([ref_frame, current_frame])


        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        skip_info = f" (skip:{self.skip_frames})" if self.skip_frames > 0 else ""
        scale_info = f" (scale:{self.infer_scale:.1f})" if self.infer_scale != 1.0 else ""
        self.putText(canvas=matched_frame, text="%s Matches: %d%s%s"%(self.args.method, len(good_matches), skip_info, scale_info), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        return matched_frame


    def main_loop(self):
        self.current_frame = self.frame_grabber.get_last_frame()
        if self.ref_frame is None:
            self.ref_frame = self.current_frame.copy()
            # Apply inference scaling for reference frame feature extraction
            if self.infer_scale != 1.0 and self.args.method == 'XFeat':
                ref_for_inference = cv2.resize(self.ref_frame, (self.infer_width, self.infer_height))
                self.ref_precomp = self.method.descriptor.detectAndCompute(ref_for_inference, None)
            else:
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

        while True:
            if self.current_frame is None:
                break

            t0 = time()
            self.process()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()  # Update reference frame
                # Apply inference scaling for reference frame feature extraction
                if self.infer_scale != 1.0 and self.args.method == 'XFeat':
                    ref_for_inference = cv2.resize(self.ref_frame, (self.infer_width, self.infer_height))
                    self.ref_precomp = self.method.descriptor.detectAndCompute(ref_for_inference, None)
                else:
                    self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

            self.current_frame = self.frame_grabber.get_last_frame()

            #Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        self.cap.release()
        if self.hand_tracker:
            self.hand_tracker.cleanup()
        if self.audio_player:
            self.audio_player.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = MatchingDemo(args = argparser())
    demo.main_loop()
