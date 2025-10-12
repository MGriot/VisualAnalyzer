import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Dict
import os
import math

# Imports for robust patch alignment
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, deltaE_cie76

# Assuming these utilities are in your project structure
from src import config
from src.utils.image_utils import load_image, save_image

# --- NEW IMPORTS for ArUco-based detection ---
from src.color_correction.patch_detector import ColorCheckerAligner, get_or_generate_reference_checker


def _sort_patches(
    patches: List[np.ndarray], patch_coords: List[Tuple[int, int, int, int]]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Sorts patches based on their y and then x coordinates."""
    if not patches:
        return [], []
    # Combine patches and coordinates for sorting
    combined = sorted(
        zip(patches, patch_coords), key=lambda item: (item[1][1], item[1][0])
    )
    # Unzip after sorting
    sorted_patches, sorted_coords = zip(*combined)
    return list(sorted_patches), list(sorted_coords)


class ColorCorrector:
    def __init__(self):
        self.model = YOLO(str(config.YOLO_MODEL_PATH))

    # --- New Step: Robust Patch Alignment ---
    def _align_color_patches_by_color(
        self, 
        detected_colors_bgr: List[np.ndarray], 
        reference_colors_bgr: List[np.ndarray], 
        debug_mode: bool = False
    ) -> Tuple[List, List, List]:
        """ 
        Aligns detected patches to reference patches using color similarity (deltaE) and the Hungarian algorithm.
        This is robust to mis-ordered, missing, or extra detected patches.
        """
        # Convert BGR to RGB then to LAB
        detected_colors_rgb = [c[::-1] for c in detected_colors_bgr]
        reference_colors_rgb = [c[::-1] for c in reference_colors_bgr]
        
        reference_lab = rgb2lab(np.array(reference_colors_rgb, dtype=np.uint8).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        detected_lab = rgb2lab(np.array(detected_colors_rgb, dtype=np.uint8).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

        # 1. Create Cost Matrix with a Rejection Threshold
        REJECTION_THRESHOLD = 50.0  # deltaE_cie76: a large but plausible color difference
        num_detected = len(detected_lab)
        num_reference = len(reference_lab)

        cost_matrix = np.zeros((num_detected, num_reference))
        for i in range(num_detected):
            for j in range(num_reference):
                cost_matrix[i, j] = deltaE_cie76(detected_lab[i], reference_lab[j])
        
        cost_matrix[cost_matrix > REJECTION_THRESHOLD] = np.inf

        # 2. Pad the Cost Matrix to Make it Square
        DUMMY_COST = REJECTION_THRESHOLD - 1.0
        size = max(num_detected, num_reference)
        padded_cost_matrix = np.full((size, size), DUMMY_COST, dtype=float)
        padded_cost_matrix[:num_detected, :num_reference] = cost_matrix

        # 3. Solve and Filter the Assignments
        row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
        
        valid_pairs = []
        if debug_mode: print("\n--- Filtering Color Patch Assignments ---")
        for r, c in zip(row_ind, col_ind):
            cost = padded_cost_matrix[r, c]
            if r < num_detected and c < num_reference and cost != np.inf:
                valid_pairs.append((r, c))  # (detected_index, reference_index)
                if debug_mode: print(f"✔️ Valid: Detected patch {r} -> Reference patch {c} (Cost: {cost:.2f})")
            else:
                if debug_mode: print(f"❌ Discarded: Assignment of row {r} to col {c} (Cost: {cost:.2f})")

        # 4. Create aligned lists from valid pairs
        valid_detected_indices = [pair[0] for pair in valid_pairs]
        valid_reference_indices = [pair[1] for pair in valid_pairs]

        aligned_detected_colors = [detected_colors_bgr[i] for i in valid_detected_indices]
        aligned_reference_colors = [reference_colors_bgr[i] for i in valid_reference_indices]

        return aligned_detected_colors, aligned_reference_colors, valid_pairs

    # --- Step 1: Detection Helpers (Unchanged) ---
    def _find_checker_bounds(
        self, image: np.ndarray
    ) -> Tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        inverted_gray = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(inverted_gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        return cv2.boundingRect(approx)

    # --- NEW: Tier 1 ArUco Detection Method ---
    def _detect_patches_with_aruco(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        """
        Attempts to detect and extract color checker patches by first aligning the
        checker using ArUco markers.
        """
        try:
            if debug_mode:
                print("[DEBUG] Attempting ArUco-based alignment and patch detection.")
            
            # Get the ideal digital reference checker with ArUco markers
            reference_checker_img = get_or_generate_reference_checker()
            
            # Use the aligner from patch_detector.py
            aligner = ColorCheckerAligner(image)
            aligned_image = aligner.align_with_aruco(reference_checker_img)

            if aligned_image is None:
                if debug_mode:
                    print("[DEBUG] ArUco alignment failed to produce an image.")
                return {"patches": [], "patch_coords": []}

            if debug_mode and output_dir:
                save_image(os.path.join(output_dir, "debug_aruco_aligned_checker.png"), aligned_image)

            # Detect patches on the new, aligned image
            # Increase margin_ratio to crop out the ArUco markers before patch detection.
            patch_infos = aligner.detect_patches(adaptive=True, margin_ratio=0.15)
            
            if not patch_infos:
                if debug_mode:
                    print("[DEBUG] ArUco alignment succeeded, but patch detection on the aligned image failed.")
                return {"patches": [], "patch_coords": []}

            patches = []
            patch_coords = []
            for p in patch_infos:
                x, y, w, h = p.bounding_box
                # Extract patch from the ALIGNED image
                patch = aligned_image[y:y+h, x:x+w]
                patches.append(patch)
                # Coords are relative to the aligned image, which is fine for sorting and debug
                patch_coords.append((x, y, x+w, y+h))
            
            # The patches are already ordered by index from detect_patches, so sorting is not strictly
            # necessary but we do it anyway to conform to the expected output format of other methods.
            return {"patches": patches, "patch_coords": patch_coords}

        except Exception as e:
            if debug_mode:
                import traceback
                print(f"[DEBUG] ArUco detection process failed with an error: {e}")
                traceback.print_exc()
            return {"patches": [], "patch_coords": []}


    def detect_color_checker_patches(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        """
        Detects color checker patches using a tiered approach, starting with the most robust.
        1. ArUco marker alignment.
        2. Robust OpenCV grid-line detection.
        3. YOLO model if other methods fail.
        4. Simple grid-based OpenCV detection as a final fallback.
        """
        # --- TIER 1: Try ArUco Method First ---
        if debug_mode:
            print("[DEBUG] Tier 1: Attempting ArUco-based detection.")
        aruco_result = self._detect_patches_with_aruco(
            image, output_dir=output_dir, debug_mode=debug_mode
        )
        # Check for a good number of patches (e.g., at least 20)
        if len(aruco_result.get("patches", [])) >= 20:
            if debug_mode:
                print(f"[DEBUG] ArUco detection succeeded with {len(aruco_result['patches'])} patches.")
            aruco_result["detection_method"] = "aruco"
            return aruco_result

        # --- TIER 2: Fallback to Robust OpenCV ---
        if debug_mode:
            print("[DEBUG] ArUco failed. Tier 2: Attempting robust OpenCV grid-line detection.")
        robust_result = self._detect_patches_opencv_robust(
            image, output_dir=output_dir, debug_mode=debug_mode
        )
        if len(robust_result.get("patches", [])) == 24:
            if debug_mode:
                print(
                    f"[DEBUG] Robust OpenCV succeeded with {len(robust_result['patches'])} patches."
                )
            robust_result["detection_method"] = "robust_opencv"
            return robust_result

        # --- TIER 3: Fallback to YOLO ---
        if debug_mode:
            print("[DEBUG] Robust OpenCV failed. Tier 3: Falling back to YOLO detection.")
        patches, patch_coords = [], []
        try:
            results = self.model(image, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    patch = image[y1:y2, x1:x2]
                    if patch.size > 0:
                        patches.append(patch)
                        patch_coords.append((x1, y1, x2, y2))
        except Exception as e:
            if debug_mode:
                print(f"[DEBUG] YOLO detection failed with error: {e}")
            patches = []

        if len(patches) >= 18:
            if debug_mode:
                print(f"[DEBUG] YOLO detection succeeded with {len(patches)} patches.")
            return {
                "patches": patches,
                "patch_coords": patch_coords,
                "detection_method": "yolo",
            }

        # --- TIER 4: Final Fallback to Simple Grid ---
        if debug_mode:
            print(
                "[DEBUG] YOLO detection also failed. Tier 4: Falling back to simple grid method."
            )
        simple_result = self._detect_patches_opencv(
            image, output_dir=output_dir, debug_mode=debug_mode
        )
        simple_result["detection_method"] = "simple_grid"
        return simple_result

    def _detect_patches_opencv_robust(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        bounds = self._find_checker_bounds(image)
        if not bounds:
            return {"patches": [], "patch_coords": []}
        x, y, w, h = bounds
        cropped_checker = image[y : y + h, x : x + w]
        gray = cv2.cvtColor(cropped_checker, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40, minLineLength=h // 6, maxLineGap=25
        )
        if lines is None:
            if debug_mode:
                print("[DEBUG] Robust method found no lines with Hough Transform.")
            return {"patches": [], "patch_coords": []}
        vertical_x = []
        horizontal_y = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:
                horizontal_y.append(y1)
                horizontal_y.append(y2)
            elif abs(abs(angle) - 90) < 15:
                vertical_x.append(x1)
                vertical_x.append(x2)
        def cluster_and_average(coords, threshold=10):
            if not coords:
                return []
            coords.sort()
            clusters = []
            current_cluster = [coords[0]]
            for coord in coords[1:]:
                if coord - current_cluster[-1] < threshold:
                    current_cluster.append(coord)
                else:
                    clusters.append(int(np.mean(current_cluster)))
                    current_cluster = [coord]
            clusters.append(int(np.mean(current_cluster)))
            return clusters
        vertical_x.extend([0, w])
        horizontal_y.extend([0, h])
        x_coords = sorted(
            cluster_and_average(vertical_x, threshold=w / 20)
        )
        y_coords = sorted(
            cluster_and_average(horizontal_y, threshold=h / 15)
        )
        if len(x_coords) != 7 or len(y_coords) != 5:
            if debug_mode:
                print(
                    f"[DEBUG] Robust grid validation failed. Expected 7 vertical and 5 horizontal lines, but found {len(x_coords)} and {len(y_coords)}."
                )
            return {"patches": [], "patch_coords": []}
        if debug_mode and output_dir:
            debug_img = cropped_checker.copy()
            for x_c in x_coords:
                cv2.line(debug_img, (x_c, 0), (x_c, h), (0, 255, 255), 2)
            for y_c in y_coords:
                cv2.line(debug_img, (0, y_c), (w, y_c), (255, 0, 255), 2)
            path = os.path.join(output_dir, "debug_robust_grid_lines.png")
            save_image(path, debug_img)
        patches, patch_coords = [], []
        inset = int(min(w / 6, h / 4) * 0.1)
        for r_idx in range(len(y_coords) - 1):
            for c_idx in range(len(x_coords) - 1):
                y1, y2 = y_coords[r_idx] + inset, y_coords[r_idx + 1] - inset
                x1, x2 = x_coords[c_idx] + inset, x_coords[c_idx + 1] - inset
                if x1 >= x2 or y1 >= y2:
                    continue
                patch = cropped_checker[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)
                    patch_coords.append((x + x1, y + y1, x + x2, y + y2))
        if debug_mode:
            print(
                f"[DEBUG] Successfully extracted {len(patches)} patches with robust method."
            )
        return {"patches": patches, "patch_coords": patch_coords}

    def _detect_patches_opencv(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        bounds = self._find_checker_bounds(image)
        if not bounds:
            return {"patches": [], "patch_coords": []}
        x, y, w, h = bounds
        cropped_color_checker = image[y : y + h, x : x + w]
        rows, cols = 4, 6
        patch_height = h // rows
        patch_width = w // cols
        patches, patch_coords = [], []
        inset = int(min(patch_height, patch_width) * 0.15)
        for r_idx in range(rows):
            for c_idx in range(cols):
                y1 = r_idx * patch_height + inset
                y2 = (r_idx + 1) * patch_height - inset
                x1 = c_idx * patch_width + inset
                x2 = (c_idx + 1) * patch_width - inset
                patch = cropped_color_checker[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)
                    patch_coords.append((x + x1, y + y1, x + x2, y + y2))
        return {"patches": patches, "patch_coords": patch_coords}

    def calculate_average_colors(
        self, patches: List[np.ndarray], color_space="bgr"
    ) -> List[np.ndarray]:
        average_colors = []
        for patch in patches:
            img = patch[:, :, :3]
            if color_space == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            avg_color = np.mean(img, axis=(0, 1))
            average_colors.append(avg_color.astype(np.uint8))
        return average_colors

    def calculate_correction_model(
        self, source, target, method: str = "linear"
    ) -> Dict:
        if method == "histogram":
            return {"luts": self._calculate_histogram_luts(source, target)}
        if method == "linear":
            return {"matrix": self._calculate_linear_matrix(source, target)}
        elif method == "polynomial":
            return {"matrix": self._calculate_root_polynomial_matrix(source, target)}
        elif method == "hsv":
            return {"luts": self._calculate_hsv_luts(source, target)}
        else:
            raise ValueError(f"Unknown color correction method: {method}")

    def _calculate_histogram_luts(
        self, source_roi: np.ndarray, target_roi: np.ndarray
    ) -> List[np.ndarray]:
        luts = []
        for i in range(3):
            src_hist = cv2.calcHist([source_roi], [i], None, [256], [0, 256])
            tgt_hist = cv2.calcHist([target_roi], [i], None, [256], [0, 256])
            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()
            src_cdf_norm = (src_cdf * tgt_cdf.max()) / src_cdf.max()
            lut = np.zeros(256, dtype=np.uint8)
            j = 0
            for val in range(256):
                while j < 255 and src_cdf_norm[val] > tgt_cdf[j]:
                    j += 1
                lut[val] = j
            luts.append(lut)
        return luts

    def _calculate_linear_matrix(self, source_colors, target_colors):
        source_matrix = np.array(source_colors, dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_root_polynomial_matrix(self, source_colors, target_colors):
        source_matrix = np.array([np.sqrt(c) for c in source_colors], dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_hsv_luts(self, source_colors_hsv, target_colors_hsv):
        luts = []
        source_channels = cv2.split(
            np.array(source_colors_hsv, dtype=np.uint8).reshape(-1, 1, 3)
        )
        target_channels = cv2.split(
            np.array(target_colors_hsv, dtype=np.uint8).reshape(-1, 1, 3)
        )
        for i in range(3):
            src_hist, _ = np.histogram(
                source_channels[i].flatten(), bins=256, range=[0, 256]
            )
            tgt_hist, _ = np.histogram(
                target_channels[i].flatten(), bins=256, range=[0, 256]
            )
            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()
            src_cdf_norm = src_cdf * (tgt_cdf.max() / src_cdf.max())
            lut = np.zeros(256, dtype=np.uint8)
            j = 0
            for val in range(256):
                while j < 255 and src_cdf_norm[val] > tgt_cdf[j]:
                    j += 1
                lut[val] = j
            luts.append(lut)
        return luts

    def apply_correction_model(
        self, image: np.ndarray, model: Dict, method: str = "linear"
    ) -> np.ndarray:
        if method == "histogram":
            return self._apply_histogram_luts(image, model["luts"])
        elif method == "linear":
            return self._apply_linear_matrix(image, model["matrix"])
        elif method == "polynomial":
            return self._apply_root_polynomial_matrix(image, model["matrix"])
        elif method == "hsv":
            return self._apply_hsv_luts(image, model["luts"])
        else:
            raise ValueError(f"Unknown color correction method: {method}")

    def _apply_histogram_luts(
        self, image: np.ndarray, luts: List[np.ndarray]
    ) -> np.ndarray:
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        b, g, r = cv2.split(bgr_image)
        b_corr = cv2.LUT(b, luts[0])
        g_corr = cv2.LUT(g, luts[1])
        r_corr = cv2.LUT(r, luts[2])
        corrected_bgr = cv2.merge([b_corr, g_corr, r_corr])
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_linear_matrix(self, image, matrix):
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(pixels, matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_root_polynomial_matrix(self, image, matrix):
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(np.sqrt(pixels), matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_hsv_luts(self, image, luts):
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h_corr = cv2.LUT(h, luts[0])
        s_corr = cv2.LUT(s, luts[1])
        v_corr = cv2.LUT(v, luts[2])
        corrected_hsv = cv2.merge([h_corr, s_corr, v_corr])
        corrected_bgr = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def calculate_correction_from_images(
        self,
        source_image_path: str,
        reference_image_path: str,
        output_dir: str | None = None,
        debug_mode: bool = False,
        method: str = "linear",
    ) -> Dict:
        all_debug_paths = {}
        source_image, _ = load_image(source_image_path)
        reference_image, _ = load_image(reference_image_path)
        source_detection_method, ref_detection_method = None, None

        if method == "histogram":
            src_bounds = self._find_checker_bounds(source_image)
            ref_bounds = self._find_checker_bounds(reference_image)
            if not src_bounds or not ref_bounds:
                raise ValueError(
                    "Could not find color checker bounds for histogram alignment."
                )
            src_roi = source_image[
                src_bounds[1] : src_bounds[1] + src_bounds[3],
                src_bounds[0] : src_bounds[0] + src_bounds[2],
            ]
            ref_roi = reference_image[
                ref_bounds[1] : ref_bounds[1] + ref_bounds[3],
                ref_bounds[0] : ref_bounds[0] + ref_bounds[2],
            ]
            correction_model = self.calculate_correction_model(src_roi, ref_roi, method)
        else:
            source_detection = self.detect_color_checker_patches(
                source_image, output_dir, debug_mode
            )
            ref_detection = self.detect_color_checker_patches(
                reference_image, output_dir, debug_mode
            )
            source_detection_method = source_detection.get("detection_method")
            ref_detection_method = ref_detection.get("detection_method")

            sorted_src_patches, sorted_src_coords = _sort_patches(
                source_detection["patches"], source_detection["patch_coords"]
            )
            sorted_ref_patches, _ = _sort_patches(
                ref_detection["patches"], ref_detection["patch_coords"]
            )

            if debug_mode and output_dir:
                src_patch_debug_img = source_image.copy()
                for i, (x1, y1, x2, y2) in enumerate(sorted_src_coords):
                    cv2.rectangle(
                        src_patch_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        src_patch_debug_img,
                        str(i),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                src_patch_debug_path = os.path.join(
                    output_dir, f"01_debug_source_sorted_patches_{method}.png"
                )
                save_image(src_patch_debug_path, src_patch_debug_img)
                all_debug_paths[f"source_sorted_patches"] = src_patch_debug_path

            color_space = "hsv" if method == "hsv" else "bgr"
            source_colors = self.calculate_average_colors(
                sorted_src_patches, color_space
            )
            ref_colors = self.calculate_average_colors(
                sorted_ref_patches, color_space
            )

            aligned_source_colors, aligned_ref_colors, valid_pairs = self._align_color_patches_by_color(
                source_colors, ref_colors, debug_mode
            )

            if debug_mode and output_dir:
                src_match_debug_img = source_image.copy()
                for detected_idx, ref_idx in valid_pairs:
                    x1, y1, x2, y2 = sorted_src_coords[detected_idx]
                    cv2.rectangle(src_match_debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        src_match_debug_img,
                        f"->{ref_idx}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
                match_debug_path = os.path.join(output_dir, f"03_debug_source_final_matches_{method}.png")
                save_image(match_debug_path, src_match_debug_img)
                all_debug_paths["source_final_matches"] = match_debug_path

            if len(aligned_source_colors) < 6:
                raise ValueError(f"Not enough valid patch pairs found ({len(aligned_source_colors)}). Need at least 6 for a stable correction.")

            correction_model = self.calculate_correction_model(
                aligned_source_colors, aligned_ref_colors, method
            )

        return {
            "correction_model": correction_model,
            "debug_paths": all_debug_paths,
            "source_detection_method": source_detection_method,
            "reference_detection_method": ref_detection_method,
        }