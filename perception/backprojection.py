import numpy as np

def get_ai2_thor_intrinsics(width: int, height: int, fov: float) -> np.ndarray:
    """
    Build camera intrinsic matrix from AI2-THOR parameters.
    AI2-THOR uses horizontal FoV.

    Returns:
        K: (3, 3) intrinsic matrix
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
    """
    fov_rad = np.deg2rad(fov)  # change degree to radian
    fx = (width / 2) / np.tan(fov_rad / 2)
    fy = fx # square pixels
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
        ], dtype=np.float32)
    
    return K

def backproject_point(px: int, py: int, depth: float, K: np.ndarray) -> np.ndarray:
    """
    Backproject a single pixel + depth value into 3D camera space.

    Args:
        px : pixel x  (column)
        py : pixel y  (row)
        depth : depth value in meters
        K : (3, 3) intrinsic matrix

    Returns:
        point_3d: (3, ) array - [X, Y, Z] in camera coordinates (meters)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (px - cx) * depth / fx
    Y = (py - cy) * depth / fy
    Z = depth
    
    return np.array([X, Y, Z], dtype=np.float32)

def backproject_detections(pixel_boxes: list, depth_map: np.ndarray, K: np.ndarray, phrases: list) -> list:
    """
    Backproject all detected objects into 3D using their center pixel + depth.

    Args:
        pixel_boxes : list of dicts with center_px, center_py
        depth_map   : (H, W) float32 depth image in meters
        K           : (3, 3) intrinsic matrix
        phrases     : list of object names
    """
    results = []
    h, w = depth_map.shape

    for phrase, pb in zip(phrases, pixel_boxes):
        px = np.clip(pb["center_px"], 0, w - 1)
        py = np.clip(pb["center_py"], 0, h - 1)

        depth = float(depth_map[py, px])

        if depth <= 0 or depth > 10.0: # invalid depth
            continue

        point_3d = backproject_point(px, py, depth, K)

        results.append({
            "object":   phrase,
            "pixel":    (px, py),
            "depth_m":  round(depth, 3),
            "X":        round(float(point_3d[0]), 3),
            "Y":        round(float(point_3d[1]), 3),
            "Z":        round(float(point_3d[2]), 3),
        })

    return results


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from perception.detector import load_detector, detect_objects, boxes_to_pixel

    # AI2-THOR camera settings (must match explore_env.py)
    WIDTH, HEIGHT, FOV = 640, 480, 90

    STEP = "step005"
    DATA_DIR = Path("data/episodes/d1_exploration")

    # Load depth
    depth_map = np.load(DATA_DIR / f"{STEP}_depth_raw.npy")

    # Load detector and run detection
    model = load_detector()
    image_path = str(DATA_DIR / f"{STEP}_rgb.png")
    image_source, boxes, logits, phrases = detect_objects(model, image_path)

    # Convert boxes to pixel coordinates
    h, w = image_source.shape[:2]
    pixel_boxes = boxes_to_pixel(boxes, w, h)

    # Build intrinsic matrix
    K = get_ai2_thor_intrinsics(WIDTH, HEIGHT, FOV)
    print(f"\nIntrinsic Matrix K:\n{K}\n")

    # Backproject
    results = backproject_detections(pixel_boxes, depth_map, K, phrases)

    # Print results
    print(f"{'Object':<25} {'Pixel':^18} {'Depth(m)':>8} {'X':>8} {'Y':>8} {'Z':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['object']:<25} ({r['pixel'][0]}, {r['pixel'][1]}) "
              f"{r['depth_m']:>8.3f} {r['X']:>8.3f} {r['Y']:>8.3f} {r['Z']:>8.3f}")