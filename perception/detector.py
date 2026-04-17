import torch
import numpy as np
import cv2
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict, annotate


# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = "external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# Detectable Home Objects (sep: . )
HOME_OBJECTS = "cup . mug . bottle . bowl . plate . pan . pot . chair . drawer . table"

# Threshold
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15

def load_detector(config_path = CONFIG_PATH, 
                  weights_path = WEIGHTS_PATH, 
                  device = DEVICE):
    """Load GroundingDINO Model"""
    model = load_model(config_path, weights_path, device)
    print(f"[INFO] Detector loaded on {DEVICE}")
    return model

def detect_objects(model, image_path: str):
    """
    Detect Object in the image
    Returns:
        boxes   : (N, 4)   tensor - cx, cy, w, h (normalized 0-1)
        logits  : (N, )    tensor - confidence score
        phrases : list     str - detected objects
    """
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=HOME_OBJECTS,
        box_threshold=BOX_THRESHOLD,
        text_threshold =TEXT_THRESHOLD,
        device=DEVICE
    )
    return image_source, boxes, logits, phrases

def save_annotated(image_source, boxes, logits, phrases, save_path: str):
    """Draw detected result to the image and save"""
    annotated = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )
    cv2.imwrite(save_path, annotated)
    print(f"[INFO] Saved annotated image: {save_path}")

def boxes_to_pixel(boxes, img_w: int, img_h: int):
    """
    normalized box (cx, cy, w, h) -> pixel box (x1, y1, x2, y2)
    To use as central pixel in backprojection
    """
    result = []
    for box in boxes:
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        px = int(cx * img_w)  # center pixel x
        py = int(cy * img_h)  # center pixel y
        result.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_px": px, "center_py": py,
        })
    return result

if __name__ == "__main__":
    # Test
    TEST_IMAGE = "data/episodes/d1_exploration/step005_rgb.png"
    SAVE_DIR   = Path("data/episodes/d1_exploration")

    model = load_detector()
    image_source, boxes, logits, phrases = detect_objects(model, TEST_IMAGE)
    print(f"Detected {len(phrases)} objects: {phrases}")

    print(f"=== Test Result ===")
    print(f"{'Object':<20} {'confidence':>10}")
    print("-" * 32)
    for phrase, logit in zip(phrases, logits):
        print(f"{phrase:<20} {logit.item():>10.3f}")

    save_annotated(
        image_source, boxes, logits, phrases,
        str(SAVE_DIR / "step005_detected.png")
    )

    # Change to pixel coordinate
    h, w = image_source.shape[:2]
    pixel_boxes = boxes_to_pixel(boxes, w, h)
    print(f"\n=== Pixel Coordinate ===")
    for phrase, pb in zip(phrases, pixel_boxes):
        print(f"{phrase:<20} center=({pb['center_px']}, {pb['center_py']})")
        