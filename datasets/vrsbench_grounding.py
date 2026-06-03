"""
Insert VRSBench objects as grounding samples into a single VLM dataset.

Dataset: https://huggingface.co/datasets/xiang709/VRSBench

For each object in each image we create ONE sample shaped as:
  - input_meta_data: {"question": <referring sentence or built phrase>}
  - variables:
      label = obj_cls
      xmin, ymin, xmax, ymax = obj_coord (normalized 0-1, rounded to 2 decimals)

The dataset's result prompt template renders these as:
  [{"label": "{{label}}", "bbox": [{{xmin}}, {{ymin}}, {{xmax}}, {{ymax}}]}]
"""

import io
import json
import os
import zipfile

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from ximilar.client import RecognitionClient
from ximilar.client.detection import DetectionClient
from ximilar.client.vlm import VLMClient


WORKSPACE_ID = os.environ.get("XIMILAR_WORKSPACE_ID", "default")
DATASET_ID = os.environ["XIMILAR_DATASET_ID"]
DETECTION_LABEL_ID = os.environ["XIMILAR_DETECTION_LABEL_ID"]

API_KEY = os.environ["XIMILAR_API_KEY"]
VAR_LABEL = os.environ["XIMILAR_VAR_LABEL"]
VAR_XMIN  = os.environ["XIMILAR_VAR_XMIN"]
VAR_YMIN  = os.environ["XIMILAR_VAR_YMIN"]
VAR_XMAX  = os.environ["XIMILAR_VAR_XMAX"]
VAR_YMAX  = os.environ["XIMILAR_VAR_YMAX"]

REPO_ID = "xiang709/VRSBench"

SPLITS = [
    {
        "name": "train",
        "annotations_zip": "Annotations_train.zip",
        "images_zip": "Images_train.zip",
        "force_test": False,
    },
]

LIMIT = 1000
TEST_EVERY = 10
MIN_BBOX_PIXELS = 16


def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_image_from_zip(zf, image_id):
    candidates = [image_id, f"Images_train/{image_id}", f"images/{image_id}"]
    names = zf.namelist()
    match = next((n for n in candidates if n in names), None)
    if match is None:
        match = next((n for n in names if n.endswith("/" + image_id) or n == image_id), None)
    if match is None:
        raise FileNotFoundError(f"Could not find '{image_id}'. First 5 zip entries: {names[:5]}")
    with zf.open(match) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def build_question(obj):
    """Return just the target description (lowercased); backend wraps it in the grounding prompt."""
    referring = (obj.get("referring_sentence") or "").strip()
    if referring:
        return referring.rstrip(".").lower()
    obj_cls = (obj.get("obj_cls") or "object").strip()
    obj_size = (obj.get("obj_size") or "").strip()
    target = f"the {obj_size} {obj_cls}".replace("  ", " ") if obj_size else f"the {obj_cls}"
    return target.lower()


def create_grounding_sample(
    client_vlm,
    uploaded_image_id,
    image_filename,
    obj_id,
    question,
    label,
    bbox_norm,
    test=False,
):
    sample = client_vlm.create_sample(
        DATASET_ID, name=f"{image_filename}_obj{obj_id}", test=test
    )
    sample.add_images([uploaded_image_id])
    sample.add_input_meta_data({"question": question})
    sample.add_value(VAR_LABEL, label)
    sample.add_value(VAR_XMIN, bbox_norm[0])
    sample.add_value(VAR_YMIN, bbox_norm[1])
    sample.add_value(VAR_XMAX, bbox_norm[2])
    sample.add_value(VAR_YMAX, bbox_norm[3])
    return sample


def main():
    client = RecognitionClient(API_KEY, workspace=WORKSPACE_ID)
    client.max_image_size = 0

    client_vlm = VLMClient(API_KEY, workspace=WORKSPACE_ID)
    client_vlm.max_image_size = 0

    client_detection = DetectionClient(API_KEY, workspace=WORKSPACE_ID)
    client_detection.max_image_size = 0

    totals = {"succeeded": 0, "skipped": 0, "failed": 0}

    for split in SPLITS:
        split_name = split["name"]
        print(f"\n=== Split: {split_name} ===")

        print("Fetching files (cached)...")
        annotations_zip_path = hf_hub_download(
            repo_id=REPO_ID, filename=split["annotations_zip"], repo_type="dataset"
        )
        images_zip_path = hf_hub_download(
            repo_id=REPO_ID, filename=split["images_zip"], repo_type="dataset"
        )

        annotations_zf = zipfile.ZipFile(annotations_zip_path)
        images_zf = zipfile.ZipFile(images_zip_path)

        ann_names = [
            n for n in annotations_zf.namelist()
            if n.endswith(".json") and not n.startswith("__MACOSX/") and not os.path.basename(n).startswith("._")
        ]
        ann_names = list(reversed(ann_names))
        if LIMIT is not None:
            ann_names = ann_names[:LIMIT]
        print(f"Processing {len(ann_names)} images")

        skipped = 0
        failed = 0
        succeeded = 0

        for idx, ann_name in enumerate(tqdm(ann_names, desc=split_name)):
            is_test = split["force_test"] or (idx % TEST_EVERY) == 0
            try:
                with annotations_zf.open(ann_name) as f:
                    entry = json.load(f)

                image_filename = entry.get("image_id") or os.path.basename(ann_name).replace(".json", "") + ".png"
                objects = entry.get("objects") or []
                if not objects:
                    skipped += 1
                    continue

                try:
                    pil_image = load_image_from_zip(images_zf, image_filename)
                except FileNotFoundError as e:
                    tqdm.write(f"[skip] {image_filename}: {e}")
                    skipped += 1
                    continue

                images, status = client.upload_images(
                    [{
                        "_img_data": pil_to_cv2(pil_image),
                        "_color_space": "BGR",
                    }]
                )
                if not images:
                    tqdm.write(f"[skip] image upload failed/exists for {image_filename}: {status}")
                    skipped += 1
                    continue
                uploaded_image_id = images[0].id

                existing_objects, _ = client_detection.get_objects_of_image(uploaded_image_id)
                skip_detection_objects = bool(existing_objects)
                if skip_detection_objects:
                    tqdm.write(
                        f"[info] {image_filename}: image already has "
                        f"{len(existing_objects)} detection object(s), skipping creation"
                    )

                image_width, image_height = pil_image.size

                created = 0
                for obj in objects:
                    obj_coord = obj.get("obj_coord")
                    obj_cls = (obj.get("obj_cls") or "object").strip()
                    obj_id = obj.get("obj_id", created)

                    if not obj_coord or len(obj_coord) != 4:
                        continue
                    x1, y1, x2, y2 = obj_coord
                    x1, y1 = max(0.0, x1), max(0.0, y1)
                    x2, y2 = min(1.0, x2), min(1.0, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    px = [
                        int(round(x1 * image_width)),
                        int(round(y1 * image_height)),
                        int(round(x2 * image_width)),
                        int(round(y2 * image_height)),
                    ]
                    if px[2] - px[0] < MIN_BBOX_PIXELS or px[3] - px[1] < MIN_BBOX_PIXELS:
                        tqdm.write(f"[skip] obj {obj_id}: bbox too small ({px})")
                        continue

                    if not skip_detection_objects:
                        det_obj, st = client_detection.create_object(
                            label_id=DETECTION_LABEL_ID, image_id=uploaded_image_id, data=px
                        )
                        if det_obj is None:
                            tqdm.write(f"[fail] create_object failed for {obj_cls}: {st}")
                            continue

                    bbox_norm = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    question = build_question(obj)

                    create_grounding_sample(
                        client_vlm,
                        uploaded_image_id,
                        image_filename,
                        obj_id,
                        question,
                        obj_cls,
                        bbox_norm,
                        test=is_test,
                    )
                    created += 1

                tqdm.write(
                    f"[ok][{split_name}] {image_filename}: {created} sample(s)"
                    f"{' (TEST)' if is_test else ''}"
                )
                succeeded += 1
            except Exception as e:
                tqdm.write(f"[fail][{split_name}] {ann_name}: {e}")
                failed += 1

        annotations_zf.close()
        images_zf.close()

        print(f"Split {split_name} done. succeeded={succeeded} skipped={skipped} failed={failed}")
        totals["succeeded"] += succeeded
        totals["skipped"] += skipped
        totals["failed"] += failed

    print(
        f"\nAll splits done. "
        f"succeeded={totals['succeeded']} skipped={totals['skipped']} failed={totals['failed']}"
    )


if __name__ == "__main__":
    main()
