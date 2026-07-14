from pathlib import Path
import argparse
import csv
import json
import re
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog


def parse_filename(path: Path):

    stem = path.stem

    if "_" not in stem:
        raise ValueError(f"Invalid filename format: {path.name}")

    house_id, room_id = stem.rsplit("_", 1)

    if not room_id.isdigit():
        raise ValueError(f"Room id is not numeric in filename: {path.name}")

    return house_id, int(room_id)


def load_npz_array(path: Path):

    data = np.load(path)

    candidate_keys = []
    for key in data.files:
        arr = data[key]
        if arr.ndim == 2:
            candidate_keys.append(key)

    if not candidate_keys:
        raise ValueError(f"No 2D array found in {path.name}. Keys: {data.files}")

    key = candidate_keys[0]
    arr = data[key]

    return key, arr


def get_bbox(mask: np.ndarray):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return {
        "min_x": int(xs.min()),
        "min_y": int(ys.min()),
        "max_x": int(xs.max()),
        "max_y": int(ys.max())
    }


def inspect_room_file(path: Path):

    house_id, room_id = parse_filename(path)
    array_key, arr = load_npz_array(path)

    if arr.shape != (256, 256):
        shape_warning = True
    else:
        shape_warning = False

    foreground = arr != 0
    nonzero_values = np.unique(arr[foreground])

    if len(nonzero_values) == 0:
        dominant_room_value = None
        foreground_pixels = 0
    else:
        values, counts = np.unique(arr[foreground], return_counts=True)
        dominant_room_value = int(values[np.argmax(counts)])
        foreground_pixels = int(foreground.sum())

    return {
        "file": path.name,
        "house_id": house_id,
        "room_id": room_id,
        "array_key": array_key,
        "shape": list(arr.shape),
        "shape_warning": shape_warning,
        "room_values": [int(v) for v in nonzero_values],
        "dominant_room_value": dominant_room_value,
        "foreground_pixels": foreground_pixels,
        "bbox": get_bbox(foreground)
    }


def make_house_preview(house_id, room_records, data_dir: Path, output_dir: Path):

    canvas = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(canvas)

    colours = [
        (230, 80, 80),
        (80, 160, 230),
        (90, 190, 120),
        (230, 170, 70),
        (170, 110, 220),
        (80, 200, 200),
        (220, 120, 180),
        (160, 160, 80),
    ]

    for i, record in enumerate(sorted(room_records, key=lambda r: r["room_id"])):
        path = data_dir / record["file"]
        _, arr = load_npz_array(path)
        mask = arr != 0

        colour = colours[i % len(colours)]
        img_arr = np.array(canvas)

        img_arr[mask] = colour
        canvas = Image.fromarray(img_arr)

        bbox = record["bbox"]
        if bbox:
            draw = ImageDraw.Draw(canvas)
            draw.text((bbox["min_x"], bbox["min_y"]), str(record["room_id"]), fill=(0, 0, 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"house_{house_id}.png")

def choose_dataset_folder():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(title="Select MagicPlan dataset folder containing .npz files")

    root.destroy()

    if not folder:
        raise SystemExit("No folder selected. Exiting.")

    return folder

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="Path to MagicPlan folder containing .npz files")
    parser.add_argument("--out", default="magicplan_profile", help="Output folder for profile results")
    parser.add_argument("--preview_count", type=int, default=20, help="Number of house previews to save")

    args = parser.parse_args()

    if args.data:
        data_dir = Path(args.data)
    else:
        data_dir = Path(choose_dataset_folder())

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    room_records = []
    errors = []

    for i, path in enumerate(files, start=1):
        try:
            room_records.append(inspect_room_file(path))
        except Exception as e:
            errors.append({
                "file": path.name,
                "error": str(e)
            })

        if i % 1000 == 0:
            print(f"Processed {i}/{len(files)} files...")

    houses = {}

    for record in room_records:
        houses.setdefault(record["house_id"], []).append(record)

    house_records = []

    for house_id, records in houses.items():
        masks = []

        for record in records:
            path = data_dir / record["file"]
            _, arr = load_npz_array(path)
            masks.append(arr != 0)

        if masks:
            stacked = np.stack(masks, axis=0)
            union_mask = stacked.any(axis=0)
            overlap_pixels = int((stacked.sum(axis=0) > 1).sum())
            union_pixels = int(union_mask.sum())
        else:
            overlap_pixels = 0
            union_pixels = 0

        room_values = sorted({
            value
            for record in records
            for value in record["room_values"]
        })

        house_records.append({
            "house_id": house_id,
            "room_count": len(records),
            "room_ids": [record["room_id"] for record in records],
            "room_values_present": room_values,
            "union_pixels": union_pixels,
            "overlap_pixels": overlap_pixels
        })

    # Save room-level CSV
    with open(output_dir / "rooms.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "house_id",
                "room_id",
                "array_key",
                "shape",
                "shape_warning",
                "room_values",
                "dominant_room_value",
                "foreground_pixels",
                "bbox"
            ]
        )
        writer.writeheader()
        writer.writerows(room_records)

    # Save house-level CSV
    with open(output_dir / "houses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "house_id",
                "room_count",
                "room_ids",
                "room_values_present",
                "union_pixels",
                "overlap_pixels"
            ]
        )
        writer.writeheader()
        writer.writerows(house_records)

    # Save errors
    with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)

    # Save summary
    room_counts = [h["room_count"] for h in house_records]

    summary = {
        "total_npz_files": len(files),
        "valid_room_files": len(room_records),
        "error_files": len(errors),
        "total_houses": len(house_records),
        "min_rooms_per_house": min(room_counts) if room_counts else None,
        "max_rooms_per_house": max(room_counts) if room_counts else None,
        "mean_rooms_per_house": float(np.mean(room_counts)) if room_counts else None,
        "unique_room_values": sorted({
            value
            for record in room_records
            for value in record["room_values"]
        }),
        "shape_warnings": sum(1 for r in room_records if r["shape_warning"]),
        "houses_with_overlap_pixels": sum(1 for h in house_records if h["overlap_pixels"] > 0)
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save previews
    preview_dir = output_dir / "previews"

    for house_id in sorted(houses.keys())[:args.preview_count]:
        make_house_preview(house_id, houses[house_id], data_dir, preview_dir)

    print("MagicPlan dataset profile complete.")
    print(f"Total .npz files: {summary['total_npz_files']}")
    print(f"Valid room files: {summary['valid_room_files']}")
    print(f"Error files: {summary['error_files']}")
    print(f"Total houses: {summary['total_houses']}")
    print(f"Mean rooms per house: {summary['mean_rooms_per_house']:.2f}")
    print(f"Unique room values: {summary['unique_room_values']}")
    print(f"Houses with overlapping room pixels: {summary['houses_with_overlap_pixels']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()