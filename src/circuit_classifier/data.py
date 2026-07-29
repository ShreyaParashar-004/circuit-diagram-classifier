"""
Dataset exploration and preparation utilities.

The source dataset is expected to be a folder of subfolders, one per
class, named like `<component>_r<rotation>` (e.g. `ac_src_r0`), each
containing image files for that class/rotation.
"""

import os
import shutil


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def explore_dataset(base_path):
    """Explore and analyze the dataset structure.

    Returns a stats dict (per-component totals, rotation totals, and
    overall image count) plus the list of raw class folders found.
    """
    stats = {
        "components": {},
        "rotations": {0: 0, 1: 0, 2: 0, 3: 0},
        "total_images": 0,
    }

    folders = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    print(f"Total folders: {len(folders)}\n")

    for folder in sorted(folders):
        folder_path = os.path.join(base_path, folder)
        num_images = len(
            [f for f in os.listdir(folder_path) if f.endswith(IMAGE_EXTENSIONS)]
        )

        # parse folder name, e.g. 'ac_src_r0' -> component='ac_src', rotation=0
        parts = folder.rsplit("_r", 1)
        if len(parts) == 2:
            component_name = parts[0]
            rotation = int(parts[1])

            if component_name not in stats["components"]:
                stats["components"][component_name] = {"total": 0, "rotations": {}}

            stats["components"][component_name]["total"] += num_images
            stats["components"][component_name]["rotations"][rotation] = num_images
            stats["rotations"][rotation] += num_images

        stats["total_images"] += num_images
        print(f"{folder}: {num_images} images")

    return stats, folders


def print_dataset_summary(stats):
    """Pretty-print the summary produced by explore_dataset."""
    print(f"\n{'=' * 50}")
    print(f"Total images: {stats['total_images']}")
    print(f"Unique components: {len(stats['components'])}")
    print("\nComponent distribution:")
    for comp, info in sorted(stats["components"].items()):
        print(f"  {comp}: {info['total']} images")


def prepare_data_structure(base_path, organized_path):
    """Copy the raw class folders into a clean working directory.

    Returns (organized_path, class_names).
    """
    if os.path.exists(organized_path):
        shutil.rmtree(organized_path)

    os.makedirs(organized_path)

    folders = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    for folder in folders:
        src = os.path.join(base_path, folder)
        dst = os.path.join(organized_path, folder)
        shutil.copytree(src, dst)

    class_names = folders
    print(f"Data organized into {len(class_names)} classes")
    print(f"Classes: {class_names}")

    return organized_path, class_names


def get_classes_info(base_path):
    """List class names (subfolder names) at base_path, sorted."""
    class_names = sorted(
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    )
    num_classes = len(class_names)

    print(f"Found {num_classes} classes.")
    print(f"Classes: {class_names}")

    return class_names, num_classes
