import os
import sys
import tempfile
import csv
import SimpleITK as sitk
from moosez import moose
import torch
import numpy as np
# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

BASE_DIR = r""  #enter your dir 

# DICOM CT input folder (put your .dcm series here)
dicom_folder = os.path.join(BASE_DIR, "input_dicom_ct")

# MooseZ output folder (organ + ribs segmentations)
output_segmented_folder = os.path.join(BASE_DIR, "output_segmented")

# Tumour segmentation NIfTI (from Slicer Segmentations export)
tumor_seg_path = os.path.join(BASE_DIR, "input_tumour_nii", "Segmentation.nii")
# change filename above if different

# Where to save tumour centroid FCSV
centroid_fcsv_out = os.path.join(
    BASE_DIR, "output_tumourcentroid_fcsv", "tumor_centroid.fcsv"
)

# Where to save candidate + valid entry FCSVs
candidate_fcsv_dir = os.path.join(BASE_DIR, "output_candidate_valid_points_fcsv")

models = ["clin_ct_organs", "clin_ct_ribs"]
accelerator = "cuda"
LIVER_LABEL = 8  # clin_ct_organs → liver = 8 (official MOOSE label)
ALLOWED_ORGAN_LABELS = {LIVER_LABEL}
def ensure_dirs():
    os.makedirs(output_segmented_folder, exist_ok=True)
    os.makedirs(os.path.dirname(centroid_fcsv_out), exist_ok=True)
    os.makedirs(candidate_fcsv_dir, exist_ok=True)


def pick_largest_series(folder):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        return None, None
    best_id = None
    best_count = -1
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(folder, sid)
        if len(files) > best_count:
            best_count = len(files)
            best_id = sid
    return best_id, best_count

def run_moose_segmentation():
    # 0) checks
    if not os.path.exists(dicom_folder) or not os.path.isdir(dicom_folder):
        print("DICOM folder not found or not a directory:", dicom_folder)
        sys.exit(1)

    sid, count = pick_largest_series(dicom_folder)
    if sid is None:
        print("No DICOM series found in folder. Ensure the folder contains .dcm files.")
        sys.exit(1)
    print(f"Selected DICOM series id={sid} with {count} files.")

    # 1) read series
    try:
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(dicom_folder, sid)
        reader.SetFileNames(file_names)
        print("Reading DICOM series...")
        image = reader.Execute()
    except PermissionError as e:
        print("Permission error reading files. Try running as Administrator or change folder permissions.")
        print("Exception:", e)
        sys.exit(1)
    except Exception as e:
        print("Failed to read DICOM series with SimpleITK. Exception:", repr(e))
        sys.exit(1)

    # 2) write temporary NIfTI
    try:
        tmp_nifti = os.path.join(tempfile.gettempdir(), "moose_input_tmp.nii.gz")
        sitk.WriteImage(image, tmp_nifti)
        print("Wrote temporary NIfTI:", tmp_nifti)
    except Exception as e:
        print("Failed to write temporary NIfTI. Exception:", repr(e))
        sys.exit(1)

    # 3) run MOOSE
    try:
        print("Calling moose(...) with signature: (input, models, output_dir)")
        moose(tmp_nifti, models, output_segmented_folder, accelerator)
        print("MOOSE completed. Output should be in:", output_segmented_folder)
    except Exception as e:
        print("MOOSE run failed. Exception:")
        print(repr(e))
        sys.exit(1)
    finally:
        # cleanup temp file
        try:
            if os.path.exists(tmp_nifti):
                os.remove(tmp_nifti)
        except Exception:
            pass

    return image  # CT image

def compute_tumor_volume(seg_path, label_value=1):
    # Load segmentation
    seg_img = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg_img)  # shape: (z, y, x)

    # Extract spacing (mm)
    sx, sy, sz = seg_img.GetSpacing()  # spacing in x, y, z directions
    voxel_volume_mm3 = sx * sy * sz    # volume of 1 voxel in mm³

    # Count voxels belonging to tumour
    num_voxels = np.sum(seg_arr == label_value)

    # Total tumour volume
    total_volume_mm3 = num_voxels * voxel_volume_mm3
    total_volume_cm3 = total_volume_mm3 / 1000.0  # convert mm³ → cm³

    return num_voxels, voxel_volume_mm3, total_volume_mm3, total_volume_cm3

def load_tumor_centroid_from_seg(seg_path):

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Tumour segmentation file not found: {seg_path}")

    seg_img = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    tumor_voxels = np.argwhere(seg_arr == 1)  # assumes tumour label = 1
    if len(tumor_voxels) == 0:
        raise ValueError("Tumour segmentation is empty (no label = 1 voxels).")

    centroid_voxel = tumor_voxels.mean(axis=0)  # (z,y,x)

    centroid_mm = seg_img.TransformIndexToPhysicalPoint(
        (int(centroid_voxel[2]),
         int(centroid_voxel[1]),
         int(centroid_voxel[0]))
    )
    return centroid_mm

def save_centroid_as_fcsv(centroid_mm, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# name = TumorCentroid\n")
        f.write("# coordinateSystem = 0\n")
        f.write("# columns = label,x,y,z,ow,ox,oy,oz,vis,sel,lock,desc,associatedNodeID\n")

        label = "TumorCentroid"
        x, y, z = centroid_mm
        ow, ox, oy, oz = 0, 0, 0, 1
        vis, sel, lock = 1, 1, 0
        desc = "Tumour centroid from segmentation"
        associatedNodeID = ""

        line = (
            f"{label},{x},{y},{z},"
            f"{ow},{ox},{oy},{oz},"
            f"{vis},{sel},{lock},"
            f"\"{desc}\",{associatedNodeID}\n"
        )
        f.write(line)
        print("Saved tumour centroid FCSV:", out_path)
        seg_path = "D:\data\saves\maxio_path_planning\input_tumour_nii\Segmentation.nii"
        num_vox, vox_vol, vol_mm3, vol_cm3 = compute_tumor_volume(seg_path)

        print("Tumour voxel count:", num_vox)
        print("Voxel volume (mm³):", vox_vol)
        print("Tumour volume (mm³):", vol_mm3)
        print("Tumour volume (cm³):", vol_cm3)

def load_ct_from_dicom(dicom_folder_path):
    sid, count = pick_largest_series(dicom_folder_path)
    if sid is None:
        raise RuntimeError(f"No DICOM series found in folder: {dicom_folder_path}")

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(dicom_folder_path, sid)
    reader.SetFileNames(file_names)
    ct_img = reader.Execute()
    return ct_img


def create_body_mask_from_ct(ct_img, threshold=-400):
    ct_arr = sitk.GetArrayFromImage(ct_img)
    body_arr = (ct_arr > threshold).astype(np.uint8)
    body_img = sitk.GetImageFromArray(body_arr)
    body_img.CopyInformation(ct_img)
    return body_img

def sample_directions(num_theta=10, num_phi=18):
    """
    Sample directions on the unit sphere.
    Returns a list of 3D unit vectors (numpy arrays).
    """
    dirs = []
    for i in range(num_theta):
        theta = np.pi * (i + 0.5) / num_theta   # polar angle (0..pi)
        for j in range(num_phi):
            phi = 2 * np.pi * j / num_phi       # azimuth (0..2pi)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v = np.array([x, y, z], dtype=float)
            v /= np.linalg.norm(v)
            dirs.append(v)
    return dirs


def _is_inside_index(idx, size):
    return all(0 <= idx[d] < size[d] for d in range(3))


def find_entry_point_along_direction(tumor_mm, direction, body_img,
                                     step_mm=1.0, max_dist_mm=200.0):
    """
    Starting from tumor_mm, move along +direction until leaving the body.
    Returns the last 'inside body' point in mm (approximate skin entry),
    or None if no valid entry found within max_dist_mm.
    """
    size = body_img.GetSize()
    body_arr = sitk.GetArrayFromImage(body_img)

    tumor = np.array(tumor_mm, dtype=float)
    last_inside_point = None
    last_inside = False

    for d in np.arange(0.0, max_dist_mm, step_mm):
        p = tumor + d * direction  # physical coordinates (mm)
        idx = body_img.TransformPhysicalPointToIndex(tuple(p))  # (x, y, z) indices

        if not _is_inside_index(idx, size):
            # left the volume; if we were inside before, last_inside_point is entry
            return last_inside_point

        inside = body_arr[idx[2], idx[1], idx[0]] > 0  # [z,y,x]

        if inside:
            last_inside_point = p
            last_inside = True
        else:
            # transitioned from inside -> outside (body -> air)
            if last_inside and last_inside_point is not None:
                return last_inside_point
            else:
                # never really inside in this direction
                return None

    # did not leave body within max_dist
    return None

def compute_needle_angle_deg(entry_mm, target_mm, ct_img):
    """
    Angle (in degrees) between needle direction (entry->target)
    and the axial/gantry axis (slice normal).
    0°  = parallel to gantry axis
    90° = in axial plane
    """
    entry = np.array(entry_mm, dtype=float)
    target = np.array(target_mm, dtype=float)
    v = target - entry
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return None
    v /= norm_v

    # Direction matrix 3x3
    direction = np.array(ct_img.GetDirection()).reshape(3, 3)
    axial_normal = direction[:, 2]  # k-axis (slice direction)
    norm_n = np.linalg.norm(axial_normal)
    if norm_n == 0:
        axial_normal = np.array([0., 0., 1.])
        norm_n = 1.0
    axial_normal /= norm_n

    # angle between v and axial_normal
    cos_theta = np.clip(abs(np.dot(v, axial_normal)), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def generate_candidate_paths_with_angle_and_length(
        dicom_folder_path,
        tumor_point_mm,
        max_angle_deg=60.0,
        min_length_mm=30.0,
        max_length_mm=150.0,
        num_theta=10,
        num_phi=18,
        step_mm=1.0,
        max_dist_mm=200.0,
        min_entry_separation_mm=5.0):
    """
    1) Load CT from DICOM
    2) Create body mask
    3) Sample directions around tumor
    4) For each direction, find skin entry
    5) Apply (in this order):
         a) ANGLE check (<= max_angle_deg w.r.t. axial/gantry axis)
         b) LENGTH check (min_length_mm <= L <= max_length_mm)
         c) Remove near-duplicate entry points
    """
    # 1) Load CT
    ct_img = load_ct_from_dicom(dicom_folder_path)

    # 2) Body mask
    body_img = create_body_mask_from_ct(ct_img)

    # 3) Tumor point
    tumor = np.array(tumor_point_mm, dtype=float)

    # 4) Directions
    directions = sample_directions(num_theta=num_theta, num_phi=num_phi)

    candidate_paths = []
    entry_points = []  # to avoid clustering of very similar entry points

    for direction in directions:
        entry = find_entry_point_along_direction(
            tumor_point_mm, direction, body_img,
            step_mm=step_mm,
            max_dist_mm=max_dist_mm
        )
        if entry is None:
            continue

        entry = np.array(entry, dtype=float)

        # ---- a) ANGLE CHECK ----
        angle_deg = compute_needle_angle_deg(entry, tumor, ct_img)
        if angle_deg is None or angle_deg > max_angle_deg:
            continue

        # ---- b) LENGTH CHECK ----
        length = np.linalg.norm(tumor - entry)
        if length < min_length_mm or length > max_length_mm:
            continue

        # ---- c) DIVERSITY CHECK (avoid near duplicates) ----
        too_close = False
        for e_prev in entry_points:
            if np.linalg.norm(entry - e_prev) < min_entry_separation_mm:
                too_close = True
                break
        if too_close:
            continue

        entry_points.append(entry)
        candidate_paths.append({
            "entry": tuple(entry),
            "target": tuple(tumor),
            "length": float(length),
            "angle_deg": float(angle_deg),
        })

    return candidate_paths

def save_candidate_entries_to_fcsv(candidate_paths, out_path, list_name="CandidateEntries"):
    """
    Save all candidate entry points to an FCSV file that 3D Slicer can open
    as a Markups Fiducial list.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write(f"# name = {list_name}\n")
        f.write("# coordinateSystem = 0\n")
        f.write("# columns = label,x,y,z,ow,ox,oy,oz,vis,sel,lock,desc,associatedNodeID\n")

        for i, p in enumerate(candidate_paths):
            x, y, z = p["entry"]
            label = f"Entry_{i}"
            ow, ox, oy, oz = 0, 0, 0, 1
            vis, sel, lock = 1, 1, 0
            desc = f"len={p['length']:.1f}mm,angle={p['angle_deg']:.1f}"
            associatedNodeID = ""

            line = (
                f"{label},{x},{y},{z},"
                f"{ow},{ox},{oy},{oz},"
                f"{vis},{sel},{lock},"
                f"\"{desc}\",{associatedNodeID}\n"
            )
            f.write(line)

    print(f"Saved {len(candidate_paths)} candidate entry points to FCSV:", out_path)

def find_segmentation_file(root_folder, keyword):
    """
    Search recursively in root_folder for the first .nii/.nii.gz file
    whose name contains the given keyword (e.g. 'clin_ct_organs').
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fn in filenames:
            low = fn.lower()
            if keyword.lower() in low and (low.endswith(".nii") or low.endswith(".nii.gz")):
                path = os.path.join(dirpath, fn)
                print(f"Found segmentation for '{keyword}': {path}")
                return path
    raise FileNotFoundError(f"No NIfTI file found in {root_folder} containing '{keyword}'")


def load_organs_and_ribs_segmentations(output_folder):
    """
    Load MooseZ organ and ribs segmentations produced in output_folder.
    Returns:
      organs_img, organs_arr, ribs_img, ribs_arr
    """
    organs_path = find_segmentation_file(output_folder, "clin_ct_organs")
    ribs_path   = find_segmentation_file(output_folder, "clin_ct_ribs")

    organs_img = sitk.ReadImage(organs_path)
    ribs_img   = sitk.ReadImage(ribs_path)

    organs_arr = sitk.GetArrayFromImage(organs_img)  # [z,y,x]
    ribs_arr   = sitk.GetArrayFromImage(ribs_img)

    if organs_img.GetSize() != ribs_img.GetSize():
        print("WARNING: organs and ribs volumes have different sizes!")

    return organs_img, organs_arr, ribs_img, ribs_arr


def path_has_collision(entry_mm, target_mm,
                       organs_img, organs_arr,
                       ribs_img, ribs_arr,
                       allowed_organ_labels={8},
                       num_samples=2000):
    """
    Returns True if the straight path entry->target intersects:
    - ANY organ NOT in allowed_organ_labels
    - ANY ribs

    Liver (label=8) is explicitly allowed.
    """

    entry = np.array(entry_mm, dtype=float)
    target = np.array(target_mm, dtype=float)
    size = organs_img.GetSize()

    for t in np.linspace(0.0, 1.0, num_samples):
        p = entry + t * (target - entry)

        idx_org = organs_img.TransformPhysicalPointToIndex(tuple(p))
        idx_rib = ribs_img.TransformPhysicalPointToIndex(tuple(p))

        # Outside volume → unsafe
        if not (_is_inside_index(idx_org, size) and _is_inside_index(idx_rib, size)):
            return True

        organ_label = int(organs_arr[idx_org[2], idx_org[1], idx_org[0]])
        rib_label   = int(ribs_arr[idx_rib[2], idx_rib[1], idx_rib[0]])

        # ❌ ribs always block
        if rib_label > 0:
            return True

        # ❌ organs block EXCEPT liver
        if organ_label > 0 and organ_label not in allowed_organ_labels:
            return True

    return False

def filter_paths_by_collision(candidate_paths, output_folder):
    """
    Loads organ + ribs segmentations from output_folder and
    keeps only paths that do NOT touch any organ or ribs.
    """
    organs_img, organs_arr, ribs_img, ribs_arr = load_organs_and_ribs_segmentations(output_folder)

    valid = []
    invalid = []

    for p in candidate_paths:
        entry = p["entry"]
        target = p["target"]

        has_collision = path_has_collision(
            entry, target,
            organs_img, organs_arr,
            ribs_img, ribs_arr,
            allowed_organ_labels=ALLOWED_ORGAN_LABELS
        )

        if has_collision:
            invalid.append(p)
        else:
            valid.append(p)

    print(f"\nCollision check: {len(valid)} valid paths, {len(invalid)} colliding paths")
    return valid, invalid

def run_candidate_generation(tumor_point_mm):
    """
    Uses tumor_point_mm and dicom_folder to generate candidate
    needle paths, then filters them by collision with organs + ribs.
    """
    print("\nGenerating candidate needle paths...")
    candidate_paths = generate_candidate_paths_with_angle_and_length(
        dicom_folder_path=dicom_folder,
        tumor_point_mm=tumor_point_mm,
        max_angle_deg=60.0,
        min_length_mm=30.0,
        max_length_mm=150.0,
    )

    print("\n=== CANDIDATE NEEDLE PATHS (BEFORE COLLISION CHECK) ===")
    print("Number of candidate paths:", len(candidate_paths))
    for i, p in enumerate(candidate_paths):
        print(
            f"Path {i}: "
            f"length={p['length']:.1f} mm, "
            f"angle={p['angle_deg']:.1f}°, "
            f"entry={p['entry']}"
        )

    # Collision filtering
    valid_paths, invalid_paths = filter_paths_by_collision(candidate_paths, output_segmented_folder)

    print("\n=== COLLISION-FREE PATHS ===")
    for i, p in enumerate(valid_paths):
        print(
            f"Valid Path {i}: "
            f"length={p['length']:.1f} mm, "
            f"angle={p['angle_deg']:.1f}°, "
            f"entry={p['entry']}"
        )

    print("\n=== COLLIDING PATHS (DEBUG) ===")
    for i, p in enumerate(invalid_paths):
        print(
            f"Colliding Path {i}: "
            f"length={p['length']:.1f} mm, "
            f"angle={p['angle_deg']:.1f}°, "
            f"entry={p['entry']}"
        )

    # Save entries so you can see them in Slicer
    out_fcsv_all = os.path.join(candidate_fcsv_dir, "candidate_entries.fcsv")
    save_candidate_entries_to_fcsv(candidate_paths, out_fcsv_all, list_name="AllCandidates")

    # Save only VALID entries
    out_fcsv_valid = os.path.join(candidate_fcsv_dir, "valid_candidate_entries.fcsv")
    save_candidate_entries_to_fcsv(valid_paths, out_fcsv_valid, list_name="ValidEntries")
   # Input and output files
    input_fcsv = r""    #enter your input dir and output dir 
    output_fcsv = r""

    def extract_length_from_desc(desc):
        """
        desc example: 'len=51.0mm,angle=9.0'
        returns length as float (e.g. 51.0)
        """
        desc = desc.strip().strip('"')
        parts = desc.split(',')
        length_part = None
        for p in parts:
            p = p.strip()
            if p.startswith("len="):
                length_part = p
                break
        if length_part is None:
            return float('inf')  # if something wrong, push to end
        # len=51.0mm -> 51.0
        val_str = length_part.replace("len=", "").replace("mm", "")
        return float(val_str)

    # Read the file
    header_lines = []
    data_rows = []

    with open(input_fcsv, "r", newline="") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
            else:
                # Use csv.reader to handle quoted desc with commas
                reader = csv.reader([line])
                row = next(reader)
                if len(row) == 0:
                    continue
                data_rows.append(row)

    print(f"Total valid entries read: {len(data_rows)}")

    # Each row format (13 cols):
    # label,x,y,z,ow,ox,oy,oz,vis,sel,lock,desc,associatedNodeID

    # Compute length for each row
    rows_with_length = []
    for row in data_rows:
        desc = row[11]  # desc column
        length = extract_length_from_desc(desc)
        rows_with_length.append((length, row))

    # Sort by length (ascending = best first)
    rows_with_length.sort(key=lambda t: t[0])

    # Take best 5
    top_k = 5
    best_rows = [row for length, row in rows_with_length[:top_k]]

    print("Selected top 5 paths (by minimum length):")
    for length, row in rows_with_length[:top_k]:
        print(row[0], "-> length:", length, "mm")

    # Write to new FCSV
    os.makedirs(os.path.dirname(output_fcsv), exist_ok=True)

    with open(output_fcsv, "w", newline="") as f:
        # Write header lines
        for hl in header_lines:
            f.write(hl + "\n")

        # Optionally: change the name line to "RankedFivePaths"
        # If there's a "# name = ..." line, override it:
        # (simpler: just append another name line)
        # f.write("# name = RankedFivePaths\n")

        writer = csv.writer(f)
        for i, row in enumerate(best_rows):
            # Optionally rename label to Rank_0, Rank_1, ...
            row[0] = f"Rank_{i}"
            writer.writerow(row)
    print("✅ Saved ranked top 5 paths to:", output_fcsv)
if __name__ == "__main__":
    ensure_dirs()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    # 1) Run MooseZ segmentation (organs + ribs)
    run_moose_segmentation()

    # 2) Load tumour centroid from segmentation
    tumor_point_mm = load_tumor_centroid_from_seg(tumor_seg_path)
    print("Tumour centroid (mm):", tumor_point_mm)

    # 3) Save tumour centroid as FCSV for visualization
    save_centroid_as_fcsv(tumor_point_mm, centroid_fcsv_out)

    # 4) Candidate paths + collision checking
    run_candidate_generation(tumor_point_mm)
