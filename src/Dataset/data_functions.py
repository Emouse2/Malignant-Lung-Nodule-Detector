import pandas as pd
import os
import pydicom
import numpy as np
import ast
import matplotlib.pyplot as plt

def heading_to_heading(labels_path, heading_in, heading_out, data):
    df = pd.read_csv(labels_path)

    if not isinstance(heading_in, list):
        heading_in = [heading_in]
    if not isinstance(data, list):
        data = [data]
    if not isinstance(heading_out, list):
        heading_out = [heading_out]

    mask = pd.Series([True] * len(df))
    for col, val in zip(heading_in, data):
        mask &= (df[col] == val)

    target_rows = df[mask]
    return target_rows[heading_out]

def get_ct_folder(images_path, labels_path, patient_id):
    target_instance_uid = heading_to_heading(
        labels_path=labels_path,
        heading_in=["patient_id"],
        heading_out=["instance_uid"],
        data=[patient_id]
    )
    instance_uid = target_instance_uid.values.tolist()[0][0]
    patient_folder = os.path.join(images_path, patient_id)

    dicom_folder = ""
    for folder in os.listdir(patient_folder):
        image_scan_folders_parent = os.path.join(patient_folder, folder)
        if not os.path.isdir(image_scan_folders_parent):
            continue
        for folder_2 in os.listdir(image_scan_folders_parent):
            folder_2_dir = os.path.join(image_scan_folders_parent, folder_2)
            if os.path.isdir(folder_2_dir) and folder_2 == instance_uid:
                dicom_folder = os.path.join(image_scan_folders_parent, folder_2)
    if dicom_folder != "":
        return dicom_folder

def load_ct_series(ct_folder):
    slices = []
    for f in os.listdir(ct_folder):
        if f.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(ct_folder, f))
            z = float(ds.ImagePositionPatient[2])
            slices.append((z, ds))
    slices.sort(key=lambda x: x[0])
    volume = np.stack([ds.pixel_array for (_, ds) in slices], axis=0)
    return slices, volume

def get_slice_index(slices, z_slice):
    z_positions = [z for (z, ds) in slices]
    diff = [abs(z - z_slice) for z in z_positions]
    slice_index = diff.index(min(diff))
    return slice_index

def plot_nodule(highlight: bool, images_path, labels_path, instance_uid, nodule_id):
    target_plot_data = heading_to_heading(
        labels_path,
        heading_in=["instance_uid", "nodule_id"],
        heading_out=["xs", "ys", "z-slice", "patient_id"],
        data=[instance_uid, nodule_id]
    )
    plot_data = target_plot_data.values.tolist()[0]
    xs = ast.literal_eval(plot_data[0])
    ys = ast.literal_eval(plot_data[1])
    z_slice = plot_data[2]
    patient_id = plot_data[3]
    ct_folder = get_ct_folder(
        images_path,
        labels_path,
        patient_id
    )

    slices, volume = load_ct_series(ct_folder)
    slice_index = get_slice_index(slices, z_slice)

    image = volume[int(slice_index)]
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    if highlight:
        print(f"Plotting slice {slice_index + 1} of {len(slices)} (Z = {z_slice} mm), nodule outlined in yellow")
        print(f"Nodule at x: {xs[0]}, Nodule at y: {ys[0]}")
        plt.plot(
            xs + [xs[0]], 
            ys + [ys[0]], 
            linewidth=1, 
            color='yellow',
            alpha=1
        )
        plt.title(f"Slice {slice_index + 1} of {len(slices)} (Z = {z_slice} mm), nodule outlined in yellow")
    else:
        print(f"Plotting slice {slice_index + 1} of {len(slices)} (Z = {z_slice} mm")
        plt.title(f"Slice {slice_index + 1} of {len(slices)} (Z = {z_slice} mm)")