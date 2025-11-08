#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys

from pathlib import Path
from typing import NamedTuple

import numpy as np

from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB

from .bbox_tool import BBoxTool


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    bbox_model: BBoxTool = None  # only used in hyfluid


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir)
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"] if "camera_angle_x" in contents else None

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            if fovx is None:
                assert "camera_angle_x" in frame, "camera_angle_x not found in frame"
                fovx = frame["camera_angle_x"]
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransformsHyFluid(path, transformsfile, white_background, extension=".png", frame_idx=100):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    voxel_scale = np.array(contents["voxel_scale"])
    voxel_scale = np.broadcast_to(voxel_scale, [3])

    voxel_matrix = np.array(contents["voxel_matrix"])
    voxel_matrix = np.stack([voxel_matrix[:, 2], voxel_matrix[:, 1], voxel_matrix[:, 0], voxel_matrix[:, 3]], axis=1)
    # points are in OpenGL coordinate system, but the camera is in colmap coordinate system
    # voxel_matrix[:3, 1:3] *= -1
    voxel_matrix_inv = np.linalg.inv(voxel_matrix)
    bbox_model = BBoxTool(voxel_matrix_inv, voxel_scale)

    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, "colmap_frames", f"colmap_{frame_idx}", frame["file_path"] + extension)
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovx = frame["camera_angle_x"]
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos, bbox_model


def readNerfSyntheticInfoHyFluid(path, white_background, eval, extension=".png", frame_idx=100):
    print("Reading Training Transforms")
    train_cam_infos, bbox_model = readCamerasFromTransformsHyFluid(
        path, "transforms_train_hyfluid.json", white_background, extension, frame_idx
    )
    print("Reading Test Transforms")
    test_cam_infos, _ = readCamerasFromTransformsHyFluid(
        path, "transforms_test_hyfluid.json", white_background, extension, frame_idx
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    # hyfluid recreate the points every time
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

    # points are in OpenGL coordinate system, but the camera is in colmap coordinate system
    # this is just OpenGL coordinate system, but it works, although the camera has to convert from OpenGL to colmap
    # x = np.random.random((num_pts, 1)) * 0.35 + 0.15  # [0.15, 0.5]
    radius = 0.02  # default value 0.18  source region 0.02
    x_mid = 0.34  # default value 0.325  source region 0.34
    y_min = -0.01  # default value -0.05  source region -0.01
    y_max = 0.02  # default value 0.7  source region 0.05
    z_mid = -0.225  # default value -0.25  source region -0.225

    y = np.random.uniform(y_min, y_max, (num_pts, 1))  # [-0.05, 0.15] [-0.05, 0.7]

    radius = np.random.random((num_pts, 1)) * radius  # * 0.03 # 0.18
    theta = np.random.random((num_pts, 1)) * 2 * np.pi
    x = radius * np.cos(theta) + x_mid
    z = radius * np.sin(theta) + z_mid

    print(f"Points init x: {x.min()}, {x.max()}")
    print(f"Points init y: {y.min()}, {y.max()}")
    print(f"Points init z: {z.min()}, {z.max()}")

    xyz = np.concatenate((x, y, z), axis=1)

    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        bbox_model=bbox_model,
    )
    return scene_info


def readCamerasFromTransformsRealCapture(path, transformsfile, white_background, extension=".png", frame_idx=0):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    def rotate_camera_around_optical_axis(C2W, theta):
        # Rotation matrix around Z-axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_roll = np.array([[cos_theta, -sin_theta, 0, 0], [sin_theta, cos_theta, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Update camera-to-world matrix
        C2W_new = np.dot(C2W, R_roll)
        return C2W_new

    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        cam_name = frame["file_path"][-1]
        # frame["file_path"]: camera00~camera04

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        #### real_capture_black hacks
        if cam_name == "0":
            degree = np.deg2rad(4)
            c2w = rotate_camera_around_optical_axis(c2w, degree)
        elif cam_name == "1":
            degree = np.deg2rad(2)
            c2w = rotate_camera_around_optical_axis(c2w, degree)
        elif cam_name == "3":
            degree = np.deg2rad(-3)
            c2w = rotate_camera_around_optical_axis(c2w, degree)
        elif cam_name == "4":
            degree = np.deg2rad(-3)
            c2w = rotate_camera_around_optical_axis(c2w, degree)

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = f"{frame['file_path']}_{frame_idx:03d}"
        image_path = os.path.join(path, frame["file_path"], f"{frame_idx:03d}" + extension)
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovx = frame["camera_angle_x"]
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos


def readNerfSyntheticInfoRealCapture(path, white_background, eval, extension=".png", frame_idx=0, all_cam=False):
    print("Reading Training Transforms")
    train_json = "transforms_aligned_train.json"
    if all_cam:
        train_json = "transforms_aligned.json"
        print("Reading All Transforms")
    train_cam_infos = readCamerasFromTransformsRealCapture(path, train_json, white_background, extension, frame_idx)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransformsRealCapture(
        path, "transforms_aligned_test.json", white_background, extension, frame_idx
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    # hyfluid recreate the points every time
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")

    x_min = -0.5
    x_max = 2.5
    y_min = -0.2
    y_max = 2.5
    z_plane = -0.43

    x = np.random.uniform(x_min, x_max, (num_pts, 1))
    y = np.random.uniform(y_min, y_max, (num_pts, 1))
    z = np.ones((num_pts, 1)) * z_plane + np.random.random((num_pts, 1)) * 0.01

    print(f"Points init x: {x.min()}, {x.max()}")
    print(f"Points init y: {y.min()}, {y.max()}")
    print(f"Points init z: {z.min()}, {z.max()}")

    xyz = np.concatenate((x, y, z), axis=1)

    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        bbox_model=None,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "HyFluid": readNerfSyntheticInfoHyFluid,
    "RealCapture": readNerfSyntheticInfoRealCapture,
}
