from typing import Sequence, Tuple, Dict, Optional, Union
import os
import glob
import pathlib
import numpy as np
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2
import matplotlib.cm as cm
colormap = np.array(cm.get_cmap("inferno").colors)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform

from scipy.spatial.transform import Rotation as R

def pine_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore] = None,
        out_resolutions: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        lowdim_keys: Optional[Sequence[str]] = None,
        image_keys: Optional[Sequence[str]] = None,
        lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
        image_compressor: Optional[numcodecs.abc.Codec] = None,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        n_encoding_threads: int = multiprocessing.cpu_count(),
        max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
        verify_read: bool = True
) -> ReplayBuffer:
    """
    Load pine demo dataset into ReplayBuffer (numpy backend compatible)
    """
    n_encoding_threads = 8
    dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
    task_name = dataset_path.stem
    dataset_path = dataset_path.parent
    data_path = dataset_path / 'train'
    demo_dirs = glob.glob(os.path.join(data_path,f'{task_name}*.npz'))

    # 创建空的 ReplayBuffer
    out_replay_buffer = ReplayBuffer.create_empty_numpy()

    lock = multiprocessing.Lock()

    def add_episode_to_buffer(episode_data):
        with lock:
            out_replay_buffer.add_episode(episode_data)


    pbar = tqdm(total=len(demo_dirs), desc="Processing demos", ncols=100)

    def process_demo(demo_path):
        # print(demo_path)
        episode_data = dict()

        data_path = demo_path
        depth_path = demo_path.replace('train','depths')
        wrist_path = demo_path.replace('train','wrist')

        data = np.load(data_path)
        depth = np.load(depth_path)
        wrist = np.load(wrist_path)
        data = np.load(data_path)
        main_camera = data['obs_rgb']
        depth_camera = depth['depths']
        wrist_camera = wrist['obs_rgb']

        action = data['actions']
        # load lowdim
        episode_data['action'] = action
        episode_data['robot_eef_pose'] = action[:,:3]

        assert out_resolutions

        # load images
        if image_keys:
            for key in image_keys:
                if isinstance(out_resolutions, dict):
                    out_res = out_resolutions.get(key, None)
                else:
                    out_res = out_resolutions
                if key == 'main_camera':
                    resized_frames = [cv2.resize(frame, out_res, interpolation=cv2.INTER_NEAREST) for frame in main_camera]
                    imgs = np.stack(resized_frames, axis=0)  # (N, newH, newW, C)
                    cv2.imwrite('test_r.png',imgs[0])
                elif key == 'depth_camera':
                    resized_frames = [cv2.resize(frame, out_res, interpolation=cv2.INTER_NEAREST) for frame in depth_camera]
                    imgs = np.stack(resized_frames, axis=0)  # (N, newH, newW, C)
                    imgs = imgs / 10
                    imgs = imgs.clip(min=0, max=1)
                    imgs = (imgs * 255).astype(np.uint8)
                    imgs = imgs[...,np.newaxis].repeat(3,axis=-1)
                    # imgs = (colormap[imgs.astype(np.uint8)]*255).astype(np.uint8)
                    # cv2.imwrite('test_d.png',imgs[0])
                elif key == 'wrist_camera':
                    resized_frames = [cv2.resize(frame, out_res, interpolation=cv2.INTER_NEAREST) for frame in wrist_camera]
                    imgs = np.stack(resized_frames, axis=0)  # (N, newH, newW, C)
                    # cv2.imwrite('test_w.png',imgs[0])
                episode_data[key] = imgs

        min_len = min(arr.shape[0] for arr in episode_data.values())
        for key in episode_data:
            episode_data[key] = episode_data[key][:min_len]

        add_episode_to_buffer(episode_data)
        pbar.update(1)

    # 多线程处理 demo
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
        futures = [executor.submit(process_demo, d) for d in demo_dirs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    pbar.close()
    return out_replay_buffer

def rollout_pine_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore] = None,
        out_resolutions: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        lowdim_keys: Optional[Sequence[str]] = None,
        image_keys: Optional[Sequence[str]] = None,
        lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
        image_compressor: Optional[numcodecs.abc.Codec] = None,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        n_encoding_threads: int = multiprocessing.cpu_count(),
        max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
        verify_read: bool = True
) -> ReplayBuffer:
    """
    Load pine demo dataset into ReplayBuffer (numpy backend compatible)
    """
    n_encoding_threads = 1
    dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
    task_name = dataset_path.stem
    dataset_path = dataset_path.parent
    demo_dirs = glob.glob(os.path.join(dataset_path,f'rollout_{task_name}*.npz'))

    # 创建空的 ReplayBuffer
    out_replay_buffer = ReplayBuffer.create_empty_numpy()

    lock = multiprocessing.Lock()

    def add_episode_to_buffer(episode_data):
        with lock:
            out_replay_buffer.add_episode(episode_data)


    pbar = tqdm(total=len(demo_dirs), desc="Processing demos", ncols=100)

    def process_demo(demo_path):
        # print(demo_path)
        episode_data = dict()

        data_path = demo_path
        demo_name = demo_path.split('rollout_')[-1]
        depth_path = os.path.join('/home/chuanrui001/code/diffusion_policy/data/pine_real_npz/depths', demo_name)
        wrist_path = os.path.join('/home/chuanrui001/code/diffusion_policy/data/pine_real_npz/wrist', demo_name)

        data = np.load(data_path)
        wrist = np.load(wrist_path)
        main_camera = data['frames']
        wrist_camera = wrist['obs_rgb']

        if 'ivideo' in demo_path:
            depth = np.load(depth_path)
            depth_camera = depth['depths']
        elif 'i4d' in demo_path:
            depth = main_camera[:,3:6]
            main_camera = main_camera[:,:3]
        else:
            raise NotImplementedError

        action = data['actions']
        # load lowdim
        episode_data['action'] = action
        episode_data['robot_eef_pose'] = action[:,:3]

        assert out_resolutions

        # load images
        if image_keys:
            for key in image_keys:
                if isinstance(out_resolutions, dict):
                    out_res = out_resolutions.get(key, None)
                else:
                    out_res = out_resolutions
                if key == 'main_camera':
                    imgs = (main_camera * 255).astype(np.uint8)
                    imgs = imgs.transpose(0, 2, 3, 1)  # shape: [91, 256, 256, 3]
                    # cv2.imwrite('test_r.png',imgs[0])
                elif key == 'depth_camera':
                    if 'ivideo' in demo_path:
                        resized_frames = [cv2.resize(frame, out_res, interpolation=cv2.INTER_NEAREST) for frame in depth_camera]
                        imgs = np.stack(resized_frames, axis=0)  # (N, newH, newW, C)
                        imgs = imgs / 10
                        imgs = imgs.clip(min=0, max=1)
                        imgs = (imgs * 255).astype(np.uint8)
                        imgs = imgs[...,np.newaxis].repeat(3,axis=-1)
                    elif 'i4d' in demo_path:
                        imgs = (depth * 255).astype(np.uint8)
                        imgs = imgs.transpose(0, 2, 3, 1)  # shape: [91, 256, 256, 3]
                    else:
                        raise NotImplementedError
                    # imgs = (colormap[imgs.astype(np.uint8)]*255).astype(np.uint8)
                    # cv2.imwrite('test_dd.png',imgs[0])
                elif key == 'wrist_camera':
                    resized_frames = [cv2.resize(frame, out_res, interpolation=cv2.INTER_NEAREST) for frame in wrist_camera]
                    imgs = np.stack(resized_frames, axis=0)  # (N, newH, newW, C)
                    # cv2.imwrite('test_w.png',imgs[0])
                episode_data[key] = imgs

        min_len = min(arr.shape[0] for arr in episode_data.values())
        for key in episode_data:
            episode_data[key] = episode_data[key][:min_len]

        add_episode_to_buffer(episode_data)
        pbar.update(1)

    # 多线程处理 demo
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
        futures = [executor.submit(process_demo, d) for d in demo_dirs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    pbar.close()
    return out_replay_buffer