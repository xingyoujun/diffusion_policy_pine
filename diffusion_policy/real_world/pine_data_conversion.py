from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform

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
    demo_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    # 创建空的 ReplayBuffer
    out_replay_buffer = ReplayBuffer.create_empty_numpy()

    # helper functions
    def read_lowdim_file(path):
        return np.loadtxt(str(path), dtype=np.float32)

    def read_image_file(path, output_res=None):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if output_res is not None:
            img = cv2.resize(img, output_res, interpolation=cv2.INTER_NEAREST)
        return img

    lock = multiprocessing.Lock()

    def add_episode_to_buffer(episode_data):
        with lock:
            out_replay_buffer.add_episode(episode_data)

    # 40 212 is Error
    # progress bar
    # demo_dirs = demo_dirs[175:180]
    pbar = tqdm(total=len(demo_dirs), desc="Processing demos", ncols=100)

    def process_demo(demo_path):
        # print(demo_path)
        episode_data = dict()

        # load lowdim
        if lowdim_keys:
            for key in lowdim_keys:
                if key == 'action':
                    key_data = 'joint_states_right_arm'
                else:
                    key_data = key
                file_path = demo_path / f"{key_data}.txt"
                if file_path.exists():
                    lowdim_data = read_lowdim_file(file_path)
                    episode_data[key] = lowdim_data[:,1:]

        # load images
        if image_keys:
            for key in image_keys:
                img_dir = demo_path / key
                img_paths = sorted(img_dir.glob("*"))
                imgs = []
                for p in img_paths:
                    if 'depth_colormap' in str(p):
                        continue
                    out_res = None
                    if out_resolutions:
                        if isinstance(out_resolutions, dict):
                            out_res = out_resolutions.get(key, None)
                        else:
                            out_res = out_resolutions
                    imgs.append(read_image_file(p, output_res=out_res))
                # cv2.imwrite('test.png',imgs[0])
                imgs = np.stack(imgs, axis=0)
                if len(imgs.shape) == 3:
                    imgs = np.repeat(imgs[..., np.newaxis], 3, axis=-1)
                episode_data[key] = imgs

        add_episode_to_buffer(episode_data)
        pbar.update(1)

    # 多线程处理 demo
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
        futures = [executor.submit(process_demo, d) for d in demo_dirs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    pbar.close()
    return out_replay_buffer