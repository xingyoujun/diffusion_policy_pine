from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import imageio
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()


def pine_data_to_replay_buffer(
        dataset_path: str, 
        out_store: Optional[zarr.ABSStore]=None, 
        out_resolutions: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
        lowdim_keys: Optional[Sequence[str]]=None,
        image_keys: Optional[Sequence[str]]=None,
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        image_compressor: Optional[numcodecs.abc.Codec]=None,
        n_decoding_threads: int=multiprocessing.cpu_count(),
        n_encoding_threads: int=multiprocessing.cpu_count(),
        max_inflight_tasks: int=multiprocessing.cpu_count()*5,
        verify_read: bool=True
        ) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    import pdb
    pdb.set_trace()
    # ====== BEGIN CHANGED PART ======
    input = pathlib.Path(os.path.expanduser(dataset_path))
    demo_dirs = sorted([d for d in input.iterdir() if d.is_dir() and d.name.startswith("demo_")])
    
    # load first demo to get shape
    first_demo = demo_dirs[0]
    if image_keys is None:
        image_keys = ['left_camera', 'right_camera', 'depth_camera']
    n_cameras = len(image_keys)

    # calculate total steps & episode_ends
    episode_ends = []
    total_steps = 0
    for demo in demo_dirs:
        first_cam_dir = demo / image_keys[0]
        n_steps = len(list(first_cam_dir.glob("*.*")))
        total_steps += n_steps
        episode_ends.append(total_steps)

    # prepare lowdim_keys
    if lowdim_keys is None:
        lowdim_keys = ['eef_pose_left', 'eef_pose_right', 'joint_left', 'joint_right', 'reward']

    # create empty ReplayBuffer
    out_replay_buffer = ReplayBuffer.create_from_store(out_store, mode='w')

    # write lowdim datasets
    for key in lowdim_keys:
        example_file = first_demo / f"{key}.txt" if key != 'reward' else first_demo / "task_reward.txt"
        example_data = np.loadtxt(example_file)
        shape = (total_steps, example_data.shape[1] if example_data.ndim>1 else 1)
        out_replay_buffer.data.require_dataset(
            name=key, shape=shape, dtype=example_data.dtype, compressor=lowdim_compressor
        )

    # write image datasets
    for cam in image_keys:
        example_img = imageio.imread(first_demo / cam / os.listdir(first_demo / cam)[0])
        h, w = example_img.shape[:2]
        c = example_img.shape[2] if example_img.ndim==3 else 1
        out_replay_buffer.data.require_dataset(
            name=cam, shape=(total_steps,h,w,c), chunks=(1,h,w,c),
            compressor=image_compressor, dtype=np.uint8
        )
    # ====== END CHANGED PART ======

    
    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    
    n_cameras = 0
    camera_idxs = set() 
    if image_keys is not None:
        n_cameras = len(image_keys)
        camera_idxs = set(int(x.split('_')[-1]) for x in image_keys)
    else:
        # estimate number of cameras
        episode_video_dir = in_video_dir.joinpath(str(0))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        camera_idxs = set(int(x.stem) for x in episode_video_paths)
        n_cameras = len(episode_video_paths)
    
    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    timestamps = in_replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]

    with tqdm(total=n_steps*n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]

                episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
                this_camera_idxs = set(int(x.stem) for x in episode_video_paths)
                if image_keys is None:
                    for i in this_camera_idxs - camera_idxs:
                        print(f"Unexpected camera {i} at episode {episode_idx}")
                for i in camera_idxs - this_camera_idxs:
                    print(f"Missing camera {i} at episode {episode_idx}")
                    if image_keys is not None:
                        raise RuntimeError(f"Missing camera {i} at episode {episode_idx}")

                for video_path in episode_video_paths:
                    camera_idx = int(video_path.stem)
                    if image_keys is not None:
                        # if image_keys provided, skip not used cameras
                        if camera_idx not in camera_idxs:
                            continue

                    # read resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'camera_{camera_idx}'
                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name])
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        _ = out_replay_buffer.data.require_dataset(
                            name=arr_name,
                            shape=(n_steps,oh,ow,3),
                            chunks=(1,oh,ow,3),
                            compressor=image_compressor,
                            dtype=np.uint8
                        )
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(
                        input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(read_video(
                            video_path=str(video_path),
                            dt=dt,
                            img_transform=image_tf,
                            thread_type='FRAME',
                            thread_count=n_decoding_threads
                        )):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))
                        
                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))

                        if step_idx == (episode_length - 1):
                            break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
    return out_replay_buffer

