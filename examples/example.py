#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gym
import imageio
import numpy as np

import habitat.gym  # noqa: F401
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    images_to_video,
)


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README
    
    # Create output directory for videos
    output_dir = "./data/example_videos"
    os.makedirs(output_dir, exist_ok=True)

    with gym.make("RearrangePddlTask-v0") as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        
        # Method 1: Using imageio for direct video writing
        video_file_path = os.path.join(output_dir, "example_pick_task_imageio.mp4")
        video_writer = imageio.get_writer(video_file_path, fps=30)
        
        # Method 2: Collect frames for batch video creation
        frames = []
        
        while not terminal:
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )  # noqa: F841
            
            # Method 1: Direct video writing with imageio
            # Convert observations to renderable image
            render_obs = observations_to_image(observations, info)
            # Add text overlay with metrics
            render_obs = overlay_frame(render_obs, info)
            # Write frame to video
            video_writer.append_data(render_obs)
            
            # Method 2: Store frames for batch processing
            frames.append(render_obs.copy())
            
            count_steps += 1
            
            # Optional: Limit episode length for demo
            if count_steps >= 100:  # Stop after 100 steps for demo
                terminal = True
                
        print("Episode finished after {} steps.".format(count_steps))
        
        # Finalize Method 1 video
        video_writer.close()
        print(f"Video saved using imageio: {video_file_path}")
        
        # Method 2: Create video using habitat utilities
        if frames:
            batch_video_name = "example_pick_task_habitat_utils"
            images_to_video(frames, output_dir, batch_video_name, fps=30, quality=9)
            print(f"Video saved using habitat utils: {output_dir}/{batch_video_name}.mp4")
        
        # Method 3: Using env.render() for raw sensor data
        env.reset()
        raw_video_path = os.path.join(output_dir, "example_pick_task_raw.mp4")
        raw_video_writer = imageio.get_writer(raw_video_path, fps=30)
        
        print("Recording raw sensor video...")
        terminal = False
        count_steps = 0
        while not terminal and count_steps < 50:  # Shorter demo for raw video
            observations, reward, terminal, info = env.step(env.action_space.sample())
            # Get raw RGB sensor data
            raw_frame = env.render(mode="rgb_array")
            raw_video_writer.append_data(raw_frame)
            count_steps += 1
            
        raw_video_writer.close()
        print(f"Raw sensor video saved: {raw_video_path}")
        
        print(f"\n✅ All videos saved in: {output_dir}")
        print("📁 Files created:")
        print("  • example_pick_task_imageio.mp4 (with overlays)")
        print("  • example_pick_task_habitat_utils.mp4 (with overlays)")  
        print("  • example_pick_task_raw.mp4 (raw sensor data)")


if __name__ == "__main__":
    example()
