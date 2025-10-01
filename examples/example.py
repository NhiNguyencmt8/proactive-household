#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import imageio
import numpy as np

import habitat
from habitat import get_config
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

    # Load the social navigation configuration with Spot robot
    config = get_config("habitat-lab/habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav.yaml")
    
    # The social nav config has RGB sensors configured for agent_0 (Spot robot)
    # We'll use the existing head_rgb_sensor for video recording
    
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        
        # Method 1: Using imageio for direct video writing
        video_file_path = os.path.join(output_dir, "social_nav_task_imageio.mp4")
        video_writer = imageio.get_writer(video_file_path, fps=30)
        
        # Method 2: Collect frames for batch video creation
        frames = []
        
        while not env.episode_over:
            # Sample a random action from the action space
            action = env.action_space.sample()
            observations = env.step(action)
            
            # Method 1: Direct video writing with imageio
            # Convert observations to renderable image
            render_obs = observations_to_image(observations, {})
            # Add text overlay with metrics
            render_obs = overlay_frame(render_obs, {})
            # Write frame to video
            video_writer.append_data(render_obs)
            
            # Method 2: Store frames for batch processing
            frames.append(render_obs.copy())
            
            count_steps += 1
            
            # Optional: Limit episode length for demo
            # For longer videos, increase this number or comment out to run full episodes
            if count_steps >= 1000:  # Stop after 1000 steps for very long demo (33+ seconds)
                break
                
        print("Episode finished after {} steps.".format(count_steps))
        
        # Finalize Method 1 video
        video_writer.close()
        print(f"Video saved using imageio: {video_file_path}")
        
        # Method 2: Create video using habitat utilities
        if frames:
            batch_video_name = "social_nav_task_habitat_utils"
            images_to_video(frames, output_dir, batch_video_name, fps=30, quality=9)
            print(f"Video saved using habitat utils: {output_dir}/{batch_video_name}.mp4")
        

        print(f" Videos saved in: {output_dir}")


if __name__ == "__main__":
    example()
