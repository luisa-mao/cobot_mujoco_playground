from cobot_networks import Generator, Encoder, Decoder
import jax
import jax.numpy as jnp
from utils import load_student_params, denormalize
from mujoco_playground_vision_autoencoder_trainer import collect_vision_data
from cobot_env import CobotEnv, default_config
import numpy as np
from utils import load_inference_without_env, get_obs_shape 
import mediapy as media
import orbax.checkpoint as ocp
import cv2
import os
from vision_utils import tile

checkpointer = ocp.PyTreeCheckpointer()


model = Generator()
params = model.init({'params':jax.random.PRNGKey(0), 'dropout':jax.random.PRNGKey(1)}, jnp.ones((1, 256, 256, 3)), train=True)

encoder = Encoder()
decoder = Decoder()

checkpoint_dir = "/home/luisamao/villa_spaces/sim_ws/checkpoints/autoencoder_run/"

abstract_target = {
                'wristcam_params': params["params"],
                'basecam_params': params["params"],
                'iteration': 0,
            }
print(params['params'].keys())
# debug_print_ckpt_structure(abstract_target["params"])
restored_checkpoint = load_student_params(checkpoint_dir, abstract_target)
basecam_params = restored_checkpoint['basecam_params']

from_env = False

if from_env:
    env_cfg = default_config()
    num_envs = 2
    env_cfg['vision'] = True
    env_cfg['vision_config']['nworld'] = num_envs
    env_cfg['include_teacher_obs'] = True
    env = CobotEnv(config=env_cfg)

    restore_checkpoint_path = "/home/luisamao/villa_spaces/sim_ws/checkpoints/classic-river-146/000045547520"
    teacher_obs_shape = get_obs_shape(env, num_envs, obs_key="teacher_obs")
    teacher_jit_inference_fn = load_inference_without_env(restore_checkpoint_path, obs_dim=teacher_obs_shape, action_dim= env.action_size)

    collect_key = jax.random.PRNGKey(0)
    traj_pixels = collect_vision_data(env, teacher_jit_inference_fn, num_envs=num_envs, key=collect_key, episode_length=250)

    # 2. Flatten for the Autoencoder: (T, E, H, W, C) -> (T*E, H, W, C)
    batch_pixels = traj_pixels["basecam"][:, 0]
    print("batch pixels shape:", batch_pixels.shape)
    print("batch pixels min max", batch_pixels.min(), batch_pixels.max())
    recon = model.apply({"params": basecam_params }, batch_pixels, train = False)

    encoded = encoder.apply({"params": basecam_params["Encoder_0"]}, batch_pixels, train = False)
    decoded = decoder.apply({"params": basecam_params["Decoder_0"]}, encoded, train = False)
    print("out min max", decoded.min(), decoded.max())

    input_vis = denormalize(jax.device_get(batch_pixels))
    output_vis = denormalize(recon)
    output_vis2 = denormalize(decoded)
    vis = np.concatenate([input_vis, output_vis, output_vis2], axis=1)  
    print(vis.shape, vis.dtype, vis.min(), vis.max())
    media.write_video("reconstruction.mp4", vis, fps=30)

else:
    import os
    import numpy as np
    from PIL import Image
    import mediapy as media


    images_dir = "/scratch/luisamao/cobot_jenga_dataset/trainB"
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 1. Take a sample of 64 images
    random_indices = np.random.choice(len(images), size=64, replace=False)
    sample_paths = [os.path.join(images_dir, images[i]) for i in random_indices]

    # 2. Load, resize (optional/recommended), and preprocess
    pixel_list = []
    for p in sample_paths:
        img = Image.open(p).convert('RGB')
        # Ensure they match the model's expected input size, e.g., 256x256
        img = img.resize((256, 256)) 
        pixel_list.append(np.array(img))

    # 3. Convert to float32 and scale to [-1, 1]
    batch_pixels = np.stack(pixel_list).astype(np.float32)
    batch_pixels = (batch_pixels / 127.5) - 1.0
    print("inp min max", batch_pixels.min(), batch_pixels.max())

    # 4. Forward pass through Encoder and Decoder
    encoded = encoder.apply({"params": basecam_params["Encoder_0"]}, batch_pixels, train=False)
    decoded = decoder.apply({"params": basecam_params["Decoder_0"]}, encoded, train=False)
    print("out min max", decoded.min(), decoded.max())

    output_vis = denormalize(decoded)
    print("denormalized min max", output_vis.min(), output_vis.max())
    output_vis = tile(output_vis[:64], 8) # 8x8 grid

    media.write_image("real_autoencoder_output.jpg", output_vis)
