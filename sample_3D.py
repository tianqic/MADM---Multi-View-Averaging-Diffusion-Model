"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import numpy as np
import torch as th
import torch.distributed as dist
import argparse
import os
import torch.nn as nn
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataloader_scripts.load_pet_2_5D import LoadTestData
import nibabel as nib

def main():
    args = create_argparser().parse_args()

    args.in_channels = args.load_adj * 2 + 1 + args.out_channels

    dist_util.setup_dist()
    logger.configure()
    logger.log("loading model and diffusion...")
    models = []
    diffusion = None
    for i in args.model_axis:
        model_path = os.path.join(args.model_root, f"model_{i}.pt")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model.requires_grad_(False)
        models.append(model)

    logger.log("sampling...")
    test_dir = os.path.join(args.data_root, "test")
    test_input = LoadTestData(root_dir=test_dir, load_adj=args.load_adj)

    for idx in range(len(test_input)):
        print(f"idx: {idx} / {len(test_input)}")
        whole_image = None
        sample_fn = diffusion.p_sample_loop
        model_kwargs = {}
        test_input.idx = idx
        
        shape = (args.image_size, args.image_size, test_input.get_zsize())
        prior_nii = nib.load(os.path.join(args.load_prior_root, f"{test_input.get_name(idx)}umap_pred.nii"))
        prior_numpy = prior_nii.get_fdata()
        comb_img = np.zeros(shape)
        for i in range(args.sample_num):
            prior = th.zeros(shape).to(dist_util.dev())
            prior[:,:,:test_input.get_original_z()]= th.from_numpy(prior_numpy).to(dist_util.dev())
            noisy_priors = []
            for n in range(args.avg_start_number): 
                if args.prior_start_t != None:
                    noisy_priors.append(diffusion.q_sample(prior, th.tensor(args.prior_start_t).to(dist_util.dev())))
                else:
                    noisy_priors.append(th.randn(shape).to(dist_util.dev()))
            whole_image = sample_fn(
                models,
                args.model_axis,
                test_input,
                shape,
                args.batch_size,
                args.prior_start_t,
                noise=noisy_priors,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            whole_image = whole_image.cpu().numpy()
            comb_img += whole_image
            whole_image = whole_image[:,:,:test_input.get_original_z()]
            whole_image = nib.Nifti1Image(whole_image, affine=np.eye(4))

            axis = "".join(args.model_axis)

            # Save individual sampled volume
            output_dir = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_ave_first_ddpm_full_single")
            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(output_dir, f"{test_input.get_name(idx)}pred_{i}.nii")
            nib.save(whole_image, save_path)

        # Save averaged combined volume
        comb_img /= args.sample_num
        comb_img = comb_img[:, :, :test_input.get_original_z()]
        comb_img[comb_img < 0] = 0
        comb_img = nib.Nifti1Image(comb_img, affine=np.eye(4))

        output_dir_comb = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_comb")
        os.makedirs(output_dir_comb, exist_ok=True)

        save_path_comb = os.path.join(output_dir_comb, f"{test_input.get_name(idx)}pred.nii")
        nib.save(comb_img, save_path_comb)




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=64,
        use_ddim=False,
        out_channels=1,
        model_root="models",
        model_axis=["x", "y", "z"],
        prior_start_t=200, #range from 1 to 999 for starting noise level add to prior, None for no prior
        load_adj=8,
        avg_start_number=2,
        sample_num=1,
        save_root="MADM",
        load_prior_root="cGAN_prior",
        data_root="NAC_data",
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
