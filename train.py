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

import os
import torch
from random import randint
from utils.loss_utils import (
    l1_loss,
    ssim,
    predicted_normal_loss,
    delta_normal_loss,
    zero_one_loss,
    first_order_edge_aware_loss,
    entropy_loss,
    tv_loss,
    neutral_color_loss,
    avg_light_intensity_loss,
    metallic_loss,
    roughness_loss,
    ambient_occlusion_loss,
)
from gaussian_renderer import render, network_gui, render_lighting
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time

from scene.cameras import Camera

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9999, stdoutToServer=True, stderrToServer=True, suspend=False)


def training(
        dataset, opt, pipe, testing_iterations, saving_iterations,
        hdr_path=None,
):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.brdf_dim,
        dataset.brdf_mode,
        dataset.brdf_envmap_res,
        hdr_path=hdr_path,
    )

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    #bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    #bg_color = [0, 0, 0]
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1): 
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) 

        if pipe.brdf:
            gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
            gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
            if gaussians.brdf_mode=="envmap":
                gaussians.brdf_mlp.build_mips()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, debug=False)

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        loss, losses_extra = _calculate_loss(
            iteration, opt, pipe, render_pkg, viewpoint_cam,
            gaussians.brdf_mlp.base,
            gaussians,
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            gt_image = viewpoint_cam.original_image.cuda()
            losses_extra['psnr'] = psnr(image, gt_image).mean()

            training_report(tb_writer, iteration, loss, losses_extra, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)

            if pipe.brdf and pipe.brdf_mode=="envmap":
                gaussians.brdf_mlp.clamp_(min=0.0, max=1.0)


def _calculate_loss(iteration, opt, pipe, render_pkg, viewpoint_camera: Camera, env_light, gaussians):
    image, viewspace_point_tensor, visibility_filter, radii = (
        render_pkg["render"],
        render_pkg["viewspace_points"],
        render_pkg["visibility_filter"],
        render_pkg["radii"],
    )
    gt_image = viewpoint_camera.original_image.cuda()

    losses_extra = {}

    if pipe.brdf and iteration > opt.normal_reg_from_iter:
        if iteration < opt.normal_reg_util_iter:
            losses_extra['predicted_normal'] = predicted_normal_loss(render_pkg["normal"], render_pkg["normal_ref"],
                                                                     render_pkg["alpha"])
        losses_extra['zero_one'] = entropy_loss(render_pkg["alpha"])
        #losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])

        if "delta_normal_norm" in render_pkg.keys():
            losses_extra['delta_reg'] = delta_normal_loss(render_pkg["delta_normal_norm"], render_pkg["alpha"])
        else:
            assert opt.lambda_delta_reg == 0

    if opt.lambda_opacity_zero_one > 0:
        losses_extra['opacity_zero_one'] = entropy_loss(gaussians.get_opacity)

    if opt.lambda_diffuse_smooth > 0:
        diffuse = render_pkg["diffuse"]
        #image_mask = viewpoint_camera.gt_alpha_mask.cuda()

        loss_diffuse_smooth = first_order_edge_aware_loss(diffuse, gt_image)
        #loss_diffuse_smooth = first_order_edge_aware_loss(diffuse * image_mask, gt_image)
        losses_extra["diffuse_smooth"] = loss_diffuse_smooth

    if opt.lambda_specular_smooth > 0:
        specular = render_pkg["specular"]
        #image_mask = viewpoint_camera.gt_alpha_mask.cuda()

        #loss_specular_smooth = first_order_edge_aware_loss(specular * image_mask, gt_image)
        loss_specular_smooth = first_order_edge_aware_loss(specular, gt_image)

        losses_extra["specular_smooth"] = loss_specular_smooth

    if opt.lambda_env_neutral > 0:
        specular_env = render_pkg["specular_env_color"]
        #image_mask = viewpoint_camera.gt_alpha_mask.cuda()

        loss_neutral_color = neutral_color_loss(specular_env)
        #loss_neutral_color = neutral_color_loss(specular_env * image_mask)
        losses_extra["env_neutral"] = loss_neutral_color

    if opt.lambda_env_smooth > 0:
        loss_env_smooth = tv_loss(env_light.permute(3, 0, 1, 2))
        losses_extra["env_smooth"] = loss_env_smooth

    if opt.lambda_specular_color_smooth > 0:
        specular_color = render_pkg["specular_env_color"]
        #image_mask = viewpoint_camera.gt_alpha_mask.cuda()
        #loss_specular_color_smooth = first_order_edge_aware_loss(specular_color * image_mask, gt_image)
        loss_specular_color_smooth = first_order_edge_aware_loss(specular_color, gt_image)
        losses_extra["specular_color_smooth"] = loss_specular_color_smooth

    if opt.lambda_avg_light_intensity > 0:
        loss_light_intensity = avg_light_intensity_loss(env_light)
        losses_extra["avg_light_intensity"] = loss_light_intensity

    if opt.lambda_metallic > 0:
        losses_extra["metallic"] = metallic_loss(gaussians)

    if opt.lambda_roughness > 0:
        losses_extra["roughness"] = roughness_loss(gaussians)

    if opt.lambda_ambient_occlusion > 0:
        losses_extra["ambient_occlusion"] = ambient_occlusion_loss(gaussians)

    # Loss

    render_image_l1 = l1_loss(image, gt_image)
    losses_extra['l1'] = render_image_l1

    loss = (1.0 - opt.lambda_dssim) * render_image_l1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    for k in losses_extra.keys():
        loss += getattr(opt, f'lambda_{k}') * losses_extra[k]

    return loss, losses_extra


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, losses_extra, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses_extra['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in losses_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', losses_extra[k].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        for k in render_pkg.keys():
                            if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                                continue
                            if k == "depth":
                                image_k = apply_depth_colormap(-render_pkg[k][0][...,None])
                                image_k = image_k.permute(2,0,1)
                            elif k == "alpha":
                                image_k = apply_depth_colormap(render_pkg[k][0][...,None], min=0., max=1.)
                                image_k = image_k.permute(2,0,1)
                            else:
                                if "normal" in k:
                                    render_pkg[k] = 0.5 + (0.5*render_pkg[k]) # (-1, 1) -> (0, 1)
                                image_k = torch.clamp(render_pkg[k], 0.0, 1.0)
                            tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, k), image_k[None], global_step=iteration)
                        
                        if renderArgs[0].brdf:
                            lighting = render_lighting(scene.gaussians, resolution=(512, 1024))
                            if tb_writer:
                                tb_writer.add_images(config['name'] + "/lighting", lighting[None], global_step=iteration)
                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()  
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 100, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 100, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--hdr_path', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        hdr_path=args.hdr_path,
    )

    # All done
    print("\nTraining complete.")
