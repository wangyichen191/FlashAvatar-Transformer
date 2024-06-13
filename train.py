import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips
import imageio
from tqdm import tqdm

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep
from utils.camera_utils import angle2cam_mv, PanoHeadcam2Gaussiancam
from datetime import datetime

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)
    
    ## dataloader
    # data_dir = os.path.join('dataset', args.idname)
    # mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    data_dir = os.path.join('data', args.idname)
    mica_datadir = os.path.join(data_dir, "tracker_output", args.idname)
    log_dir = os.path.join(data_dir, f"log_{datetime.now().strftime('@%Y%m%d-%H%M%S')}")
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    ply_dir = os.path.join(log_dir, "ply")
    video_dir = os.path.join(log_dir, "video")
    desc_input = input("Mission Description:")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'description.txt'), 'w') as f:
        f.writelines(desc_input)
    scene = Scene_mica(data_dir, mica_datadir, train_type=0, white_background=lpt.white_background, device = args.device)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000
    for iteration in range(first_iter, opt.iterations + 1):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera when train
        if not args.test:
            if not viewpoint_stack:
                viewpoint_stack = scene.getCameras().copy()
                random.shuffle(viewpoint_stack)
                if len(viewpoint_stack)>2000: 
                    viewpoint_stack = viewpoint_stack[:2000]
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)) 
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getCameras().copy()
                if len(viewpoint_stack)>2000: 
                    viewpoint_stack = viewpoint_stack[:2000]
            viewpoint_cam = viewpoint_stack.pop() 
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose 
        # verts_final, rot_delta, scale_coef = DeformModel.decode(codedict, scaling=10-(iteration / opt.iterations)*9)
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict, scaling=10)
        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        mouth_mask = viewpoint_cam.mouth_mask
        
        loss_huber = huber_loss(image, gt_image, 0.1) + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)

        # loss_huber = huber_loss(image, gt_image, 0.1) 
        
        loss_G = 0.
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image*head_mask)
        gt_image_percep = normalize_for_percep(gt_image*head_mask)
        if iteration>mid_num:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep))*0.05

        loss = loss_huber*1 + loss_G*1

        loss.backward()

        DeformModel.eval()
        with torch.no_grad():
            # Optimizer step
            if iteration < opt.iterations :
                gaussians.optimizer.step()
                DeformModel.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                DeformModel.optimizer.zero_grad(set_to_none = True)
            
            # print loss
            if iteration % 500 == 0:
                if iteration<=mid_num:
                    print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
                else:
                    print("step: %d, huber: %.5f, percep: %.5f" %(iteration, loss_huber.item(), loss_G.item()))
            
            # visualize results
            if iteration % 500 == 0 or iteration==1:
                save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                save_image[:, :args.image_res, :] = gt_image_np
                save_image[:, args.image_res:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.png"), save_image[:,:,[2,1,0]])
            
            # save checkpoint
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")

            # save gaussian
            # if iteration % 5000 == 0:
            #     gaussians.save_ply(os.path.join(ply_dir, f"{iteration}" + ".ply"))
            if iteration % (50 if args.test else 5000) == 0:
                # compute average exp
                # expr = DeformModel.default_expr_code
                # eyelids = torch.zeros_like(viewpoint_cam.eyelids).to(args.device)
                # eyes_pose = torch.zeros_like(viewpoint_cam.eyes_pose).to(args.device)
                # jaw_pose = torch.zeros_like(viewpoint_cam.jaw_pose).to(args.device)
                viewpoint_list = scene.getCameras().copy()
                # for viewpoint in viewpoint_list:
                #     expr += viewpoint.exp_param
                #     eyelids += viewpoint.eyelids
                #     eyes_pose += viewpoint.eyes_pose
                #     jaw_pose += viewpoint.jaw_pose
                # average_expr = expr/len(viewpoint_list)
                # average_eyelids = eyelids/len(viewpoint_list)
                # average_eyes_pose = eyes_pose/len(viewpoint_list)
                # average_jaw_pose = jaw_pose/len(viewpoint_list)

                I = matrix_to_rotation_6d(torch.cat([torch.eye(3)[None]], dim=0).to(args.device))
                codedict['expr'] = viewpoint_list[457].exp_param
                codedict['eyelids'] = viewpoint_list[457].eyelids
                codedict['eyes_pose'] = viewpoint_list[457].eyes_pose
                codedict['jaw_pose'] = viewpoint_list[457].jaw_pose
                verts_final, rot_delta, scale_coef = DeformModel.decode(codedict, scaling=10-(iteration / opt.iterations)*9)
                gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

                print("\n[ITER {}] Saving Gaussian".format(iteration))
                gaussians.save_ply(os.path.join(ply_dir, f"{iteration}_neural" + ".ply"))

                # render video
                video_path = os.path.join(video_dir, f"{iteration}_457" + ".mp4")
                with imageio.get_writer(video_path, mode='I', fps=60, codec='libx264') as video_out:
                    yaw_scale = 50.0
                    pitch_scale = 20.0
                    num_frames = 300
                    for idx in tqdm(range(num_frames), desc="render video"):
                        c = angle2cam_mv(
                                yaw=torch.tensor(yaw_scale * np.sin(idx*np.pi*2/num_frames), device='cuda').unsqueeze(0).type(torch.float32)/180 * np.pi, 
                                # pitch=torch.tensor(max(pitch_scale, ) * np.sin(idx*np.pi*4/num_frames), device='cuda').unsqueeze(0).type(torch.float32)/180 * np.pi, 
                                pitch=torch.tensor( pitch_scale * np.sin(idx*np.pi*4/num_frames), device='cuda').unsqueeze(0).type(torch.float32)/180 * np.pi, 
                                radius=1.3
                            )
                        cam_list = PanoHeadcam2Gaussiancam(c, 'cuda', viewpoint_cam)
                        view = cam_list[0]
                        img = render(view, gaussians, ppt, background)["render"]
                        img = (img * 255).clamp(0, 255).squeeze(0).to(torch.uint8)
                        img = img.permute(1,2,0).to('cpu').numpy()
                        video_out.append_data(img)
        DeformModel.train()

                

           