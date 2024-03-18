''' Inference script for DiffPortrait3D'''
import os
import argparse
# torch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# distributed 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP 
# data
from dataset import pano_head
# utilsl_datas
from utils.checkpoint import load_from_pretrain
from utils.utils import set_seed, count_param, print_peak_memory
# model
from control_model.ControlNet.cldm.model import create_model
import imageio

def tensor_to_image(tensor):
    # Assuming tensor shape is [1, 3, 64, 64]
    # Convert the tensor to a numpy array and move the channel dimension to the last axis
    image_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Rescale the values from [-1, 1] to [0, 255]
    image_np = ((image_np + 1) * 0.5 * 255).clip(0,255).astype('uint8')

    return image_np


TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16 if TORCH_VERSION == "1" else torch.bfloat16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")


def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict   

def visualize(args, name, batch_data, infer_model, global_step, nSample):
    infer_model.eval()
    # video length #max(nSample, batch_data["image"].squeeze().shape[0])
    target_imgs = batch_data["condition_image"].squeeze().cuda()
    conditions = batch_data["condition"].squeeze().cuda()
    if args.denoise_from_fea_map:
        fea_condtion = batch_data['fea_condition'].squeeze().cuda()
    text = batch_data["text_blip"]
    c_cross = infer_model.get_learned_conditioning(text)
    c_cross = c_cross.repeat(nSample, 1, 1)
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    gene_img_list = []
    cond_img = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(target_imgs.unsqueeze(0)))
    cond_img = cond_img.repeat(nSample, 1, 1, 1)
    cond_img_cat = [cond_img]
    for i in range(conditions.shape[0] // nSample):
        print("Generate Image {} in {} images".format(nSample * i, conditions.shape[0])) 
        inpaint = None
        if args.denoise_from_fea_map:
            fea_map_enc = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(fea_condtion[i*nSample: i*nSample+nSample]))
            c = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [c_cross], "image_control": cond_img_cat, 'feature_control':fea_map_enc}
            if args.control_mode == "controlnet_important":
                uc = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [uc_cross]}
            else:
                uc = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [uc_cross], "image_control": cond_img_cat}
            c['wonoise'] = True
            uc['wonoise'] = True
        # generate images
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            infer_model.to(args.device)
            infer_model.eval()
            gene_img, _ = infer_model.sample_log(cond=c,
                                    batch_size=nSample, ddim=True,
                                    ddim_steps=50, eta=0.5,
                                    unconditional_guidance_scale=3,
                                    unconditional_conditioning=uc,
                                    inpaint=inpaint
                                    )
            gene_img = infer_model.decode_first_stage(gene_img)
            for j in range(nSample):
                gene_img_list.append(gene_img[j].clamp(-1, 1).cpu())
    image = target_imgs.unsqueeze(0)
    latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
    rec_image = infer_model.decode_first_stage(latent) # no rec_img since it is 
    writer = imageio.get_writer(f"{args.local_image_dir}/{str(batch_data['infer_img_name'][0]).split('.')[0]}.mp4", fps=10)
    for idm, tensor in enumerate(gene_img_list):
            tensorimg = tensor.cpu() #torch.cat((target_imgs.cpu(), tensor.cpu()), dim=2)
            writer.append_data(tensor_to_image(tensorimg.cpu())) 
def main(args):
    # ******************************
    # initializing
    # ******************************
    # assign rank
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.rank = int(os.environ['RANK'])
    args.device = torch.device("cuda", args.local_rank)
    os.makedirs(args.local_image_dir,exist_ok=True)
    #os.makedirs(args.local_log_dir,exist_ok=True)
    if args.rank == 0:
        print(args)
    # initial distribution comminucation
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # set seed for reproducibility
    set_seed(args.seed)
    # ******************************
    # create model
    # ******************************
    model = create_model(args.model_config).cpu()
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.local_rank)
    if args.local_rank == 0:
        print('Total base  parameters {:.02f}M'.format(count_param([model])))
    # ******************************
    # load pre-trained models
    # ******************************
    ckpt_path = args.resume_dir #os.path.join('/home/ygu/Documents/CVPR2024/model_state-540000-001.th')
    print('loading state dict from {} ...'.format(ckpt_path))
    load_state_dict(model, ckpt_path, strict=False) 
    torch.cuda.empty_cache()
    # ******************************
    # create DDP model
    # ******************************
    #torch.cuda.set_device(args.local_rank)
    model = DDP(model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
                gradient_as_bucket_view=True # will save memory
    )
    print_peak_memory("Max memory allocated after creating DDP", args.local_rank)
    # ******************************
    # create dataset and dataloader
    # ******************************
    image_transform = T.Compose([
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.test_dataset == "pano_head":
        test_dataset_cls = getattr(pano_head, args.test_dataset+'_val_pose_sequence_batch_mm')
        test_image_dataset = test_dataset_cls(image_transform=image_transform,
                                image_folder = args.image_folder,
                                sequence_path = args.sequence_path,
                                fea_condition_root = args.fea_condition_root,
                                )
    else:
        print("find the appropriate dataset class!")
        return
    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=args.val_batch_size,
                                  num_workers=1,
                                  pin_memory=True,
                                  shuffle=False)
    test_image_dataloader_iter = iter(test_image_dataloader)
    if args.local_rank == 0:
        print(f"image dataloader created: dataset={args.test_dataset} batch_size={args.train_batch_size} ")
    dist.barrier()
    first_print = True
    infer_model = model.module if hasattr(model, "module") else model
    print(f"[rank{args.rank}] start training loop!")
    for itr in range(0, 100000):
        test_batch_data = next(test_image_dataloader_iter)
        with torch.no_grad():
            nSample = 8 # video length 
            visualize(args, "val_images", test_batch_data, infer_model, itr, nSample=nSample)
        # Clear cache
        if first_print or itr % 200 == 0:
            torch.cuda.empty_cache()
            print_peak_memory("Max memory allocated After running {} steps:".format(itr), args.local_rank)
        first_print = False
if __name__ == "__main__":
    str2bool = lambda arg: bool(int(arg))
    parser = argparse.ArgumentParser(description='Control Net training')
    ## Model
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video.yaml",
                        help="The path of model config file")
    parser.add_argument('--reinit_hint_block', action='store_true', default=False,
                        help="Re-initialize hint blocks for channel mis-match")
    parser.add_argument('--sd_locked', type =str2bool, default=True,
                        help='Freeze parameters in original stable-diffusion decoder')
    parser.add_argument('--only_mid_control', type =str2bool, default=False,
                        help='Only control middle blocks')
    parser.add_argument("--control_mode", type=str, default="balance",
                        help="Set controlnet is more important or balance.")
    ## Training
    parser.add_argument('--num_workers', type = int, default = 1,
                        help='total number of workers for dataloaders')
    parser.add_argument('--train_batch_size', type = int, default = 16,
                        help='batch size for each gpu in distributed training')
    parser.add_argument('--val_batch_size', type = int, default = 1,
                        help='batch size for each gpu during inference(must be set to 1)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument('--global_step', type=int, default=0,
                        help='initial global step to start with (use with --init_path)')
    ## Data
    parser.add_argument("--test_dataset", type=str, default="pano_head",
                        help="The dataset class for training.")
    parser.add_argument("--local_image_dir", type=str, default=None, required=True,
                        help="The local output directory where generated images will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument('--image_folder', type=str, default=None, help='Image folder for in the wild inference')
    parser.add_argument('--denoise_from_fea_map', action='store_true', default=False, help='Denoise from feature map')
    parser.add_argument('--sequence_path', type=str, default=None, help='Denoise from feature map')   
    parser.add_argument('--fea_condition_root', type=str, default=None, help='Denoise from feature map') 
    args = parser.parse_args()
    main(args)