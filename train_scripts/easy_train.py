import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import random
import tqdm
import mimetypes
import json

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image
from torch.utils.data import RandomSampler

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

warnings.filterwarnings("ignore")

def pyramid_noise_like(noise, device, iterations=6, discount=0.4):
    b,c,w,h = noise.shape
    u = torch.nn.Upsample(size=(w,h), mode="bicubic").to(device)
    for i in range(iterations):
        r = random.random() * (1.618 * torch.pi) + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()

def parse_args():
    parser = argparse.ArgumentParser(description="Parsing arguments ... ")
    parser.add_argument("--config", type=str, help="Config file for training")
    # parser.add_argument("--base_model", type=str, help="Path to base model")
    # parser.add_argument('--text_encoder', type=str, help="Path to text encoder model")
    
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args

def generate_prompt_metadata(config):
    json_entries = []
    for (dirpath, dirnames, filenames) in os.walk(config.dataset):
        for filename in tqdm.tqdm(filenames):
            filepath = Path(dirpath).joinpath(filename)
            mime_type, _ = mimetypes.guess_type(filepath)
            if 'image/' in mime_type:
                entry = {}
                img = Image.open(filepath)
                entry['width'] = img.width
                entry['height'] = img.height
                entry['ratio'] = img.width / img.height
                entry['path'] = filename
                
                caption_file = filepath.with_suffix(config.caption_extension)
                
                
                with open(caption_file) as prompt:
                    entry['prompt'] = prompt.read().strip()
                
                entry['sharegpt4v'] = ''
                if config.caption_extension2 is not None:
                    filename_only = filepath.with_suffix('')
                    second_caption_file = str(filename_only) + config.caption_extension2
                    with open(Path(second_caption_file)) as prompt2:
                        entry['sharegpt4v'] = prompt2.read().strip()
                json_entries.append(entry)
    
    metadata_json_file = Path(config.captions_metadata_file)
    with open(metadata_json_file, 'w') as json_file:
        json_file.write(json.dumps(json_entries, indent=4))
    return True

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    load_t5_feat = getattr(train_dataloader.dataset, 'load_t5_feat', False)
    
    model.train()
    # optimizer.optimizer.train()
    
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                global_step += 1
                continue    # skip data in the resumed ckpt
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            clean_images = z * config.scale_factor
            data_info = batch[3]

            if load_t5_feat:
                y = batch[1]
                y_mask = batch[2]
            else:
                with torch.no_grad():
                    txt_tokens = tokenizer(
                        batch[1], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    y = text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                noise = torch.randn_like(clean_images, device=clean_images.device)
                noise = pyramid_noise_like(noise, clean_images)
                loss_term = train_diffusion.training_losses(
                    model, 
                    clean_images, 
                    timesteps, 
                    noise=noise,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                )
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()
                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}"
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()

            if global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    name=config.output_model_name
                                    )
            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                #     log_validation(model, global_step, device=accelerator.device, vae=vae)

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                name=config.output_model_name
                                )
        accelerator.wait_for_everyone()

if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    init_train = 'DDP'
    fsdp_plugin = None
    
    even_batches = False if config.multi_scale else True
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )
    
    if config.generate_captions_metadata:
        accelerator.print('Generating image prompt metadata file ... ')
        generate_prompt_metadata(config)
    
    current_time = datetime.datetime.now()
    formatted_dt = current_time.strftime('%Y%m%d-%H%M%S')
    
    log_name = f"{config.output_model_name}_{formatted_dt}_trainlog.log"
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)
    
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
        config.dump(os.path.join(config.work_dir, f"{config.output_model_name}_{formatted_dt}_config.py"))
    
    logger = get_root_logger(os.path.join(config.work_dir, log_name))
    logger.info(accelerator.state)
    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    
    image_size = config.image_size # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, "pred_sigma", True)
    learn_sigma = getattr(config, "learn_sigma", True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    
    accelerator.print(f'Loading vae from {config.vae_file} ... ')
    vae = AutoencoderKL.from_pretrained(config.vae_file, torch_dtype=torch.float16).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor
    logger.info(f"VAE Scale Factor: {config.scale_factor}")
    accelerator.print('VAE LOADED!')
    
    accelerator.print(f'Loading TE/TOKENIZER from {config.te_file} ... ')
    tokenizer = T5Tokenizer.from_pretrained(config.te_file, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(config.te_file, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
    accelerator.print('TE/TOKENIZER LOADED!')
    
    # TODO: ADD VISUALIZE/VALIDATION LOSS GENERATION DATA

    # GENERATE THE NULL TOKEN EMBEDS USED TO CAPTION DROPOUT
    if not os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth'):
        null_tokens = tokenizer(
            "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to(accelerator.device)
        null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
        torch.save(
            {"uncond_prompt_embeds": null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
            f"output/pretrained_models/null_embed_diffusers_{max_length}token.pth"
        )
    
    model_kwargs = {
        "config": config,
        "pe_interpolation": config.pe_interpolation,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "micro_condition": config.micro_condition,
    }
    
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss,
        )
    model = build_model(config.model_type,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    model.requires_grad_(True)
    
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    missing, unexpected = load_checkpoint(
        config.model,
        model,
        load_ema=config.get("load_ema", False),
        max_length=max_length
    )
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    
    # TODO: ADD OPTION FOR FSDP clip grad norm calculation
    
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data, 
        resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio, max_length=max_length, 
        config=config,
    )
    
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset), dataset=dataset,
            batch_size=config.train_batch_size, 
            aspect_ratios=dataset.aspect_ratio, drop_last=True,
            ratio_nums=dataset.ratio_nums, config=config, 
            valid_num=config.valid_num
        )
        train_dataloader = build_dataloader(
            dataset, 
            batch_sampler=batch_sampler, 
            num_workers=config.num_workers,
            # shuffle=config.shuffle_dataset
        )
    else:
        train_dataloader = build_dataloader(
            dataset, 
            num_workers=config.num_workers, 
            batch_size=config.train_batch_size, 
            # shuffle=config.shuffle_dataset
        )

    # Setup optimizer and LR Scheduler
    lr_scale_ratio = 1
    if config.get("auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer, 
            **config.auto_lr
        )
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")
    
    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs
        
    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()