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
import math
import torchvision.utils as tutils

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
from diffusers import DDPMScheduler
import torch.nn.functional as F

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

def make_bucket_resolutions(max_res, min_size=256, max_size=1024, step_size=64):
    max_w, max_h = max_res
    max_area = (max_w // step_size) * (max_h // step_size)
    resolutions = set()
    
    size = int(math.sqrt(max_area)) * step_size
    resolutions.add((size, size, size / size))
    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // step_size)) * step_size)
        resolutions.add((width, height, width / height))
        resolutions.add((height, width, height / width))
        
        size += step_size
    resolutions = list(resolutions)
    resolutions.sort()
    return resolutions #(size, size, aspect_ratio)

def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def apply_snr_weight(loss, timesteps, noise_scheduler, gamma):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float().to(loss.device)  # from paper
    loss = loss * snr_weight
    return loss

def apply_debiased_estimation_loss(loss, timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1/(torch.sqrt(snr_t) + 1e-10)
    loss = weight * loss
    return loss

def training_losses_ddpm(model, latents, timesteps, noise, model_kwargs):
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    if config.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=config.prediction_type)    
    
    if noise_scheduler.config.prediction_type == 'epsilon':
        target = noise
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    noise_pred = model(
        noisy_latents, 
        timesteps, 
        **model_kwargs
    ).chunk(2, 1)[0]
    
    if config.use_huber_loss is False:
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    else:
        huber_c = 0.002
        alpha = -math.log(huber_c) / noise_scheduler.config.num_train_timesteps
        huber_c = math.exp(-alpha * timesteps[0])
    
        loss = torch.mean(
            huber_c * (torch.sqrt((noise_pred.float() - target.float()) ** 2 + huber_c**2) - huber_c)
        ).to(accelerator.device)
    
    if config.snr_loss:
        loss = apply_snr_weight(loss, timesteps, noise_scheduler, config.min_snr_gamma)
        
    if config.debiased_estimation_loss:
        loss = apply_debiased_estimation_loss(loss, timesteps, noise_scheduler)
        
    loss = loss.mean().to(accelerator.device)
    return {"loss": loss}

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
    
    progress_bar = tqdm.tqdm(range(total_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="Steps")
    
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        # accelerator.print(f"\nEpoch: {epoch} / {config.num_epochs + 1}")
        model.train()
        train_loss = 0.0
        if config.optimizer["type"].lower().endswith("schedulefree"):
            optimizer.optimizer.train()
        
        for step, batch in enumerate(train_dataloader):
            # start = time.process_time()
            # if step < skip_step:
            #     global_step += 1
            #     continue    # skip data in the resumed ckpt
            
            with accelerator.accumulate(model):
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        latents = vae.encode(batch[0]).latent_dist.sample().mul_(config.scale_factor)
                    
                grad_norm = None
                noise = torch.randn_like(latents, device=latents.device)
                noise = pyramid_noise_like(noise, latents, 10, 0.8)
                # add input perturbation https://arxiv.org/abs/2301.11706
                noise += config.input_perturbation * torch.rand_like(latents, device=latents.device)
                
                bsz = latents.shape[0]
                timesteps = torch.randint(0, config.train_sampling_steps, (bsz,), device=latents.device).long()
                
                # added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                data_info = batch[3]
                with accelerator.autocast():
                    with torch.set_grad_enabled(False):
                        txt_tokens = tokenizer(
                            batch[1], 
                            max_length=max_length, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt"
                        ).to(accelerator.device)
                        y = text_encoder(
                            txt_tokens.input_ids, 
                            attention_mask=txt_tokens.attention_mask
                        )[0][:, None]
                        y_mask = txt_tokens.attention_mask[:, None, None]
                    
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                    if config.use_iddpm:
                        loss_term = train_diffusion.training_losses(
                            model,
                            latents,
                            timesteps,
                            noise=noise,
                            model_kwargs=model_kwargs
                        )
                    else:
                        loss_term = training_losses_ddpm(
                            model, 
                            latents, 
                            timesteps,
                            noise=noise,
                            model_kwargs=model_kwargs
                        )
                    
                    loss = loss_term["loss"].mean().to(accelerator.device)
                
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                optimizer.step()
                lr_scheduler.step()
                if config.optimizer["type"].lower().endswith("sophia"):
                    optimizer.optimizer.update_hessian()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    progress_bar.update(1)
                    global_step += 1
                    train_loss = 0.0
            if grad_norm is None:
                grad_norm = 0.0
            else:
                grad_norm = accelerator.gather(grad_norm).mean().item()
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm}
            progress_bar.set_postfix(**logs)
            # print(time.process_time() - start)
            
            # =================================
            # if load_vae_feat:
            #     z = batch[0]
            # else:
            #     with torch.no_grad():
            #         with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
            #             posterior = vae.encode(batch[0]).latent_dist
            #             if config.sample_posterior:
            #                 z = posterior.sample()
            #             else:
            #                 z = posterior.mode()

            # clean_images = z * config.scale_factor
            # data_info = batch[3]
            # # tutils.save_image(batch[0], f"{config.data_root}/latents/latent_{step}.png")

            # if load_t5_feat:
            #     y = batch[1]
            #     y_mask = batch[2]
            # else:
            #     with torch.no_grad():
            #         txt_tokens = tokenizer(
            #             batch[1], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            #         ).to(accelerator.device)
            #         y = text_encoder(
            #             txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
            #         y_mask = txt_tokens.attention_mask[:, None, None]

            # # Sample a random timestep for each image
            # bs = clean_images.shape[0]
            # timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            # grad_norm = None
            # data_time_all += time.time() - data_time_start
            # with accelerator.accumulate(model):
            #     # Predict the noise residual
            #     noise = torch.randn_like(clean_images, device=clean_images.device)
            #     noise = pyramid_noise_like(noise, clean_images)
            #     # add input perturbation https://arxiv.org/abs/2301.11706
            #     noise = noise + config.input_perturbation * torch.rand_like(clean_images, device=clean_images.device)
            #     with accelerator.autocast():
            #         loss_term = train_diffusion.training_losses(
            #             model, 
            #             clean_images, 
            #             timesteps, 
            #             noise=noise,
            #             model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
            #         )
            #     loss = loss_term['loss'].mean()
            #     accelerator.backward(loss)
                
            #     optimizer.step()
            #     lr_scheduler.step()
            #     optimizer.zero_grad()
                
            #     if accelerator.sync_gradients:
            #         accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
            #     progress_bar.update(1)


            # lr = lr_scheduler.get_last_lr()[0]
            
            # logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            # if grad_norm is not None:
            #     logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            # log_buffer.update(logs)
            # if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
            #     t = (time.time() - last_tic) / config.log_interval
            #     t_d = data_time_all / config.log_interval
            #     avg_time = (time.time() - time_start) / (global_step + 1)
            #     eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
            #     eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
            #     log_buffer.average()
            #     info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
            #            f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}"
            #     info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
            #     logger.info(info)
            #     last_tic = time.time()
            #     log_buffer.clear()
            #     data_time_all = 0
            # logs.update(lr=lr)
            # accelerator.log(logs, step=global_step)

            # global_step += 1
            # data_time_start = time.time()
            # =====================================
            if config.optimizer["type"].lower().endswith("schedulefree"):
                optimizer.optimizer.eval()
                model.eval()
            if global_step % config.save_model_steps == 0:
                # model.to(torch.float32)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    model=accelerator.unwrap_model(model),
                                    epoch=epoch,
                                    name=config.output_model_name
                                    )
            # if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
            #     accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                #     log_validation(model, global_step, device=accelerator.device, vae=vae)

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            # model.to(torch.float32)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if config.optimizer["type"].lower().endswith("schedulefree"):
                    optimizer.optimizer.eval()
                    model.eval()
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                model=accelerator.unwrap_model(model),
                                epoch=epoch,
                                name=config.output_model_name
                                )
        accelerator.wait_for_everyone()
    accelerator.end_training()


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
    
    set_data_root(config.data_root)
    buckets = make_bucket_resolutions((config.image_size,config.image_size), config.bucket_size_min, config.bucket_size_max, config.bucket_res_steps)
    
    accelerator.print(f"Dataset buckets: {buckets}")
    
    dataset = build_dataset(
        config.data, 
        resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio, max_length=max_length, 
        config=config, buckets=buckets
    )
    
    # print(len(dataset), dataset.ratio_nums, sum(dataset.ratio_nums.values()))
    for bk in dataset.ratio_nums:
        if dataset.ratio_nums[bk] != 0:
            accelerator.print(f"({bk}: {dataset.ratio_nums[bk]} / {len(dataset)}) - {dataset.aspect_ratio['{:.2f}'.format(bk)]}")
    
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
            shuffle=config.shuffle_dataset
        )
    else:
        train_dataloader = build_dataloader(
            dataset, 
            num_workers=config.num_workers, 
            batch_size=config.train_batch_size, 
            shuffle=config.shuffle_dataset
        )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    accelerator.print(f'Loading vae from {config.vae_file} ... ')
    vae = AutoencoderKL.from_pretrained(config.vae_file, torch_dtype=weight_dtype).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor
    logger.info(f"VAE Scale Factor: {config.scale_factor}")
    accelerator.print('VAE LOADED!')

    accelerator.print(f'Loading Tokenizer from {config.te_file} ... ')
    tokenizer = T5Tokenizer.from_pretrained(config.te_file, subfolder="tokenizer")    
    accelerator.print(f'Loading Text Encoder from {config.te_file} ... ')
    text_encoder = T5EncoderModel.from_pretrained(config.te_file, subfolder="text_encoder", torch_dtype=weight_dtype).to(accelerator.device)
    
    text_encoder.requires_grad_(False)
    if config.grad_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        text_encoder.train()
    
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
    
    train_diffusion = None
    noise_scheduler = None
    if config.use_iddpm:
        train_diffusion = IDDPM(
            str(config.train_sampling_steps),
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            snr=config.snr_loss,
            rescale_learned_sigmas=True,
        )
    else:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=config.train_sampling_steps, clip_sample=True, timestep_spacing='trailing', rescale_betas_zero_snr=True,
        )
        prepare_scheduler_for_custom_training(noise_scheduler=noise_scheduler, device=accelerator.device)
    
    model = build_model(config.model_type,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        gc_step=config.gc_step,
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    model.to(accelerator.device)
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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * config.num_epochs
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    train()