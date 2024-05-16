# ----------------------------------
# ---- SOURCE MODEL AND MODEL TYPE
model_type = 'PixArtMS_XL_2'
model = '/path/to/sigma/checkpoint/based/on/img/size'
resume_from = dict(checkpoint=model, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
# VAE AND TEXT ENCODER MODELS
vae_file = '/path/to/vae/folder/'
te_file = '/path/to/text_encoder/T5-XXL-SIGMA'


# ----------------------------------
# --- FOLDERS
work_dir="/path/to/training/"
data_root = 'training/datasets/<datasetname>'
# TRAINING IMAGE FOLDER
dataset = 'training/datasets/<datasetname>'
caption_extension = '.txt'
caption_extension2 = '_wd.txt' #can be None if no secondary captions
# TRAINING CONFIG FOLDER - meta data json file
generate_captions_metadata = True
captions_metadata_file = 'training/datasets/<datasetname>/InternData/data_info.json'
# LOGGING FOLDER

logs_dir="logs"
# MODEL OUTPUT NAME
output_model_name = "MODEL_OUTPUT_NAME"


# ----------------------------------
# ---- DATASET PREPARATION
image_size = 512 #image resolution [256,512,1024]
bucket_size_min = 384
bucket_size_max = 1536
bucket_upscale = False # do NOT upscale image
bucket_res_steps = 64 #must be multiple of 8, e.g.: 8, 16, 24, 32, 40, ...
multi_scale = True #multiscale images
aspect_ratio_type = "ASPECT_RATIO_512" # [ASPECT_RATIO_1024, ASPECT_RATIO_512, ASPECT_RATIO_256]

data = dict(
    type='InternalDataMSSigmaCustom',  #use InternalData<<MS>>Sigma for multi-size images
    root='InternData', #this is where the data.json metadata file is contained 
    image_list_json=['data_info.json'], 
    transform='default_train', 
    load_vae_feat=False, 
    load_t5_feat=False
    )
shuffle_dataset = True

# ----------------------------------
# ---- PARAMETERS

# --- BASIC
train_batch_size = 1 #16
num_epochs = 10
mixed_precision = 'bf16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = False

# optimizer = dict(type='AdamWSchedulerFreeWrapper', lr=1e-5, weight_decay=3e-2, eps=1e-10)
# optimizer = dict(type='CAMEWrapper', lr=1.618e-6, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
optimizer = dict(type='AdamW8bit', lr=1.618e-6, weight_decay=3e-2, eps=1e-10)
auto_lr = dict(rule="sqrt") #linear|sqrt
lr_schedule = "constant"
lr_schedule_args = dict(num_warmup_steps=0)
lr_warmup_steps = 0
use_iddpm = False

# --- ADVANCED
gradient_accumulation_steps = 1
save_model_epochs = 5
save_model_steps = 5000

model_max_length = 300 #text embedding tokens max length
real_prompt_ratio = 0.7 #ratio to switch between prompt and sharegpt4v caption
grad_checkpointing = True
train_sampling_steps=1000
snr_loss=False
min_snr_gamma=3
sample_posterior=True
use_huber_loss=True
debiased_estimation_loss=True

scale_factor = 0.13025 # SDXL VAE scaling factor, should NOT be changed
pe_interpolation = 1.0 #positional embedding interpolation
num_workers = 4
class_dropout_prob = 0.1
gradient_clip = 0.1
gc_step = 1
valid_num = 0 #take as valid aspect-ratio when sample number >= valid_num
pred_sigma = True
learn_sigma = True
input_perturbation = 0.1 #https://arxiv.org/abs/2301.11706
prediction_type=None # [epsilon, v_prediction, None]

kv_compress = False
kv_compress_config = {
    'sampling': "conv", # ['conv', 'uniform', 'ave']
    'scale_factor': 2,
    'kv_compress_layer': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
}
visualize = False
skip_step = 0  # skip steps during training for checkpoint resumed training, should be 0 for new finetune
qk_norm = False
micro_condition = False

# ---- SAMPLING AND VALIDATION
report_to = "tensorboard"
eval_sampling_steps = 250
visualize = True
log_interval = 10
validation_prompts = [
    "dog",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

# LCM
loss_type = 'huber'
huber_c = 0.001
num_ddim_timesteps=50
w_max = 15.0
w_min = 3.0
ema_decay = 0.95

