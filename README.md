# AI_FINAL_PROJECT
text to image

environment:
python : https://www.python.org/downloads/
torch : pip install torch
torchvision : pip install torchvision
diffusers : pip install diffusers
transformers : pip install transformers
accelerate : pip install accelerate
safetensors : pip install safetensors

using this code download the stable-diffusion-v1-5 from hugging face:

        pip install huggingface_hub
        
        from huggingface_hub import snapshot_download
        
        snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", local_dir="./models/stable-diffusion-v1-5")

for cpu:

        from diffusers import StableDiffusionPipeline
        import torch
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "./models/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
            local_files_only=True
        ).to("cpu") 

for gpu:

        from diffusers import StableDiffusionPipeline
        import torch
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "./models/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  
            use_safetensors=True,
            local_files_only=True
        ).to("cuda") 

file structure
AI_FINAL_PROJECT/
├── app.py
└── models/
    └── stable-diffusion-v1-5/


if you run successfully, you will see (http://127.0.0.1:7860) this on your terminal
