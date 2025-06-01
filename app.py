import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import gc
import os

# 定義可用的模型 (添加管道类型信息)
MODELS = {
    "Stable Diffusion v1.5": {
        "path": "./models/stable-diffusion-v1-5",
        "class": StableDiffusionPipeline
    },
    "Stable Diffusion XL": {
        "path": "./models/sdxl-base-1.0",
        "class": StableDiffusionXLPipeline
    }
}

# 可選風格提示
STYLE_PRESETS = {
    "原始風格": "",
    "動漫風": "anime style, cel-shading, colorful",
    "現實攝影": "photorealistic, DSLR, 8k",
    "油畫風": "oil painting, canvas texture, brush strokes",
    "皮克斯風": "pixar style, cartoon, 3d rendered, expressive eyes",
    "浮世繪": "ukiyo-e, japanese woodblock print, traditional style",
    "馬賽克風": "mosaic style, broken tiles, abstract shapes",
    "吉普力風": "Studio Ghibli style, whimsical, detailed environment, soft colors",
    "科幻風": "sci-fi, futuristic, neon lights, cyberpunk",
    "奇幻風": "fantasy, magical, mythical creatures, epic landscapes",  
}

# 全域變數儲存當前模型
current_pipe = None
current_model = None

def load_model(model_name):
    global current_pipe, current_model
    
    if current_model == model_name and current_pipe is not None:
        return f"✅ 模型 {model_name} 已載入"
    
    # 清除前一個模型
    if current_pipe is not None:
        del current_pipe
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        model_info = MODELS.get(model_name)
        if not model_info:
            return f"❌ 找不到模型設定: {model_name}"
        
        model_path = model_info["path"]
        model_class = model_info["class"]
        
        # 判斷是資料夾還是單一檔案
        if os.path.isfile(os.path.join(model_path, "model.safetensors")):
            # 使用 from_single_file 載入
            current_pipe = model_class.from_single_file(
                os.path.join(model_path, "model.safetensors"),
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True
            ).to("cuda")
        else:
            # 使用 from_pretrained 載入資料夾
            current_pipe = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True
            ).to("cuda")
        
        current_model = model_name
        return f"✅ 成功載入模型: {model_name}"
        
    except Exception as e:
        return f"❌ 載入模型失敗: {str(e)}"


def generate_image(prompt, style, model_name):
    global current_pipe, current_model
    
    # 檢查是否需要切換模型
    if current_model != model_name or current_pipe is None:
        load_status = load_model(model_name)
        if load_status is None or "❌" in load_status:
            return None, load_status or "❌ 未知錯誤"
    
    try:
        full_prompt = f"{prompt}, {STYLE_PRESETS.get(style, '')}"
        
        with torch.inference_mode():
            # 添加异常处理以确保图像生成
            try:
                image = current_pipe(
                    full_prompt,
                    num_inference_steps=25 if "XL" in model_name else 20,
                    guidance_scale=8.0 if "XL" in model_name else 7.5,
                    width=1024 if "XL" in model_name else 512,
                    height=1024 if "XL" in model_name else 512
                ).images[0]
            except Exception as e:
                return None, f"❌ 生成過程中出錯: {str(e)}"
        
        return image, f"✅ 圖片生成完成！使用模型: {current_model}"
    except Exception as e:
        return None, f"❌ 生成失敗: {str(e)}"

# 初始化預設模型
def initialize_default_model():
    """初始化預設模型"""
    default_model = list(MODELS.keys())[0]  # 使用第一個模型作為預設
    return load_model(default_model)

# Gradio UI 設計
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ 文生圖：本地 AI 圖片生成器")
    gr.Markdown("支援多模型切換的 Stable Diffusion 圖片生成工具")
    
    # 模型選擇區域
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODELS.keys()), 
            label="🤖 選擇模型", 
            value=list(MODELS.keys())[0],
            info="選擇要使用的 AI 模型"
        )
        load_model_btn = gr.Button("🔄 載入模型", variant="secondary")
    
    # 模型狀態顯示
    model_status = gr.Textbox(
        label="模型狀態", 
        value="請選擇模型並點擊「載入模型」按鈕",
        interactive=False
    )
    
    gr.Markdown("---")
    
    # 圖片生成區域
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="📝 Prompt 描述", 
                placeholder="例如: a beautiful landscape with mountains and lakes",
                lines=3
            )
            style_dropdown = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()), 
                label="🎨 風格選擇", 
                value="原始風格"
            )
            
            with gr.Row():
                generate_button = gr.Button("🖼️ 生成圖片", variant="primary", scale=2)
                clear_button = gr.Button("🗑️ 清除", variant="secondary", scale=1)
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="生成結果", type="pil", height=400)
            generation_status = gr.Textbox(
                label="生成狀態", 
                value="準備就緒",
                interactive=False
            )
    
    # 範例提示詞
    gr.Markdown("### 💡 範例提示詞")
    example_prompts = [
        "1girl, anime style, cute face, big eyes, looking at viewer, detailed eyes, long hair, school uniform, sitting on a bench, soft light, cherry blossoms, spring, highly detailed, perfect lighting, masterpiece, anime background",
        "A peaceful countryside landscape, in the style of Studio Ghibli, soft colors, dreamy atmosphere, detailed environment, anime style, hand-drawn look, gentle lighting, cinematic composition，a Boy and a girl running hand in hand on forest path",
        "anime style, warrior girl, long flowing hair, armor, holding sword, dynamic pose, glowing effects, fantasy landscape, dramatic lighting, cinematic, epic, intricate details, masterpiece",
        "anime style, night cityscape, neon lights, futuristic, girl walking in the rain, umbrella, reflections on the ground, detailed background, soft light, cyberpunk vibes, high quality, masterpiece"
    ]
    
    with gr.Row():
        for prompt in example_prompts:
            gr.Button(prompt, size="sm").click(
                lambda p=prompt: p, 
                outputs=prompt_input
            )
    
    # 事件處理
    load_model_btn.click(
        fn=load_model,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, style_dropdown, model_dropdown],
        outputs=[output_image, generation_status]
    )
    
    clear_button.click(
        lambda: (None, "", "已清除"),
        outputs=[output_image, prompt_input, generation_status]
    )
    
    # 模型下拉選單變更時的提示
    model_dropdown.change(
        lambda model: f"💡 已選擇 {model}，請點擊「載入模型」按鈕",
        inputs=[model_dropdown],
        outputs=[model_status]
    )

    # 啟動時初始化預設模型
    demo.load(
        fn=initialize_default_model,
        outputs=[model_status]
    )

# 啟動本地服務
if __name__ == "__main__":
    demo.launch()