# app_design.py - Aug4Sat: AI-Powered Synthetic Satellite Imagery Generation
import gradio as gr
import base64
import os
import gc
import torch
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL STATE & SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global models (loaded on demand)
sdxl_pipe = None
prompt_generator = None

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_sdxl_with_lora():
    """Load SDXL + LoRA (called once at startup or on first generation)"""
    global sdxl_pipe
    
    if sdxl_pipe is not None:
        return "✅ SDXL already loaded"
    
    from diffusers import StableDiffusionXLPipeline
    
    yield "📥 Downloading SDXL base model..."
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        add_watermarker=False
    )
    
    yield "📥 Downloading LoRA weights from Hugging Face..."
    # Your LoRA from HuggingFace
    pipe.load_lora_weights("Sam-Roblex/LoRA-Sat")
    
    yield "🔧 Moving to GPU..."
    pipe = pipe.to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    sdxl_pipe = pipe
    
    mem_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    yield f"✅ SDXL + LoRA loaded ({mem_allocated:.1f} GB)"

# ============================================================================
# PROMPT GENERATOR CLASS
# ============================================================================

class SatellitePromptGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load Qwen model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )
    
    def unload(self):
        """Free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def generate_single_prompt(self, feature_dict, scene_type, density_bias):
        """Generate ONE prompt with special vegetation handling"""
        
        # Check if vegetation-focused
        is_vegetation_focused = (
            feature_dict.get('dense_forest') or 
            feature_dict.get('sparse_trees') or 
            feature_dict.get('grassland')
        ) and scene_type == 'rural-dominant' and density_bias == 'dense'
        
        if is_vegetation_focused:
            # Special vegetation prompt
            prompt_text = """Generate a satellite image description (3-4 sentences, 50-80 words) for a heavily vegetated rural landscape.

REQUIREMENTS:
- PRIMARY: Dense vegetation coverage (forests, agricultural fields, grasslands)
- Scene: Rural/agricultural landscape
- Density: Dense/thick vegetation
- AVOID: Urban elements, buildings, paved roads

VOCABULARY: "dense forest canopy", "thick tree coverage", "continuous vegetation", "agricultural fields with mature crops", "grassland", "wooded areas"

EXAMPLE: "Rural landscape with dense forest coverage dominating the terrain. Agricultural fields showing thick crop vegetation interspersed with wooded sections. Continuous tree canopy with sparse clearings and minimal exposed soil throughout."

Generate ONE description:"""
        else:
            # Standard prompt
            features = []
            if feature_dict.get('coastal_water') or feature_dict.get('rivers') or feature_dict.get('lakes'):
                features.append("water body")
            if feature_dict.get('dense_forest') or feature_dict.get('sparse_trees') or feature_dict.get('grassland'):
                features.append(f"{density_bias} vegetation")
            if feature_dict.get('paved_roads'):
                features.append("paved roads")
            if feature_dict.get('unpaved_roads'):
                features.append("unpaved roads")
            if feature_dict.get('residential') or feature_dict.get('commercial') or feature_dict.get('industrial'):
                features.append(f"{density_bias} building density")
            
            features_str = ", ".join(features) if features else "bareland"
            
            prompt_text = f"""Generate a satellite image description (2-3 sentences) for:
- Scene type: {scene_type}
- Features: {features_str}

Use terminology: "high/moderate/low building density", "sparse/moderate/dense vegetation", "paved/unpaved roads", "bareland", "water body"

Example: "Urban settlement with central vegetated area surrounded by dense buildings. Unpaved roads creating grid pattern. Mixed bareland and moderate vegetation throughout residential zone."

Generate description:"""
        
        messages = [{"role": "user", "content": prompt_text}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract description
        parts = generated.split("Generate description:")
        if len(parts) > 1:
            description = parts[-1].strip()
        else:
            description = generated.split('\n')[-1].strip()
        
        # Clean up
        description = description.replace('"', '').replace("'", "").strip()
        
        return description
    
    def generate_prompts(self, feature_dict, num_prompts, scene_type, density_bias):
        """Generate prompts one by one"""
        if self.model is None:
            self.load()
        
        prompts = []
        for i in range(num_prompts):
            try:
                prompt = self.generate_single_prompt(feature_dict, scene_type, density_bias)
                prompts.append(prompt)
            except Exception as e:
                # Fallback
                prompts.append(f"{scene_type} scene with {density_bias} features including typical landscape elements.")
        
        return prompts

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_dataset(
    # Water features
    rivers, lakes, coastal, ponds,
    # Vegetation features
    forest, sparse_trees, grass, ag_fields,
    # Road features
    paved, unpaved, highways,
    # Building features
    residential, commercial, industrial,
    # Generation settings
    count, scene, density, steps, guidance, width, height, seed, neg_prompt
):
    """Main generation function - yields status updates"""
    
    global sdxl_pipe, prompt_generator
    
    try:
        # ====================================================================
        # STEP 1: LOAD SDXL + LORA
        # ====================================================================
        
        if sdxl_pipe is None:
            for status in load_sdxl_with_lora():
                yield status, ""
        
        # ====================================================================
        # STEP 2: GENERATE PROMPTS WITH QWEN
        # ====================================================================
        
        yield "📝 Loading Qwen for prompt generation...", ""
        
        if prompt_generator is None:
            prompt_generator = SatellitePromptGenerator()
        
        # Build feature dictionary
        feature_dict = {
            'rivers': rivers,
            'lakes': lakes,
            'coastal_water': coastal,
            'ponds': ponds,
            'dense_forest': forest,
            'sparse_trees': sparse_trees,
            'grassland': grass,
            'agricultural_fields': ag_fields,
            'paved_roads': paved,
            'unpaved_roads': unpaved,
            'highways': highways,
            'residential': residential,
            'commercial': commercial,
            'industrial': industrial
        }
        
        yield f"📝 Generating {count} prompts...", ""
        
        prompts = prompt_generator.generate_prompts(
            feature_dict=feature_dict,
            num_prompts=int(count),
            scene_type=scene,
            density_bias=density
        )
        
        # Show sample prompts
        sample_text = "\n".join([f"  {i+1}. {p[:70]}..." for i, p in enumerate(prompts[:3])])
        yield f"✅ Generated {len(prompts)} prompts\n\nSample prompts:\n{sample_text}", ""
        
        # ====================================================================
        # STEP 3: UNLOAD QWEN
        # ====================================================================
        
        yield "🧹 Unloading Qwen to free memory...", ""
        prompt_generator.unload()
        torch.cuda.empty_cache()
        gc.collect()
        
        # ====================================================================
        # STEP 4: GENERATE IMAGES WITH SDXL + LORA
        # ====================================================================
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = f"{OUTPUT_DIR}/satellite_dataset_{timestamp}"
        os.makedirs(f"{exp_dir}/images", exist_ok=True)
        
        # Save prompts
        with open(f"{exp_dir}/prompts.txt", "w") as f:
            for i, p in enumerate(prompts, 1):
                f.write(f"{i}. {p}\n\n")
        
        # Prepare metadata
        metadata = {
            "config": {
                "features": feature_dict,
                "scene_type": scene,
                "density_bias": density,
                "steps": int(steps),
                "guidance": float(guidance),
                "resolution": f"{int(width)}x{int(height)}",
                "seed": seed if seed else "random",
                "negative_prompt": neg_prompt,
                "timestamp": timestamp
            },
            "images": []
        }
        
        # Parse seed
        base_seed = int(seed) if seed and seed.strip().isdigit() else 42
        
        successful = 0
        failed = 0
        
        yield f"🎨 Starting image generation (0/{len(prompts)})...", ""
        
        for i, prompt in enumerate(prompts):
            try:
                result = sdxl_pipe(
                    prompt=f"satellite imagery, aerial view, {prompt}",
                    negative_prompt=neg_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=torch.manual_seed(base_seed + i)
                )
                
                img = result.images[0]
                filename = f"image_{i+1:04d}.png"
                img.save(f"{exp_dir}/images/{filename}")
                
                metadata["images"].append({
                    "filename": filename,
                    "prompt": prompt,
                    "seed": base_seed + i
                })
                
                successful += 1
                
                # Cleanup
                del result, img
                torch.cuda.empty_cache()
                
                # Update progress every 5 images or at the end
                if (successful % 5 == 0) or (successful == len(prompts)):
                    progress_text = f"🎨 Generating images: {successful}/{len(prompts)} complete"
                    if failed > 0:
                        progress_text += f" ({failed} failed)"
                    yield progress_text, ""
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    failed += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # Save metadata
        with open(f"{exp_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # ====================================================================
        # FINAL STATUS
        # ====================================================================
        
        final_status = f"""✅ GENERATION COMPLETE!

📊 Summary:
  • Generated: {successful}/{len(prompts)} images
  • Failed: {failed}
  • Resolution: {width}x{height} px
  • Steps: {steps} | Guidance: {guidance}

📁 Output Directory:
  {exp_dir}

📝 Files:
  • {successful} images in /images/
  • prompts.txt (all prompts)
  • metadata.json (full config)

You can now download the entire folder or individual images."""
        
        yield final_status, exp_dir
        
    except Exception as e:
        error_msg = f"""❌ ERROR OCCURRED:

{str(e)}

Please check:
1. GPU memory available
2. All inputs are valid
3. Try reducing image count or resolution"""
        
        yield error_msg, ""

# ============================================================================
# UI COMPONENTS
# ============================================================================

def get_logo_base64():
    """Encode logo to base64"""
    try:
        with open("aug4sat.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        # Fallback if logo not found
        return ""

css = """
.gradio-container {
    max-width: 1600px !important;
}

.header-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 1rem 0 1rem 0;
}

.logo-img {
    width: 90px;
    height: 90px;
    object-fit: contain;
}

.title-section {
    display: flex;
    flex-direction: column;
}

.app-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #111827;
    margin: 0;
    line-height: 1;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}


.app-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 400;
    margin-top: 0.5rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

input::-webkit-inner-spin-button,
input::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}

input[type=number] {
    -moz-appearance: textfield !important;
}

.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    margin-top: 2rem;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 0.875rem;
}

.footer a {
    color: #2563eb;
    text-decoration: none;
    margin: 0 0.5rem;
}

.footer a:hover {
    text-decoration: underline;
}
"""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(css=css, title="Aug4Sat", theme=gr.themes.Default()) as demo:
    
    logo_b64 = get_logo_base64()
    
    # Header
    if logo_b64:
        gr.HTML(f"""
        <div class="header-wrapper">
            <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="Aug4Sat Logo">
            <div class="title-section">
                <h1 class="app-title">Aug4Sat</h1>
                <p class="app-subtitle">AI-Powered Synthetic Satellite Imagery Generation</p>
            </div>
        </div>
        """)
    else:
        gr.Markdown("# Aug4Sat\n### AI-Powered Synthetic Satellite Imagery Generation")
    
    with gr.Row(equal_height=False):
        # Left Column - Features
        with gr.Column(scale=1):
            
            gr.Markdown("### Water Bodies")
            with gr.Group():
                rivers = gr.Checkbox(label="Rivers")
                lakes = gr.Checkbox(label="Lakes")
                coastal = gr.Checkbox(label="Coastal Waters", value=True)
                ponds = gr.Checkbox(label="Ponds")
            
            gr.Markdown("### Vegetation")
            with gr.Group():
                forest = gr.Checkbox(label="Dense Forest")
                sparse_trees = gr.Checkbox(label="Sparse Trees", value=True)
                grass = gr.Checkbox(label="Grassland", value=True)
                ag_fields = gr.Checkbox(label="Agricultural Fields")
            
            gr.Markdown("### Roads & Infrastructure")
            with gr.Group():
                paved = gr.Checkbox(label="Paved Roads", value=True)
                unpaved = gr.Checkbox(label="Unpaved Roads", value=True)
                highways = gr.Checkbox(label="Highways")
            
            gr.Markdown("### Buildings")
            with gr.Group():
                residential = gr.Checkbox(label="Residential", value=True)
                commercial = gr.Checkbox(label="Commercial")
                industrial = gr.Checkbox(label="Industrial")
        
        # Right Column - Settings
        with gr.Column(scale=1):
            
            gr.Markdown("### Generation Settings")
            
            with gr.Row():
                count = gr.Number(label="Images", value=10, minimum=1, maximum=100, precision=0)
                steps = gr.Number(label="Steps", value=30, minimum=20, maximum=50, precision=0)
            
            with gr.Row():
                guidance = gr.Number(label="Guidance", value=7.5, minimum=5.0, maximum=15.0)
                seed = gr.Textbox(label="Seed (optional)", placeholder="Leave empty for random")
            
            scene = gr.Dropdown(
                ["balanced", "coastal-focused", "urban-heavy", "rural-dominant"],
                value="balanced",
                label="Scene Distribution"
            )
            
            density = gr.Dropdown(
                ["sparse", "moderate", "dense", "mixed"],
                value="moderate",
                label="Feature Density"
            )
            
            with gr.Row():
                width = gr.Radio([768, 1024], value=1024, label="Width")
                height = gr.Radio([768, 1024], value=1024, label="Height")
            
            gr.Markdown("### Advanced Settings")
            gr.Markdown("⚠️ **Warning:** Only modify if you understand prompt engineering")
            
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, distorted, text, watermark",
                lines=2
            )
    
    # Status display
    status_box = gr.Textbox(
        label="Status",
        lines=10,
        value="Ready to generate. Click 'Generate Dataset' to begin.",
        interactive=False
    )
    
    path_box = gr.Textbox(label="Output Path", lines=1, interactive=False)
    
    gen_btn = gr.Button("🚀 Generate Dataset", variant="primary", size="lg")
    
    # Footer
    gr.HTML("""
    <div class="footer">
        Made by <strong>SAM Qureshi</strong><br>
        <a href="https://samqureshi.me/" target="_blank">Website</a> | 
        <a href="https://www.linkedin.com/in/sam-qureshi/" target="_blank">LinkedIn</a> | 
        <a href="mailto:sam.qureshi.grad@gmail.com">Email</a> |
        <a href="https://github.com/AmirRoblex/Aug4Sat" target="_blank">GitHub</a>
    </div>
    """)
    
    # Wire up generation
    gen_btn.click(
        fn=generate_dataset,
        inputs=[
            rivers, lakes, coastal, ponds,
            forest, sparse_trees, grass, ag_fields,
            paved, unpaved, highways,
            residential, commercial, industrial,
            count, scene, density, steps, guidance, width, height, seed, neg_prompt
        ],
        outputs=[status_box, path_box]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Aug4Sat - AI-Powered Synthetic Satellite Imagery Generation")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*70)
    
    demo.launch(share=True)