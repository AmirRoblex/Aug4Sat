# app.py - Aug4Sat: AI-Powered Synthetic Satellite Imagery Generation
import gradio as gr
import base64
import os
import gc
import torch
import json
from datetime import datetime
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
template_generator = None

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
# TEMPLATE VARIANT GENERATOR
# ============================================================================

import random

class TemplateVariantGenerator:
    """Generate diverse prompts using template variants and synonym substitution"""
    
    def __init__(self):
        # Synonym dictionary from training data
        self.synonyms = {
            # Scene types
            'Coastal urban settlement': ['Coastal urban settlement', 'Coastal settlement', 'Coastal urban zone', 'Semi-arid coastal landscape'],
            'Urban settlement': ['Urban settlement', 'Urban area', 'Urban core', 'Peri-urban settlement'],
            'Rural arid landscape': ['Rural arid landscape', 'Rural landscape', 'Arid landscape', 'Rural agricultural area'],
            'Mixed-use settlement': ['Mixed-use settlement', 'Mixed settlement', 'Mixed-use urban settlement'],
            
            # Density
            'low building density': ['low building density', 'sparse building density', 'minimal building density'],
            'moderate building density': ['moderate building density', 'moderate development', 'moderate density'],
            'high building density': ['high building density', 'dense building density', 'dense development'],
            
            # Buildings
            'featuring': ['featuring', 'with', 'including'],
            'structures': ['structures', 'buildings', 'development'],
            'isolated structures': ['isolated structures', 'scattered structures', 'dispersed structures'],
            'tightly packed structures': ['tightly packed structures', 'dense building clusters', 'densely packed buildings'],
            'clusters of': ['clusters of', 'groups of', 'collections of'],
            
            # Roads
            'Paved roads form grid-like network': ['Paved roads form grid-like network', 'Grid-like paved road network', 'Paved road network forms grid pattern'],
            'run along the periphery': ['run along the periphery', 'line the edges', 'border the area'],
            'weave through': ['weave through', 'crisscross through', 'wind through', 'traverse'],
            'dirt tracks': ['dirt tracks', 'unpaved tracks', 'dirt roads'],
            
            # Vegetation
            'Sparse vegetation': ['Sparse vegetation', 'Scattered vegetation', 'Limited vegetation'],
            'consisting of': ['consisting of', 'including', 'with'],
            'scattered shrubs and bushes': ['scattered shrubs and bushes', 'isolated shrubs', 'sparse shrubs'],
            'across predominantly': ['across predominantly', 'on mostly', 'across primarily'],
            'bareland': ['bareland', 'bare terrain', 'exposed terrain', 'bare soil'],
            'interspersed with': ['interspersed with', 'mixed with', 'dotted with', 'scattered among'],
            'patches': ['patches', 'areas', 'sections'],
            
            # Water
            'adjacent water body': ['adjacent water body', 'nearby water body', 'water body adjacent'],
            'developed shoreline': ['developed shoreline', 'shoreline development', 'built shoreline']
        }
        
        # Template structures
        self.templates = {
            'coastal-moderate': [
                "{scene} with {density} {buildings}{water}. {roads}. {vegetation}.",
                "{scene}{water}, with {density} {buildings}. {roads}. {vegetation}.",
                "{scene} with {density} {buildings} along {roads_short}{water}. {vegetation}.",
            ],
            'urban-moderate': [
                "{scene} with {density}, {buildings}. {roads}. {vegetation}.",
                "{scene} with {density}. {roads} connect {buildings_short}. {vegetation}.",
                "{scene} with {density} {buildings}. {roads}. {vegetation}.",
            ],
            'rural-sparse': [
                "{scene} with {density}, {buildings}. {roads}. {vegetation}. {closure}",
                "{scene} with {density} {buildings} connected by {roads_short}. {vegetation}. {closure}",
                "{scene} with {buildings} connected by {roads_short}. {vegetation} across predominantly bareland. {closure}",
            ]
        }
    
    def _substitute(self, text):
        """Replace phrases with synonyms"""
        phrases = sorted(self.synonyms.keys(), key=len, reverse=True)
        
        for phrase in phrases:
            if phrase in text:
                replacement = random.choice(self.synonyms[phrase])
                text = text.replace(phrase, replacement, 1)
        
        return text
    
    def _get_scene_phrase(self, scene_type):
        mapping = {
            'coastal-focused': 'Coastal urban settlement',
            'urban-heavy': 'Urban settlement',
            'rural-dominant': 'Rural arid landscape',
            'balanced': 'Mixed-use settlement'
        }
        return mapping.get(scene_type, 'Settlement')
    
    def _get_density_phrase(self, density_bias):
        mapping = {
            'sparse': 'low building density',
            'moderate': 'moderate building density',
            'dense': 'high building density',
            'mixed': 'varied building density'
        }
        return mapping.get(density_bias, 'moderate building density')
    
    def _get_building_phrase(self, features, density_bias):
        has_buildings = any([features.get('residential'), features.get('commercial'), features.get('industrial')])
        
        if not has_buildings:
            return None
        
        if density_bias == 'sparse':
            return "featuring isolated structures and dispersed development"
        elif density_bias == 'dense':
            if features.get('industrial'):
                return "featuring dense building clusters and industrial facilities"
            else:
                return "featuring tightly packed structures"
        else:
            types = []
            if features.get('residential'): types.append("residential")
            if features.get('commercial'): types.append("commercial")
            if features.get('industrial'): types.append("industrial")
            
            if len(types) > 1:
                return f"featuring clusters of {' and '.join(types)} structures"
            else:
                return "featuring moderate development"
    
    def _get_road_phrase(self, features):
        has_paved = features.get('paved_roads') or features.get('highways')
        has_unpaved = features.get('unpaved_roads')
        
        if has_paved and has_unpaved:
            return "Paved roads run along the periphery while unpaved dirt tracks weave through the interior"
        elif has_paved:
            if features.get('highways'):
                return "Paved road network forms grid pattern with major highways"
            else:
                return "Paved roads form grid-like network connecting neighborhoods"
        elif has_unpaved:
            return "Unpaved tracks and dirt roads crisscross through the area"
        return None
    
    def _get_vegetation_phrase(self, features, density_bias):
        has_dense = features.get('dense_forest')
        has_sparse = features.get('sparse_trees') or features.get('grassland')
        
        if has_dense:
            return "Dense vegetation consisting of thick forest coverage and agricultural fields"
        elif has_sparse:
            if density_bias == 'sparse':
                return "Sparse vegetation consisting of scattered shrubs and bushes across predominantly bareland"
            else:
                return "Sparse vegetation patches interspersed with bareland"
        else:
            return "Minimal vegetation, predominantly bareland"
    
    def _get_water_phrase(self, features):
        if features.get('coastal_water'):
            return "adjacent water body with developed shoreline"
        elif features.get('rivers') or features.get('lakes'):
            return "with visible water bodies throughout the terrain"
        return None
    
    def _select_template_category(self, scene_type, density_bias):
        """Choose appropriate template category"""
        if 'coastal' in scene_type and density_bias in ['moderate', 'dense']:
            return 'coastal-moderate'
        elif 'urban' in scene_type:
            return 'urban-moderate'
        elif 'rural' in scene_type and density_bias == 'sparse':
            return 'rural-sparse'
        else:
            return 'urban-moderate'
    
    def _fill_template(self, template, features, scene_type, density_bias):
        """Fill template with feature descriptions"""
        replacements = {
            '{scene}': self._get_scene_phrase(scene_type),
            '{density}': self._get_density_phrase(density_bias),
            '{buildings}': self._get_building_phrase(features, density_bias) or "featuring minimal development",
            '{buildings_short}': "neighborhoods" if features.get('residential') else "structures",
            '{roads}': self._get_road_phrase(features) or "Dirt tracks traverse the area",
            '{roads_short}': "paved roads" if features.get('paved_roads') else "unpaved tracks",
            '{vegetation}': self._get_vegetation_phrase(features, density_bias) or "Minimal vegetation",
            '{water}': f", {self._get_water_phrase(features)}" if self._get_water_phrase(features) else "",
            '{closure}': "No visible water bodies; development is minimal and dispersed" if not features.get('coastal_water') and scene_type == 'rural-dominant' else ""
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        # Clean up
        result = result.replace("  ", " ").replace(" .", ".").replace("..", ".").strip()
        
        return result
    
    def generate(self, features, scene_type, density_bias):
        """Generate a single prompt with template variants and synonyms"""
        # Select template category
        category = self._select_template_category(scene_type, density_bias)
        
        # Pick random template
        template = random.choice(self.templates[category])
        
        # Fill template
        filled = self._fill_template(template, features, scene_type, density_bias)
        
        # Apply synonym substitution
        varied = self._substitute(filled)
        
        return varied

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
    
    global sdxl_pipe, template_generator
    
    try:
        # ====================================================================
        # STEP 1: LOAD SDXL + LORA
        # ====================================================================
        
        if sdxl_pipe is None:
            for status in load_sdxl_with_lora():
                yield status, ""
        
        # ====================================================================
        # STEP 2: GENERATE PROMPTS
        # ====================================================================
        
        global template_generator
        
        if template_generator is None:
            template_generator = TemplateVariantGenerator()
        
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
        
        # Generate prompts
        prompts = []
        for i in range(int(count)):
            prompt = template_generator.generate(feature_dict, scene, density)
            prompts.append(prompt)
        
        # Show sample prompts
        sample_text = "\n".join([f"  {i+1}. {p[:70]}..." for i, p in enumerate(prompts[:3])])
        yield f"✅ Generated {len(prompts)} prompts\n\nSample prompts:\n{sample_text}", ""
        
        # ====================================================================
        # STEP 3: GENERATE IMAGES WITH SDXL + LORA
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
                
                # Update progress after each image
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