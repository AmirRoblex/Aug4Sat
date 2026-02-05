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
# TEMPLATE VARIANT GENERATOR (USING ACTUAL TRAINING DATA VOCABULARY)
# ============================================================================

import random

class TemplateVariantGenerator:
    """Generate diverse prompts using actual training data phrases for LoRA compatibility"""
    
    def __init__(self):
        # ACTUAL phrases from training data for synonym substitution
        self.synonyms = {
            # Scene types (from training data)
            'Rural arid landscape': ['Rural arid landscape', 'Arid rural landscape', 'Rural settlement in an arid landscape', 'Sparse rural settlement in an arid landscape'],
            'Urban settlement': ['Urban settlement', 'Urban residential settlement', 'Mixed-use urban settlement', 'Semi-urban settlement'],
            'Coastal scene': ['Coastal scene', 'Coastal urban settlement', 'Coastal scene with a water body'],
            
            # Building density (exact training phrases)
            'low building density': ['low building density', 'sparse building density', 'minimal building density', 'low to moderate building density'],
            'moderate building density': ['moderate building density', 'moderate to high building density', 'moderate development'],
            'high building density': ['high building density', 'dense building density', 'tightly packed'],
            
            # Buildings (from training)
            'scattered': ['scattered', 'dispersed', 'isolated'],
            'clusters': ['clusters', 'groups', 'patches'],
            'tightly packed': ['tightly packed', 'densely packed', 'compact'],
            'residential structures': ['residential structures', 'residential buildings', 'houses', 'dwellings'],
            
            # Roads (exact training phrases)
            'Unpaved roads': ['Unpaved roads', 'Unpaved dirt roads', 'Dirt roads', 'Unpaved tracks'],
            'Paved roads': ['Paved roads', 'A paved road', 'Paved road network'],
            'form a grid': ['form a grid', 'form a grid-like network', 'form a loose grid', 'form an irregular grid'],
            'connect': ['connect', 'connecting', 'traverse'],
            
            # Vegetation (exact training phrases)
            'Sparse vegetation': ['Sparse vegetation', 'Minimal vegetation', 'Limited vegetation coverage'],
            'scattered shrubs and bushes': ['scattered shrubs and bushes', 'scattered shrubs', 'scattered shrubs and trees', 'isolated shrubs'],
            'Moderate vegetation': ['Moderate vegetation', 'Vegetation coverage', 'Moderate vegetation coverage'],
            'patches of trees': ['patches of trees', 'scattered trees', 'patches of trees and shrubs'],
            'Dense vegetation': ['Dense vegetation', 'Vegetation is dense', 'Thick vegetation'],
            
            # Terrain (exact training phrases)
            'predominantly bareland': ['predominantly bareland', 'predominantly bare', 'extensive bareland', 'mostly bare'],
            'bare, sandy terrain': ['bare, sandy terrain', 'bare, arid land', 'dry, sandy terrain', 'bareland'],
            'arid': ['arid', 'dry', 'semi-arid'],
            
            # Water (exact training phrases)
            'adjacent to': ['adjacent to', 'near', 'along'],
            'a water body': ['a water body', 'water', 'the water'],
            'shoreline': ['shoreline', 'coast', 'coastal area'],
            'No visible water bodies': ['No visible water bodies', 'No water bodies are visible', 'with no visible water bodies']
        }
        
        # Template structures - 2-3 SENTENCES matching actual training prompt structure
        self.templates = {
            'rural-sparse': [
                # Pattern: Scene + feature. Infrastructure. Vegetation/terrain.
                "{scene} with {density} and {buildings}{water_s1}. {roads} {road_action} the {building_location}, {road_detail}. {veg_terrain}{spatial_closure}",
                "{scene} with {density}, featuring {buildings}{water_s1}. {roads} visible throughout terrain, {road_detail}. {veg_terrain}{spatial_closure}",
                "Arid landscape with {buildings} and {roads_lower}{water_s1}. {veg_terrain} {closure_phrase}{spatial_closure}",
                "{scene} with {buildings} connected by {roads_lower}{water_s1}. {veg_terrain} {density} throughout rural zone{spatial_closure}",
            ],
            'urban-moderate': [
                # Pattern: Scene + development. Roads/buildings. Vegetation/terrain.
                "{scene} with {density} featuring {buildings}{water_s1}. {roads} creating grid pattern, {road_detail}. {veg_terrain}{spatial_closure}",
                "{scene} with {density} and {buildings}{water_s1}. {roads} visible throughout area, {road_detail}. Mixed {terrain} and {veg_lower}{spatial_closure}",
                "Urban area with {buildings} and {density}{water_s1}. {roads} {road_action} residential zone, {road_detail}. {veg_terrain}{spatial_closure}",
                "{scene} with {density}{water_s1}. {buildings} and {roads_lower} throughout settlement, {road_detail}. {veg_terrain}{spatial_closure}",
            ],
            'coastal': [
                # Pattern: Coastal scene with water. Buildings/roads along shoreline. Vegetation.
                "Coastal scene with water body adjacent to {shoreline_desc}. {buildings} and {roads_lower} visible along shoreline, {road_detail}. {veg_terrain}",
                "Coastal scene with water body along developed {shore_position}. {density} with {buildings} near water, {road_detail}. {roads} visible throughout area. {veg_terrain}",
                "Coastal urban settlement adjacent to water body. {buildings} and {roads_lower} along {shore_position}, {road_detail}. {veg_terrain}",
                "Coastal scene with water body {water_position} {shoreline_desc}. {density} featuring {buildings}, {road_detail}. {roads} visible along coast. {veg_terrain}",
            ]
        }
    

    def _substitute(self, text):
        """Replace phrases with synonyms for diversity (avoiding duplicate substitutions)"""
        import re
        phrases = sorted(self.synonyms.keys(), key=len, reverse=True)
        
        # Track what we've already replaced to avoid duplicates
        already_replaced = set()
        
        for phrase in phrases:
            # Skip if we already replaced a phrase that contains this one
            if any(phrase in existing for existing in already_replaced):
                continue
            
            if phrase in text:
                replacement = random.choice(self.synonyms[phrase])
                # Only replace if not creating duplicate words
                test_text = text.replace(phrase, replacement, 1)
                # Check for duplicate words (e.g., "coverage coverage")
                words = test_text.split()
                if len(words) != len(set(words)):
                    continue  # Skip this replacement if it creates duplicates
                text = test_text
                already_replaced.add(phrase)
        
        # Clean up any remaining duplicate words
        words = text.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def _get_scene_type(self, scene, has_coastal_water):
        """Determine scene template type"""
        if has_coastal_water:
            return 'coastal'
        elif scene == 'rural':
            return 'rural-sparse'
        else:
            return 'urban-moderate'
    
    def _build_scene_phrase(self, scene):
        """Build scene opening"""
        if scene == 'rural':
            return random.choice(self.synonyms['Rural arid landscape'])
        else:
            return random.choice(self.synonyms['Urban settlement'])
    
    def _build_density_phrase(self, density):
        """Build building density phrase"""
        density_map = {
            'low': random.choice(self.synonyms['low building density']),
            'moderate': random.choice(self.synonyms['moderate building density']),
            'high': random.choice(self.synonyms['high building density'])
        }
        return density_map.get(density, 'moderate building density')
    
    def _build_buildings_phrase(self, density):
        """Build buildings description"""
        if density == 'low':
            return f"{random.choice(self.synonyms['scattered'])} {random.choice(self.synonyms['residential structures'])}"
        elif density == 'high':
            return f"{random.choice(self.synonyms['tightly packed'])} {random.choice(self.synonyms['residential structures'])}"
        else:
            return f"{random.choice(self.synonyms['clusters'])} of {random.choice(self.synonyms['residential structures'])}"
    
    def _build_roads_phrase(self, paved, unpaved):
        """Build roads description"""
        if paved:
            return random.choice(self.synonyms['Paved roads'])
        elif unpaved:
            return random.choice(self.synonyms['Unpaved roads'])
        else:
            return 'Roads'
    
    def _build_road_action(self, paved):
        """Get road action verb"""
        if paved:
            actions = ['form a grid throughout', 'form a grid-like network in', 'form a loose grid in', 'create a grid pattern in']
            return random.choice(actions)
        else:
            actions = ['connect', 'link', 'traverse', 'cross']
            return random.choice(actions)
    
    def _build_road_detail(self):
        """Build road detail phrases for added description"""
        details = [
            "connecting the dispersed structures",
            "forming irregular pathways",
            "interspersed throughout the scene",
            "dividing the terrain into irregular plots",
            "traversing the landscape",
            "winding through the area",
            "cutting through the scene",
            "running through the terrain",
            "connecting scattered clusters",
            "forming a loose grid pattern"
        ]
        return random.choice(details)
    
    def _build_spatial_closure(self, density):
        """Build closure phrase with spatial detail"""
        if density == 'low':
            closures = [
                " Minimal development with scattered structures.",
                " Development is minimal and dispersed.",
                " Development is sparse and dispersed across the terrain.",
                ""
            ]
        else:
            closures = ["", ""]  # Less closure needed for higher density
        return random.choice(closures)
    
    def _build_vegetation_phrase(self, veg_level):
        """Build vegetation description"""
        if veg_level == 'sparse':
            return random.choice(self.synonyms['Sparse vegetation'])
        elif veg_level == 'moderate':
            return random.choice(self.synonyms['Moderate vegetation'])
        elif veg_level == 'dense':
            return random.choice(self.synonyms['Dense vegetation'])
        else:
            return 'Minimal vegetation'
    
    def _build_veg_detail(self, veg_level):
        """Build vegetation detail"""
        if veg_level == 'sparse':
            return random.choice(self.synonyms['scattered shrubs and bushes'])
        elif veg_level == 'moderate':
            return random.choice(self.synonyms['patches of trees'])
        elif veg_level == 'dense':
            return 'with thick tree coverage'
        else:
            return 'with scattered shrubs'
    
    def _build_terrain_phrase(self, veg_level):
        """Build terrain description"""
        if veg_level in ['sparse', None]:
            return random.choice(self.synonyms['predominantly bareland'])
        else:
            return random.choice(self.synonyms['bare, sandy terrain'])
    
    def _build_water_components(self, has_coastal):
        """Build water-related phrases - integrated into FIRST sentence"""
        if has_coastal:
            return {
                'water_position': random.choice(['adjacent to', 'near']),
                'shoreline_desc': 'developed shoreline',
                'shore_position': random.choice(self.synonyms['shoreline']),
                'water_s1': '',  # Coastal templates already have "with water body"
                'spatial_closure': ''
            }
        else:
            return {
                'water_s1': '',  # No water mention
                'water_position': '',
                'shoreline_desc': '',
                'shore_position': '',
                'spatial_closure': ''
            }
    
    def _build_veg_terrain_phrase(self, veg_level, is_coastal=False):
        """Build combined vegetation + terrain phrase (Sentence 3 style)"""
        if is_coastal:
            # Coastal: "Minimal vegetation with predominantly water and built environment"
            if veg_level == 'dense':
                return "Moderate vegetation with predominantly water and built environment"
            elif veg_level == 'moderate':
                return f"{random.choice(self.synonyms['Moderate vegetation'])} with water and developed area"
            else:
                return "Minimal vegetation with predominantly water and built environment"
        else:
            # Rural/Urban: "Sparse vegetation on bareland"
            if veg_level == 'dense':
                return f"{random.choice(self.synonyms['Dense vegetation'])} throughout area"
            elif veg_level == 'moderate':
                veg = random.choice(self.synonyms['Moderate vegetation'])
                terrain = random.choice(self.synonyms['bare, sandy terrain'])
                return f"{veg} on {terrain}"
            elif veg_level == 'sparse':
                veg = random.choice(self.synonyms['Sparse vegetation'])
                detail = random.choice(self.synonyms['scattered shrubs and bushes'])
                terrain = random.choice(self.synonyms['predominantly bareland'])
                return f"{veg} consisting of {detail} on {terrain}"
            else:
                return f"Minimal vegetation on {random.choice(self.synonyms['predominantly bareland'])}"
    
    def generate(self, features):
        """Generate diverse prompts using actual training data structure (2-3 sentences)
        
        Args:
            features: dict with keys:
                - scene: 'rural' or 'urban'
                - density: 'low', 'moderate', 'high'
                - coastal_water: bool
                - veg_level: 'sparse', 'moderate', 'dense', or None
                - paved_roads: bool
                - unpaved_roads: bool
        """
        # Determine template type
        scene_type = self._get_scene_type(features['scene'], features.get('coastal_water', False))
        is_coastal = scene_type == 'coastal'
        
        # Select random template
        template = random.choice(self.templates[scene_type])
        
        # Build all components
        scene = self._build_scene_phrase(features['scene'])
        density = self._build_density_phrase(features.get('density', 'moderate'))
        buildings = self._build_buildings_phrase(features.get('density', 'moderate'))
        roads = self._build_roads_phrase(features.get('paved_roads', False), features.get('unpaved_roads', False))
        roads_lower = roads.lower()
        road_action = self._build_road_action(features.get('paved_roads', False))
        road_detail = self._build_road_detail()
        veg_lower = self._build_vegetation_phrase(features.get('veg_level')).lower()
        terrain = self._build_terrain_phrase(features.get('veg_level'))
        building_location = random.choice(['structures', 'buildings', 'settlements', 'dispersed dwellings', 'residential clusters'])
        closure_phrase = random.choice([
            'Minimal development with scattered structures',
            'Development is minimal and dispersed',
            'Development is sparse and dispersed across the terrain',
            'indicating a dry environment'
        ])
        
        # Build combined vegetation + terrain phrase (sentence 3)
        veg_terrain = self._build_veg_terrain_phrase(features.get('veg_level'), is_coastal)
        
        # Water components (integrated into sentence 1)
        water_components = self._build_water_components(
            features.get('coastal_water', False)
        )
        
        # Fill template with all components
        prompt = template.format(
            scene=scene,
            density=density,
            buildings=buildings,
            roads=roads,
            roads_lower=roads_lower,
            road_action=road_action,
            road_detail=road_detail,
            veg_lower=veg_lower,
            terrain=terrain,
            veg_terrain=veg_terrain,
            building_location=building_location,
            closure_phrase=closure_phrase,
            **water_components
        )
        
        # Apply synonym substitution for additional diversity
        prompt = self._substitute(prompt)
        
        return prompt

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_dataset(
    # Water features
    coastal,
    # Vegetation features (simplified to match training data)
    veg_sparse, veg_moderate, veg_dense,
    # Road features
    paved, unpaved, highways,
    # Building features
    residential, commercial, industrial,
    # Generation settings
    count, scene, density, steps, guidance, height, seed, neg_prompt
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
        
        # Build feature dictionary for template generator
        # Determine vegetation level
        veg_level = None
        if veg_dense:
            veg_level = 'dense'
        elif veg_moderate:
            veg_level = 'moderate'
        elif veg_sparse:
            veg_level = 'sparse'
        
        feature_dict = {
            'scene': scene,
            'density': density,
            'coastal_water': coastal,
            'veg_level': veg_level,
            'paved_roads': paved,
            'unpaved_roads': unpaved or highways  # highways are paved roads
        }
        
        yield f"📝 Generating {count} prompts...", ""
        
        # Generate prompts (each call creates diversity through templates + synonyms + randomization)
        prompts = []
        for i in range(int(count)):
            prompt = template_generator.generate(feature_dict)
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
                "resolution": f"{int(height)}x{int(height)}",
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
                    width=int(height),  # Use height for both (square images)
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
        
        final_status = f"""Generation Complete!

Summary:
  - Generated: {successful}/{len(prompts)} images
  - Failed: {failed}
  - Resolution: {height}x{height} px
  - Steps: {steps} | Guidance: {guidance}

Output Directory:
  {exp_dir}

Files:
  - {successful} images in /images/
  - prompts.txt (all prompts)
  - metadata.json (full config)

You can now download the entire folder or individual images."""
        
        yield final_status, exp_dir
        
    except Exception as e:
        error_msg = f"""ERROR OCCURRED:

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
                coastal = gr.Checkbox(label="Coastal Water", value=True)
                gr.Markdown("""
                <div style='background: #eff6ff; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem; color: #1e40af;'>
                    💡 <strong>Note:</strong> Coastal water requires sparse vegetation only (no moderate/dense) and no highways, based on training data patterns.
                </div>
                """)
                # Hidden - not in training data
                rivers = gr.Checkbox(visible=False)
                lakes = gr.Checkbox(visible=False)
                ponds = gr.Checkbox(visible=False)
            
            gr.Markdown("### Vegetation Coverage")
            with gr.Group():
                veg_sparse = gr.Checkbox(label="Sparse (scattered shrubs)", value=True)
                veg_moderate = gr.Checkbox(label="Moderate (patches of trees)")
                veg_dense = gr.Checkbox(label="Dense (forests/agriculture)")
                gr.Markdown("""
                <div style='background: #fef3c7; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem; color: #92400e;'>
                    ⚠️ <strong>Constraint:</strong> Moderate/Dense vegetation cannot be combined with coastal water (incompatible with training data).
                </div>
                """)
                # Hidden - simplified from original
                forest = gr.Checkbox(visible=False)
                sparse_trees = gr.Checkbox(visible=False)
                grass = gr.Checkbox(visible=False)
                ag_fields = gr.Checkbox(visible=False)
            
            gr.Markdown("### Roads & Infrastructure")
            with gr.Group():
                paved = gr.Checkbox(label="Paved Roads", value=True)
                unpaved = gr.Checkbox(label="Unpaved/Dirt Roads", value=True)
                highways = gr.Checkbox(label="Highways")
                gr.Markdown("""
                <div style='background: #fef3c7; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem; color: #92400e;'>
                    ⚠️ <strong>Constraint:</strong> Highways cannot be combined with coastal water, dense vegetation, or high density (0% in training data).
                </div>
                """)
            
            gr.Markdown("### Buildings")
            with gr.Group():
                residential = gr.Checkbox(label="Residential", value=True)
                commercial = gr.Checkbox(label="Commercial")
                industrial = gr.Checkbox(label="Industrial")
                gr.Markdown("""
                <div style='background: #fef3c7; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem; color: #92400e;'>
                    ⚠️ <strong>Constraint:</strong> Commercial/Industrial buildings cannot be combined with coastal water (0% in training data).
                </div>
                """)
        
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
                ["rural", "urban"],
                value="rural",
                label="Scene Type (rural = arid landscape, urban = settlement)"
            )
            
            density = gr.Dropdown(
                ["low", "moderate", "high"],
                value="moderate",
                label="Building Density"
            )
            gr.Markdown("""
            <div style='background: #fef3c7; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem; color: #92400e;'>
                ⚠️ <strong>Constraint:</strong> High density cannot be combined with coastal water, rural scenes, or sparse vegetation (0% in training data).
            </div>
            """)
            
            height = gr.Radio([768, 1024], value=1024, label="Image Size")
            
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
    
    # ========================================================================
    # CONSTRAINT LOGIC - Based on Training Data Analysis
    # ========================================================================
    # Comprehensive rules extracted from comprehensive_rules.py analysis:
    # - Coastal: 3/10 (30%), Sparse veg: 10/10 (100%), Dense veg: 0/10 (0%)
    # - Highways: 0/10 (0%), High density: 0/10 (0%)
    # - Commercial: 0/10 (0%), Industrial: 0/10 (0%)
    
    def apply_coastal_constraints(coastal_enabled):
        """When coastal water is selected, disable ALL incompatible options."""
        if coastal_enabled:
            return {
                veg_moderate: gr.update(value=False, interactive=False),
                veg_dense: gr.update(value=False, interactive=False),
                highways: gr.update(value=False, interactive=False),
                commercial: gr.update(value=False, interactive=False),
                industrial: gr.update(value=False, interactive=False),
                density: gr.update(value="low" if density.value == "high" else density.value)
            }
        else:
            return {
                veg_moderate: gr.update(interactive=True),
                veg_dense: gr.update(interactive=True),
                highways: gr.update(interactive=True),
                commercial: gr.update(interactive=True),
                industrial: gr.update(interactive=True),
                density: gr.update()
            }
    
    def apply_vegetation_constraints(veg_mod, veg_dense):
        """When moderate/dense vegetation is selected, disable coastal water."""
        # Moderate veg: very rare with coastal (1/10)
        # Dense veg: NEVER with coastal, urban, or highways (0/10)
        if veg_mod or veg_dense:
            updates = {coastal: gr.update(value=False, interactive=False)}
            if veg_dense:
                # Dense veg also blocks highways
                updates[highways] = gr.update(value=False, interactive=False)
            return updates
        else:
            return {
                coastal: gr.update(interactive=True),
                highways: gr.update(interactive=True)
            }
    
    def apply_highway_constraints(highway_enabled):
        """When highways are selected, disable coastal, dense veg, and high density."""
        if highway_enabled:
            return {
                coastal: gr.update(value=False, interactive=False),
                veg_dense: gr.update(value=False, interactive=False),
                commercial: gr.update(value=False, interactive=False),
                density: gr.update(value="low" if density.value == "high" else density.value)
            }
        else:
            return {
                coastal: gr.update(interactive=True),
                veg_dense: gr.update(interactive=True),
                commercial: gr.update(interactive=True),
                density: gr.update()
            }
    
    def apply_density_constraints(density_value):
        """When high density is selected, disable coastal, rural, and sparse veg."""
        if density_value == "high":
            return {
                coastal: gr.update(value=False, interactive=False),
                highways: gr.update(value=False, interactive=False),
                veg_sparse: gr.update(value=False, interactive=False),
                scene: gr.update(value="urban" if scene.value == "rural" else scene.value)
            }
        else:
            return {
                coastal: gr.update(interactive=True),
                highways: gr.update(interactive=True),
                veg_sparse: gr.update(interactive=True),
                scene: gr.update()
            }
    
    def apply_building_type_constraints(comm, ind):
        """When commercial/industrial selected, disable coastal only."""
        if comm or ind:
            return gr.update(value=False, interactive=False)
        else:
            return gr.update(interactive=True)
    
    def apply_scene_constraints(scene_value):
        """When rural/urban selected, apply scene-specific constraints."""
        if scene_value == "rural":
            # Rural cannot have high density
            return {
                density: gr.update(value="low" if density.value == "high" else density.value)
            }
        else:  # urban
            # Urban cannot have dense vegetation
            return {
                veg_dense: gr.update(value=False, interactive=False)
            }
    
    # Wire up ALL constraint listeners
    coastal.change(
        fn=apply_coastal_constraints,
        inputs=[coastal],
        outputs=[veg_moderate, veg_dense, highways, commercial, industrial, density]
    )
    
    veg_moderate.change(
        fn=apply_vegetation_constraints,
        inputs=[veg_moderate, veg_dense],
        outputs=[coastal, highways]
    )
    
    veg_dense.change(
        fn=apply_vegetation_constraints,
        inputs=[veg_moderate, veg_dense],
        outputs=[coastal, highways]
    )
    
    highways.change(
        fn=apply_highway_constraints,
        inputs=[highways],
        outputs=[coastal, veg_dense, commercial, density]
    )
    
    density.change(
        fn=apply_density_constraints,
        inputs=[density],
        outputs=[coastal, highways, veg_sparse, scene]
    )
    
    commercial.change(
        fn=apply_building_type_constraints,
        inputs=[commercial, industrial],
        outputs=[coastal]
    )
    
    industrial.change(
        fn=apply_building_type_constraints,
        inputs=[commercial, industrial],
        outputs=[coastal]
    )
    
    scene.change(
        fn=apply_scene_constraints,
        inputs=[scene],
        outputs=[density, veg_dense]
    )
    
    # Wire up generation - NEW: Simplified inputs matching training data
    gen_btn.click(
        fn=generate_dataset,
        inputs=[
            coastal,  # Water features (coastal only)
            veg_sparse, veg_moderate, veg_dense,  # Vegetation levels
            paved, unpaved, highways,  # Roads
            residential, commercial, industrial,  # Buildings
            count, scene, density, steps, guidance, height, seed, neg_prompt  # Settings
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