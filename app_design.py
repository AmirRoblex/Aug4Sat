# app_design.py
import gradio as gr
import base64

# Encode logo to base64
def get_logo_base64():
    with open("aug4sat.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

# Mock function for testing UI
def mock_generate(
    rivers, lakes, coastal, ponds,
    forest, sparse_trees, grass, ag_fields,
    paved, unpaved, highways,
    residential, commercial, industrial,
    count, scene, density, steps, guidance, width, height, seed, neg_prompt
):
    """Mock generation for UI testing"""
    
    import time
    time.sleep(2)  # Simulate processing
    
    status = f"""GENERATION COMPLETE (MOCK)

Generated: {count} / {count} images
Resolution: {width} x {height} px
Steps: {steps} | Guidance: {guidance}

Output: /mock/path/satellite_dataset_20260203_123456"""
    
    path = "/mock/path/satellite_dataset_20260203_123456"
    
    return status, path

css = """
.gradio-container {
    max-width: 1600px !important;
}

/* Header styling */
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

.dark .app-title {
    color: #f9fafb;
}

.dark .app-subtitle {
    color: #9ca3af;
}

.theme-toggle-wrapper {
    position: absolute;
    top: 1rem;
    right: 2rem;
}

/* Remove ALL spinners from ALL inputs */
input::-webkit-inner-spin-button,
input::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
    display: none !important;
}
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
    display: none !important;
}
input[type=text]::-webkit-inner-spin-button,
input[type=text]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
    display: none !important;
}
input[type=number] {
    -moz-appearance: textfield !important;
}
input[type=text] {
    -moz-appearance: textfield !important;
}
input {
    -moz-appearance: textfield !important;
}

/* Dark theme */
body.dark, .dark .gradio-container, .dark {
    background-color: #1f2937 !important;
}

.dark .gradio-container,
.dark * {
    color: #f3f4f6 !important;
}

.dark input,
.dark textarea,
.dark select {
    background-color: #374151 !important;
    border-color: #4b5563 !important;
    color: #f3f4f6 !important;
}

.dark .block,
.dark .form {
    background-color: #374151 !important;
    border-color: #4b5563 !important;
}

.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    margin-top: 2rem;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 0.875rem;
}
.dark .footer {
    border-top-color: #4b5563;
    color: #9ca3af;
}
.footer a {
    color: #2563eb;
    text-decoration: none;
    margin: 0 0.5rem;
}
.dark .footer a {
    color: #60a5fa;
}
.footer a:hover {
    text-decoration: underline;
}
"""

js_theme = """
function(theme) {
    if (theme === 'Dark') {
        document.body.classList.add('dark');
        document.querySelector('.gradio-container').classList.add('dark');
    } else {
        document.body.classList.remove('dark');
        document.querySelector('.gradio-container').classList.remove('dark');
    }
    return theme;
}
"""

with gr.Blocks(css=css, title="Aug4Sat", theme=gr.themes.Default()) as demo:
    
    logo_b64 = get_logo_base64()
    
    # Header with logo
    gr.HTML(f"""
    <div class="header-wrapper">
        <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="Aug4Sat Logo">
        <div class="title-section">
            <h1 class="app-title">Aug4Sat</h1>
            <p class="app-subtitle">AI-Powered Synthetic Satellite Imagery Generation</p>
        </div>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        # Left Column - All Features
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
        
        # Right Column - Settings Only
        with gr.Column(scale=1):
            
            gr.Markdown("### Generation Settings")
            
            with gr.Row():
                count = gr.Number(label="Images", value=10, minimum=1, maximum=100, precision=0)
                steps = gr.Number(label="Steps", value=25, minimum=20, maximum=50, precision=0)
            
            with gr.Row():
                guidance = gr.Number(label="Guidance", value=7.5, minimum=5.0, maximum=15.0)
                seed = gr.Textbox(label="Seed", placeholder="Optional")
            
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
    
    # Status and Generate button span full width below
    status_box = gr.Textbox(
        label="Status",
        lines=6,
        value="Ready to generate."
    )
    
    path_box = gr.Textbox(label="Output Path", lines=2)
    
    gen_btn = gr.Button("Generate Dataset", variant="primary", size="lg")
    
    # Footer
    gr.HTML("""
    <div class="footer">
        Made by <strong>SAM Qureshi</strong><br>
        <a href="https://samqureshi.me/" target="_blank">Website</a> | 
        <a href="https://www.linkedin.com/in/sam-qureshi/" target="_blank">LinkedIn</a> | 
        <a href="mailto:sam.qureshi.grad@gmail.com">Email</a>
    </div>
    """)
    
    # Wire up with MOCK function
    gen_btn.click(
        fn=mock_generate,
        inputs=[
            rivers, lakes, coastal, ponds,
            forest, sparse_trees, grass, ag_fields,
            paved, unpaved, highways,
            residential, commercial, industrial,
            count, scene, density, steps, guidance, width, height, seed, neg_prompt
        ],
        outputs=[status_box, path_box]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861, inbrowser=False)