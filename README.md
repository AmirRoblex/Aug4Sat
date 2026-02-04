# Aug4Sat

**AI-Powered Synthetic Satellite Imagery Generation**

Aug4Sat generates realistic synthetic satellite imagery using AI. Control land cover types, urban features, and environmental elements to create custom satellite image datasets.

## Features

- **Land Cover Control**: Water bodies, vegetation types, infrastructure, and urban development zones
- **Generation Parameters**: Customizable resolution, batch processing, scene variety control
- **Advanced Settings**: Fine-tune with steps, guidance scale, seed control, and negative prompts
- **User Interface**: Clean Gradio interface with real-time progress monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AmirRoblex/Aug4Sat.git
cd Aug4Sat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the interface through the provided URL (local and public Gradio link)

## Usage

1. Select land cover features (water bodies, vegetation, roads, buildings)
2. Set generation parameters (image count, resolution, quality settings)
3. Configure advanced options (steps, guidance scale, seed)
4. Click generate to create your dataset
5. Download generated images from the output directory

## Use Cases

- Machine learning training datasets
- Urban planning visualizations
- Environmental monitoring simulations
- Remote sensing algorithm development
- Educational demonstrations

## Technical Stack

- SDXL (Stable Diffusion XL) with custom LoRA
- Qwen 2.5-3B for prompt generation
- Gradio for web interface
- PyTorch for GPU acceleration

## Author

**SAM Qureshi**
- GitHub: [@AmirRoblex](https://github.com/AmirRoblex)
- Website: [samqureshi.me](https://samqureshi.me/)
- LinkedIn: [sam-qureshi](https://www.linkedin.com/in/sam-qureshi/)

## Demo Video

Demo video will be added soon.

## License

Open source for educational and research purposes.
