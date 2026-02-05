# test_prompts.py - Test prompt generation and compare with training data
import json
import sys
from collections import Counter
import random

# ============================================================================
# TEMPLATE GENERATOR (Copied from app.py to avoid dependencies)
# ============================================================================

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
        
        # Spatial positioning phrases for water bodies
        self.water_spatial_positions = [
            "visible in the lower-left",
            "visible on the left",
            "visible in the upper portion of the scene",
            "visible near the central cluster",
            "running along the left edge",
            "visible in the lower portion",
            "meandering through the upper area",
            "visible on one side",
            "adjacent to the settlement",
            "bordering the developed area"
        ]
        
        # Road detail phrases
        self.road_details = [
            "connecting the dispersed structures",
            "forming irregular pathways",
            "interspersed throughout the scene",
            "dividing the terrain",
            "traversing the landscape",
            "winding through the area",
            "cutting through the scene",
            "running through the terrain"
        ]
        
        # Closure phrases
        self.closure_phrases = [
            "Minimal development with scattered structures",
            "Development is minimal and dispersed",
            "Development is sparse and dispersed across the terrain",
            "indicating a dry environment"
        ]
        
        # Building location descriptors
        self.building_locations = [
            "structures",
            "buildings",
            "settlements",
            "dispersed dwellings",
            "residential clusters"
        ]
    
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
    
    def _build_terrain_phrase(self, veg_level):
        """Build terrain description"""
        if veg_level in ['sparse', None]:
            return random.choice(self.synonyms['predominantly bareland'])
        else:
            return random.choice(self.synonyms['bare, sandy terrain'])
    
    def _build_water_components(self, has_coastal, has_small):
        """Build water-related phrases - integrated into FIRST sentence with spatial positioning"""
        if has_coastal:
            return {
                'water_position': random.choice(['adjacent to', 'near']),
                'shoreline_desc': 'developed shoreline',
                'shore_position': random.choice(self.synonyms['shoreline']),
                'water_s1': '',  # Coastal templates already have "with water body"
                'spatial_closure': ''
            }
        elif has_small:
            # Small water goes in sentence 1 as scene-defining feature with spatial location
            positions = [
                "visible in the lower-left",
                "visible on the left",
                "visible in the upper portion of the scene",
                "visible near the central cluster",
                "running along the left edge",
                "visible in the lower portion",
                "meandering through the upper area",
                "visible on one side",
                "adjacent to the settlement",
                "bordering the developed area"
            ]
            position = random.choice(positions)
            small_phrases = [
                f", with a small water body {position}",
                f"; a small water body is {position}",
                f", with a narrow water body {position}",
            ]
            return {
                'water_s1': random.choice(small_phrases),
                'water_position': '',
                'shoreline_desc': '',
                'shore_position': '',
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
        """Generate diverse prompts using actual training data structure (2-3 sentences)"""
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
        veg_terrain_coastal = veg_terrain  # Same for coastal
        
        # Water components (integrated into sentence 1)
        water_components = self._build_water_components(
            features.get('coastal_water', False),
            features.get('small_water', False)
        )
        
        # Ensure spatial_closure is in water_components (from method above)
        # No need to add separately since it's already in water_components
        
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
            veg_terrain_coastal=veg_terrain_coastal,
            building_location=building_location,
            closure_phrase=closure_phrase,
            **water_components
        )
        
        # Apply synonym substitution for additional diversity
        prompt = self._substitute(prompt)
        
        return prompt

# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# All possible setting combinations to test
TEST_CONFIGS = [
    # Rural configurations
    {
        'name': 'Rural Sparse No Water',
        'features': {
            'scene': 'rural',
            'density': 'low',
            'coastal_water': False,
            'small_water': False,
            'veg_level': 'sparse',
            'paved_roads': False,
            'unpaved_roads': True
        }
    },
    {
        'name': 'Rural Sparse Small Water',
        'features': {
            'scene': 'rural',
            'density': 'low',
            'coastal_water': False,
            'small_water': True,
            'veg_level': 'sparse',
            'paved_roads': False,
            'unpaved_roads': True
        }
    },
    {
        'name': 'Rural Moderate Sparse Veg',
        'features': {
            'scene': 'rural',
            'density': 'moderate',
            'coastal_water': False,
            'small_water': False,
            'veg_level': 'sparse',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    {
        'name': 'Rural Moderate Small Water',
        'features': {
            'scene': 'rural',
            'density': 'moderate',
            'coastal_water': False,
            'small_water': True,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': True
        }
    },
    # Urban configurations
    {
        'name': 'Urban Low Density',
        'features': {
            'scene': 'urban',
            'density': 'low',
            'coastal_water': False,
            'small_water': False,
            'veg_level': 'sparse',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    {
        'name': 'Urban Moderate Density',
        'features': {
            'scene': 'urban',
            'density': 'moderate',
            'coastal_water': False,
            'small_water': False,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': True
        }
    },
    {
        'name': 'Urban High Density',
        'features': {
            'scene': 'urban',
            'density': 'high',
            'coastal_water': False,
            'small_water': False,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    {
        'name': 'Urban Moderate Small Water',
        'features': {
            'scene': 'urban',
            'density': 'moderate',
            'coastal_water': False,
            'small_water': True,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    # Coastal configurations
    {
        'name': 'Coastal Low Density',
        'features': {
            'scene': 'urban',
            'density': 'low',
            'coastal_water': True,
            'small_water': False,
            'veg_level': 'sparse',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    {
        'name': 'Coastal Moderate Density',
        'features': {
            'scene': 'urban',
            'density': 'moderate',
            'coastal_water': True,
            'small_water': False,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': False
        }
    },
    {
        'name': 'Coastal High Density',
        'features': {
            'scene': 'urban',
            'density': 'high',
            'coastal_water': True,
            'small_water': False,
            'veg_level': 'moderate',
            'paved_roads': True,
            'unpaved_roads': True
        }
    },
]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_training_prompts():
    """Load actual training prompts from image_descriptions.json if available"""
    try:
        with open('image_descriptions.json', 'r') as f:
            data = json.load(f)
            return [item['description'] for item in data]
    except FileNotFoundError:
        print("⚠️  image_descriptions.json not found - skipping comparison")
        return []

def analyze_prompt_structure(prompts):
    """Analyze structural properties of prompts"""
    analysis = {
        'total_prompts': len(prompts),
        'unique_prompts': len(set(prompts)),
        'duplicate_count': len(prompts) - len(set(prompts)),
        'avg_length': sum(len(p) for p in prompts) / len(prompts),
        'sentence_counts': [],
        'avg_sentences': 0,
        'has_water': 0,
        'water_in_first_sentence': 0,
    }
    
    for prompt in prompts:
        # Count sentences
        sentences = prompt.count('.') + prompt.count('!') + prompt.count('?')
        analysis['sentence_counts'].append(sentences)
        
        # Check water mentions
        if 'water' in prompt.lower():
            analysis['has_water'] += 1
            # Check if water is in first sentence
            first_sentence = prompt.split('.')[0] if '.' in prompt else prompt
            if 'water' in first_sentence.lower():
                analysis['water_in_first_sentence'] += 1
    
    analysis['avg_sentences'] = sum(analysis['sentence_counts']) / len(analysis['sentence_counts'])
    
    return analysis

def extract_vocabulary(prompts):
    """Extract common words and phrases from prompts"""
    vocab = {
        'density_terms': [],
        'vegetation_terms': [],
        'road_terms': [],
        'building_terms': [],
        'water_terms': [],
        'terrain_terms': []
    }
    
    density_keywords = ['low', 'moderate', 'high', 'sparse', 'dense', 'minimal', 'tightly packed']
    veg_keywords = ['vegetation', 'shrubs', 'bushes', 'trees', 'bareland', 'bare']
    road_keywords = ['road', 'paved', 'unpaved', 'dirt', 'track', 'highway', 'grid']
    building_keywords = ['building', 'structure', 'residential', 'commercial', 'dwelling', 'house']
    water_keywords = ['water', 'coastal', 'shoreline', 'coast', 'body']
    terrain_keywords = ['bareland', 'terrain', 'arid', 'sandy', 'landscape']
    
    for prompt in prompts:
        prompt_lower = prompt.lower()
        for word in density_keywords:
            if word in prompt_lower:
                vocab['density_terms'].append(word)
        for word in veg_keywords:
            if word in prompt_lower:
                vocab['vegetation_terms'].append(word)
        for word in road_keywords:
            if word in prompt_lower:
                vocab['road_terms'].append(word)
        for word in building_keywords:
            if word in prompt_lower:
                vocab['building_terms'].append(word)
        for word in water_keywords:
            if word in prompt_lower:
                vocab['water_terms'].append(word)
        for word in terrain_keywords:
            if word in prompt_lower:
                vocab['terrain_terms'].append(word)
    
    return vocab

def compare_with_training(generated_prompts, training_prompts):
    """Compare generated prompts with training data"""
    print("\n" + "="*80)
    print("COMPARISON WITH TRAINING DATA")
    print("="*80)
    
    gen_analysis = analyze_prompt_structure(generated_prompts)
    train_analysis = analyze_prompt_structure(training_prompts)
    
    print(f"\n📊 STRUCTURAL COMPARISON:")
    print(f"  Training Data:")
    print(f"    - Avg length: {train_analysis['avg_length']:.0f} chars")
    print(f"    - Avg sentences: {train_analysis['avg_sentences']:.1f}")
    print(f"  Generated Data:")
    print(f"    - Avg length: {gen_analysis['avg_length']:.0f} chars")
    print(f"    - Avg sentences: {gen_analysis['avg_sentences']:.1f}")
    
    print(f"\n💧 WATER MENTIONS:")
    print(f"  Training Data:")
    print(f"    - Has water: {train_analysis['has_water']}/{train_analysis['total_prompts']} ({train_analysis['has_water']/train_analysis['total_prompts']*100:.1f}%)")
    print(f"    - Water in 1st sentence: {train_analysis['water_in_first_sentence']}/{train_analysis['has_water']}")
    print(f"  Generated Data:")
    print(f"    - Has water: {gen_analysis['has_water']}/{gen_analysis['total_prompts']} ({gen_analysis['has_water']/gen_analysis['total_prompts']*100:.1f}%)")
    print(f"    - Water in 1st sentence: {gen_analysis['water_in_first_sentence']}/{gen_analysis['has_water']}")
    
    # Vocabulary comparison
    gen_vocab = extract_vocabulary(generated_prompts)
    train_vocab = extract_vocabulary(training_prompts)
    
    print(f"\n📚 VOCABULARY COMPARISON:")
    
    for category in ['density_terms', 'vegetation_terms', 'road_terms', 'building_terms', 'water_terms', 'terrain_terms']:
        gen_counter = Counter(gen_vocab[category])
        train_counter = Counter(train_vocab[category])
        
        print(f"\n  {category.replace('_', ' ').title()}:")
        print(f"    Training: {dict(train_counter.most_common(5))}")
        print(f"    Generated: {dict(gen_counter.most_common(5))}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests and generate report"""
    print("="*80)
    print("PROMPT GENERATION TEST SUITE")
    print("="*80)
    
    # Initialize generator
    generator = TemplateVariantGenerator()
    
    # Load training prompts
    training_prompts = load_training_prompts()
    
    # Store all generated prompts for overall analysis
    all_generated_prompts = []
    
    # Test each configuration
    for config in TEST_CONFIGS:
        print(f"\n{'='*80}")
        print(f"TEST: {config['name']}")
        print(f"{'='*80}")
        print(f"Settings: {config['features']}")
        
        # Generate 10 prompts
        prompts = []
        for i in range(10):
            prompt = generator.generate(config['features'])
            prompts.append(prompt)
            all_generated_prompts.append(prompt)
        
        # Display prompts
        print(f"\n📝 Generated Prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")
        
        # Analyze this batch
        analysis = analyze_prompt_structure(prompts)
        
        print(f"\n📊 Analysis:")
        print(f"  - Unique prompts: {analysis['unique_prompts']}/10")
        print(f"  - Duplicates: {analysis['duplicate_count']}")
        print(f"  - Avg length: {analysis['avg_length']:.0f} chars")
        print(f"  - Avg sentences: {analysis['avg_sentences']:.1f}")
        
        if config['features'].get('small_water') or config['features'].get('coastal_water'):
            print(f"  - Water mentions: {analysis['has_water']}/10")
            print(f"  - Water in 1st sentence: {analysis['water_in_first_sentence']}/{analysis['has_water']}")
            
            if analysis['has_water'] < 10:
                print(f"  ⚠️  WARNING: {10 - analysis['has_water']} prompts missing water!")
        
        # Check for common issues
        issues = []
        for prompt in prompts:
            if '  ' in prompt:
                issues.append("Double spaces detected")
            if prompt.count('.') > 5:
                issues.append("Too many sentences")
            if len(prompt) < 100:
                issues.append("Very short prompt")
        
        if issues:
            print(f"\n⚠️  Issues Detected:")
            for issue in set(issues):
                print(f"    - {issue}")
    
    # Overall analysis
    print(f"\n\n{'='*80}")
    print("OVERALL ANALYSIS")
    print(f"{'='*80}")
    
    overall_analysis = analyze_prompt_structure(all_generated_prompts)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  - Total prompts: {overall_analysis['total_prompts']}")
    print(f"  - Unique prompts: {overall_analysis['unique_prompts']}")
    print(f"  - Duplicate rate: {overall_analysis['duplicate_count']/overall_analysis['total_prompts']*100:.1f}%")
    print(f"  - Avg length: {overall_analysis['avg_length']:.0f} chars")
    print(f"  - Avg sentences: {overall_analysis['avg_sentences']:.1f}")
    
    # Compare with training if available
    if training_prompts:
        compare_with_training(all_generated_prompts, training_prompts)
    
    # Final recommendations
    print(f"\n\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = []
    
    if overall_analysis['duplicate_count'] / overall_analysis['total_prompts'] > 0.1:
        recommendations.append("🔴 HIGH: Duplicate rate > 10% - Need more diversity mechanisms")
    
    if overall_analysis['avg_sentences'] < 2 or overall_analysis['avg_sentences'] > 4:
        recommendations.append(f"🟡 MEDIUM: Avg sentences ({overall_analysis['avg_sentences']:.1f}) not in 2-3 range")
    
    if training_prompts:
        train_analysis = analyze_prompt_structure(training_prompts)
        length_diff = abs(overall_analysis['avg_length'] - train_analysis['avg_length'])
        if length_diff > 50:
            recommendations.append(f"🟡 MEDIUM: Length differs from training by {length_diff:.0f} chars")
    
    # Check water integration
    water_configs = [c for c in TEST_CONFIGS if c['features'].get('small_water') or c['features'].get('coastal_water')]
    water_prompts = []
    for config in water_configs:
        for _ in range(10):
            water_prompts.append(generator.generate(config['features']))
    
    water_analysis = analyze_prompt_structure(water_prompts)
    if water_analysis['has_water'] < len(water_prompts):
        missing_pct = (len(water_prompts) - water_analysis['has_water']) / len(water_prompts) * 100
        recommendations.append(f"🔴 HIGH: {missing_pct:.0f}% of water prompts missing water keyword")
    
    if water_analysis['water_in_first_sentence'] < water_analysis['has_water'] * 0.8:
        recommendations.append(f"🟡 MEDIUM: Only {water_analysis['water_in_first_sentence']}/{water_analysis['has_water']} water mentions in 1st sentence")
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  ✅ All checks passed! Prompts look good.")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_tests()
