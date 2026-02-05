# Prompt Diversity Examples

This document shows examples of how the template generator creates diverse prompts from the same settings, using actual training data vocabulary.

## Example 1: Rural Arid with Low Density, No Water

**Settings:**
- Scene: rural
- Density: low
- Coastal Water: No
- Small Water: No
- Vegetation: sparse
- Roads: unpaved
- Buildings: residential

**Generated Prompts (3 variations):**

1. `Rural arid landscape with low building density and scattered residential structures. Unpaved dirt roads connect the structures. Sparse vegetation consisting of scattered shrubs and bushes on predominantly bareland. No visible water bodies.`

2. `Arid rural landscape with sparse building density and dispersed residential buildings. Dirt roads traverse the terrain, with Minimal vegetation with scattered shrubs on extensive bareland. No water bodies are visible.`

3. `Sparse rural settlement in an arid landscape with minimal building density and isolated dwellings. Unpaved tracks connecting the structures, with Limited vegetation coverage consisting of isolated shrubs on mostly bare. with no visible water bodies.`

## Example 2: Urban Moderate Density with Coastal Water

**Settings:**
- Scene: urban
- Density: moderate
- Coastal Water: Yes
- Small Water: No
- Vegetation: moderate
- Roads: paved
- Buildings: residential

**Generated Prompts (3 variations):**

1. `Coastal scene with a water body adjacent to a developed shoreline. moderate building density with clusters of residential structures along the shoreline. Paved roads form a grid-like network coastal structures. Moderate vegetation on bare, sandy terrain.`

2. `Coastal scene with a developed shoreline. moderate to high building density featuring groups of residential buildings. Paved roads visible along the coast. Vegetation coverage with patches of trees dry, sandy terrain.`

3. `Coastal urban settlement with moderate building density and clusters of houses near the shoreline. Paved roads form a network through the area. Moderate vegetation coverage on surrounding bare, arid land.`

## Example 3: Urban High Density, No Coastal Water, Small Water Body

**Settings:**
- Scene: urban
- Density: high
- Coastal Water: No
- Small Water: Yes
- Vegetation: moderate
- Roads: paved + unpaved
- Buildings: residential + commercial

**Generated Prompts (3 variations):**

1. `Urban settlement with high building density, featuring tightly packed residential structures. Paved roads form a loose grid structures. Moderate vegetation with patches of trees and shrubs bare, sandy terrain. A small water body is visible in the area.`

2. `Semi-urban settlement with dense building density and densely packed residential buildings. Paved road network form a grid throughout the area. Vegetation coverage interspersed with dry, sandy terrain. A small water body is visible in the area.`

3. `Urban residential settlement with tightly packed featuring compact dwellings. A paved road form an irregular grid through the area. Moderate vegetation coverage scattered among bareland. A small water body is visible in the area.`

## Diversity Mechanisms

The template generator achieves prompt diversity through **multiple layers**:

### 1. **Multiple Template Structures (3-4 per scene type)**
   - Different sentence patterns
   - Different ordering of features
   - Different connector phrases

### 2. **Synonym Substitution (10-20 per phrase)**
   Examples:
   - "Rural arid landscape" → "Rural arid landscape" / "Arid rural landscape" / "Sparse rural settlement in an arid landscape"
   - "low building density" → "low building density" / "sparse building density" / "minimal building density"
   - "scattered" → "scattered" / "dispersed" / "isolated"
   - "predominantly bareland" → "predominantly bareland" / "predominantly bare" / "extensive bareland" / "mostly bare"

### 3. **Phrase-Level Variations**
   - Roads: "Paved roads" / "A paved road" / "Paved road network"
   - Action verbs: "form a grid" / "form a grid-like network" / "form a loose grid"
   - Vegetation detail: "scattered shrubs and bushes" / "scattered shrubs" / "isolated shrubs"

### 4. **Random Selection at Multiple Levels**
   - Template selection (random)
   - Synonym selection (random for each phrase)
   - Component building (different logic paths)

### Expected Result for 100 Images with Same Settings:
- **0-5 exact duplicates** (extremely rare due to multiple randomization layers)
- **90+ unique prompts** using same vocabulary but different phrasing
- **All prompts use actual training data vocabulary** ensuring LoRA recognizes them

## Vocabulary Source

All phrases are extracted from actual LoRA training data (image_descriptions.json):
- ✅ "scattered shrubs and bushes"
- ✅ "predominantly bareland"
- ✅ "tightly packed residential structures"
- ✅ "unpaved dirt roads"
- ✅ "paved roads form a grid-like network"
- ✅ "coastal scene with a water body"
- ✅ "Rural arid landscape"
- ✅ "Urban settlement with moderate building density"

This ensures maximum compatibility with the LoRA while maintaining natural linguistic variation.
