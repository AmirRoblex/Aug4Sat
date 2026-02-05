# Training Data Analysis

## Features Present in Training Data:

### Water Bodies:
- ✅ Coastal water body (very common)
- ✅ Small water bodies (ponds, irrigation channels, drainage)
- ❌ Rivers (rare, maybe 2-3 instances)
- ❌ Lakes (not clearly present)

### Vegetation:
- ✅ Sparse vegetation (most common: "scattered shrubs and bushes")
- ✅ Moderate vegetation (patches of trees, shrubs)
- ✅ Dense vegetation (forests, agricultural fields with crops)
- ✅ Agricultural fields/plots

### Roads:
- ✅ Paved roads (very common)
- ✅ Unpaved roads/dirt roads/tracks (very common)
- ✅ Highways (mentioned several times)

### Buildings:
- ✅ Low building density (scattered, isolated structures)
- ✅ Moderate building density (clusters)
- ✅ High building density (tightly packed)
- ✅ Residential structures
- ✅ Commercial structures
- ✅ Industrial structures/facilities

### Scene Types:
- ✅ Rural arid landscape
- ✅ Urban settlement
- ✅ Coastal scene/urban settlement
- ✅ Mixed-use settlement
- ✅ Semi-urban/peri-urban

### Common Elements:
- Bareland (appears in 90%+ of descriptions)
- "No visible water bodies" (very common in rural scenes)

## Recommended UI Updates:
1. Keep coastal water, add small water bodies option
2. Remove rivers and lakes completely
3. Simplify vegetation to: sparse/moderate/dense
4. Keep roads as-is (paved/unpaved/highways)
5. Keep buildings with density options

## Template Strategy:
Use actual phrases from training data to ensure LoRA recognition
