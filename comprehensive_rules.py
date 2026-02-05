"""
Comprehensive analysis of ALL feature constraints from training data
"""

import json
import re

# Load training data
with open('image_descriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Analyzing {len(data)} training samples for ALL feature combinations...")
print("=" * 80)

# Initialize counters for ALL features
features = {
    # Water
    'coastal': 0,
    'river': 0,
    'lake': 0,
    'pond': 0,
    'water_body': 0,
    
    # Vegetation
    'sparse_veg': 0,
    'moderate_veg': 0,
    'dense_veg': 0,
    'forest': 0,
    'agriculture': 0,
    'trees': 0,
    
    # Roads
    'paved': 0,
    'unpaved': 0,
    'highway': 0,
    'dirt_road': 0,
    
    # Buildings
    'residential': 0,
    'commercial': 0,
    'industrial': 0,
    
    # Density
    'low_density': 0,
    'moderate_density': 0,
    'high_density': 0,
    
    # Scene types
    'rural': 0,
    'urban': 0,
    'coastal_scene': 0,
}

# Combination counters
combinations = {}

def detect_features(desc):
    """Extract all features from a description."""
    desc_lower = desc.lower()
    found = {}
    
    # Water
    found['coastal'] = 'coastal' in desc_lower or 'shoreline' in desc_lower or 'beach' in desc_lower
    found['river'] = 'river' in desc_lower
    found['lake'] = 'lake' in desc_lower
    found['pond'] = 'pond' in desc_lower
    found['water_body'] = 'water' in desc_lower and not found['coastal']
    
    # Vegetation - be more specific
    found['sparse_veg'] = 'sparse' in desc_lower and 'vegetation' in desc_lower
    found['moderate_veg'] = 'moderate' in desc_lower and ('vegetation' in desc_lower or 'cover' in desc_lower)
    found['dense_veg'] = 'dense' in desc_lower and ('vegetation' in desc_lower or 'cover' in desc_lower)
    found['forest'] = 'forest' in desc_lower
    found['agriculture'] = 'agriculture' in desc_lower or 'cultivated' in desc_lower or 'farm' in desc_lower
    found['trees'] = 'tree' in desc_lower
    
    # Roads
    found['paved'] = 'paved' in desc_lower
    found['unpaved'] = 'unpaved' in desc_lower or 'dirt' in desc_lower
    found['highway'] = 'highway' in desc_lower
    
    # Buildings
    found['residential'] = 'residential' in desc_lower
    found['commercial'] = 'commercial' in desc_lower
    found['industrial'] = 'industrial' in desc_lower
    
    # Density
    found['low_density'] = 'low' in desc_lower and 'density' in desc_lower
    found['moderate_density'] = 'moderate' in desc_lower and 'density' in desc_lower
    found['high_density'] = 'high' in desc_lower and 'density' in desc_lower
    
    # Scene
    found['rural'] = 'rural' in desc_lower
    found['urban'] = 'urban' in desc_lower
    found['coastal_scene'] = found['coastal']
    
    return found

# Analyze each sample
all_feature_sets = []
for item in data:
    desc = item['description']
    found = detect_features(desc)
    all_feature_sets.append(found)
    
    # Count individual features
    for feature, present in found.items():
        if present:
            features[feature] += 1

# Print individual feature statistics
print("\n📊 INDIVIDUAL FEATURE OCCURRENCE:")
print("=" * 80)

print("\n🌊 WATER FEATURES:")
print(f"  Coastal:      {features['coastal']:3d} ({features['coastal']/len(data)*100:.1f}%)")
print(f"  River:        {features['river']:3d} ({features['river']/len(data)*100:.1f}%)")
print(f"  Lake:         {features['lake']:3d} ({features['lake']/len(data)*100:.1f}%)")
print(f"  Pond:         {features['pond']:3d} ({features['pond']/len(data)*100:.1f}%)")
print(f"  Water body:   {features['water_body']:3d} ({features['water_body']/len(data)*100:.1f}%)")

print("\n🌿 VEGETATION FEATURES:")
print(f"  Sparse:       {features['sparse_veg']:3d} ({features['sparse_veg']/len(data)*100:.1f}%)")
print(f"  Moderate:     {features['moderate_veg']:3d} ({features['moderate_veg']/len(data)*100:.1f}%)")
print(f"  Dense:        {features['dense_veg']:3d} ({features['dense_veg']/len(data)*100:.1f}%)")
print(f"  Forest:       {features['forest']:3d} ({features['forest']/len(data)*100:.1f}%)")
print(f"  Agriculture:  {features['agriculture']:3d} ({features['agriculture']/len(data)*100:.1f}%)")
print(f"  Trees:        {features['trees']:3d} ({features['trees']/len(data)*100:.1f}%)")

print("\n🛣️ ROAD FEATURES:")
print(f"  Paved:        {features['paved']:3d} ({features['paved']/len(data)*100:.1f}%)")
print(f"  Unpaved:      {features['unpaved']:3d} ({features['unpaved']/len(data)*100:.1f}%)")
print(f"  Highway:      {features['highway']:3d} ({features['highway']/len(data)*100:.1f}%)")

print("\n🏢 BUILDING FEATURES:")
print(f"  Residential:  {features['residential']:3d} ({features['residential']/len(data)*100:.1f}%)")
print(f"  Commercial:   {features['commercial']:3d} ({features['commercial']/len(data)*100:.1f}%)")
print(f"  Industrial:   {features['industrial']:3d} ({features['industrial']/len(data)*100:.1f}%)")

print("\n🏘️ DENSITY:")
print(f"  Low:          {features['low_density']:3d} ({features['low_density']/len(data)*100:.1f}%)")
print(f"  Moderate:     {features['moderate_density']:3d} ({features['moderate_density']/len(data)*100:.1f}%)")
print(f"  High:         {features['high_density']:3d} ({features['high_density']/len(data)*100:.1f}%)")

print("\n🗺️ SCENE TYPES:")
print(f"  Rural:        {features['rural']:3d} ({features['rural']/len(data)*100:.1f}%)")
print(f"  Urban:        {features['urban']:3d} ({features['urban']/len(data)*100:.1f}%)")
print(f"  Coastal:      {features['coastal_scene']:3d} ({features['coastal_scene']/len(data)*100:.1f}%)")

# Now check ALL critical combinations
print("\n" + "=" * 80)
print("🔍 CRITICAL FEATURE COMBINATIONS:")
print("=" * 80)

def check_combo(feature_sets, feat1, feat2):
    """Check how many times two features appear together."""
    count = sum(1 for fs in feature_sets if fs[feat1] and fs[feat2])
    return count

# Check all meaningful combinations
combos_to_check = [
    # Coastal combinations
    ('coastal', 'sparse_veg', "Coastal + Sparse Vegetation"),
    ('coastal', 'moderate_veg', "Coastal + Moderate Vegetation"),
    ('coastal', 'dense_veg', "Coastal + Dense Vegetation"),
    ('coastal', 'highway', "Coastal + Highway"),
    ('coastal', 'high_density', "Coastal + High Density"),
    ('coastal', 'moderate_density', "Coastal + Moderate Density"),
    ('coastal', 'low_density', "Coastal + Low Density"),
    ('coastal', 'commercial', "Coastal + Commercial"),
    ('coastal', 'industrial', "Coastal + Industrial"),
    ('coastal', 'residential', "Coastal + Residential"),
    
    # Dense vegetation combinations
    ('dense_veg', 'coastal', "Dense Vegetation + Coastal"),
    ('dense_veg', 'urban', "Dense Vegetation + Urban"),
    ('dense_veg', 'rural', "Dense Vegetation + Rural"),
    ('dense_veg', 'highway', "Dense Vegetation + Highway"),
    ('dense_veg', 'high_density', "Dense Vegetation + High Density"),
    
    # Moderate vegetation combinations
    ('moderate_veg', 'coastal', "Moderate Vegetation + Coastal"),
    ('moderate_veg', 'urban', "Moderate Vegetation + Urban"),
    ('moderate_veg', 'high_density', "Moderate Vegetation + High Density"),
    
    # Highway combinations
    ('highway', 'coastal', "Highway + Coastal"),
    ('highway', 'urban', "Highway + Urban"),
    ('highway', 'rural', "Highway + Rural"),
    ('highway', 'high_density', "Highway + High Density"),
    ('highway', 'commercial', "Highway + Commercial"),
    
    # High density combinations
    ('high_density', 'coastal', "High Density + Coastal"),
    ('high_density', 'rural', "High Density + Rural"),
    ('high_density', 'sparse_veg', "High Density + Sparse Vegetation"),
    
    # Commercial/Industrial
    ('commercial', 'coastal', "Commercial + Coastal"),
    ('commercial', 'rural', "Commercial + Rural"),
    ('commercial', 'high_density', "Commercial + High Density"),
    ('industrial', 'coastal', "Industrial + Coastal"),
    ('industrial', 'rural', "Industrial + Rural"),
    ('industrial', 'high_density', "Industrial + High Density"),
    ('industrial', 'residential', "Industrial + Residential"),
]

print("\n🚫 CONSTRAINT RULES (Combinations that NEVER appear):")
print("-" * 80)

never_combos = []
rare_combos = []

for feat1, feat2, label in combos_to_check:
    count = check_combo(all_feature_sets, feat1, feat2)
    percentage = count / len(data) * 100
    
    if count == 0:
        never_combos.append((label, count))
        print(f"❌ {label:45s} : {count:2d} occurrences (NEVER)")
    elif count <= 1:
        rare_combos.append((label, count))
        print(f"⚠️  {label:45s} : {count:2d} occurrences (VERY RARE - {percentage:.1f}%)")

# Print summary
print("\n" + "=" * 80)
print("📋 SUMMARY OF MISSING CONSTRAINTS:")
print("=" * 80)

print(f"\n❌ NEVER OCCURS ({len(never_combos)} combinations):")
for label, count in never_combos:
    print(f"   - {label}")

if rare_combos:
    print(f"\n⚠️  VERY RARE ({len(rare_combos)} combinations):")
    for label, count in rare_combos:
        print(f"   - {label} ({count} occurrence)")

# Extract actionable rules
print("\n" + "=" * 80)
print("✅ ACTIONABLE UI CONSTRAINTS TO IMPLEMENT:")
print("=" * 80)

print("\n1. When COASTAL WATER is selected:")
print("   - ❌ Disable: Moderate Vegetation")
print("   - ❌ Disable: Dense Vegetation")
print("   - ❌ Disable: Highways")
print("   - ❌ Disable: High Density")
if check_combo(all_feature_sets, 'coastal', 'commercial') == 0:
    print("   - ❌ Disable: Commercial Buildings")
if check_combo(all_feature_sets, 'coastal', 'industrial') == 0:
    print("   - ❌ Disable: Industrial Buildings")

print("\n2. When MODERATE VEGETATION is selected:")
print("   - ❌ Disable: Coastal Water")
if check_combo(all_feature_sets, 'moderate_veg', 'high_density') == 0:
    print("   - ❌ Disable: High Density")

print("\n3. When DENSE VEGETATION is selected:")
print("   - ❌ Disable: Coastal Water")
print("   - ❌ Disable: Urban Scene")
print("   - ❌ Disable: Highways")
if check_combo(all_feature_sets, 'dense_veg', 'high_density') == 0:
    print("   - ❌ Disable: High Density")

print("\n4. When HIGHWAYS are selected:")
print("   - ❌ Disable: Coastal Water")
if check_combo(all_feature_sets, 'highway', 'high_density') == 0:
    print("   - ❌ Disable: High Density")

print("\n5. When HIGH DENSITY is selected:")
print("   - ❌ Disable: Coastal Water")
print("   - ❌ Disable: Rural Scene")
if check_combo(all_feature_sets, 'high_density', 'sparse_veg') == 0:
    print("   - ❌ Disable: Sparse Vegetation")

print("\n6. When COMMERCIAL BUILDINGS are selected:")
if check_combo(all_feature_sets, 'commercial', 'coastal') == 0:
    print("   - ❌ Disable: Coastal Water")
if check_combo(all_feature_sets, 'commercial', 'rural') == 0:
    print("   - ❌ Disable: Rural Scene")

print("\n7. When INDUSTRIAL BUILDINGS are selected:")
if check_combo(all_feature_sets, 'industrial', 'coastal') == 0:
    print("   - ❌ Disable: Coastal Water")
if check_combo(all_feature_sets, 'industrial', 'rural') == 0:
    print("   - ❌ Disable: Rural Scene")

print("\n" + "=" * 80)
print("⚠️  CURRENT IMPLEMENTATION STATUS:")
print("=" * 80)
print("✅ IMPLEMENTED: Coastal + Moderate/Dense Veg")
print("✅ IMPLEMENTED: Coastal + Highways")
print("❓ MISSING: High Density constraints")
print("❓ MISSING: Commercial/Industrial constraints")
print("❓ MISSING: Dense Vegetation constraints")
print("=" * 80)
