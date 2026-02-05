"""
FINAL COMPREHENSIVE ANALYSIS - Real Training Data (322 samples)
"""

import json
from collections import Counter

with open('image_descriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"{'='*80}")
print(f"COMPREHENSIVE TRAINING DATA ANALYSIS - {len(data)} SAMPLES")
print(f"{'='*80}")

# Feature detection function
def analyze_features(desc):
    desc_lower = desc.lower()
    features = {
        # Water
        'coastal': 'coastal' in desc_lower or 'shoreline' in desc_lower or 'beach' in desc_lower or 'shore' in desc_lower,
        'water': 'water' in desc_lower,
        
        # Vegetation - be very specific
        'sparse_veg': 'sparse' in desc_lower and ('vegetation' in desc_lower or 'veg' in desc_lower or 'shrub' in desc_lower),
        'moderate_veg': 'moderate' in desc_lower and ('vegetation' in desc_lower or 'veg' in desc_lower or 'cover' in desc_lower),
        'dense_veg': 'dense' in desc_lower and ('vegetation' in desc_lower or 'veg' in desc_lower or 'cover' in desc_lower or 'forest' in desc_lower),
        'forest': 'forest' in desc_lower,
        'agriculture': 'agriculture' in desc_lower or 'cultivated' in desc_lower or 'crop' in desc_lower or 'farm' in desc_lower,
        
        # Roads
        'paved': 'paved' in desc_lower,
        'unpaved': 'unpaved' in desc_lower or 'dirt' in desc_lower or 'track' in desc_lower,
        'highway': 'highway' in desc_lower,
        
        # Buildings
        'residential': 'residential' in desc_lower or 'housing' in desc_lower or 'houses' in desc_lower,
        'commercial': 'commercial' in desc_lower,
        'industrial': 'industrial' in desc_lower or 'factory' in desc_lower or 'warehouse' in desc_lower,
        'mixed_use': 'mixed-use' in desc_lower or 'mixed use' in desc_lower,
        
        # Density
        'low_density': 'low' in desc_lower and 'density' in desc_lower,
        'moderate_density': 'moderate' in desc_lower and 'density' in desc_lower,
        'high_density': 'high' in desc_lower and 'density' in desc_lower,
        
        # Scene types
        'rural': 'rural' in desc_lower,
        'urban': 'urban' in desc_lower or 'city' in desc_lower or 'town' in desc_lower,
        'arid': 'arid' in desc_lower,
        'bareland': 'bareland' in desc_lower or 'bare land' in desc_lower,
    }
    return features

# Analyze all samples
all_features = [analyze_features(item['description']) for item in data]

# Count individual features
feature_counts = {}
for feature in all_features[0].keys():
    feature_counts[feature] = sum(1 for f in all_features if f[feature])

# Print individual feature statistics
print(f"\n{'='*80}")
print("INDIVIDUAL FEATURE OCCURRENCE")
print(f"{'='*80}")

print("\n🌊 WATER FEATURES:")
print(f"  Coastal:          {feature_counts['coastal']:4d} ({feature_counts['coastal']/len(data)*100:5.1f}%)")
print(f"  Water (any):      {feature_counts['water']:4d} ({feature_counts['water']/len(data)*100:5.1f}%)")

print("\n🌿 VEGETATION:")
print(f"  Sparse:           {feature_counts['sparse_veg']:4d} ({feature_counts['sparse_veg']/len(data)*100:5.1f}%)")
print(f"  Moderate:         {feature_counts['moderate_veg']:4d} ({feature_counts['moderate_veg']/len(data)*100:5.1f}%)")
print(f"  Dense:            {feature_counts['dense_veg']:4d} ({feature_counts['dense_veg']/len(data)*100:5.1f}%)")
print(f"  Forest:           {feature_counts['forest']:4d} ({feature_counts['forest']/len(data)*100:5.1f}%)")
print(f"  Agriculture:      {feature_counts['agriculture']:4d} ({feature_counts['agriculture']/len(data)*100:5.1f}%)")

print("\n🛣️ ROADS:")
print(f"  Paved:            {feature_counts['paved']:4d} ({feature_counts['paved']/len(data)*100:5.1f}%)")
print(f"  Unpaved:          {feature_counts['unpaved']:4d} ({feature_counts['unpaved']/len(data)*100:5.1f}%)")
print(f"  Highway:          {feature_counts['highway']:4d} ({feature_counts['highway']/len(data)*100:5.1f}%)")

print("\n🏢 BUILDINGS:")
print(f"  Residential:      {feature_counts['residential']:4d} ({feature_counts['residential']/len(data)*100:5.1f}%)")
print(f"  Commercial:       {feature_counts['commercial']:4d} ({feature_counts['commercial']/len(data)*100:5.1f}%)")
print(f"  Industrial:       {feature_counts['industrial']:4d} ({feature_counts['industrial']/len(data)*100:5.1f}%)")
print(f"  Mixed-use:        {feature_counts['mixed_use']:4d} ({feature_counts['mixed_use']/len(data)*100:5.1f}%)")

print("\n🏘️ DENSITY:")
print(f"  Low:              {feature_counts['low_density']:4d} ({feature_counts['low_density']/len(data)*100:5.1f}%)")
print(f"  Moderate:         {feature_counts['moderate_density']:4d} ({feature_counts['moderate_density']/len(data)*100:5.1f}%)")
print(f"  High:             {feature_counts['high_density']:4d} ({feature_counts['high_density']/len(data)*100:5.1f}%)")

print("\n🗺️ SCENE TYPES:")
print(f"  Rural:            {feature_counts['rural']:4d} ({feature_counts['rural']/len(data)*100:5.1f}%)")
print(f"  Urban:            {feature_counts['urban']:4d} ({feature_counts['urban']/len(data)*100:5.1f}%)")
print(f"  Arid:             {feature_counts['arid']:4d} ({feature_counts['arid']/len(data)*100:5.1f}%)")
print(f"  Bareland:         {feature_counts['bareland']:4d} ({feature_counts['bareland']/len(data)*100:5.1f}%)")

# Check critical combinations
print(f"\n{'='*80}")
print("CRITICAL FEATURE COMBINATIONS")
print(f"{'='*80}")

def count_combo(feat1, feat2):
    return sum(1 for f in all_features if f[feat1] and f[feat2])

combinations = [
    ('coastal', 'sparse_veg', "Coastal + Sparse Vegetation"),
    ('coastal', 'moderate_veg', "Coastal + Moderate Vegetation"),
    ('coastal', 'dense_veg', "Coastal + Dense Vegetation"),
    ('coastal', 'highway', "Coastal + Highway"),
    ('coastal', 'high_density', "Coastal + High Density"),
    ('coastal', 'commercial', "Coastal + Commercial"),
    ('coastal', 'industrial', "Coastal + Industrial"),
    ('coastal', 'residential', "Coastal + Residential"),
    ('coastal', 'rural', "Coastal + Rural"),
    ('coastal', 'urban', "Coastal + Urban"),
    
    ('dense_veg', 'coastal', "Dense Veg + Coastal"),
    ('dense_veg', 'urban', "Dense Veg + Urban"),
    ('dense_veg', 'highway', "Dense Veg + Highway"),
    ('dense_veg', 'high_density', "Dense Veg + High Density"),
    
    ('moderate_veg', 'coastal', "Moderate Veg + Coastal"),
    ('moderate_veg', 'urban', "Moderate Veg + Urban"),
    ('moderate_veg', 'high_density', "Moderate Veg + High Density"),
    
    ('highway', 'coastal', "Highway + Coastal"),
    ('highway', 'dense_veg', "Highway + Dense Veg"),
    ('highway', 'high_density', "Highway + High Density"),
    ('highway', 'rural', "Highway + Rural"),
    ('highway', 'urban', "Highway + Urban"),
    ('highway', 'commercial', "Highway + Commercial"),
    
    ('high_density', 'coastal', "High Density + Coastal"),
    ('high_density', 'rural', "High Density + Rural"),
    ('high_density', 'sparse_veg', "High Density + Sparse Veg"),
    
    ('commercial', 'coastal', "Commercial + Coastal"),
    ('commercial', 'rural', "Commercial + Rural"),
    ('commercial', 'urban', "Commercial + Urban"),
    
    ('industrial', 'coastal', "Industrial + Coastal"),
    ('industrial', 'rural', "Industrial + Rural"),
    ('industrial', 'urban', "Industrial + Urban"),
    
    ('rural', 'commercial', "Rural + Commercial"),
    ('rural', 'industrial', "Rural + Industrial"),
    ('rural', 'high_density', "Rural + High Density"),
    
    ('urban', 'dense_veg', "Urban + Dense Veg"),
]

print("\n📊 COMBINATION ANALYSIS:")
never_combos = []
rare_combos = []
common_combos = []

for feat1, feat2, label in combinations:
    count = count_combo(feat1, feat2)
    pct = count / len(data) * 100
    
    if count == 0:
        never_combos.append((label, count, pct))
        print(f"❌ {label:40s} : {count:4d} ({pct:5.1f}%) - NEVER")
    elif pct < 1.0:
        rare_combos.append((label, count, pct))
        print(f"⚠️  {label:40s} : {count:4d} ({pct:5.1f}%) - VERY RARE")
    elif pct < 5.0:
        print(f"📉 {label:40s} : {count:4d} ({pct:5.1f}%) - RARE")
    else:
        common_combos.append((label, count, pct))
        print(f"✅ {label:40s} : {count:4d} ({pct:5.1f}%) - COMMON")

# Print actionable constraints
print(f"\n{'='*80}")
print("RECOMMENDED UI CONSTRAINTS (0% or <1% occurrence)")
print(f"{'='*80}")

print("\n🚫 NEVER OCCURS (should disable):")
for label, count, pct in never_combos:
    print(f"   - {label}")

print(f"\n⚠️  VERY RARE (<1%, consider warning):")
for label, count, pct in rare_combos:
    print(f"   - {label} ({count} occurrences)")

print(f"\n✅ COMMON (allow freely):")
for label, count, pct in common_combos[:10]:  # Show first 10
    print(f"   - {label} ({count} occurrences, {pct:.1f}%)")

print(f"\n{'='*80}")
print("FINAL CONSTRAINT RECOMMENDATIONS")
print(f"{'='*80}")
print("\nBased on 0% occurrence in 322 samples:")

constraint_rules = {}
for label, count, pct in never_combos:
    parts = label.split(' + ')
    if len(parts) == 2:
        feat1, feat2 = parts
        if feat1 not in constraint_rules:
            constraint_rules[feat1] = []
        constraint_rules[feat1].append(feat2)

for feature, blocked in sorted(constraint_rules.items()):
    print(f"\n{feature}:")
    for block in blocked:
        print(f"   ❌ Disable: {block}")
