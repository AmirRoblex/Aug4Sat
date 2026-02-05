import json

# Load training data
with open('image_descriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total training samples: {len(data)}\n")

# Coastal analysis
coastal = [d for d in data if 'oastal' in d['description']]
print(f"=== COASTAL WATER RULES ===")
print(f"Coastal scenes: {len(coastal)}")
print(f"Coastal + Sparse vegetation: {len([d for d in coastal if 'parse' in d['description']])}")
print(f"Coastal + Moderate vegetation: {len([d for d in coastal if 'oderate vegetation' in d['description']])}")
print(f"Coastal + Dense vegetation: {len([d for d in coastal if 'ense' in d['description'] and 'vegetation' in d['description']])}")
print(f"Coastal + Highways: {len([d for d in coastal if 'ighway' in d['description']])}")
print(f"Coastal + High density: {len([d for d in coastal if 'igh building density' in d['description']])}")

# Dense vegetation analysis  
dense = [d for d in data if 'ense vegetation' in d['description'] or 'ense forest' in d['description']]
print(f"\n=== DENSE VEGETATION RULES ===")
print(f"Dense vegetation scenes: {len(dense)}")
print(f"Dense veg + Coastal: {len([d for d in dense if 'oastal' in d['description']])}")
print(f"Dense veg + Rural: {len([d for d in dense if 'ural' in d['description']])}")
print(f"Dense veg + Urban: {len([d for d in dense if 'rban' in d['description']])}")

# Highway analysis
highway = [d for d in data if 'ighway' in d['description'].lower() or 'overpass' in d['description'].lower()]
print(f"\n=== HIGHWAY RULES ===")
print(f"Highway scenes: {len(highway)}")
print(f"Highway + Coastal: {len([d for d in highway if 'oastal' in d['description']])}")
print(f"Highway + Urban: {len([d for d in highway if 'rban' in d['description']])}")
print(f"Highway + High density: {len([d for d in highway if 'igh building density' in d['description']])}")

# Vegetation distribution
sparse = len([d for d in data if 'parse vegetation' in d['description']])
moderate = len([d for d in data if 'oderate vegetation' in d['description']])
dense_total = len(dense)

print(f"\n=== VEGETATION DISTRIBUTION ===")
print(f"Sparse: {sparse} ({sparse/len(data)*100:.1f}%)")
print(f"Moderate: {moderate} ({moderate/len(data)*100:.1f}%)")
print(f"Dense: {dense_total} ({dense_total/len(data)*100:.1f}%)")

# Density patterns
print(f"\n=== BUILDING DENSITY PATTERNS ===")
low = len([d for d in data if 'ow building density' in d['description']])
mod_density = len([d for d in data if 'oderate building density' in d['description']])
high = len([d for d in data if 'igh building density' in d['description']])
print(f"Low: {low} ({low/len(data)*100:.1f}%)")
print(f"Moderate: {mod_density} ({mod_density/len(data)*100:.1f}%)")
print(f"High: {high} ({high/len(data)*100:.1f}%)")

print(f"\n=== EXTRACTED RULES ===")
print("1. Coastal Water + Dense Vegetation: NEVER (0 occurrences)")
print("2. Coastal Water + Moderate Vegetation: RARE/NEVER")
print("3. Coastal Water + Highways: NEVER (0 occurrences)")
print("4. Coastal Water → Usually Sparse Vegetation only")
print("5. Dense Vegetation → Never Coastal")
print("6. Highways → Rare (2 occurrences), never with coastal")
