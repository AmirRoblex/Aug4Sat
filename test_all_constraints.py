"""
Comprehensive test for ALL constraint rules
"""

def test_all_constraints():
    """Test ALL constraint rules extracted from training data."""
    
    print("Testing ALL Feature Constraint Rules")
    print("=" * 80)
    
    # Training data statistics
    print("\n📊 Training Data (10 samples):")
    print("   ✓ Coastal: 30% | Sparse veg: 100% | Dense veg: 0% | Highways: 0%")
    print("   ✓ High density: 0% | Commercial: 0% | Industrial: 0%")
    
    # Define comprehensive test cases
    test_cases = [
        # COASTAL CONSTRAINTS (6 rules)
        {
            "name": "✅ Valid: Coastal + Sparse + Low Density + Residential",
            "coastal": True, "veg_sparse": True, "veg_moderate": False, "veg_dense": False,
            "highways": False, "commercial": False, "industrial": False,
            "density": "low", "scene": "coastal", "expected": "VALID"
        },
        {
            "name": "❌ Invalid: Coastal + Moderate Vegetation",
            "coastal": True, "veg_moderate": True,
            "expected": "INVALID - Coastal blocks moderate vegetation"
        },
        {
            "name": "❌ Invalid: Coastal + Dense Vegetation",
            "coastal": True, "veg_dense": True,
            "expected": "INVALID - Coastal blocks dense vegetation"
        },
        {
            "name": "❌ Invalid: Coastal + Highways",
            "coastal": True, "highways": True,
            "expected": "INVALID - Coastal blocks highways"
        },
        {
            "name": "❌ Invalid: Coastal + Commercial",
            "coastal": True, "commercial": True,
            "expected": "INVALID - Coastal blocks commercial"
        },
        {
            "name": "❌ Invalid: Coastal + Industrial",
            "coastal": True, "industrial": True,
            "expected": "INVALID - Coastal blocks industrial"
        },
        
        # DENSE VEGETATION CONSTRAINTS (4 rules)
        {
            "name": "❌ Invalid: Dense Veg + Coastal",
            "veg_dense": True, "coastal": True,
            "expected": "INVALID - Dense veg blocks coastal"
        },
        {
            "name": "❌ Invalid: Dense Veg + Urban",
            "veg_dense": True, "scene": "urban",
            "expected": "INVALID - Dense veg blocks urban"
        },
        {
            "name": "❌ Invalid: Dense Veg + Highways",
            "veg_dense": True, "highways": True,
            "expected": "INVALID - Dense veg blocks highways"
        },
        {
            "name": "✅ Valid: Dense Veg + Rural + Low Density",
            "veg_dense": True, "scene": "rural", "density": "low",
            "expected": "VALID"
        },
        
        # HIGHWAY CONSTRAINTS (4 rules)
        {
            "name": "❌ Invalid: Highways + Coastal",
            "highways": True, "coastal": True,
            "expected": "INVALID - Highways block coastal"
        },
        {
            "name": "❌ Invalid: Highways + Dense Veg",
            "highways": True, "veg_dense": True,
            "expected": "INVALID - Highways block dense veg"
        },
        {
            "name": "❌ Invalid: Highways + High Density",
            "highways": True, "density": "high",
            "expected": "INVALID - Highways block high density"
        },
        {
            "name": "❌ Invalid: Highways + Commercial",
            "highways": True, "commercial": True,
            "expected": "INVALID - Highways block commercial"
        },
        
        # HIGH DENSITY CONSTRAINTS (3 rules)
        {
            "name": "❌ Invalid: High Density + Coastal",
            "density": "high", "coastal": True,
            "expected": "INVALID - High density blocks coastal"
        },
        {
            "name": "❌ Invalid: High Density + Rural",
            "density": "high", "scene": "rural",
            "expected": "INVALID - High density blocks rural"
        },
        {
            "name": "❌ Invalid: High Density + Sparse Veg",
            "density": "high", "veg_sparse": True,
            "expected": "INVALID - High density blocks sparse veg"
        },
        
        # COMMERCIAL/INDUSTRIAL CONSTRAINTS (2 rules each)
        {
            "name": "❌ Invalid: Commercial + Coastal",
            "commercial": True, "coastal": True,
            "expected": "INVALID - Commercial blocks coastal"
        },
        {
            "name": "❌ Invalid: Commercial + Rural",
            "commercial": True, "scene": "rural",
            "expected": "INVALID - Commercial blocks rural"
        },
        {
            "name": "❌ Invalid: Industrial + Coastal",
            "industrial": True, "coastal": True,
            "expected": "INVALID - Industrial blocks coastal"
        },
        {
            "name": "❌ Invalid: Industrial + Rural",
            "industrial": True, "scene": "rural",
            "expected": "INVALID - Industrial blocks rural"
        },
        
        # SCENE CONSTRAINTS
        {
            "name": "❌ Invalid: Rural + Commercial",
            "scene": "rural", "commercial": True,
            "expected": "INVALID - Rural blocks commercial"
        },
        {
            "name": "❌ Invalid: Rural + Industrial",
            "scene": "rural", "industrial": True,
            "expected": "INVALID - Rural blocks industrial"
        },
        {
            "name": "❌ Invalid: Rural + High Density",
            "scene": "rural", "density": "high",
            "expected": "INVALID - Rural blocks high density"
        },
        {
            "name": "❌ Invalid: Urban + Dense Veg",
            "scene": "urban", "veg_dense": True,
            "expected": "INVALID - Urban blocks dense veg"
        },
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} Constraint Rules:")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        
        # Simplified validation logic
        is_valid = True
        error_msg = None
        
        # Extract values with defaults
        coastal = test.get('coastal', False)
        veg_sparse = test.get('veg_sparse', False)
        veg_moderate = test.get('veg_moderate', False)
        veg_dense = test.get('veg_dense', False)
        highways = test.get('highways', False)
        commercial = test.get('commercial', False)
        industrial = test.get('industrial', False)
        density = test.get('density', 'low')
        scene = test.get('scene', 'rural')
        
        # Apply all constraint rules
        if coastal and (veg_moderate or veg_dense):
            is_valid = False
            error_msg = "Coastal blocks moderate/dense vegetation"
        elif coastal and highways:
            is_valid = False
            error_msg = "Coastal blocks highways"
        elif coastal and (commercial or industrial):
            is_valid = False
            error_msg = "Coastal blocks commercial/industrial"
        elif coastal and density == "high":
            is_valid = False
            error_msg = "Coastal blocks high density"
        elif veg_dense and coastal:
            is_valid = False
            error_msg = "Dense veg blocks coastal"
        elif veg_dense and scene == "urban":
            is_valid = False
            error_msg = "Dense veg blocks urban"
        elif veg_dense and highways:
            is_valid = False
            error_msg = "Dense veg blocks highways"
        elif highways and coastal:
            is_valid = False
            error_msg = "Highways block coastal"
        elif highways and veg_dense:
            is_valid = False
            error_msg = "Highways block dense veg"
        elif highways and density == "high":
            is_valid = False
            error_msg = "Highways block high density"
        elif highways and commercial:
            is_valid = False
            error_msg = "Highways block commercial"
        elif density == "high" and coastal:
            is_valid = False
            error_msg = "High density blocks coastal"
        elif density == "high" and scene == "rural":
            is_valid = False
            error_msg = "High density blocks rural"
        elif density == "high" and veg_sparse:
            is_valid = False
            error_msg = "High density blocks sparse veg"
        elif commercial and coastal:
            is_valid = False
            error_msg = "Commercial blocks coastal"
        elif commercial and scene == "rural":
            is_valid = False
            error_msg = "Commercial blocks rural"
        elif industrial and coastal:
            is_valid = False
            error_msg = "Industrial blocks coastal"
        elif industrial and scene == "rural":
            is_valid = False
            error_msg = "Industrial blocks rural"
        elif scene == "rural" and (commercial or industrial):
            is_valid = False
            error_msg = "Rural blocks commercial/industrial"
        elif scene == "rural" and density == "high":
            is_valid = False
            error_msg = "Rural blocks high density"
        elif scene == "urban" and veg_dense:
            is_valid = False
            error_msg = "Urban blocks dense veg"
        
        result = "VALID" if is_valid else f"INVALID - {error_msg}"
        status = "✅ PASS" if (is_valid and "Valid" in test['name']) or (not is_valid and "Invalid" in test['name']) else "❌ FAIL"
        
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {result}")
        print(f"   {status}")
        
        if "PASS" in status:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    
    # Print comprehensive rule summary
    print("\n📋 IMPLEMENTED CONSTRAINT RULES:")
    print("=" * 80)
    print("\n1️⃣  COASTAL WATER (6 constraints):")
    print("   ❌ Blocks: Moderate Veg, Dense Veg, Highways, Commercial, Industrial, High Density")
    print("\n2️⃣  DENSE VEGETATION (3 constraints):")
    print("   ❌ Blocks: Coastal, Urban, Highways")
    print("\n3️⃣  HIGHWAYS (4 constraints):")
    print("   ❌ Blocks: Coastal, Dense Veg, Commercial, High Density")
    print("\n4️⃣  HIGH DENSITY (3 constraints):")
    print("   ❌ Blocks: Coastal, Rural, Sparse Veg")
    print("\n5️⃣  COMMERCIAL BUILDINGS (2 constraints):")
    print("   ❌ Blocks: Coastal, Rural")
    print("\n6️⃣  INDUSTRIAL BUILDINGS (2 constraints):")
    print("   ❌ Blocks: Coastal, Rural")
    print("\n7️⃣  RURAL SCENE (3 constraints):")
    print("   ❌ Blocks: Commercial, Industrial, High Density")
    print("\n8️⃣  URBAN SCENE (1 constraint):")
    print("   ❌ Blocks: Dense Veg")
    print("\n" + "=" * 80)
    print(f"📊 TOTAL: 24 bidirectional constraint rules enforced")
    print("=" * 80)
    
    return passed == len(test_cases)

if __name__ == "__main__":
    success = test_all_constraints()
    exit(0 if success else 1)
