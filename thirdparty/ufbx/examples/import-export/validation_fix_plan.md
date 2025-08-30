# FBX Validation Failure Fix Plan

## Analysis Summary
Out of 371 test files, 152 (41%) are failing validation due to data loss during roundtrip.

## Root Cause Categories

### 1. Animation Data Loss (Major Issue)
**Files affected**: Files with "anim", "tangent", "extrapolation", "keyframe", "motionbuilder_*"
**Issues identified**:
- Animation curves only copying 3 components (X,Y,Z) but some properties need more
- Missing animation-to-node connections 
- Not preserving extrapolation modes and tangent types
- Animation layer blending and weights not copied
- Missing animation props like stepped tangents, weighted tangents

### 2. Transform/Pivot Data Loss (Major Issue)  
**Files affected**: Files with "pivot", "transform", "inherit", "scale"
**Issues identified**:
- Only copying basic local_transform, missing:
  - Pre/post rotation matrices
  - Rotation and scaling pivots
  - Transform inheritance modes
  - Constraint connections

### 3. Skin Deformer Issues (Medium Issue)
**Files affected**: Files with "skin", "instanced_skin", "transformed_skin" 
**Issues identified**:
- Incomplete skin weight copying
- Missing proper cluster binding setup
- Not handling all skin deformer connection types

### 4. Missing Element Types (Medium Issue)
**Files affected**: Various files with specific element types
**Issues identified**:  
- Lights and cameras not being copied
- User properties and custom attributes missing
- Audio elements not handled

### 5. Material/Texture Connection Issues (Minor Issue)
**Files affected**: Files with complex material setups
**Issues identified**:
- Some material properties not fully copied
- Texture layer blending modes missing

## Fix Implementation Priority

### Phase 1: Animation System Fixes (High Impact)
1. **Fix animation curve component handling**
   - Dynamically determine component count based on property type
   - Copy all components, not just X,Y,Z

2. **Add animation-node connections**
   - Connect animation values to proper node properties
   - Preserve animation connection types (translation, rotation, scaling, etc.)

3. **Copy animation metadata** 
   - Extrapolation modes (constant, linear, cyclic, etc.)
   - Tangent types and weights
   - Layer blending modes and weights

### Phase 2: Transform System Fixes (High Impact)
1. **Copy complete transform data**
   - Pre/post rotation matrices
   - Rotation and scaling pivots  
   - Transform inheritance modes

2. **Add constraint support**
   - Copy constraint connections between nodes
   - Preserve constraint weights and settings

### Phase 3: Deformer System Fixes (Medium Impact)
1. **Improve skin deformer copying**
   - Better cluster connection handling
   - Complete weight data transfer
   - Handle instanced/shared skins

2. **Add blend deformer details**
   - Copy blend shape target data
   - Preserve in-between targets

### Phase 4: Element Type Coverage (Medium Impact)
1. **Add missing element types**
   - Light copying with all properties
   - Camera copying with lens settings
   - Audio clip handling

2. **User properties and metadata**
   - Custom attributes on nodes
   - User-defined properties

## Expected Impact
- **Phase 1**: Should fix ~60-80 files (animation-related failures)
- **Phase 2**: Should fix ~40-60 files (transform-related failures) 
- **Phase 3**: Should fix ~15-25 files (deformer-related failures)
- **Phase 4**: Should fix ~5-15 files (misc element failures)

**Total expected**: Reduce failures from 152 to ~20-40 files (73-87% improvement)

## Implementation Strategy
1. Fix one category at a time
2. Test fixes against specific failing files for that category
3. Run full validation suite after each phase
4. Measure improvement and identify remaining issues
