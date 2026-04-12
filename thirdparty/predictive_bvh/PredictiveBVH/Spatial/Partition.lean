-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.Formula

-- ============================================================================
-- PARTITION NODE VOCABULARY
--
-- 3D partition node types for SAH-BVH construction.
-- Every split node stores a `parent` BoundingBox as the Hilbert cell volume for RDO.
-- ============================================================================

/-- 3D BVH partition node vocabulary covering 2-, 3-, 4-, 8-way splits and leaf.
    Every split node stores a `parent` BoundingBox (the Hilbert cell AABB) for RDO cost. -/
inductive PartitionNode where
  | none_split (data : LeafData)
  | horz       (parent : BoundingBox) (top bot : EClassId)      -- Y-axis 2-way
  | vert       (parent : BoundingBox) (left right : EClassId)   -- X-axis 2-way
  | depth      (parent : BoundingBox) (front back : EClassId)   -- Z-axis 2-way
  | horz_a     (parent : BoundingBox) (tl tr bot : EClassId)    -- XY T-shape: top split X, bottom full
  | vert_b     (parent : BoundingBox) (left tr br : EClassId)   -- XY T-shape: left full, right split Y
  | xz_a       (parent : BoundingBox) (fl fr back : EClassId)   -- XZ T-shape: front split X, back full
  | horz_4     (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 Y-strips
  | vert_4     (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 X-strips
  | depth_4    (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 Z-strips
  | oct        (parent : BoundingBox) (s1 s2 s3 s4 s5 s6 s7 s8 : EClassId)
  deriving Inhabited, Repr

-- ============================================================================
-- SAH COST MODEL
-- ============================================================================

/-- Predictive SAH cost for a leaf: surface area × traversal cost + predictive expansion.
    This is the cost used by the E-graph to select optimal partitions. -/
def predictiveSAH (data : LeafData) : Int :=
  bvhTraversalCost * surfaceArea data.bounds

/-- Full SAH cost evaluator for a partition node; returns `none` when any child EClassId is out of bounds.
    Traversal cost = surfaceArea(parent); leaf cost = predictiveSAH. -/
def evalNodeCost? (node : PartitionNode) (classes : Array EClass) : Option Int :=
  match node with
  | .none_split d => some (predictiveSAH d)
  | .horz p t b =>
    if hT : t < classes.size ∧ hB : b < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[t]!'.minCost + classes[b]!'.minCost)
    else none
  | .vert p l r =>
    if hL : l < classes.size ∧ hR : r < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[l]!'.minCost + classes[r]!'.minCost)
    else none
  | .depth p f b =>
    if hF : f < classes.size ∧ hB : b < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[f]!'.minCost + classes[b]!'.minCost)
    else none
  | .horz_a p tl tr b =>
    if hTL : tl < classes.size ∧ hTR : tr < classes.size ∧ hB : b < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[tl]!'.minCost + classes[tr]!'.minCost + classes[b]!'.minCost)
    else none
  | .vert_b p l tr br =>
    if hL : l < classes.size ∧ hTR : tr < classes.size ∧ hBR : br < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[l]!'.minCost + classes[tr]!'.minCost + classes[br]!'.minCost)
    else none
  | .xz_a p fl fr b =>
    if hFL : fl < classes.size ∧ hFR : fr < classes.size ∧ hB : b < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[fl]!'.minCost + classes[fr]!'.minCost + classes[b]!'.minCost)
    else none
  | .horz_4 p s1 s2 s3 s4 =>
    if h1 : s1 < classes.size ∧ h2 : s2 < classes.size ∧ h3 : s3 < classes.size ∧ h4 : s4 < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[s1]!'.minCost + classes[s2]!'.minCost +
            classes[s3]!'.minCost + classes[s4]!'.minCost)
    else none
  | .vert_4 p s1 s2 s3 s4 =>
    if h1 : s1 < classes.size ∧ h2 : s2 < classes.size ∧ h3 : s3 < classes.size ∧ h4 : s4 < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[s1]!'.minCost + classes[s2]!'.minCost +
            classes[s3]!'.minCost + classes[s4]!'.minCost)
    else none
  | .depth_4 p s1 s2 s3 s4 =>
    if h1 : s1 < classes.size ∧ h2 : s2 < classes.size ∧ h3 : s3 < classes.size ∧ h4 : s4 < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[s1]!'.minCost + classes[s2]!'.minCost +
            classes[s3]!'.minCost + classes[s4]!'.minCost)
    else none
  | .oct p s1 s2 s3 s4 s5 s6 s7 s8 =>
    if h1 : s1 < classes.size ∧ h2 : s2 < classes.size ∧ h3 : s3 < classes.size ∧ h4 : s4 < classes.size ∧
       h5 : s5 < classes.size ∧ h6 : s6 < classes.size ∧ h7 : s7 < classes.size ∧ h8 : s8 < classes.size then
      some (bvhTraversalCost * surfaceArea p + classes[s1]!'.minCost + classes[s2]!'.minCost +
            classes[s3]!'.minCost + classes[s4]!'.minCost + classes[s5]!'.minCost + classes[s6]!'.minCost +
            classes[s7]!'.minCost + classes[s8]!'.minCost)
    else none
