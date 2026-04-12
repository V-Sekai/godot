-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Spatial.Partition

-- ============================================================================
-- E-GRAPH SATURATION
--
-- Apply partition rewrite rules to find optimal (lowest-cost) partitions.
-- Uses worklist algorithm to reach fixed point.
-- ============================================================================

/-- E-graph saturation state. -/
structure SaturateState where
  nodes     : Array PartitionNode := #[]
  classes   : Array EClass := #[]
  worklist  : Array EClassId := #[]
  iteration : Nat := 0

/-- Add a node to an EClass, updating minCost if this node is cheaper. -/
def SaturateState.addNode (s : SaturateState) (node : PartitionNode) (classId : EClassId) : SaturateState :=
  let nodeCost := evalNodeCost? node s.classes
  match nodeCost with
  | none => s
  | some cost =>
    let oldClass := if classId < s.classes.size then s.classes[classId]! else
      { id := classId, nodes := #[], minCost := Int.maxVal, bestNode := none,
        bounds := { minX := 0, maxX := 1, minY := 0, maxY := 1, minZ := 0, maxZ := 1 },
        firstCode := 0, lastCode := 0 }
    let newNodeId := s.nodes.size
    let newNodes := oldClass.nodes.push newNodeId
    let (newMinCost, newBestNode) := if cost < oldClass.minCost then (cost, some newNodeId) else (oldClass.minCost, oldClass.bestNode)
    let newClass := { oldClass with nodes := newNodes, minCost := newMinCost, bestNode := newBestNode }
    let newClasses := if classId < s.classes.size then
      s.classes.set classId newClass
    else
      s.classes.push newClass
    { s with nodes := s.nodes.push node, classes := newClasses }

/-- Check if saturation has reached fixed point. -/
def SaturateState.isSaturated (s : SaturateState) : Bool :=
  s.worklist.isEmpty

/-- Get the root EClass (lowest cost among all classes). -/
def SaturateState.getRootClass (s : SaturateState) : Option EClassId :=
  if s.classes.isEmpty then none
  else
    let mut bestId := 0
    let mut bestCost := s.classes[0]!'.minCost
    for i in List.range s.classes.size do
      let cost := s.classes[i]!'.minCost
      if cost < bestCost then
        bestId := i
        bestCost := cost
    some bestId

-- ── AV1-style axis variant rewrite ─────────────────────────────────────────
--
-- For each 2-way split node (.horz, .vert, .depth), insert the two
-- alternative axis splits into the same EClass.  The E-graph keeps the
-- cheapest.  This is the BVH analogue of AV1's partition mode search:
-- the encoder tries horizontal, vertical, and square splits at each
-- superblock and picks the best RDO cost.

/-- Generate axis variants for a 2-way split node.  Given a node that
    splits on one axis, return the two alternative axis nodes. -/
private def axisVariants (node : PartitionNode) : List PartitionNode :=
  match node with
  | .horz  p l r => [.vert p l r, .depth p l r]
  | .vert  p l r => [.horz p l r, .depth p l r]
  | .depth p l r => [.horz p l r, .vert  p l r]
  | other        => []

/-- One saturation pass: for every EClass, try axis variants of each
    2-way node.  Returns the updated state and whether any cost improved. -/
def SaturateState.axisRewritePass (s : SaturateState) : SaturateState × Bool :=
  let mut state := s
  let mut improved := false
  for classId in List.range s.classes.size do
    let cls := state.classes[classId]!
    let oldCost := cls.minCost
    -- Collect node IDs before mutating (snapshot).
    let nodeIds := cls.nodes.toList
    for nid in nodeIds do
      if nid < state.nodes.size then
        let node := state.nodes[nid]!
        for variant in axisVariants node do
          state := state.addNode variant classId
    if state.classes[classId]!.minCost < oldCost then
      improved := true
  (state, improved)

/-- Run axis variant saturation until fixed point (no cost improves).
    Bounded by `fuel` iterations to guarantee termination. -/
def saturateAxes (s : SaturateState) (fuel : Nat := 3) : SaturateState :=
  match fuel with
  | 0 => s
  | n + 1 =>
    let (s', improved) := s.axisRewritePass
    if improved then saturateAxes s' n else s'
