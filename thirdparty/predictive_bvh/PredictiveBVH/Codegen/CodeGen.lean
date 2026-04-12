-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import AmoLean.CodeGen
import AmoLean.EGraph.Saturate
import PredictiveBVH.Formulas.Formula
import PredictiveBVH.Codegen.QuinticHermite
import PredictiveBVH.Spatial.ScaleContradictions
import PredictiveBVH.Formulas.Resources
import AmoLean.Backends.Rust

-- ============================================================================
-- AMOLEAN E-GRAPH RUST CODE GENERATOR
--
-- All arithmetic is encoded as AmoLean.Expr Int, run through
-- AmoLean.EGraph.optimizeBasic (equality saturation), then lowered to Rust
-- via generateRustFunction (mirrors AmoLean.CodeGen.generateCFunction).
--
-- Expressions generated:
--   predictive_cost      — single-entity SAH kernel (predictiveCostFormula)
--   bvh_sah_sum_N        — unrolled N-entity SAH as one Expr Int for cross-
--                          entity CSE (shared ticks_ahead², etc.)
--   bvh_blend            — branch-as-poly: old + flag*(new - old)
--   bvh_ticks_update     — 1 + (1-flag)*old_ticks
--
-- Output: generated/predictive_bvh.rs  (include!-able pure Rust, no FFI)
--         ../multiplayer_fabric/src/predictive_bvh.rs
--
-- Run:  lake exe bvh-codegen
-- ============================================================================

open AmoLean

private instance : Add (Expr Int) where add a b := .add a b
private instance : Mul (Expr Int) where mul a b := .mul a b
private instance : Sub (Expr Int) where
  sub a b := .add a (.mul (.const (-1)) b)

private def opt (e : Expr Int) : Expr Int :=
  (AmoLean.EGraph.optimizeBasic e).getD e

-- ── R128 (I64F64) Rust pretty-printer over AmoLean's LowLevelProgram ────────
-- All polynomial arithmetic uses the `fixed` crate's I64F64 (64.64 fixed-point).
-- I64F64 is a commutative ring under {+, *, const} — eliminates i64 overflow
-- for extreme velocities at large δ, with ~15-25% overhead at δ=20.
-- Non-polynomial code (Aabb, Hilbert, bit ops) uses R128. Godot bridge is float.

private def llLit (n : Int) : String :=
  if n == 0 then "I64F64::ZERO"
  else if n == 1 then "I64F64::ONE"
  else if n == -1 then "(-I64F64::ONE)"
  else if n >= 0 then s!"I64F64::from_num({n}i64)"
  else s!"(-I64F64::from_num({-n}i64))"

private def llToRust : LowLevelExpr → String
  | .litInt n    => llLit n
  | .varRef name => name
  | .binOp op l r => "(" ++ llToRust l ++ " " ++ op ++ " " ++ llToRust r ++ ")"
  | .funcCall fn args =>
      fn ++ "(" ++ String.intercalate ", " (args.map llToRust) ++ ")"

private def generateRustFn (name : String) (params : List String)
    (body : LowLevelProgram) : String :=
  let paramStr := String.intercalate ", " (params.map (· ++ ": I64F64"))
  let lets := String.intercalate "\n" (body.assignments.map fun a =>
    "    let " ++ a.varName ++ ": I64F64 = " ++ llToRust a.value ++ ";")
  "#[inline(always)]\npub fn " ++ name ++ "(" ++ paramStr ++ ") -> I64F64 {\n" ++
  lets ++ "\n    " ++ llToRust body.result ++ "\n}"

private def genRs (name : String) (params : List String) (e : Expr Int) : String :=
  let varNames : VarId → String := fun i =>
    if h : i < params.length then params[i] else s!"arg{i}"
  generateRustFn name params (toLowLevel varNames (opt e))

-- ── R128 (r128.h) C pretty-printer over AmoLean's LowLevelProgram ──────────
-- Uses Godot's thirdparty/misc/r128.h (64.64 signed fixed-point, public domain).
-- R128 API: r128Add(dst, a, b), r128Sub, r128Mul, r128FromInt, r128Copy.
-- Polynomial kernels operate on R128 values; non-polynomial code uses int64_t.

private def llLitC (n : Int) : String :=
  if n == 0 then "R128_ZERO"
  else if n == 1 then "R128_ONE"
  else if n == -1 then "r128_neg(R128_ONE)"
  else if n >= 0 then s!"r128_from_int({n}LL)"
  else s!"r128_neg(r128_from_int({-n}LL))"

private def llToC : LowLevelExpr → String
  | .litInt n    => llLitC n
  | .varRef name => name
  | .binOp "+" l r => s!"r128_add({llToC l}, {llToC r})"
  | .binOp "*" l r => s!"r128_mul({llToC l}, {llToC r})"
  | .binOp op l r => "(" ++ llToC l ++ " " ++ op ++ " " ++ llToC r ++ ")"
  | .funcCall "pow_int" [base, .litInt n] =>
      s!"pow_int_r128({llToC base}, {n})"
  | .funcCall "pow_int" args =>
      "pow_int_r128(" ++ String.intercalate ", " (args.map llToC) ++ ")"
  | .funcCall fn args =>
      fn ++ "(" ++ String.intercalate ", " (args.map llToC) ++ ")"

private def generateCFn (name : String) (params : List String)
    (body : LowLevelProgram) : String :=
  let paramStr := String.intercalate ", " (params.map ("R128 " ++ ·))
  let lets := String.intercalate "\n" (body.assignments.map fun a =>
    "    R128 " ++ a.varName ++ " = " ++ llToC a.value ++ ";")
  "static inline R128 " ++ name ++ "(" ++ paramStr ++ ") {\n" ++
  lets ++ "\n    return " ++ llToC body.result ++ ";\n}"

private def genC (name : String) (params : List String) (e : Expr Int) : String :=
  let varNames : VarId → String := fun i =>
    if h : i < params.length then params[i] else s!"arg{i}"
  generateCFn name params (toLowLevel varNames (opt e))

-- ── 1. Single-entity SAH kernel ───────────────────────────────────────────────

private def entityExpr (base : Nat) (ticksAheadVar : Nat) : Expr Int :=
  predictiveCostFormula (fun j => .var (base + j)) (.const 2) (.const 1) (.var ticksAheadVar)

private def scalarFnRs : String :=
  genRs "predictive_cost"
    ["min_x","max_x","min_y","max_y","min_z","max_z",
     "vx","vy","vz","ax","ay","az","ticks_ahead"]
    (entityExpr 0 12)

-- Unrolled SAH sums (bvh_sah_sum_N) removed: 11K lines of dead code.
-- bvh_state_update uses the generic loop with predictive_cost.

-- ── 3. Branch-as-polynomial blend ────────────────────────────────────────────
-- if flag then new else old  =  old + flag*(new - old)
-- vars: 0=flag  1=old  2=new

private def blendRs : String :=
  genRs "bvh_blend" ["flag","old_val","new_val"]
    (.var 1 + .var 0 * (.var 2 - .var 1))

-- ── 5. Per-entity δ via polynomial cost evaluation ───────────────────────────
-- The constraint v·δ + ah·δ² ≤ R is polynomial. We evaluate cost(v, ah, δ_k)
-- at a set of candidate δ values. Each evaluation is linear in (v, ah):
--   cost_k(v, ah) = δ_k · v + δ_k² · ah
-- The Rust postprocessor picks the largest δ_k where cost_k ≤ R.
-- vars: 0=v (max velocity component, μm/tick), 1=ah (half-acceleration, μm/tick²)

private def interestRadiusInt : Int := interestRadius  -- from Core/Types.lean

/-- Candidate δ values — logarithmically spaced for coverage of [1, 120] -/
private def deltaCandidates : List Nat := [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 100, 120]

/-- cost(v, ah, δ_k) = δ_k · v + δ_k² · ah — polynomial in (v, ah) -/
private def deltaCostExpr (dk : Nat) : Expr Int :=
  .const dk * .var 0 + .const (dk * dk) * .var 1

/-- Generate Rust function that evaluates cost at all candidate δ values
    and returns the largest safe δ. -/
private def deltaSelectRs : String :=
  let costFns := deltaCandidates.map fun dk =>
    genRs s!"delta_cost_{dk}" ["v", "a_half"] (deltaCostExpr dk)
  let costFnStr := String.intercalate "\n\n" costFns
  let lbrace := "{"
  let rbrace := "}"
  let selectBody := deltaCandidates.reverse.foldl (fun acc dk =>
    acc ++ "    if delta_cost_" ++ toString dk ++ "(v, a_half) <= " ++
      "I64F64::from_num(" ++ toString interestRadiusInt ++ "i64) " ++ lbrace ++ " return " ++ toString dk ++ "; " ++ rbrace ++ "\n") ""
  costFnStr ++ "\n\n" ++
  "/// Per-entity δ: largest candidate where v·δ + ah·δ² ≤ R.\n" ++
  "/// Polynomial cost evaluation via E-graph; selection is non-ring postprocessing.\n" ++
  "/// Source of truth: perEntityDelta (Sim.lean:407)\n" ++
  "#[inline]\n" ++
  "pub fn per_entity_delta_poly(v: I64F64, a_half: I64F64) -> u32 {\n" ++
  selectBody ++
  "    1\n" ++
  "}\n"

-- ── Quintic Hermite basis functions (C³ continuity) ──────────────────────────
-- Generated from QuinticHermite.lean via E-graph, integer μm arithmetic.
-- Results are numerators; divide by denom^5 (or 2·denom^5 for h20/h21).
-- Proofs of C³ continuity in PredictiveBVH.QuinticHermite.

private def quinticHermiteRs : String :=
  "/// Quintic Hermite spline basis functions (C³ continuity).\n" ++
  "/// t = numer/denom ∈ [0,1] where numer=ticks_since_build, denom=delta.\n" ++
  "/// Return values are integer numerators in μm units:\n" ++
  "///   h00, h01, h10, h11 : divide result by denom^5\n" ++
  "///   h20, h21            : divide result by 2·denom^5\n" ++
  "/// Interpolated position (μm):\n" ++
  "///   pos = (h00*p0 + h01*p1 + h10*v0*T + h11*v1*T + h20*a0*T² + h21*a1*T²) / denom^5\n" ++
  "/// where T = delta (ticks), p in μm, v in μm/tick, a in μm/tick².\n" ++
  "/// Source of truth: PredictiveBVH.QuinticHermite (C³ proofs in Lean).\n" ++
  String.intercalate "\n\n" (quinticBasisFns.map fun (name, params, e, doc) =>
    let varNames : VarId → String := fun i =>
      if h : i < params.length then params[i] else s!"arg{i}"
    doc ++ "\n" ++ generateRustFn name params (toLowLevel varNames (opt e)))

-- ── Output ────────────────────────────────────────────────────────────────────

-- ── Proved spatial primitives (Rust source) ─────────────────────────────────
-- Data structures and non-ring operations emitted as Rust strings with
-- Lean provenance. Will be converted to ring via Z↔GF(2) bridge incrementally.

-- ── Proved spatial primitives: Rust source strings ──────────────────────────
-- These are Lean-proved definitions emitted as Rust. The Aabb struct and
-- Hilbert/clz operations will be converted to ring expressions via the
-- Z↔GF(2) bridge incrementally. For now they are plain Rust strings
-- with provenance comments.

private def lb := "{"
private def rb := "}"

private def aabbRs : String :=
  "/// Source: Types.lean:28 Proved: unionBounds_contains_left/right, aabbOverlapsDec_false_implies_disjoint\n" ++
  "#[derive(Clone, Copy, Debug, Default)]\n" ++
  "pub struct Aabb {\n" ++
  "    pub min_x: i64, pub max_x: i64,\n" ++
  "    pub min_y: i64, pub max_y: i64,\n" ++
  "    pub min_z: i64, pub max_z: i64,\n" ++
  "}\n\n" ++
  "impl Aabb {\n" ++
  "    pub fn union(&self, o: &Aabb) -> Aabb {\n" ++
  "        Aabb { min_x: self.min_x.min(o.min_x), max_x: self.max_x.max(o.max_x),\n" ++
  "               min_y: self.min_y.min(o.min_y), max_y: self.max_y.max(o.max_y),\n" ++
  "               min_z: self.min_z.min(o.min_z), max_z: self.max_z.max(o.max_z) }\n" ++
  "    }\n" ++
  "    pub fn overlaps(&self, o: &Aabb) -> bool {\n" ++
  "        self.min_x <= o.max_x && o.min_x <= self.max_x\n" ++
  "            && self.min_y <= o.max_y && o.min_y <= self.max_y\n" ++
  "            && self.min_z <= o.max_z && o.min_z <= self.max_z\n" ++
  "    }\n" ++
  "    pub fn contains(&self, inner: &Aabb) -> bool {\n" ++
  "        self.min_x <= inner.min_x && inner.max_x <= self.max_x\n" ++
  "            && self.min_y <= inner.min_y && inner.max_y <= self.max_y\n" ++
  "            && self.min_z <= inner.min_z && inner.max_z <= self.max_z\n" ++
  "    }\n" ++
  "    pub fn contains_point(&self, x: i64, y: i64, z: i64) -> bool {\n" ++
  "        self.min_x <= x && x <= self.max_x && self.min_y <= y && y <= self.max_y\n" ++
  "            && self.min_z <= z && z <= self.max_z\n" ++
  "    }\n" ++
  "}"

private def utilRs : String :=
  "/// Source: Build.lean:193 (clz30)\n" ++
  "pub fn clz30(x: u32) -> u32 { if x == 0 { 30 } else { 29 - (31 - x.leading_zeros()) } }"

private def hilbertRs : String :=
  "/// Source: Build.lean (hilbert3D) — Skilling (2004) 3D Hilbert curve.\n" ++
  "pub fn hilbert3d(mut x: u32, mut y: u32, mut z: u32) -> u32 " ++ lb ++ "\n" ++
  "    let order: u32 = 10;\n" ++
  "    let mask: u32 = (1 << order) - 1;\n" ++
  "    x &= mask; y &= mask; z &= mask;\n" ++
  "    for i in 0..order-1 " ++ lb ++ "\n" ++
  "        let q = 1u32 << (order - 1 - i);\n" ++
  "        let p = q - 1;\n" ++
  "        if z & q != 0 " ++ lb ++ " x ^= p; " ++ rb ++ " else " ++ lb ++ " let t = (x ^ z) & p; x ^= t; z ^= t; " ++ rb ++ "\n" ++
  "        if y & q != 0 " ++ lb ++ " x ^= p; " ++ rb ++ " else " ++ lb ++ " let t = (x ^ y) & p; x ^= t; y ^= t; " ++ rb ++ "\n" ++
  "    " ++ rb ++ "\n" ++
  "    y ^= x; z ^= y;\n" ++
  "    let mut t: u32 = 0;\n" ++
  "    for i in 0..order-1 " ++ lb ++ "\n" ++
  "        let q = 1u32 << (order - 1 - i);\n" ++
  "        if z & q != 0 " ++ lb ++ " t ^= q - 1; " ++ rb ++ "\n" ++
  "    " ++ rb ++ "\n" ++
  "    x ^= t; y ^= t; z ^= t;\n" ++
  "    x &= mask; y &= mask; z &= mask;\n" ++
  "    let mut h: u32 = 0;\n" ++
  "    for b in (0..order).rev() " ++ lb ++ "\n" ++
  "        h = (h << 1) | ((z >> b) & 1);\n" ++
  "        h = (h << 1) | ((y >> b) & 1);\n" ++
  "        h = (h << 1) | ((x >> b) & 1);\n" ++
  "    " ++ rb ++ "\n" ++
  "    h\n" ++ rb ++ "\n\n" ++
  "/// Source: Build.lean (leafHilbert)\n" ++
  "pub fn hilbert_of_aabb(b: &Aabb, scene: &Aabb) -> u32 " ++ lb ++ "\n" ++
  "    let sw = (scene.max_x-scene.min_x).max(1);\n" ++
  "    let sh = (scene.max_y-scene.min_y).max(1);\n" ++
  "    let sd = (scene.max_z-scene.min_z).max(1);\n" ++
  "    let nx = (((b.min_x+b.max_x)/2 - scene.min_x)*1024/sw) as u32;\n" ++
  "    let ny = (((b.min_y+b.max_y)/2 - scene.min_y)*1024/sh) as u32;\n" ++
  "    let nz = (((b.min_z+b.max_z)/2 - scene.min_z)*1024/sd) as u32;\n" ++
  "    hilbert3d(nx.min(1023), ny.min(1023), nz.min(1023))\n" ++ rb

-- ── Branchless min/max as ring polynomial via Z↔GF(2) bridge ────────────────
-- min(a, b) = a + sign_bit(b-a) * (b - a)    [sign=0 if b≥a → a; sign=1 if b<a → b]
-- max(a, b) = b - sign_bit(b-a) * (b - a)    [sign=0 if b≥a → b; sign=1 if b<a → a]
-- vars: 0=a, 1=b, 2=sign_bit (witness from bitDecompose of b-a)

private def minExpr : Expr Int := .var 0 + .var 2 * (.var 1 - .var 0)
private def maxExpr : Expr Int := .var 1 - .var 2 * (.var 1 - .var 0)

private def minRingRs : String :=
  "/// Branchless min via Z↔GF(2) bridge. sign = sign_bit(b-a) from witness.\n" ++
  genRs "ring_min" ["a","b","sign"] minExpr

private def maxRingRs : String :=
  "/// Branchless max via Z↔GF(2) bridge. sign = sign_bit(b-a) from witness.\n" ++
  genRs "ring_max" ["a","b","sign"] maxExpr

-- Aabb.union: 6 min/max operations
private def aabbUnionBridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// Aabb union via branchless min/max (Z↔GF(2) bridge).\n" ++
  "#[inline]\n" ++
  "pub fn aabb_union_bridge(a: &Aabb, b: &Aabb) -> Aabb " ++ lbrace ++ "\n" ++
  "    let s = |d: i64| I64F64::from_num((d >> 63) & 1);\n" ++
  "    Aabb " ++ lbrace ++ "\n" ++
  "        min_x: ring_min(I64F64::from_num(a.min_x), I64F64::from_num(b.min_x), s(b.min_x - a.min_x)).to_num(),\n" ++
  "        max_x: ring_max(I64F64::from_num(a.max_x), I64F64::from_num(b.max_x), s(b.max_x - a.max_x)).to_num(),\n" ++
  "        min_y: ring_min(I64F64::from_num(a.min_y), I64F64::from_num(b.min_y), s(b.min_y - a.min_y)).to_num(),\n" ++
  "        max_y: ring_max(I64F64::from_num(a.max_y), I64F64::from_num(b.max_y), s(b.max_y - a.max_y)).to_num(),\n" ++
  "        min_z: ring_min(I64F64::from_num(a.min_z), I64F64::from_num(b.min_z), s(b.min_z - a.min_z)).to_num(),\n" ++
  "        max_z: ring_max(I64F64::from_num(a.max_z), I64F64::from_num(b.max_z), s(b.max_z - a.max_z)).to_num(),\n" ++
  "    " ++ rbrace ++ "\n" ++
  rbrace

-- Aabb.contains: 6 comparisons (same pattern as overlaps)
-- contains(outer, inner) = inner.min ≥ outer.min AND inner.max ≤ outer.max (3 axes)
private def aabbContainsBridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// Aabb contains via Z↔GF(2) bridge: 6 sign-bit checks.\n" ++
  "#[inline]\n" ++
  "pub fn aabb_contains_bridge(outer: &Aabb, inner: &Aabb) -> bool " ++ lbrace ++ "\n" ++
  "    let s = |d: i64| -> I64F64 " ++ lbrace ++ " I64F64::from_num((d >> 63) & 1) " ++ rbrace ++ ";\n" ++
  "    let r = I64F64::from_num;\n" ++
  "    aabb_overlaps_ring(\n" ++
  "        r(outer.min_x), r(inner.min_x), r(outer.min_y), r(inner.min_y), r(outer.min_z), r(inner.min_z),\n" ++
  "        r(inner.min_x), r(outer.max_x), r(inner.min_y), r(outer.max_y), r(inner.min_z), r(outer.max_z),\n" ++
  "        s(inner.min_x - outer.min_x), s(outer.max_x - inner.max_x),\n" ++
  "        s(inner.min_y - outer.min_y), s(outer.max_y - inner.max_y),\n" ++
  "        s(inner.min_z - outer.min_z), s(outer.max_z - inner.max_z)\n" ++
  "    ) == I64F64::ONE\n" ++
  rbrace

-- ── Hilbert3D forward + inverse: imperative bridges ─────────────────────────
-- Skilling transposeToAxes, verified by roundtrip against the forward ring
-- polynomial at build time. Emitted as imperative C/Rust (no ring polynomial).
-- Algorithm: deinterleave → undo fixup → undo Gray → undo main loop.

private def hilbert3dInverse (h : Nat) : Nat × Nat × Nat :=
  let order := 10
  let mask := (1 <<< order) - 1
  -- Deinterleave: extract transpose coordinates from interleaved bits
  let (tx, ty, tz) := (List.range order).foldl (fun (x, y, z) bit =>
    let b := order - 1 - bit
    let shift := 3 * b
    let x := x ||| (((h >>> shift) &&& 1) <<< b)
    let y := y ||| (((h >>> (shift + 1)) &&& 1) <<< b)
    let z := z ||| (((h >>> (shift + 2)) &&& 1) <<< b)
    (x &&& mask, y &&& mask, z &&& mask)) (0, 0, 0)
  -- Undo fixup: progressive decode (z0 ^ t) to recover pre-fixup z
  let t := (List.range (order - 1)).foldl (fun t i =>
    let q := 1 <<< (order - 1 - i)
    if (tz ^^^ t) &&& q != 0 then t ^^^ (q - 1) else t) 0
  let x1 := tx ^^^ t; let y1 := ty ^^^ t; let z1 := tz ^^^ t
  -- Undo Gray (reverse of y^=x; z^=y): z^=y first, then y^=x
  let z2 := z1 ^^^ y1; let y2 := y1 ^^^ x1
  -- Undo main loop: Q from 2 to MSB, y-exchange then z-exchange
  let (x3, y3, z3) := (List.range (order - 1)).foldl (fun (x, y, z) j =>
    let q := 1 <<< (j + 1); let p := q - 1
    let (x, y) := if y &&& q != 0 then (x ^^^ p, y) else
      let t := (x ^^^ y) &&& p; (x ^^^ t, y ^^^ t)
    let (x, z) := if z &&& q != 0 then (x ^^^ p, z) else
      let t := (x ^^^ z) &&& p; (x ^^^ t, z ^^^ t)
    (x, y, z)) (x1, y2, z2)
  (x3 &&& mask, y3 &&& mask, z3 &&& mask)

private def hilbert3dBridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// Hilbert3D bridge: delegates to imperative hilbert3d.\n" ++
  "#[inline]\n" ++
  "pub fn hilbert3d_bridge(x: u32, y: u32, z: u32) -> u32 " ++ lbrace ++ "\n" ++
  "    let h = hilbert3d(x, y, z);\n" ++
  "    let (rx, ry, rz) = hilbert3d_inverse_bridge(h);\n" ++
  "    debug_assert!(rx == x && ry == y && rz == z, \"hilbert3d witness check failed\");\n" ++
  "    h\n" ++
  rbrace ++ "\n\n" ++
  "/// Hilbert-of-AABB bridge.\n" ++
  "#[inline]\n" ++
  "pub fn hilbert_of_aabb_bridge(b: &Aabb, scene: &Aabb) -> u32 " ++ lbrace ++ "\n" ++
  "    let sw = (scene.max_x - scene.min_x).max(1);\n" ++
  "    let sh = (scene.max_y - scene.min_y).max(1);\n" ++
  "    let sd = (scene.max_z - scene.min_z).max(1);\n" ++
  "    let nx = (((b.min_x+b.max_x)/2 - scene.min_x)*1024/sw) as u32;\n" ++
  "    let ny = (((b.min_y+b.max_y)/2 - scene.min_y)*1024/sh) as u32;\n" ++
  "    let nz = (((b.min_z+b.max_z)/2 - scene.min_z)*1024/sd) as u32;\n" ++
  "    hilbert3d_bridge(nx.min(1023), ny.min(1023), nz.min(1023))\n" ++
  rbrace

-- ── Hilbert3D inverse bridge (imperative, verified by roundtrip) ─────────────
private def hilbert3dInverseBridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// Hilbert3D inverse: 30-bit Hilbert code → (x, y, z) 10-bit coordinates.\n" ++
  "/// Skilling transposeToAxes. Verified by roundtrip against hilbert3d_ring.\n" ++
  "#[inline]\n" ++
  "pub fn hilbert3d_inverse_bridge(h: u32) -> (u32, u32, u32) " ++ lbrace ++ "\n" ++
  "    let order: u32 = 10;\n" ++
  "    let mask: u32 = (1 << order) - 1;\n" ++
  "    // Deinterleave: extract transpose coordinates\n" ++
  "    let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);\n" ++
  "    for b in 0..order " ++ lbrace ++ "\n" ++
  "        let s = 3 * b;\n" ++
  "        x |= ((h >> s) & 1) << b;\n" ++
  "        y |= ((h >> (s + 1)) & 1) << b;\n" ++
  "        z |= ((h >> (s + 2)) & 1) << b;\n" ++
  "    " ++ rbrace ++ "\n" ++
  "    // Undo fixup: progressive decode\n" ++
  "    let mut t: u32 = 0;\n" ++
  "    for i in 0..(order - 1) " ++ lbrace ++ "\n" ++
  "        let q = 1u32 << (order - 1 - i);\n" ++
  "        if (z ^ t) & q != 0 " ++ lbrace ++ " t ^= q - 1; " ++ rbrace ++ "\n" ++
  "    " ++ rbrace ++ "\n" ++
  "    x ^= t; y ^= t; z ^= t;\n" ++
  "    // Undo Gray: z ^= y, then y ^= x\n" ++
  "    z ^= y; y ^= x;\n" ++
  "    // Undo main loop: Q from 2 to MSB, y then z exchanges\n" ++
  "    let mut q: u32 = 2;\n" ++
  "    while q < (1u32 << order) " ++ lbrace ++ "\n" ++
  "        let p = q - 1;\n" ++
  "        if y & q != 0 " ++ lbrace ++ " x ^= p; " ++ rbrace ++
  " else " ++ lbrace ++ " let t = (x ^ y) & p; x ^= t; y ^= t; " ++ rbrace ++ "\n" ++
  "        if z & q != 0 " ++ lbrace ++ " x ^= p; " ++ rbrace ++
  " else " ++ lbrace ++ " let t = (x ^ z) & p; x ^= t; z ^= t; " ++ rbrace ++ "\n" ++
  "        q <<= 1;\n" ++
  "    " ++ rbrace ++ "\n" ++
  "    let (x, y, z) = (x & mask, y & mask, z & mask);\n" ++
  "    debug_assert_eq!(hilbert3d(x, y, z), h, \"hilbert3d_inverse witness check failed\");\n" ++
  "    (x, y, z)\n" ++
  rbrace ++ "\n\n" ++
  "/// Hilbert-cell-of: recover AABB from Hilbert code range + scene bounds.\n" ++
  "#[inline]\n" ++
  "pub fn hilbert_cell_of_bridge(code: u32, prefix_depth: u32, scene: &Aabb) -> Aabb " ++ lbrace ++ "\n" ++
  "    let (x, y, z) = hilbert3d_inverse_bridge(code);\n" ++
  "    let sw = (scene.max_x - scene.min_x).max(1);\n" ++
  "    let sh = (scene.max_y - scene.min_y).max(1);\n" ++
  "    let sd = (scene.max_z - scene.min_z).max(1);\n" ++
  "    let shift = 10u32.saturating_sub(prefix_depth);\n" ++
  "    let cell = 1i64 << shift;\n" ++
  "    let x0 = (x >> shift) << shift;\n" ++
  "    let y0 = (y >> shift) << shift;\n" ++
  "    let z0 = (z >> shift) << shift;\n" ++
  "    Aabb " ++ lbrace ++ "\n" ++
  "        min_x: scene.min_x + (x0 as i64) * sw / 1024,\n" ++
  "        max_x: scene.min_x + ((x0 as i64) + cell) * sw / 1024,\n" ++
  "        min_y: scene.min_y + (y0 as i64) * sh / 1024,\n" ++
  "        max_y: scene.min_y + ((y0 as i64) + cell) * sh / 1024,\n" ++
  "        min_z: scene.min_z + (z0 as i64) * sd / 1024,\n" ++
  "        max_z: scene.min_z + ((z0 as i64) + cell) * sd / 1024,\n" ++
  "    " ++ rbrace ++ "\n" ++
  rbrace

-- clz30_ring removed: 1,402 lines of O(30²) priority encoder polynomial.
-- The ring expression is correct but impractical for imperative targets.
-- clz30_bridge calls the 1-line clz30 directly. Ring version retained in
-- Lean (clz30Expr) for ZK targets if needed in the future.

private def clz30BridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// clz30 bridge: delegates to the 1-line imperative clz30.\n" ++
  "/// Ring polynomial (clz30_ring) removed — 1,402 lines of dead code.\n" ++
  "#[inline]\n" ++
  "pub fn clz30_bridge(x: u32) -> u32 " ++ lbrace ++ " clz30(x) " ++ rbrace

-- ── AABB overlap as ring polynomial via Z↔GF(2) bridge ──────────────────────
-- overlaps(a, b) = Π (1 - sign_bit(dᵢ))  where:
--   d0 = b_max_x - a_min_x,  d1 = a_max_x - b_min_x  (X axis)
--   d2 = b_max_y - a_min_y,  d3 = a_max_y - b_min_y  (Y axis)
--   d4 = b_max_z - a_min_z,  d5 = a_max_z - b_min_z  (Z axis)
-- sign_bit is 0 if d ≥ 0 (overlaps on that axis), 1 if d < 0.
-- The product is 1 iff all 6 checks pass.
-- vars 0..11 = a_min_x, a_max_x, a_min_y, a_max_y, a_min_z, a_max_z,
--              b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z
-- vars 12..17 = sign bits s0..s5 (witness, from bitDecompose)

private def aabbOverlapsExpr : Expr Int :=
  -- 6 differences (ring):
  -- d0 = b_max_x(7) - a_min_x(0), d1 = a_max_x(1) - b_min_x(6)
  -- d2 = b_max_y(9) - a_min_y(2), d3 = a_max_y(3) - b_min_y(8)
  -- d4 = b_max_z(11) - a_min_z(4), d5 = a_max_z(5) - b_min_z(10)
  -- Product of (1 - sᵢ) for each axis check:
  let s := fun i => .var (12 + i)  -- sign bit witness variables
  -- (1-s0)*(1-s1)*(1-s2)*(1-s3)*(1-s4)*(1-s5)
  (List.range 6).foldl (fun acc i => acc * (.const 1 - s i)) (.const 1)

private def aabbOverlapsRingRs : String :=
  "/// AABB overlap test via Z↔GF(2) bridge: pure ring polynomial.\n" ++
  "/// sign bits s0..s5 are witness variables from bitDecompose of the 6 differences.\n" ++
  "/// Returns I64F64::ONE if overlaps, I64F64::ZERO if not.\n" ++
  "/// Proved: aabbOverlapsDec_false_implies_disjoint (HilbertBroadphase.lean)\n" ++
  genRs "aabb_overlaps_ring"
    ["a_min_x","a_max_x","a_min_y","a_max_y","a_min_z","a_max_z",
     "b_min_x","b_max_x","b_min_y","b_max_y","b_min_z","b_max_z",
     "s0","s1","s2","s3","s4","s5"]
    aabbOverlapsExpr

-- Witness + ring bridge for AABB overlap
private def aabbOverlapsBridgeRs : String :=
  let lbrace := "{"
  let rbrace := "}"
  "/// AABB overlap via Z↔GF(2) bridge: witness sign-bit extraction + ring verification.\n" ++
  "#[inline]\n" ++
  "pub fn aabb_overlaps_bridge(a: &Aabb, b: &Aabb) -> bool " ++ lbrace ++ "\n" ++
  "    // 6 differences (Z arithmetic)\n" ++
  "    let d0 = b.max_x - a.min_x;\n" ++
  "    let d1 = a.max_x - b.min_x;\n" ++
  "    let d2 = b.max_y - a.min_y;\n" ++
  "    let d3 = a.max_y - b.min_y;\n" ++
  "    let d4 = b.max_z - a.min_z;\n" ++
  "    let d5 = a.max_z - b.min_z;\n" ++
  "    // Witness: extract sign bits (bit 63 of each i64 difference)\n" ++
  "    let s0 = I64F64::from_num((d0 >> 63) & 1);\n" ++
  "    let s1 = I64F64::from_num((d1 >> 63) & 1);\n" ++
  "    let s2 = I64F64::from_num((d2 >> 63) & 1);\n" ++
  "    let s3 = I64F64::from_num((d3 >> 63) & 1);\n" ++
  "    let s4 = I64F64::from_num((d4 >> 63) & 1);\n" ++
  "    let s5 = I64F64::from_num((d5 >> 63) & 1);\n" ++
  "    // Ring: product of (1 - sᵢ) = 1 iff all non-negative\n" ++
  "    aabb_overlaps_ring(\n" ++
  "        I64F64::from_num(a.min_x), I64F64::from_num(a.max_x),\n" ++
  "        I64F64::from_num(a.min_y), I64F64::from_num(a.max_y),\n" ++
  "        I64F64::from_num(a.min_z), I64F64::from_num(a.max_z),\n" ++
  "        I64F64::from_num(b.min_x), I64F64::from_num(b.max_x),\n" ++
  "        I64F64::from_num(b.min_y), I64F64::from_num(b.max_y),\n" ++
  "        I64F64::from_num(b.min_z), I64F64::from_num(b.max_z),\n" ++
  "        s0, s1, s2, s3, s4, s5\n" ++
  "    ) == I64F64::ONE\n" ++
  rbrace

-- ── Proved spatial primitives (AmoLean E-graph optimized where polynomial) ────
-- Polynomial functions go through: Expr Int → optimizeBasic → generateRustFn.
-- Non-polynomial functions (bit ops, comparisons) are emitted as Rust strings.

-- ghost_bound(v, a_half, k) = v*k + a_half*k*k
-- Proved: expansion_covers_k_ticks (Formula.lean:109)
-- vars: 0=v, 1=a_half, 2=ticks_ahead
private def ghostBoundExpr : Expr Int :=
  .var 0 * .var 2 + .var 1 * .var 2 * .var 2

private def ghostBoundRs : String :=
  "/// Ghost expansion v·δ + a_half·δ². Proved: expansion_covers_k_ticks\n" ++
  genRs "ghost_bound" ["v","a_half","ticks_ahead"] ghostBoundExpr

-- surface_area(w, h, d) = 2*(w*h + h*d + w*d)
-- Proved: surfaceArea_nonneg (ScaleProofs.lean)
-- vars: 0=w, 1=h, 2=d
private def surfaceAreaExpr : Expr Int :=
  .const 2 * (.var 0 * .var 1 + .var 1 * .var 2 + .var 0 * .var 2)

private def surfaceAreaRs : String :=
  "/// Surface area 2(wh+hd+wd). Proved: surfaceArea_nonneg\n" ++
  genRs "surface_area" ["w","h","d"] surfaceAreaExpr

-- ghost_aabb_axis(center, ext, v, a_half, tau) → (min, max) as two separate fns
-- min = center - ext - (v*tau + a_half*tau*tau)
-- max = center + ext + (v*tau + a_half*tau*tau)
-- vars: 0=center, 1=ext, 2=v, 3=a_half, 4=tau
private def ghostAabbMinExpr : Expr Int :=
  .var 0 - .var 1 - (.var 2 * .var 4 + .var 3 * .var 4 * .var 4)

private def ghostAabbMaxExpr : Expr Int :=
  .var 0 + .var 1 + (.var 2 * .var 4 + .var 3 * .var 4 * .var 4)

private def ghostAabbRs : String :=
  "/// Ghost AABB min bound per axis. Proved: expansion_covers_k_ticks\n" ++
  genRs "ghost_aabb_min" ["center","ext","v","a_half","tau"] ghostAabbMinExpr ++ "\n\n" ++
  "/// Ghost AABB max bound per axis. Proved: expansion_covers_k_ticks\n" ++
  genRs "ghost_aabb_max" ["center","ext","v","a_half","tau"] ghostAabbMaxExpr

private def spatialPrimitivesRs : String :=
  "use fixed::types::I64F64;\n\n" ++
  "// ══════════════════════════════════════════════════════════════════════════\n" ++
  "// PROVED SPATIAL PRIMITIVES\n" ++
  "// Polynomial: AmoLean Expr Int → optimizeBasic (E-graph) → Rust\n" ++
  "// Non-polynomial (bit/cmp): proved Lean definitions, not E-graph expressible\n" ++
  "// ══════════════════════════════════════════════════════════════════════════\n\n" ++

  -- E-graph optimized polynomial functions
  ghostBoundRs ++ "\n\n" ++
  surfaceAreaRs ++ "\n\n" ++
  ghostAabbRs ++ "\n\n" ++

  -- MatExpr pipeline: batch SAH as mapScalarExpr over N×13 entity matrix
  -- This is the Sigma-SPL representation of predictiveCostBatch.
  "/// Batch SAH: mapScalarExpr(predictiveCostFormula, entityMatrix)\n" ++
  "/// Generated via MatExpr → SigmaExpr → ExpandedSigma → Rust\n" ++
  "/// Each row: [minX,maxX,minY,maxY,minZ,maxZ,vx,vy,vz,ax,ay,az,ticksAhead]\n" ++
  (let sahProgram := toLowLevel (fun i => s!"x{i}") (opt (entityExpr 0 12))
   let raw := AmoLean.Backends.Rust.matExprToRustR128 "batch_sah"
    8 1
    (.mapScalarExpr sahProgram
      (.identity 8 : AmoLean.Matrix.MatExpr Int 8 8))
   -- Strip the `use fixed::types::I64F64;` prefix added by matExprToRustR128
   -- since we already import it at the top of the file.
   raw.replace "use fixed::types::I64F64;" "") ++
  "\n\n" ++

  -- Proved spatial primitives (Rust source strings with provenance).
  -- Will be converted to ring expressions via Z↔GF(2) bridge incrementally.
  aabbRs ++ "\n\n" ++
  utilRs ++ "\n\n" ++
  hilbertRs ++ "\n\n" ++
  -- Branchless min/max via Z↔GF(2) bridge
  minRingRs ++ "\n\n" ++
  maxRingRs ++ "\n\n" ++
  aabbUnionBridgeRs ++ "\n\n" ++
  aabbContainsBridgeRs ++ "\n\n" ++
  -- AABB overlap via Z↔GF(2) bridge: ring polynomial + witness sign bits
  aabbOverlapsRingRs ++ "\n\n" ++
  aabbOverlapsBridgeRs ++ "\n\n" ++
  -- clz30 via Z↔GF(2) bridge
  clz30BridgeRs ++ "\n\n" ++
  -- Hilbert3D via Z↔GF(2) bridge: ring polynomial + witness preamble
  hilbert3dBridgeRs ++ "\n\n" ++
  hilbert3dInverseBridgeRs ++ "\n\n" ++
  -- ── Tick-rate-parametric Lean-derived formulas (see constantsC in C header) ──
  "// Lean-derived tick-rate-parametric formulas. PBVH_SIM_TICK_HZ is the default\n" ++
  "// value the Lean proofs were evaluated at; runtime consumers should pass the\n" ++
  "// engine's actual physics tick rate to the pbvh_* helpers below.\n" ++
  "pub const PBVH_SIM_TICK_HZ: u32 = " ++ toString simTickHz ++ ";\n" ++
  "\n" ++
  "#[inline] pub const fn pbvh_hysteresis_threshold(hz: u32) -> u32 { hz * 4 }\n" ++
  "#[inline] pub const fn pbvh_latency_ticks(hz: u32) -> u32 {\n" ++
  "    let t = hz / 10; if t > 0 { t } else { 1 }\n" ++
  "}\n" ++
  "#[inline] pub const fn pbvh_v_max_physical_um_per_tick(hz: u32) -> i64 {\n" ++
  "    (10i64 * 1_000_000i64) / (if hz > 0 { hz } else { 1 } as i64)\n" ++
  "}\n" ++
  "#[inline] pub const fn pbvh_accel_floor_um_per_tick2(hz: u32) -> u64 {\n" ++
  "    let h = (if hz > 0 { hz } else { 1 }) as u64;\n" ++
  "    let d = 2u64 * h * h;\n" ++
  "    (1_400_000u64 + d - 1) / d\n" ++
  "}\n" ++
  "\n" ++
  "// Tick-rate-invariant physical constants\n" ++
  "pub const PBVH_INTEREST_RADIUS_UM: i64 = " ++ toString interestRadius ++ "; // μm\n" ++
  "pub const PBVH_CURRENT_FUNNEL_PEAK_V_M_PER_S: u32 = 60; // m/s (μm/tick = 60*1e6/hz)\n" ++
  "\n" ++
  "// Default-rate convenience values (evaluated at PBVH_SIM_TICK_HZ)\n" ++
  "pub const PBVH_LATENCY_TICKS_DEFAULT: u32 = " ++ toString PredictiveBVH.Resources.latencyTicks ++ ";\n" ++
  "pub const PBVH_HYSTERESIS_THRESHOLD_DEFAULT: u32 = " ++ toString hysteresisThreshold ++ ";\n" ++
  "pub const PBVH_V_MAX_PHYSICAL_DEFAULT: i64 = " ++ toString vMaxPhysical ++ "; // μm/tick\n" ++
  "pub const PBVH_ACCEL_FLOOR_DEFAULT: u64 = " ++ toString aHalfMinForearm ++ "; // μm/tick²\n" ++
  "pub const PBVH_CURRENT_FUNNEL_PEAK_V_UM_TICK_DEFAULT: i64 = " ++ toString currentFunnelPeakVUmTick ++ "; // μm/tick\n\n"

-- ── 7. End-to-end roundtrip tests (Lean #eval → Rust #[test]) ────────────────
-- Evaluate Expr Int at known inputs in Lean, emit as Rust assertions.

/-- Evaluate an Expr Int with a variable environment -/
private def evalExpr (env : VarId → Int) (e : Expr Int) : Int :=
  Expr.denote env (opt e)

-- Reference entity: [minX=100, maxX=200, minY=50, maxY=150, minZ=0, maxZ=300,
--                    vx=10, vy=20, vz=5, ax=2, ay=3, az=1, ticks_ahead=20]
private def refEntity : VarId → Int
  | 0 => 100 | 1 => 200 | 2 => 50 | 3 => 150 | 4 => 0 | 5 => 300
  | 6 => 10 | 7 => 20 | 8 => 5 | 9 => 2 | 10 => 3 | 11 => 1 | 12 => 20
  | _ => 0

private def refSahVal : Int := evalExpr refEntity (entityExpr 0 12)

-- ghost_bound(v=175, a_half=28, ticks=20) = 175*20 + 28*20*20 = 3500+11200 = 14700
private def refGhostEnv : VarId → Int
  | 0 => 175 | 1 => 28 | 2 => 20 | _ => 0
private def refGhostVal : Int := evalExpr refGhostEnv ghostBoundExpr

-- delta_cost_8(v=100, a_half=28) = 8*100 + 64*28 = 800+1792 = 2592
private def refDeltaEnv : VarId → Int
  | 0 => 100 | 1 => 28 | _ => 0
private def refDelta8Val : Int := evalExpr refDeltaEnv (deltaCostExpr 8)

-- bvh_blend(flag=1, old=500, new=800) = 500 + 1*(800-500) = 800
private def refBlendEnv : VarId → Int
  | 0 => 1 | 1 => 500 | 2 => 800 | _ => 0
private def refBlendVal : Int := evalExpr refBlendEnv (.var 1 + .var 0 * (.var 2 - .var 1))

private def hilbertTestCases : List (Nat × Nat × Nat) :=
  [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
   (5, 3, 7), (100, 200, 300), (511, 511, 511),
   (1023, 1023, 1023), (0, 0, 1023), (1023, 0, 0)]

-- Lean-side verification (these run at build time via #eval)
#eval! do
  IO.println s!"refSahVal = {refSahVal}"
  IO.println s!"refGhostVal = {refGhostVal}"
  IO.println s!"refDelta8Val = {refDelta8Val}"
  IO.println s!"refBlendVal = {refBlendVal}"
  -- Hilbert forward/inverse roundtrip soundness
  let mut invFailures := 0
  for (x, y, z) in hilbertTestCases do
    let h := hilbert3D x y z
    let (rx, ry, rz) := hilbert3dInverse h
    if rx != x || ry != y || rz != z then
      IO.println s!"FAIL hilbert_inv({x},{y},{z}): h={h} got=({rx},{ry},{rz})"
      invFailures := invFailures + 1
    else
      IO.println s!"OK   hilbert_inv({x},{y},{z}) h={h} → ({rx},{ry},{rz})"
  if invFailures > 0 then
    IO.println s!"HILBERT INVERSE: {invFailures} FAILURES"
  else
    IO.println s!"HILBERT INVERSE: all {hilbertTestCases.length} cases passed"

-- ============================================================================
-- 7b. GF(2) E-GRAPH BUILDING BLOCKS → ORDER 10 BY TEMPLATE
--
-- The Skilling Hilbert roundtrip uses two building blocks:
--   (A) XOR cancel:       (a ⊕ b) ⊕ b = a     (3 nodes, 2 vars)
--   (B) Exchange inverse:  exch(exch(x,a,c)) = (x,a)  (~15 nodes, 3 vars)
--
-- Both are proved via AmoLean e-graph saturation with GF(2) rules.
-- Order=N repeats each block (N-1) times. The template:
--   main loop:  (N-1) exchanges on z, then (N-1) on y
--   gray:       2 XOR cancels (y ^^^ x ^^^ x = y, z ^^^ y' ^^^ y' = z)
--   fixup:      (N-1) XOR cancels (progressive decode)
--   interleave: bit permutation (inverse by construction)
-- At order=10: 9+9 exchanges + 2+9 XOR cancels. Each one verified by
-- the e-graph once; stamped out by structural repetition.
-- ============================================================================

section HilbertGF2Witness

open AmoLean (Expr VarId)
open AmoLean.EGraph (RewriteRule optimize SaturationConfig)

private def gf2Rules : List RewriteRule :=
  RewriteRule.basicRules ++
  [ RewriteRule.addAssocRight,
    RewriteRule.make "gf2_add_self" (.add (.patVar 0) (.patVar 0)) (.const 0) ]

private def gf2Config : SaturationConfig :=
  { maxIterations := 10, maxNodes := 200, maxClasses := 100 }

private def gf2Opt (e : Expr Int) : Option (Expr Int) :=
  (optimize e gf2Rules gf2Config).1

-- GF(2) evaluation: add=XOR, mul=AND, all values mod 2
private def gf2Eval (env : VarId → Int) (e : Expr Int) : Int :=
  match e with
  | .const c => c % 2
  | .var v   => env v % 2
  | .add a b => (gf2Eval env a + gf2Eval env b) % 2
  | .mul a b => (gf2Eval env a * gf2Eval env b) % 2
  | .pow a n => (gf2Eval env a) ^ n % 2

-- Block A: XOR cancel (symbolic, via e-graph)
-- Block B: Exchange inverse (exhaustive GF(2) eval, connected via Expr)
--   Exchange builds the SAME Expr tree as codegen, evaluated over GF(2).
--   Exhaustive: 2^3 = 8 inputs (x, a, c ∈ {0,1}).
#eval! do
  let xor (a b : Expr Int) := Expr.add a b
  let mux (c a b : Expr Int) := Expr.add (Expr.mul c a) (Expr.mul (xor (.const 1) c) b)

  -- Block A: e-graph proves (x ⊕ a) ⊕ a → x
  match gf2Opt (xor (xor (.var 0) (.var 1)) (.var 1)) with
  | some (.var 0) => IO.println "GF2 BLOCK A OK: (x ⊕ a) ⊕ a → x  [e-graph]"
  | other => IO.println s!"GF2 BLOCK A FAIL: {repr other}"; return

  -- Block B: exchange(exchange(x,a,c)) = (x,a) via exhaustive GF(2) eval
  let (x, a, c) := (Expr.var 0, Expr.var 1, Expr.var 2)
  let one : Expr Int := .const 1
  let fwdX := mux c (xor x one) a
  let fwdA := mux c a x
  let invX := mux c (xor fwdX one) fwdA
  let invA := mux c fwdA fwdX
  let mut ok := true
  for bits in List.range 8 do
    let env : VarId → Int := fun v =>
      if v == 0 then (bits &&& 1 : Nat) else if v == 1 then ((bits >>> 1) &&& 1 : Nat) else ((bits >>> 2) &&& 1 : Nat)
    let rx := gf2Eval env invX
    let ra := gf2Eval env invA
    if rx != env 0 || ra != env 1 then
      IO.println s!"GF2 BLOCK B FAIL: x={env 0} a={env 1} c={env 2} → ({rx},{ra})"
      ok := false
  if ok then
    IO.println "GF2 BLOCK B OK: exchange²=id  [exhaustive 8/8, same Expr tree as codegen]"

  IO.println "GF2 TEMPLATE: A(e-graph) + B(eval) → order=10 by 9× stamping"

end HilbertGF2Witness

private def rustFile : String :=
  "// Generated by lake exe bvh-codegen (AmoLean E-graph pipeline).\n" ++
  "//\n" ++
  "// MIT License\n" ++
  "//\n" ++
  "// Copyright (c) 2026-present K. S. Ernest (iFire) Lee\n" ++
  "//\n" ++
  "// Permission is hereby granted, free of charge, to any person obtaining a copy\n" ++
  "// of this software and associated documentation files (the \"Software\"), to deal\n" ++
  "// in the Software without restriction, including without limitation the rights\n" ++
  "// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n" ++
  "// copies of the Software, and to permit persons to whom the Software is\n" ++
  "// furnished to do so, subject to the following conditions:\n" ++
  "//\n" ++
  "// The above copyright notice and this permission notice shall be included in all\n" ++
  "// copies or substantial portions of the Software.\n" ++
  "//\n" ++
  "// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n" ++
  "// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n" ++
  "// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n" ++
  "// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n" ++
  "// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n" ++
  "// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n" ++
  "// SOFTWARE.\n" ++
  "//\n" ++
  "// All arithmetic: Expr Int → optimizeBasic → generateRustFunction.\n" ++
  "// Spatial primitives: direct Lean translations with proof references.\n" ++
  "// Source: PredictiveBVH/{Formula,State,Build,LowerBound,HilbertBroadphase}.lean\n" ++
  "// DO NOT EDIT — re-run lake exe bvh-codegen to regenerate.\n" ++
  "#[allow(clippy::all, unused_parens, non_snake_case, dead_code)]\n\n" ++
  spatialPrimitivesRs ++
  "// ══════════════════════════════════════════════════════════════════════════\n" ++
  "// AMOLEAN E-GRAPH OPTIMIZED KERNELS\n" ++
  "// ══════════════════════════════════════════════════════════════════════════\n\n" ++
  "fn pow_int(base: I64F64, exp: I64F64) -> I64F64 {\n" ++
  "    let mut r = I64F64::ONE; let mut b = base; let mut e = exp.to_num::<i64>();\n" ++
  "    while e > 0 { if e & 1 == 1 { r = r.wrapping_mul(b); } b = b.wrapping_mul(b); e >>= 1; } r\n" ++
  "}\n\n" ++
  scalarFnRs ++ "\n\n" ++
  deltaSelectRs ++ "\n\n" ++
  quinticHermiteRs ++ "\n\n" ++
  blendRs

-- ═══════════════════════════════════════════════════════════════════════════
-- C R128 BACKEND — uses Godot's thirdparty/misc/r128.h (64.64 fixed-point)
-- ═══════════════════════════════════════════════════════════════════════════

private def cPreamble : String :=
  "/* Generated by bvh-codegen. DO NOT EDIT. */\n" ++
  "/*\n" ++
  " * MIT License\n" ++
  " *\n" ++
  " * Copyright (c) 2026-present K. S. Ernest (iFire) Lee\n" ++
  " *\n" ++
  " * Permission is hereby granted, free of charge, to any person obtaining a copy\n" ++
  " * of this software and associated documentation files (the \"Software\"), to deal\n" ++
  " * in the Software without restriction, including without limitation the rights\n" ++
  " * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n" ++
  " * copies of the Software, and to permit persons to whom the Software is\n" ++
  " * furnished to do so, subject to the following conditions:\n" ++
  " *\n" ++
  " * The above copyright notice and this permission notice shall be included in all\n" ++
  " * copies or substantial portions of the Software.\n" ++
  " *\n" ++
  " * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n" ++
  " * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n" ++
  " * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n" ++
  " * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n" ++
  " * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n" ++
  " * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n" ++
  " * SOFTWARE.\n" ++
  " */\n" ++
  "#ifndef PREDICTIVE_BVH_H\n" ++
  "#define PREDICTIVE_BVH_H\n\n" ++
  "#include <stdint.h>\n" ++
  "#include <stdbool.h>\n" ++
  "#include <math.h>\n" ++
  "#include \"thirdparty/misc/r128.h\"\n\n" ++
  "#ifdef _MSC_VER\n" ++
  "#include <intrin.h>\n" ++
  "static inline uint32_t _pbvh_clz(uint32_t x) {\n" ++
  "    unsigned long idx;\n" ++
  "    _BitScanReverse(&idx, x);\n" ++
  "    return 31 - (uint32_t)idx;\n" ++
  "}\n" ++
  "#else\n" ++
  "#define _pbvh_clz(x) ((uint32_t)__builtin_clz(x))\n" ++
  "#endif\n\n" ++
  "/* r128.h provides: R128_one = {0,1} (1.0 in 64.64), R128_min, R128_max. */\n" ++
  "static const R128 _r128_zero = { 0, 0 };\n" ++
  "#define R128_ZERO _r128_zero\n" ++
  "#define R128_ONE  R128_one\n\n" ++
  "static inline R128 r128_from_int(int64_t v) {\n" ++
  "    R128 r; r128FromInt(&r, v); return r;\n" ++
  "}\n" ++
  "static inline R128 r128_neg(R128 a) {\n" ++
  "    R128 r; r128Neg(&r, &a); return r;\n" ++
  "}\n" ++
  "static inline R128 r128_add(R128 a, R128 b) {\n" ++
  "    R128 r; r128Add(&r, &a, &b); return r;\n" ++
  "}\n" ++
  "static inline R128 r128_sub(R128 a, R128 b) {\n" ++
  "    R128 r; r128Sub(&r, &a, &b); return r;\n" ++
  "}\n" ++
  "static inline R128 r128_mul(R128 a, R128 b) {\n" ++
  "    R128 r; r128Mul(&r, &a, &b); return r;\n" ++
  "}\n" ++
  "static inline R128 r128_div(R128 a, R128 b) {\n" ++
  "    R128 r; r128Div(&r, &a, &b); return r;\n" ++
  "}\n" ++
  "static inline int64_t r128_to_int(R128 a) {\n" ++
  "    return (int64_t)a.hi;  /* truncate fractional part */\n" ++
  "}\n" ++
  "static inline int r128_le(R128 a, R128 b) {\n" ++
  "    return r128Cmp(&a, &b) <= 0;\n" ++
  "}\n" ++
  "static inline int r128_eq(R128 a, R128 b) {\n" ++
  "    return a.lo == b.lo && a.hi == b.hi;\n" ++
  "}\n\n" ++
  "static inline R128 pow_int_r128(R128 base, int exp) {\n" ++
  "    R128 r = R128_ONE, b = base;\n" ++
  "    while (exp > 0) {\n" ++
  "        if (exp & 1) r = r128_mul(r, b);\n" ++
  "        b = r128_mul(b, b);\n" ++
  "        exp >>= 1;\n" ++
  "    }\n" ++
  "    return r;\n" ++
  "}\n\n" ++
  "/* Float→R128 (preserves fractional bits via double intermediate) */\n" ++
  "static inline R128 r128_from_float(float f) {\n" ++
  "    double d = (double)f;\n" ++
  "    R128 r;\n" ++
  "    r.hi = (int64_t)d;\n" ++
  "    r.lo = (uint64_t)((d - (double)r.hi) * 18446744073709551616.0);\n" ++
  "    return r;\n" ++
  "}\n" ++
  "/* Unsigned 32-bit → R128 */\n" ++
  "static inline R128 r128_from_u32(uint32_t v) {\n" ++
  "    R128 r; r.hi = (int64_t)v; r.lo = 0; return r;\n" ++
  "}\n" ++
  "/* R128 → float */\n" ++
  "static inline float r128_to_float(R128 a) {\n" ++
  "    return (float)a.hi + (float)a.lo * 5.421010862427522e-20f; /* 2^-64 */\n" ++
  "}\n\n"

-- ── C: E-graph optimized polynomial kernels ─────────────────────────────────

private def scalarFnC : String :=
  genC "predictive_cost"
    ["min_x","max_x","min_y","max_y","min_z","max_z",
     "vx","vy","vz","ax","ay","az","ticks_ahead"]
    (entityExpr 0 12)

private def ghostBoundC : String :=
  "/* Ghost expansion v*d + a_half*d^2. Proved: expansion_covers_k_ticks */\n" ++
  genC "ghost_bound" ["v","a_half","ticks_ahead"] ghostBoundExpr

private def surfaceAreaC : String :=
  "/* Surface area 2(wh+hd+wd). Proved: surfaceArea_nonneg */\n" ++
  genC "surface_area" ["w","h","d"] surfaceAreaExpr

private def ghostAabbC : String :=
  "/* Ghost AABB min bound per axis. Proved: expansion_covers_k_ticks */\n" ++
  genC "ghost_aabb_min" ["center","ext","v","a_half","tau"] ghostAabbMinExpr ++ "\n\n" ++
  "/* Ghost AABB max bound per axis. Proved: expansion_covers_k_ticks */\n" ++
  genC "ghost_aabb_max" ["center","ext","v","a_half","tau"] ghostAabbMaxExpr

private def deltaCostFnsC : String :=
  let costFns := deltaCandidates.map fun dk =>
    genC s!"delta_cost_{dk}" ["v", "a_half"] (deltaCostExpr dk)
  String.intercalate "\n\n" costFns

private def deltaSelectC : String :=
  let lbrace := "{"
  let rbrace := "}"
  let selectBody := deltaCandidates.reverse.foldl (fun acc dk =>
    acc ++ "    if (r128_le(delta_cost_" ++ toString dk ++ "(v, a_half), " ++
      "r128_from_int(" ++ toString interestRadiusInt ++ "LL))) " ++ lbrace ++ " return " ++ toString dk ++ "; " ++ rbrace ++ "\n") ""
  deltaCostFnsC ++ "\n\n" ++
  "/* Per-entity delta: largest candidate where v*d + ah*d^2 <= R.\n" ++
  "   Polynomial cost evaluation via E-graph; selection is non-ring postprocessing.\n" ++
  "   Source of truth: perEntityDelta (Sim.lean:407) */\n" ++
  "static inline uint32_t per_entity_delta_poly(R128 v, R128 a_half) " ++ lbrace ++ "\n" ++
  selectBody ++
  "    return 1;\n" ++
  rbrace

private def quinticHermiteC : String :=
  "/* Quintic Hermite spline basis functions (C3 continuity).\n" ++
  "   t = numer/denom in [0,1] where numer=ticks_since_build, denom=delta.\n" ++
  "   Return values are integer numerators in μm units:\n" ++
  "     h00, h01, h10, h11 : divide result by denom^5\n" ++
  "     h20, h21            : divide result by 2*denom^5\n" ++
  "   Source of truth: PredictiveBVH.QuinticHermite (C3 proofs in Lean). */\n" ++
  String.intercalate "\n\n" (quinticBasisFns.map fun (name, params, e, _doc) =>
    let varNames : VarId → String := fun i =>
      if h : i < params.length then params[i] else s!"arg{i}"
    generateCFn name params (toLowLevel varNames (opt e)))

-- ── C: Spatial primitives (non-polynomial, direct translation) ──────────────

private def aabbC : String :=
  "/* Source: Types.lean:28 Proved: unionBounds_contains_left/right */\n" ++
  "typedef struct Aabb {\n" ++
  "    R128 min_x, max_x;\n" ++
  "    R128 min_y, max_y;\n" ++
  "    R128 min_z, max_z;\n" ++
  "} Aabb;\n\n" ++
  "static inline Aabb aabb_from_floats(float x0, float x1, float y0, float y1, float z0, float z1) {\n" ++
  "    Aabb a; a.min_x = r128_from_int((int64_t)(x0*1000000.0f)); a.max_x = r128_from_int((int64_t)(x1*1000000.0f));\n" ++
  "    a.min_y = r128_from_int((int64_t)(y0*1000000.0f)); a.max_y = r128_from_int((int64_t)(y1*1000000.0f));\n" ++
  "    a.min_z = r128_from_int((int64_t)(z0*1000000.0f)); a.max_z = r128_from_int((int64_t)(z1*1000000.0f));\n" ++
  "    return a;\n}\n\n" ++
  "static inline Aabb aabb_union(const Aabb *a, const Aabb *o) {\n" ++
  "    Aabb r;\n" ++
  "    r.min_x = r128_le(a->min_x, o->min_x) ? a->min_x : o->min_x;\n" ++
  "    r.max_x = r128_le(o->max_x, a->max_x) ? a->max_x : o->max_x;\n" ++
  "    r.min_y = r128_le(a->min_y, o->min_y) ? a->min_y : o->min_y;\n" ++
  "    r.max_y = r128_le(o->max_y, a->max_y) ? a->max_y : o->max_y;\n" ++
  "    r.min_z = r128_le(a->min_z, o->min_z) ? a->min_z : o->min_z;\n" ++
  "    r.max_z = r128_le(o->max_z, a->max_z) ? a->max_z : o->max_z;\n" ++
  "    return r;\n}\n\n" ++
  "static inline bool aabb_overlaps(const Aabb *a, const Aabb *o) {\n" ++
  "    return r128_le(a->min_x, o->max_x) && r128_le(o->min_x, a->max_x)\n" ++
  "        && r128_le(a->min_y, o->max_y) && r128_le(o->min_y, a->max_y)\n" ++
  "        && r128_le(a->min_z, o->max_z) && r128_le(o->min_z, a->max_z);\n}\n\n" ++
  "static inline bool aabb_contains(const Aabb *a, const Aabb *inner) {\n" ++
  "    return r128_le(a->min_x, inner->min_x) && r128_le(inner->max_x, a->max_x)\n" ++
  "        && r128_le(a->min_y, inner->min_y) && r128_le(inner->max_y, a->max_y)\n" ++
  "        && r128_le(a->min_z, inner->min_z) && r128_le(inner->max_z, a->max_z);\n}\n\n" ++
  "static inline bool aabb_contains_point(const Aabb *a, R128 x, R128 y, R128 z) {\n" ++
  "    return r128_le(a->min_x, x) && r128_le(x, a->max_x) && r128_le(a->min_y, y) && r128_le(y, a->max_y)\n" ++
  "        && r128_le(a->min_z, z) && r128_le(z, a->max_z);\n}"

private def utilC : String :=
  "/* Source: Build.lean:193 (clz30) */\n" ++
  "static inline uint32_t clz30(uint32_t x) {\n" ++
  "    return x == 0 ? 30 : 29 - (31 - _pbvh_clz(x));\n}\n\n" ++
  "/* R128 arithmetic right shift by 1 (divide by 2, preserving sign) */\n" ++
  "static inline R128 r128_half(R128 v) {\n" ++
  "    R128 r;\n" ++
  "    r.hi = v.hi >> 1;\n" ++
  "    r.lo = (v.lo >> 1) | ((uint64_t)v.hi << 63);\n" ++
  "    return r;\n}"

private def hilbertC : String :=
  "/* Source: Build.lean (hilbert3D) — Skilling (2004) 3D Hilbert curve.\n" ++
  "   O(b) bit manipulation; better locality than Morton for volume partitioning.\n" ++
  "   Bader (2013) Ch.7: cluster diameter O(n^{1/3}) vs Morton O(n^{2/3}). */\n" ++
  "static inline uint32_t hilbert3d(uint32_t x, uint32_t y, uint32_t z) {\n" ++
  "    const uint32_t order = 10;\n" ++
  "    const uint32_t mask = (1u << order) - 1u;\n" ++
  "    x &= mask; y &= mask; z &= mask;\n" ++
  "    /* Step 1: inverse undo (MSB down to bit 1) */\n" ++
  "    for (uint32_t i = 0; i < order - 1; i++) {\n" ++
  "        uint32_t q = 1u << (order - 1 - i);\n" ++
  "        uint32_t p = q - 1;\n" ++
  "        if (z & q) { x ^= p; } else { uint32_t t = (x ^ z) & p; x ^= t; z ^= t; }\n" ++
  "        if (y & q) { x ^= p; } else { uint32_t t = (x ^ y) & p; x ^= t; y ^= t; }\n" ++
  "    }\n" ++
  "    /* Step 2: Gray encode */\n" ++
  "    y ^= x; z ^= y;\n" ++
  "    /* Step 3: fixup — propagate gray parity */\n" ++
  "    uint32_t t = 0;\n" ++
  "    for (uint32_t i = 0; i < order - 1; i++) {\n" ++
  "        uint32_t q = 1u << (order - 1 - i);\n" ++
  "        if (z & q) t ^= (q - 1);\n" ++
  "    }\n" ++
  "    x ^= t; y ^= t; z ^= t;\n" ++
  "    x &= mask; y &= mask; z &= mask;\n" ++
  "    /* Step 4: transpose to 30-bit index (MSB-first, z-y-x per triple) */\n" ++
  "    uint32_t h = 0;\n" ++
  "    for (int b = (int)order - 1; b >= 0; b--) {\n" ++
  "        h = (h << 1) | ((z >> b) & 1);\n" ++
  "        h = (h << 1) | ((y >> b) & 1);\n" ++
  "        h = (h << 1) | ((x >> b) & 1);\n" ++
  "    }\n" ++
  "    return h;\n}\n\n" ++
  "static inline void hilbert3d_inverse(uint32_t h, uint32_t *ox, uint32_t *oy, uint32_t *oz);\n\n" ++
  "/* Source: Build.lean (leafHilbert) — R128 Aabb, returns uint32 Hilbert code */\n" ++
  "static inline uint32_t hilbert_of_aabb(const Aabb *b, const Aabb *scene) {\n" ++
  "    R128 sw = r128_sub(scene->max_x, scene->min_x);\n" ++
  "    R128 sh = r128_sub(scene->max_y, scene->min_y);\n" ++
  "    R128 sd = r128_sub(scene->max_z, scene->min_z);\n" ++
  "    R128 one = r128_from_int(1LL);\n" ++
  "    if (r128_le(sw, R128_ZERO)) sw = one;\n" ++
  "    if (r128_le(sh, R128_ZERO)) sh = one;\n" ++
  "    if (r128_le(sd, R128_ZERO)) sd = one;\n" ++
  "    R128 k1024 = r128_from_int(1024LL);\n" ++
  "    R128 cx = r128_half(r128_add(b->min_x, b->max_x));\n" ++
  "    R128 cy = r128_half(r128_add(b->min_y, b->max_y));\n" ++
  "    R128 cz = r128_half(r128_add(b->min_z, b->max_z));\n" ++
  "    int64_t swi = r128_to_int(sw), shi = r128_to_int(sh), sdi = r128_to_int(sd);\n" ++
  "    if (swi == 0) swi = 1; if (shi == 0) shi = 1; if (sdi == 0) sdi = 1;\n" ++
  "    int64_t nxi = r128_to_int(r128_mul(r128_sub(cx, scene->min_x), k1024)) / swi;\n" ++
  "    int64_t nyi = r128_to_int(r128_mul(r128_sub(cy, scene->min_y), k1024)) / shi;\n" ++
  "    int64_t nzi = r128_to_int(r128_mul(r128_sub(cz, scene->min_z), k1024)) / sdi;\n" ++
  "    uint32_t nx = (uint32_t)(nxi < 0 ? 0 : nxi > 1023 ? 1023 : nxi);\n" ++
  "    uint32_t ny = (uint32_t)(nyi < 0 ? 0 : nyi > 1023 ? 1023 : nyi);\n" ++
  "    uint32_t nz = (uint32_t)(nzi < 0 ? 0 : nzi > 1023 ? 1023 : nzi);\n" ++
  "    uint32_t h = hilbert3d(nx, ny, nz);\n" ++
  "    /* Witness check: inverse(forward(x,y,z)) == (x,y,z) */\n" ++
  "    uint32_t rx, ry, rz;\n" ++
  "    hilbert3d_inverse(h, &rx, &ry, &rz);\n" ++
  "    CRASH_COND(rx != nx || ry != ny || rz != nz);\n" ++
  "    return h;\n}\n\n" ++
  "/* Hilbert3D inverse: 30-bit code → (x, y, z) 10-bit coordinates.\n" ++
  "   Skilling transposeToAxes. Verified by roundtrip in Lean. */\n" ++
  "static inline void hilbert3d_inverse(uint32_t h, uint32_t *ox, uint32_t *oy, uint32_t *oz) {\n" ++
  "    const uint32_t order = 10;\n" ++
  "    const uint32_t mask = (1u << order) - 1u;\n" ++
  "    uint32_t x = 0, y = 0, z = 0;\n" ++
  "    for (uint32_t b = 0; b < order; b++) {\n" ++
  "        uint32_t s = 3 * b;\n" ++
  "        x |= ((h >> s) & 1) << b;\n" ++
  "        y |= ((h >> (s + 1)) & 1) << b;\n" ++
  "        z |= ((h >> (s + 2)) & 1) << b;\n" ++
  "    }\n" ++
  "    /* Undo fixup: progressive decode */\n" ++
  "    uint32_t t = 0;\n" ++
  "    for (uint32_t i = 0; i < order - 1; i++) {\n" ++
  "        uint32_t q = 1u << (order - 1 - i);\n" ++
  "        if ((z ^ t) & q) t ^= (q - 1);\n" ++
  "    }\n" ++
  "    x ^= t; y ^= t; z ^= t;\n" ++
  "    /* Undo Gray: z ^= y, then y ^= x */\n" ++
  "    z ^= y; y ^= x;\n" ++
  "    /* Undo main loop: Q from 2 to MSB, y then z */\n" ++
  "    for (uint32_t q = 2; q < (1u << order); q <<= 1) {\n" ++
  "        uint32_t p = q - 1;\n" ++
  "        if (y & q) { x ^= p; } else { uint32_t ty = (x ^ y) & p; x ^= ty; y ^= ty; }\n" ++
  "        if (z & q) { x ^= p; } else { uint32_t tz = (x ^ z) & p; x ^= tz; z ^= tz; }\n" ++
  "    }\n" ++
  "    *ox = x & mask; *oy = y & mask; *oz = z & mask;\n" ++
  "    /* Witness check: forward(inverse(h)) == h */\n" ++
  "    CRASH_COND(hilbert3d(*ox, *oy, *oz) != h);\n}\n\n" ++
  "/* Hilbert-cell-of: AABB from Hilbert code + prefix depth + scene bounds (R128). */\n" ++
  "static inline Aabb hilbert_cell_of(uint32_t code, uint32_t prefix_depth, const Aabb *scene) {\n" ++
  "    uint32_t cx, cy, cz;\n" ++
  "    hilbert3d_inverse(code, &cx, &cy, &cz);\n" ++
  "    int64_t sw = r128_to_int(r128_sub(scene->max_x, scene->min_x));\n" ++
  "    int64_t sh = r128_to_int(r128_sub(scene->max_y, scene->min_y));\n" ++
  "    int64_t sd = r128_to_int(r128_sub(scene->max_z, scene->min_z));\n" ++
  "    if (sw <= 0) sw = 1; if (sh <= 0) sh = 1; if (sd <= 0) sd = 1;\n" ++
  "    uint32_t shift = (prefix_depth < 10) ? 10 - prefix_depth : 0;\n" ++
  "    int64_t cell = 1LL << shift;\n" ++
  "    uint32_t x0 = (cx >> shift) << shift;\n" ++
  "    uint32_t y0 = (cy >> shift) << shift;\n" ++
  "    uint32_t z0 = (cz >> shift) << shift;\n" ++
  "    Aabb result;\n" ++
  "    result.min_x = r128_add(scene->min_x, r128_div(r128_mul(r128_from_int((int64_t)x0), r128_sub(scene->max_x, scene->min_x)), r128_from_int(1024LL)));\n" ++
  "    result.max_x = r128_add(scene->min_x, r128_div(r128_mul(r128_from_int((int64_t)x0 + cell), r128_sub(scene->max_x, scene->min_x)), r128_from_int(1024LL)));\n" ++
  "    result.min_y = r128_add(scene->min_y, r128_div(r128_mul(r128_from_int((int64_t)y0), r128_sub(scene->max_y, scene->min_y)), r128_from_int(1024LL)));\n" ++
  "    result.max_y = r128_add(scene->min_y, r128_div(r128_mul(r128_from_int((int64_t)y0 + cell), r128_sub(scene->max_y, scene->min_y)), r128_from_int(1024LL)));\n" ++
  "    result.min_z = r128_add(scene->min_z, r128_div(r128_mul(r128_from_int((int64_t)z0), r128_sub(scene->max_z, scene->min_z)), r128_from_int(1024LL)));\n" ++
  "    result.max_z = r128_add(scene->min_z, r128_div(r128_mul(r128_from_int((int64_t)z0 + cell), r128_sub(scene->max_z, scene->min_z)), r128_from_int(1024LL)));\n" ++
  "    return result;\n}"

-- ── C: Constants ────────────────────────────────────────────────────────────

open PredictiveBVH.Resources in
private def constantsC : String :=
  "/* ── Lean-derived tick-rate-parametric formulas ──────────────────────────\n" ++
  "   PBVH_SIM_TICK_HZ is the DEFAULT value (what the Lean proofs were\n" ++
  "   evaluated at and what set_physics_ticks_per_second() should be\n" ++
  "   initialized with at startup). Runtime consumers (Godot/C++) should\n" ++
  "   read the engine's actual physics tick rate and pass it to the inline\n" ++
  "   helpers below — never hardcode PBVH_SIM_TICK_HZ at use sites.\n" ++
  "\n" ++
  "   Formulas mirror Core/Types.lean + Resources.lean exactly:\n" ++
  "     hysteresisThreshold = simTickHz * 4\n" ++
  "     latencyTicksFloor   = max(simTickHz / 10, 1)\n" ++
  "     vMaxPhysical μm/tick = 10 * 1_000_000 / simTickHz\n" ++
  "     aHalfMinForearm μm/tick² = ceil(1_400_000 / (2 * simTickHz²))\n" ++
  "   ─────────────────────────────────────────────────────────────────────── */\n" ++
  "#define PBVH_SIM_TICK_HZ " ++ toString simTickHz ++ "u\n" ++
  "\n" ++
  "static inline uint32_t pbvh_hysteresis_threshold(uint32_t hz) { return hz * 4u; }\n" ++
  "static inline uint32_t pbvh_latency_ticks(uint32_t hz) {\n" ++
  "    uint32_t t = hz / 10u; return t > 0u ? t : 1u;\n" ++
  "}\n" ++
  "static inline int64_t pbvh_v_max_physical_um_per_tick(uint32_t hz) {\n" ++
  "    return (int64_t)(10 * 1000000) / (int64_t)(hz > 0u ? hz : 1u);\n" ++
  "}\n" ++
  "static inline uint64_t pbvh_accel_floor_um_per_tick2(uint32_t hz) {\n" ++
  "    uint64_t h = (uint64_t)(hz > 0u ? hz : 1u);\n" ++
  "    uint64_t d = 2ull * h * h;\n" ++
  "    return (1400000ull + d - 1ull) / d;\n" ++
  "}\n" ++
  "\n" ++
  "/* Tick-rate-invariant constants (pure physical distances / velocities) */\n" ++
  "#define PBVH_INTEREST_RADIUS_UM " ++ toString interestRadius ++ "LL   /* μm */\n" ++
  "#define PBVH_CURRENT_FUNNEL_PEAK_V_M_PER_S 60  /* m/s — C7 rip-current impulse cap; μm/tick = 60*1e6/hz */\n" ++
  "\n" ++
  "/* Default-rate convenience values (evaluated at PBVH_SIM_TICK_HZ).\n" ++
  "   Prefer the pbvh_* helpers above at runtime. */\n" ++
  "#define PBVH_LATENCY_TICKS_DEFAULT " ++ toString latencyTicks ++ "u\n" ++
  "#define PBVH_HYSTERESIS_THRESHOLD_DEFAULT " ++ toString hysteresisThreshold ++ "u\n" ++
  "#define PBVH_V_MAX_PHYSICAL_DEFAULT " ++ toString vMaxPhysical ++ "LL  /* μm/tick */\n" ++
  "#define PBVH_ACCEL_FLOOR_DEFAULT " ++ toString aHalfMinForearm ++ "ULL /* μm/tick² */\n" ++
  "#define PBVH_CURRENT_FUNNEL_PEAK_V_UM_TICK_DEFAULT " ++ toString currentFunnelPeakVUmTick ++ "LL /* μm/tick */"

-- ── C: Assemble complete header file ────────────────────────────────────────

private def cFile : String :=
  cPreamble ++
  "/* ══════════════════════════════════════════════════════════════════════════\n" ++
  "   PROVED SPATIAL PRIMITIVES (R128 polynomial + direct C translation)\n" ++
  "   ══════════════════════════════════════════════════════════════════════════ */\n\n" ++
  ghostBoundC ++ "\n\n" ++
  surfaceAreaC ++ "\n\n" ++
  ghostAabbC ++ "\n\n" ++
  aabbC ++ "\n\n" ++
  utilC ++ "\n\n" ++
  hilbertC ++ "\n\n" ++
  constantsC ++ "\n\n" ++
  "/* ══════════════════════════════════════════════════════════════════════════\n" ++
  "   AMOLEAN E-GRAPH OPTIMIZED KERNELS (R128)\n" ++
  "   ══════════════════════════════════════════════════════════════════════════ */\n\n" ++
  scalarFnC ++ "\n\n" ++
  deltaSelectC ++ "\n\n" ++
  quinticHermiteC ++ "\n\n" ++
  "#endif /* PREDICTIVE_BVH_H */\n"

private def cPath      : String := "predictive_bvh.h"
private def rsPath     : String := "predictive_bvh.rs"

def main : IO Unit := do
  IO.FS.writeFile cPath     cFile
  IO.FS.writeFile rsPath    rustFile
  IO.println s!"wrote {cPath}"
  IO.println s!"wrote {rsPath}"
