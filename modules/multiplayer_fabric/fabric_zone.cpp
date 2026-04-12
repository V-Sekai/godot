/**************************************************************************/
/*  fabric_zone.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "fabric_zone.h"

#include "fabric_snapshot.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/typedefs.h"
#include "core/variant/variant.h"

#include <cstring> // memcpy

// ── Static peer callback table — filled by multiplayer_fabric module ─────────
FabricZonePeerCallbacks FabricZone::peer_callbacks;

void FabricZone::register_peer_callbacks(const FabricZonePeerCallbacks &p_cb) {
	peer_callbacks = p_cb;
}

// Channel IDs (mirrored from fabric_multiplayer_peer.h)
// Godot's ENetMultiplayerPeer reserves ENet wire channels 0 (SYSCH_RELIABLE) and 1
// (SYSCH_UNRELIABLE) for internal system messages. set_transfer_channel(N) maps to
// ENet wire channel SYSCH_MAX+N-1 = N+1, so N=0 → wire ch 1 (swallowed as system msg).
// Use N≥1 to land on wire channels ≥2 which are safe user channels.
static constexpr int CH_MIGRATION = 1; // → ENet wire ch 2, reliable STAGING intents
static constexpr int CH_INTEREST = 2; // → ENet wire ch 3, unreliable entity snapshots
static constexpr int CH_PLAYER = 3; // → ENet wire ch 4, unreliable player state

// Runtime physics tick rate — the single authoritative "hz" source for tick-rate
// arithmetic. set_physics_ticks_per_second() is initialized to PBVH_SIM_TICK_HZ
// at startup; after that, all tick-rate-dependent calculations go through this
// accessor, so the values in this file never duplicate what the engine already knows.
static inline uint32_t _fz_hz() {
	return (uint32_t)Engine::get_singleton()->get_physics_ticks_per_second();
}

// ── Diagnostic knobs for jellyfish_zone_crossing minimal-repro tiers ────────
// s_xing_count: how many of the XING_ID_LO..XING_ID_HI tracers use the
//   crossing behavior. Remaining IDs in the range fall through to the
//   centering-drift branch and stay in their home zone.
// s_xing_mode:
//   0 = default (bell-jet + clump + cohesion, the 144-tracer chokepoint)
//   1 = fixed linear trajectory (tier 0): constant velocity, one crossing,
//       no bell, no spring, no jitter. Isolates the migration state machine.
//   2 = oscillating (tier 1): sinusoidal velocity that overshoots and
//       reverses. Isolates hysteresis reset-on-return behavior.
static int s_xing_count = 144;
static int s_xing_mode = 0;

// ══════════════════════════════════════════════════════════════════════════════
// Entity stepping — scenario functions (translated from Sim.lean via Rust)
// ══════════════════════════════════════════════════════════════════════════════

// ── Stepping helper: clamp position to bounds, bounce velocity ──────────────

static void _clamp_bounce(real_t &cx, real_t &nvx, real_t bound) {
	if (cx > bound) {
		cx = bound;
		nvx = -Math::abs(nvx);
	}
	if (cx < -bound) {
		cx = -bound;
		nvx = Math::abs(nvx);
	}
}

void FabricZone::_step_entity(FabricEntity &e, uint32_t p_tick, Scenario p_scenario, int p_n,
		const real_t p_waypoints[][3], int p_zone_count, const real_t *p_local_clump) {
	// Stroke knots: spring toward their spawn anchor + heavy damping.
	// Physics IS applied (not skipped) — keeps ghost-bound proofs valid.
	// Spring coefficient 0.05 + damping 0.85 → knot stays within ~0.02 m of anchor.
	if ((uint32_t)e.global_id >= (uint32_t)STROKE_ENTITY_BASE) {
		const real_t FIXED_SCALE = SIM_BOUND / 32767.0f;
		int16_t ax16, ay16, az16;
		memcpy(&ax16, (const uint8_t *)&e.payload[2], 2);
		memcpy(&ay16, (const uint8_t *)&e.payload[2] + 2, 2);
		memcpy(&az16, (const uint8_t *)&e.payload[3], 2);
		real_t anc_x = (real_t)ax16 * FIXED_SCALE;
		real_t anc_y = (real_t)ay16 * FIXED_SCALE;
		real_t anc_z = (real_t)az16 * FIXED_SCALE;
		real_t fx = (anc_x - e.cx) * 0.05f;
		real_t fy = (anc_y - e.cy) * 0.05f;
		real_t fz = (anc_z - e.cz) * 0.05f;
		real_t nvx = CLAMP((e.vx + fx) * 0.85f, -0.02f, 0.02f);
		real_t nvy = CLAMP((e.vy + fy) * 0.85f, -0.02f, 0.02f);
		real_t nvz = CLAMP((e.vz + fz) * 0.85f, -0.02f, 0.02f);
		e.ax = nvx - e.vx;
		e.ay = nvy - e.vy;
		e.az = nvz - e.vz;
		e.cx += nvx;
		e.cy += nvy;
		e.cz += nvz;
		e.vx = nvx;
		e.vy = nvy;
		e.vz = nvz;
		return;
	}

	switch (p_scenario) {
		case SCENARIO_JELLYFISH_BLOOM: {
			// ~1 Hz beat with a short pulse (~80 ms). Both expressed in terms of
			// the engine's physics tick rate so the rhythm is wall-clock stable.
			const uint32_t hz = _fz_hz();
			const uint32_t beat_period = hz * (uint32_t)BLOOM_BEAT_PERIOD_SECONDS;
			const uint32_t beat_on = MAX((BLOOM_BEAT_ON_MS * hz + 999u) / 1000u, 1u);
			real_t rx = -(e.cx / 0.8f);
			real_t ry = -(e.cy / 0.8f);
			real_t beat_ax = (p_tick % beat_period < beat_on) ? -(e.cx / 0.2f) : 0.0f;
			real_t beat_ay = (p_tick % beat_period < beat_on) ? -(e.cy / 0.2f) : 0.0f;
			bool gate = (((int)p_tick * 7 + e.global_id * 13) % 8) == 0;
			real_t kx = gate ? (real_t)(((int)p_tick * 3 + e.global_id * 11) % 25 - 12) * 0.001f : 0.0f;
			real_t ky = gate ? (real_t)(((int)p_tick * 5 + e.global_id * 17) % 25 - 12) * 0.001f : 0.0f;
			real_t vx = CLAMP(e.vx + kx + rx + beat_ax, -V_MAX, V_MAX);
			real_t vy = CLAMP(e.vy + ky + ry + beat_ay, -V_MAX, V_MAX);
			real_t vz = CLAMP(e.vz, -0.1f, 0.1f);
			real_t cx = e.cx + vx;
			real_t cy = e.cy + vy;
			real_t cz = e.cz + vz;
			real_t nvx = vx, nvy = vy, nvz = vz;
			_clamp_bounce(cx, nvx, 3.0f); // 6m x 6m venue
			_clamp_bounce(cy, nvy, 3.0f);
			_clamp_bounce(cz, nvz, 0.3f);
			e.ax = nvx - e.vx;
			e.ay = nvy - e.vy;
			e.az = nvz - e.vz;
			e.cx = cx;
			e.cy = cy;
			e.cz = cz;
			e.vx = nvx;
			e.vy = nvy;
			e.vz = nvz;
		} break;

		case SCENARIO_WHALE_WITH_SHARKS: {
			// Multi-cabin articulated vehicle (train) with passengers.
			// Each 14-entity group is one train:
			//   slot 0..2  → three cabins, slot 0 is the lead cabin.
			//                Each trailing cabin samples the lead-cabin path at an
			//                earlier effective tick so it trails spatially.
			//   slot 3..6  → 4 passengers riding cabin 0
			//   slot 7..9  → 3 passengers riding cabin 1
			//   slot 10..13→ 4 passengers riding cabin 2
			// Passengers steer toward their assigned cabin's target with a small
			// transverse offset so they sit in distinguishable seats.
			// Leader path is a pure function of (vehicle_idx, tick, hz) so followers
			// read no other entity's state — slot-order-independent.
			const uint32_t hz = _fz_hz();
			const int phase_period = MAX((int)(hz * (uint32_t)WHALE_PHASE_PERIOD_SECONDS), 1);
			const int group_size = 14;
			const int vehicle_idx = e.global_id / group_size;
			const int slot = e.global_id % group_size;
			// No output velocity clamp: terminal velocity is the natural equilibrium
			// between bounded steering force and drag. A C7 impulse (up to
			// CURRENT_FUNNEL_PEAK_V) simply rides through and decays over ~1 second,
			// so the BVH observes the true peak magnitude instead of a truncated value.
			// DRAG chosen so |steer_max| / (1 - DRAG) ≈ V_MAX (terminal ≈ 10 m/s).
			const real_t DRAG = 0.92f;

			// Role dispatch: cabin_idx ∈ {0,1,2}; is_cabin distinguishes leader body from rider.
			int cabin_idx;
			int passenger_seat = 0;
			bool is_cabin;
			if (slot < 3) {
				cabin_idx = slot;
				is_cabin = true;
			} else if (slot <= 6) {
				cabin_idx = 0;
				passenger_seat = slot - 3;
				is_cabin = false;
			} else if (slot <= 9) {
				cabin_idx = 1;
				passenger_seat = slot - 7;
				is_cabin = false;
			} else {
				cabin_idx = 2;
				passenger_seat = slot - 10;
				is_cabin = false;
			}

			// Each trailing cabin lags the lead cabin by a quarter-phase in tick-time.
			const int cabin_delay_ticks = phase_period / 4;
			int effective_tick = (int)p_tick - cabin_idx * cabin_delay_ticks;
			int phase = ((effective_tick + vehicle_idx * phase_period) / phase_period) % 4;
			if (phase < 0) {
				phase += 4;
			}
			real_t tx = (phase == 0) ? 10.0f : (phase == 2) ? -10.0f
															: 0.0f;
			real_t ty = (phase == 1) ? 10.0f : (phase == 3) ? -10.0f
															: 0.0f;
			if (!is_cabin) {
				// Distribute passengers in ~0.3 m seat offsets transverse to the cabin.
				const real_t seat_offset = ((real_t)passenger_seat - 1.5f) * 0.3f;
				tx += seat_offset;
				ty += seat_offset * 0.5f;
			}

			// Bounded steering forces model finite actuator strength (a motor with max
			// torque). This is a physical input, not a velocity truncation.
			real_t steer_x = CLAMP((tx - e.cx) / 0.1f, -0.04f, 0.04f);
			real_t steer_y = CLAMP((ty - e.cy) / 0.1f, -0.04f, 0.04f);
			real_t vx = (e.vx + steer_x) * DRAG;
			real_t vy = (e.vy + steer_y) * DRAG;
			real_t vz = e.vz * DRAG;
			real_t cx = e.cx + vx;
			real_t cy = e.cy + vy;
			real_t cz = e.cz + vz;
			real_t nvx = vx, nvy = vy, nvz = vz;
			_clamp_bounce(cx, nvx, SIM_BOUND);
			_clamp_bounce(cy, nvy, SIM_BOUND);
			_clamp_bounce(cz, nvz, SIM_BOUND);
			e.ax = nvx - e.vx;
			e.ay = nvy - e.vy;
			e.az = nvz - e.vz;
			e.cx = cx;
			e.cy = cy;
			e.cz = cz;
			e.vx = nvx;
			e.vy = nvy;
			e.vz = nvz;
		} break;

		case SCENARIO_CURRENT_FUNNEL: {
			const uint32_t prone_period = _fz_hz() * (uint32_t)FUNNEL_PRONE_PERIOD_SECONDS;
			bool is_prone = (p_tick / prone_period + (uint32_t)e.global_id) % 4 == 3;
			bool just_entered = is_prone && ((p_tick - 1) / prone_period + (uint32_t)e.global_id) % 4 != 3;
			bool spike = just_entered && ((e.global_id * 7 + (int)p_tick) % 10 == 0);
			real_t v_cap = spike ? CURRENT_FUNNEL_PEAK_V : V_MAX;
			real_t rx = -(e.cx / 0.8f);
			real_t ry = -(e.cy / 0.8f);
			real_t kx = (real_t)(((int)p_tick * 7 + e.global_id * 13) % 25 - 12) * 0.001f;
			real_t ky = (real_t)(((int)p_tick * 11 + e.global_id * 17) % 25 - 12) * 0.001f;
			real_t vx = CLAMP(e.vx + kx + rx, -v_cap, v_cap);
			real_t vy = CLAMP(e.vy + ky + ry, -v_cap, v_cap);
			real_t vz = CLAMP(e.vz, -0.1f, 0.1f);
			real_t cx = e.cx + vx;
			real_t cy = e.cy + vy;
			real_t cz = e.cz + vz;
			real_t nvx = vx, nvy = vy, nvz = vz;
			_clamp_bounce(cx, nvx, SIM_BOUND);
			_clamp_bounce(cy, nvy, SIM_BOUND);
			_clamp_bounce(cz, nvz, SIM_BOUND);
			e.ax = nvx - e.vx;
			e.ay = nvy - e.vy;
			e.az = nvz - e.vz;
			e.cx = cx;
			e.cy = cy;
			e.cz = cz;
			e.vx = nvx;
			e.vy = nvy;
			e.vz = nvz;
		} break;

		case SCENARIO_DEFAULT:
			// Default: gentle centering drift, no zone-crossing intent.
			{
				real_t rx = -(e.cx / 0.8f);
				real_t ry = -(e.cy / 0.8f);
				real_t kx = (real_t)(((int)p_tick * 7 + e.global_id * 13) % 25 - 12) * 0.001f;
				real_t ky = (real_t)(((int)p_tick * 11 + e.global_id * 17) % 25 - 12) * 0.001f;
				real_t vx = CLAMP(e.vx + kx + rx, -V_MAX, V_MAX);
				real_t vy = CLAMP(e.vy + ky + ry, -V_MAX, V_MAX);
				real_t vz = CLAMP(e.vz, -0.1f, 0.1f);
				real_t cx = e.cx + vx;
				real_t cy = e.cy + vy;
				real_t cz = e.cz + vz;
				real_t nvx = vx, nvy = vy, nvz = vz;
				_clamp_bounce(cx, nvx, SIM_BOUND);
				_clamp_bounce(cy, nvy, SIM_BOUND);
				_clamp_bounce(cz, nvz, SIM_BOUND);
				e.ax = nvx - e.vx;
				e.ay = nvy - e.vy;
				e.az = nvz - e.vz;
				e.cx = cx;
				e.cy = cy;
				e.cz = cz;
				e.vx = nvx;
				e.vy = nvy;
				e.vz = nvz;
			}
			break;

		case SCENARIO_JELLYFISH_ZONE_CROSSING: {
			// Chokepoint stress test — 144 jellyfish_zone_crossing entities (IDs 256-399)
			// clump-spring toward (0,0,0) and cross zone boundaries in a burst.
			// All other entities use default centering drift so zone density stays even:
			// only the 144 tracers move cross-zone; the rest stay in their home zone.
			const uint32_t hz = _fz_hz();
			const int BELL_PERIOD = (int)(hz * (uint32_t)JET_BELL_PERIOD_SECONDS);
			const int JET_TICKS_N = (int)MAX((JET_PULSE_MS * hz + 999u) / 1000u, 1u);
			const int WP_PERIOD_N = (int)(hz * (uint32_t)WP_PERIOD_SECONDS);
			static constexpr real_t JET_JITTER = 0.08f; // small per-entity noise
			static constexpr real_t CLUMP_K = 0.06f; // spring toward curtain
			static constexpr real_t DAMPING = 0.88f; // velocity decay

			if (e.global_id >= XING_ID_LO && e.global_id < XING_ID_LO + s_xing_count) {
				if (s_xing_mode == 1 || s_xing_mode == 2) {
					// ── Minimal-repro positions ──
					// Ignore the seeded position entirely — compute an absolute target
					// position from tick so both zones agree deterministically.
					//
					// Tier 0 (mode=1): linear ramp x = -5 + 0.1*t. Crosses x=0 at t=50,
					//   stays past the curtain forever. No bell, no spring, no jitter.
					//   Should land in Zone B with a single STAGING→OWNED transition.
					//
					// Tier 1 (mode=2): sinusoid x = 3 * sin(t/15). Period ~94 ticks ≈ 4.7 s.
					//   Overshoots past x=0 into Zone B then returns to Zone A. Probes
					//   hysteresis reset-on-return.
					const int32_t t = (int32_t)p_tick;
					real_t new_cx;
					if (s_xing_mode == 1) {
						new_cx = (real_t)(-5.0 + 0.1 * (double)t);
					} else {
						new_cx = (real_t)(3.0 * Math::sin((double)t / 15.0));
					}
					real_t new_vx = new_cx - e.cx;
					e.ax = new_vx - e.vx;
					e.ay = (real_t)0.0 - e.vy;
					e.az = (real_t)0.0 - e.vz;
					e.cx = new_cx;
					e.cy = 0.0f;
					e.cz = 0.0f;
					e.vx = new_vx;
					e.vy = 0.0f;
					e.vz = 0.0f;
					break;
				}
				// ── Crossing tracers: bell-jet clump, shuffled cyclic between zone centroids ──
				// Each WP_PERIOD half-cycle picks a target zone via LCG hash of the cycle
				// number — deterministic across all zone processes, different each cycle.
				// WP_PERIOD_SECONDS is proved valid by WaypointBound.lean (wpPeriodMin
				// is simTickHz-parametric; 10 s margin holds at any configured tick rate).
				static constexpr real_t COHESION_K = 0.03f; // spring toward local tracer clump
				uint32_t cycle = (uint32_t)((int)p_tick / WP_PERIOD_N);
				// LCG: well-distributed, cheap, same across all zone processes.
				int target_zone_idx = (int)((cycle * 1664525u + 1013904223u) % (uint32_t)MAX(p_zone_count, 1));
				real_t tx = p_waypoints ? p_waypoints[target_zone_idx][0] : 0.0f;
				real_t ty = p_waypoints ? p_waypoints[target_zone_idx][1] : 0.0f;
				real_t tz = p_waypoints ? p_waypoints[target_zone_idx][2] : 0.0f;

				// Shared bell phase — all fire together, maximizing burst.
				int bell_phase = (int)p_tick % BELL_PERIOD;
				bool jetting = bell_phase < JET_TICKS_N;

				real_t vx = e.vx * DAMPING;
				real_t vy = e.vy * DAMPING;
				real_t vz = e.vz * DAMPING;

				if (jetting) {
					// Tiny per-entity jitter prevents exact stacking but keeps the clump.
					real_t jx = (real_t)((e.global_id * 7 + bell_phase) % 9 - 4) * JET_JITTER;
					real_t jy = (real_t)((e.global_id * 11 + bell_phase) % 9 - 4) * JET_JITTER;
					real_t jz = (real_t)((e.global_id * 17 + bell_phase) % 9 - 4) * JET_JITTER;
					vx += jx;
					vy += jy;
					vz += jz;
				}

				// Migration spring: pull toward shuffled target zone centroid.
				vx += (tx - e.cx) * CLUMP_K;
				vy += (ty - e.cy) * CLUMP_K;
				vz += (tz - e.cz) * CLUMP_K;

				// Cohesion spring: pull toward local tracer clump centroid.
				// COHESION_K < CLUMP_K so migration wins at long range but
				// tracers attract each other at short range → natural clumping.
				if (p_local_clump) {
					vx += (p_local_clump[0] - e.cx) * COHESION_K;
					vy += (p_local_clump[1] - e.cy) * COHESION_K;
					vz += (p_local_clump[2] - e.cz) * COHESION_K;
				}

				vx = CLAMP(vx, -V_MAX, V_MAX);
				vy = CLAMP(vy, -V_MAX, V_MAX);
				vz = CLAMP(vz, -V_MAX, V_MAX);
				real_t cx = e.cx + vx;
				real_t cy = e.cy + vy;
				real_t cz = e.cz + vz;
				real_t nvx = vx, nvy = vy, nvz = vz;
				_clamp_bounce(cx, nvx, SIM_BOUND);
				_clamp_bounce(cy, nvy, SIM_BOUND);
				_clamp_bounce(cz, nvz, SIM_BOUND);
				e.ax = nvx - e.vx;
				e.ay = nvy - e.vy;
				e.az = nvz - e.vz;
				e.cx = cx;
				e.cy = cy;
				e.cz = cz;
				e.vx = nvx;
				e.vy = nvy;
				e.vz = nvz;
			} else {
				// ── Non-crossing entities: gentle centering drift, stay in home zone ──
				real_t rx = -(e.cx / 0.8f);
				real_t ry = -(e.cy / 0.8f);
				real_t kx = (real_t)(((int)p_tick * 7 + e.global_id * 13) % 25 - 12) * 0.001f;
				real_t ky = (real_t)(((int)p_tick * 11 + e.global_id * 17) % 25 - 12) * 0.001f;
				real_t vx = CLAMP(e.vx + kx + rx, -V_MAX, V_MAX);
				real_t vy = CLAMP(e.vy + ky + ry, -V_MAX, V_MAX);
				real_t vz = CLAMP(e.vz, -0.1f, 0.1f);
				real_t cx = e.cx + vx;
				real_t cy = e.cy + vy;
				real_t cz = e.cz + vz;
				real_t nvx = vx, nvy = vy, nvz = vz;
				_clamp_bounce(cx, nvx, SIM_BOUND);
				_clamp_bounce(cy, nvy, SIM_BOUND);
				_clamp_bounce(cz, nvz, SIM_BOUND);
				e.ax = nvx - e.vx;
				e.ay = nvy - e.vy;
				e.az = nvz - e.vz;
				e.cx = cx;
				e.cy = cy;
				e.cz = cz;
				e.vx = nvx;
				e.vy = nvy;
				e.vz = nvz;
			}
		} break;

		case SCENARIO_MIXED: {
			int n = p_n;
			int jellyfish_bloom_end = n / 2;
			int jellyfish_zone_crossing_end = jellyfish_bloom_end + n * 28 / 100;
			if (e.global_id < jellyfish_bloom_end) {
				_step_entity(e, p_tick, SCENARIO_JELLYFISH_BLOOM, n);
			} else if (e.global_id < jellyfish_zone_crossing_end) {
				_step_entity(e, p_tick, SCENARIO_CURRENT_FUNNEL, n);
			} else {
				_step_entity(e, p_tick, SCENARIO_WHALE_WITH_SHARKS, n);
			}
		} break;
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// Ghost snap system
// ══════════════════════════════════════════════════════════════════════════════

uint32_t FabricZone::_snap_delta(real_t v, real_t ah) {
	return per_entity_delta_poly(r128_from_real_um(v), r128_from_real_um(ah));
}

FabricZone::GhostSnap FabricZone::_make_ghost_snap(const FabricEntity &e) {
	real_t ahx = MAX(ACCEL_FLOOR_M, Math::abs(e.ax));
	real_t ahy = MAX(ACCEL_FLOOR_M, Math::abs(e.ay));
	real_t ahz = MAX(ACCEL_FLOOR_M, Math::abs(e.az));
	float v = MAX(MAX(Math::abs(e.vx), Math::abs(e.vy)), Math::abs(e.vz));
	real_t ah = MAX(MAX(ahx, ahy), ahz);
	uint32_t d = _snap_delta(v, ah);
	R128 r_dk = r128_from_u32(d);

	// Cache worst-case expansion (proved by expansion_covers_k_ticks). R128 in μm.
	R128 gex = ghost_bound(r128_from_real_um(Math::abs(e.vx)), r128_from_real_um(ahx), r_dk);
	R128 gey = ghost_bound(r128_from_real_um(Math::abs(e.vy)), r128_from_real_um(ahy), r_dk);
	R128 gez = ghost_bound(r128_from_real_um(Math::abs(e.vz)), r128_from_real_um(ahz), r_dk);

	// Build R128 ghost AABB (μm) and compute Hilbert code.
	R128 cx_um = r128_from_real_um(e.cx);
	R128 cy_um = r128_from_real_um(e.cy);
	R128 cz_um = r128_from_real_um(e.cz);
	Aabb ghost;
	ghost.min_x = r128_sub(cx_um, gex);
	ghost.max_x = r128_add(cx_um, gex);
	ghost.min_y = r128_sub(cy_um, gey);
	ghost.max_y = r128_add(cy_um, gey);
	ghost.min_z = r128_sub(cz_um, gez);
	ghost.max_z = r128_add(cz_um, gez);

	Aabb scene = aabb_from_floats(-SIM_BOUND, SIM_BOUND, -SIM_BOUND, SIM_BOUND, -SIM_BOUND, SIM_BOUND);
	uint32_t hcode = hilbert_of_aabb(&ghost, &scene);

	GhostSnap snap;
	snap.cx = e.cx;
	snap.cy = e.cy;
	snap.cz = e.cz;
	snap.vx = Math::abs(e.vx);
	snap.vy = Math::abs(e.vy);
	snap.vz = Math::abs(e.vz);
	snap.max_ahx = ahx;
	snap.max_ahy = ahy;
	snap.max_ahz = ahz;
	snap.per_delta = d;
	snap.ticks_since_snap = 0;
	snap.ghost_ey = r128_to_real_m(gey);
	snap.ghost_hilbert = hcode;
	return snap;
}

void FabricZone::_update_snap(GhostSnap &snap, const FabricEntity &e) {
	uint32_t t = snap.ticks_since_snap + 1;
	real_t ahx = MAX(ACCEL_FLOOR_M, Math::abs(e.ax));
	real_t ahy = MAX(ACCEL_FLOOR_M, Math::abs(e.ay));
	real_t ahz = MAX(ACCEL_FLOOR_M, Math::abs(e.az));
	real_t new_max_ahx = MAX(snap.max_ahx, ahx);
	real_t new_max_ahy = MAX(snap.max_ahy, ahy);
	real_t new_max_ahz = MAX(snap.max_ahz, ahz);

	real_t old_scalar_max = MAX(MAX(snap.max_ahx, snap.max_ahy), snap.max_ahz);
	real_t new_scalar_max = MAX(MAX(new_max_ahx, new_max_ahy), new_max_ahz);
	uint32_t new_per_delta;
	if (new_scalar_max > old_scalar_max) {
		new_per_delta = _snap_delta(MAX(MAX(snap.vx, snap.vy), snap.vz), new_scalar_max);
	} else {
		new_per_delta = snap.per_delta;
	}

	if (t >= new_per_delta) {
		real_t old_ahx = new_max_ahx;
		real_t old_ahy = new_max_ahy;
		real_t old_ahz = new_max_ahz;
		snap = _make_ghost_snap(e);
		snap.max_ahx = MAX(snap.max_ahx, old_ahx);
		snap.max_ahy = MAX(snap.max_ahy, old_ahy);
		snap.max_ahz = MAX(snap.max_ahz, old_ahz);
	} else {
		snap.max_ahx = new_max_ahx;
		snap.max_ahy = new_max_ahy;
		snap.max_ahz = new_max_ahz;
		snap.per_delta = new_per_delta;
		snap.ticks_since_snap = t;
	}
}

Aabb FabricZone::_ghost_aabb_from_snap(const GhostSnap &snap) {
	R128 t = r128_from_u32(snap.ticks_since_snap);
	R128 ex = ghost_bound(r128_from_real_um(snap.vx), r128_from_real_um(snap.max_ahx), t);
	R128 ey = ghost_bound(r128_from_real_um(snap.vy), r128_from_real_um(snap.max_ahy), t);
	R128 ez = ghost_bound(r128_from_real_um(snap.vz), r128_from_real_um(snap.max_ahz), t);
	R128 cx = r128_from_real_um(snap.cx);
	R128 cy = r128_from_real_um(snap.cy);
	R128 cz = r128_from_real_um(snap.cz);
	Aabb aabb;
	aabb.min_x = r128_sub(cx, ex);
	aabb.max_x = r128_add(cx, ex);
	aabb.min_y = r128_sub(cy, ey);
	aabb.max_y = r128_add(cy, ey);
	aabb.min_z = r128_sub(cz, ez);
	aabb.max_z = r128_add(cz, ez);
	return aabb;
}

// ══════════════════════════════════════════════════════════════════════════════
// Zone partitioning — Hilbert-ordered (Skilling 2004)
// ══════════════════════════════════════════════════════════════════════════════

Aabb FabricZone::_scene_aabb() {
	return aabb_from_floats(
			(float)-SIM_BOUND, (float)SIM_BOUND,
			(float)-SIM_BOUND, (float)SIM_BOUND,
			(float)-SIM_BOUND, (float)SIM_BOUND);
}

::AABB FabricZone::hilbert_cell_of_aabb(int p_code, int p_prefix_depth) {
	Aabb scene = _scene_aabb();
	Aabb cell = hilbert_cell_of((uint32_t)p_code, (uint32_t)p_prefix_depth, &scene);
	auto um_to_m = [](R128 v) -> real_t { return (real_t)((int64_t)v.hi) * (real_t)0.000001; };
	Vector3 min_pt(um_to_m(cell.min_x), um_to_m(cell.min_y), um_to_m(cell.min_z));
	Vector3 max_pt(um_to_m(cell.max_x), um_to_m(cell.max_y), um_to_m(cell.max_z));
	return ::AABB(min_pt, max_pt - min_pt);
}

int FabricZone::hilbert_of_point(float p_x, float p_y, float p_z) {
	// Delegate to _entity_hilbert so the ClassDB binding and the server
	// always agree on the Hilbert code for a given position. The old
	// float-normalization path diverged from the R128 path by a few codes.
	FabricEntity e;
	e.cx = (real_t)p_x;
	e.cy = (real_t)p_y;
	e.cz = (real_t)p_z;
	return (int)_entity_hilbert(e);
}

uint32_t FabricZone::_entity_hilbert(const FabricEntity &e) {
	Aabb b = aabb_from_floats(
			(float)e.cx, (float)e.cx,
			(float)e.cy, (float)e.cy,
			(float)e.cz, (float)e.cz);
	Aabb scene = _scene_aabb();
	return hilbert_of_aabb(&b, &scene);
}

int FabricZone::_zone_for_hilbert(uint32_t hcode, int count) {
	// Matches Fabric.lean assignToZone / zonePrefixDepth:
	//   depth = ceil(log2(count)) = bit-length of (count - 1)
	//   zone  = min(count - 1, hcode >> (30 - depth))
	if (count <= 1) {
		return 0;
	}
	int depth = 0;
	uint32_t x = (uint32_t)(count - 1);
	while (x > 0) {
		depth++;
		x >>= 1;
	}
	int z = (int)(hcode >> (30 - depth));
	return CLAMP(z, 0, count - 1);
}

void FabricZone::_hilbert_aoi_band(uint32_t &out_lo, uint32_t &out_hi, int my_id, int count) {
	// Hilbert AOI band for zone `my_id` in an `count`-zone fabric.
	// depth = ceil(log2(count)); cell_w = 2^(30 - depth).
	// Own range is [my_id*cell_w, (my_id+1)*cell_w); the band pads by
	// AOI_CELLS*cell_w on each side, clamped to [0, 2^30).
	uint32_t total = 1u << 30;
	if (count <= 1) {
		out_lo = 0u;
		out_hi = total;
		return;
	}
	int depth = 0;
	uint32_t x = (uint32_t)(count - 1);
	while (x > 0) {
		depth++;
		x >>= 1;
	}
	uint32_t cell_w = 1u << (30 - depth);
	uint32_t pad = (uint32_t)AOI_CELLS * cell_w;
	uint32_t zone_lo = (uint32_t)my_id * cell_w;
	uint32_t zone_hi = zone_lo + cell_w;
	out_lo = zone_lo > pad ? zone_lo - pad : 0u;
	out_hi = (total - zone_hi) > pad ? zone_hi + pad : total;
}

// ══════════════════════════════════════════════════════════════════════════════
// Migration intent serialization — little-endian byte packing
// ══════════════════════════════════════════════════════════════════════════════

Vector<uint8_t> FabricZone::_pack_intent(int eid, int to, uint32_t arrival, const FabricEntity &e) {
	Vector<uint8_t> b;
	b.resize(INTENT_SIZE); // 8(eid) + 4(to) + 4(arrival) + 9×8(double xyz/vxyz/axyz) = 88 bytes
	uint8_t *w = b.ptrw();
	uint64_t eid64 = (uint64_t)eid;
	uint32_t to32 = (uint32_t)to;
	memcpy(w + 0, &eid64, 8);
	memcpy(w + 8, &to32, 4);
	memcpy(w + 12, &arrival, 4);
	double vals[9] = { (double)e.cx, (double)e.cy, (double)e.cz, (double)e.vx, (double)e.vy, (double)e.vz, (double)e.ax, (double)e.ay, (double)e.az };
	for (int i = 0; i < 9; i++) {
		memcpy(w + 16 + i * 8, &vals[i], 8);
	}
	return b;
}

bool FabricZone::_unpack_intent(const uint8_t *p_data, int p_size, int &r_eid, int &r_to, uint32_t &r_arrival, FabricEntity &r_entity) {
	if (p_size < INTENT_SIZE) {
		return false;
	}
	uint64_t eid64;
	uint32_t to32;
	memcpy(&eid64, p_data + 0, 8);
	memcpy(&to32, p_data + 8, 4);
	memcpy(&r_arrival, p_data + 12, 4);
	r_eid = (int)eid64;
	r_to = (int)to32;

	double vals[9];
	for (int i = 0; i < 9; i++) {
		memcpy(&vals[i], p_data + 16 + i * 8, 8);
	}
	r_entity.cx = (real_t)vals[0];
	r_entity.cy = (real_t)vals[1];
	r_entity.cz = (real_t)vals[2];
	r_entity.vx = (real_t)vals[3];
	r_entity.vy = (real_t)vals[4];
	r_entity.vz = (real_t)vals[5];
	r_entity.ax = (real_t)vals[6];
	r_entity.ay = (real_t)vals[7];
	r_entity.az = (real_t)vals[8];
	r_entity.global_id = (int)eid64;
	return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// GDScript bindings
// ══════════════════════════════════════════════════════════════════════════════

void FabricZone::_bind_methods() {
	ClassDB::bind_static_method("FabricZone", D_METHOD("hilbert_cell_of_aabb", "code", "prefix_depth"), &FabricZone::hilbert_cell_of_aabb);
	ClassDB::bind_static_method("FabricZone", D_METHOD("hilbert_of_point", "x", "y", "z"), &FabricZone::hilbert_of_point);
}

// ══════════════════════════════════════════════════════════════════════════════
// Destructor
// ══════════════════════════════════════════════════════════════════════════════

FabricZone::~FabricZone() {
	if (fabric_peer.is_valid()) {
		fabric_peer->close();
		fabric_peer.unref();
	}
	if (slots) {
		for (int i = 0; i < _zone_capacity; i++) {
			slots[i].~EntitySlot();
		}
		memfree(slots);
		slots = nullptr;
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// Graceful shutdown: drain consolidation
// ══════════════════════════════════════════════════════════════════════════════

void FabricZone::_begin_drain() {
	if (_draining) {
		return;
	}
	_draining = true;
	_drain_start_tick = tick;
	print_line(vformat("[zone %d] drain started at tick %d — migrating entities toward zone 0", zone_id, tick));

	// Broadcast DRAIN_MAGIC to neighbors.
	if (fabric_peer.is_valid()) {
		uint8_t buf[8];
		uint32_t zid = (uint32_t)zone_id;
		uint32_t mag = DRAIN_MAGIC;
		memcpy(buf + 0, &zid, 4);
		memcpy(buf + 4, &mag, 4);
		peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_MIGRATION, buf, 8);
	}
}

void FabricZone::_drain_collect_entity(const FabricEntity &e) {
	_drain_buffer.push_back(e);
}

void FabricZone::_drain_save_snapshot() {
	Ref<FabricSnapshot> snap;
	snap.instantiate();

	// Collect own remaining active entities into drain buffer.
	for (int i = 0; i < _zone_capacity; i++) {
		if (slots[i].active) {
			_drain_buffer.push_back(slots[i].entity);
		}
	}

	int n = (int)_drain_buffer.size();
	print_line(vformat("[zone %d] saving snapshot: %d entities to %s", zone_id, n, SNAPSHOT_PATH));

	PackedInt32Array ids;
	PackedFloat64Array pos, vel, acc;
	PackedInt32Array pay;
	ids.resize(n);
	pos.resize(n * 3);
	vel.resize(n * 3);
	acc.resize(n * 3);
	pay.resize(n * 14);

	for (int i = 0; i < n; i++) {
		const FabricEntity &e = _drain_buffer[i];
		ids.set(i, e.global_id);
		pos.set(i * 3 + 0, (double)e.cx);
		pos.set(i * 3 + 1, (double)e.cy);
		pos.set(i * 3 + 2, (double)e.cz);
		vel.set(i * 3 + 0, (double)e.vx);
		vel.set(i * 3 + 1, (double)e.vy);
		vel.set(i * 3 + 2, (double)e.vz);
		acc.set(i * 3 + 0, (double)e.ax);
		acc.set(i * 3 + 1, (double)e.ay);
		acc.set(i * 3 + 2, (double)e.az);
		for (int p = 0; p < 14; p++) {
			pay.set(i * 14 + p, (int32_t)e.payload[p]);
		}
	}

	snap->set_global_ids(ids);
	snap->set_positions(pos);
	snap->set_velocities(vel);
	snap->set_accelerations(acc);
	snap->set_payloads(pay);

	Ref<Resource> res = snap;
	Error err = ResourceSaver::save(res, SNAPSHOT_PATH, ResourceSaver::FLAG_COMPRESS);
	if (err != OK) {
		print_line(vformat("[zone %d] ERROR: failed to save snapshot (err=%d)", zone_id, (int)err));
	} else {
		print_line(vformat("[zone %d] snapshot saved: %d entities", zone_id, n));
	}
	_drain_buffer.clear();
}

int FabricZone::_load_snapshot() {
	if (!ResourceLoader::exists(SNAPSHOT_PATH)) {
		return 0;
	}
	Ref<FabricSnapshot> snap = ResourceLoader::load(SNAPSHOT_PATH);
	if (snap.is_null()) {
		print_line(vformat("[zone %d] WARNING: snapshot exists but failed to load", zone_id));
		return 0;
	}
	if (snap->get_version() != FabricSnapshot::CURRENT_VERSION) {
		print_line(vformat("[zone %d] WARNING: snapshot version %d != expected %d, skipping",
				zone_id, snap->get_version(), FabricSnapshot::CURRENT_VERSION));
		return 0;
	}
	int n = snap->get_entity_count();
	PackedInt32Array ids = snap->get_global_ids();
	PackedFloat64Array pos = snap->get_positions();
	PackedFloat64Array vel = snap->get_velocities();
	PackedFloat64Array acc = snap->get_accelerations();
	PackedInt32Array pay = snap->get_payloads();

	int loaded = 0;
	for (int i = 0; i < n && entity_count < _zone_capacity; i++) {
		FabricEntity e;
		e.global_id = ids[i];
		e.cx = (real_t)pos[i * 3 + 0];
		e.cy = (real_t)pos[i * 3 + 1];
		e.cz = (real_t)pos[i * 3 + 2];
		e.vx = (real_t)vel[i * 3 + 0];
		e.vy = (real_t)vel[i * 3 + 1];
		e.vz = (real_t)vel[i * 3 + 2];
		e.ax = (real_t)acc[i * 3 + 0];
		e.ay = (real_t)acc[i * 3 + 1];
		e.az = (real_t)acc[i * 3 + 2];
		for (int p = 0; p < 14; p++) {
			e.payload[p] = (uint32_t)pay[i * 14 + p];
		}
		// Only load entities that belong to this zone by Hilbert assignment.
		int target = _zone_for_hilbert(_entity_hilbert(e), zone_count);
		if (target != zone_id) {
			continue; // Will be loaded by the correct zone.
		}
		int idx = _alloc_entity_slot();
		if (idx < 0) {
			break;
		}
		slots[idx].entity = e;
		slots[idx].snap = _make_ghost_snap(e);
		loaded++;
	}
	print_line(vformat("[zone %d] loaded %d/%d entities from snapshot", zone_id, loaded, n));
	return loaded;
}

void FabricZone::finalize() {
	if (!_draining && !is_player) {
		// Synchronous drain: dump own entities directly (no time for migration).
		if (zone_id == 0) {
			_drain_save_snapshot();
		} else {
			// Non-zero zone in abrupt shutdown: begin drain so at least own entities are logged.
			print_line(vformat("[zone %d] finalize: abrupt shutdown, %d entities lost (not zone 0)", zone_id, entity_count));
		}
	} else if (_draining && zone_id == 0) {
		// Already draining — save whatever we've collected plus our own.
		_drain_save_snapshot();
	}
	SceneTree::finalize();
}

// ── Protected slot helpers (for MMOG-layer subclasses) ──────────────────────

int FabricZone::_alloc_entity_slot() {
	ERR_FAIL_NULL_V(slots, -1);
	if (entity_count >= _zone_capacity) {
		return -1;
	}
	for (int fi = 0; fi < _zone_capacity; fi++) {
		int idx = (free_hint + fi) % _zone_capacity;
		if (!slots[idx].active) {
			slots[idx] = EntitySlot();
			slots[idx].active = true;
			free_hint = (idx + 1) % _zone_capacity;
			entity_count++;
			return idx;
		}
	}
	return -1;
}

void FabricZone::_free_entity_slot(int p_idx) {
	ERR_FAIL_NULL(slots);
	ERR_FAIL_INDEX(p_idx, _zone_capacity);
	if (slots[p_idx].active) {
		slots[p_idx].active = false;
		entity_count--;
	}
}

FabricZone::FabricEntity &FabricZone::_slot_entity_ref(int p_idx) {
	DEV_ASSERT(slots && p_idx >= 0 && p_idx < _zone_capacity);
	return slots[p_idx].entity;
}

void FabricZone::_send_to_peer_raw(int p_peer_id, int p_channel, const uint8_t *p_data, int p_size) {
	if (!fabric_peer.is_valid() ||
			fabric_peer->get_connection_status() != MultiplayerPeer::CONNECTION_CONNECTED) {
		return;
	}
	fabric_peer->set_target_peer(p_peer_id);
	fabric_peer->set_transfer_channel(p_channel);
	fabric_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
	fabric_peer->put_packet(p_data, p_size);
	fabric_peer->set_target_peer(0); // restore: broadcast to all
}

// ══════════════════════════════════════════════════════════════════════════════
// initialize() — parse CLI, create entities, connect ENet
// ══════════════════════════════════════════════════════════════════════════════

void FabricZone::initialize() {
	SceneTree::initialize();

	// ── Parse optional overrides from command line ──────────────────────
	// get_cmdline_args()      = args before "--" (engine args, may include zone flags if no "--" separator)
	// get_cmdline_user_args() = args after  "--" (user args, the conventional way to pass zone flags)
	// Merge both so zone flags work whether or not the "--" separator is used.
	Vector<String> args;
	for (const String &a : OS::get_singleton()->get_cmdline_args()) {
		args.push_back(a);
	}
	for (const String &a : OS::get_singleton()->get_cmdline_user_args()) {
		args.push_back(a);
	}

	int explicit_id = -1;
	int explicit_count = -1;
	bool player_mode = false;
	int explicit_player_id = -1;

	for (int i = 0; i < args.size(); i++) {
		if (args[i] == "--zone-id" && i + 1 < args.size()) {
			explicit_id = args[i + 1].to_int();
		} else if (args[i] == "--zone-count" && i + 1 < args.size()) {
			explicit_count = args[i + 1].to_int();
		} else if (args[i] == "--player") {
			player_mode = true;
		} else if (args[i] == "--player-id" && i + 1 < args.size()) {
			player_mode = true;
			explicit_player_id = args[i + 1].to_int();
		} else if (args[i] == "--scenario" && i + 1 < args.size()) {
			String s = args[i + 1];
			if (s == "jellyfish_bloom" || s == "jellyfish_bloom_concert") {
				scenario = SCENARIO_JELLYFISH_BLOOM;
			} else if (s == "jellyfish_zone_crossing") {
				scenario = SCENARIO_JELLYFISH_ZONE_CROSSING;
			} else if (s == "whale_with_sharks") {
				scenario = SCENARIO_WHALE_WITH_SHARKS;
			} else if (s == "current_funnel") {
				scenario = SCENARIO_CURRENT_FUNNEL;
			} else if (s == "mixed") {
				scenario = SCENARIO_MIXED;
			} else {
				scenario = SCENARIO_DEFAULT;
			}
		} else if (args[i] == "--zone-capacity" && i + 1 < args.size()) {
			int cap = args[i + 1].to_int();
			_zone_capacity = CLAMP(cap, 1, ZONE_CAPACITY);
			if (cap < 896) {
				ERR_PRINT(vformat("--zone-capacity %d is below the SLA minimum of 896 (16 players × 56 entities). Concert scenario will not fit.", cap));
			}
		} else if (args[i] == "--xing-count" && i + 1 < args.size()) {
			s_xing_count = CLAMP(args[i + 1].to_int(), 0, XING_TOTAL);
		} else if (args[i] == "--xing-mode" && i + 1 < args.size()) {
			String m = args[i + 1];
			if (m == "fixed") {
				s_xing_mode = 1;
			} else if (m == "osc") {
				s_xing_mode = 2;
			} else {
				s_xing_mode = 0;
			}
		} else if (args[i] == "--drain-at-tick" && i + 1 < args.size()) {
			_drain_at_tick = (uint32_t)args[i + 1].to_int();
		}
	}
	print_line(vformat("[zone init] xing_count=%d xing_mode=%d", s_xing_count, s_xing_mode));

	is_player = player_mode;

	// ── Player mode ─────────────────────────────────────────────────────
	if (is_player) {
		int pid = (explicit_player_id >= 0) ? explicit_player_id : 0;
		player_id = pid;
		player_cx = (real_t)((pid * 137) % 30000 - 15000) * 0.001;
		player_cy = (real_t)((pid * 53) % 30000 - 15000) * 0.001;
		// Connect to target zone.
		uint16_t base_port = 17500;
		for (int i = 0; i < args.size(); i++) {
			if (args[i] == "--base-port" && i + 1 < args.size()) {
				base_port = (uint16_t)args[i + 1].to_int();
			}
		}
		int target_zone = 0;
		for (int i = 0; i < args.size(); i++) {
			if (args[i] == "--target-zone" && i + 1 < args.size()) {
				target_zone = args[i + 1].to_int();
			}
		}
		// Player connects to exactly ONE zone — superlinear scaling model intact.
		// Each zone handles 1/N of the world; players connect to their zone only.
		if (peer_callbacks.is_valid()) {
			fabric_peer = peer_callbacks.create_client("127.0.0.1", (int)base_port + target_zone);
		}
		Engine::get_singleton()->set_physics_ticks_per_second((int)PBVH_SIM_TICK_HZ);
		wall_start_usec = OS::get_singleton()->get_ticks_usec();
		print_line(vformat("[player %d] ready at (%.3f, %.3f) -> zone %d", pid, player_cx, player_cy, target_zone));
		return;
	}

	// ── Zone ID: explicit via CLI args ──────────────────────────────────
	int my_id = (explicit_id >= 0) ? explicit_id : 0;
	int zcount = (explicit_count >= 0) ? explicit_count : DEFAULT_ZONE_COUNT;

	zone_id = my_id;
	zone_count = zcount;

	// ── Allocate fixed-size slot array ──────────────────────────────────
	// _zone_capacity is set by --zone-capacity (default = ZONE_CAPACITY).
	// Allocating exactly _zone_capacity slots ensures constant-work at the configured limit.
	slots = (EntitySlot *)memalloc(sizeof(EntitySlot) * _zone_capacity);
	for (int i = 0; i < _zone_capacity; i++) {
		new (&slots[i]) EntitySlot();
	}
	entity_count = 0;

	// ── Create entities, keep our partition; accumulate per-zone centroids ──
	// All zone centroids are computed here so every zone process knows the
	// waypoint targets for jellyfish_zone_crossing without any inter-zone comms.
	double centroid_sum[MAX_ZONES][3] = {};
	int centroid_cnt[MAX_ZONES] = {};
	for (int i = 0; i < N_TOTAL; i++) {
		FabricEntity candidate;
		candidate.cx = (real_t)((i * 137) % 30000 - 15000) * 0.001;
		candidate.cy = (real_t)((i * 53) % 30000 - 15000) * 0.001;
		candidate.cz = (real_t)((i * 29) % 30000 - 15000) * 0.001;
		int z = _zone_for_hilbert(_entity_hilbert(candidate), zone_count);
		if (z < MAX_ZONES) {
			centroid_sum[z][0] += candidate.cx;
			centroid_sum[z][1] += candidate.cy;
			centroid_sum[z][2] += candidate.cz;
			centroid_cnt[z]++;
		}
		if (z == zone_id && entity_count < _zone_capacity - MIGRATION_HEADROOM) {
			FabricEntity e = candidate;
			e.vx = 0;
			e.vy = 0;
			e.vz = 0;
			e.ax = 0;
			e.ay = 0;
			e.az = 0;
			e.global_id = i;
			slots[entity_count].entity = e;
			slots[entity_count].snap = _make_ghost_snap(e);
			slots[entity_count].hysteresis = 0;
			slots[entity_count].active = true;
			entity_count++;
		}
	}
	for (int z = 0; z < zone_count && z < MAX_ZONES; z++) {
		if (centroid_cnt[z] > 0) {
			_zone_centroid[z][0] = (real_t)(centroid_sum[z][0] / centroid_cnt[z]);
			_zone_centroid[z][1] = (real_t)(centroid_sum[z][1] / centroid_cnt[z]);
			_zone_centroid[z][2] = (real_t)(centroid_sum[z][2] / centroid_cnt[z]);
		}
	}
	print_line(vformat("[zone %d] centroids: z0=(%.2f,%.2f,%.2f) z%d=(%.2f,%.2f,%.2f)",
			zone_id,
			_zone_centroid[0][0], _zone_centroid[0][1], _zone_centroid[0][2],
			zone_count - 1,
			_zone_centroid[zone_count - 1][0], _zone_centroid[zone_count - 1][1],
			_zone_centroid[zone_count - 1][2]));

	// ── Load snapshot from previous graceful shutdown (if any) ──────────
	// Snapshot entities are filtered by Hilbert zone — each zone loads only
	// the entities that belong to it, so all zones can read the same file.
	int snapshot_loaded = _load_snapshot();
	if (snapshot_loaded > 0) {
		print_line(vformat("[zone %d] restored %d entities from snapshot", my_id, snapshot_loaded));
	}
	int n = entity_count;

	// ── ENet: always create server (for player connections + zone neighbors) ─
	{
		uint16_t base_port = 17500; // Default; override with --base-port
		for (int i = 0; i < args.size(); i++) {
			if (args[i] == "--base-port" && i + 1 < args.size()) {
				base_port = (uint16_t)args[i + 1].to_int();
			}
		}
		cluster_base_port = base_port;
		if (peer_callbacks.is_valid()) {
			fabric_peer = peer_callbacks.create_server((int)base_port + my_id, 32);
		}
		if (fabric_peer.is_valid() && zcount > 1) {
			// Hilbert-AOI-derived neighbor topology. For each other zone nb,
			// compute nb's Hilbert range and connect iff it intersects this
			// zone's AOI band. Reduces to full-mesh when AOI_CELLS covers the
			// whole Hilbert space (the 3-zone smoke case) and to local-only
			// linear connectivity at AOI_CELLS=1 in a large fabric, without
			// any zone_count special case.
			uint32_t band_lo = 0, band_hi = 0;
			_hilbert_aoi_band(band_lo, band_hi, my_id, zcount);
			int depth = 0;
			uint32_t x = (uint32_t)(zcount - 1);
			while (x > 0) {
				depth++;
				x >>= 1;
			}
			uint32_t cell_w = 1u << (30 - depth);
			for (int nb = 0; nb < zcount; nb++) {
				if (nb == my_id) {
					continue;
				}
				uint32_t nb_lo = (uint32_t)nb * cell_w;
				uint32_t nb_hi = nb_lo + cell_w;
				if (nb_hi > band_lo && nb_lo < band_hi) {
					peer_callbacks.connect_to_zone(fabric_peer.ptr(), nb,
							(int)base_port + nb);
				}
			}
		}
	}

	Engine::get_singleton()->set_physics_ticks_per_second((int)PBVH_SIM_TICK_HZ);
	wall_start_usec = OS::get_singleton()->get_ticks_usec();
	print_line(vformat("[zone %d] ready: %d entities", my_id, (int)n));

	{
		uint32_t bl = 0, bh = 0;
		_hilbert_aoi_band(bl, bh, my_id, zone_count);
		::AABB lo_cell = hilbert_cell_of_aabb((int)bl, 10);
		::AABB hi_cell = hilbert_cell_of_aabb((int)(bh > 0 ? bh - 1 : 0), 10);
		print_line(vformat("[zone %d] aoi_band: codes=[%d,%d) lo_cell=(%.1f,%.1f,%.1f) hi_cell=(%.1f,%.1f,%.1f)",
				zone_id, bl, bh,
				lo_cell.position.x, lo_cell.position.y, lo_cell.position.z,
				hi_cell.position.x, hi_cell.position.y, hi_cell.position.z));
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// physics_process() — main tick logic
// ══════════════════════════════════════════════════════════════════════════════

bool FabricZone::physics_process(double p_time) {
	if (done) {
		return false;
	}

	// ── Player tick ─────────────────────────────────────────────────────
	if (is_player) {
		int pid = player_id;
		// Client-side prediction.
		real_t rx = -(player_cx / 0.8f);
		real_t ry = -(player_cy / 0.8f);
		real_t kx = (real_t)(((int)tick * 7 + pid * 31) % 25 - 12) * 0.001f;
		real_t ky = (real_t)(((int)tick * 11 + pid * 37) % 25 - 12) * 0.001f;
		player_vx = CLAMP(player_vx + kx + rx, -0.2f, 0.2f);
		player_vy = CLAMP(player_vy + ky + ry, -0.2f, 0.2f);
		player_cx = CLAMP(player_cx + player_vx, -SIM_BOUND, SIM_BOUND);
		player_cy = CLAMP(player_cy + player_vy, -SIM_BOUND, SIM_BOUND);

		// Publish state to zones (public game topic -- no fabric cookie).
		uint64_t t0 = OS::get_singleton()->get_ticks_usec();
		{
			// Player state: 100-byte packet matching CH_INTEREST skeleton.
			// [u32 player_id][f64 cx/cy/cz][i16×6 vx/vy/vz/ax/ay/az][u32 hlc][u32×14 payload]
			Vector<uint8_t> buf;
			buf.resize(100);
			buf.fill(0);
			uint8_t *w = buf.ptrw();
			uint32_t pid32 = (uint32_t)pid;
			memcpy(w + 0, &pid32, 4);
			double pcx_d = (double)player_cx, pcy_d = (double)player_cy, pcz_d = 0.0;
			memcpy(w + 4, &pcx_d, 8);
			memcpy(w + 12, &pcy_d, 8);
			memcpy(w + 20, &pcz_d, 8);
			int16_t ivx = (int16_t)CLAMP((int)(player_vx * V_SCALE), -32767, 32767);
			int16_t ivy = (int16_t)CLAMP((int)(player_vy * V_SCALE), -32767, 32767);
			memcpy(w + 28, &ivx, 2);
			memcpy(w + 30, &ivy, 2);
			// vz, ax, ay, az left 0
			uint32_t hlc_val = ((tick & 0x00FFFFFFu) << 8);
			memcpy(w + 40, &hlc_val, 4);
			// payload[0..13] = 0 (no cmd)
			// Only send once connected (avoids put_packet errors during ENet handshake).
			if (fabric_peer.is_valid() &&
					fabric_peer->get_connection_status() == MultiplayerPeer::CONNECTION_CONNECTED) {
				peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_PLAYER, buf.ptr(), buf.size());
			}
		}
		total_compute_us += OS::get_singleton()->get_ticks_usec() - t0;

		// Poll single-zone peer and drain CH_INTEREST.
		if (fabric_peer.is_valid() && fabric_peer->get_connection_status() != MultiplayerPeer::CONNECTION_DISCONNECTED) {
			fabric_peer->poll();
			LocalVector<Vector<uint8_t>> interest_pkts =
					peer_callbacks.drain_channel_raw(fabric_peer.ptr(), CH_INTEREST);
			for (uint32_t pi = 0; pi < interest_pkts.size(); pi++) {
				const Vector<uint8_t> &pkt = interest_pkts[pi];
				int offset = 0;
				while (offset + 100 <= pkt.size()) {
					uint32_t gid;
					memcpy(&gid, pkt.ptr() + offset, 4);
					double cx_d, cy_d, cz_d;
					memcpy(&cx_d, pkt.ptr() + offset + 4, 8);
					memcpy(&cy_d, pkt.ptr() + offset + 12, 8);
					memcpy(&cz_d, pkt.ptr() + offset + 20, 8);
					real_t cx = (real_t)cx_d, cy = (real_t)cy_d, cz = (real_t)cz_d;
					entities_received++;
					offset += 100;

					// Phase-1 pass-condition: track zone-crossing entities (256–399).
					if (gid >= (uint32_t)XING_ID_LO && gid <= (uint32_t)XING_ID_HI) {
						int idx = (int)gid - XING_ID_LO;
						bool was_seen = _p1_seen[idx];
						const uint32_t snap_absence_ticks = MAX((SNAP_ABSENCE_MS * _fz_hz() + 999u) / 1000u, 1u);
						bool gap = was_seen && (tick - _p1_last_tick[idx] > snap_absence_ticks);
						if (was_seen && !gap) {
							// Continuous stream — check for teleport.
							real_t dx = cx - _p1_cx[idx];
							real_t dy = cy - _p1_cy[idx];
							real_t dz = cz - _p1_cz[idx];
							real_t jump = Math::sqrt(dx * dx + dy * dy + dz * dz);
							if (jump > SNAP_THRESHOLD_M) {
								_p1_snap_count++;
								print_line(vformat("SNAP gid=%d jump=%.2fm tick=%d",
										gid, (double)jump, tick));
							}
						} else if (!was_seen) {
							_p1_seen[idx] = true;
							_p1_seen_count++;
						}
						// gap==true: entity reappeared after absence — reset position, no snap.
						_p1_cx[idx] = cx;
						_p1_cy[idx] = cy;
						_p1_cz[idx] = cz;
						_p1_last_tick[idx] = tick;
						if (!_p1_pass_logged && _p1_seen_count >= XING_TOTAL) {
							_p1_pass_logged = true;
							print_line(vformat(
									"PHASE1 PASS: all %d zone-crossing entities received (snaps=%d) at tick %d",
									XING_TOTAL, _p1_snap_count, tick));
						}
					}
				}
			}
		}

		tick += 1;
		// Semantic log cadence: every 20 wall-seconds at the engine's current tick rate.
		const uint32_t log_interval_ticks = MAX(_fz_hz() * (uint32_t)STATS_LOG_INTERVAL_SECONDS, 1u);
		if (tick % log_interval_ticks == 0) {
			print_line(vformat("[player %d] tick %d pos=(%.1f,%.1f) received=%d xing_seen=%d",
					pid, tick, (double)player_cx, (double)player_cy,
					(int)entities_received, _p1_seen_count));
		}
		return false;
	}

	// ── Server tick ─────────────────────────────────────────────────────
	// Drain trigger: --drain-at-tick N starts graceful shutdown at tick N.
	if (_drain_at_tick > 0 && tick >= _drain_at_tick && !_draining) {
		_begin_drain();
	}

	uint64_t t0 = OS::get_singleton()->get_ticks_usec();
	// Runtime tick-rate-dependent bounds — all derived from the engine's
	// physics tick rate (single source of runtime truth).
	const uint32_t hz = _fz_hz();
	const uint32_t latency_ticks_runtime = pbvh_latency_ticks(hz);
	const uint32_t hysteresis_runtime = pbvh_hysteresis_threshold(hz);
	const uint32_t retry_cap_ticks = RETRY_CAP_SECONDS * hz;
	const uint32_t ping_interval_ticks = PING_INTERVAL_SECONDS * hz;
	const uint32_t player_slot_timeout = PLAYER_SLOT_TIMEOUT_SECONDS * hz;
	const uint32_t log_interval_ticks = MAX(hz * (uint32_t)STATS_LOG_INTERVAL_SECONDS, 1u);

	// Per-tick diagnostic counters (minimal-repro instrumentation for the
	// xing_in saturation bug). Reset each tick; consumed in the log line below.
	int diag_wrong_zone = 0; // target != my_id this tick
	int diag_hys_reset = 0; // hysteresis was > 0 and got zeroed at the else branch
	int diag_gate_skip = 0; // !resnap && hysteresis == 0 continue
	int diag_ready_block = 0; // hysteresis >= runtime but outbound_budget == 0
	int diag_max_hys = 0; // peak hysteresis across all wrong-zone slots this tick
	const uint64_t xing_received_at_tick_start = xing_received;

	// Poll ENet every tick.
	if (fabric_peer.is_valid() && fabric_peer->get_connection_status() != MultiplayerPeer::CONNECTION_DISCONNECTED) {
		fabric_peer->poll();
		// Reconnect to neighbors with exponential backoff — handles startup ordering
		// where a neighbor zone may not have been listening at initialize() time.
		// Starts at pbvh_latency_ticks(hz), doubles each failed attempt, caps at retry_cap_ticks.
		if (zone_count > 1) {
			int neighbors[2] = { zone_id - 1, zone_id + 1 };
			bool valid[2] = { zone_id > 0, zone_id + 1 < zone_count };
			for (int ni = 0; ni < 2; ni++) {
				if (!valid[ni]) {
					continue;
				}
				int nb = neighbors[ni];
				if (peer_callbacks.is_zone_connected(fabric_peer.ptr(), nb)) {
					// Connected — reset backoff for next potential disconnect.
					_retry_next[ni] = tick;
					_retry_interval[ni] = latency_ticks_runtime;
					// Send HLC ping periodically to measure RTT → _neighbor_latency_ticks.
					if (tick >= _ping_next[ni]) {
						uint8_t ping_buf[8];
						uint32_t t32 = tick;
						memcpy(ping_buf + 0, &t32, 4);
						uint32_t mag = PING_MAGIC;
						memcpy(ping_buf + 4, &mag, 4);
						peer_callbacks.send_to_zone_raw(fabric_peer.ptr(), nb, CH_MIGRATION,
								ping_buf, 8);
						_ping_send_tick[ni] = tick;
						_ping_next[ni] = tick + ping_interval_ticks;
					}
				} else if (tick >= _retry_next[ni]) {
					peer_callbacks.connect_to_zone(fabric_peer.ptr(), nb,
							(int)cluster_base_port + nb);
					_retry_interval[ni] = MIN(_retry_interval[ni] * 2, retry_cap_ticks);
					_retry_next[ni] = tick + _retry_interval[ni];
				}
			}
		}
		LocalVector<Vector<uint8_t>> migration_data = peer_callbacks.drain_channel_raw(fabric_peer.ptr(), CH_MIGRATION);
		for (uint32_t i = 0; i < migration_data.size(); i++) {
			const Vector<uint8_t> &mpkt = migration_data[i];
			if (mpkt.size() == 8) {
				// 8-byte control packets: PING, PONG, ACK (< INTENT_SIZE, not migration intents).
				uint32_t field0, magic;
				memcpy(&field0, mpkt.ptr() + 0, 4);
				memcpy(&magic, mpkt.ptr() + 4, 4);
				if (magic == PING_MAGIC) {
					// Reflect back as pong to the sending zone.
					uint8_t pong_buf[8];
					memcpy(pong_buf + 0, &field0, 4);
					uint32_t pmag = PONG_MAGIC;
					memcpy(pong_buf + 4, &pmag, 4);
					// We don't know which neighbor sent this (any connected neighbor may have).
					// Broadcast pong to all neighbors â they discard by magic check.
					peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_MIGRATION, pong_buf, 8);
				} else if (magic == PONG_MAGIC) {
					// Compute RTT and update _neighbor_latency_ticks for the matching neighbor.
					uint32_t rtt_ticks = tick - field0;
					// perNeighborLatencyTicks: max(pbvh_latency_ticks(hz), ceil(rtt_ms/2 * hz/1000))
					// rtt_ticks is full round trip; one-way = rtt_ticks/2 ticks.
					uint32_t one_way = (rtt_ticks + 1) / 2;
					uint32_t lt = one_way < latency_ticks_runtime ? latency_ticks_runtime : one_way;
					for (int ni = 0; ni < 2; ni++) {
						if (_ping_send_tick[ni] == field0) {
							_neighbor_latency_ticks[ni] = lt;
							_rtt_measured[ni] = true;
							print_line(vformat("[FabricZone] zone=%d neighbor_idx=%d rtt_ticks=%d latency_ticks=%d",
									zone_id, ni, rtt_ticks, lt));
						}
					}
				} else if (magic == ACK_MAGIC) {
					// Zone B confirmed receipt of migration (INCOMINGâOWNED on zone B).
					// field0 = eid. Immediately resolve the matching STAGING slot on zone A.
					uint32_t acked_eid = field0;
					for (int si = 0; si < _zone_capacity; si++) {
						if (slots[si].active && slots[si].is_staging &&
								(uint32_t)slots[si].entity.global_id == acked_eid) {
							print_line(vformat("[XING_DONE]  zone=%d tick=%d eid=%d (ACK->zone=%d) latency=%d",
									zone_id, tick, slots[si].entity.global_id,
									slots[si].migration_target_zone,
									(int)(tick - slots[si].staging_send_tick)));
							xing_done++;
							slots[si].active = false;
							slots[si].is_staging = false;
							slots[si].migration_target_zone = -1;
							entity_count--;
							free_hint = si;
							break;
						}
					}
				} else if (magic == DRAIN_MAGIC) {
					// Neighbor zone is shutting down — enter drain mode ourselves.
					uint32_t draining_zone = field0;
					print_line(vformat("[zone %d] received DRAIN from zone %d", zone_id, draining_zone));
					_begin_drain();
				} else if (magic == DRAIN_DONE_MAGIC) {
					// Neighbor zone has finished draining (zero entities).
					uint32_t done_zone = field0;
					int ni = (done_zone == (uint32_t)(zone_id - 1)) ? 0 : 1;
					_neighbor_drain_done[ni] = true;
					print_line(vformat("[zone %d] neighbor zone %d drain done", zone_id, done_zone));
				} else {
					intent_inbox.push_back(mpkt);
				}
				continue;
			}
			intent_inbox.push_back(mpkt);
		}

		// ── CH_INTEREST relay — forward neighbor rows inside the Hilbert AOI band ──
		// Any CH_INTEREST packet that landed in this zone's inbox came from a
		// neighbor zone's server broadcast. Filter each 100-byte row by its
		// Hilbert position: if the entity's Hilbert code is inside this zone's
		// AOI band, copy the row into a local-only relay buffer and dispatch
		// it to the attached game clients via local_broadcast_raw. Packets
		// must NOT go through broadcast_raw, which would re-fan back to
		// neighbors and loop — late-duplication violation. Players and NPCs
		// share one codepath; only (cx, cy, cz) is read.
		{
			LocalVector<Vector<uint8_t>> interest_pkts =
					peer_callbacks.drain_channel_raw(fabric_peer.ptr(), CH_INTEREST);
			uint32_t band_lo = 0, band_hi = 0;
			_hilbert_aoi_band(band_lo, band_hi, zone_id, zone_count);
			Aabb scene = _scene_aabb();
			Vector<uint8_t> relay_buf;
			relay_buf.resize(1200);
			int relay_pos = 0;
			for (uint32_t pi = 0; pi < interest_pkts.size(); pi++) {
				const Vector<uint8_t> &pkt = interest_pkts[pi];
				int offset = 0;
				while (offset + 100 <= pkt.size()) {
					double ecx_d, ecy_d, ecz_d;
					memcpy(&ecx_d, pkt.ptr() + offset + 4, 8);
					memcpy(&ecy_d, pkt.ptr() + offset + 12, 8);
					memcpy(&ecz_d, pkt.ptr() + offset + 20, 8);
					Aabb b = aabb_from_floats(
							(float)ecx_d, (float)ecx_d,
							(float)ecy_d, (float)ecy_d,
							(float)ecz_d, (float)ecz_d);
					uint32_t em = hilbert_of_aabb(&b, &scene);
					if (em >= band_lo && em < band_hi) {
						if (relay_pos + 100 > relay_buf.size()) {
							if (fabric_peer.is_valid()) {
								peer_callbacks.local_broadcast_raw(fabric_peer.ptr(),
										CH_INTEREST, relay_buf.ptr(), relay_pos);
							}
							relay_pos = 0;
						}
						memcpy(relay_buf.ptrw() + relay_pos, pkt.ptr() + offset, 100);
						relay_pos += 100;
					}
					offset += 100;
				}
			}
			if (relay_pos > 0 && fabric_peer.is_valid()) {
				peer_callbacks.local_broadcast_raw(fabric_peer.ptr(),
						CH_INTEREST, relay_buf.ptr(), relay_pos);
			}
		}

		// Drain player commands (CH_PLAYER, 100-byte packets — same skeleton as CH_INTEREST).
		// Format: [u32 player_id][f64 cx/cy/cz][i16×6 vx/vy/vz/ax/ay/az][u32 hlc][u32×14 payload]
		// cmd lives in low byte of payload[0] (offset 44).
		// cmd=1: current_funnel — inject C7 velocity spike into nearby jellyfish_zone_crossing (IDs 256-399).
		// cmd=2: nudge — payload[1] = target_entity_id, add Y velocity impulse.
		// cmd=3: spawn_stroke_knot — payload[1] = stroke_id, payload[2] = RGBA8888 color.
		LocalVector<Vector<uint8_t>> player_pkts = peer_callbacks.drain_channel_raw(fabric_peer.ptr(), CH_PLAYER);
		for (uint32_t pi = 0; pi < player_pkts.size(); pi++) {
			const Vector<uint8_t> &p = player_pkts[pi];
			if (p.size() < 100) {
				continue;
			}
			uint8_t cmd = p[44]; // low byte of payload[0]

			// cmd=0: player position update.
			// Upsert a player entity slot so the player appears in CH_INTEREST
			// broadcasts to all zone clients (concert scenario: players see each other).
			// Lean: AbyssalSLA.entitiesPerPlayer=56, targetPlayersPerZone=16.
			// global_id = PLAYER_ENTITY_BASE + player_id.
			uint32_t player_id_field;
			memcpy(&player_id_field, p.ptr() + 0, 4);
			double plr_cx_d, plr_cy_d, plr_cz_d;
			memcpy(&plr_cx_d, p.ptr() + 4, 8);
			memcpy(&plr_cy_d, p.ptr() + 12, 8);
			memcpy(&plr_cz_d, p.ptr() + 20, 8);
			int16_t plr_vxi, plr_vyi, plr_vzi;
			memcpy(&plr_vxi, p.ptr() + 28, 2);
			memcpy(&plr_vyi, p.ptr() + 30, 2);
			memcpy(&plr_vzi, p.ptr() + 32, 2);
			{
				// Find or allocate a slot for this player.
				int slot_idx = -1;
				HashMap<uint32_t, int>::Iterator pit = _player_slot_map.find(player_id_field);
				if (pit != _player_slot_map.end()) {
					slot_idx = pit->value;
					if (!slots[slot_idx].active || !slots[slot_idx].is_player_slot) {
						// Slot was recycled — remove stale mapping.
						_player_slot_map.remove(pit);
						slot_idx = -1;
					}
				}
				if (slot_idx < 0) {
					// Allocate a new slot.
					for (int fi = 0; fi < _zone_capacity; fi++) {
						int idx = (free_hint + fi) % _zone_capacity;
						if (!slots[idx].active) {
							slot_idx = idx;
							free_hint = (idx + 1) % _zone_capacity;
							break;
						}
					}
					if (slot_idx >= 0) {
						slots[slot_idx].active = true;
						slots[slot_idx].is_player_slot = true;
						slots[slot_idx].is_staging = false;
						slots[slot_idx].hysteresis = 0;
						slots[slot_idx].entity.global_id = (int)((uint32_t)PLAYER_ENTITY_BASE + player_id_field);
						// payload[0]: class=1 (player-owned) | player_id in bits 16-31
						slots[slot_idx].entity.payload[0] = (1u << 24) | ((player_id_field & 0xFFFFu) << 8);
						entity_count++;
						_player_slot_map.insert(player_id_field, slot_idx);
					}
				}
				if (slot_idx >= 0) {
					FabricEntity &pe = slots[slot_idx].entity;
					pe.cx = (real_t)plr_cx_d;
					pe.cy = (real_t)plr_cy_d;
					pe.cz = (real_t)plr_cz_d;
					pe.vx = (real_t)plr_vxi / V_SCALE;
					pe.vy = (real_t)plr_vyi / V_SCALE;
					pe.vz = (real_t)plr_vzi / V_SCALE;
					slots[slot_idx].last_update_tick = tick;
					slots[slot_idx].snap = _make_ghost_snap(pe);
				}
			}
			if (cmd == 0) {
				continue; // position update only — no command to dispatch
			}
			double player_cx_d = 0.0, player_cy_d = 0.0, player_cz_d = 0.0;
			memcpy(&player_cx_d, p.ptr() + 4, 8);
			memcpy(&player_cy_d, p.ptr() + 12, 8);
			memcpy(&player_cz_d, p.ptr() + 20, 8);
			real_t pcx = (real_t)player_cx_d;
			real_t pcy = (real_t)player_cy_d;
			real_t pcz = (real_t)player_cz_d;

			if (cmd == 1) {
				// current_funnel: C7 velocity spike. Hits every entity within the
				// interest radius, no exceptions — humans (jellyfish), vehicles
				// (whale+remoras = train+passengers), pen annotations, everything.
				// There is no "static" entity class in this system: every entity is
				// dynamic and can be displaced by a world-scale event.
				// Downstream _step_entity clamps must have v_cap ≥ CURRENT_FUNNEL_PEAK_V
				// so the spike survives the next tick and the BVH registers its true
				// magnitude (see static_assert near SCENARIO_WHALE_WITH_SHARKS).
				for (int si = 0; si < _zone_capacity; si++) {
					if (!slots[si].active) {
						continue;
					}
					FabricEntity &e = slots[si].entity;
					real_t dx = e.cx - pcx, dy = e.cy - pcy, dz = e.cz - pcz;
					real_t dist2 = dx * dx + dy * dy + dz * dz;
					if (dist2 > INTEREST_RADIUS * INTEREST_RADIUS) {
						continue;
					}
					e.vy = CURRENT_FUNNEL_PEAK_V;
				}
			} else if (cmd == 2) {
				// nudge: add Y impulse to target entity to push it toward zone boundary.
				uint32_t target_id = 0;
				memcpy(&target_id, p.ptr() + 48, 4); // payload[1]
				for (int si = 0; si < _zone_capacity; si++) {
					if (!slots[si].active) {
						continue;
					}
					if ((uint32_t)slots[si].entity.global_id == target_id) {
						slots[si].entity.vy += V_MAX * 2.0f; // 2× normal cap nudge
						break;
					}
				}
			} else if (cmd == 3) {
				// spawn_stroke_knot: place a knot entity at player position.
				// payload[1] (offset 48): stroke_id (u32)
				// payload[2] (offset 52): stroke_color RGBA8888 (u32)
				uint32_t stroke_id = 0, stroke_color = 0;
				memcpy(&stroke_id, p.ptr() + 48, 4);
				memcpy(&stroke_color, p.ptr() + 52, 4);

				// Find a free slot.
				int free_idx = -1;
				for (int fi = 0; fi < _zone_capacity; fi++) {
					int idx = (free_hint + fi) % _zone_capacity;
					if (!slots[idx].active) {
						free_idx = idx;
						break;
					}
				}
				if (free_idx >= 0) {
					FabricEntity ent;
					ent.cx = pcx;
					ent.cy = pcy;
					ent.cz = pcz;
					ent.global_id = STROKE_ENTITY_BASE + _stroke_entity_counter++;

					// Pack entity payload[].
					ent.payload[0] = stroke_id;
					ent.payload[1] = stroke_color;
					// Anchor position as int16 fixed-point: 1 unit = SIM_BOUND/32767 m.
					const real_t FIXED_SCALE = 32767.0f / SIM_BOUND;
					int16_t ax16 = (int16_t)CLAMP(pcx * FIXED_SCALE, -32767.0f, 32767.0f);
					int16_t ay16 = (int16_t)CLAMP(pcy * FIXED_SCALE, -32767.0f, 32767.0f);
					int16_t az16 = (int16_t)CLAMP(pcz * FIXED_SCALE, -32767.0f, 32767.0f);
					memcpy((uint8_t *)&ent.payload[2], &ax16, 2);
					memcpy((uint8_t *)&ent.payload[2] + 2, &ay16, 2);
					memcpy((uint8_t *)&ent.payload[3], &az16, 2);

					slots[free_idx].entity = ent;
					slots[free_idx].snap = _make_ghost_snap(ent);
					slots[free_idx].hysteresis = 0;
					slots[free_idx].active = true;
					entity_count++;
					free_hint = (free_idx + 1) % _zone_capacity;

					// Snake chain: push new head, remove tail when over MAX_STROKE_KNOTS.
					LocalVector<int> &chain = _stroke_chains[stroke_id];
					chain.push_back(free_idx);
					if ((int)chain.size() > MAX_STROKE_KNOTS) {
						int tail_idx = chain[0];
						chain.remove_at(0);
						if (slots[tail_idx].active) {
							slots[tail_idx].active = false;
							entity_count--;
						}
					}
				}
			}
		}
	}

	// Step own entities.
	Scenario cur_scenario = scenario;
	int n = entity_count;

	// For jellyfish_zone_crossing: precompute local tracer clump centroid (O(N), not O(N²)).
	// Tracers in this zone attract each other via cohesion spring → natural clumping.
	real_t local_clump[3] = {};
	bool has_local_clump = false;
	if (cur_scenario == SCENARIO_JELLYFISH_ZONE_CROSSING) {
		double tc[3] = {};
		int tc_n = 0;
		for (int i = 0; i < _zone_capacity; i++) {
			if (!slots[i].active) {
				continue;
			}
			int gid = slots[i].entity.global_id;
			if (gid >= 256 && gid <= 399) {
				tc[0] += slots[i].entity.cx;
				tc[1] += slots[i].entity.cy;
				tc[2] += slots[i].entity.cz;
				tc_n++;
			}
		}
		if (tc_n > 0) {
			local_clump[0] = (real_t)(tc[0] / tc_n);
			local_clump[1] = (real_t)(tc[1] / tc_n);
			local_clump[2] = (real_t)(tc[2] / tc_n);
			has_local_clump = true;
		}
	}

	for (int i = 0; i < _zone_capacity; i++) {
		if (!slots[i].active) {
			continue;
		}
		// Player slots: position is authoritative from CH_PLAYER; skip simulation.
		if (slots[i].is_player_slot) {
			continue;
		}
		_step_entity(slots[i].entity, tick, cur_scenario, n, _zone_centroid, zone_count,
				has_local_clump ? local_clump : nullptr);
	}
	// Per-entity ghost snap update (independent rebuild clocks).
	for (int i = 0; i < _zone_capacity; i++) {
		if (!slots[i].active) {
			continue;
		}
		_update_snap(slots[i].snap, slots[i].entity);
	}

	// ── Detect boundary crossings -> begin_migration (Lean protocol) ────
	int my_id = zone_id;
	int p = zone_count;

	struct IntentRecord {
		int target;
		uint32_t local_idx;
		Vector<uint8_t> data;
	};
	Vector<IntentRecord> intents;
	int outbound_budget = MAX_MIGRATIONS_PER_TICK;

	for (int i = 0; i < _zone_capacity; i++) {
		if (!slots[i].active) {
			continue;
		}
		const FabricEntity &e = slots[i].entity;
		bool resnap = (slots[i].snap.ticks_since_snap == 0);
		if (!resnap && slots[i].hysteresis == 0) {
			diag_gate_skip++;
			continue;
		}
		// Player slots stay on their connected zone — never migrate.
		if (slots[i].is_player_slot) {
			continue;
		}
		int target = _zone_for_hilbert(_entity_hilbert(e), p);
		if (target != my_id) {
			diag_wrong_zone++;
			slots[i].hysteresis += 1;
			if ((int)slots[i].hysteresis > diag_max_hys) {
				diag_max_hys = (int)slots[i].hysteresis;
			}
			if (!slots[i].is_staging && slots[i].hysteresis >= hysteresis_runtime && outbound_budget == 0) {
				diag_ready_block++;
			}
			if (!slots[i].is_staging && slots[i].hysteresis >= hysteresis_runtime && outbound_budget > 0) {
				// OWNED → STAGING (proved: owned_to_staging).
				// Entity stays active for per-neighbor latency ticks so both zones hold valid ghost snaps.
				print_line(vformat("[XING_START] zone=%d tick=%d eid=%d -> zone=%d pos=(%.2f,%.2f,%.2f)",
						my_id, tick, e.global_id, target, (double)e.cx, (double)e.cy, (double)e.cz));
				xing_started++;
				slots[i].is_staging = true;
				slots[i].staging_send_tick = tick;
				slots[i].migration_target_zone = target;
				migrations += 1;
				IntentRecord rec;
				rec.target = target;
				rec.local_idx = i;
				// Use per-neighbor latency ticks (RTT-based) for arrival deadline.
				int ni_tgt = (target == my_id - 1) ? 0 : 1;
				uint32_t lt = _neighbor_latency_ticks[ni_tgt];
				rec.data = _pack_intent(e.global_id, target, tick + lt, e);
				intents.push_back(rec);
				outbound_budget--;
			}
		} else {
			if (slots[i].hysteresis > 0) {
				diag_hys_reset++;
			}
			slots[i].hysteresis = 0;
		}
	}

	// STAGING entities stay active — deactivated by the STAGING timeout below
	// after per-neighbor latency ticks (proved: staging_resolves_to_single_owner).

	// ── Drain: force-migrate all OWNED entities toward zone 0 ───────────
	// Bypasses hysteresis — every non-staging, non-player entity is pushed
	// toward zone 0 at MAX_MIGRATIONS_PER_TICK per tick. Zone 0 collects
	// into _drain_buffer (write-through) when its slots are full.
	if (_draining && zone_id > 0 && fabric_peer.is_valid()) {
		int drain_budget = MAX_MIGRATIONS_PER_TICK - (int)intents.size();
		int fwd_target = zone_id - 1; // always toward zone 0
		for (int i = 0; i < _zone_capacity && drain_budget > 0; i++) {
			if (!slots[i].active || slots[i].is_staging || slots[i].is_incoming || slots[i].is_player_slot) {
				continue;
			}
			const FabricEntity &e = slots[i].entity;
			slots[i].is_staging = true;
			slots[i].staging_send_tick = tick;
			slots[i].migration_target_zone = fwd_target;
			migrations++;
			IntentRecord rec;
			rec.target = fwd_target;
			rec.local_idx = i;
			uint32_t lt = _neighbor_latency_ticks[0]; // neighbor index 0 = zone_id-1
			rec.data = _pack_intent(e.global_id, fwd_target, tick + lt, e);
			intents.push_back(rec);
			drain_budget--;
		}
		// Check drain completion: no active non-staging entities remain.
		bool all_drained = true;
		for (int i = 0; i < _zone_capacity; i++) {
			if (slots[i].active && !slots[i].is_staging) {
				all_drained = false;
				break;
			}
		}
		if (all_drained && entity_count == 0) {
			// Broadcast DRAIN_DONE and exit.
			uint8_t done_buf[8];
			uint32_t zid = (uint32_t)zone_id;
			uint32_t mag = DRAIN_DONE_MAGIC;
			memcpy(done_buf + 0, &zid, 4);
			memcpy(done_buf + 4, &mag, 4);
			peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_MIGRATION, done_buf, 8);
			print_line(vformat("[zone %d] drain complete at tick %d — shutting down", zone_id, tick));
			done = true;
			return true; // Signal quit.
		}
		// Drain timeout: force quit after DRAIN_TIMEOUT_SECONDS.
		if (tick - _drain_start_tick >= DRAIN_TIMEOUT_SECONDS * hz) {
			print_line(vformat("[zone %d] drain timeout at tick %d — %d entities lost", zone_id, tick, entity_count));
			done = true;
			return true;
		}
	}
	// Zone 0 drain: once all neighbors report DRAIN_DONE, save snapshot and quit.
	if (_draining && zone_id == 0) {
		bool all_neighbors_done = true;
		if (zone_count > 1 && !_neighbor_drain_done[1]) {
			all_neighbors_done = false; // zone 1 not done yet
		}
		// Zone 0 has no lower neighbor (index 0 is always "done").
		_neighbor_drain_done[0] = true;
		if (all_neighbors_done) {
			_drain_save_snapshot();
			print_line(vformat("[zone 0] all neighbors drained — snapshot saved, shutting down"));
			done = true;
			return true;
		}
		if (tick - _drain_start_tick >= DRAIN_TIMEOUT_SECONDS * hz) {
			_drain_save_snapshot();
			print_line(vformat("[zone 0] drain timeout at tick %d — saving partial snapshot (%d buffered + %d local)",
					tick, (int)_drain_buffer.size(), entity_count));
			done = true;
			return true;
		}
	}

	// ── Constant-work: shed excess entities to maintain _zone_capacity ───
	// The zone always iterates exactly _zone_capacity slots per tick.
	// When entity_count exceeds capacity (from inbound migrations), the
	// excess are immediately forwarded to the nearest neighbor. The zone
	// never does more work than its fixed budget — no modes, no scaling.
	// See: https://aws.amazon.com/builders-library/reliability-and-constant-work/
	int shed_budget = MAX_MIGRATIONS_PER_TICK;
	while (entity_count > _zone_capacity && fabric_peer.is_valid() && zone_count > 1 && shed_budget > 0) {
		shed_budget--;
		// Find an active slot to shed (search from end).
		int shed_idx = -1;
		for (int si = _zone_capacity - 1; si >= 0; si--) {
			if (slots[si].active) {
				shed_idx = si;
				break;
			}
		}
		if (shed_idx < 0) {
			break;
		}
		const FabricEntity &e = slots[shed_idx].entity;
		int target = (e.cy < 0 && zone_id > 0) ? zone_id - 1
				: (zone_id + 1 < zone_count)   ? zone_id + 1
				: (zone_id > 0)				   ? zone_id - 1
											   : zone_id;
		if (target == zone_id) {
			break; // No neighbor to shed to.
		}
		Vector<uint8_t> pkt = _pack_intent(e.global_id, target, tick + latency_ticks_runtime, e);
		peer_callbacks.send_to_zone_raw(fabric_peer.ptr(), target, CH_MIGRATION, pkt.ptr(), pkt.size());
		slots[shed_idx].active = false;
		entity_count--;
		free_hint = shed_idx;
		migrations++;
	}
	if (fabric_peer.is_valid()) {
		/* flush handled by poll */;
	}

	total_compute_us += OS::get_singleton()->get_ticks_usec() - t0;
	uint64_t t1 = OS::get_singleton()->get_ticks_usec();

	// ── Publish intents -- batched per target zone ──────────────────────
	if (!intents.is_empty()) {
		// Batch by target zone.
		HashMap<int, Vector<uint8_t>> batches;
		for (int i = 0; i < intents.size(); i++) {
			int tgt = intents[i].target;
			if (!batches.has(tgt)) {
				batches.insert(tgt, Vector<uint8_t>());
			}
			Vector<uint8_t> &batch = batches[tgt];
			int old_size = batch.size();
			batch.resize(old_size + intents[i].data.size());
			memcpy(batch.ptrw() + old_size, intents[i].data.ptr(), intents[i].data.size());
		}
		if (fabric_peer.is_valid()) {
			for (const KeyValue<int, Vector<uint8_t>> &kv : batches) {
				peer_callbacks.send_to_zone_raw(fabric_peer.ptr(), kv.key, CH_MIGRATION, kv.value.ptr(), kv.value.size());
			}
			/* flush handled by poll */;
		}
	}

	// ── Drain incoming migration intents (constant-work: capped per tick) ─
	// Process at most MAX_MIGRATIONS_PER_TICK inbound entities. Remaining
	// intents stay in inbox for next tick. This keeps migration cost fixed.
	{
		// No inbound cap — MUST drain within latency_ticks_runtime (proved invariant).
		// Inbound rate bounded by: outbound cap (50) × max neighbors (2) = 100/tick.
		for (uint32_t di = 0; di < intent_inbox.size(); di++) {
			const Vector<uint8_t> &data = intent_inbox[di];
			int offset = 0;
			while (offset + INTENT_SIZE <= data.size()) {
				int eid, to_zone;
				uint32_t arrival;
				FabricEntity ent;
				if (_unpack_intent(data.ptr() + offset, INTENT_SIZE, eid, to_zone, arrival, ent)) {
					if (to_zone != zone_id) {
						// Intent addressed to a different zone — misdirected packet, skip.
						offset += INTENT_SIZE;
						continue;
					}
					int free_idx = -1;
					for (int fi = 0; fi < _zone_capacity; fi++) {
						int idx = (free_hint + fi) % _zone_capacity;
						if (!slots[idx].active) {
							free_idx = idx;
							break;
						}
					}
					if (free_idx >= 0) {
						// OWNED → INCOMING (Lean: MigrationState.incoming).
						// Snap regenerated — zone B is now the authoritative holder.
						xing_received++;
						slots[free_idx].entity = ent;
						slots[free_idx].snap = _make_ghost_snap(ent);
						slots[free_idx].hysteresis = 0;
						slots[free_idx].active = true;
						slots[free_idx].is_incoming = true;
						entity_count++;
						free_hint = (free_idx + 1) % _zone_capacity;
					} else if (_draining) {
						// No free slot during drain — handle based on zone role.
						if (zone_id == 0) {
							// Zone 0: write-through to drain buffer (no slot needed).
							_drain_collect_entity(ent);
						} else if (zone_id > 0 && fabric_peer.is_valid()) {
							// Intermediate zone: passthrough — forward toward zone 0.
							int fwd_target = zone_id - 1;
							Vector<uint8_t> fwd_pkt = _pack_intent(ent.global_id, fwd_target,
									tick + latency_ticks_runtime, ent);
							peer_callbacks.send_to_zone_raw(fabric_peer.ptr(), fwd_target,
									CH_MIGRATION, fwd_pkt.ptr(), fwd_pkt.size());
						}
					}
				}
				offset += INTENT_SIZE;
			}
		}
		intent_inbox.clear();
	}

	// ── INCOMING finalize: INCOMING â OWNED (zone B side, Lean: MigrationState.incoming) ──
	// Zone B received the migration intent last tick (is_incoming=true).
	// This tick: fully activate as OWNED, send ACK_MAGIC back so zone A can resolve STAGING.
	for (int i = 0; i < _zone_capacity; i++) {
		if (slots[i].active && slots[i].is_incoming) {
			slots[i].is_incoming = false; // INCOMING â OWNED
			print_line(vformat("[XING_IN]    zone=%d tick=%d eid=%d pos=(%.2f,%.2f,%.2f)",
					my_id, tick, slots[i].entity.global_id,
					(double)slots[i].entity.cx, (double)slots[i].entity.cy, (double)slots[i].entity.cz));
			// Send ACK so zone A resolves STAGING immediately (ACK_MAGIC = 'ACKK').
			if (fabric_peer.is_valid()) {
				uint8_t ack_buf[8];
				uint32_t eid = (uint32_t)slots[i].entity.global_id;
				uint32_t amag = ACK_MAGIC;
				memcpy(ack_buf + 0, &eid, 4);
				memcpy(ack_buf + 4, &amag, 4);
				peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_MIGRATION, ack_buf, 8);
			}
		}
	}

	// ── STAGING timeout: rollback to OWNED if no ACK received within deadline ──
	// Normal path: ACK arrives from zone B and resolves STAGING immediately (above).
	// Timeout path: zone B was full or packet lost. Rollback: entity stays on zone A.
	// Timeout = 4 Ã latency_ticks (generous window to survive packet loss).
	for (int i = 0; i < _zone_capacity; i++) {
		if (slots[i].active && slots[i].is_staging) {
			int ni_tgt = (slots[i].migration_target_zone == my_id - 1) ? 0 : 1;
			uint32_t timeout = _staging_timeout(_neighbor_latency_ticks[ni_tgt], _rtt_measured[ni_tgt], hz);
			if (tick - slots[i].staging_send_tick >= timeout) {
				// No ACK — rollback: STAGING â OWNED on zone A.
				print_line(vformat("[XING_ROLLBACK] zone=%d tick=%d eid=%d (no ACK from zone=%d)",
						my_id, tick, slots[i].entity.global_id,
						slots[i].migration_target_zone));
				slots[i].is_staging = false;
				slots[i].migration_target_zone = -1;
				slots[i].hysteresis = 0; // Reset hysteresis — will re-attempt next crossing
			}
		}
	}

	// ── Player slot expiry (~3 s without CH_PLAYER update) ──────────────────
	// PLAYER_SLOT_TIMEOUT = PLAYER_SLOT_TIMEOUT_SECONDS (converted to ticks at runtime).
	// Players that disconnect or crash stop sending CH_PLAYER; their slot
	// is freed so the capacity budget returns to NPC/entity use.
	for (int i = 0; i < _zone_capacity; i++) {
		if (slots[i].active && slots[i].is_player_slot) {
			if (tick - slots[i].last_update_tick >= player_slot_timeout) {
				uint32_t pid = (uint32_t)(slots[i].entity.global_id - PLAYER_ENTITY_BASE);
				_player_slot_map.erase(pid);
				slots[i].active = false;
				slots[i].is_player_slot = false;
				entity_count--;
				free_hint = i;
			}
		}
	}

	total_sync_us += OS::get_singleton()->get_ticks_usec() - t1;
	tick += 1;

	// Per-tick minimal-repro diag line. Emitted only on ticks where something
	// migration-relevant happened, so the 144-tracer chokepoint doesn't spam
	// but tier-0/tier-1 single-tracer runs show every interesting tick.
	{
		const int xing_in_delta = (int)(xing_received - xing_received_at_tick_start);
		if (diag_wrong_zone > 0 || diag_hys_reset > 0 || xing_in_delta > 0 || diag_ready_block > 0) {
			print_line(vformat("[diag zone %d] tick=%d wrong_zone=%d hys_reset=%d gate_skip=%d ready_block=%d max_hys=%d xing_in_dt=%d xing_received=%d",
					zone_id, tick, diag_wrong_zone, diag_hys_reset,
					diag_gate_skip, diag_ready_block, diag_max_hys,
					xing_in_delta, (int)xing_received));
		}
	}

	// Semantic log cadence: every 20 wall-seconds at the engine's current tick rate.
	if (tick % log_interval_ticks == 0) {
		int resnaps = 0;
		for (int i = 0; i < _zone_capacity; i++) {
			if (slots[i].active && slots[i].snap.ticks_since_snap == 0) {
				resnaps++;
			}
		}
		const char *scenario_name;
		switch (scenario) {
			case SCENARIO_JELLYFISH_BLOOM:
				scenario_name = "jellyfish_bloom_concert";
				break;
			case SCENARIO_JELLYFISH_ZONE_CROSSING:
				scenario_name = "jellyfish_zone_crossing";
				break;
			case SCENARIO_WHALE_WITH_SHARKS:
				scenario_name = "whale_with_sharks";
				break;
			case SCENARIO_CURRENT_FUNNEL:
				scenario_name = "current_funnel";
				break;
			case SCENARIO_MIXED:
				scenario_name = "mixed";
				break;
			default:
				scenario_name = "default";
				break;
		}
		uint64_t avg_compute = total_compute_us / tick;
		uint64_t avg_sync = total_sync_us / tick;
		int ent_count = entity_count;
		uint64_t ns_per_ent = ent_count > 0 ? (avg_compute * 1000) / ent_count : 0;
		// Broadphase pair counts: naive O(N^2) vs BVH O(N*log(N)/(2*delta))
		uint64_t naive_pairs = (uint64_t)ent_count * ((uint64_t)ent_count - 1) / 2;
		uint32_t log2n = 0;
		{
			uint32_t tmp = (uint32_t)ent_count + 1;
			while (tmp >>= 1) {
				log2n++;
			}
		}
		uint64_t bvh_pairs = (uint64_t)ent_count * log2n / 40; // /2*delta, delta~20
		// mspf: measured compute time in ms per physics tick.
		double mspf = (double)avg_compute * 0.001;
		print_line(vformat("[zone %d %s] tick %d cap=%d ents=%d mig=%d resnaps=%d mspf=%.2f compute=%d us/tick sync=%d us/tick (%d ns/ent) naive_pairs=%d bvh_pairs=%d xing_start=%d xing_done=%d xing_in=%d",
				my_id, scenario_name, tick, _zone_capacity, ent_count, (int)migrations, resnaps,
				mspf, (int)avg_compute, (int)avg_sync, (int)ns_per_ent,
				(int)naive_pairs, (int)bvh_pairs,
				(int)xing_started, (int)xing_done, (int)xing_received));
	}

	// ── Interest-culled entity publishing ───────────────────────────────
	if (!is_player && tick % INTEREST_PUBLISH_INTERVAL == 0) {
		Vector<uint8_t> buf;
		// 100 bytes per entity:
		//   4(gid) + 24(cx/cy/cz f64×3) + 12(vx/vy/vz/ax/ay/az int16×6) + 4(hlc) + 56(payload)
		buf.resize(_zone_capacity * 100); // Constant-work: fixed allocation regardless of occupancy.
		uint8_t *w = buf.ptrw();
		int write_pos = 0;
		uint8_t hlc_counter = 0;

		for (int i = 0; i < _zone_capacity; i++) {
			if (!slots[i].active) {
				continue;
			}
			Aabb g = _ghost_aabb_from_snap(slots[i].snap);
			if (!r128_eq(g.max_x, g.min_x) || !r128_eq(g.max_y, g.min_y)) {
				const FabricEntity &e = slots[i].entity;
				uint32_t gid = (uint32_t)e.global_id;
				double ecx = (double)e.cx, ecy = (double)e.cy, ecz = (double)e.cz;
				// Velocity and acceleration: quantise to int16 (μm/tick scale).
				int16_t ivx = (int16_t)CLAMP((int)(e.vx * V_SCALE), -32767, 32767);
				int16_t ivy = (int16_t)CLAMP((int)(e.vy * V_SCALE), -32767, 32767);
				int16_t ivz = (int16_t)CLAMP((int)(e.vz * V_SCALE), -32767, 32767);
				int16_t iax = (int16_t)CLAMP((int)(e.ax * A_SCALE), -32767, 32767);
				int16_t iay = (int16_t)CLAMP((int)(e.ay * A_SCALE), -32767, 32767);
				int16_t iaz = (int16_t)CLAMP((int)(e.az * A_SCALE), -32767, 32767);
				// HLC: tick(24b) | counter(8b). Counter disambiguates same-tick publishes.
				uint32_t hlc = ((tick & 0x00FFFFFFu) << 8) | hlc_counter++;
				memcpy(w + write_pos, &gid, 4);
				write_pos += 4;
				memcpy(w + write_pos, &ecx, 8);
				write_pos += 8;
				memcpy(w + write_pos, &ecy, 8);
				write_pos += 8;
				memcpy(w + write_pos, &ecz, 8);
				write_pos += 8;
				memcpy(w + write_pos, &ivx, 2);
				write_pos += 2;
				memcpy(w + write_pos, &ivy, 2);
				write_pos += 2;
				memcpy(w + write_pos, &ivz, 2);
				write_pos += 2;
				memcpy(w + write_pos, &iax, 2);
				write_pos += 2;
				memcpy(w + write_pos, &iay, 2);
				write_pos += 2;
				memcpy(w + write_pos, &iaz, 2);
				write_pos += 2;
				memcpy(w + write_pos, &hlc, 4);
				write_pos += 4;
				memcpy(w + write_pos, e.payload, sizeof(e.payload));
				write_pos += (int)sizeof(e.payload);
			}
		}

		if (fabric_peer.is_valid() && write_pos > 0) {
			// Chunk to fit within ENet MTU (~1200 bytes per packet).
			int chunk_offset = 0;
			while (chunk_offset < write_pos) {
				int chunk_size = MIN(1200, write_pos - chunk_offset);
				peer_callbacks.broadcast_raw(fabric_peer.ptr(), CH_INTEREST, w + chunk_offset, chunk_size);
				chunk_offset += chunk_size;
			}
		}
	}

	return false; // Don't quit.
}

// ── Migration sub-routines (public static for testability) ──────────────────

void FabricZone::_resolve_staging_timeouts_s(EntitySlot *p_slots, int p_capacity,
		int p_zone_id, const uint32_t p_neighbor_latency[2],
		const bool p_rtt_measured[2], uint32_t p_hz, uint32_t p_tick) {
	for (int i = 0; i < p_capacity; i++) {
		if (p_slots[i].active && p_slots[i].is_staging) {
			int ni_tgt = (p_slots[i].migration_target_zone == p_zone_id - 1) ? 0 : 1;
			uint32_t timeout = _staging_timeout(p_neighbor_latency[ni_tgt], p_rtt_measured[ni_tgt], p_hz);
			if (p_tick - p_slots[i].staging_send_tick >= timeout) {
				print_line(vformat("[XING_ROLLBACK] zone=%d tick=%d eid=%d (no ACK from zone=%d)",
						p_zone_id, p_tick, p_slots[i].entity.global_id,
						p_slots[i].migration_target_zone));
				p_slots[i].is_staging = false;
				p_slots[i].migration_target_zone = -1;
				p_slots[i].hysteresis = 0;
			}
		}
	}
}

int FabricZone::_accept_incoming_intents_s(EntitySlot *p_slots, int p_capacity,
		int &r_entity_count, int &r_free_hint, int p_zone_id,
		LocalVector<Vector<uint8_t>> &r_inbox, uint64_t &r_xing_received) {
	int accepted = 0;
	for (uint32_t di = 0; di < r_inbox.size(); di++) {
		const Vector<uint8_t> &data = r_inbox[di];
		int offset = 0;
		while (offset + INTENT_SIZE <= data.size()) {
			int eid, to_zone;
			uint32_t arrival;
			FabricEntity ent;
			if (_unpack_intent(data.ptr() + offset, INTENT_SIZE, eid, to_zone, arrival, ent)) {
				if (to_zone != p_zone_id) {
					offset += INTENT_SIZE;
					continue;
				}
				int free_idx = -1;
				for (int fi = 0; fi < p_capacity; fi++) {
					int idx = (r_free_hint + fi) % p_capacity;
					if (!p_slots[idx].active) {
						free_idx = idx;
						break;
					}
				}
				if (free_idx >= 0) {
					r_xing_received++;
					p_slots[free_idx].entity = ent;
					p_slots[free_idx].snap = _make_ghost_snap(ent);
					p_slots[free_idx].hysteresis = 0;
					p_slots[free_idx].active = true;
					p_slots[free_idx].is_incoming = true;
					r_entity_count++;
					r_free_hint = (free_idx + 1) % p_capacity;
					accepted++;
				}
			}
			offset += INTENT_SIZE;
		}
	}
	r_inbox.clear();
	return accepted;
}

int FabricZone::_collect_migration_intents_s(EntitySlot *p_slots, int p_capacity,
		int p_zone_id, int p_zone_count, const uint32_t p_neighbor_latency[2],
		uint32_t p_tick, uint32_t p_hz, int p_budget,
		uint64_t &r_xing_started, uint64_t &r_migrations,
		LocalVector<Vector<uint8_t>> &r_outbox) {
	int sent = 0;
	uint32_t hysteresis_runtime = pbvh_hysteresis_threshold(p_hz);

	for (int i = 0; i < p_capacity && p_budget > 0; i++) {
		if (!p_slots[i].active || p_slots[i].is_staging || p_slots[i].is_incoming || p_slots[i].is_player_slot) {
			continue;
		}
		const FabricEntity &e = p_slots[i].entity;
		int target = _zone_for_hilbert(_entity_hilbert(e), p_zone_count);
		if (target != p_zone_id) {
			p_slots[i].hysteresis += 1;
			if (p_slots[i].hysteresis >= hysteresis_runtime && p_budget > 0) {
				r_xing_started++;
				p_slots[i].is_staging = true;
				p_slots[i].staging_send_tick = p_tick;
				p_slots[i].migration_target_zone = target;
				r_migrations += 1;
				int ni_tgt = (target == p_zone_id - 1) ? 0 : 1;
				uint32_t lt = p_neighbor_latency[ni_tgt];
				Vector<uint8_t> pkt = _pack_intent(e.global_id, target, p_tick + lt, e);
				r_outbox.push_back(pkt);
				p_budget--;
				sent++;
			}
		} else {
			p_slots[i].hysteresis = 0;
		}
	}
	return sent;
}
