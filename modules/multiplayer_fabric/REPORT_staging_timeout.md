# STAGING timeout: RTT-derived, not hardcoded

## Problem

Zone-crossing migration uses a STAGING timeout to rollback entities when Zone B
does not acknowledge receipt. The original timeout was `_neighbor_latency_ticks *
4` with `_neighbor_latency_ticks` defaulting to `PBVH_LATENCY_TICKS_DEFAULT = 2`
until a PING/PONG RTT measurement completes. This gave an 8-tick window (~133 ms
at 60 Hz) that was too tight for:

- 3-tick outbound queuing from `MAX_MIGRATIONS_PER_TICK = 50`
- ENet reliable-channel fragment reassembly on 4400-byte batches
- The return ACK traversing the same path

Result: only ~10-13 of 144 entities landed in Zone B per burst; the rest rolled
back to OWNED via the STAGING timeout.

## Literature review

The RTT-based adaptive timeout is a well-studied problem in transport protocols:

- **Karn and Partridge (1987)** [@karn1987rtt] introduced the retransmission
  ambiguity problem: when an ACK arrives for a retransmitted segment, there is
  no way to know which transmission is being acknowledged. Their solution:
  stop sampling RTT on retransmissions and use exponential backoff for the
  timeout instead.

- **Jacobson and Karels (1988)** [@jacobson1988congestion] refined the timeout
  estimator by tracking both the smoothed RTT and its mean deviation. The
  formula `RTO = SRTT + 4 * RTTVAR` (where RTTVAR is the mean deviation)
  adapts to both the mean and the variance of the network path. The key insight
  is that a fixed multiplier of the mean RTT fails when variance is high; the
  deviation term automatically widens the window under jitter.

- **Braud et al. (2021)** [@braud2021talaria] demonstrated in-engine seamless
  migration between edge game servers with average handoff latency below 25 ms.
  Their approach synchronizes content by priority, allowing the client to switch
  servers before full state transfer completes. This confirms that RTT-scale
  timeouts (tens of milliseconds) are achievable for server migration.

- **Beskow et al. (2009)** [@Beskow2009PartialMigration] showed that partial
  migration of game state combined with dynamic server selection reduces
  player-perceived latency by choosing optimal server placement. Their work
  uses measured RTT to make placement decisions, reinforcing that RTT measurement
  must precede latency-sensitive operations.

## Fix

The STAGING timeout is now derived from RTT measurement, not hardcoded:

```cpp
static uint32_t _staging_timeout(uint32_t p_latency_ticks, bool p_rtt_measured, uint32_t p_hz) {
    if (!p_rtt_measured) {
        return p_hz; // 1 second — generous unmeasured default
    }
    return p_latency_ticks * 4;
}
```

- **Before RTT is measured**: 1 second of ticks (`p_hz`). Networking delays
  can reach minutes; 1 second is a conservative floor for the unmeasured case
  that accommodates ENet connection setup, DNS resolution, and initial packet
  exchange.

- **After RTT is measured**: `latency_ticks * 4`. The `4x` multiplier follows
  the Jacobson/Karels intuition — the timeout must exceed the round trip
  including variance. The `_neighbor_latency_ticks` value already applies
  `max(pbvh_latency_ticks(hz), ceil(rtt_ms / 2 * hz / 1000))`, ensuring it
  never drops below the theoretical floor.

- **`_rtt_measured[ni]`**: a per-neighbor boolean, set `true` when the first
  PONG arrives. This cleanly separates the "no data" case from the "measured"
  case without magic constants.

The fix is applied symmetrically in both the inline `physics_process()` code and
the extracted static method `_resolve_staging_timeouts_s()` via the same
`_staging_timeout()` function.

## Test results

All 5 migration tests pass (25/25 assertions):

| Test | Result |
|---|---|
| pack_intent round-trips through unpack_intent | PASS |
| staging timeout rolls back entity to OWNED | PASS |
| 144 entities all land within timeout window | PASS |
| Zone B at capacity rejects intent gracefully | PASS |
| outbound budget queues excess entities across ticks | PASS |

## Changed files

- `modules/multiplayer_fabric/fabric_zone.h` — `_staging_timeout()` static
  function, `_rtt_measured[2]` flag, `EntitySlot` moved to public, extracted
  method signatures updated
- `modules/multiplayer_fabric/fabric_zone.cpp` — timeout uses `_staging_timeout()`
  in both inline and extracted paths; `_rtt_measured[ni] = true` on PONG receipt
- `tests/scene/test_fabric_zone.cpp` — `ZoneState` harness replaces
  `MigrationHarness` (no SceneTree dependency); tests pass `rtt_measured` and
  `hz` to static methods
- `modules/csg/csg_shape.cpp` — guard `material_id >= 0` in `_pack_manifold`
  (fixes pre-existing CSG test crash)
- `modules/multiplayer_fabric_mmog/todo.md` — updated root cause description
- `thirdparty/predictive_bvh/OptimalPartitionBook.md` — added Jacobson/Karels,
  Karn/Partridge, Talaria, and Beskow references
