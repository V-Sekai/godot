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

The STAGING timeout now implements the Jacobson/Karels (1988) adaptive RTO
estimator, tracking both smoothed RTT and its mean deviation per neighbor:

### Per-neighbor state

```cpp
uint32_t _srtt_ticks[2];    // smoothed one-way latency (EWMA, alpha = 1/8)
uint32_t _rttvar_ticks[2];  // mean deviation          (EWMA, beta  = 1/4)
bool     _rtt_measured[2];  // true once first PONG arrives
```

### PONG handler (Jacobson/Karels EWMA update)

On the first PONG, SRTT is initialized to the sample and RTTVAR to half the
sample (RFC 6298 Section 2.2). Subsequent PONGs apply the standard EWMA:

```
RTTVAR = 3/4 * RTTVAR + 1/4 * |SRTT - sample|
SRTT   = 7/8 * SRTT   + 1/8 * sample
```

SRTT is floored at `pbvh_latency_ticks(hz)` (the proved minimum for 0 ms RTT).

### Timeout calculation

```cpp
static uint32_t _staging_timeout(uint32_t p_srtt, uint32_t p_rttvar,
        bool p_rtt_measured, uint32_t p_hz) {
    if (!p_rtt_measured) {
        return p_hz; // 1 second — generous unmeasured default
    }
    // Jacobson/Karels: RTO = SRTT + 4 * RTTVAR, floor of 1 tick.
    uint32_t rto = p_srtt + 4 * p_rttvar;
    return rto < 1 ? 1 : rto;
}
```

- **Before RTT is measured**: 1 second of ticks (`p_hz`). Networking delays
  can reach minutes; 1 second is a conservative floor for the unmeasured case
  that accommodates ENet connection setup, DNS resolution, and initial packet
  exchange.

- **After RTT is measured**: `SRTT + 4 * RTTVAR`. The variance term
  automatically widens the timeout window under jitter (packet reordering,
  GC pauses, ENet retransmits) and tightens it on stable links. This directly
  implements the Jacobson/Karels formula rather than using a fixed multiplier
  on the raw latency sample.

- **SRTT (arrival deadlines)**: `_srtt_ticks` is also used as the expected
  one-way transit time in `_pack_intent` arrival tick calculations, replacing
  the former raw `_neighbor_latency_ticks` value.

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

- `modules/multiplayer_fabric/fabric_zone.h` — `_staging_timeout()` takes
  SRTT + RTTVAR; `_srtt_ticks[2]`/`_rttvar_ticks[2]` replace
  `_neighbor_latency_ticks[2]`; `EntitySlot` moved to public; extracted
  method signatures carry SRTT/RTTVAR arrays
- `modules/multiplayer_fabric/fabric_zone.cpp` — PONG handler implements
  Jacobson/Karels EWMA (first sample initializes, subsequent samples smooth);
  all timeout and arrival deadline sites use `_srtt_ticks`/`_rttvar_ticks`
- `tests/scene/test_fabric_zone.cpp` — `ZoneState` harness carries
  `srtt[2]`/`rttvar[2]`; tests pass SRTT, RTTVAR, and `rtt_measured` to
  static methods
- `modules/csg/csg_shape.cpp` — guard `material_id >= 0` in `_pack_manifold`
  (fixes pre-existing CSG test crash)
- `modules/multiplayer_fabric_mmog/todo.md` — updated root cause description
- `thirdparty/predictive_bvh/OptimalPartitionBook.md` — added Jacobson/Karels,
  Karn/Partridge, Talaria, and Beskow references
