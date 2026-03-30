/*
 * Thin wrapper around the VL53L8CX ULD that exposes a simple opaque-handle
 * API for the Rust FFI layer.
 */

#ifndef _TOF_WRAPPER_H_
#define _TOF_WRAPPER_H_
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * tof_open  — open /dev/i2c-<bus>, configure VL53L8CX at 7-bit <addr>,
 *             and start 8×8 ranging.
 *
 *   ranging_mode    : 1 = continuous, 3 = autonomous
 *   integration_ms  : 2–1000 ms
 *
 * Returns an opaque handle on success, NULL on failure.
 * On failure an error string is written into errbuf (if non-NULL, <errbuf_len>
 * bytes including NUL terminator).
 */
void *tof_open(uint8_t bus, uint8_t addr_7bit,
               uint8_t ranging_mode, uint32_t integration_ms,
               char *errbuf, int errbuf_len);

/*
 * tof_read  — poll once for a ready scan (up to 200 ms) and return the
 *             8 column-minimum distances in metres.
 *
 *   out_ranges_m   : caller-supplied float[8], filled on success
 *   out_t_ms       : milliseconds since tof_open (for scan timestamp)
 *   row_min        : first row to include in per-column minimum (0–7)
 *   row_max        : last row to include (inclusive, 0–7, must be >= row_min)
 *
 * Row 0 is the topmost zone (sensor aperture looking forward on a flat robot).
 * Row 7 is the bottommost zone — typically angled downward toward the floor.
 * Use row_min=0, row_max=5 to skip the two bottom floor-facing rows.
 *
 * Returns 0 on success, nonzero on error.
 */
int tof_read(void *handle, float *out_ranges_m, uint64_t *out_t_ms,
             int row_min, int row_max);

/*
 * tof_close  — stop ranging and free all resources.
 */
void tof_close(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* _TOF_WRAPPER_H_ */
