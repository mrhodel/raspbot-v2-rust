/*
 * Linux i2c-dev platform layer for VL53L8CX ULD.
 * Replaces the STM32-specific platform.h from ST's example.
 *
 * BSD 3-clause licence (same as VL53L8CX ULD).
 */

#ifndef _PLATFORM_H_
#define _PLATFORM_H_
#pragma once

#include <stdint.h>
#include <string.h>

/*
 * Platform handle: i2c-dev file descriptor + 7-bit I2C address.
 */
typedef struct {
    uint16_t address;   /* 7-bit I2C address (e.g. 0x29) */
    int      fd;        /* /dev/i2c-N file descriptor     */
} VL53L8CX_Platform;

/* One target per zone — keep I2C traffic minimal. */
#define VL53L8CX_NB_TARGET_PER_ZONE  1U

/* Disable every optional output except DISTANCE to shrink the per-scan
 * I2C payload to ~184 bytes (vs ~1400 bytes with everything on). */
#define VL53L8CX_DISABLE_AMBIENT_PER_SPAD
#define VL53L8CX_DISABLE_NB_SPADS_ENABLED
#define VL53L8CX_DISABLE_NB_TARGET_DETECTED
#define VL53L8CX_DISABLE_SIGNAL_PER_SPAD
#define VL53L8CX_DISABLE_RANGE_SIGMA_MM
/* VL53L8CX_DISABLE_DISTANCE_MM  — intentionally NOT defined */
#define VL53L8CX_DISABLE_REFLECTANCE_PERCENT
#define VL53L8CX_DISABLE_TARGET_STATUS
#define VL53L8CX_DISABLE_MOTION_INDICATOR

/* ── Mandatory platform functions ──────────────────────────────────────── */

uint8_t VL53L8CX_RdByte(
        VL53L8CX_Platform *p_platform,
        uint16_t           RegisterAddress,
        uint8_t           *p_value);

uint8_t VL53L8CX_WrByte(
        VL53L8CX_Platform *p_platform,
        uint16_t           RegisterAddress,
        uint8_t            value);

uint8_t VL53L8CX_RdMulti(
        VL53L8CX_Platform *p_platform,
        uint16_t           RegisterAddress,
        uint8_t           *p_values,
        uint32_t           size);

uint8_t VL53L8CX_WrMulti(
        VL53L8CX_Platform *p_platform,
        uint16_t           RegisterAddress,
        uint8_t           *p_values,
        uint32_t           size);

uint8_t VL53L8CX_Reset_Sensor(VL53L8CX_Platform *p_platform);

void    VL53L8CX_SwapBuffer(uint8_t *buffer, uint16_t size);

uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform *p_platform, uint32_t TimeMs);

#endif /* _PLATFORM_H_ */
