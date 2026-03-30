/*
 * Thin wrapper around the VL53L8CX ULD for use from Rust FFI.
 *
 * Allocates VL53L8CX_Configuration + VL53L8CX_ResultsData on the heap,
 * opens /dev/i2c-N, runs the full init/set_resolution/start_ranging sequence,
 * and exposes tof_open / tof_read / tof_close.
 */

#include "wrapper.h"
#include "vl53l8cx_api.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

/* ── internal handle ───────────────────────────────────────────────────── */

typedef struct {
    VL53L8CX_Configuration  dev;
    VL53L8CX_ResultsData    results;
    struct timespec          t_open;
} TofHandle;

/* ── helpers ────────────────────────────────────────────────────────────── */

static uint64_t ms_since(const struct timespec *t0)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t diff = (int64_t)(now.tv_sec  - t0->tv_sec ) * 1000
                 + (int64_t)(now.tv_nsec - t0->tv_nsec) / 1000000;
    return diff > 0 ? (uint64_t)diff : 0;
}

static void fmt_err(char *buf, int len, const char *msg, int rc)
{
    if (buf && len > 0)
        snprintf(buf, (size_t)len, "%s (rc=%d)", msg, rc);
}

/* ── public API ─────────────────────────────────────────────────────────── */

void *tof_open(uint8_t bus, uint8_t addr_7bit,
               uint8_t ranging_mode, uint32_t integration_ms,
               char *errbuf, int errbuf_len)
{
    TofHandle *h = calloc(1, sizeof(TofHandle));
    if (!h) {
        fmt_err(errbuf, errbuf_len, "calloc failed", 0);
        return NULL;
    }

    /* Open /dev/i2c-<bus> and configure slave address. */
    char dev_path[32];
    snprintf(dev_path, sizeof(dev_path), "/dev/i2c-%u", (unsigned)bus);
    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        fmt_err(errbuf, errbuf_len, "open i2c device failed", errno);
        free(h);
        return NULL;
    }
    if (ioctl(fd, I2C_SLAVE, (long)addr_7bit) < 0) {
        fmt_err(errbuf, errbuf_len, "ioctl I2C_SLAVE failed", errno);
        close(fd);
        free(h);
        return NULL;
    }

    h->dev.platform.fd      = fd;
    h->dev.platform.address = addr_7bit;

    uint8_t status = 0;

    /* 1. Init (SW reboot, firmware upload, calibration). */
    status = vl53l8cx_init(&h->dev);
    if (status) {
        fmt_err(errbuf, errbuf_len, "vl53l8cx_init failed", (int)status);
        goto fail;
    }

    /* 2. Switch to 8×8 resolution. */
    status = vl53l8cx_set_resolution(&h->dev, VL53L8CX_RESOLUTION_8X8);
    if (status) {
        fmt_err(errbuf, errbuf_len, "set_resolution 8x8 failed", (int)status);
        goto fail;
    }

    /* 3. Ranging mode (1=continuous, 3=autonomous). */
    uint8_t uld_mode = (ranging_mode == 3)
                       ? VL53L8CX_RANGING_MODE_AUTONOMOUS
                       : VL53L8CX_RANGING_MODE_CONTINUOUS;
    status = vl53l8cx_set_ranging_mode(&h->dev, uld_mode);
    if (status) {
        fmt_err(errbuf, errbuf_len, "set_ranging_mode failed", (int)status);
        goto fail;
    }

    /* 4. Integration time (2–1000 ms). */
    if (integration_ms < 2)  integration_ms = 2;
    if (integration_ms > 1000) integration_ms = 1000;
    status = vl53l8cx_set_integration_time_ms(&h->dev, integration_ms);
    if (status) {
        fmt_err(errbuf, errbuf_len, "set_integration_time_ms failed", (int)status);
        goto fail;
    }

    /* 5. Start ranging. */
    status = vl53l8cx_start_ranging(&h->dev);
    if (status) {
        fmt_err(errbuf, errbuf_len, "start_ranging failed", (int)status);
        goto fail;
    }

    /* Let the sensor produce its first frame before the caller polls. */
    VL53L8CX_WaitMs(&h->dev.platform, 100);

    clock_gettime(CLOCK_MONOTONIC, &h->t_open);
    return h;

fail:
    close(fd);
    free(h);
    return NULL;
}

int tof_read(void *handle, float *out_ranges_m, uint64_t *out_t_ms,
             int row_min, int row_max)
{
    TofHandle *h = (TofHandle *)handle;

    /* Clamp row range to valid bounds. */
    if (row_min < 0) row_min = 0;
    if (row_max > 7) row_max = 7;
    if (row_max < row_min) row_max = row_min;

    /* Poll for data ready — up to 1 s (100 × 10 ms). */
    uint8_t is_ready = 0;
    for (int i = 0; i < 100; i++) {
        uint8_t st = vl53l8cx_check_data_ready(&h->dev, &is_ready);
        if (st) return (int)st;
        if (is_ready) break;

        struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000L }; /* 10 ms */
        nanosleep(&ts, NULL);
    }
    if (!is_ready) return -1;   /* timeout */

    /* Fetch results. */
    uint8_t st = vl53l8cx_get_ranging_data(&h->dev, &h->results);
    if (st) return (int)st;

    if (out_t_ms)
        *out_t_ms = ms_since(&h->t_open);

    /*
     * distance_mm is a flat row-major array [row * 8 + col], int16_t,
     * already divided by 4 (mm units) by the ULD.
     * Compute the column-minimum across the caller-specified row range.
     * Row 0 = topmost zone (forward/upward); Row 7 = bottommost (floor-facing).
     * Excluding bottom rows (e.g. row_max=5) avoids false floor detections.
     */
    for (int col = 0; col < 8; col++) {
        int16_t min_mm = INT16_MAX;
        for (int row = row_min; row <= row_max; row++) {
            int16_t v = h->results.distance_mm[row * 8 + col];
            if (v > 0 && v < min_mm)
                min_mm = v;
        }
        out_ranges_m[col] = (min_mm == INT16_MAX)
                            ? 3.0f                         /* no target */
                            : ((float)min_mm / 1000.0f);   /* mm → m    */
        /* Clamp to [0.01, 4.0] m */
        if (out_ranges_m[col] < 0.01f) out_ranges_m[col] = 0.01f;
        if (out_ranges_m[col] > 4.0f)  out_ranges_m[col] = 4.0f;
    }

    return 0;
}

void tof_close(void *handle)
{
    if (!handle) return;
    TofHandle *h = (TofHandle *)handle;
    vl53l8cx_stop_ranging(&h->dev);
    close(h->dev.platform.fd);
    free(h);
}
