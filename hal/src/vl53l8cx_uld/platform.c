/*
 * Linux i2c-dev platform layer for VL53L8CX ULD.
 *
 * I2C protocol: every transaction starts with a 2-byte big-endian register
 * address followed by the data bytes (write) or just the address (read).
 * For large writes (firmware upload ~32 KB per page) we chunk at 4 KB so
 * the kernel i2c-dev ioctl stays inside its usual buffer limits.
 */

#include "platform.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

/* Maximum bytes of *payload* per I2C write transaction (2-byte addr + this). */
#define CHUNK_SIZE  4096U

/* ── helpers ────────────────────────────────────────────────────────────── */

/* Write [addr_hi, addr_lo, data...] as a single I2C message. */
static uint8_t i2c_write_chunk(int fd, uint16_t reg, const uint8_t *data, uint32_t len)
{
    uint32_t buf_len = len + 2;
    uint8_t *buf = malloc(buf_len);
    if (!buf) return 1;

    buf[0] = (uint8_t)(reg >> 8);
    buf[1] = (uint8_t)(reg & 0xFF);
    memcpy(buf + 2, data, len);

    ssize_t ret = write(fd, buf, buf_len);
    free(buf);
    return (ret == (ssize_t)buf_len) ? 0 : 1;
}

/* ── platform API ───────────────────────────────────────────────────────── */

uint8_t VL53L8CX_WrByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t value)
{
    return i2c_write_chunk(p->fd, reg, &value, 1);
}

uint8_t VL53L8CX_RdByte(VL53L8CX_Platform *p, uint16_t reg, uint8_t *out)
{
    return VL53L8CX_RdMulti(p, reg, out, 1);
}

uint8_t VL53L8CX_WrMulti(VL53L8CX_Platform *p, uint16_t reg,
                          uint8_t *data, uint32_t size)
{
    uint32_t offset = 0;
    while (offset < size) {
        uint32_t chunk = (size - offset > CHUNK_SIZE) ? CHUNK_SIZE : (size - offset);
        uint16_t chunk_reg = (uint16_t)(reg + offset);
        if (i2c_write_chunk(p->fd, chunk_reg, data + offset, chunk))
            return 1;
        offset += chunk;
    }
    return 0;
}

uint8_t VL53L8CX_RdMulti(VL53L8CX_Platform *p, uint16_t reg,
                          uint8_t *out, uint32_t size)
{
    /* Write the 2-byte register address, then read the data. */
    uint8_t addr[2] = { (uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF) };

    struct i2c_msg msgs[2] = {
        { .addr  = p->address,
          .flags = 0,
          .len   = 2,
          .buf   = addr },
        { .addr  = p->address,
          .flags = I2C_M_RD,
          .len   = (uint16_t)size,
          .buf   = out },
    };
    struct i2c_rdwr_ioctl_data xfer = { .msgs = msgs, .nmsgs = 2 };

    return (ioctl(p->fd, I2C_RDWR, &xfer) < 0) ? 1 : 0;
}

uint8_t VL53L8CX_Reset_Sensor(VL53L8CX_Platform *p)
{
    (void)p;
    return 0;  /* No XSHUT on Pololu #3419 carrier */
}

void VL53L8CX_SwapBuffer(uint8_t *buf, uint16_t size)
{
    /* Reverse byte order within each 4-byte group (host↔sensor endian). */
    for (uint16_t i = 0; i < size; i += 4) {
        uint8_t t0 = buf[i], t1 = buf[i+1];
        buf[i]   = buf[i+3];
        buf[i+1] = buf[i+2];
        buf[i+2] = t1;
        buf[i+3] = t0;
    }
}

uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform *p, uint32_t ms)
{
    (void)p;
    struct timespec ts = { .tv_sec = ms / 1000,
                           .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
    return 0;
}
