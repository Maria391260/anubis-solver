/* Wall-clock without an OS: the TSC, calibrated once against a 10 ms firmware
 * stall so the JSON can report microseconds instead of raw cycles. */
#pragma once

#include <stdint.h>

void timer_init(void);
uint64_t timer_now(void);
uint64_t timer_us(uint64_t delta);
