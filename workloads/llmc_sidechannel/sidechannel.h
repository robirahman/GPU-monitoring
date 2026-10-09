/*
Side-channel instrumentation and workload shaping for power-trace experiments.

Two jobs:

  1. MARKER OUTPUT. Assert a hardware line for exactly the region of interest
     (one forward+backward), so the marked window lands in the *same timebase*
     as the current-clamp trace. This removes the host-clock-to-scope offset
     problem entirely: you no longer have to trust that the scope started
     sampling when the host thought it did.

     The line is the DTR pin of a USB-UART (CP2102/FTDI). No extra hardware
     beyond a $2 adapter. Feed DTR into a spare scope channel.

  2. WORKLOAD SHAPING. Optionally reshape the *cadence* of training so that its
     power envelope resembles request-driven inference: idle gaps between
     micro-steps, with jitter. The arithmetic performed is bit-for-bit
     unchanged — that is the entire point of the experiment. What changes is
     only when it happens.

  3. TIMELINE LOG. A CSV of host timestamps at every phase boundary, plus the
     CUDA-event elapsed time for the same region, so the two clocks can be
     cross-checked against each other and against the scope.

IMPORTANT — why no explicit cudaDeviceSynchronize() calls appear around the
marker: llm.c is already host-synchronous at every phase boundary.
gpt2_forward(), gpt2_backward_and_reduce() and gpt2_update() each END with
cudaCheck(cudaDeviceSynchronize()). So when control returns to the host, the
GPU work for that phase is genuinely finished, and a marker toggle placed
between those calls brackets *computation*, not merely kernel launches.
If you port this to a codebase that does NOT sync, you must add the syncs
yourself or the marker will fall while the GPU is still busy.

MEASURED CAVEAT: toggling DTR is a USB control transfer, so it costs on the
order of 0.1-1 ms with jitter. Against a ~100-300 ms pass that is well under
1%, but it is not zero. Call sidechannel_calibrate() once to emit a known
pulse train and measure the real latency and jitter of *your* adapter from the
scope trace, rather than assuming.
*/
#ifndef SIDECHANNEL_H
#define SIDECHANNEL_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <termios.h>
#endif

// ----------------------------------------------------------------------------
// workload shaping modes

typedef enum {
    SC_MODE_TRAINING = 0,           // natural cadence, no gaps
    SC_MODE_MODIFIED_INFERENCE = 1, // same math, reshaped into gapped bursts
} ScWorkloadMode;

static const char* sc_mode_name(int m) {
    switch (m) {
        case SC_MODE_TRAINING:           return "training";
        case SC_MODE_MODIFIED_INFERENCE: return "modified-inference";
        default:                         return "unknown";
    }
}

// ----------------------------------------------------------------------------
// global state (single process, single scope — a struct would be overkill)

typedef struct {
    int   fd;                 // serial fd for the marker line, -1 if disabled
    int   mode;               // ScWorkloadMode
    long  gap_us;             // base inter-micro-step gap
    long  gap_jitter_us;      // uniform jitter added to the gap
    unsigned long long rng;   // xorshift state for the jitter
    FILE* timeline;           // CSV log, or NULL
    int   marker_state;       // current asserted state, for sanity checks
    long  toggles;            // count, for the summary line
} ScState;

static ScState sc = {-1, SC_MODE_TRAINING, 0, 0, 0x2545F4914F6CDD1DULL, NULL, 0, 0};

// ----------------------------------------------------------------------------
// clocks

// CLOCK_REALTIME, to correlate with the scope's unix capture timestamp
static double sc_unix_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// CLOCK_MONOTONIC, for durations (immune to NTP steps)
static double sc_mono_ms(void) {
#ifdef CLOCK_MONOTONIC
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
#else
    return sc_unix_s() * 1e3;
#endif
}

static unsigned long long sc_rand(void) {
    unsigned long long x = sc.rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    sc.rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// ----------------------------------------------------------------------------
// marker line

// Open the USB-UART and drop DTR to the de-asserted state.
static int sidechannel_open_marker(const char* device) {
#ifdef _WIN32
    (void)device; fprintf(stderr, "[sc] marker output not implemented on Windows\n");
    return -1;
#else
    if (device == NULL || device[0] == '\0') return -1;
    int fd = open(device, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "[sc] could not open marker device %s: %s\n",
                device, strerror(errno));
        return -1;
    }
    // Some drivers assert DTR on open; force it low so the trace starts clean.
    int bits = 0;
    if (ioctl(fd, TIOCMGET, &bits) == 0) {
        bits &= ~TIOCM_DTR;
        ioctl(fd, TIOCMSET, &bits);
    }
    sc.fd = fd;
    sc.marker_state = 0;
    fprintf(stderr, "[sc] marker line on %s (DTR)\n", device);
    return fd;
#endif
}

static void sidechannel_marker(int high) {
#ifndef _WIN32
    if (sc.fd < 0) { sc.marker_state = high; return; }
    int bits = 0;
    if (ioctl(sc.fd, TIOCMGET, &bits) != 0) return;
    if (high) bits |= TIOCM_DTR; else bits &= ~TIOCM_DTR;
    ioctl(sc.fd, TIOCMSET, &bits);
#endif
    sc.marker_state = high;
    sc.toggles++;
}

// ----------------------------------------------------------------------------
// timeline log

static void sidechannel_open_timeline(const char* path) {
    if (path == NULL || path[0] == '\0') return;
    sc.timeline = fopen(path, "w");
    if (sc.timeline == NULL) {
        fprintf(stderr, "[sc] could not open timeline log %s\n", path);
        return;
    }
    // t_unix_s correlates with the scope capture; t_mono_ms measures durations;
    // cuda_ms is the GPU-side elapsed time for the same region, where known.
    fprintf(sc.timeline,
            "step,micro_step,phase,marker,t_unix_s,t_mono_ms,cuda_ms,note\n");
    fflush(sc.timeline);
}

static void sc_log(int step, int micro, const char* phase, double cuda_ms,
                   const char* note) {
    if (sc.timeline == NULL) return;
    fprintf(sc.timeline, "%d,%d,%s,%d,%.6f,%.4f,%.4f,%s\n",
            step, micro, phase, sc.marker_state, sc_unix_s(), sc_mono_ms(),
            cuda_ms, note ? note : "");
}

// Marker toggle + log in one call, so the log records the state transition.
static void sc_mark(int step, int micro, const char* phase, int high) {
    sidechannel_marker(high);
    sc_log(step, micro, phase, -1.0, "");
}

// ----------------------------------------------------------------------------
// idle gap (modified-inference shaping)

static void sidechannel_gap(int step, int micro) {
    if (sc.mode != SC_MODE_MODIFIED_INFERENCE) return;
    if (sc.gap_us <= 0 && sc.gap_jitter_us <= 0) return;
    long us = sc.gap_us;
    if (sc.gap_jitter_us > 0) us += (long)(sc_rand() % (unsigned long long)sc.gap_jitter_us);
    sc_log(step, micro, "gap_begin", (double)us / 1000.0, "");
#ifndef _WIN32
    struct timespec req;
    req.tv_sec  = us / 1000000L;
    req.tv_nsec = (us % 1000000L) * 1000L;
    nanosleep(&req, NULL);
#endif
    sc_log(step, micro, "gap_end", -1.0, "");
}

// ----------------------------------------------------------------------------
// calibration: known pulse train, so marker latency/jitter can be measured

static void sidechannel_calibrate(int pulses, long width_us) {
    if (sc.fd < 0) {
        fprintf(stderr, "[sc] calibrate: no marker device, skipping\n");
        return;
    }
    fprintf(stderr, "[sc] calibration: %d pulses of %ld us, "
                    "50%% duty — measure these on the scope to obtain the "
                    "true marker latency and jitter\n", pulses, width_us);
    struct timespec req;
    req.tv_sec  = width_us / 1000000L;
    req.tv_nsec = (width_us % 1000000L) * 1000L;
    for (int i = 0; i < pulses; i++) {
        double t0 = sc_mono_ms();
        sidechannel_marker(1);
        double t1 = sc_mono_ms();
        nanosleep(&req, NULL);
        double t2 = sc_mono_ms();
        sidechannel_marker(0);
        double t3 = sc_mono_ms();
        nanosleep(&req, NULL);
        if (sc.timeline) {
            fprintf(sc.timeline, "-1,%d,calib,0,%.6f,%.4f,%.4f,"
                    "toggle_high_cost_ms=%.4f;toggle_low_cost_ms=%.4f\n",
                    i, sc_unix_s(), t0, t2 - t1, t1 - t0, t3 - t2);
        }
    }
    fflush(sc.timeline ? sc.timeline : stderr);
}

// ----------------------------------------------------------------------------
// lifecycle

static void sidechannel_init(const char* marker_device, const char* timeline_path,
                            int mode, long gap_us, long gap_jitter_us,
                            unsigned long long seed) {
    sc.mode = mode;
    sc.gap_us = gap_us;
    sc.gap_jitter_us = gap_jitter_us;
    if (seed) sc.rng = seed;
    sidechannel_open_timeline(timeline_path);
    sidechannel_open_marker(marker_device);
    fprintf(stderr, "[sc] mode=%s gap=%ld us jitter=%ld us marker=%s timeline=%s\n",
            sc_mode_name(mode), gap_us, gap_jitter_us,
            sc.fd >= 0 ? "on" : "off",
            sc.timeline ? timeline_path : "off");
}

static void sidechannel_free(void) {
    if (sc.fd >= 0) {
        sidechannel_marker(0);
#ifndef _WIN32
        close(sc.fd);
#endif
        sc.fd = -1;
    }
    if (sc.timeline) { fclose(sc.timeline); sc.timeline = NULL; }
    fprintf(stderr, "[sc] %ld marker toggles\n", sc.toggles);
}

#endif // SIDECHANNEL_H
