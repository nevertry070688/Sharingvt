/*
 * rtos_sim.c
 *
 * Emulates a hard-RTOS style single-CPU cooperative scheduler on top of
 * Linux/PREEMPT_RT pthreads.
 *
 *   - Exactly one thread is ever "running" (holds the CPU token) at a time.
 *   - A thread keeps the token until it calls one of: rtos_delay(),
 *     rtos_sleep(), or rtos_timesleep().
 *   - rtos_timesleep() blocks until rtos_wakeup() is called on it, or the
 *     timeout expires -- whichever comes first.
 *   - rtos_delay()/rtos_sleep() are NOT interruptible by wakeup(); a
 *     wakeup() delivered during a delay is latched (pending_flag) and
 *     consumed by the *next* timesleep() call, matching typical RTOS
 *     "pending event" semantics.
 *   - rtos_rcv_msg()/rtos_snd_msg() are built from the exact same
 *     block/wake machinery as timesleep/wakeup: rcv_msg blocks until a
 *     message is available or timeout, snd_msg enqueues a message and
 *     wakes a blocked receiver, with the same call-time ordering
 *     guarantee. The mailbox's own occupancy count plays the role that
 *     pending_flag plays for wakeup -- see the comments on each function.
 *   - Ordering guarantee: if you call rtos_wakeup(A) then rtos_wakeup(B),
 *     A is placed in the ready queue before B, regardless of which OS
 *     thread the Linux scheduler happens to run first. This is done by
 *     pushing the target directly onto the ready queue *inside*
 *     rtos_wakeup(), under the global lock -- not by letting the woken
 *     thread push itself later (which would only reflect Linux scheduling
 *     order, not call order).
 *
 * Build:   gcc -O2 -Wall -o rtos_sim rtos_sim.c -lpthread
 * Run as root (or with CAP_SYS_NICE) to get SCHED_FIFO.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

#define NTHREADS   4
#define RQ_CAP     NTHREADS   /* at most NTHREADS threads can ever be ready */
#define MSG_MAX_LEN 32        /* max bytes per message */
#define MBOX_DEPTH  8         /* messages a mailbox can hold before full */

typedef enum {
    TH_RUNNING,
    TH_READY,
    TH_BLOCKED_SLEEP,      /* rtos_delay/rtos_sleep - not wakeable */
    TH_BLOCKED_TIMESLEEP,  /* rtos_timesleep - wakeable by rtos_wakeup */
    TH_BLOCKED_RECV        /* rtos_rcv_msg - wakeable by rtos_snd_msg */
} state_t;

typedef struct {
    int           sender_id;
    size_t        len;
    unsigned char data[MSG_MAX_LEN];
} msg_t;

typedef struct {
    int             id;
    pthread_t       tid;
    pthread_cond_t  cv;
    volatile int    pending_flag;  /* count of outstanding wakeups; each
                                     * timesleep() consumes exactly one */
    state_t         state;

    /* Private mailbox for rtos_snd_msg/rtos_rcv_msg (FIFO ring buffer).
     * mb_count plays exactly the role pending_flag plays for wakeup:
     * its 0 -> nonzero transition is what triggers waking a blocked
     * receiver, and rcv_msg's fast path is "mb_count > 0 already". */
    msg_t           mailbox[MBOX_DEPTH];
    int             mb_head, mb_tail, mb_count;
} thread_t;

static thread_t threads[NTHREADS];

static pthread_mutex_t g_lock;        /* single lock protects everything below */
static int  rq[RQ_CAP];
static int  rq_head = 0, rq_tail = 0, rq_count = 0;
static int  current = -1;             /* id of thread currently holding CPU */

/* ---------- time helpers ---------- */

static struct timespec deadline_from_now(int ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec  += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_nsec -= 1000000000L;
        ts.tv_sec  += 1;
    }
    return ts;
}

/* ---------- ready queue (must hold g_lock) ---------- */

static void rq_push(int id)
{
    rq[rq_tail] = id;
    rq_tail = (rq_tail + 1) % RQ_CAP;
    rq_count++;
}

static int rq_pop(void)
{
    int id = rq[rq_head];
    rq_head = (rq_head + 1) % RQ_CAP;
    rq_count--;
    return id;
}

/* Only call when current == -1 (CPU idle). Hands the token to the head
 * of the FIFO ready queue, if any. */
static void schedule_next_locked(void)
{
    if (current == -1 && rq_count > 0) {
        int id = rq_pop();
        threads[id].state = TH_RUNNING;
        current = id;
        pthread_cond_signal(&threads[id].cv);
