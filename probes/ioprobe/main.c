/* PS5LM ioprobe: how fast can expert weights come off a drive?
 *
 * Streaming MoE experts from storage (docs/OFFLOAD.md) reads one expert slab
 * per tensor, about 0.6 MiB at Q4_K_M for Qwen3.6/Ornith 35B-A3B, from random
 * places in the GGUF, from a few threads at once. This reads a big file that
 * way: sequential 8 MiB reads first, then random 4 KiB-aligned reads of 0.5,
 * 1 and 2 MiB from 1, 4 and 8 threads, with O_DIRECT (no page cache) and
 * buffered. 512 MiB per run, a 2 MiB buffer per thread: safe next to the app.
 *
 * usage (elfldr): ioprobe.elf?args=/data/PS5LM/models/<big>.gguf
 * Pick a file larger than the console's memory to keep buffered runs honest.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MiB ((size_t)1024 * 1024)
#define TOTAL (512 * MiB)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct job {
    int fd;
    off_t file_len;
    size_t chunk;
    size_t reads;
    unsigned seed;
    int seq;     // sequential from offset 0 instead of random
    int failed;
};

static void *reader(void *arg) {
    struct job *j = arg;
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, j->chunk) != 0) {
        j->failed = 1;
        return NULL;
    }
    const off_t span = (j->file_len - (off_t)j->chunk) / 4096;
    for (size_t i = 0; i < j->reads; i++) {
        off_t off = j->seq ? (off_t)(i * j->chunk) : (off_t)(rand_r(&j->seed) % span) * 4096;
        if (pread(j->fd, buf, j->chunk, off) != (ssize_t)j->chunk) {
            j->failed = 1;
            break;
        }
    }
    free(buf);
    return NULL;
}

static void run(const char *path, off_t len, int direct, int seq, size_t chunk, int threads) {
    pthread_t th[8];
    struct job jobs[8];
    int fds[8];
    for (int i = 0; i < threads; i++) {
        fds[i] = open(path, O_RDONLY | (direct ? O_DIRECT : 0));
        if (fds[i] < 0) {
            printf("%-8s open failed\n", direct ? "direct" : "buffered");
            for (int k = 0; k < i; k++)
                close(fds[k]);
            return;
        }
        jobs[i] = (struct job){fds[i], len, chunk, TOTAL / chunk / threads, 12345u + i * 7919u, seq, 0};
    }
    const double t0 = now_s();
    for (int i = 0; i < threads; i++)
        pthread_create(&th[i], NULL, reader, &jobs[i]);
    int failed = 0;
    for (int i = 0; i < threads; i++) {
        pthread_join(th[i], NULL);
        failed |= jobs[i].failed;
        close(fds[i]);
    }
    const double dt = now_s() - t0;
    printf("%-8s %-10s %4zu KiB x %d threads: %7.0f MB/s%s\n", direct ? "direct" : "buffered",
           seq ? "sequential" : "random", chunk / 1024, threads, TOTAL / dt / 1e6, failed ? " (a read failed)" : "");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: ioprobe <file larger than 1 GiB>\n");
        return 1;
    }
    struct stat st;
    if (stat(argv[1], &st) != 0 || st.st_size < (off_t)(2 * TOTAL)) {
        printf("ioprobe: %s is missing or under 1 GiB\n", argv[1]);
        return 1;
    }
    printf("PS5LM ioprobe: %s, %.2f GiB\n", argv[1], st.st_size / (double)(1024 * MiB));
    static const size_t chunks[] = {MiB / 2, MiB, 2 * MiB};
    static const int threads[] = {1, 4, 8};
    for (int direct = 1; direct >= 0; direct--) {
        run(argv[1], st.st_size, direct, 1, 8 * MiB, 1);
        for (size_t c = 0; c < 3; c++)
            for (size_t t = 0; t < 3; t++)
                run(argv[1], st.st_size, direct, 0, chunks[c], threads[t]);
    }
    printf("done\n");
    return 0;
}
