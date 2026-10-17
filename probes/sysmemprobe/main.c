/* PS5LM sysmemprobe: where the memory outside the title's 12 GiB goes.
 *
 * Read-only: allocates nothing beyond its own buffers. Prints the kernel's
 * page counters, every sysctl whose name looks memory-related (the PS5 keeps
 * FreeBSD's tree and adds its own), and every process by resident size.
 * Run it while the app is open to see the title next to the system.
 *
 * Every line goes to stdout (elfldr streams it back to scripts/send.sh) and to
 * /data/PS5LM/sysmemprobe.txt.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/user.h>
#include <unistd.h>

#define OUT_FILE "/data/PS5LM/sysmemprobe.txt"
#define MiB      (1024.0 * 1024.0)

static FILE *out_file;
static long  page_bytes = 4096;

static void out(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (out_file) {
        va_start(ap, fmt);
        vfprintf(out_file, fmt, ap);
        va_end(ap);
    }
}

/* An int or a long sysctl, little endian either way; -1 when absent. */
static long long number(const char *name) {
    unsigned long long v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0) != 0 || len > sizeof(v))
        return -1;
    return len == 4 ? (long long)(uint32_t)v : (long long)v;
}

static void counters(void) {
    static const char *const pages[] = {
        "vm.stats.vm.v_page_count",     "vm.stats.vm.v_free_count",   "vm.stats.vm.v_active_count",
        "vm.stats.vm.v_inactive_count", "vm.stats.vm.v_laundry_count", "vm.stats.vm.v_wire_count",
        "vm.stats.vm.v_cache_count",    "vm.max_wired",
    };
    static const char *const bytes[] = {"hw.physmem", "hw.realmem", "hw.usermem"};

    out("== page counters (MiB)\n");
    for (size_t i = 0; i < sizeof(pages) / sizeof(pages[0]); i++) {
        const long long v = number(pages[i]);
        if (v < 0)
            out("%-30s unavailable (errno %d)\n", pages[i], errno);
        else
            out("%-30s %10.1f  (%lld pages)\n", pages[i], v * page_bytes / MiB, v);
    }
    for (size_t i = 0; i < sizeof(bytes) / sizeof(bytes[0]); i++) {
        const long long v = number(bytes[i]);
        if (v < 0)
            out("%-30s unavailable (errno %d)\n", bytes[i], errno);
        else
            out("%-30s %10.1f\n", bytes[i], v / MiB);
    }
}

/* Walks the whole sysctl tree ({0,2} gives the next OID, {0,1} its name) and
 * prints the names that look memory-related, with their value when numeric. */
static void memory_sysctls(void) {
    static const char *const words[] = {"mem", "budget", "dmem", "flex", "wire", "swap", "page", "game", "vm.kmem"};
    int    oid[CTL_MAXNAME + 2] = {0, 2, 1};
    size_t oid_len = 1;

    out("== memory-related sysctls\n");
    for (;;) {
        int    next[CTL_MAXNAME];
        size_t next_len = sizeof(next);
        if (sysctl(oid, (u_int)(oid_len + 2), next, &next_len, NULL, 0) != 0) {
            out("walk ended (errno %d)\n", errno);
            break;
        }
        oid_len = next_len / sizeof(int);
        memcpy(oid + 2, next, next_len);

        int  q[CTL_MAXNAME + 2] = {0, 1};
        char name[256];
        size_t name_len = sizeof(name);
        memcpy(q + 2, next, next_len);
        if (sysctl(q, (u_int)(oid_len + 2), name, &name_len, NULL, 0) != 0)
            continue;
        int hit = 0;
        for (size_t i = 0; i < sizeof(words) / sizeof(words[0]) && !hit; i++)
            hit = strstr(name, words[i]) != NULL;
        if (!hit)
            continue;

        unsigned long long v = 0;
        size_t len = sizeof(v);
        const int rc = sysctl(next, (u_int)oid_len, &v, &len, NULL, 0);
        if (rc == 0 && (len == 4 || len == 8))
            out("%-44s %llu\n", name, len == 4 ? (unsigned long long)(uint32_t)v : v);
        else
            out("%-44s (%s)\n", name, rc == 0 ? "not a number" : "unreadable");
    }
}

/* The biggest processes, for the per-process calls below. */
static pid_t top_pids[10];
static char  top_names[10][COMMLEN + 1];
static int   top_count;

struct proc_row {
    pid_t  pid;
    double rss, virt;
    char   comm[COMMLEN + 1];
};

static int by_rss(const void *a, const void *b) {
    const struct proc_row *x = a, *y = b;
    return (y->rss > x->rss) - (y->rss < x->rss);
}

/* The PS5's kinfo_proc is longer than the SDK's (1096 bytes against 1088), so
 * records are stepped by the size the kernel reports; the fields read here
 * sit in the shared prefix, and a bad pid or name ends the walk. */
static void processes(void) {
    int    mib[3] = {CTL_KERN, KERN_PROC, KERN_PROC_PROC};
    size_t len = 0;

    out("== processes by resident size\n");
    if (sysctl(mib, 3, NULL, &len, NULL, 0) != 0) {
        out("kern.proc unavailable (errno %d)\n", errno);
        return;
    }
    len += len / 4;
    char *buf = malloc(len);
    if (!buf || sysctl(mib, 3, buf, &len, NULL, 0) != 0) {
        out("kern.proc unreadable (errno %d)\n", errno);
        free(buf);
        return;
    }
    const size_t stride = len ? (size_t)((struct kinfo_proc *)buf)->ki_structsize : 0;
    if (stride < sizeof(struct kinfo_proc) || stride > 4096) {
        out("kinfo_proc stride %zu: not walking it\n", stride);
        free(buf);
        return;
    }
    const size_t n = len / stride;
    struct proc_row *rows = calloc(n, sizeof(*rows));
    size_t m = 0;
    for (size_t i = 0; rows && i < n; i++) {
        const struct kinfo_proc *k = (const void *)(buf + i * stride);
        if (k->ki_structsize != (int)stride || k->ki_pid < 0 || memchr(k->ki_comm, 0, sizeof(k->ki_comm)) == NULL) {
            out("record %zu looks wrong, stopping\n", i);
            break;
        }
        rows[m].pid = k->ki_pid;
        rows[m].rss = k->ki_rssize * (double)page_bytes / MiB;
        rows[m].virt = k->ki_size / MiB;
        memcpy(rows[m].comm, k->ki_comm, sizeof(rows[m].comm));
        m++;
    }
    if (rows)
        qsort(rows, m, sizeof(rows[0]), by_rss);
    double total = 0;
    out("%6s %10s %10s  %s\n", "pid", "rss MiB", "virt MiB", "name");
    for (size_t i = 0; i < m; i++) {
        total += rows[i].rss;
        out("%6d %10.1f %10.1f  %s\n", rows[i].pid, rows[i].rss, rows[i].virt, rows[i].comm);
    }
    out("%zu processes, %.1f MiB resident in all\n", m, total);
    for (size_t i = 0; i < m && top_count < 10; i++, top_count++) {
        top_pids[top_count] = rows[i].pid;
        memcpy(top_names[top_count], rows[i].comm, sizeof(top_names[0]));
    }
    free(rows);
    free(buf);
}

/* Sony's "get" syscalls, numbered as in the PS4 table (elfldr's
 * sys_budget_set is 0x23b there too). The argument shapes are guesses, so
 * every buffer is oversized and zeroed and only nonzero qwords are printed;
 * an error return only means the guess was wrong. Nothing here sets,
 * creates or deletes. */
#define SYS_BUDGET_GET             0x23a
#define SYS_BUDGET_GETID           0x261
#define SYS_BUDGET_GET_PTYPE       0x262
#define SYS_GET_PROC_TYPE_INFO     0x264
#define SYS_GET_RESIDENT_COUNT     0x265
#define SYS_GET_RESIDENT_FMEM      0x267
#define SYS_GET_MAP_STATISTICS     0x282
#define SYS_GET_KERNEL_MEM_STATS   0x286
#define SYS_BUDGET_PTYPE_OF_BUDGET 0x28a

#define BUF_BYTES (64 * 1024)
static uint64_t scratch[BUF_BYTES / 8];

static void dump_nonzero(const char *what) {
    int shown = 0;
    for (size_t i = 0; i < BUF_BYTES / 8 && shown < 48; i++) {
        if (scratch[i] == 0)
            continue;
        out("  %s +0x%04zx  0x%016llx  %llu\n", what, i * 8, (unsigned long long)scratch[i],
            (unsigned long long)scratch[i]);
        shown++;
    }
    if (shown == 0)
        out("  %s: buffer untouched\n", what);
}

static void call_into_buffer(const char *what, int nr, long a, long b) {
    memset(scratch, 0, sizeof(scratch));
    errno = 0;
    const long rc = syscall(nr, a, b);
    out("%s = %ld (errno %d)\n", what, rc, errno);
    if (rc >= 0)
        dump_nonzero(what);
}

static void budgets(void) {
    out("== budgets and kernel memory\n");
    errno = 0;
    const long self = syscall(SYS_BUDGET_GETID);
    out("budget_getid (self) = %ld (errno %d)\n", self, errno);
    for (long id = 0; id < 16; id++) {
        errno = 0;
        const long ptype = syscall(SYS_BUDGET_GET_PTYPE, id);
        const int  e1 = errno;
        errno = 0;
        const long ptype2 = syscall(SYS_BUDGET_PTYPE_OF_BUDGET, id);
        out("budget %2ld: get_ptype %ld (errno %d), ptype_of_budget %ld (errno %d)\n", id, ptype, e1, ptype2, errno);
    }
    /* Each record: u32 resource, u32 flags, u64 limit, u64 used. Only ids
     * that answer 0 are shown. */
    for (long id = 0; id < 4096; id++) {
        int count = 64;
        memset(scratch, 0, sizeof(scratch));
        const long rc = syscall(SYS_BUDGET_GET, id, scratch, &count);
        if (rc != 0)
            continue;
        out("budget %ld: %d resources\n", id, count);
        for (int r = 0; r < count && r < 64; r++) {
            const uint64_t *e = scratch + r * 3;
            out("  resource %2u flags %u  limit %14llu (%9.1f MiB)  used %14llu (%9.1f MiB)\n", (unsigned)(e[0] & 0xffffffff),
                (unsigned)(e[0] >> 32), (unsigned long long)e[1], e[1] / MiB, (unsigned long long)e[2], e[2] / MiB);
        }
    }
    call_into_buffer("get_kernel_mem_statistics(buf)", SYS_GET_KERNEL_MEM_STATS, (long)scratch, 0);
    call_into_buffer("get_kernel_mem_statistics(buf, size)", SYS_GET_KERNEL_MEM_STATS, (long)scratch, BUF_BYTES);
    call_into_buffer("get_map_statistics(buf)", SYS_GET_MAP_STATISTICS, (long)scratch, 0);
    call_into_buffer("get_proc_type_info(buf)", SYS_GET_PROC_TYPE_INFO, (long)scratch, 0);
}

int main(void) {
    out_file = fopen(OUT_FILE, "w");
    const long long ps = number("hw.pagesize");
    if (ps > 0)
        page_bytes = ps;
    out("sysmemprobe: page size %ld\n", page_bytes);
    counters();
    processes();
    memory_sysctls();
    budgets();
    out("sysmemprobe: done\n");
    if (out_file)
        fclose(out_file);
    return 0;
}
