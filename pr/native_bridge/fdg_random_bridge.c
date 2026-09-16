/* Independent from-scratch native FDG bridge for random-assignment bundles.
 *
 * The input is deliberately not the 022 warm-start FDGBIN1 contract: it has
 * no coordinate payload.  The target map is built from an explicit bead grid
 * and raw integer edges, then hk_fdg is called with src == NULL.
 */
#include "hickit.h"
#include "hkpriv.h"
#include "krng.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_MAGIC "RNDINP1\0"
#define OUTPUT_MAGIC "RNDOUT1\0"
#define INPUT_VERSION 1u
#define OUTPUT_VERSION 1u
#define MAX_TRACKS 1024u
#define MAX_BEADS 10000000u
#define MAX_RAW_PAIRS 20000000u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t bin_size;
    uint32_t n_tracks;
    uint32_t n_beads;
    uint32_t n_raw_pairs;
    uint32_t flags;
} input_header_t;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t n_beads;
    uint32_t n_binned_pairs;
    uint32_t n_raw_pairs;
    uint32_t n_iter;
    uint32_t flags;
    float native_unit;
    float init_max_norm;
    float final_max_norm;
    float reserved_float;
} output_header_t;

/* hk_fdg_init is an exported symbol in the frozen vendored fdg.o. */
extern fvec3_t *hk_fdg_init(krng_t *rng, int n_beads, float max);

static int read_exact(FILE *fp, void *dst, size_t n)
{
    return n == 0 || fread(dst, 1, n, fp) == n;
}

static int read_u32_le(FILE *fp, uint32_t *out)
{
    uint8_t b[4];
    if (!read_exact(fp, b, sizeof(b))) return 0;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_i32_le(FILE *fp, int32_t *out)
{
    uint32_t raw;
    if (!read_u32_le(fp, &raw)) return 0;
    memcpy(out, &raw, sizeof(raw));
    return 1;
}

static int read_u64_decimal(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out = (uint64_t)value;
    return 1;
}

static int write_u32_le(FILE *fp, uint32_t value)
{
    uint8_t b[4] = {
        (uint8_t)value, (uint8_t)(value >> 8),
        (uint8_t)(value >> 16), (uint8_t)(value >> 24)
    };
    return fwrite(b, 1, sizeof(b), fp) == sizeof(b);
}

static int write_f32_le(FILE *fp, float value)
{
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    return write_u32_le(fp, raw);
}

static int file_exists(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return 0;
    fclose(fp);
    return 1;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --inspect --input FILE\n"
            "       %s --input FILE --output FILE --iterations N --seed N\n",
            prog, prog);
}

static int parse_args(int argc, char **argv, const char **input_path,
                      const char **output_path, uint32_t *iterations,
                      uint64_t *seed, int *inspect)
{
    int have_input = 0, have_output = 0, have_iterations = 0, have_seed = 0;
    int i;
    *inspect = 0;
    for (i = 1; i < argc; ++i) {
        uint64_t value;
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            *input_path = argv[++i];
            have_input = 1;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            *output_path = argv[++i];
            have_output = 1;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!read_u64_decimal(argv[++i], &value) || value == 0 || value > INT_MAX)
                return 0;
            *iterations = (uint32_t)value;
            have_iterations = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!read_u64_decimal(argv[++i], seed)) return 0;
            have_seed = 1;
        } else if (strcmp(argv[i], "--inspect") == 0) {
            *inspect = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            return 0;
        } else {
            return 0;
        }
    }
    if (!have_input) return 0;
    if (*inspect) return !have_output && !have_iterations && !have_seed;
    return have_output && have_iterations && have_seed;
}

static int fail_message(const char *message)
{
    fprintf(stderr, "fdg_random_bridge: %s\n", message);
    return 1;
}

static int validate_grid(const int32_t *lengths, uint32_t n_tracks,
                         uint32_t n_beads, const struct hk_bead *beads,
                         uint32_t bin_size)
{
    uint64_t expected_total = 0;
    uint32_t t;
    uint32_t offset = 0;
    if (n_tracks == 0 || (n_tracks & 1u) != 0 || bin_size == 0) return 0;
    for (t = 0; t < n_tracks; ++t) {
        uint64_t count;
        if (lengths[t] <= 0) return 0;
        count = ((uint64_t)(uint32_t)lengths[t] + bin_size - 1u) / bin_size;
        expected_total += count;
        if (expected_total > n_beads) return 0;
        {
            uint64_t k;
            for (k = 0; k < count; ++k) {
                uint64_t start = k * bin_size;
                uint64_t end = start + bin_size;
                if (end > (uint32_t)lengths[t]) end = (uint32_t)lengths[t];
                if (beads[offset + k].chr != (int32_t)t ||
                    beads[offset + k].st != (int32_t)start ||
                    beads[offset + k].en != (int32_t)end)
                    return 0;
            }
        }
        offset += (uint32_t)count;
    }
    return expected_total == n_beads && offset == n_beads;
}

static int build_dictionary(struct hk_sdict **out, const int32_t *lengths,
                            uint32_t n_tracks)
{
    struct hk_sdict *d = hk_sd_init();
    uint32_t t;
    if (d == NULL) return 0;
    for (t = 0; t < n_tracks; ++t) {
        char name[32];
        int chromosome = (int)(t / 2u) + 1;
        char copy = (t & 1u) ? 'b' : 'a';
        int32_t id;
        snprintf(name, sizeof(name), "c%02d%c", chromosome, copy);
        id = hk_sd_put(d, name, lengths[t]);
        if (id != (int32_t)t) {
            hk_sd_destroy(d);
            return 0;
        }
    }
    *out = d;
    return 1;
}

static int make_raw_pairs(const struct hk_bead *beads, uint32_t n_beads,
                          const uint32_t *bid0, const uint32_t *bid1,
                          uint32_t n_raw, struct hk_pair *raw)
{
    uint32_t i;
    for (i = 0; i < n_raw; ++i) {
        uint32_t a = bid0[i], b = bid1[i];
        struct hk_pair *p = &raw[i];
        if (a >= n_beads || b >= n_beads || a >= b) return 0;
        memset(p, 0, sizeof(*p));
        p->chr = ((uint64_t)(uint32_t)beads[a].chr << 32) |
                 (uint32_t)beads[b].chr;
        p->pos = ((uint64_t)(uint32_t)beads[a].st << 32) |
                 (uint32_t)beads[b].st;
        p->phase[0] = -1;
        p->phase[1] = -1;
        p->_.phased_prob = 0.0f;
    }
    return 1;
}

static int validate_coordinates(const fvec3_t *coords, uint32_t n_beads)
{
    uint32_t i;
    for (i = 0; i < n_beads; ++i) {
        if (!isfinite(coords[i][0]) || !isfinite(coords[i][1]) ||
            !isfinite(coords[i][2]))
            return 0;
    }
    return 1;
}

static float max_norm(const fvec3_t *coords, uint32_t n_beads)
{
    float maximum = 0.0f;
    uint32_t i;
    for (i = 0; i < n_beads; ++i) {
        float norm = sqrtf(coords[i][0] * coords[i][0] +
                           coords[i][1] * coords[i][1] +
                           coords[i][2] * coords[i][2]);
        if (norm > maximum) maximum = norm;
    }
    return maximum;
}

static int write_output(const char *path, uint32_t n_beads,
                        uint32_t n_binned_pairs, uint32_t n_raw_pairs,
                        uint32_t n_iter, float native_unit,
                        float init_max_norm, float final_max_norm,
                        const fvec3_t *native_init,
                        const fvec3_t *native_final)
{
    FILE *fp;
    uint32_t i, j;
    if (file_exists(path)) {
        fprintf(stderr, "fdg_random_bridge: refusing to overwrite output %s\n", path);
        return 0;
    }
    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "fdg_random_bridge: cannot open output %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    if (fwrite(OUTPUT_MAGIC, 1, 8, fp) != 8 ||
        !write_u32_le(fp, OUTPUT_VERSION) ||
        !write_u32_le(fp, n_beads) ||
        !write_u32_le(fp, n_binned_pairs) ||
        !write_u32_le(fp, n_raw_pairs) ||
        !write_u32_le(fp, n_iter) ||
        !write_u32_le(fp, 0) ||
        !write_f32_le(fp, native_unit) ||
        !write_f32_le(fp, init_max_norm) ||
        !write_f32_le(fp, final_max_norm) ||
        !write_f32_le(fp, 0.0f)) {
        fclose(fp);
        return 0;
    }
    for (i = 0; i < n_beads; ++i)
        for (j = 0; j < 3; ++j)
            if (!write_f32_le(fp, native_init[i][j])) {
                fclose(fp);
                return 0;
            }
    for (i = 0; i < n_beads; ++i)
        for (j = 0; j < 3; ++j)
            if (!write_f32_le(fp, native_final[i][j])) {
                fclose(fp);
                return 0;
            }
    if (fclose(fp) != 0) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    const char *input_path = NULL, *output_path = NULL;
    uint32_t iterations = 0;
    uint64_t seed = 0;
    int inspect = 0;
    FILE *fp = NULL;
    input_header_t header;
    int32_t *lengths = NULL;
    struct hk_bead *beads = NULL;
    uint32_t *bid0 = NULL, *bid1 = NULL;
    struct hk_pair *raw = NULL;
    fvec3_t *native_init = NULL;
    struct hk_sdict *dict = NULL;
    struct hk_bmap *target = NULL;
    krng_t rng;
    int64_t total_count = 0;
    int status = 1;
    int32_t i;

    if (!parse_args(argc, argv, &input_path, &output_path, &iterations,
                    &seed, &inspect)) {
        usage(argv[0]);
        return 2;
    }
    fp = fopen(input_path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "fdg_random_bridge: cannot open input %s: %s\n",
                input_path, strerror(errno));
        goto cleanup;
    }
    if (!read_exact(fp, header.magic, sizeof(header.magic)) ||
        memcmp(header.magic, INPUT_MAGIC, sizeof(header.magic)) != 0 ||
        !read_u32_le(fp, &header.version) ||
        !read_u32_le(fp, &header.bin_size) ||
        !read_u32_le(fp, &header.n_tracks) ||
        !read_u32_le(fp, &header.n_beads) ||
        !read_u32_le(fp, &header.n_raw_pairs) ||
        !read_u32_le(fp, &header.flags)) {
        fail_message("invalid or truncated input header");
        goto cleanup;
    }
    if (header.version != INPUT_VERSION || header.bin_size == 0 ||
        header.n_tracks == 0 || header.n_tracks > MAX_TRACKS ||
        header.n_beads == 0 || header.n_beads > MAX_BEADS ||
        header.n_raw_pairs == 0 || header.n_raw_pairs > MAX_RAW_PAIRS ||
        header.flags != 0) {
        fail_message("input header values are outside the random contract");
        goto cleanup;
    }
    lengths = (int32_t *)calloc(header.n_tracks, sizeof(*lengths));
    beads = (struct hk_bead *)calloc(header.n_beads, sizeof(*beads));
    bid0 = (uint32_t *)calloc(header.n_raw_pairs, sizeof(*bid0));
    bid1 = (uint32_t *)calloc(header.n_raw_pairs, sizeof(*bid1));
    raw = (struct hk_pair *)calloc(header.n_raw_pairs, sizeof(*raw));
    if (lengths == NULL || beads == NULL || bid0 == NULL || bid1 == NULL ||
        raw == NULL) {
        fail_message("allocation failed");
        goto cleanup;
    }
    for (i = 0; i < (int32_t)header.n_tracks; ++i)
        if (!read_i32_le(fp, &lengths[i])) {
            fail_message("truncated track lengths");
            goto cleanup;
        }
    for (i = 0; i < (int32_t)header.n_beads; ++i)
        if (!read_i32_le(fp, &beads[i].chr) ||
            !read_i32_le(fp, &beads[i].st) ||
            !read_i32_le(fp, &beads[i].en)) {
            fail_message("truncated explicit bead inventory");
            goto cleanup;
        }
    if (!validate_grid(lengths, header.n_tracks, header.n_beads, beads,
                       header.bin_size)) {
        fail_message("bead inventory is not the complete origin-0 grid");
        goto cleanup;
    }
    for (i = 0; i < (int32_t)header.n_raw_pairs; ++i)
        if (!read_u32_le(fp, &bid0[i]) || !read_u32_le(fp, &bid1[i])) {
            fail_message("truncated canonical raw edge list");
            goto cleanup;
        }
    if (fgetc(fp) != EOF) {
        fail_message("input blob contains trailing bytes");
        goto cleanup;
    }
    fclose(fp);
    fp = NULL;
    if (!make_raw_pairs(beads, header.n_beads, bid0, bid1,
                        header.n_raw_pairs, raw)) {
        fail_message("raw force edges must be canonical distinct bead ids");
        goto cleanup;
    }
    if (!build_dictionary(&dict, lengths, header.n_tracks)) {
        fail_message("could not build the explicit track dictionary");
        goto cleanup;
    }
    target = hk_bmap_gen_from_beads(dict, (int32_t)header.n_beads, beads,
                                    (int32_t)header.n_raw_pairs, raw);
    if (target == NULL || (uint32_t)target->n_beads != header.n_beads) {
        fail_message("hk_bmap_gen_from_beads failed or changed bead inventory");
        goto cleanup;
    }
    for (i = 0; i < target->n_pairs; ++i) {
        if (target->pairs[i].bid[0] == target->pairs[i].bid[1] ||
            target->pairs[i].n <= 0) {
            fail_message("native aggregation produced an invalid edge");
            goto cleanup;
        }
        total_count += target->pairs[i].n;
    }
    if (total_count != (int64_t)header.n_raw_pairs) {
        fail_message("native binned integer count is not raw-edge conserving");
        goto cleanup;
    }
    if (inspect) {
        printf("{\"status\":\"inspected\",\"n_beads\":%u,"
               "\"n_raw_pairs\":%u,\"n_binned_pairs\":%u,"
               "\"native_count_sum\":%" PRId64 "}\n",
               header.n_beads, header.n_raw_pairs,
               (uint32_t)target->n_pairs, total_count);
        status = 0;
        goto cleanup;
    }

    {
        struct hk_fdg_conf conf;
        float init_max, final_max;
        hk_fdg_conf_init(&conf);
        conf.n_iter = (int)iterations;
        conf.backend = HK_FDG_BACKEND_CPU;
        fprintf(stderr,
                "fdg_random_bridge: mode=from_scratch source=NULL "
                "target_radius=%.9g iterations=%u seed=%" PRIu64 " backend=CPU\n",
                conf.target_radius, iterations, seed);

        kr_srand_r(&rng, seed);
        native_init = hk_fdg_init(&rng, target->n_beads, conf.target_radius);
        if (native_init == NULL ||
            !validate_coordinates((const fvec3_t *)native_init, header.n_beads)) {
            fail_message("hk_fdg_init returned invalid coordinates");
            goto cleanup;
        }
        init_max = max_norm((const fvec3_t *)native_init, header.n_beads);
        if (!isfinite(init_max) || init_max <= 0.0f) {
            fail_message("native init radius is invalid");
            goto cleanup;
        }

        /* Reset so hk_fdg's internal NULL-source init uses exactly this seed. */
        kr_srand_r(&rng, seed);
        hk_fdg(&conf, target, NULL, &rng);
        if (target->x == NULL ||
            !validate_coordinates((const fvec3_t *)target->x, header.n_beads) ||
            !isfinite(target->unit) || target->unit <= 0.0f) {
            fail_message("native FDG returned invalid coordinates or unit");
            goto cleanup;
        }
        final_max = max_norm((const fvec3_t *)target->x, header.n_beads);
        if (!isfinite(final_max) || final_max <= 0.0f) {
            fail_message("native final radius is invalid");
            goto cleanup;
        }
        if (!write_output(output_path, header.n_beads,
                          (uint32_t)target->n_pairs, header.n_raw_pairs,
                          iterations, target->unit, init_max, final_max,
                          (const fvec3_t *)native_init,
                          (const fvec3_t *)target->x)) {
            fail_message("could not write random native output blob");
            goto cleanup;
        }
    }
    status = 0;

cleanup:
    if (fp != NULL) fclose(fp);
    if (target != NULL) hk_bmap_destroy(target);
    if (dict != NULL) hk_sd_destroy(dict);
    free(lengths);
    free(beads);
    free(bid0);
    free(bid1);
    free(raw);
    free(native_init);
    return status;
}
