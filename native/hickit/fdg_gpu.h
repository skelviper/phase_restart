#ifndef HK_FDG_GPU_H
#define HK_FDG_GPU_H

#include <stdint.h>
#include <stddef.h>
#include "hickit.h"

#define HK_FDG_PAIR_TYPE_BACKBONE 0
#define HK_FDG_PAIR_TYPE_CONTACT  1
#define HK_FDG_PAIR_TYPE_REPEL    2
#define HK_FDG_PAIR_TYPE_COUNT    3

#define HK_FDG_GPU_FORCE_LEGACY   0
#define HK_FDG_GPU_FORCE_PHYSICAL 1

struct hk_fdg_gpu_pair {
	int32_t i;
	int32_t j;
	float k;
	float d_scale;
	uint8_t type;
	uint8_t pad[3];
};

struct hk_fdg_gpu_stats {
	double energy[HK_FDG_PAIR_TYPE_COUNT];
	float dist_sum[HK_FDG_PAIR_TYPE_COUNT];
	uint32_t active[HK_FDG_PAIR_TYPE_COUNT];
	unsigned long long repulsion_considered;
	unsigned long long repulsion_blocked;
	unsigned long long repulsion_partial;
	unsigned long long repulsion_active;
	double repulsion_keep_weight_sum;
	double force_sq_sum;
};

struct hk_fdg_gpu_ctx;

#ifdef __cplusplus
extern "C" {
#endif

int hk_fdg_gpu_is_available(void);
struct hk_fdg_gpu_ctx *hk_fdg_gpu_create(int32_t n_beads);
void hk_fdg_gpu_destroy(struct hk_fdg_gpu_ctx *ctx);
int hk_fdg_gpu_prepare(struct hk_fdg_gpu_ctx *ctx, int32_t n_beads, size_t n_repulsion_keys);
int hk_fdg_gpu_set_blocklist(struct hk_fdg_gpu_ctx *ctx, const uint64_t *keys, size_t n_keys);
int hk_fdg_gpu_set_repulsion_weights(struct hk_fdg_gpu_ctx *ctx, const uint64_t *keys,
									 const double *keep_weights, size_t n_keys);
int hk_fdg_gpu_upload_positions(struct hk_fdg_gpu_ctx *ctx, const fvec3_t *pos_host, int32_t n_beads);
int hk_fdg_gpu_download_positions(const struct hk_fdg_gpu_ctx *ctx, fvec3_t *pos_host, int32_t n_beads);
int hk_fdg_gpu_snapshot_best(struct hk_fdg_gpu_ctx *ctx, int32_t n_beads);
int hk_fdg_gpu_download_best_positions(const struct hk_fdg_gpu_ctx *ctx, fvec3_t *pos_host, int32_t n_beads);
int hk_fdg_gpu_ensure_capacity(struct hk_fdg_gpu_ctx *ctx, size_t n_pairs);
int hk_fdg_gpu_compute(struct hk_fdg_gpu_ctx *ctx,
					   const struct hk_fdg_conf *opt,
					   const struct hk_fdg_gpu_pair *pairs,
					   size_t n_pairs,
					   float unit,
					   int force_unit_mode,
					   float rel_rep_k,
					   float rep_radius,
					   struct hk_fdg_gpu_stats *stats,
					   double *rms_force,
					   int need_sync);
int hk_fdg_gpu_pairs_ready(const struct hk_fdg_gpu_ctx *ctx);
size_t hk_fdg_gpu_get_active_pairs(const struct hk_fdg_gpu_ctx *ctx);
void hk_fdg_gpu_set_pair_totals(struct hk_fdg_gpu_ctx *ctx, const uint32_t totals[HK_FDG_PAIR_TYPE_COUNT]);
void hk_fdg_gpu_get_pair_totals(const struct hk_fdg_gpu_ctx *ctx, uint32_t totals[HK_FDG_PAIR_TYPE_COUNT]);

#ifdef __cplusplus
}
#endif

#endif
