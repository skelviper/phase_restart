#include "fdg_gpu.h"

struct hk_fdg_gpu_ctx {
	int dummy;
};

int hk_fdg_gpu_is_available(void)
{
	return 0;
}

struct hk_fdg_gpu_ctx *hk_fdg_gpu_create(int32_t n_beads)
{
	(void)n_beads;
	return 0;
}

void hk_fdg_gpu_destroy(struct hk_fdg_gpu_ctx *ctx)
{
	(void)ctx;
}

int hk_fdg_gpu_prepare(struct hk_fdg_gpu_ctx *ctx, int32_t n_beads, size_t n_repulsion_keys)
{
	(void)ctx;
	(void)n_beads;
	(void)n_repulsion_keys;
	return -1;
}

int hk_fdg_gpu_set_blocklist(struct hk_fdg_gpu_ctx *ctx, const uint64_t *keys, size_t n_keys)
{
	(void)ctx;
	(void)keys;
	(void)n_keys;
	return -1;
}

int hk_fdg_gpu_set_repulsion_weights(struct hk_fdg_gpu_ctx *ctx, const uint64_t *keys,
									 const double *keep_weights, size_t n_keys)
{
	(void)ctx;
	(void)keys;
	(void)keep_weights;
	(void)n_keys;
	return -1;
}

int hk_fdg_gpu_upload_positions(struct hk_fdg_gpu_ctx *ctx, const fvec3_t *pos_host, int32_t n_beads)
{
	(void)ctx;
	(void)pos_host;
	(void)n_beads;
	return -1;
}

int hk_fdg_gpu_download_positions(const struct hk_fdg_gpu_ctx *ctx, fvec3_t *pos_host, int32_t n_beads)
{
	(void)ctx;
	(void)pos_host;
	(void)n_beads;
	return -1;
}

int hk_fdg_gpu_snapshot_best(struct hk_fdg_gpu_ctx *ctx, int32_t n_beads)
{
	(void)ctx;
	(void)n_beads;
	return -1;
}

int hk_fdg_gpu_download_best_positions(const struct hk_fdg_gpu_ctx *ctx, fvec3_t *pos_host, int32_t n_beads)
{
	(void)ctx;
	(void)pos_host;
	(void)n_beads;
	return -1;
}

int hk_fdg_gpu_ensure_capacity(struct hk_fdg_gpu_ctx *ctx, size_t n_pairs)
{
	(void)ctx;
	(void)n_pairs;
	return -1;
}

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
					   int need_sync)
{
	(void)ctx;
	(void)opt;
	(void)pairs;
	(void)n_pairs;
	(void)unit;
	(void)force_unit_mode;
	(void)rel_rep_k;
	(void)rep_radius;
	(void)stats;
	(void)rms_force;
	(void)need_sync;
	return -1;
}

int hk_fdg_gpu_pairs_ready(const struct hk_fdg_gpu_ctx *ctx)
{
	(void)ctx;
	return 0;
}

size_t hk_fdg_gpu_get_active_pairs(const struct hk_fdg_gpu_ctx *ctx)
{
	(void)ctx;
	return 0;
}

void hk_fdg_gpu_set_pair_totals(struct hk_fdg_gpu_ctx *ctx, const uint32_t totals[HK_FDG_PAIR_TYPE_COUNT])
{
	(void)ctx;
	(void)totals;
}

void hk_fdg_gpu_get_pair_totals(const struct hk_fdg_gpu_ctx *ctx, uint32_t totals[HK_FDG_PAIR_TYPE_COUNT])
{
	(void)ctx;
	if (!totals) return;
	for (int i = 0; i < HK_FDG_PAIR_TYPE_COUNT; ++i)
		totals[i] = 0;
}
