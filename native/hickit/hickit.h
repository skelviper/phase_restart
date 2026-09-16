#ifndef HICKIT_H
#define HICKIT_H

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include "krng.h"

#define HK_SUB_DELIM    '!'
#define HK_MAX_LOOP_RES 8
#define HK_GC_MAX_DEGREE 10

#define HK_DBG_VAL   1

#define HK_DIPLOID_N_COPY 2
#define HK_DIPLOID_COPY0  0
#define HK_DIPLOID_COPY1  1
#define HK_BLIND_N_STATE  4
#define HK_BLIND_STATE_00 0
#define HK_BLIND_STATE_01 1
#define HK_BLIND_STATE_10 2
#define HK_BLIND_STATE_11 3
#define HK_BLIND_STATE_MASK(state) ((uint8_t)(1u << (state)))
#define HK_BLIND_RHO_TRAIN_DEFAULT_FLOOR 0.5f
#define HK_BLIND_REPULSION_MAP_MARGIN_EPS 1e-6f
#define HK_BLIND_N_CONTACT_CLASS 2

enum hk_blind_force_mode {
	HK_BLIND_FORCE_PHYSICAL = 0,
	HK_BLIND_FORCE_LEGACY_CHAINRULE_OMITTED = 1
};

enum hk_blind_repulsion_mode {
	HK_BLIND_REPULSION_NONE = 0,
	HK_BLIND_REPULSION_N2 = 1,
	HK_BLIND_REPULSION_CELL = 2
};

enum hk_blind_repulsion_policy {
	HK_BLIND_REPULSION_POLICY_LEGACY_BINARY_BLOCK = 0,
	HK_BLIND_REPULSION_POLICY_INDEPENDENT = 1,
	HK_BLIND_REPULSION_POLICY_EXPECTED = 2,
	HK_BLIND_REPULSION_POLICY_MAP_ONLY = 3
};

enum hk_blind_softall_filter_mode {
	HK_BLIND_SOFTALL_FILTER_LEGACY_POSTERIOR = 0,
	HK_BLIND_SOFTALL_FILTER_OFF = 1,
	HK_BLIND_SOFTALL_FILTER_RAW_SUPPORT = 2
};

enum hk_blind_rho_train_mode {
	HK_BLIND_RHO_TRAIN_CONSTANT = 0,
	HK_BLIND_RHO_TRAIN_ENTROPY = 1,
	HK_BLIND_RHO_TRAIN_ENTROPY_WITH_FLOOR = 2,
	HK_BLIND_RHO_TRAIN_ENTROPY_CIS_CONSTANT_TRANS = 3,
	HK_BLIND_RHO_TRAIN_ENTROPY_CIS_FLOOR_TRANS = 4,
	HK_BLIND_RHO_TRAIN_TRANS_ENTROPY = 5,
	HK_BLIND_RHO_TRAIN_TRANS_ENTROPY_WITH_FLOOR = 6
};

enum hk_blind_contact_class {
	HK_BLIND_CONTACT_CIS = 0,
	HK_BLIND_CONTACT_TRANS = 1
};

enum hk_blind_d_scale_mode {
	HK_BLIND_D_SCALE_RAW_COUNT = 0,
	HK_BLIND_D_SCALE_POSTERIOR_COUNT = 1,
	HK_BLIND_D_SCALE_EXPECTED_COUNT = HK_BLIND_D_SCALE_POSTERIOR_COUNT,
	HK_BLIND_D_SCALE_DENSITY_NORMALIZED_RAW_COUNT = 2,
	HK_BLIND_D_SCALE_CAPPED_DENSITY_RAW_COUNT = 3,
	HK_BLIND_D_SCALE_TEMPERED_POSTERIOR_COUNT = 4
};

#define HK_BLIND_D_SCALE_DEFAULT_COUNT_CAP 64.0f

enum hk_blind_estep_score_mode {
	HK_BLIND_ESTEP_SCORE_FDG_FLAT = 0,
	HK_BLIND_ESTEP_SCORE_LOGDIST2 = 1,
	HK_BLIND_ESTEP_SCORE_DIST2 = 2,
	HK_BLIND_ESTEP_SCORE_MONOTONIC_POWERLAW = 3,
	HK_BLIND_ESTEP_SCORE_CONTACT_LOG_INTENSITY = HK_BLIND_ESTEP_SCORE_MONOTONIC_POWERLAW
};

enum hk_blind_prior_mode {
	HK_BLIND_PRIOR_UNIFORM = 0,
	HK_BLIND_PRIOR_CIS_INTER_RATIO = 1
};

enum hk_blind_cis_rate_mode {
	HK_BLIND_CIS_RATE_UNIFORM = HK_BLIND_PRIOR_UNIFORM,
	HK_BLIND_CIS_RATE_BLIND_INTER_RATIO = HK_BLIND_PRIOR_CIS_INTER_RATIO,
	HK_BLIND_CIS_RATE_FIXED_ALPHA = 2
};

enum hk_blind_prior_semantics {
	HK_BLIND_PRIOR_SEMANTICS_LEGACY_NORMALIZED = 0,
	HK_BLIND_PRIOR_SEMANTICS_EXPLICIT_RATE = 1
};

enum hk_blind_init_mode {
	HK_BLIND_INIT_TOY_SPLIT = 0,
	HK_BLIND_INIT_UNPHASED_SCAFFOLD_SPLIT = 1,
	HK_BLIND_INIT_RANDOM_DIPLOID = 2,
	HK_BLIND_INIT_RANDOM_HAPLOID_SPLIT = 3,
	HK_BLIND_INIT_FIXED_SOURCE_COORDS = 4
};

enum hk_blind_relax_step_policy {
	HK_BLIND_RELAX_STEP_FIXED = 0,
	HK_BLIND_RELAX_STEP_ENDPOINT_BACKTRACKING = 1
};

enum hk_blind_base_k_mode {
	HK_BLIND_BASE_K_UNIFORM = 0,
	HK_BLIND_BASE_K_NEIGHBOR_MEDIAN = 1
};

enum hk_blind_readgroup_mode {
	HK_BLIND_READGROUP_OFF = 0,
	HK_BLIND_READGROUP_JOINT_MARGINAL = 1
};

enum hk_blind_trans_top1_mode {
	HK_BLIND_TRANS_TOP1_OFF = 0,
	HK_BLIND_TRANS_TOP1_HARD = 1,
	HK_BLIND_TRANS_TOP1_MIX = 2
};

enum hk_blind_trans_gate_mode {
	HK_BLIND_TRANS_GATE_OFF = 0,
	HK_BLIND_TRANS_GATE_PMAX = 1,
	HK_BLIND_TRANS_GATE_MARGIN = 2,
	HK_BLIND_TRANS_GATE_NEG_ENTROPY = 3,
	HK_BLIND_TRANS_GATE_PMAX_MARGIN = 4
};

enum hk_blind_trans_chr_pair_mstep_mode {
	HK_BLIND_TRANS_CHR_PAIR_MSTEP_STATE4 = 0,
	HK_BLIND_TRANS_CHR_PAIR_MSTEP_SAME_CROSS = 1
};

enum hk_blind_trans_callable_anchor_mode {
	HK_BLIND_TRANS_CALLABLE_ANCHOR_OFF = 0,
	HK_BLIND_TRANS_CALLABLE_ANCHOR_NEG_ENTROPY = 1,
	HK_BLIND_TRANS_CALLABLE_ANCHOR_PMAX = 2,
	HK_BLIND_TRANS_CALLABLE_ANCHOR_MARGIN = 3
};

enum hk_blind_phase_lock_mode {
	HK_BLIND_PHASE_LOCK_NONE = 0,
	HK_BLIND_PHASE_LOCK_FIXED_P4 = 1
};

// Copy must be 0 or 1. Assertions catch invalid use in debug/test builds.
static inline int32_t hk_diploid_bid(int32_t haploid_bid, int32_t copy)
{
	assert(haploid_bid >= 0);
	assert(haploid_bid <= INT32_MAX / HK_DIPLOID_N_COPY);
	assert(copy == HK_DIPLOID_COPY0 || copy == HK_DIPLOID_COPY1);
	return HK_DIPLOID_N_COPY * haploid_bid + copy;
}

static inline int32_t hk_diploid_haploid_bid(int32_t diploid_bid)
{
	assert(diploid_bid >= 0);
	return diploid_bid / HK_DIPLOID_N_COPY;
}

static inline int32_t hk_diploid_copy(int32_t diploid_bid)
{
	assert(diploid_bid >= 0);
	return diploid_bid % HK_DIPLOID_N_COPY;
}

#ifdef __cplusplus
extern "C" {
#endif

struct hk_popt {
	int min_dist, max_seg, min_mapq, min_flt_cnt;
	int min_tad_size;
	float area_weight;
	int min_radius, max_radius, max_nei;
	float pseudo_cnt;
	int n_iter;
};

struct hk_sdict {     // sequence dictionary
	int32_t n, m;
	char **name;
	int32_t *len;
	void *h;
};

struct hk_seg {       // a segment
	int32_t frag_id, chr, st, en;
	int8_t strand, phase;
	int16_t mapq;
};

struct hk_pair {      // a contact pair
	uint64_t chr;     // chr1<<32 | chr2
	uint64_t pos;     // pos1<<32 | pos2
	uint64_t read_name_hash;
	int8_t strand[2]; // strand
	int8_t phase[2];  // phase
	int32_t read_group_id;
	int16_t read_seg[2];
	int16_t read_n_segments;
	uint32_t n_ctn:31, tad_marked:1;
	uint32_t n_nei, n_nei_corner;
	union {
		float p4[4];
		float phased_prob;
		int32_t n_nei[2][2];
		struct { float q; int32_t r, n; } loop;
		struct { double e; int32_t n[2]; } ecnt;
	} _;
};

struct hk_blind_pair {
	int32_t chr[2];
	int32_t pos[2];
	uint64_t read_name_hash;
	int8_t strand[2];
	int32_t read_group_id;
	int16_t read_seg[2];
	int16_t read_n_segments;
};

struct hk_blind_bpair_key {
	int32_t bid[2]; // Canonical haploid bead ids: bid[0] <= bid[1].
};

struct hk_blind_raw2binned {
	int32_t bpair_id; // The array index is the raw contact id.
	uint8_t swapped;  // Raw endpoint order was reversed relative to the canonical key.
};

struct hk_blind_readgroup_entry {
	int32_t group_id;
	int32_t raw_id;
};

struct hk_blind_bpair {
	struct hk_blind_bpair_key key;
	int32_t n_raw;
	int32_t n_phase_locked_raw;
	float base_d_scale;
	float base_k;
	int8_t contact_class;
	int8_t phase_lock_mode;
	int8_t phase_locked_state;
	uint8_t phase_lock_conflict;
	int32_t density_child_count[2];
	float density_exposure;
	float log_prior[HK_BLIND_N_STATE];
	float p4[HK_BLIND_N_STATE];
	float entropy;
	float pmax;
	float margin;
	float rho_output;
	float pU;
	float real_log_norm;
	float log_rate_weight[HK_BLIND_N_STATE]; // Unnormalized intensity weights applied outside temperature.
};

struct hk_blind_phase_lock_diag {
	int enabled;
	int32_t sample_size_percent;
	uint64_t seed;
	int64_t n_raw_total;
	int64_t n_sampled_raw;
	int64_t n_full_phase_raw;
	int64_t n_partial_phase_raw;
	int64_t n_unphased_raw;
	int64_t n_sampled_full_phase_raw;
	int64_t n_sampled_partial_phase_raw;
	int64_t n_sampled_unphased_raw;
	int64_t n_locked_raw;
	int64_t n_conflict_raw;
	int64_t n_same_bin_locked_raw;
	int32_t n_locked_bpair;
	int32_t n_conflict_bpair;
	int32_t n_same_bin_locked_bpair;
	int32_t state_bpair_count[HK_BLIND_N_STATE];
	int64_t state_raw_count[HK_BLIND_N_STATE];
};

struct hk_blind_bpair_set {
	struct hk_blind_bpair *bpairs;
	int32_t n_bpairs;
	struct hk_blind_raw2binned *raw2binned;
	struct hk_blind_pair *raw;
	struct hk_blind_readgroup_entry *readgroup_entries;
	float *raw_p4;
	int8_t *raw_locked_state;
	int32_t n_raw;
	int32_t n_readgroup_entries;
	int32_t n_readgroup_groups;
	int32_t n_readgroup_groups_used;
	int64_t n_readgroup_raw_decoded;
	int64_t n_readgroup_groups_skipped_too_large;
	int64_t n_readgroup_groups_skipped_bad;
	int same_bin_filter_enabled;
	int64_t n_raw_same_bin_excluded;
	int32_t n_bpair_same_bin_excluded;
	int64_t n_raw_cis;
	int64_t n_raw_trans;
	int32_t n_bpair_cis;
	int32_t n_bpair_trans;
	int prior_semantics;
	struct hk_blind_phase_lock_diag phase_lock_diag;
};

struct hk_blind_wedge {
	int32_t bid[2]; // Diploid bead ids.
	float k;
	float d_scale;
	float repulsion_block_prob; // Posterior occupancy q before rho/contact-strength scaling.
	double repulsion_keep_weight; // Authoritative product-space keep weight for expected repulsion.
	int8_t state;     // Representative state after sorting; use state_mask for full provenance.
	uint8_t state_mask; // Bitwise OR of contributing states: 1u << state.
	uint8_t repulsion_map; // At least one contribution has this pair as its unique posterior MAP state.
};

struct hk_blind_softall_graph_diag {
	int64_t candidate_raw_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t candidate_state_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t filter1_raw_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t filter1_state_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t filter2_raw_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t filter2_state_rows[HK_BLIND_N_CONTACT_CLASS];
	int64_t binned_edges[HK_BLIND_N_CONTACT_CLASS];
	double candidate_posterior_mass[HK_BLIND_N_CONTACT_CLASS];
	double filter1_posterior_mass[HK_BLIND_N_CONTACT_CLASS];
	double filter2_posterior_mass[HK_BLIND_N_CONTACT_CLASS];
};

struct hk_blind_wedge_list {
	struct hk_blind_wedge *edges;
	int32_t n_edges, m_edges;
	int64_t n_input_bpair;
	int64_t n_expanded_edges;
	int64_t n_softall_candidate_pairs;
	int64_t n_softall_filter1_pairs;
	int64_t n_softall_filter2_pairs;
	int64_t n_softall_bmap_pairs;
	int64_t n_softall_selected_raw;
	int64_t n_softall_gate_skip_raw;
	int64_t n_softall_same_bin_skip_raw;
	int64_t n_softall_state_raw_count[HK_BLIND_N_STATE];
	int64_t n_skipped_self_edges;
	int64_t n_skipped_same_bin_bpairs;
	int64_t n_edges_before_aggregation;
	int64_t n_aggregated_edges_removed;
	double sum_rho_train_bpair;
	float mean_rho_train_bpair;
	float min_rho_train_bpair;
	float max_rho_train_bpair;
	struct hk_blind_softall_graph_diag softall_graph_diag;
};

struct hk_blind_base_k_stats {
	int32_t n;
	int32_t n_nonfinite;
	float min;
	float max;
	float mean;
};

struct hk_blind_heldout_diag {
	int enabled;
	float fraction;
	uint64_t seed;
	int64_t n_input_raw_total;
	int32_t n_raw_train;
	int32_t n_raw_heldout;
	int32_t n_bpair_heldout;
	int32_t n_bpair_eval;
	int64_t n_raw_eval;
	int32_t n_bpair_same_bin_skipped;
	int64_t n_raw_same_bin_skipped;
	double diagnostic_expected_energy;
	double mean_min_energy;
	double mean_entropy;
	double mean_pU;
	double mean_best_normalized_distance;
	double short_distance_frac;
};

struct hk_blind_prior_diag {
	int prior_mode;
	int32_t n_distance_bins;
	int32_t n_alpha;
	double observed_inter;
	double possible_inter;
	double inter_density;
	float alpha_min;
	float alpha_median;
	float alpha_max;
	float eps_prior;
};

struct hk_blind_homolog_sep_stats {
	int32_t n_haploid;
	int32_t n_finite;
	int32_t n_nonfinite;
	int32_t n_collapsed;
	float min_sep;
	float max_sep;
	float mean_sep;
	float collapse_threshold;
};

struct hk_blind_gauge_stats {
	int32_t n_chr;
	int32_t n_flipped;
	double mismatch_no_flip;
	double mismatch_flip;
	double mismatch_chosen;
};

struct hk_blind_iter_diag {
	int32_t n_bpair;
	int32_t n_raw;
	float mean_entropy;
	float mean_pmax;
	float mean_margin;
	float mean_pU;
	float mean_rho_output;
	int32_t n_posterior_nonfinite;
	int32_t n_posterior_bad_sum;
	int32_t n_posterior_out_of_range;
	int32_t n_uncertainty_nonfinite;
	int32_t n_uncertainty_out_of_range;
	int32_t n_five_state_bad_sum;
	struct hk_blind_homolog_sep_stats sep_stats;
	float sep_energy;
	float sep_force_l1;
	int32_t sep_force_nonfinite;
	int32_t n_wedges;
	int32_t n_wedges_before_aggregation;
	int64_t n_expanded_edges;
	int64_t n_softall_candidate_pairs;
	int64_t n_softall_filter1_pairs;
	int64_t n_softall_filter2_pairs;
	int64_t n_softall_bmap_pairs;
	int64_t n_softall_selected_raw;
	int64_t n_softall_gate_skip_raw;
	int64_t n_softall_same_bin_skip_raw;
	int64_t n_softall_state_raw_count[HK_BLIND_N_STATE];
	struct hk_blind_softall_graph_diag softall_graph_diag;
	int64_t n_skipped_self_edges;
	int64_t n_skipped_same_bin_bpairs;
	double sum_wedge_k;
	float mean_rho_train_bpair;
	float min_rho_train_bpair;
	float max_rho_train_bpair;
	int32_t n_wedge_nonfinite;
	int32_t n_wedge_bad_k;
	int32_t n_wedge_bad_d_scale;
	struct hk_blind_gauge_stats gauge_stats;
	int32_t n_chr_flipped;
};

struct hk_blind_step_diag {
	float contact_energy;
	float backbone_energy;
	double repulsion_energy;
	float bending_energy;
	float confinement_energy;
	float sep_energy;
	float copytrack_energy;
	float global_copytrack_energy;
	float normdir_copytrack_energy;
	float anchor_energy;
	float total_energy;
	float force_l1;
	float backbone_force_l1;
	float repulsion_force_l1;
	float bending_force_l1;
	float confinement_force_l1;
	float sep_force_l1;
	float copytrack_force_l1;
	float global_copytrack_force_l1;
	float normdir_copytrack_force_l1;
	float anchor_force_l1;
	int32_t n_force_nonfinite;
	int32_t n_contact_nonfinite;
	int32_t n_backbone_nonfinite;
	int32_t n_repulsion_nonfinite;
	int32_t n_bending_nonfinite;
	int32_t n_confinement_nonfinite;
	int32_t n_sep_nonfinite;
	int32_t n_copytrack_nonfinite;
	int32_t n_global_copytrack_nonfinite;
	int32_t n_normdir_copytrack_nonfinite;
	int32_t n_anchor_nonfinite;
	int64_t n_backbone_edges;
	int64_t n_bending_triplets;
	int64_t n_confinement_active;
	int64_t n_repulsion_pairs_considered;
	int64_t n_repulsion_pairs_blocked;
	int64_t n_repulsion_pairs_partial;
	int64_t n_repulsion_pairs_active;
	double repulsion_keep_weight_sum;
	int32_t repulsion_mode;
};

struct hk_blind_backbone_diag {
	int64_t n_edges;
	int32_t n_nonfinite;
	float energy;
	float force_l1;
};

struct hk_blind_repulsion_diag {
	int64_t n_pairs_considered;
	int64_t n_pairs_blocked;
	int64_t n_pairs_partial;
	int64_t n_pairs_active;
	double keep_weight_sum;
	int32_t n_nonfinite;
	float energy;
	float force_l1;
};

struct hk_blind_contact_class_diag {
	int64_t n_wedges_cis;
	int64_t n_wedges_trans;
	int32_t n_nonfinite;
	double sum_wedge_k_cis;
	double sum_wedge_k_trans;
	float contact_energy_cis;
	float contact_energy_trans;
	float contact_force_l1_cis;
	float contact_force_l1_trans;
};

struct hk_blind_coarse_to_fine_map {
	int32_t n_coarse;
	int32_t n_fine;
	int32_t *fine_to_coarse;
	int32_t *child_rank;
	int32_t *parent_first_fine;
	int32_t *parent_child_count;
};

// Scheduled energy fields preserve the relaxation schedule provenance. Comparable
// fields sample the pre-update and post-update coordinates at one fixed objective:
// the final scheduled repulsion scale. For n_steps == 0, summaries remain zero.
struct hk_blind_relax_diag {
	int32_t n_steps;
	int32_t n_completed;
	float initial_total_energy;
	float final_total_energy;
	float initial_contact_energy;
	float final_contact_energy;
	float initial_backbone_energy;
	float final_backbone_energy;
	double initial_repulsion_energy;
	double final_repulsion_energy;
	float initial_bending_energy;
	float final_bending_energy;
	float initial_confinement_energy;
	float final_confinement_energy;
	float initial_sep_energy;
	float final_sep_energy;
	float initial_copytrack_energy;
	float final_copytrack_energy;
	float initial_global_copytrack_energy;
	float final_global_copytrack_energy;
	float initial_normdir_copytrack_energy;
	float final_normdir_copytrack_energy;
	float initial_anchor_energy;
	float final_anchor_energy;
	float max_force_l1;
	float final_force_l1;
	float max_backbone_force_l1;
	float final_backbone_force_l1;
	float max_repulsion_force_l1;
	float final_repulsion_force_l1;
	float max_bending_force_l1;
	float final_bending_force_l1;
	float max_confinement_force_l1;
	float final_confinement_force_l1;
	float max_sep_force_l1;
	float final_sep_force_l1;
	float max_copytrack_force_l1;
	float final_copytrack_force_l1;
	float max_global_copytrack_force_l1;
	float final_global_copytrack_force_l1;
	float max_normdir_copytrack_force_l1;
	float final_normdir_copytrack_force_l1;
	float max_anchor_force_l1;
	float final_anchor_force_l1;
	int64_t final_n_repulsion_pairs_considered;
	int64_t final_n_repulsion_pairs_blocked;
	int64_t final_n_repulsion_pairs_partial;
	int64_t final_n_repulsion_pairs_active;
	double final_repulsion_keep_weight_sum;
	int32_t n_nonfinite_step;
	int32_t n_backbone_nonfinite_step;
	int32_t n_repulsion_nonfinite_step;
	int32_t n_bending_nonfinite_step;
	int32_t n_confinement_nonfinite_step;
	int32_t n_sep_nonfinite_step;
	int32_t n_copytrack_nonfinite_step;
	int32_t n_global_copytrack_nonfinite_step;
	int32_t n_normdir_copytrack_nonfinite_step;
	int32_t n_anchor_nonfinite_step;
	int32_t n_coord_nonfinite;
	int32_t repulsion_mode;
	int force_mode;
	float scheduled_initial_repulsion_scale;
	float scheduled_final_repulsion_scale;
	float comparable_repulsion_scale;
	float comparable_initial_total_energy;
	float comparable_final_total_energy;
	float comparable_initial_contact_energy;
	float comparable_final_contact_energy;
	float comparable_initial_backbone_energy;
	float comparable_final_backbone_energy;
	double comparable_initial_repulsion_energy;
	double comparable_final_repulsion_energy;
	float comparable_initial_bending_energy;
	float comparable_final_bending_energy;
	float comparable_initial_confinement_energy;
	float comparable_final_confinement_energy;
	float comparable_initial_sep_energy;
	float comparable_final_sep_energy;
	float comparable_initial_copytrack_energy;
	float comparable_final_copytrack_energy;
	float comparable_initial_global_copytrack_energy;
	float comparable_final_global_copytrack_energy;
	float comparable_initial_normdir_copytrack_energy;
	float comparable_final_normdir_copytrack_energy;
	float comparable_initial_anchor_energy;
	float comparable_final_anchor_energy;
	int32_t comparable_energy_valid;
	int relax_step_policy;
	float requested_relax_step;
	float accepted_relax_step;
	int32_t relax_attempts;
	int32_t relax_rejected_attempts;
	int32_t relax_rollback_count;
	int32_t relax_nonfinite_rejected_attempts;
	int32_t endpoint_acceptance_evaluated;
	int32_t relax_accepted;
	double endpoint_pre_energy;
	double endpoint_post_energy;
	double endpoint_acceptance_threshold;
};

struct hk_blind_single_iter_conf {
	float unit;
	float d_scale;
	float base_k;
	float temperature;
	float rho_train;
	float rho_train_floor;
	float min_sep_unit;
	float lambda_sep;
	float lambda_copytrack;
	float lambda_global_copytrack;
	float lambda_normdir_copytrack;
	float normdir_copytrack_eps_unit;
	float trans_chr_pair_prior_lambda;
	float trans_chr_pair_prior_eps;
	float trans_chr_pair_prior_power;
	int32_t trans_chr_pair_prior_warmup_iter;
	float trans_chr_pair_mstep_lambda;
	float trans_chr_pair_mstep_eps;
	float trans_chr_pair_mstep_power;
	int32_t trans_chr_pair_mstep_warmup_iter;
	int trans_chr_pair_mstep_mode;
	float chr_sep_unit;
	float lambda_chr_sep;
	float relax_step;
	int32_t relax_steps;
	int force_mode;
	int relax_step_policy;
	float relax_backtrack_factor;
	int32_t relax_backtrack_max_retries;
	float relax_backtrack_min_step;
	double relax_backtrack_rtol;
	double relax_backtrack_atol;
	int enable_repulsion;
	int repulsion_mode;
	int repulsion_policy;
	float repulsion_block_k_min;
	int softall_filter_mode;
	int rho_train_mode;
		int d_scale_mode;
		float d_scale_eps_count;
		float d_scale_posterior_gamma;
		float trans_d_scale_posterior_gamma;
		float trans_k_multiplier;
		float trans_dscale_multiplier;
	int trans_top1_mode;
	float trans_top1_min_pmax;
	float trans_top1_min_margin;
	float trans_top1_mix_weight;
	int trans_callable_anchor_mode;
	float trans_callable_anchor_top_frac;
	float trans_callable_anchor_mix_weight;
	int32_t trans_callable_anchor_min_n_raw;
	int32_t trans_callable_anchor_warmup_iter;
	int trans_gate_mode;
	float trans_gate_min_pmax;
	float trans_gate_min_margin;
	float trans_gate_min_neg_entropy;
	int readgroup_mode;
	int32_t readgroup_max_segments;
	float readgroup_eps;
	int estep_score_mode;
};

struct hk_blind_single_iter_diag {
	struct hk_blind_iter_diag pre_relax_diag;
	struct hk_blind_relax_diag relax_diag;
	struct hk_blind_gauge_stats gauge_stats;
	int32_t n_chr_flipped;
	int32_t n_wedges;
	int32_t n_wedges_before_aggregation;
	int64_t n_expanded_edges;
	int64_t n_softall_candidate_pairs;
	int64_t n_softall_filter1_pairs;
	int64_t n_softall_filter2_pairs;
	int64_t n_softall_bmap_pairs;
	int64_t n_softall_selected_raw;
	int64_t n_softall_gate_skip_raw;
	int64_t n_softall_same_bin_skip_raw;
	int64_t n_softall_state_raw_count[HK_BLIND_N_STATE];
	struct hk_blind_softall_graph_diag softall_graph_diag;
	int64_t n_skipped_self_edges;
	int64_t n_skipped_same_bin_bpairs;
	double sum_wedge_k;
	float mean_rho_train_bpair;
	float min_rho_train_bpair;
	float max_rho_train_bpair;
};

struct hk_blind_iter_loop_conf {
	int32_t n_iter;
	struct hk_blind_single_iter_conf single_iter_conf;
};

struct hk_blind_iter_schedule_conf {
	int32_t n_iter;
	struct hk_blind_single_iter_conf base_conf;
	float temperature_start;
	float temperature_end;
	float rho_train_start;
	float rho_train_end;
};

struct hk_blind_iter_loop_diag {
	int32_t n_iter;
	int32_t n_completed;
	float initial_temperature;
	float final_temperature;
	float initial_rho_train;
	float final_rho_train;
	float initial_mean_entropy;
	float final_mean_entropy;
	float initial_mean_pU;
	float final_mean_pU;
	int32_t total_chr_flipped;
	int32_t n_bad_iter;
	int32_t n_relax_nonfinite_iter;
	int32_t n_coord_nonfinite;
	float final_mean_sep;
	float final_min_sep;
	float final_max_sep;
	double final_sum_wedge_k;
	float final_mean_rho_train_bpair;
	float final_min_rho_train_bpair;
	float final_max_rho_train_bpair;
	int64_t final_n_expanded_edges;
	int64_t final_n_softall_candidate_pairs;
	int64_t final_n_softall_filter1_pairs;
	int64_t final_n_softall_filter2_pairs;
	int64_t final_n_softall_bmap_pairs;
	int64_t final_n_softall_selected_raw;
	int64_t final_n_softall_gate_skip_raw;
	int64_t final_n_softall_same_bin_skip_raw;
	int64_t final_n_softall_state_raw_count[HK_BLIND_N_STATE];
	struct hk_blind_softall_graph_diag final_softall_graph_diag;
	int64_t final_n_skipped_same_bin_bpairs;
	double final_repulsion_energy;
	float final_repulsion_force_l1;
	int64_t final_n_repulsion_pairs_considered;
	int64_t final_n_repulsion_pairs_blocked;
	int64_t final_n_repulsion_pairs_partial;
	int64_t final_n_repulsion_pairs_active;
	double final_repulsion_keep_weight_sum;
	float final_anchor_energy;
	float final_anchor_force_l1;
	float final_copytrack_energy;
	float final_copytrack_force_l1;
	float final_global_copytrack_energy;
	float final_global_copytrack_force_l1;
	float final_normdir_copytrack_energy;
	float final_normdir_copytrack_force_l1;
	int32_t n_repulsion_nonfinite_step;
	int32_t n_copytrack_nonfinite_step;
	int32_t n_global_copytrack_nonfinite_step;
	int32_t n_normdir_copytrack_nonfinite_step;
	int32_t n_anchor_nonfinite_step;
	int32_t repulsion_mode;
	int posterior_refreshed_after_final_relax;
	float posterior_refresh_temperature;
	double posterior_refresh_mean_kl;
	float posterior_refresh_top_state_switch_frac;
	float posterior_refresh_mean_pU_before;
	float posterior_refresh_mean_pU_after;
	struct hk_blind_heldout_diag heldout_diag;
};

static inline void hk_blind_pair_from_pair(struct hk_blind_pair *dst, const struct hk_pair *src)
{
	assert(dst);
	assert(src);
	dst->chr[0] = (int32_t)(src->chr >> 32);
	dst->chr[1] = (int32_t)src->chr;
	dst->pos[0] = (int32_t)(src->pos >> 32);
	dst->pos[1] = (int32_t)src->pos;
	dst->read_name_hash = src->read_name_hash;
	dst->strand[0] = src->strand[0];
	dst->strand[1] = src->strand[1];
	dst->read_group_id = src->read_group_id;
	dst->read_seg[0] = src->read_seg[0];
	dst->read_seg[1] = src->read_seg[1];
	dst->read_n_segments = src->read_n_segments;
}

struct hk_map {
	uint32_t cols;
	struct hk_sdict *d;
	int32_t n_frags, n_segs, n_pairs;
	struct hk_seg  *segs;
	struct hk_pair *pairs;
};

struct hk_bpair {
	int32_t bid[2];
	int32_t n, max_nei;
	int8_t phase[2];
	float p;
	float gc_exp;      // expected multiplicative bias from CpG fit
	float gc_norm_n;   // contact count after GC normalization
};

struct hk_bead {
	int32_t chr, st, en;
};

typedef float fvec3_t[3];

struct hk_bmap {
	int32_t n_beads, n_pairs;
	float unit;
	struct hk_sdict *d;
	struct hk_bead *beads;
	uint64_t *offcnt; // index into beads
	struct hk_bpair *pairs;
	fvec3_t *x;
	float *feat;
	float *cpg;        // CpG density per bead (optional)
	float *gc_bias;    // per-bead GC bias normalized to mean 1
	int gc_corrected;  // GC correction applied
};

struct hk_fdg_conf;

int hk_bmap_pos2bid(const struct hk_bmap *m, int32_t chr, int32_t pos);

static inline void hk_blind_pair_to_bids(const struct hk_bmap *m, const struct hk_blind_pair *p, int32_t bid[2])
{
	assert(m && m->d);
	assert(p);
	assert(bid);
	assert(p->chr[0] >= 0 && p->chr[0] < m->d->n);
	assert(p->chr[1] >= 0 && p->chr[1] < m->d->n);
	assert(p->pos[0] >= 0);
	assert(p->pos[1] >= 0);
	bid[0] = hk_bmap_pos2bid(m, p->chr[0], p->pos[0]);
	bid[1] = hk_bmap_pos2bid(m, p->chr[1], p->pos[1]);
}

static inline void hk_blind_bpair_key_from_bids(struct hk_blind_bpair_key *key, int32_t bid0, int32_t bid1, uint8_t *swapped)
{
	assert(key);
	assert(bid0 >= 0);
	assert(bid1 >= 0);
	if (bid1 < bid0) {
		key->bid[0] = bid1;
		key->bid[1] = bid0;
		if (swapped) *swapped = 1;
	} else {
		key->bid[0] = bid0;
		key->bid[1] = bid1;
		if (swapped) *swapped = 0;
	}
}

static inline void hk_blind_pair_to_bpair_key(const struct hk_bmap *m, const struct hk_blind_pair *p, struct hk_blind_bpair_key *key, uint8_t *swapped)
{
	int32_t bid[2];
	hk_blind_pair_to_bids(m, p, bid);
	hk_blind_bpair_key_from_bids(key, bid[0], bid[1], swapped);
}

static inline void hk_blind_raw2binned_set(struct hk_blind_raw2binned *dst, int32_t bpair_id, uint8_t swapped)
{
	assert(dst);
	assert(bpair_id >= 0);
	dst->bpair_id = bpair_id;
	dst->swapped = swapped != 0;
}

static inline int hk_blind_bpair_key_is_same_bin(const struct hk_blind_bpair_key *key)
{
	assert(key);
	return key->bid[0] == key->bid[1];
}

static inline int hk_blind_bpair_is_same_bin(const struct hk_blind_bpair *bp)
{
	assert(bp);
	return hk_blind_bpair_key_is_same_bin(&bp->key);
}

const char *hk_blind_init_mode_name(int mode);
const char *hk_blind_relax_step_policy_name(int policy);
const char *hk_blind_cis_rate_mode_name(int mode);
const char *hk_blind_prior_mode_name(int mode);
const char *hk_blind_prior_semantics_name(int semantics);
const char *hk_blind_repulsion_policy_name(int policy);
const char *hk_blind_rho_train_mode_name(int mode);
const char *hk_blind_d_scale_mode_name(int mode);
const char *hk_blind_d_scale_effective_count_formula(int mode, float gamma);
const char *hk_blind_estep_score_mode_name(int mode);
const char *hk_blind_base_k_mode_name(int mode);
const char *hk_blind_contact_class_name(int contact_class);
const char *hk_blind_softall_filter_mode_name(int mode);
const char *hk_blind_readgroup_mode_name(int mode);
const char *hk_blind_trans_top1_mode_name(int mode);
const char *hk_blind_trans_gate_mode_name(int mode);
const char *hk_blind_trans_chr_pair_mstep_mode_name(int mode);
const char *hk_blind_trans_callable_anchor_mode_name(int mode);
int hk_blind_init_mode_valid(int mode);
int hk_blind_cis_rate_mode_valid(int mode);
int hk_blind_prior_mode_valid(int mode);
int hk_blind_prior_semantics_valid(int semantics);
int hk_blind_repulsion_policy_valid(int policy);
int hk_blind_rho_train_mode_valid(int mode);
int hk_blind_d_scale_mode_valid(int mode);
int hk_blind_estep_score_mode_valid(int mode);
int hk_blind_base_k_mode_valid(int mode);
int hk_blind_contact_class_valid(int contact_class);
int hk_blind_softall_filter_mode_valid(int mode);
int hk_blind_readgroup_mode_valid(int mode);
int hk_blind_trans_chr_pair_mstep_mode_valid(int mode);
int hk_blind_relax_step_policy_valid(int policy);
int hk_blind_trans_callable_anchor_mode_valid(int mode);
struct hk_blind_bpair_set *hk_blind_bpair_set_build(const struct hk_bmap *m, int32_t n_raw, const struct hk_blind_pair *raw);
void hk_blind_bpair_set_destroy(struct hk_blind_bpair_set *set);
int hk_blind_bpair_set_build_readgroup_index(struct hk_blind_bpair_set *set);
int hk_blind_bpair_set_apply_readgroup_marginals(struct hk_blind_bpair_set *set,
												 int32_t max_segments,
												 float eps);
int hk_blind_bpair_set_aggregate_raw_p4_to_bpair(struct hk_blind_bpair_set *set);
void hk_blind_init_uniform_log_prior(float log_prior[HK_BLIND_N_STATE]);
void hk_blind_bpair_set_init_uniform_prior(struct hk_blind_bpair_set *set);
void hk_blind_bpair_set_init_uniform_rate_score(struct hk_blind_bpair_set *set);
int hk_blind_bpair_set_init_cis_fixed_alpha_prior(struct hk_blind_bpair_set *set,
														 float alpha, float eps_prior,
														 struct hk_blind_prior_diag *diag);
int hk_blind_bpair_set_apply_trans_chr_pair_prior(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
												  float lambda, float eps_count, float power);
int hk_blind_bpair_set_init_cis_inter_ratio_prior(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
												  float eps_prior, struct hk_blind_prior_diag *diag);
int hk_blind_bpair_set_apply_base_k_mode(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
										 int base_k_mode);
int hk_blind_bpair_set_apply_neighbor_median_base_k(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set);
void hk_blind_bpair_set_base_k_stats(const struct hk_blind_bpair_set *set, struct hk_blind_base_k_stats *stats);
void hk_blind_heldout_diag_init(struct hk_blind_heldout_diag *diag);
int hk_blind_eval_heldout_bpair_set(const struct hk_bmap *bmap, const struct hk_blind_bpair_set *set,
									const struct hk_fdg_conf *conf, const fvec3_t *coords,
									float unit, float temperature,
									struct hk_blind_heldout_diag *diag);
void hk_blind_posterior_from_energy(const float energy[HK_BLIND_N_STATE], const float log_prior[HK_BLIND_N_STATE], float temperature,
									float p4[HK_BLIND_N_STATE], float *entropy, float *pmax, float *margin, float *rho_output, float *pU);
double hk_blind_contact_kernel_sum_from_distances(const struct hk_fdg_conf *conf,
														 const float distance[HK_BLIND_N_STATE],
														 float unit, float d_scale, float k,
														 float temperature, int estep_score_mode,
														 int contact_class, float cis_alpha);
void hk_blind_bpair_posterior_from_coords(const struct hk_fdg_conf *conf, int32_t bid0, int32_t bid1, const fvec3_t *coords,
										  float unit, float d_scale, float k, const float log_prior[HK_BLIND_N_STATE], float temperature,
										  float energy_out[HK_BLIND_N_STATE], float p4[HK_BLIND_N_STATE],
										  float *entropy, float *pmax, float *margin, float *rho_output, float *pU);
void hk_blind_bpair_posterior_from_coords_score_mode(const struct hk_fdg_conf *conf, int32_t bid0, int32_t bid1,
													 const fvec3_t *coords, float unit, float d_scale, float k,
													 const float log_prior[HK_BLIND_N_STATE], float temperature,
													 int estep_score_mode, float energy_out[HK_BLIND_N_STATE],
													 float p4[HK_BLIND_N_STATE], float *entropy, float *pmax,
													 float *margin, float *rho_output, float *pU);
void hk_blind_bpair_set_update_posterior_from_coords(struct hk_blind_bpair_set *set, const struct hk_fdg_conf *conf,
													const fvec3_t *coords, float unit, float d_scale, float k,
													const float log_prior[HK_BLIND_N_STATE], float temperature);
void hk_blind_bpair_set_update_posterior_from_coords_params(struct hk_blind_bpair_set *set, const struct hk_fdg_conf *conf,
															const fvec3_t *coords, float unit,
															const float log_prior[HK_BLIND_N_STATE], float temperature);
void hk_blind_bpair_set_update_posterior_from_coords_params_score_mode(struct hk_blind_bpair_set *set,
																	   const struct hk_fdg_conf *conf,
																	   const fvec3_t *coords, float unit,
																	   const float log_prior[HK_BLIND_N_STATE],
																	   float temperature, int estep_score_mode);
// eps and noise_scale are raw coordinate units; later callers should derive them from bd_unit.
int hk_blind_init_diploid_coords_from_haploid(const struct hk_bmap *bmap, const fvec3_t *haploid_x, int32_t n_haploid,
											 fvec3_t *diploid_x, float eps, float noise_scale, uint64_t seed);
int hk_blind_init_toy_haploid_scaffold(const struct hk_bmap *bmap, fvec3_t *haploid_x);
int hk_blind_init_random_haploid_scaffold(const struct hk_bmap *bmap, fvec3_t *haploid_x, float scale, uint64_t seed);
int hk_blind_init_random_diploid_coords(const struct hk_bmap *bmap, fvec3_t *diploid_x, float scale, uint64_t seed);
int hk_blind_init_haploid_scaffold_from_bmap_fdg(struct hk_bmap *bmap, const struct hk_fdg_conf *fdg_conf,
												 fvec3_t *haploid_x, uint64_t seed);
float hk_blind_homolog_sep_energy(const fvec3_t x0, const fvec3_t x1, float unit, float min_sep_unit, float lambda_sep);
float hk_blind_homolog_sep_energy_force(const fvec3_t x0, const fvec3_t x1, float unit, float min_sep_unit, float lambda_sep,
										 fvec3_t f0, fvec3_t f1);
float hk_blind_homolog_sep_accumulate_force(int32_t n_haploid, const fvec3_t *coords, fvec3_t *force,
											float unit, float min_sep_unit, float lambda_sep);
float hk_blind_homolog_sep_accumulate_force_ex(int32_t n_haploid, const fvec3_t *coords, fvec3_t *force,
											  float unit, float min_sep_unit, float lambda_sep,
											  float *force_l1, int32_t *n_nonfinite);
float hk_blind_copytrack_accumulate_force(const struct hk_bmap *bmap, const fvec3_t *coords,
										  fvec3_t *force, float unit, float lambda_copytrack,
										  float *force_l1, int32_t *n_nonfinite);
float hk_blind_global_copytrack_accumulate_force(const struct hk_bmap *bmap, const fvec3_t *coords,
												 fvec3_t *force, float unit, float lambda_global_copytrack,
												 float *force_l1, int32_t *n_nonfinite);
float hk_blind_normdir_copytrack_accumulate_force(const struct hk_bmap *bmap, const fvec3_t *coords,
												  fvec3_t *force, float unit,
												  float lambda_normdir_copytrack,
												  float eps_unit,
												  float *force_l1,
												  int32_t *n_nonfinite);
void hk_blind_homolog_sep_stats_init(struct hk_blind_homolog_sep_stats *stats);
int hk_blind_homolog_sep_compute_stats(int32_t n_haploid, const fvec3_t *coords, float collapse_threshold,
									   struct hk_blind_homolog_sep_stats *stats);
void hk_blind_gauge_stats_init(struct hk_blind_gauge_stats *stats);
int hk_blind_temporal_gauge_stabilize_coords(int32_t n_haploid, const int32_t *chr_by_haploid, int32_t n_chr,
											 const fvec3_t *prev_coords, fvec3_t *cur_coords,
											 uint8_t *chr_flipped, struct hk_blind_gauge_stats *stats);
// Coordinates are expected to already be in the same normalized coordinate frame.
int hk_blind_temporal_gauge_stabilize_bmap(const struct hk_bmap *bmap, const fvec3_t *prev_coords, fvec3_t *cur_coords,
										   uint8_t *chr_flipped, struct hk_blind_gauge_stats *stats);
void hk_blind_p4_apply_endpoint_flips(const float in_p4[HK_BLIND_N_STATE], int flip_endpoint0, int flip_endpoint1,
									  float out_p4[HK_BLIND_N_STATE]);
// Applies gauge flips to canonical binned p4; raw endpoint order is handled later by hk_blind_p4_to_raw_order().
int hk_blind_bpair_set_apply_chr_flips(struct hk_blind_bpair_set *set, const struct hk_bmap *bmap,
									   const uint8_t *chr_flipped, int32_t n_chr);
int hk_blind_raw_state_to_canonical_state(int raw_state, uint8_t swapped);
void hk_blind_p4_to_raw_order(const float canonical_p4[HK_BLIND_N_STATE], uint8_t swapped, float raw_p4[HK_BLIND_N_STATE]);
void hk_blind_phase_lock_diag_init(struct hk_blind_phase_lock_diag *diag);
int hk_blind_bpair_set_apply_raw_phase_state_freq_locks(
	struct hk_blind_bpair_set *set, const struct hk_pair *pairs, int32_t n_raw,
	int sample_size_percent, uint64_t seed, struct hk_blind_phase_lock_diag *diag);
void hk_blind_bpair_expand_weighted_edges(const struct hk_blind_bpair *bp, float base_k, float base_d_scale, float rho_train,
										  struct hk_blind_wedge out_edges[HK_BLIND_N_STATE]);
float hk_blind_bpair_effective_rho_train(const struct hk_blind_bpair *bp, float rho_train, int rho_train_mode);
float hk_blind_bpair_effective_rho_train_floor(const struct hk_blind_bpair *bp, float rho_train,
											   int rho_train_mode, float rho_train_floor);
void hk_blind_bpair_expand_weighted_edges_mode(const struct hk_blind_bpair *bp, float base_k, float base_d_scale,
											   float rho_train, int rho_train_mode, int d_scale_mode,
											   float d_scale_eps_count,
											   struct hk_blind_wedge out_edges[HK_BLIND_N_STATE]);
void hk_blind_bpair_expand_weighted_edges_mode_ex(const struct hk_blind_bpair *bp, float base_k, float base_d_scale,
												  float rho_train, int rho_train_mode, float rho_train_floor,
												  int d_scale_mode, float d_scale_eps_count,
												  struct hk_blind_wedge out_edges[HK_BLIND_N_STATE]);
void hk_blind_bpair_expand_weighted_edges_mode_gamma_ex(const struct hk_blind_bpair *bp, float base_k, float base_d_scale,
														float rho_train, int rho_train_mode, float rho_train_floor,
														int d_scale_mode, float d_scale_eps_count,
														float d_scale_posterior_gamma,
														struct hk_blind_wedge out_edges[HK_BLIND_N_STATE]);
float hk_blind_wedge_contact_energy(const struct hk_fdg_conf *conf, const struct hk_blind_wedge *edge, float distance, float unit);
int hk_blind_force_mode_valid(int force_mode);
const char *hk_blind_force_mode_name(int force_mode);
float hk_blind_wedge_list_accumulate_contact_force_cpu_mode(const struct hk_fdg_conf *conf,
														const struct hk_blind_wedge_list *edges,
														const fvec3_t *coords, int32_t n_diploid, float unit,
														int force_mode, fvec3_t *force, int32_t *n_nonfinite);
float hk_blind_wedge_list_accumulate_contact_force_cpu(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
													   const fvec3_t *coords, int32_t n_diploid, float unit,
													   fvec3_t *force, int32_t *n_nonfinite);
void hk_blind_contact_class_diag_init(struct hk_blind_contact_class_diag *diag);
int hk_blind_contact_class_diag_accumulate_force_mode(const struct hk_fdg_conf *conf, const struct hk_bmap *bmap,
													  const struct hk_blind_wedge_list *edges, const fvec3_t *coords,
													  int32_t n_haploid, float unit, int force_mode,
													  struct hk_blind_contact_class_diag *diag);
int hk_blind_contact_class_diag_accumulate(const struct hk_fdg_conf *conf, const struct hk_bmap *bmap,
										   const struct hk_blind_wedge_list *edges, const fvec3_t *coords,
										   int32_t n_haploid, float unit, struct hk_blind_contact_class_diag *diag);
void hk_blind_backbone_diag_init(struct hk_blind_backbone_diag *diag);
float hk_blind_backbone_energy_force_cpu_mode(const struct hk_fdg_conf *conf, const fvec3_t x0, const fvec3_t x1,
													 float unit, float d_scale, int force_mode,
													 fvec3_t f0, fvec3_t f1);
float hk_blind_backbone_energy_force_cpu(const struct hk_fdg_conf *conf, const fvec3_t x0, const fvec3_t x1,
										 float unit, float d_scale, fvec3_t f0, fvec3_t f1);
float hk_blind_backbone_accumulate_force_cpu(const struct hk_fdg_conf *conf, const struct hk_bmap *bmap,
											 const fvec3_t *coords, float unit, fvec3_t *force,
											 struct hk_blind_backbone_diag *diag);
float hk_blind_bending_accumulate_force_cpu(const struct hk_bmap *bmap, const fvec3_t *coords,
											 float unit, float k_bend, fvec3_t *force,
											 float *force_l1, int32_t *n_nonfinite, int64_t *n_triplets);
float hk_blind_confinement_accumulate_force_cpu(int32_t n_diploid, const fvec3_t *coords,
												 float unit, float radius_unit, float k_confine,
												 fvec3_t *force, float *force_l1, int32_t *n_nonfinite,
												 int64_t *n_active);
float hk_blind_rel_rep_schedule_at(int32_t iter, int32_t n_iter);
void hk_blind_repulsion_diag_init(struct hk_blind_repulsion_diag *diag);
float hk_blind_repulsion_energy_force_cpu_mode(const struct hk_fdg_conf *conf, const fvec3_t x0, const fvec3_t x1,
													  float unit, float rel_rep_k, int force_mode,
													  fvec3_t f0, fvec3_t f1);
float hk_blind_repulsion_energy_force_cpu(const struct hk_fdg_conf *conf, const fvec3_t x0, const fvec3_t x1,
										  float unit, float rel_rep_k, fvec3_t f0, fvec3_t f1);
float hk_blind_repulsion_accumulate_force_cpu(const struct hk_fdg_conf *conf, int32_t n_diploid,
											  const fvec3_t *coords, float unit, float rel_rep_k,
											  const struct hk_blind_wedge_list *contact_edges_or_null,
											  const struct hk_bmap *bmap_or_null, fvec3_t *force,
											  struct hk_blind_repulsion_diag *diag);
float hk_blind_repulsion_accumulate_force_cell_cpu(const struct hk_fdg_conf *conf, int32_t n_diploid,
												   const fvec3_t *coords, float unit, float rel_rep_k,
												   const struct hk_blind_wedge_list *contact_edges_or_null,
												   const struct hk_bmap *bmap_or_null, fvec3_t *force,
												   struct hk_blind_repulsion_diag *diag);
void hk_blind_step_diag_init(struct hk_blind_step_diag *diag);
int hk_blind_relax_step_cpu(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
							const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords, float unit, float step,
							float min_sep_unit, float lambda_sep, int enable_repulsion, int repulsion_mode, float rel_rep_k,
							struct hk_blind_step_diag *diag);
int hk_blind_relax_step_cpu_policy(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
								   const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords,
								   float unit, float step, float min_sep_unit, float lambda_sep,
								   int enable_repulsion, int repulsion_mode, float rel_rep_k,
								   float repulsion_block_k_min, int repulsion_policy,
								   struct hk_blind_step_diag *diag);
int hk_blind_relax_step_parent_anchor_cpu(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
										  const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords,
										  float unit, float step, float min_sep_unit, float lambda_sep,
										  int enable_repulsion, int repulsion_mode, float rel_rep_k,
										  const struct hk_blind_coarse_to_fine_map *anchor_map,
										  const fvec3_t *coarse_diploid_coords, float anchor_k,
										  struct hk_blind_step_diag *diag);
void hk_blind_relax_diag_init(struct hk_blind_relax_diag *diag);
int hk_blind_relax_cpu(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
						   const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords, float unit, float step, int32_t n_steps,
						   float min_sep_unit, float lambda_sep, int enable_repulsion, int repulsion_mode, struct hk_blind_relax_diag *diag);
/* CPU-only fixed-posterior callers need the Wave1 blocking policy without an E-step. */
int hk_blind_relax_cpu_policy(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
							  const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords,
							  float unit, float step, int32_t n_steps, float min_sep_unit, float lambda_sep,
							  int enable_repulsion, int repulsion_mode, float repulsion_block_k_min,
							  int repulsion_policy, struct hk_blind_relax_diag *diag);
int hk_blind_relax_cpu_policy_force_mode(const struct hk_fdg_conf *conf,
										 const struct hk_blind_wedge_list *edges,
										 const struct hk_bmap *bmap_or_null, int32_t n_haploid,
										 fvec3_t *coords, float unit, float step, int32_t n_steps,
										 float min_sep_unit, float lambda_sep, int enable_repulsion,
										 int repulsion_mode, float repulsion_block_k_min,
										 int repulsion_policy, int force_mode,
										 struct hk_blind_relax_diag *diag);
int hk_blind_relax_parent_anchor_cpu(const struct hk_fdg_conf *conf, const struct hk_blind_wedge_list *edges,
									 const struct hk_bmap *bmap_or_null, int32_t n_haploid, fvec3_t *coords,
									 float unit, float step, int32_t n_steps, float min_sep_unit, float lambda_sep,
									 int enable_repulsion, int repulsion_mode,
									 const struct hk_blind_coarse_to_fine_map *anchor_map,
									 const fvec3_t *coarse_diploid_coords, float anchor_k,
									 struct hk_blind_relax_diag *diag);
int hk_blind_endpoint_energy_acceptable(double pre, double post, double rtol, double atol,
										double *threshold_out);
/* Compatibility wrapper: force semantics are always physical. */
int hk_blind_relax_with_step_policy(const struct hk_fdg_conf *conf,
									const struct hk_blind_wedge_list *edges,
									const struct hk_bmap *bmap_or_null, int32_t n_haploid,
									fvec3_t *coords,
									const struct hk_blind_single_iter_conf *iter_conf,
									const struct hk_blind_coarse_to_fine_map *anchor_map,
									const fvec3_t *coarse_diploid_coords, float anchor_k,
									struct hk_blind_relax_diag *diag);
int hk_blind_relax_with_step_policy_force_mode(const struct hk_fdg_conf *conf,
											   const struct hk_blind_wedge_list *edges,
											   const struct hk_bmap *bmap_or_null, int32_t n_haploid,
											   fvec3_t *coords,
											   const struct hk_blind_single_iter_conf *iter_conf,
											   const struct hk_blind_coarse_to_fine_map *anchor_map,
											   const fvec3_t *coarse_diploid_coords, float anchor_k,
											   int force_mode, struct hk_blind_relax_diag *diag);
void hk_blind_wedge_list_init(struct hk_blind_wedge_list *list);
void hk_blind_wedge_list_destroy(struct hk_blind_wedge_list *list);
// Build expects a zero-initialized list or one initialized with hk_blind_wedge_list_init().
// It clears current contents, preserves allocated capacity, and refills stats.
int hk_blind_wedge_list_build_from_bpair_set(struct hk_blind_wedge_list *out, const struct hk_blind_bpair_set *set,
											 float base_k, float base_d_scale, float rho_train);
int hk_blind_wedge_list_build_from_bpair_set_params(struct hk_blind_wedge_list *out, const struct hk_blind_bpair_set *set,
													float rho_train);
int hk_blind_wedge_list_build_from_bpair_set_params_mode(struct hk_blind_wedge_list *out, const struct hk_blind_bpair_set *set,
														 float rho_train, int rho_train_mode,
														 int d_scale_mode, float d_scale_eps_count);
int hk_blind_wedge_list_build_from_bpair_set_params_mode_ex(struct hk_blind_wedge_list *out,
															const struct hk_blind_bpair_set *set,
															float rho_train, int rho_train_mode,
															float rho_train_floor, int d_scale_mode,
															float d_scale_eps_count);
int hk_blind_wedge_list_build_softall(struct hk_blind_wedge_list *out,
									  const struct hk_bmap *bmap,
									  const struct hk_blind_bpair_set *set);
int hk_blind_wedge_list_build_softall_mode(struct hk_blind_wedge_list *out,
										   const struct hk_bmap *bmap,
										   const struct hk_blind_bpair_set *set,
										   int d_scale_mode,
										   float d_scale_eps_count);
int hk_blind_wedge_list_build_softall_mode_gamma(struct hk_blind_wedge_list *out,
												 const struct hk_bmap *bmap,
												 const struct hk_blind_bpair_set *set,
												 int d_scale_mode,
												 float d_scale_eps_count,
												 float d_scale_posterior_gamma);
int hk_blind_wedge_list_build_softall_mode_gamma_trans_scale(struct hk_blind_wedge_list *out,
															 const struct hk_bmap *bmap,
															 const struct hk_blind_bpair_set *set,
															 int d_scale_mode,
															 float d_scale_eps_count,
															 float d_scale_posterior_gamma,
															 float trans_d_scale_posterior_gamma,
															 float trans_k_multiplier,
															 float trans_dscale_multiplier);
int hk_blind_wedge_list_build_softall_mode_gamma_trans_control(struct hk_blind_wedge_list *out,
															  const struct hk_bmap *bmap,
															  const struct hk_blind_bpair_set *set,
															  int d_scale_mode,
															  float d_scale_eps_count,
															  float d_scale_posterior_gamma,
															  float trans_d_scale_posterior_gamma,
															  float trans_k_multiplier,
															  float trans_dscale_multiplier,
															  float rho_train,
															  int rho_train_mode,
															  float rho_train_floor,
															  int trans_top1_mode,
															  float trans_top1_min_pmax,
															  float trans_top1_min_margin,
															  float trans_top1_mix_weight);
int hk_blind_wedge_list_build_softall_mode_gamma_trans_control_chrpair(struct hk_blind_wedge_list *out,
																	  const struct hk_bmap *bmap,
																	  const struct hk_blind_bpair_set *set,
																		  int d_scale_mode,
																		  float d_scale_eps_count,
																		  float d_scale_posterior_gamma,
																		  float trans_d_scale_posterior_gamma,
																		  float trans_k_multiplier,
																		  float trans_dscale_multiplier,
																	  float rho_train,
																	  int rho_train_mode,
																	  float rho_train_floor,
																	  int trans_top1_mode,
																	  float trans_top1_min_pmax,
																	  float trans_top1_min_margin,
																	  float trans_top1_mix_weight,
																	  int trans_callable_anchor_mode,
																	  float trans_callable_anchor_top_frac,
																	  float trans_callable_anchor_mix_weight,
																	  int32_t trans_callable_anchor_min_n_raw,
																	  int trans_gate_mode,
																	  float trans_gate_min_pmax,
																	  float trans_gate_min_margin,
																	  float trans_gate_min_neg_entropy,
																	  float trans_chr_pair_mstep_lambda,
																	  float trans_chr_pair_mstep_eps,
																	  float trans_chr_pair_mstep_power,
																	  int trans_chr_pair_mstep_mode);
int hk_blind_wedge_list_build_softall_mode_gamma_trans_control_chrpair_filter(struct hk_blind_wedge_list *out,
																			  const struct hk_bmap *bmap,
																			  const struct hk_blind_bpair_set *set,
																			  int d_scale_mode,
																			  float d_scale_eps_count,
																			  float d_scale_posterior_gamma,
																			  float trans_d_scale_posterior_gamma,
																			  float trans_k_multiplier,
																			  float trans_dscale_multiplier,
																			  float rho_train,
																			  int rho_train_mode,
																			  float rho_train_floor,
																			  int trans_top1_mode,
																			  float trans_top1_min_pmax,
																			  float trans_top1_min_margin,
																			  float trans_top1_mix_weight,
																			  int trans_callable_anchor_mode,
																			  float trans_callable_anchor_top_frac,
																			  float trans_callable_anchor_mix_weight,
																			  int32_t trans_callable_anchor_min_n_raw,
																			  int trans_gate_mode,
																			  float trans_gate_min_pmax,
																			  float trans_gate_min_margin,
																			  float trans_gate_min_neg_entropy,
																			  float trans_chr_pair_mstep_lambda,
																			  float trans_chr_pair_mstep_eps,
																			  float trans_chr_pair_mstep_power,
																			  int trans_chr_pair_mstep_mode,
																			  int softall_filter_mode);
int hk_blind_wedge_list_aggregate_exact(struct hk_blind_wedge_list *list);
void hk_blind_iter_diag_init(struct hk_blind_iter_diag *diag);
void hk_blind_iter_diag_validate_bpair_set(const struct hk_blind_bpair_set *set, struct hk_blind_iter_diag *diag);
void hk_blind_iter_diag_validate_wedge_list(const struct hk_blind_wedge_list *list, struct hk_blind_iter_diag *diag);
int hk_blind_iter_diag_snprintf(char *buf, size_t buf_size, const struct hk_blind_iter_diag *diag);
int hk_blind_run_single_iter_diag(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
								  const struct hk_fdg_conf *conf, const fvec3_t *prev_coords,
								  fvec3_t *cur_coords, float unit, float d_scale, float base_k,
								  const float log_prior[HK_BLIND_N_STATE], float temperature,
								  float rho_train, float min_sep_unit, float lambda_sep,
								  struct hk_blind_iter_diag *diag);
void hk_blind_single_iter_diag_init(struct hk_blind_single_iter_diag *diag);
void hk_blind_single_iter_conf_init(struct hk_blind_single_iter_conf *conf);
int hk_blind_run_single_iter_cpu(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
								 const struct hk_fdg_conf *fdg_conf, fvec3_t *coords,
								 const float log_prior[HK_BLIND_N_STATE],
								 const struct hk_blind_single_iter_conf *iter_conf,
								 struct hk_blind_single_iter_diag *diag);
int hk_blind_run_single_iter_parent_anchor_cpu(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
											   const struct hk_fdg_conf *fdg_conf, fvec3_t *coords,
											   const float log_prior[HK_BLIND_N_STATE],
											   const struct hk_blind_single_iter_conf *iter_conf,
											   const struct hk_blind_coarse_to_fine_map *anchor_map,
											   const fvec3_t *coarse_diploid_coords, float anchor_k,
											   struct hk_blind_single_iter_diag *diag);
void hk_blind_iter_loop_diag_init(struct hk_blind_iter_loop_diag *diag);
int hk_blind_run_iter_loop_cpu(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
							   const struct hk_fdg_conf *fdg_conf, fvec3_t *coords,
							   const float log_prior[HK_BLIND_N_STATE],
							   const struct hk_blind_iter_loop_conf *loop_conf,
							   struct hk_blind_single_iter_diag *per_iter_diag_or_null,
							   struct hk_blind_iter_loop_diag *loop_diag);
float hk_blind_iter_schedule_temperature_at(const struct hk_blind_iter_schedule_conf *schedule_conf, int32_t t);
float hk_blind_iter_schedule_rho_train_at(const struct hk_blind_iter_schedule_conf *schedule_conf, int32_t t);
int hk_blind_run_iter_loop_scheduled_cpu(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
										 const struct hk_fdg_conf *fdg_conf, fvec3_t *coords,
										 const float log_prior[HK_BLIND_N_STATE],
										 const struct hk_blind_iter_schedule_conf *schedule_conf,
										 struct hk_blind_single_iter_diag *per_iter_diag_or_null,
										 struct hk_blind_iter_loop_diag *loop_diag);
int hk_blind_run_iter_loop_scheduled_parent_anchor_cpu(const struct hk_bmap *bmap, struct hk_blind_bpair_set *set,
													   const struct hk_fdg_conf *fdg_conf, fvec3_t *coords,
													   const float log_prior[HK_BLIND_N_STATE],
													   const struct hk_blind_iter_schedule_conf *schedule_conf,
													   const struct hk_blind_coarse_to_fine_map *anchor_map,
													   const fvec3_t *coarse_diploid_coords, float anchor_k,
													   struct hk_blind_single_iter_diag *per_iter_diag_or_null,
													   struct hk_blind_iter_loop_diag *loop_diag);
int hk_blind_write_bpair_posterior_tsv(FILE *fp, const struct hk_bmap *bmap,
									   const struct hk_blind_bpair_set *set);
int hk_blind_write_diploid_coords_tsv(FILE *fp, const struct hk_bmap *bmap, const fvec3_t *coords);
int hk_blind_write_diploid_coords_tsv_gz(const char *path, const struct hk_bmap *bmap, const fvec3_t *coords);
int hk_blind_write_iter_loop_diag_tsv(FILE *fp, const struct hk_blind_iter_loop_diag *diag);
int hk_blind_write_raw_contact_posterior_tsv(FILE *fp, const struct hk_blind_pair *raw, int32_t n_raw,
											 const struct hk_bmap *bmap, const struct hk_blind_bpair_set *set);
int hk_blind_write_bmap_summary_tsv(FILE *fp, const struct hk_bmap *bmap,
									const int32_t *n_child_1mb_or_null);
struct hk_blind_coarse_to_fine_map *hk_blind_coarse_to_fine_map_build(const struct hk_bmap *coarse,
																	  const struct hk_bmap *fine);
void hk_blind_coarse_to_fine_map_destroy(struct hk_blind_coarse_to_fine_map *map);
int hk_blind_write_coarse_to_fine_map_tsv(FILE *fp, const struct hk_bmap *coarse,
										  const struct hk_bmap *fine,
										  const struct hk_blind_coarse_to_fine_map *map);
float hk_blind_density_exposure_from_child_counts(int32_t child_count0, int32_t child_count1,
												  int same_parent);
float hk_blind_bpair_density_exposure(const struct hk_blind_bpair *bp);
float hk_blind_bpair_density_normalized_raw_count(const struct hk_blind_bpair *bp,
												  float eps_count);
float hk_blind_bpair_capped_density_raw_count(const struct hk_blind_bpair *bp,
											  float eps_count);
int hk_blind_bpair_set_apply_density_exposure_from_map(struct hk_blind_bpair_set *set,
													   const struct hk_blind_coarse_to_fine_map *map);
int hk_blind_bpair_set_apply_density_d_scale_from_map(struct hk_blind_bpair_set *set,
													  const struct hk_blind_coarse_to_fine_map *map,
													  int d_scale_mode, float eps_count);
int hk_blind_lift_from_4mb(const struct hk_bmap *coarse,
						   const struct hk_blind_coarse_to_fine_map *map,
						   const fvec3_t *coarse_diploid_coords,
						   fvec3_t *fine_diploid_coords,
						   float child_offset_step);
float hk_blind_parent_centroid_anchor_accumulate_force(const struct hk_blind_coarse_to_fine_map *map,
													   const fvec3_t *fine_diploid_coords,
													   const fvec3_t *coarse_diploid_coords,
													   float k_anchor,
													   fvec3_t *fine_force,
													   int32_t *n_nonfinite,
													   float *force_l1_or_null);

enum hk_fdg_backend {
	HK_FDG_BACKEND_CPU = 0,
	HK_FDG_BACKEND_GPU = 1,
	HK_FDG_BACKEND_AUTO = 2
};

struct hk_fdg_conf {
	float target_radius;
	int n_iter;
	float step;
	float coef_moment;
	float max_f;
	float contact_target;

	float k_rel_rep;
	float d_r;
	float k_bend;
	float k_confine;
	float d_confine;
	float d_b1, d_b2;
	float d_c1, d_c2, d_c3;

	float c_c1, c_c2;
	enum hk_fdg_backend backend;
};

struct hk_v3d_opt {
	int width;
	float line_width;
	float bead_radius;
};

extern int hk_verbose, hk_dbg_flag;

void hk_popt_init(struct hk_popt *opt);

int32_t *hk_sd_ploidy_XY(const struct hk_sdict *d, int32_t *sex_flag);
struct hk_sdict *hk_sd_dup(const struct hk_sdict *d);

struct hk_map *hk_map_read(const char *fn);
void hk_map_destroy(struct hk_map *m);
void hk_map_phase_male_XY(struct hk_map *m);

struct hk_pair *hk_seg2pair(int32_t n_segs, const struct hk_seg *segs, int min_dist, int max_seg, int min_mapq, int32_t *n_pairs_, int all_close_leg);
int32_t hk_pair_dedup(int n_pairs, struct hk_pair *pairs, int min_dist);
int32_t hk_pair_filter_close_legs(int32_t n_pairs, struct hk_pair *pairs, int min_dist, int all_close_leg);
int32_t hk_pair_filter_isolated(int32_t n_pairs, struct hk_pair *pairs, int32_t max_radius, int32_t min_cnt, float drop_frac);
void hk_pair_mark_close(int32_t n_pairs, struct hk_pair *pairs, int radius);
void hk_pair_count_contained(int32_t n_pairs, struct hk_pair *pairs);
void hk_pair_count_nei(int32_t n_pairs, struct hk_pair *pairs, int r1, int r2);
void hk_expected_count(int32_t n_pairs, struct hk_pair *pairs, int radius);
struct hk_map *hk_pair_split_phase(const struct hk_map *m, float phase_thres);

struct hk_pair *hk_pair2tad(const struct hk_sdict *d, int32_t n_pairs, struct hk_pair *pairs, float min_cnt_weight, float area_weight, int32_t *n_tads_);
void hk_mark_by_tad(int32_t n_tads, const struct hk_pair *tads, int32_t n_pairs, struct hk_pair *pairs);
struct hk_pair *hk_pair2loop(const struct hk_sdict *d, int32_t n_pairs, struct hk_pair *pairs, int n_r, const int *r, float min_loop_q, int32_t *n_loops_);

void hk_impute(int32_t n_pairs, struct hk_pair *pairs, int max_radius, int min_radius, int max_nei, int n_iter, float pseudo_cnt, int use_spacial);

void hk_validate_holdback(krng_t *r, float ratio, int32_t n_pairs, struct hk_pair *pairs);
void hk_validate_roc(FILE *fp, int32_t n_pairs, struct hk_pair *pairs);

struct hk_bmap *hk_3dg_read(const char *fn);
struct hk_bmap *hk_bmap_gen(const struct hk_sdict *d, int32_t n_pairs, const struct hk_pair *pairs, int size, int bmap_skip_merge_flag);
struct hk_bmap *hk_bmap_gen_from_beads(const struct hk_sdict *d, int32_t n_beads, const struct hk_bead *beads,
									   int32_t n_pairs, const struct hk_pair *pairs);
struct hk_bmap *hk_bmap_bead_dup(const struct hk_bmap *m0);
int32_t hk_pair_flt_3d(const struct hk_bmap *m, int32_t n_pairs, struct hk_pair *pairs, float max_factor);
void hk_bmap_destroy(struct hk_bmap *m);
int hk_bmap_apply_gc_correction(struct hk_bmap *m, const char *cpg_fn, int degree);

void hk_fdg_conf_init(struct hk_fdg_conf *opt);
void hk_fdg_cal_c(struct hk_fdg_conf *opt);
void hk_fdg(const struct hk_fdg_conf *opt, struct hk_bmap *m, const struct hk_bmap *src, krng_t *rng);
void hk_check_dist(struct hk_bmap *m);

void hk_print_seg(FILE *fp, const struct hk_sdict *d, int32_t n_segs, const struct hk_seg *segs);
void hk_print_pair(FILE *fp, int flag, const struct hk_sdict *d, int32_t n_pairs, const struct hk_pair *pairs);
void hk_print_bmap(FILE *fp, const struct hk_bmap *m);
void hk_print_3dg(FILE *fp, const struct hk_bmap *m);

void hk_pair_image(const struct hk_sdict *d, int32_t n_pairs, const struct hk_pair *pairs, int w, float phase_thres, int no_grad,
				   int n_tads, const struct hk_pair *tads, int n_tads_prev, const struct hk_pair *tads_prev, const char *fn);

void hk_v3d_opt_init(struct hk_v3d_opt *opt);
void hk_v3d_prep(int *argc, char *argv[]);
void hk_v3d_view(struct hk_bmap *m, const struct hk_v3d_opt *opt, int color_seed, const char *hl);
void hk_fdg_normalize(struct hk_bmap *m);

#ifdef __cplusplus
}
#endif

#endif
