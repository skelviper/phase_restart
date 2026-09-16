#include <string.h>
#include <assert.h>
#include <math.h>
#include <zlib.h>
#include "hkpriv.h"
#include "khash.h"
#include "ksort.h"

#include "kseq.h"
KSTREAM_INIT(gzFile, gzread, 0x10000)

extern int hk_verbose;

#define bpair_lt(a, b) ((a).bid[0] < (b).bid[0] || ((a).bid[0] == (b).bid[0] && (a).bid[1] < (b).bid[1]))
KSORT_INIT(bpair, struct hk_bpair, bpair_lt)

KSORT_INIT_GENERIC(int32_t)

void hk_bpair_sort(int32_t n_pairs, struct hk_bpair *pairs)
{
	ks_introsort_bpair(n_pairs, pairs);
}

struct cnt_aux {
	int32_t bid[2];
	int32_t n, max_nei;
	float p_sum;
};

static inline uint32_t hash_pair(struct cnt_aux x)
{
	return (uint32_t)hash64((uint64_t)x.bid[0] << 32 | x.bid[1]);
}

#define pair_eq(a, b) ((a).bid[0] == (b).bid[0] && (a).bid[1] == (b).bid[1])
KHASH_INIT(bin_cnt, struct cnt_aux, char, 0, hash_pair, pair_eq)

void hk_bmap_set_offcnt(struct hk_bmap *m)
{
	int32_t i, off = 0;
	assert(m && m->d && m->beads);
	if (m->offcnt) free(m->offcnt);
	m->offcnt = CALLOC(uint64_t, m->d->n + 1);
	for (i = 1; i <= m->n_beads; ++i) {
		if (i == m->n_beads || m->beads[i].chr != m->beads[i-1].chr) {
			m->offcnt[m->beads[off].chr] = (uint64_t)off << 32 | (i - off);
			off = i;
		}
	}
	m->offcnt[m->d->n] = (uint64_t)off << 32 | 0;
}

void hk_bmap_gen_beads_uniform(struct hk_bmap *m, int size)
{
	int32_t i, m_beads = 0;
	assert(m && m->d);
	for (i = 0; i < m->d->n; ++i) {
		int32_t st = 0, len = m->d->len[i];
		while (st < len) {
			int32_t l = size * 1.5 < len - st? size : len - st;
			struct hk_bead *p;
			if (m->n_beads == m_beads)
				EXPAND(m->beads, m_beads);
			p = &m->beads[m->n_beads++];
			p->chr = i, p->st = st, p->en = st + l;
			st += l;
		}
	}
	hk_bmap_set_offcnt(m);
}

int hk_bmap_pos2bid(const struct hk_bmap *m, int32_t chr, int32_t pos)
{
	int32_t lo, hi;
	lo = m->offcnt[chr] >> 32;
	hi = lo + (int32_t)m->offcnt[chr] - 1;
	assert(pos < m->beads[hi].en);
	while (lo <= hi) {
		int32_t mid = (lo + hi) / 2;
		const struct hk_bead *p = &m->beads[mid];
		if (p->st <= pos && pos < p->en) return mid;
		else if (p->st < pos) lo = mid + 1;
		else hi = mid - 1;
	}
	abort(); // if here, it is a bug
	return -1;
}

static khash_t(bin_cnt) *hash_count(struct hk_bmap *m, int32_t n_pairs, const struct hk_pair *pairs)
{
	int32_t i;
	khash_t(bin_cnt) *h;
	h = kh_init(bin_cnt);
	for (i = 0; i < n_pairs; ++i) {
		const struct hk_pair *p = &pairs[i];
		struct cnt_aux c;
		int absent;
		khint_t k;
		c.bid[0] = hk_bmap_pos2bid(m, p->chr>>32,      hk_ppos1(p));
		c.bid[1] = hk_bmap_pos2bid(m, (int32_t)p->chr, hk_ppos2(p));
		c.n = 1, c.p_sum = p->_.phased_prob;
		c.max_nei = p->n_nei;
		k = kh_put(bin_cnt, h, c, &absent);
		if (!absent) {
			struct cnt_aux *q = &kh_key(h, k);
			++q->n, q->p_sum += c.p_sum;
			q->max_nei = q->max_nei > c.max_nei? q->max_nei : c.max_nei;
		}
	}
	return h;
}

void hk_bmap_merge_beads(struct hk_bmap *m, int32_t n_pairs, const struct hk_pair *pairs)
{
	int32_t i, n, *cnt;
	khash_t(bin_cnt) *h;
	khint_t k;

	h = hash_count(m, n_pairs, pairs);
	cnt = CALLOC(int32_t, m->n_beads);
	for (k = 0; k < kh_end(h); ++k) {
		struct cnt_aux *q;
		if (!kh_exist(h, k)) continue;
		q = &kh_key(h, k);
		if (q->bid[0] == q->bid[1]) continue;
		if (m->beads[q->bid[0]].chr == m->beads[q->bid[1]].chr && q->bid[1] == q->bid[0] + 1) continue;
		++cnt[q->bid[0]];
		++cnt[q->bid[1]];
	}

	for (i = n = 0; i < m->n_beads; ++i) {
		if (cnt[i] < 1) {
			if (i == 0 || m->beads[i-1].chr != m->beads[i].chr)
				m->beads[n++] = m->beads[i];
			else m->beads[n-1].en = m->beads[i].en;
		} else m->beads[n++] = m->beads[i];
	}
	if (hk_verbose >= 3)
		fprintf(stderr, "[M::%s] %d => %d\n", __func__, m->n_beads, n);
	m->n_beads = n;

	free(cnt);
	kh_destroy(bin_cnt, h);
	hk_bmap_set_offcnt(m);
}

struct hk_bmap *hk_bmap_gen(const struct hk_sdict *d, int32_t n_pairs, const struct hk_pair *pairs, int size, int bmap_skip_merge_flag)
{
	int32_t n_del, m_pairs = 0;
	khash_t(bin_cnt) *h;
	struct hk_bmap *m;
	khint_t k;

	m = CALLOC(struct hk_bmap, 1);
	m->d = hk_sd_dup(d);

	hk_bmap_gen_beads_uniform(m, size);
	if (hk_verbose >= 3)
		fprintf(stderr, "[M::%s] generated %d beads\n", __func__, m->n_beads);
	if (bmap_skip_merge_flag == 0) {
		hk_bmap_merge_beads(m, n_pairs, pairs);
		hk_bmap_merge_beads(m, n_pairs, pairs);
	}

	h = hash_count(m, n_pairs, pairs);
	for (k = 0, n_del = 0; k < kh_end(h); ++k) {
		struct hk_bpair *p;
		struct cnt_aux *q;
		if (!kh_exist(h, k)) continue;
		q = &kh_key(h, k);
		if (m->n_pairs == m_pairs)
			EXPAND(m->pairs, m_pairs);
		p = &m->pairs[m->n_pairs++];
		p->bid[0] = q->bid[0], p->bid[1] = q->bid[1];
		p->n = q->n, p->p = q->p_sum / q->n, p->max_nei = q->max_nei;
	}
	kh_destroy(bin_cnt, h);
	hk_bpair_sort(m->n_pairs, m->pairs);
	if (hk_verbose >= 3)
		fprintf(stderr, "[M::%s] %d bin pairs\n", __func__, m->n_pairs);
	return m;
}

struct hk_bmap *hk_bmap_gen_from_beads(const struct hk_sdict *d, int32_t n_beads, const struct hk_bead *beads,
									   int32_t n_pairs, const struct hk_pair *pairs)
{
	int32_t m_pairs = 0;
	khash_t(bin_cnt) *h;
	struct hk_bmap *m;
	khint_t k;

	assert(d);
	assert(n_beads >= 0);
	assert(n_beads == 0 || beads);
	assert(n_pairs >= 0);
	assert(n_pairs == 0 || pairs);

	m = CALLOC(struct hk_bmap, 1);
	if (m == 0)
		return 0;
	m->d = hk_sd_dup(d);
	if (m->d == 0)
		goto fail;
	m->n_beads = n_beads;
	if (n_beads > 0) {
		m->beads = MALLOC(struct hk_bead, n_beads);
		if (m->beads == 0)
			goto fail;
		memcpy(m->beads, beads, (size_t)n_beads * sizeof(*beads));
	}
	hk_bmap_set_offcnt(m);
	h = hash_count(m, n_pairs, pairs);
	if (h == 0)
		goto fail;
	for (k = 0; k < kh_end(h); ++k) {
		struct hk_bpair *p;
		struct cnt_aux *q;
		if (!kh_exist(h, k)) continue;
		q = &kh_key(h, k);
		if (m->n_pairs == m_pairs)
			EXPAND(m->pairs, m_pairs);
		p = &m->pairs[m->n_pairs++];
		p->bid[0] = q->bid[0], p->bid[1] = q->bid[1];
		p->n = q->n, p->p = q->p_sum / q->n, p->max_nei = q->max_nei;
	}
	kh_destroy(bin_cnt, h);
	hk_bpair_sort(m->n_pairs, m->pairs);
	if (hk_verbose >= 3)
		fprintf(stderr, "[M::%s] reused %d beads, %d bin pairs\n", __func__, m->n_beads, m->n_pairs);
	return m;

fail:
	if (m)
		hk_bmap_destroy(m);
	return 0;
}

static int hk_bmap_load_cpg(const struct hk_bmap *m, const char *fn, float *cpg, char *mask, int *n_used, int *n_dup)
{
	gzFile fp;
	kstream_t *ks;
	kstring_t str = {0, 0, 0};
	int dret, used = 0, dup = 0;

	fp = gzopen(fn, "rb");
	if (fp == 0) return -1;
	ks = ks_init(fp);
	while (ks_getuntil(ks, KS_SEP_LINE, &str, &dret) >= 0) {
		char *fields[4];
		int chr, n_fields = 0;
		int32_t st, en;
		float val;
		char *p, *q;
		if (str.l == 0 || str.s[0] == '#') continue;
		for (p = q = str.s;; ++q) {
			if (*q == '\t' || *q == 0) {
				int c = *q;
				*q = 0;
				if (n_fields < 4) fields[n_fields++] = p;
				if (c == 0) break;
				p = q + 1;
			}
		}
		if (n_fields < 4) continue;
		chr = hk_sd_get(m->d, fields[0]);
		int targets[2], n_targets = 0;
		if (chr >= 0 && chr < m->d->n) {
			targets[n_targets++] = chr;
		} else {
			size_t l = strlen(fields[0]);
			if (l + 2 < 256) {
				char buf[256];
				memcpy(buf, fields[0], l);
				buf[l + 1] = 0;
				buf[l] = 'a';
				chr = hk_sd_get(m->d, buf);
				if (chr >= 0 && chr < m->d->n) targets[n_targets++] = chr;
				buf[l] = 'b';
				chr = hk_sd_get(m->d, buf);
				if (chr >= 0 && chr < m->d->n) targets[n_targets++] = chr;
			}
		}
		if (n_targets == 0) continue;
		st = atoi(fields[1]);
		en = atoi(fields[2]);
		val = atof(fields[3]);
		for (int ti = 0; ti < n_targets; ++ti) {
			int tgt = targets[ti];
			if (st < 0 || st >= m->d->len[tgt]) continue;
			if (en > 0 && st >= en) continue;
			if ((m->offcnt[tgt] >> 32) == 0 && (int32_t)m->offcnt[tgt] == 0) continue;
			if (st >= (int32_t)(m->d->len[tgt])) continue;
			int32_t bid = hk_bmap_pos2bid(m, tgt, st);
			if (bid < 0 || bid >= m->n_beads) continue;
			if (mask[bid]) ++dup;
			else ++used;
			cpg[bid] = val;
			mask[bid] = 1;
		}
	}
	ks_destroy(ks);
	free(str.s);
	gzclose(fp);
	if (n_used) *n_used = used;
	if (n_dup) *n_dup = dup;
	return 0;
}

static void hk_bmap_coverage(const struct hk_bmap *m, double *cov)
{
	int32_t i;
	for (i = 0; i < m->n_pairs; ++i) {
		const struct hk_bpair *p = &m->pairs[i];
		cov[p->bid[0]] += p->n;
		cov[p->bid[1]] += p->n;
	}
}

static int solve_poly(const double *sum_x, const double *sum_xy, int degree, double *coef)
{
	int i, j, k, n;
	double *a = 0;

	n = degree + 1;
	a = CALLOC(double, n * (n + 1));
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j)
			a[i * (n + 1) + j] = sum_x[i + j];
		a[i * (n + 1) + n] = sum_xy[i];
	}

	for (i = 0; i < n; ++i) {
		int pivot = i;
		for (j = i + 1; j < n; ++j)
			if (fabs(a[j * (n + 1) + i]) > fabs(a[pivot * (n + 1) + i]))
				pivot = j;
		if (fabs(a[pivot * (n + 1) + i]) < 1e-12) {
			free(a);
			return -1;
		}
		if (pivot != i) {
			for (k = i; k <= n; ++k) {
				double tmp = a[i * (n + 1) + k];
				a[i * (n + 1) + k] = a[pivot * (n + 1) + k];
				a[pivot * (n + 1) + k] = tmp;
			}
		}
		for (k = n; k >= i; --k)
			a[i * (n + 1) + k] /= a[i * (n + 1) + i];
		for (j = 0; j < n; ++j) {
			double factor;
			if (j == i) continue;
			factor = a[j * (n + 1) + i];
			for (k = i; k <= n; ++k)
				a[j * (n + 1) + k] -= factor * a[i * (n + 1) + k];
		}
	}
	for (i = 0; i < n; ++i)
		coef[i] = a[i * (n + 1) + n];

	free(a);
	return 0;
}

static int float_cmp(const void *a, const void *b)
{
	float fa = *(const float*)a;
	float fb = *(const float*)b;
	return (fa > fb) - (fa < fb);
}

int hk_bmap_apply_gc_correction(struct hk_bmap *m, const char *cpg_fn, int degree)
{
	const double min_exp = 1e-6;
	int ret = -1, n_used = 0, n_dup = 0;
	int32_t i, n_missing;
	float *cpg = 0, *bias = 0;
	char *mask = 0;
	double *cov = 0, *sum_x = 0, *sum_xy = 0, *coef = 0;
	double pred_sum = 0.0;

	if (m == 0 || cpg_fn == 0) return -1;
	if (degree < 1 || degree > HK_GC_MAX_DEGREE) {
		if (hk_verbose >= 1)
			fprintf(stderr, "[E::%s] invalid GC degree %d (expected 1-%d)\n", __func__, degree, HK_GC_MAX_DEGREE);
		return -1;
	}
	if (m->cpg) free(m->cpg);
	if (m->gc_bias) free(m->gc_bias);
	m->cpg = 0, m->gc_bias = 0, m->gc_corrected = 0;

	cpg = CALLOC(float, m->n_beads);
	mask = CALLOC(char, m->n_beads);
	cov = CALLOC(double, m->n_beads);
	if (hk_bmap_load_cpg(m, cpg_fn, cpg, mask, &n_used, &n_dup) != 0) {
		if (hk_verbose >= 1)
			fprintf(stderr, "[E::%s] failed to read CpG file: %s\n", __func__, cpg_fn);
		goto cleanup;
	}
	if (n_used < degree + 1) {
		if (hk_verbose >= 1)
			fprintf(stderr, "[E::%s] insufficient CpG bins matched beads (%d found; need >=%d)\n",
					__func__, n_used, degree + 1);
		goto cleanup;
	}

	hk_bmap_coverage(m, cov);
	sum_x = CALLOC(double, 2 * degree + 1);
	sum_xy = CALLOC(double, degree + 1);
	coef = CALLOC(double, degree + 1);
	if (sum_x == 0 || sum_xy == 0 || coef == 0) goto cleanup;
	for (i = 0; i < m->n_beads; ++i) {
		if (!mask[i]) continue;
		{
			double x = cpg[i];
			double y = cov[i];
			double x_pow = 1.0;
			int k;
			for (k = 0; k <= 2 * degree; ++k) {
				sum_x[k] += x_pow;
				if (k <= degree) sum_xy[k] += y * x_pow;
				x_pow *= x;
			}
		}
	}
	if (solve_poly(sum_x, sum_xy, degree, coef) != 0) {
		if (hk_verbose >= 1)
			fprintf(stderr, "[E::%s] failed to fit coverage~CpG polynomial (degree %d)\n", __func__, degree);
		goto cleanup;
	}

	bias = CALLOC(float, m->n_beads);
	for (i = 0; i < m->n_beads; ++i) {
		if (!mask[i]) continue;
		double x = cpg[i];
		double pred = coef[degree];
		int k;
		for (k = degree - 1; k >= 0; --k)
			pred = pred * x + coef[k];
		if (pred < min_exp) pred = min_exp;
		bias[i] = (float)pred;
		pred_sum += pred;
	}
	{
		double mean_pred = pred_sum / n_used;
		double inv_mean = mean_pred > min_exp? 1.0 / mean_pred : 1.0;
		for (i = 0; i < m->n_beads; ++i)
			bias[i] = mask[i]? (float)(bias[i] * inv_mean) : 1.0f;
	}

	// Winsorize extreme bias values to prevent overly aggressive normalization.
	{
		const double clip_q = 0.01;
		float hard_min = 0.5f;
		float hard_max = 2.0f;
		const char *min_env = getenv("HK_GC_BIAS_MIN");
		const char *max_env = getenv("HK_GC_BIAS_MAX");
		float *bias_used = 0;
		int n_bias = 0, n_low = 0, n_high = 0;

		if (min_env && min_env[0]) hard_min = (float)atof(min_env);
		if (max_env && max_env[0]) hard_max = (float)atof(max_env);
		if (hard_min <= 0.0f) hard_min = 0.05f;
		if (hard_max < hard_min) hard_max = hard_min;

		if (n_used > 0) bias_used = MALLOC(float, n_used);
		for (i = 0; i < m->n_beads; ++i)
			if (mask[i]) bias_used[n_bias++] = bias[i];
		if (n_bias > 0) {
			int idx_low, idx_high;
			float lo, hi;
			qsort(bias_used, n_bias, sizeof(float), float_cmp);
			idx_low = (int)floor(clip_q * (n_bias - 1));
			idx_high = (int)floor((1.0 - clip_q) * (n_bias - 1));
			lo = bias_used[idx_low];
			hi = bias_used[idx_high];
			if (lo < hard_min) lo = hard_min;
			if (hi > hard_max) hi = hard_max;
			if (hi < lo) hi = lo;
			for (i = 0; i < m->n_beads; ++i) {
				if (!mask[i]) continue;
				if (bias[i] < lo) {
					bias[i] = lo;
					++n_low;
				} else if (bias[i] > hi) {
					bias[i] = hi;
					++n_high;
				}
			}
			if (hk_verbose >= 2 && (n_low || n_high))
				fprintf(stderr, "[I::%s] GC bias winsorize: lo=%.4g hi=%.4g clipped=%d/%d\n",
						__func__, lo, hi, n_low + n_high, n_bias);
		}
		free(bias_used);
	}

	for (i = 0; i < m->n_pairs; ++i) {
		struct hk_bpair *p = &m->pairs[i];
		float exp = bias[p->bid[0]] * bias[p->bid[1]];
		if (exp < min_exp) exp = (float)min_exp;
		p->gc_exp = exp;
		p->gc_norm_n = p->n / exp;
	}

	// Rescale GC-normalized counts to keep total contact weight unchanged
	{
		double sum_raw = 0.0, sum_gc = 0.0;
		for (i = 0; i < m->n_pairs; ++i) {
			const struct hk_bpair *p = &m->pairs[i];
			sum_raw += (double)p->n;
			sum_gc += (double)p->gc_norm_n;
		}
		if (sum_gc > 0.0) {
			double scale = sum_raw / sum_gc;
			for (i = 0; i < m->n_pairs; ++i)
				m->pairs[i].gc_norm_n *= (float)scale;
			if (hk_verbose >= 2)
				fprintf(stderr, "[I::%s] GC norm rescale: raw_sum=%.3g gc_sum=%.3g scale=%.6g\n",
						__func__, sum_raw, sum_gc, scale);
		}
	}

	n_missing = m->n_beads - n_used;
	if (hk_verbose >= 2) {
		int k;
		fprintf(stderr, "[I::%s] GC fit degree=%d coef:", __func__, degree);
		for (k = 0; k <= degree; ++k)
			fprintf(stderr, " %.4g", coef[k]);
		fprintf(stderr, " (used=%d missing=%d dup=%d)\n", n_used, n_missing, n_dup);
	}
	{
		double minv = 1e9, maxv = -1e9, sumv = 0.0;
		int n_bias = 0;
		for (i = 0; i < m->n_beads; ++i) {
			float b = bias[i];
			if (b <= 0.0f) continue;
			if (b < minv) minv = b;
			if (b > maxv) maxv = b;
			sumv += b;
			++n_bias;
		}
		if (hk_verbose >= 1 && n_bias > 0)
			fprintf(stderr, "[D::%s] bias n=%d min=%.4g max=%.4g mean=%.4g\n",
					__func__, n_bias, minv, maxv, sumv / n_bias);
		const char *dump = getenv("HK_GC_BIAS_OUT");
		if (dump && dump[0]) {
			FILE *fp = fopen(dump, "w");
			if (fp) {
				fprintf(fp, "bead\tchr\tstart\tbias\n");
				for (i = 0; i < m->n_beads; ++i)
					fprintf(fp, "%d\t%s\t%d\t%.6g\n", i,
							m->d->name[m->beads[i].chr], m->beads[i].st, bias[i]);
				fclose(fp);
			}
		}
	}

	m->cpg = cpg;
	m->gc_bias = bias;
	m->gc_corrected = 1;
	ret = 0;

cleanup:
	free(mask);
	free(cov);
	free(sum_x);
	free(sum_xy);
	free(coef);
	if (ret != 0) {
		if (cpg) free(cpg);
		if (bias) free(bias);
	}
	return ret;
}

struct hk_bmap *hk_bmap_bead_dup(const struct hk_bmap *m0)
{
	struct hk_bmap *m;
	m = CALLOC(struct hk_bmap, 1);
	m->unit = m0->unit;
	m->d = hk_sd_dup(m0->d);
	m->n_beads = m0->n_beads;
	m->beads = CALLOC(struct hk_bead, m->n_beads);
	memcpy(m->beads, m0->beads, m->n_beads * sizeof(struct hk_bead));
	hk_bmap_set_offcnt(m);
	if (m0->x) {
		m->x = CALLOC(fvec3_t, m->n_beads);
		memcpy(m->x, m0->x, m->n_beads * sizeof(fvec3_t));
	}
	if (m0->cpg) {
		m->cpg = CALLOC(float, m->n_beads);
		memcpy(m->cpg, m0->cpg, m->n_beads * sizeof(float));
	}
	if (m0->gc_bias) {
		m->gc_bias = CALLOC(float, m->n_beads);
		memcpy(m->gc_bias, m0->gc_bias, m->n_beads * sizeof(float));
	}
	m->gc_corrected = m0->gc_corrected;
	return m;
}

void hk_bmap_destroy(struct hk_bmap *m)
{
	if (m->d) hk_sd_destroy(m->d);
	free(m->x);
	free(m->offcnt);
	free(m->beads);
	free(m->pairs);
	free(m->feat);
	free(m->cpg);
	free(m->gc_bias);
	free(m);
}
