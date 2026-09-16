import numpy as np

from pr.allele_calibration_evaluator import _bounded_rho, _macro_summary, chromosome_metrics


class TinyTemplate:
    chromosome_names = ("chr1", "chr2")

    def chromosome_slice(self, chromosome):
        return slice(8 * chromosome, 8 * (chromosome + 1))


def _line(values, offset=0.0):
    coordinates = np.zeros((len(values), 3), dtype=np.float64)
    coordinates[:, 0] = values
    coordinates[:, 1] = offset
    return coordinates


def _fixtures():
    template = TinyTemplate()
    mat = np.concatenate([_line([0, 1, 3, 6, 10, 15, 21, 28]), _line([0, 1, 3, 6, 10, 15, 21, 28], 10)])
    pat = np.concatenate([_line([0, 2, 5, 9, 14, 20, 27, 35]), _line([0, 2, 5, 9, 14, 20, 27, 35], 20)])
    truth = np.stack([mat, pat])
    return template, mat, pat, truth


def test_candidate_label_swap_invariance_and_zero_assigned_shape_error():
    template, mat, pat, truth = _fixtures()
    direct = np.stack([mat, pat])
    swapped = np.stack([pat, mat])
    first = chromosome_metrics(template, direct, truth, same_shape=False, chromosome=0)
    second = chromosome_metrics(template, swapped, truth, same_shape=False, chromosome=0)
    assert first["orientation"] == "direct"
    assert second["orientation"] == "swapped"
    assert first["shape_error"]["mean"] == 0.0
    assert second["shape_error"]["mean"] == 0.0
    for key in ("matched", "cross", "contrast", "margin_ref1_mat", "margin_ref2_pat", "minmargin"):
        assert first[key] == second[key]


def test_swap_is_independent_per_chromosome():
    template, mat, pat, truth = _fixtures()
    candidate = np.stack([
        np.concatenate([mat[:8], pat[8:]]),
        np.concatenate([pat[:8], mat[8:]]),
    ])
    records = [
        chromosome_metrics(template, candidate, truth, same_shape=False, chromosome=index)
        for index in range(2)
    ]
    assert [record["orientation"] for record in records] == ["direct", "swapped"]
    assert all(record["shape_error"]["mean"] == 0.0 for record in records)


def test_n_common_truth_tie_and_invalid_rho_contract():
    template, mat, pat, _truth = _fixtures()
    truth = np.stack([mat, mat])
    record = chromosome_metrics(template, np.stack([mat, pat]), truth, same_shape=True, chromosome=0)
    assert record["orientation"] == "unresolved_tie"
    assert record["matched"] is not None and record["cross"] is not None
    assert record["contrast"] == 0.0
    assert record["margin_ref1_mat"] is None
    assert record["margin_ref2_pat"] is None
    assert record["minmargin"] is None
    assert _bounded_rho(np.arange(19.0), np.arange(19.0)) is None
    assert _bounded_rho(np.ones(21), np.arange(21.0)) is None


def test_all_invalid_n_macro_contrast_is_na():
    records = []
    for _index in range(2):
        records.append({
            "orientation": "unresolved_missing",
            "raw_four_rho": {"rho_A_mat": None, "rho_A_pat": None, "rho_B_mat": None, "rho_B_pat": None},
            "direct_original": None, "cross_original": None, "matched": None, "cross": None,
            "contrast": None, "margin_ref1_mat": None, "margin_ref2_pat": None, "minmargin": None,
            "pooled_candidate_offdiag_rms": None, "pooled_truth_offdiag_rms": None,
            "shape_error": {"copy_first_or_assigned_first": None, "copy_second_or_assigned_second": None,
                            "mean": None, "max": None},
            "N_only_copy_difference_rms": None,
        })
    summary = _macro_summary(records, True)
    assert summary["contrast_macro"] is None
    assert summary["contrast_valid_chr_count"] == 0
