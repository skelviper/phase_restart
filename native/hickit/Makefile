.DEFAULT_GOAL := all
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:

ifneq ($(filter test% smoke%,$(MAKECMDGOALS)),)
ifneq ($(CONDA_DEFAULT_ENV),analysis)
$(error Activate the analysis Conda environment before running tests)
endif
endif

CUDA_HOME ?= /usr/local/cuda
CUDA_LIB  ?= $(CUDA_HOME)/targets/x86_64-linux/lib
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CFLAGS   ?= -g -Wall -O2 -Wc++-compat -fno-fast-math -fno-unsafe-math-optimizations
NVCCFLAGS ?= -O3
CPPFLAGS ?=
INCLUDES ?=
LDFLAGS  += -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB)
LIBS     := -lm -lz
AUDIT_LIBS := -lm -lz
LIBS_GL  :=
ASAN_FLAG :=
TEST_RES_ROOT ?= $(if $(HK_BLIND_TEST_RES_ROOT),$(HK_BLIND_TEST_RES_ROOT),$(abspath $(CURDIR)/../test_res))
override SMOKE_RES_ROOT = $$temporary_root
P9016_MINIMAL_SMOKE_ROOT := $(SMOKE_RES_ROOT)/hk_blind_p9016_minimal_audit_smoke
P9016_SEP_ON_SMOKE_ROOT := $(SMOKE_RES_ROOT)/hk_blind_p9016_sep_on_smoke
P9016_EXPECTED_SEP_SMOKE_ROOT := $(SMOKE_RES_ROOT)/hk_blind_p9016_expected_sep_smoke
P9016_RESOLUTION_CHAIN_SMOKE_ROOT := $(SMOKE_RES_ROOT)/hk_blind_p9016_resolution_chain_smoke
GRAPH_DIAG_SMOKE_ROOT := $(SMOKE_RES_ROOT)/hk_blind_graph_diagnostics_smoke

define REBASE_AUDIT_FIXTURE_PATHS
awk -F'\t' -v OFS='\t' -v dir="$(1)" '$$1 == "output_dir" { $$2 = dir; print; next } ($$1 ~ /^output_/ || $$1 == "optimizer_trace_path") && $$2 ~ /^\// { n = split($$2, part, "/"); $$2 = dir "/" part[n] } { print }' "$(1)/p9016_full.manifest.tsv" >"$(1)/manifest.tmp" && mv "$(1)/manifest.tmp" "$(1)/p9016_full.manifest.tsv"
endef

ifeq ($(gpu),1)
	FDG_GPU_OBJ := fdg_gpu.o
	FDG_GPU_STAMP := .fdg_gpu_backend.gpu
	LIBS += -lcudart -lstdc++
else
	FDG_GPU_OBJ := fdg_gpu_stub.o
	FDG_GPU_STAMP := .fdg_gpu_backend.cpu
endif

OBJS := sdict.o io.o pair.o count.o phase.o bin.o fdg.o image.o view3d.o $(FDG_GPU_OBJ)
PROG := hickit
BLIND_COMMON_SRCS := sdict.c io.c pair.c count.c phase.c bin.c blind.c fdg.c image.c view3d.c
BLIND_COMMON_OBJS := sdict.o io.o pair.o count.o phase.o bin.o blind.o fdg.o image.o view3d.o $(FDG_GPU_OBJ)
ifneq ($(asan),)
	ASAN_FLAG = -fsanitize=address,undefined -fno-omit-frame-pointer
endif

ifneq ($(gl),)
	CPPFLAGS += -DHAVE_GL
	ifeq ($(shell uname),Darwin)
		CFLAGS += -Wno-deprecated-declarations
		LIBS_GL = -framework OpenGL -framework GLUT
	else
		LIBS_GL = -Wl,-Bstatic -lglut -Wl,-Bdynamic -lGLU -lGL -lXi
	endif
endif

.PHONY: all test test_python test_diploid_marginal test_p9016_geometry_evaluator_v1 test_p9016_unordered_homolog test_maintenance test_softall_hardpc_state_machine test_softall_hardpc_multires test_p9016_optimizer_trace test_force_causal_ablation test_optimizer_stability_control test_blind_gradient_units test_blind_gradient_units_gpu test_blind_estep_scores test_blind_relax_step_policy test_blind_relax_step_policy_gpu test_blind_endpoint_backtracking test_p9016_endpoint_backtracking test_p9016_endpoint_backtracking_gpu test_p9016_force_mode test_p9016_force_mode_gpu test_p9016_fixed_source_coords smoke_blind_estep_modes test_graph_diagnostics test_fixed_posterior_graph_ablation smoke_blind_p9016_minimal smoke_blind_p9016_sep_on smoke_blind_p9016_expected_sep smoke_blind_p9016_resolution_chain smoke_blind_resolution_ploidy test_audit_optimizer_trace_manifest audit_blind_p9016_full_cpu_output clean depend
.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) $< -o $@

$(BLIND_COMMON_OBJS) main.o: hickit.h hkpriv.h krng.h

fdg_gpu.o: fdg_gpu.cu fdg_gpu.h hickit.h
	$(NVCC) -c $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

$(FDG_GPU_STAMP):
	@rm -f .fdg_gpu_backend.*
	@touch $@

all: $(PROG) run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin

hickit: $(OBJS) main.o
	$(CC) $(LDFLAGS) -o $@ $^ $(ASAN_FLAG) $(LIBS_GL) $(LIBS)

test:
	+bash scripts/run_tests.sh all

test_python:
	bash scripts/run_tests.sh python

test_diploid_marginal:
	bash scripts/run_tests.sh python prototype/diploid_marginal/tests

test_p9016_geometry_evaluator_v1:
	bash scripts/run_tests.sh python tests/test_p9016_geometry_evaluator_v1.py

test_p9016_unordered_homolog:
	bash scripts/run_tests.sh python tests/test_p9016_unordered_homolog.py

test_maintenance: test

test_softall_hardpc_state_machine: $(BLIND_COMMON_OBJS)
	temporary_root=$$(mktemp -d /tmp/hickit-state-machine.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. tests/test_softall_hardpc_state_machine.c $(BLIND_COMMON_OBJS) -o "$$temporary_root/test.bin" $(LDFLAGS) $(LIBS)
	"$$temporary_root/test.bin"

test_softall_hardpc_multires: run_blind_p9016_minimal.bin run_softall_hardpc_positive_control.bin
	bash tests/test_softall_hardpc_multires_runner.sh

test_p9016_optimizer_trace: run_blind_p9016_minimal.bin
	bash tests/test_p9016_optimizer_trace.sh

run_blind_p9016_minimal.bin: run_blind_p9016_minimal.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) run_blind_p9016_minimal.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

run_softall_hardpc_positive_control.bin: run_blind_p9016_minimal.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) -DHK_P9016_HARDPC_POSITIVE_CONTROL=1 $(INCLUDES) run_blind_p9016_minimal.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

run_blind_p9016_fixed_posterior_graph_ablation.bin: run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

run_blind_p9016_force_causal_ablation.bin: run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) -DHK_WAVE3_FORCE_CAUSAL_ABLATION=1 $(INCLUDES) run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_force_causal_ablation: run_blind_p9016_minimal.bin run_blind_p9016_force_causal_ablation.bin tests/test_force_causal_ablation.sh testdata/p9016_blind_smoke.pairs
	bash tests/test_force_causal_ablation.sh

run_blind_p9016_optimizer_stability_control.bin: run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) -DHK_WAVE3_OPTIMIZER_STABILITY_CONTROL=1 $(INCLUDES) run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

run_blind_p9016_optimizer_stability_control_overshoot_test.bin: run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) -DHK_WAVE3_OPTIMIZER_STABILITY_CONTROL=1 -DHK_WAVE3_STABILITY_STEP=1.0f -DHK_WAVE3_STABILITY_STEPS=1 $(INCLUDES) run_blind_p9016_fixed_posterior_graph_ablation.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_optimizer_stability_control: run_blind_p9016_minimal.bin run_blind_p9016_optimizer_stability_control.bin run_blind_p9016_optimizer_stability_control_overshoot_test.bin tests/test_optimizer_stability_control.sh testdata/p9016_blind_smoke.pairs
	bash tests/test_optimizer_stability_control.sh ./run_blind_p9016_minimal.bin ./run_blind_p9016_optimizer_stability_control.bin ./run_blind_p9016_optimizer_stability_control_overshoot_test.bin

test_blind_gradient_units.bin: test_blind_gradient_units.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h fdg_gpu.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) test_blind_gradient_units.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_blind_gradient_units: test_blind_gradient_units.bin
	./test_blind_gradient_units.bin

test_blind_gradient_units_gpu: test_blind_gradient_units.bin
	./test_blind_gradient_units.bin --gpu

test_blind_estep_scores.bin: test_blind_estep_scores.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) test_blind_estep_scores.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_blind_estep_scores: test_blind_estep_scores.bin
	./test_blind_estep_scores.bin

test_blind_relax_step_policy.bin: test_blind_relax_step_policy.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h fdg_gpu.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) test_blind_relax_step_policy.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_blind_relax_step_policy: test_blind_relax_step_policy.bin
	./test_blind_relax_step_policy.bin

test_blind_relax_step_policy_gpu: test_blind_relax_step_policy.bin
	./test_blind_relax_step_policy.bin --gpu

test_blind_endpoint_backtracking.bin: test_blind_endpoint_backtracking.c $(BLIND_COMMON_OBJS) $(FDG_GPU_STAMP) hickit.h hkpriv.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) test_blind_endpoint_backtracking.c $(BLIND_COMMON_OBJS) -o $@ $(LDFLAGS) $(LIBS)

test_blind_endpoint_backtracking: test_blind_endpoint_backtracking.bin
	./test_blind_endpoint_backtracking.bin

test_p9016_endpoint_backtracking: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin tests/test_p9016_endpoint_backtracking.sh testdata/p9016_blind_smoke.pairs
	bash tests/test_p9016_endpoint_backtracking.sh ./run_blind_p9016_minimal.bin ./audit_blind_p9016_full_cpu_output.bin ./testdata/p9016_blind_smoke.pairs

test_p9016_endpoint_backtracking_gpu: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin tests/test_p9016_endpoint_backtracking.sh testdata/p9016_blind_smoke.pairs
	HK_TEST_ENDPOINT_BACKTRACKING_GPU=1 bash tests/test_p9016_endpoint_backtracking.sh ./run_blind_p9016_minimal.bin ./audit_blind_p9016_full_cpu_output.bin ./testdata/p9016_blind_smoke.pairs

test_p9016_force_mode: run_blind_p9016_minimal.bin tests/test_p9016_force_mode.sh testdata/p9016_blind_smoke.pairs
	bash tests/test_p9016_force_mode.sh ./run_blind_p9016_minimal.bin ./testdata/p9016_blind_smoke.pairs

test_p9016_force_mode_gpu: run_blind_p9016_minimal.bin tests/test_p9016_force_mode.sh testdata/p9016_blind_smoke.pairs
	HK_TEST_P9016_FORCE_MODE_GPU=1 bash tests/test_p9016_force_mode.sh ./run_blind_p9016_minimal.bin ./testdata/p9016_blind_smoke.pairs

test_p9016_fixed_source_coords: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin tests/test_p9016_fixed_source_coords.sh testdata/p9016_blind_smoke.pairs
	bash tests/test_p9016_fixed_source_coords.sh ./run_blind_p9016_minimal.bin ./testdata/p9016_blind_smoke.pairs ./audit_blind_p9016_full_cpu_output.bin

smoke_blind_estep_modes: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs testdata/p9016_resolution_mapping_fixture.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke
	mkdir -p $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke
	! env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_CIS_RATE_MODE=fixed_alpha \
	HK_BLIND_P9016_CIS_FIXED_ALPHA=0.2 \
	./run_blind_p9016_minimal.bin || exit 1
	! env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_ESTEP_SCORE_MODE=monotonic_powerlaw \
	HK_BLIND_P9016_TRANS_CHR_PAIR_PRIOR_LAMBDA=0.1 \
	./run_blind_p9016_minimal.bin || exit 1
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_explicit_heldout
	! env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_explicit_heldout \
	HK_BLIND_P9016_ESTEP_SCORE_MODE=monotonic_powerlaw \
	HK_BLIND_P9016_HELDOUT_FRACTION=0.25 \
	./run_blind_p9016_minimal.bin || exit 1
	test ! -e $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_explicit_heldout
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_resolution_mapping_fixture.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/legacy_heldout \
	HK_BLIND_P9016_CONFIG_NAME=legacy_internal_heldout \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=0 \
	HK_BLIND_P9016_ALLOW_LEGACY_INTERNAL_HELDOUT=1 \
	HK_BLIND_P9016_HELDOUT_FRACTION=0.25 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/legacy_heldout/legacy_internal_heldout
	awk -F'\t' '$$1=="legacy_estep_reproducibility_mode" && $$2=="1" { legacy=1 } $$1=="heldout_enabled" && $$2=="1" { enabled=1 } $$1=="heldout_n_raw_heldout" && ($$2+0)>0 { heldout=1 } END { exit legacy&&enabled&&heldout?0:1 }' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/legacy_heldout/legacy_internal_heldout/p9016_full.manifest.tsv
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_approved_fixed
	! env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_FORMAL_PRE_SPLIT=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_approved_fixed \
	HK_BLIND_P9016_CIS_RATE_MODE=fixed_alpha \
	HK_BLIND_P9016_CIS_FIXED_ALPHA=0.2 \
	HK_BLIND_P9016_CIS_FIXED_ALPHA_SENSITIVITY_ONLY=1 \
	./run_blind_p9016_minimal.bin 2>$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_approved_fixed.stderr || exit 1
	grep -Fq 'fixed_alpha is simulation/sensitivity only and is forbidden for a source-approved P9016 training input' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_approved_fixed.stderr
	test ! -e $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_approved_fixed
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_external_source_fixed
	printf '{}\n' > $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/policy_only_split_manifest.json
	! env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_SPLIT_MANIFEST_PATH=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/policy_only_split_manifest.json \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_external_source_fixed \
	HK_BLIND_P9016_CIS_RATE_MODE=fixed_alpha \
	HK_BLIND_P9016_CIS_FIXED_ALPHA=0.2 \
	HK_BLIND_P9016_CIS_FIXED_ALPHA_SENSITIVITY_ONLY=1 \
	./run_blind_p9016_minimal.bin 2>$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_external_source_fixed.stderr || exit 1
	grep -Fq 'fixed_alpha is simulation/sensitivity only and is forbidden for a source-approved P9016 training input' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_external_source_fixed.stderr
	test ! -e $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/rejected_external_source_fixed
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/uniform \
	HK_BLIND_P9016_CONFIG_NAME=explicit_uniform_logdist2 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=0 \
	HK_BLIND_P9016_ESTEP_SCORE_MODE=logdist2 \
	HK_BLIND_P9016_CIS_RATE_MODE=uniform \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/uniform/explicit_uniform_logdist2
	awk -F'\t' '$$1=="estep_score_mode" && $$2=="logdist2" { score=1 } $$1=="estep_prior_semantics" && $$2=="explicit_rate_score" { semantics=1 } $$1=="cis_rate_mode" && $$2=="uniform" { rate=1 } $$1=="cis_real_log_norm_target" && $$2=="log(S_cis)" { cis=1 } $$1=="trans_real_log_norm_target" && $$2=="log(S_trans)" { trans=1 } END { exit score&&semantics&&rate&&cis&&trans?0:1 }' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/uniform/explicit_uniform_logdist2/p9016_full.manifest.tsv
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/blind \
	HK_BLIND_P9016_CONFIG_NAME=blind_inter_ratio_powerlaw \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=0 \
	HK_BLIND_P9016_ESTEP_SCORE_MODE=monotonic_powerlaw \
	HK_BLIND_P9016_CIS_RATE_MODE=blind_inter_ratio \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/blind/blind_inter_ratio_powerlaw
	awk -F'\t' '$$1=="estep_score_mode" && $$2=="monotonic_powerlaw" { score=1 } $$1=="estep_prior_semantics" && $$2=="explicit_rate_score" { semantics=1 } $$1=="cis_rate_mode" && $$2=="blind_inter_ratio" { rate=1 } $$1=="cis_alpha_estimation_method" && $$2=="training_raw_density_generative_model" { method=1 } $$1=="cis_alpha_formula" && $$2=="alpha_g=lambda_cross/lambda_same(g)=r_g/(2-r_g)" { formula=1 } $$1=="cis_alpha_uses_heldout" && $$2=="0" { heldout=1 } $$1=="trans_real_log_norm_target" && $$2=="log(S_trans)" { trans=1 } END { exit score&&semantics&&rate&&method&&formula&&heldout&&trans?0:1 }' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/blind/blind_inter_ratio_powerlaw/p9016_full.manifest.tsv
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/fixed \
	HK_BLIND_P9016_CONFIG_NAME=fixed_alpha_contact_intensity \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=0 \
	HK_BLIND_P9016_ESTEP_SCORE_MODE=contact_log_intensity \
	HK_BLIND_P9016_CIS_RATE_MODE=fixed_alpha \
	HK_BLIND_P9016_CIS_FIXED_ALPHA=0.2 \
	HK_BLIND_P9016_CIS_FIXED_ALPHA_SENSITIVITY_ONLY=1 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/fixed/fixed_alpha_contact_intensity
	awk -F'\t' '$$1=="estep_score_mode" && $$2=="monotonic_powerlaw" { score=1 } $$1=="estep_score_mode_input_string" && $$2=="contact_log_intensity" { alias=1 } $$1=="cis_rate_mode" && $$2=="fixed_alpha" { rate=1 } $$1=="cis_fixed_alpha_scope" && $$2=="simulation_sensitivity_only" { scope=1 } END { exit score&&alias&&rate&&scope?0:1 }' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/fixed/fixed_alpha_contact_intensity/p9016_full.manifest.tsv
	mkdir -p $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/fixed/fixed_alpha_contact_intensity $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control)
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/missing_formula
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/missing_formula)
	sed -i '/^cis_alpha_formula\t/d' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/missing_formula/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/missing_formula || exit 1
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_semantics
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_semantics)
	sed -i 's/^estep_prior_semantics\t.*/estep_prior_semantics\tlegacy_normalized_prior/' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_semantics/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_semantics || exit 1
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_trans_prior_lambda
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_trans_prior_lambda)
	sed -i 's/^trans_chr_pair_prior_lambda\t.*/trans_chr_pair_prior_lambda\t0.25/' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_trans_prior_lambda/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_trans_prior_lambda || exit 1
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/blind/blind_inter_ratio_powerlaw $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_formula
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_formula)
	sed -i 's|^cis_alpha_formula\t.*|cis_alpha_formula\talpha_g=r_g|' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_formula/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_formula || exit 1
	cp -R $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/control $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_source_approval
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_source_approval)
	sed -i 's/^approved_p9016_source_pairs_realpath\t0/approved_p9016_source_pairs_realpath\t1/; s/^approved_p9016_raw_pairs_realpath\t0/approved_p9016_raw_pairs_realpath\t1/' $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_source_approval/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke/audit_negative/tampered_source_approval || exit 1
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_estep_modes_smoke

test_graph_diagnostics: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(GRAPH_DIAG_SMOKE_ROOT)
	mkdir -p $(GRAPH_DIAG_SMOKE_ROOT)
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(GRAPH_DIAG_SMOKE_ROOT)/off \
	HK_BLIND_P9016_CONFIG_NAME=graph_filter_off \
	HK_BLIND_P9016_BIN_SIZE_BP=1000000 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=0 \
	HK_BLIND_P9016_INIT_EPS=0.5 \
	HK_BLIND_P9016_INIT_NOISE_SCALE=0.0 \
	HK_BLIND_P9016_MIN_SEP_UNIT=0.0 \
	HK_BLIND_P9016_LAMBDA_SEP=0.0 \
	HK_BLIND_P9016_SOFTALL_FILTER_MODE=off \
	HK_BLIND_P9016_REPULSION_POLICY=independent \
	./run_blind_p9016_minimal.bin
	awk -F'\t' '$$1=="softall_filter_mode" && $$2=="off" { found=1 } END { exit found?0:1 }' $(GRAPH_DIAG_SMOKE_ROOT)/off/graph_filter_off/p9016_full.manifest.tsv
	awk -F'\t' '$$1=="repulsion_policy" && $$2=="independent" { found=1 } END { exit found?0:1 }' $(GRAPH_DIAG_SMOKE_ROOT)/off/graph_filter_off/p9016_full.manifest.tsv
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/off/graph_filter_off
	awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} $$1=="total"{ok=($$(h["candidate_raw_rows"])+0)>0 && ($$(h["candidate_state_rows"])+0)>($$(h["candidate_raw_rows"])+0) && ($$(h["filter2_state_retained_frac"])+0)==1 && ($$(h["filter2_posterior_mass_retained_frac"])+0)==1} END{exit ok?0:1}' $(GRAPH_DIAG_SMOKE_ROOT)/off/graph_filter_off/p9016_full.softall_graph_diag.tsv
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(GRAPH_DIAG_SMOKE_ROOT)/raw_support \
	HK_BLIND_P9016_CONFIG_NAME=graph_raw_support \
	HK_BLIND_P9016_BIN_SIZE_BP=1000000 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=1 \
	HK_BLIND_P9016_INIT_MODE=random_diploid \
	HK_BLIND_P9016_INIT_SCALE=0.1 \
	HK_BLIND_P9016_INIT_EPS=0.5 \
	HK_BLIND_P9016_INIT_NOISE_SCALE=0.0 \
	HK_BLIND_P9016_MIN_SEP_UNIT=0.0 \
	HK_BLIND_P9016_LAMBDA_SEP=0.0 \
	HK_BLIND_P9016_RHO_TRAIN_START=0.0 \
	HK_BLIND_P9016_RHO_TRAIN_END=0.0 \
	HK_BLIND_P9016_SOFTALL_FILTER_MODE=raw_support_filter \
	HK_BLIND_P9016_REPULSION_POLICY=expected \
	HK_BLIND_P9016_REPULSION_BLOCK_K_MIN=0.01 \
	./run_blind_p9016_minimal.bin
	awk -F'\t' '$$1=="softall_filter_mode" && $$2=="raw_support_filter" { found=1 } END { exit found?0:1 }' $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support/p9016_full.manifest.tsv
	awk -F'\t' '$$1=="repulsion_policy" && $$2=="expected" { policy=1 } $$1=="repulsion_blocking_mode" && $$2=="posterior_weighted_keep_v2" { mode=1 } $$1=="repulsion_contact_probability_source" && $$2=="raw_estep_posterior_p4_before_training_graph_modifiers" { source=1 } $$1=="repulsion_contact_keep_semantics" && $$2=="product_over_contributions_one_minus_q" { keep=1 } $$1=="repulsion_weight_aggregation" && $$2=="occupancy_union_direct_keep_product" { aggregate=1 } $$1=="repulsion_keep_storage" && $$2=="direct_double_keep_weight" { storage=1 } $$1=="repulsion_block_k_min_semantics" && $$2=="deprecated_ignored_compatibility_field" { kmin=1 } END { exit policy&&mode&&source&&keep&&aggregate&&storage&&kmin?0:1 }' $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support/p9016_full.manifest.tsv
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support
	awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} NR==2{considered=$$(h["final_n_repulsion_pairs_considered"])+0; blocked=$$(h["final_n_repulsion_pairs_blocked"])+0; partial=$$(h["final_n_repulsion_pairs_partial"])+0; active=$$(h["final_n_repulsion_pairs_active"])+0; sum_k=$$(h["final_sum_wedge_k"])+0; ok=(considered>0 && blocked+active==considered && partial>0 && sum_k==0)} END{exit ok?0:1}' $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support/p9016_full.loop_diag.tsv
	awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} $$1=="total"{ok=($$(h["candidate_raw_rows"])+0)>0 && ($$(h["filter2_raw_rows"])+0)<=($$(h["candidate_raw_rows"])+0) && ($$(h["filter2_state_rows"])+0)<=($$(h["candidate_state_rows"])+0) && ($$(h["filter2_posterior_mass_retained_frac"])+0)<=1.0000001} END{exit ok?0:1}' $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support/p9016_full.softall_graph_diag.tsv
	cp -R $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support $(GRAPH_DIAG_SMOKE_ROOT)/tampered_binary_semantics
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(GRAPH_DIAG_SMOKE_ROOT)/tampered_binary_semantics)
	sed -i 's/^repulsion_contact_keep_semantics\t.*/repulsion_contact_keep_semantics\thard_zero_for_positive_k_contact_edges/' $(GRAPH_DIAG_SMOKE_ROOT)/tampered_binary_semantics/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/tampered_binary_semantics || exit 1
	cp -R $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics)
	awk -F'\t' -v OFS='\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} NR==2{$$(h["final_n_repulsion_pairs_partial"])=$$(h["final_n_repulsion_pairs_active"])+1} {print}' $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics/p9016_full.loop_diag.tsv > $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics/loop.tmp
	mv $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics/loop.tmp $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics/p9016_full.loop_diag.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/tampered_partial_diagnostics || exit 1
	cp -R $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high)
	awk -F'\t' -v OFS='\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} NR==2{$$(h["final_repulsion_keep_weight_sum"])=$$(h["final_n_repulsion_pairs_active"])+1} {print}' $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high/p9016_full.loop_diag.tsv > $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high/loop.tmp
	mv $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high/loop.tmp $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high/p9016_full.loop_diag.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/tampered_keep_sum_high || exit 1
	cp -R $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep)
	awk -F'\t' -v OFS='\t' 'NR==1{for(i=1;i<=NF;i++) h[$$i]=i} NR==2{$$(h["final_n_repulsion_pairs_partial"])=0} {print}' $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep/p9016_full.loop_diag.tsv > $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep/loop.tmp
	mv $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep/loop.tmp $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep/p9016_full.loop_diag.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/tampered_zero_partial_keep || exit 1
	cp -R $(GRAPH_DIAG_SMOKE_ROOT)/raw_support/graph_raw_support $(GRAPH_DIAG_SMOKE_ROOT)/tampered_independent_partial
	$(call REBASE_AUDIT_FIXTURE_PATHS,$(GRAPH_DIAG_SMOKE_ROOT)/tampered_independent_partial)
	sed -i 's/^repulsion_policy\t.*/repulsion_policy\tindependent/; s/^repulsion_contact_keep_semantics\t.*/repulsion_contact_keep_semantics\tone_ignore_contact_posterior/' $(GRAPH_DIAG_SMOKE_ROOT)/tampered_independent_partial/p9016_full.manifest.tsv
	! HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(GRAPH_DIAG_SMOKE_ROOT)/tampered_independent_partial || exit 1
	rm -rf $(GRAPH_DIAG_SMOKE_ROOT)

test_fixed_posterior_graph_ablation: run_blind_p9016_minimal.bin run_blind_p9016_fixed_posterior_graph_ablation.bin tests/test_fixed_posterior_graph_ablation.sh
	bash tests/test_fixed_posterior_graph_ablation.sh ./run_blind_p9016_minimal.bin ./run_blind_p9016_fixed_posterior_graph_ablation.bin

smoke_blind_p9016_minimal: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(P9016_MINIMAL_SMOKE_ROOT)
	mkdir -p $(SMOKE_RES_ROOT)
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(P9016_MINIMAL_SMOKE_ROOT) \
	HK_BLIND_P9016_CONFIG_NAME=minimal_soft_sep_off \
	HK_BLIND_P9016_BIN_SIZE_BP=1000000 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=1 \
	HK_BLIND_P9016_INIT_EPS=0.5 \
	HK_BLIND_P9016_INIT_NOISE_SCALE=0.0 \
	HK_BLIND_P9016_MIN_SEP_UNIT=0.0 \
	HK_BLIND_P9016_LAMBDA_SEP=0.0 \
	HK_BLIND_P9016_INIT_SEED=17 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_MINIMAL_SMOKE_ROOT)/minimal_soft_sep_off
	awk -F'\t' '$$1=="estep_prior_semantics" && $$2=="legacy_normalized_prior" { semantics=1 } $$1=="legacy_estep_reproducibility_mode" && $$2=="1" { legacy=1 } END { exit semantics&&legacy?0:1 }' $(P9016_MINIMAL_SMOKE_ROOT)/minimal_soft_sep_off/p9016_full.manifest.tsv

smoke_blind_p9016_sep_on: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(P9016_SEP_ON_SMOKE_ROOT)
	mkdir -p $(SMOKE_RES_ROOT)
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(P9016_SEP_ON_SMOKE_ROOT) \
	HK_BLIND_P9016_CONFIG_NAME=minimal_soft_sep_on_smoke \
	HK_BLIND_P9016_BIN_SIZE_BP=1000000 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=1 \
	HK_BLIND_P9016_INIT_EPS=1.0 \
	HK_BLIND_P9016_INIT_NOISE_SCALE=0.05 \
	HK_BLIND_P9016_MIN_SEP_UNIT=2.0 \
	HK_BLIND_P9016_LAMBDA_SEP=0.05 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_SEP_ON_SMOKE_ROOT)/minimal_soft_sep_on_smoke
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="min_sep_unit" && abs(($$2+0)-2.0)<1e-6 { found=1 } END { exit found?0:1 }' $(P9016_SEP_ON_SMOKE_ROOT)/minimal_soft_sep_on_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="lambda_sep" && abs(($$2+0)-0.05)<1e-6 { found=1 } END { exit found?0:1 }' $(P9016_SEP_ON_SMOKE_ROOT)/minimal_soft_sep_on_smoke/p9016_full.manifest.tsv

smoke_blind_p9016_expected_sep: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(P9016_EXPECTED_SEP_SMOKE_ROOT)
	mkdir -p $(SMOKE_RES_ROOT)
	env -i PATH="$$PATH" LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(P9016_EXPECTED_SEP_SMOKE_ROOT) \
	HK_BLIND_P9016_CONFIG_NAME=scaffold_expected_msep1p5_lsep0p5_smoke \
	HK_BLIND_P9016_INIT_MODE=unphased_scaffold_split \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=1 \
	HK_BLIND_P9016_INIT_EPS=1.0 \
	HK_BLIND_P9016_INIT_NOISE_SCALE=0.05 \
	HK_BLIND_P9016_MIN_SEP_UNIT=1.5 \
	HK_BLIND_P9016_LAMBDA_SEP=0.5 \
	HK_BLIND_P9016_D_SCALE_MODE=expected_count \
	HK_BLIND_P9016_D_SCALE_EPS_COUNT=0.001 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke
	awk -F'\t' '$$1=="d_scale_mode" && $$2=="posterior_count" { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' '$$1=="d_scale_mode_input_string" && $$2=="expected_count" { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' '$$1=="legacy_expected_count_alias_used" && $$2=="1" { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="d_scale_eps_count" && abs(($$2+0)-0.001)<1e-8 { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="min_sep_unit" && abs(($$2+0)-1.5)<1e-6 { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="lambda_sep" && abs(($$2+0)-0.5)<1e-6 { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' '$$1=="init_mode" && $$2=="unphased_scaffold_split" { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="init_eps_effective" && ($$2+0)>0.0 { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	awk -F'\t' 'function abs(x){return x<0?-x:x} $$1=="init_noise_scale_effective" && abs(($$2+0)-0.05)<1e-6 { found=1 } END { exit found?0:1 }' $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.manifest.tsv
	test -s $(P9016_EXPECTED_SEP_SMOKE_ROOT)/scaffold_expected_msep1p5_lsep0p5_smoke/p9016_full.sep_diag.tsv

smoke_blind_p9016_resolution_chain: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(P9016_RESOLUTION_CHAIN_SMOKE_ROOT)
	mkdir -p $(SMOKE_RES_ROOT)
	HK_BLIND_P9016_PAIRS=testdata/p9016_blind_smoke.pairs \
	HK_BLIND_P9016_ALLOW_CUSTOM_PAIRS=1 \
	HK_BLIND_P9016_OUTPUT_ROOT=$(P9016_RESOLUTION_CHAIN_SMOKE_ROOT) \
	HK_BLIND_P9016_CHAIN=1 \
	HK_BLIND_P9016_RESOLUTION_CHAIN=4000000,1000000,200000 \
	HK_BLIND_P9016_MINIMAL_N_ITER=1 \
	HK_BLIND_P9016_MINIMAL_RELAX_STEPS=1 \
	./run_blind_p9016_minimal.bin
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_RESOLUTION_CHAIN_SMOKE_ROOT)/4m/minimal_soft_sep_off
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_RESOLUTION_CHAIN_SMOKE_ROOT)/1m/minimal_soft_sep_off
	HK_BLIND_AUDIT_ALLOW_CUSTOM_RAW_PAIRS=1 ./audit_blind_p9016_full_cpu_output.bin $(P9016_RESOLUTION_CHAIN_SMOKE_ROOT)/200k/minimal_soft_sep_off

smoke_blind_resolution_ploidy: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_resolution_mapping_fixture.pairs testdata/p9016_male_xy_ploidy_fixture.pairs testdata/p9016_female_x_ploidy_fixture.pairs scripts/check_blind_resolution_ploidy.py
	temporary_root=$$(mktemp -d /tmp/hickit-smoke.XXXXXX)
	trap 'rm -rf "$$temporary_root"' EXIT
	rm -rf $(SMOKE_RES_ROOT)/hk_blind_resolution_ploidy_smoke
	mkdir -p $(SMOKE_RES_ROOT)
	python scripts/check_blind_resolution_ploidy.py \
		--runner ./run_blind_p9016_minimal.bin \
		--audit ./audit_blind_p9016_full_cpu_output.bin \
		--root $(SMOKE_RES_ROOT)/hk_blind_resolution_ploidy_smoke

test_audit_optimizer_trace_manifest: run_blind_p9016_minimal.bin audit_blind_p9016_full_cpu_output.bin testdata/p9016_blind_smoke.pairs tests/test_audit_optimizer_trace_manifest.sh
	bash tests/test_audit_optimizer_trace_manifest.sh ./run_blind_p9016_minimal.bin ./audit_blind_p9016_full_cpu_output.bin ./testdata/p9016_blind_smoke.pairs

audit_blind_p9016_full_cpu_output: audit_blind_p9016_full_cpu_output.bin

audit_blind_p9016_full_cpu_output.bin: audit_blind_p9016_full_cpu_output.c hickit.h krng.h
	$(CC) $(CFLAGS) $(ASAN_FLAG) $(CPPFLAGS) $(INCLUDES) audit_blind_p9016_full_cpu_output.c -o $@ $(AUDIT_LIBS)

clean:
	rm -f $(PROG) *.o *.bin .fdg_gpu_backend.*

depend:
	( LC_ALL=C ; export LC_ALL; makedepend -Y -- $(CFLAGS) $(CPPFLAGS) -- *.c ) 2>/dev/null
