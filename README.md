# context-switching-control

This directory is a minimal public release bundle for the project.

## Contents

### code/
- `analyze_infoI.py`
- `analyze_infoI_outcome.py`
- `collect_debug_rollouts.py`
- `collect_infoI_full.sh`
- `env_qtow.py`
- `eval.py`
- `format_infoI_main_table.py`
- `models.py`
- `run_paper_L_M_ab25_ba30.sh`
- `run_paper_M_mem16_32_ab25_ba30.sh`
- `run_paper_M_mem64_ab25_ba30.sh`
- `run_paper_ab25_ba30.sh`
- `run_switch_generalization.sh`
- `train.py`

### results/
- `paper_summaries/combined_summary_AB25_BA30_I_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_BA30_L_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_BA30_M_mem16_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_BA30_M_mem32_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_BA30_M_mem64_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_BA30_M_phase1only001_20runs.csv`
- `paper_summaries/combined_summary_AB25_I_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_AB25_L_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_AB25_M_mem16_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_AB25_M_mem32_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_AB25_M_mem64_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_AB25_M_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_I_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_L_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_M_mem16_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_M_mem32_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_M_mem64_phase1only001_10seeds.csv`
- `paper_summaries/combined_summary_BA30_M_phase1only001_10seeds.csv`
- `results/infoI_action_phase1_summary.csv`
- `results/infoI_full_all_summary.csv`
- `results/infoI_full_phase1_summary.csv`
- `results/infoI_goal3_phase1_summary.csv`
- `results/infoI_main_all_main_table.csv`
- `results/infoI_main_phase1_main_table.csv`
- `results/infoI_targethit_phase1_summary.csv`
- `switch_generalization_eval/combined_summary_switch_generalization.csv`
- `switch_generalization_eval/combined_switch_generalization_raw.csv`

## What is included

- Source code needed to reproduce the main analyses and figures.
- Result tables / figures selected for public release.
- A lightweight project structure suitable for copying into a Git repository.

## What is not included

- Training checkpoints
- Large raw run directories
- Cache files
- Temporary files
- Private notes or unpublished scratch files

## Reproduction notes

Please edit this section before publishing.

Suggested items to add:
1. Python version
2. Main dependencies
3. Training command
4. Evaluation command
5. Figure / table generation command

## Citation

Please edit this section before publishing.
