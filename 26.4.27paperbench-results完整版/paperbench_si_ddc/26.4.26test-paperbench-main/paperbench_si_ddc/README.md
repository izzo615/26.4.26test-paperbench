# paperbench_si_ddc

This repository is an AutoDL-ready reproduction project for the paper
`Stochastic Interpolants with Data-Dependent Couplings`
([arXiv:2310.03725](https://arxiv.org/abs/2310.03725)).

The project is structured to satisfy the practical PaperBench-style
reproduction requirements for this paper:

- keep the paper's base network family fixed by using the U-Net from
  `lucidrains/denoising-diffusion-pytorch`
- train the ImageNet inpainting and super-resolution pipelines with the
  paper-aligned hyperparameters exposed in config
- export Figure 3 to Figure 7 style artifacts
- compute FID-50k and Inception Score after sampling
- generate a paper-comparison table and a PaperBench-style score report
- save all artifacts under `/AutoDL-pub` and pack them into a zip archive

The local machine can validate syntax and project wiring without GPU.
Full training, figure reproduction, and FID/IS evaluation are expected to run
later inside an AutoDL Linux GPU container.

## Paper Alignment

The implementation follows the PaperBench rubric snippets and the paper text:

- ImageNet data is accessed through Hugging Face `imagenet-1k`
- the U-Net comes from `denoising-diffusion-pytorch`
- the inpainting dependent coupling uses 64 equal tiles with tile probability
  `p = 0.3`
- all paper experiments default to batch size `32`
- all paper experiments default to `200000` gradient steps
- the inpainting interpolant uses `I_t = t x_0 + (1 - t) x_1`
- the time derivative uses `dI_t = x_1 - x_0`
- the dependent inpainting velocity is masked so observed pixels stay fixed
- the super-resolution pipeline uses a data-dependent low-resolution coupling
  and reports FID against train and validation splits

## Output Layout

The AutoDL run writes to:

```text
/AutoDL-pub/paperbench_si_ddc/
  archives/
  cache/
  checkpoints/
  collected_results/
    image/
      outputs_image/
        figure_3_inpainting_examples.png
        figure_4_super_resolution_examples.png
        figure_5_inpainting_trajectory.png
        figure_6_super_resolution_gallery.png
        figure_7_super_resolution_trajectory.png
  logs/
  metrics/
  reference_sets/
  reports/
```

The folder `collected_results/image/outputs_image` is always created so it
matches the expected persistent output convention.

## AutoDL One-Click Run

Inside the AutoDL Linux container:

```bash
cd /root/autodl-tmp/paperbench_si_ddc
bash reproduce.sh --config configs/autodl_full.json --stages all
```

The run performs:

1. dependency installation
2. ImageNet loader setup
3. training for the required paper experiments
4. Figure 3 to Figure 7 artifact export
5. FID/IS evaluation
6. paper-comparison table generation
7. PaperBench-style report generation
8. zip packaging of all saved results

## Local Offline Validation

Local validation avoids GPU execution and only checks syntax, config wiring,
and import-free project integrity:

```powershell
python .\scripts\local_smoke_test.py
```

## Main Commands

Full AutoDL pipeline:

```bash
python scripts/run_full_reproduction.py --config configs/autodl_full.json --stages all
```

Train only:

```bash
python scripts/run_full_reproduction.py --config configs/autodl_full.json --stages train
```

Sample and build figures from existing checkpoints:

```bash
python scripts/run_full_reproduction.py --config configs/autodl_full.json --stages sample,report
```

Metrics and packaging only:

```bash
python scripts/run_full_reproduction.py --config configs/autodl_full.json --stages evaluate,report,package
```

## Files

```text
paperbench_si_ddc/
  configs/
    autodl_full.json
    reference_metrics.json
  scripts/
    local_smoke_test.py
    package_results.py
    run_full_reproduction.py
    run_reproduction.py
  src/
    paperbench_si_ddc/
      config.py
      data.py
      evaluation.py
      interpolants.py
      modeling.py
      paths.py
      pipeline.py
      reporting.py
      utils.py
```

## Notes

- The repository does not ship pretrained weights.
- Exact figure and metric reproduction requires running the full GPU pipeline.
- If the paper PDF is available in the container, the report can attach it as
  provenance metadata, but the pipeline does not require internet access after
  the environment and dataset are prepared.
