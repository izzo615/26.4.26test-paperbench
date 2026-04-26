from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import pandas as pd
import torch
import torch.nn.functional as F

from paperbench_si_ddc.config import normalize_stage_argument
from paperbench_si_ddc.data import build_loader, load_imagenet_bundle, subset_dataset
from paperbench_si_ddc.evaluation import (
    build_metric_table,
    compute_torch_fidelity_metrics,
    export_reference_set,
    save_tensor_folder,
    write_metric_outputs,
)
from paperbench_si_ddc.interpolants import (
    make_inpainting_base,
    make_superres_base,
    project_observed_region,
    sample_linear_interpolant,
    sample_tile_mask,
)
from paperbench_si_ddc.modeling import build_unet, checkpoint_path, maybe_load_checkpoint, save_training_checkpoint
from paperbench_si_ddc.paths import ProjectPaths
from paperbench_si_ddc.reporting import build_artifact_manifest, compute_scorecard, package_results, write_report
from paperbench_si_ddc.utils import (
    build_class_channel,
    cycle,
    format_duration,
    now_seconds,
    save_image_grid,
    save_json,
    seed_everything,
    resolve_device,
    slice_indices,
)


@dataclass
class ExperimentArtifacts:
    checkpoints: dict[str, str]
    metric_rows: list[dict]
    metric_paths: dict
    report_paths: dict
    archive_path: str
    smoke_test_passed: bool


class ReproductionPipeline:
    def __init__(self, *, project_root: Path, config_path: Path, config: dict):
        self.project_root = project_root
        self.config_path = config_path
        self.config = config
        self.paths = ProjectPaths.from_config(project_root, config_path, config)
        self.device = resolve_device(config["runtime"]["device"])
        self.dataset_cache: dict[int, object] = {}
        self.smoke_test_passed = True
        seed_everything(int(config["seed"]))

    def run(self, stage_argument: str) -> None:
        stages = normalize_stage_argument(stage_argument)
        artifacts = ExperimentArtifacts(
            checkpoints={},
            metric_rows=[],
            metric_paths={},
            report_paths={},
            archive_path="",
            smoke_test_passed=self.smoke_test_passed,
        )

        if any(stage in stages for stage in ["train", "sample", "evaluate"]):
            artifacts.checkpoints = self._train_or_load_models(run_training="train" in stages)

        if "sample" in stages:
            self._render_required_figures()

        metrics_df = None
        if "evaluate" in stages:
            artifacts.metric_rows = self._evaluate_metrics()
            metrics_df = build_metric_table(artifacts.metric_rows, self.config.get("reference_metrics", {}))
            artifacts.metric_paths = write_metric_outputs(metrics_df, self.paths.metrics_root)

        if "report" in stages:
            scorecard = compute_scorecard(
                self.paths,
                self.config,
                artifacts.checkpoints,
                metrics_df,
                smoke_test_passed=self.smoke_test_passed,
            )
            artifacts.report_paths = write_report(
                self.paths,
                self.config,
                scorecard,
                artifacts.metric_paths,
                artifacts.checkpoints,
            )
            build_artifact_manifest(
                self.paths,
                self.config,
                artifacts.checkpoints,
                artifacts.report_paths,
                artifacts.metric_paths,
            )

        if "package" in stages:
            artifacts.archive_path = str(package_results(self.paths))

        save_json(
            {
                "device": str(self.device),
                "checkpoints": artifacts.checkpoints,
                "metric_paths": artifacts.metric_paths,
                "report_paths": artifacts.report_paths,
                "archive_path": artifacts.archive_path,
            },
            self.paths.report_root / "run_summary.json",
        )

    def _autocast_context(self):
        if self.device.type == "cuda" and bool(self.config["runtime"].get("amp", True)):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _build_dataset_bundle(self, resolution: int):
        if resolution not in self.dataset_cache:
            self.dataset_cache[resolution] = load_imagenet_bundle(self.config, image_size=resolution)
        return self.dataset_cache[resolution]

    def _experiment_input_channels(self, experiment: dict) -> int:
        task = experiment["task"]
        coupling = experiment["coupling"]
        if task == "inpainting" and coupling == "uncoupled":
            return 4
        if task == "inpainting" and coupling == "dependent":
            return 5
        if task == "super_resolution":
            return 7
        raise ValueError(f"Unsupported experiment shape for {experiment}")

    def _loader_batch_size(self) -> int:
        runtime_batch = int(self.config["runtime"].get("loader_batch_size", self.config["paper_hparams"]["batch_size"]))
        return max(1, runtime_batch)

    def _accumulation_steps(self) -> int:
        effective_batch = int(self.config["paper_hparams"]["batch_size"])
        loader_batch = self._loader_batch_size()
        return max(1, math.ceil(effective_batch / loader_batch))

    def _train_or_load_models(self, run_training: bool) -> dict[str, str]:
        checkpoints = {}
        for name, experiment in self.config["experiments"].items():
            if not experiment.get("enabled", False):
                continue
            ckpt = checkpoint_path(self.paths.checkpoint_root, experiment["checkpoint_name"])
            checkpoints[name] = str(ckpt)
            if run_training:
                self._train_single_experiment(name, experiment, ckpt)
        return checkpoints

    def _train_single_experiment(self, name: str, experiment: dict, ckpt_path: Path) -> None:
        resolution = int(experiment["resolution"])
        bundle = self._build_dataset_bundle(resolution)
        loader = build_loader(
            bundle.train,
            batch_size=self._loader_batch_size(),
            num_workers=int(self.config["runtime"]["num_workers"]),
            pin_memory=bool(self.config["runtime"]["pin_memory"]) and self.device.type == "cuda",
            shuffle=True,
            image_size=resolution,
        )
        iterator = cycle(loader)

        model = build_unet(
            input_channels=self._experiment_input_channels(experiment),
            output_channels=3,
            model_config=self.config["model"],
            resnet_block_groups=int(self.config["paper_hparams"]["resnet_block_groups"]),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.config["paper_hparams"]["learning_rate"]),
            weight_decay=float(self.config["paper_hparams"]["weight_decay"]),
        )
        start_step = maybe_load_checkpoint(ckpt_path, model, optimizer=optimizer)

        train_steps = int(self.config["paper_hparams"]["train_steps"])
        save_every = int(self.config["runtime"]["save_every"])
        log_every = int(self.config["runtime"]["log_every"])
        num_classes = bundle.num_classes
        grad_accum_steps = self._accumulation_steps()
        start_time = now_seconds()
        rows = []
        optimizer.zero_grad(set_to_none=True)

        for step in range(start_step + 1, train_steps + 1):
            model.train()
            total_loss = 0.0

            for _ in range(grad_accum_steps):
                x1, labels = next(iterator)
                x1 = x1.to(self.device, non_blocking=self.device.type == "cuda")
                labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)

                model_input, target, extra = self._prepare_training_batch(x1, labels, num_classes, experiment)
                t_values = extra["t_values"]

                with self._autocast_context():
                    prediction = model(model_input, t_values)
                    if "velocity_mask" in extra:
                        prediction = prediction * extra["velocity_mask"]
                    loss = F.mse_loss(prediction, target) / grad_accum_steps

                loss.backward()
                total_loss += float(loss.detach().cpu().item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0 or step == 1 or step == train_steps:
                rows.append(
                    {
                        "step": step,
                        "loss": total_loss,
                        "elapsed": format_duration(now_seconds() - start_time),
                    }
                )
                pd.DataFrame(rows).to_csv(self.paths.log_root / f"{name}_train_log.csv", index=False)

            if step % save_every == 0 or step == train_steps:
                save_training_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metadata={"experiment": name, "config_path": str(self.config_path)},
                )

    def _prepare_training_batch(self, x1: torch.Tensor, labels: torch.Tensor, num_classes: int, experiment: dict):
        height, width = x1.shape[-2:]
        class_channel = build_class_channel(labels, height, width, num_classes)
        batch_size = x1.shape[0]
        t_values = torch.rand((batch_size,), device=x1.device, dtype=x1.dtype)
        extra = {"t_values": t_values}

        if experiment["task"] == "inpainting":
            if experiment["coupling"] == "uncoupled":
                x0 = torch.randn_like(x1)
                xt, target = sample_linear_interpolant(x0, x1, t_values)
                return torch.cat([xt, class_channel], dim=1), target, extra

            observed_mask = sample_tile_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                tile_grid=int(self.config["paper_hparams"]["tile_grid"]),
                keep_probability=float(self.config["paper_hparams"]["tile_keep_probability"]),
                device=x1.device,
            ).to(dtype=x1.dtype)
            x0 = make_inpainting_base(x1, observed_mask)
            xt, target = sample_linear_interpolant(x0, x1, t_values)
            extra["velocity_mask"] = 1.0 - observed_mask
            return torch.cat([xt, class_channel, observed_mask], dim=1), target, extra

        if experiment["task"] == "super_resolution":
            x0, lowres = make_superres_base(
                x1,
                low_resolution=int(experiment["condition_resolution"]),
                high_resolution=height,
            )
            xt, target = sample_linear_interpolant(x0, x1, t_values)
            return torch.cat([xt, class_channel, lowres], dim=1), target, extra

        raise ValueError(f"Unsupported task: {experiment['task']}")

    def _load_model_for_inference(self, name: str, experiment: dict) -> tuple[torch.nn.Module, int]:
        bundle = self._build_dataset_bundle(int(experiment["resolution"]))
        model = build_unet(
            input_channels=self._experiment_input_channels(experiment),
            output_channels=3,
            model_config=self.config["model"],
            resnet_block_groups=int(self.config["paper_hparams"]["resnet_block_groups"]),
        ).to(self.device)
        ckpt = checkpoint_path(self.paths.checkpoint_root, experiment["checkpoint_name"])
        step = maybe_load_checkpoint(ckpt, model)
        if step <= 0:
            raise FileNotFoundError(
                f"Checkpoint for experiment '{name}' was not found at {ckpt}. Run the train stage first."
            )
        model.eval()
        return model, bundle.num_classes

    @torch.no_grad()
    def _sample_sequence(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        experiment: dict,
        slice_count: int,
    ):
        model.eval()
        batch_size = images.shape[0]
        height, width = images.shape[-2:]
        class_channel = build_class_channel(labels, height, width, num_classes).to(images.dtype)
        steps = int(self.config["paper_hparams"]["sample_steps"])
        slices = []

        if experiment["task"] == "inpainting":
            observed_mask = sample_tile_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                tile_grid=int(self.config["paper_hparams"]["tile_grid"]),
                keep_probability=float(self.config["paper_hparams"]["tile_keep_probability"]),
                device=images.device,
            ).to(dtype=images.dtype)
            if experiment["coupling"] == "dependent":
                current = make_inpainting_base(images, observed_mask)
                conditioning = torch.cat([class_channel, observed_mask], dim=1)
            else:
                current = torch.randn_like(images)
                conditioning = class_channel
            observed_pixels = observed_mask * images
            velocity_mask = 1.0 - observed_mask
        else:
            current, lowres = make_superres_base(
                images,
                low_resolution=int(experiment["condition_resolution"]),
                high_resolution=height,
            )
            conditioning = torch.cat([class_channel, lowres], dim=1)
            observed_pixels = None
            observed_mask = None
            velocity_mask = None

        capture_steps = set(slice_indices(steps, slice_count))
        if 0 in capture_steps:
            slices.append(current.detach().cpu())

        for step in range(1, steps + 1):
            t_value = torch.full((batch_size,), float(step - 1) / float(steps), device=images.device, dtype=images.dtype)
            model_input = torch.cat([current, conditioning], dim=1)
            velocity = model(model_input, t_value)
            if velocity_mask is not None:
                velocity = velocity * velocity_mask
            current = current + velocity / float(steps)
            if observed_pixels is not None and observed_mask is not None:
                current = project_observed_region(current, observed_pixels, observed_mask)
            if step in capture_steps:
                slices.append(current.detach().cpu())

        context = {
            "observed_mask": None if observed_mask is None else observed_mask.detach().cpu(),
            "conditioning_preview": None,
        }
        if experiment["task"] == "super_resolution":
            context["conditioning_preview"] = lowres.detach().cpu()
        else:
            context["conditioning_preview"] = project_observed_region(
                torch.zeros_like(images), observed_pixels, observed_mask
            ).detach().cpu()

        return current.detach().cpu(), slices, context

    def _render_required_figures(self) -> None:
        examples3 = int(self.config["figures"]["figure3_examples"])
        examples4 = int(self.config["figures"]["figure4_examples"])
        examples6 = int(self.config["figures"]["figure6_examples"])
        slices5 = int(self.config["figures"]["figure5_slices"])
        slices7 = int(self.config["figures"]["figure7_slices"])
        resolution = int(self.config["experiments"]["inpainting_uncoupled"]["resolution"])

        inpaint_bundle = self._build_dataset_bundle(resolution)
        valid_subset = subset_dataset(inpaint_bundle.valid, max(examples3, examples6))
        valid_loader = build_loader(valid_subset, batch_size=max(examples3, examples6), num_workers=0, pin_memory=False, shuffle=False, image_size=resolution)
        batch_images, batch_labels = next(iter(valid_loader))
        batch_images = batch_images.to(self.device)
        batch_labels = torch.as_tensor(batch_labels, device=self.device, dtype=torch.long)

        uncoupled_model, num_classes = self._load_model_for_inference("inpainting_uncoupled", self.config["experiments"]["inpainting_uncoupled"])
        dependent_model, _ = self._load_model_for_inference("inpainting_dependent", self.config["experiments"]["inpainting_dependent"])
        superres_model, _ = self._load_model_for_inference("superres_dependent", self.config["experiments"]["superres_dependent"])

        uncoupled_samples, _, _ = self._sample_sequence(
            uncoupled_model,
            batch_images[:examples3],
            batch_labels[:examples3],
            num_classes,
            self.config["experiments"]["inpainting_uncoupled"],
            slice_count=slices5,
        )
        dependent_samples, dep_slices, dep_ctx = self._sample_sequence(
            dependent_model,
            batch_images[:examples3],
            batch_labels[:examples3],
            num_classes,
            self.config["experiments"]["inpainting_dependent"],
            slice_count=slices5,
        )
        figure3 = torch.cat(
            [dep_ctx["conditioning_preview"], uncoupled_samples, dependent_samples, batch_images[:examples3].cpu()],
            dim=0,
        )
        save_image_grid(figure3, self.paths.image_root / "figure_3_inpainting_examples.png", nrow=examples3)

        save_image_grid(torch.cat(dep_slices, dim=0), self.paths.image_root / "figure_5_inpainting_trajectory.png", nrow=examples3)

        super_images = batch_images[:examples4]
        super_labels = batch_labels[:examples4]
        superres_samples, super_slices, super_ctx = self._sample_sequence(
            superres_model,
            super_images,
            super_labels,
            num_classes,
            self.config["experiments"]["superres_dependent"],
            slice_count=slices7,
        )
        figure4 = torch.cat([super_ctx["conditioning_preview"], superres_samples, super_images.cpu()], dim=0)
        save_image_grid(figure4, self.paths.image_root / "figure_4_super_resolution_examples.png", nrow=examples4)

        extra_super_images = batch_images[:examples6]
        extra_super_labels = batch_labels[:examples6]
        extra_superres, _, extra_ctx = self._sample_sequence(
            superres_model,
            extra_super_images,
            extra_super_labels,
            num_classes,
            self.config["experiments"]["superres_dependent"],
            slice_count=slices7,
        )
        figure6 = torch.cat([extra_ctx["conditioning_preview"], extra_superres, extra_super_images.cpu()], dim=0)
        save_image_grid(figure6, self.paths.image_root / "figure_6_super_resolution_gallery.png", nrow=examples6)

        save_image_grid(torch.cat(super_slices, dim=0), self.paths.image_root / "figure_7_super_resolution_trajectory.png", nrow=examples4)

    def _evaluate_metrics(self) -> list[dict]:
        rows = []
        metric_batch = int(self.config["metrics"]["batch_size"])
        target_count = int(self.config["metrics"]["num_generated_samples"])
        ref_count = int(self.config["metrics"]["num_reference_samples"])

        for name, experiment in self.config["experiments"].items():
            if not experiment.get("enabled", False):
                continue

            resolution = int(experiment["resolution"])
            bundle = self._build_dataset_bundle(resolution)
            valid_subset = subset_dataset(bundle.valid, target_count)
            loader = build_loader(valid_subset, batch_size=metric_batch, num_workers=0, pin_memory=False, shuffle=False, image_size=resolution)
            model, num_classes = self._load_model_for_inference(name, experiment)

            generated_folder = self.paths.metrics_root / f"{name}_generated"
            if generated_folder.exists():
                shutil.rmtree(generated_folder)
            generated_folder.mkdir(parents=True, exist_ok=True)

            index = 0
            for images, labels in loader:
                images = images.to(self.device)
                labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)
                generated, _slices, _ctx = self._sample_sequence(model, images, labels, num_classes, experiment, slice_count=2)
                index = save_tensor_folder(generated, generated_folder, start_index=index)
                if index >= target_count:
                    break

            valid_reference = self.paths.reference_root / f"{resolution}_valid"
            export_reference_set(bundle.valid, valid_reference, ref_count)

            if experiment["task"] == "super_resolution":
                train_reference = self.paths.reference_root / f"{resolution}_train"
                export_reference_set(bundle.train, train_reference, ref_count)
                for split_name, reference_root in [("train", train_reference), ("valid", valid_reference)]:
                    metrics = compute_torch_fidelity_metrics(generated_folder, reference_root, cuda=(self.device.type == "cuda"))
                    rows.append({"experiment": name, "split": split_name, **metrics})
            else:
                metrics = compute_torch_fidelity_metrics(generated_folder, valid_reference, cuda=(self.device.type == "cuda"))
                rows.append({"experiment": name, "split": "valid", **metrics})

        return rows
