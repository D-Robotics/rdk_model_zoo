#!/usr/bin/env python3
"""Shared ONNX calibration and BPU compilation workflow for YOLO.

The X5 and RDK S toolchains have different compiler protocols.  They still
perform the same host-side work: inspect a static NCHW ONNX input, choose
calibration images, prepare RGB/NCHW tensors, write a compiler configuration,
run the tool, and move the generated artifact and log.  This module owns that
common workflow and leaves only those protocol differences in
``ToolchainProfile``.

The functions that create a :class:`ConversionPlan` and render its YAML are
deliberately independent of OpenCV, NumPy, ONNX Runtime, and OpenExplore.  A
host test can therefore check artifact names, calibration paths, and compiler
arguments without pretending that a conversion took place.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import logging
import numbers
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Optional, Sequence, Tuple


LOGGER = logging.getLogger("MZOO")
CALIBRATION_EXTENSIONS = (".jpg", ".jpeg", ".png")
FLOAT32 = "tensor(float)"


@dataclass(frozen=True)
class ToolchainProfile:
    """The target-specific part of the conversion protocol."""

    name: str
    march: str
    compiler_command: Tuple[str, ...]
    version_command: Tuple[str, ...]
    artifact_suffix: str
    calibration_suffix: str
    default_calibration_dir: str
    normalize_calibration: bool
    default_optimize_level: str
    optimize_choices: Tuple[str, ...]
    log_name: str
    x5_protocol: bool


@dataclass(frozen=True)
class OnnxInput:
    """The reviewed static input contract read from an ONNX model."""

    name: str
    data_type: str
    height: int
    width: int


@dataclass(frozen=True)
class ConversionPlan:
    """All paths and arguments for one conversion invocation."""

    toolchain: ToolchainProfile
    onnx_path: Path
    calibration_images: Path
    output_dir: Path
    workspace: Path
    calibration_dir: Path
    bpu_output_dir: Path
    config_path: Path
    artifact_path: Path
    log_path: Path
    model_prefix: str
    compiler_command: Tuple[str, ...]
    input: OnnxInput
    quantized: str
    jobs: int
    optimize_level: str
    sample_calibration: bool
    calibration_sample_num: int
    save_cache: bool
    overwrite: bool

    @property
    def calibration_suffix(self) -> str:
        """File suffix consumed by this target's compiler."""

        return self.toolchain.calibration_suffix

    @property
    def normalize_calibration(self) -> bool:
        """Whether calibration tensors are divided by 255."""

        return self.toolchain.normalize_calibration


def x5_toolchain() -> ToolchainProfile:
    """Return the RDK X5 ``hb_mapper`` protocol profile."""

    return ToolchainProfile(
        name="x5",
        march="bayes-e",
        compiler_command=(
            "hb_mapper", "makertbin", "--config", "config.yaml",
            "--model-type", "onnx",
        ),
        version_command=("hb_mapper", "--version"),
        artifact_suffix=".bin",
        calibration_suffix=".rgbchw",
        default_calibration_dir=".calibration_data_temporary_folder",
        normalize_calibration=False,
        default_optimize_level="O3",
        optimize_choices=("O0", "O1", "O2", "O3"),
        log_name="hb_mapper_makertbin.log",
        x5_protocol=True,
    )


def s_toolchain(march: str = "nash-e") -> ToolchainProfile:
    """Return an RDK S profile for S100, S100P, or S600.

    ``hb_compile`` uses the same calibration representation on all three
    chips; ``march`` selects the compiler architecture and output filename.
    """

    allowed = {"nash-e", "nash-m", "nash-p"}
    if march not in allowed:
        raise ValueError(
            f"unsupported RDK S march {march!r}; expected one of "
            f"{', '.join(sorted(allowed))}")
    return ToolchainProfile(
        name="s",
        march=march,
        compiler_command=("hb_compile", "--config", "config.yaml"),
        version_command=("hb_compile", "--help"),
        artifact_suffix=".hbm",
        calibration_suffix=".npy",
        default_calibration_dir=".calibration_data_temporary_folder",
        normalize_calibration=True,
        default_optimize_level="O2",
        optimize_choices=("O0", "O1", "O2"),
        log_name="hb_compile.log",
        x5_protocol=False,
    )


def export_defaults(platform: str, family: str = "yolo11") -> Tuple[int, bool]:
    """Return the reviewed ONNX export defaults for a target and family.

    The ONNX graph export is shared, but the supported toolchains use
    different opset defaults.  YOLO26's detector also uses simplification on
    X5; the generic Ultralytics exporter historically leaves simplification
    disabled for both families.  Callers can still pass explicit overrides.
    """

    if platform not in {"x5", "s100", "s100p", "s600"}:
        raise ValueError(f"unsupported export platform: {platform!r}")
    if family == "yolo26":
        return (11 if platform == "x5" else 19, platform == "x5")
    return (11 if platform == "x5" else 19, False)


def parse_bool(value):
    """Parse both legacy ``--flag true`` and convenient ``--flag`` forms."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (true/false), got {value!r}")


def build_mapper_parser(
    description: str,
    profile: ToolchainProfile,
    *,
    onnx_required: bool = False,
    include_march: bool = False,
    default_calibration_dir: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build a compatible parser for a generic or YOLO26 mapper."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cal-images", default="./cal_images", type=str,
        help="Directory containing JPG/PNG calibration images (20-50 recommended).",
    )
    parser.add_argument(
        "--onnx", default=None if onnx_required else "./yolo11n.onnx",
        required=onnx_required, type=str,
        help="Path to the source float32 ONNX model.",
    )
    parser.add_argument(
        "--output-dir", default=".", type=str,
        help="Directory for the converted .bin/.hbm artifact (default: ONNX directory).",
    )
    if include_march:
        parser.add_argument(
            "--march", default=profile.march,
            choices=("nash-e", "nash-m", "nash-p"),
            help="RDK S architecture: nash-e (S100), nash-m (S100P), or nash-p (S600).",
        )
    parser.add_argument(
        "--quantized", choices=("int8", "int16"), default="int8",
        help="Calibration precision policy.",
    )
    parser.add_argument(
        "--jobs", type=int, default=16,
        help="Number of compiler jobs.",
    )
    parser.add_argument(
        "--optimize-level", choices=profile.optimize_choices,
        default=profile.default_optimize_level,
        help="OpenExplore compiler optimization level.",
    )
    parser.add_argument(
        "--cal-sample", nargs="?", const=True, default=True, type=parse_bool,
        help="Sample at most --cal-sample-num images (default: true).",
    )
    parser.add_argument(
        "--cal-sample-num", type=int, default=20,
        help="Maximum sampled calibration images.",
    )
    parser.add_argument(
        "--save-cache", nargs="?", const=True, default=False, type=parse_bool,
        help="Keep the generated workspace and intermediate calibration files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing artifact with the same name.",
    )
    parser.add_argument(
        "--cal", default=default_calibration_dir or profile.default_calibration_dir,
        type=str, help="Calibration directory inside the temporary workspace.",
    )
    parser.add_argument(
        "--ws", default=".temporary_workspace", type=str,
        help="Temporary OpenExplore workspace.",
    )
    return parser


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a user path against the invocation directory."""

    root = Path.cwd() if base_dir is None else Path(base_dir)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def inspect_onnx(path: Path, ort_module=None) -> OnnxInput:
    """Read and validate the single static float input used by these mappers."""

    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    if ort_module is None:
        try:
            import onnxruntime as ort_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required in the OpenExplore conversion "
                "environment; dependencies are never installed automatically") from exc
    session = ort_module.InferenceSession(
        str(path), providers=["CPUExecutionProvider"])
    try:
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"model has {len(inputs)} inputs; exactly one is required")
        model_input = inputs[0]
        data_type = getattr(model_input, "type", FLOAT32)
        if data_type != FLOAT32:
            raise ValueError(
                f"model input type is {data_type!r}; expected {FLOAT32!r}")
        shape = tuple(getattr(model_input, "shape", ()))
        if len(shape) != 4:
            raise ValueError(f"model input shape {shape!r} is not NCHW rank 4")
        height, width = shape[2], shape[3]
        if not all(
            isinstance(value, numbers.Integral) and not isinstance(value, bool)
            and int(value) > 0 for value in (height, width)
        ):
            raise ValueError(
                f"model input shape {shape!r} must have static positive H/W")
        return OnnxInput(
            name=str(getattr(model_input, "name", "images")),
            data_type=data_type,
            height=int(height), width=int(width),
        )
    finally:
        del session


def calibration_images(directory: Path) -> Tuple[str, ...]:
    """Return image names in stable order and reject an empty pool."""

    if not directory.is_dir():
        raise FileNotFoundError(f"calibration image directory not found: {directory}")
    names = tuple(sorted(
        item.name for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in CALIBRATION_EXTENSIONS
    ))
    if not names:
        raise ValueError(f"no JPG/PNG calibration images found in {directory}")
    return names


def select_calibration_images(
    names: Sequence[str], sample: bool, sample_num: int, np_module=None,
) -> Tuple[str, ...]:
    """Apply the old sampling policy without changing the image pool itself."""

    if sample_num <= 0:
        raise ValueError("--cal-sample-num must be positive")
    selected = list(names)
    if sample and len(selected) > sample_num:
        if np_module is None:
            import numpy as np_module  # type: ignore
        selected = list(np_module.random.choice(
            selected, size=sample_num, replace=False))
    return tuple(str(name) for name in selected)


def make_conversion_plan(
    options,
    profile: ToolchainProfile,
    model_input: OnnxInput,
    *,
    base_dir: Optional[Path] = None,
) -> ConversionPlan:
    """Create a deterministic plan shared by generic and YOLO26 mappers."""

    onnx_path = resolve_path(options.onnx, base_dir)
    calibration_dir = resolve_path(options.cal_images, base_dir)
    output_arg = str(getattr(options, "output_dir", "."))
    if output_arg == ".":
        output_dir = onnx_path.parent
    else:
        output_dir = resolve_path(output_arg, base_dir)
    workspace = resolve_path(
        str(getattr(options, "ws", ".temporary_workspace")), base_dir)
    calibration_arg = Path(str(getattr(
        options, "cal", profile.default_calibration_dir)))
    if calibration_arg.is_absolute():
        cal_data_dir = calibration_arg.resolve()
    else:
        cal_data_dir = (workspace / calibration_arg).resolve()
    bpu_output_dir = workspace / "bpu_model_output"
    config_path = workspace / "config.yaml"

    model_name = onnx_path.stem
    if profile.x5_protocol:
        model_prefix = f"{model_name}_bayese_{model_input.width}x{model_input.height}_nv12"
    else:
        model_prefix = (
            f"{model_name}_{profile.march.replace('-', '')}_"
            f"{model_input.width}x{model_input.height}_nv12"
        )
    artifact_path = output_dir / (model_prefix + profile.artifact_suffix)
    log_path = output_dir / profile.log_name
    return ConversionPlan(
        toolchain=profile,
        onnx_path=onnx_path,
        calibration_images=calibration_dir,
        output_dir=output_dir,
        workspace=workspace,
        calibration_dir=cal_data_dir,
        bpu_output_dir=bpu_output_dir,
        config_path=config_path,
        artifact_path=artifact_path,
        log_path=log_path,
        model_prefix=model_prefix,
        compiler_command=profile.compiler_command,
        input=model_input,
        quantized=str(getattr(options, "quantized", "int8")),
        jobs=int(getattr(options, "jobs", 16)),
        optimize_level=str(getattr(
            options, "optimize_level", profile.default_optimize_level)),
        sample_calibration=bool(getattr(options, "cal_sample", True)),
        calibration_sample_num=int(getattr(options, "cal_sample_num", 20)),
        save_cache=bool(getattr(options, "save_cache", False)),
        overwrite=bool(getattr(options, "overwrite", False)),
    )


def _yaml_path(path: Path) -> str:
    """Render a filesystem path safely inside a single-quoted YAML scalar."""

    return str(path).replace("\\", "/").replace("'", "''")


def render_config(plan: ConversionPlan) -> str:
    """Render the target compiler YAML while preserving historical semantics."""

    profile = plan.toolchain
    lines = [
        "model_parameters:",
        f"  onnx_model: '{_yaml_path(plan.onnx_path)}'",
        f'  march: "{profile.march}"',
        "  layer_out_dump: False",
        f"  working_dir: '{_yaml_path(plan.bpu_output_dir)}'",
        f"  output_model_file_prefix: '{plan.model_prefix}'",
        "input_parameters:",
        '  input_name: ""',
        "  input_type_rt: 'nv12'",
        "  input_type_train: 'rgb'",
        "  input_layout_train: 'NCHW'",
        "  norm_type: 'data_scale'",
        "  scale_value: 0.003921568627451",
        "calibration_parameters:",
        f"  cal_data_dir: '{_yaml_path(plan.calibration_dir)}'",
        "  cal_data_type: 'float32'",
        "  calibration_type: 'default'",
    ]
    if profile.x5_protocol:
        optimization = "set_Softmax_input_int8,set_Softmax_output_int8"
        if plan.quantized == "int16":
            optimization += ",set_all_nodes_int16"
        lines.append(f"  optimization: {optimization}")
    elif plan.quantized == "int16":
        lines.extend([
            "  quant_config:",
            '    "model_config":',
            '      "all_node_type": "int16"',
            '      "model_output_type": "int16"',
        ])
    lines.extend([
        "compiler_parameters:",
    ])
    if not profile.x5_protocol:
        lines.append(
            "  extra_params: {'input_no_padding': True, 'output_no_padding': True}")
    lines.extend([
        f"  jobs: {plan.jobs}",
        "  compile_mode: 'latency'",
        "  debug: true",
        f"  optimize_level: '{plan.optimize_level}'",
        "",
    ])
    return "\n".join(lines)


def prepare_calibration(
    plan: ConversionPlan,
    names: Iterable[str],
    *,
    cv2_module=None,
    np_module=None,
) -> Tuple[Path, ...]:
    """Create the exact calibration files consumed by the selected compiler."""

    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to prepare calibration data; "
                "dependencies are never installed automatically") from exc
    if np_module is None:
        try:
            import numpy as np_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required to prepare calibration data; "
                "dependencies are never installed automatically") from exc

    plan.calibration_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name in names:
        source = cv2_module.imread(str(plan.calibration_images / name))
        if source is None:
            LOGGER.warning("failed to load calibration image: %s", name)
            continue
        tensor = cv2_module.cvtColor(source, cv2_module.COLOR_BGR2RGB)
        tensor = cv2_module.resize(tensor, (plan.input.width, plan.input.height))
        tensor = np_module.transpose(tensor, (2, 0, 1))
        tensor = np_module.expand_dims(tensor, axis=0).astype(np_module.float32)
        if plan.normalize_calibration:
            tensor = tensor / 255.0
        destination = plan.calibration_dir / (name + plan.calibration_suffix)
        if plan.normalize_calibration:
            np_module.save(str(destination), tensor)
        else:
            tensor.tofile(str(destination))
        generated.append(destination)
    if not generated:
        raise ValueError("none of the calibration images could be decoded")
    return tuple(generated)


CommandRunner = Callable[..., object]


def _run_command(command: Sequence[str], *, cwd: Optional[Path] = None):
    return subprocess.run(
        list(command), cwd=str(cwd) if cwd is not None else None,
        capture_output=True, text=True, check=True,
    )


def check_toolchain(
    profile: ToolchainProfile,
    *,
    command_runner: Optional[CommandRunner] = None,
) -> None:
    """Verify that the matching compiler is sourced before creating artifacts."""

    runner = command_runner or _run_command
    try:
        runner(profile.version_command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"{profile.version_command[0]} is unavailable; run this mapper "
            "inside the matching OpenExplore toolchain environment") from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either path contains the other.

    The check is intentionally lexical after ``resolve``.  It is used before
    creating a temporary workspace so a user cannot accidentally point the
    workspace at the input/calibration/output tree and then lose unrelated
    files during cleanup.
    """

    left = left.resolve()
    right = right.resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_paths_before_workspace(
    workspace_parent: Path,
    plan: ConversionPlan,
    *,
    calibration_is_absolute: bool = False,
) -> None:
    """Reject workspace/calibration/output overlaps before any mutation."""

    # ``output_dir == onnx.parent`` is the documented default.  The temporary
    # workspace is an owned child under --ws, so it is safe for that parent to
    # be the output directory or an ancestor of it.  Calibration images are
    # read from the tree, however, so placing the workspace under that tree
    # would make cleanup and image enumeration ambiguous.
    for label, path in (
        ("calibration images", plan.calibration_images),
    ):
        if _paths_overlap(workspace_parent, path):
            raise ValueError(
                f"--ws {workspace_parent} overlaps the {label} path {path}; "
                "choose a separate temporary-workspace parent")

    # Calibration output is written by this process.  It must not overwrite
    # the source image pool or be placed in the final output tree.
    if _paths_overlap(plan.calibration_dir, plan.calibration_images):
        raise ValueError(
            f"calibration output {plan.calibration_dir} overlaps the source "
            f"image directory {plan.calibration_images}")
    # A relative calibration directory is inside the owned workspace and is
    # safe even when the workspace parent is also the output directory.  An
    # absolute calibration directory is user-owned and must stay separate
    # from both the workspace parent and the final output tree.
    if calibration_is_absolute and (
            _paths_overlap(plan.calibration_dir, workspace_parent)
            or _paths_overlap(plan.calibration_dir, plan.output_dir)):
        raise ValueError(
            f"absolute calibration output {plan.calibration_dir} overlaps a "
            "workspace or output path")


def _owned_workspace(path: Path, parent: Path) -> bool:
    """Check that ``path`` is exactly one newly-created temp child."""

    try:
        relative = path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return len(relative.parts) == 1 and relative.parts[0].startswith(
        ".ultralytics-yolo-")


def _validate_owned_workspace(plan: ConversionPlan) -> None:
    """Ensure generated calibration data cannot touch user-owned inputs."""

    if _paths_overlap(plan.workspace, plan.onnx_path):
        raise ValueError(
            f"owned workspace {plan.workspace} overlaps the ONNX input "
            f"{plan.onnx_path}")
    if _paths_overlap(plan.workspace, plan.calibration_images):
        raise ValueError(
            f"owned workspace {plan.workspace} overlaps the calibration image "
            f"directory {plan.calibration_images}")
    if _paths_overlap(plan.calibration_dir, plan.calibration_images):
        raise ValueError(
            f"calibration output {plan.calibration_dir} overlaps the source "
            f"image directory {plan.calibration_images}")


def run_conversion(
    options,
    profile: ToolchainProfile,
    *,
    base_dir: Optional[Path] = None,
    ort_module=None,
    cv2_module=None,
    np_module=None,
    command_runner: Optional[CommandRunner] = None,
) -> ConversionPlan:
    """Run one complete conversion and return its artifact plan.

    Dependency modules and the command runner are injectable for host tests.
    Production callers leave them unset, which loads the explicitly prepared
    conversion environment and invokes the real compiler.
    """

    check_toolchain(profile, command_runner=command_runner)
    onnx_path = resolve_path(options.onnx, base_dir)
    model_input = inspect_onnx(onnx_path, ort_module=ort_module)
    requested_plan = make_conversion_plan(
        options, profile, model_input, base_dir=base_dir)
    workspace_parent = resolve_path(
        str(getattr(options, "ws", ".temporary_workspace")), base_dir)
    calibration_arg = Path(str(getattr(
        options, "cal", profile.default_calibration_dir)))
    _validate_paths_before_workspace(
        workspace_parent,
        requested_plan,
        calibration_is_absolute=calibration_arg.is_absolute(),
    )
    if requested_plan.artifact_path.exists() and not requested_plan.overwrite:
        raise FileExistsError(
            f"output artifact already exists: {requested_plan.artifact_path}; "
            "pass --overwrite to replace it explicitly")

    # Treat --ws as a parent directory.  A unique child is created for this
    # invocation, so an interrupted or concurrent conversion cannot delete a
    # user's pre-existing workspace.  We only ever remove this owned child.
    workspace_parent.mkdir(parents=True, exist_ok=True)
    owned_workspace = Path(tempfile.mkdtemp(
        prefix=".ultralytics-yolo-", dir=str(workspace_parent)))
    execution_options = argparse.Namespace(**vars(options))
    execution_options.ws = str(owned_workspace)
    plan = make_conversion_plan(
        execution_options, profile, model_input, base_dir=base_dir)
    _validate_owned_workspace(plan)
    names = calibration_images(plan.calibration_images)
    names = select_calibration_images(
        names, plan.sample_calibration, plan.calibration_sample_num,
        np_module=np_module,
    )

    completed = False
    try:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        plan.bpu_output_dir.mkdir(parents=True, exist_ok=True)
        plan.config_path.write_text(render_config(plan), encoding="utf-8")
        prepare_calibration(
            plan, names, cv2_module=cv2_module, np_module=np_module)
        runner = command_runner or _run_command
        runner(plan.compiler_command, cwd=plan.workspace)
        generated_artifact = plan.bpu_output_dir / (
            plan.model_prefix + plan.toolchain.artifact_suffix)
        if not generated_artifact.is_file():
            raise RuntimeError(
                f"compiler completed but did not produce {generated_artifact}")
        if plan.artifact_path.exists():
            if not plan.overwrite:
                raise FileExistsError(
                    f"output artifact already exists: {plan.artifact_path}; "
                    "pass --overwrite to replace it explicitly")
            plan.artifact_path.unlink()
        shutil.move(str(generated_artifact), str(plan.artifact_path))
        generated_log = plan.workspace / plan.toolchain.log_name
        if generated_log.is_file():
            if plan.log_path.exists():
                plan.log_path.unlink()
            shutil.move(str(generated_log), str(plan.log_path))
        LOGGER.info("converted model: %s", plan.artifact_path)
        completed = True
        return plan
    finally:
        # Leave a failed workspace and compiler log for diagnosis.  On success
        # remove only the unique child created above; a user-supplied --ws
        # directory and all unrelated files remain untouched.
        if (completed and not plan.save_cache and plan.workspace.exists()
                and _owned_workspace(plan.workspace, workspace_parent)):
            shutil.rmtree(plan.workspace)


def main_for_profile(
    argv: Optional[Sequence[str]],
    profile: ToolchainProfile,
    *,
    description: str,
    onnx_required: bool = False,
    include_march: bool = False,
    default_calibration_dir: Optional[str] = None,
) -> int:
    """Parse and execute a mapper while retaining a conventional exit code."""

    parser = build_mapper_parser(
        description, profile, onnx_required=onnx_required,
        include_march=include_march,
        default_calibration_dir=default_calibration_dir,
    )
    options = parser.parse_args(argv)
    selected_profile = profile
    if include_march:
        try:
            selected_profile = s_toolchain(options.march)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        run_conversion(options, selected_profile)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("conversion failed: %s", exc)
        return 1
    return 0
