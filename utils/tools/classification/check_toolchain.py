"""Check an OE YAML and calibration loader inside its pinned container.

Only configuration validation and calibration loading run; no PTQ or compiler
is invoked. The SDK implementation stays in the installed vendor package.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Validate the config and read every calibration sample using OE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("x5", "s100", "s100p", "s600"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from horizon_tc_ui.data import data_loader_factory

    if args.platform == "x5":
        from horizon_tc_ui.config.mapper_conf_parser import MpConf
        from horizon_tc_ui.helper import get_raw_transformer

        conf = MpConf(str(args.config), model_type="onnx")
        shape = conf.input_shapes[0]
        transforms = get_raw_transformer(conf.input_type_rt[0], conf.input_type_train[0],
                                         conf.input_layout_train[0], shape[2], shape[3])
    else:
        from horizon_tc_ui.config.params_parser import ParamsParser

        parser = ParamsParser(str(args.config))
        parser.validate_parameters()
        conf = parser.conf
        shape = conf.input_shapes[0]
        transforms = []
    loader = data_loader_factory.get_raw_image_dir_loader(
        transforms, conf.cal_data_dir[0], shape, np.float32)
    count = 0
    first = None
    for sample in loader:
        array = np.asarray(sample)
        if not np.isfinite(array).all() or array.size != int(np.prod(shape)):
            raise ValueError("Invalid calibration loader output")
        count += 1
        if first is None:
            first = {"shape": list(array.shape), "dtype": str(array.dtype),
                     "min": float(array.min()), "max": float(array.max())}
    if count != 200:
        raise ValueError(f"Expected 200 calibration samples, got {count}")
    print(json.dumps({"stage": "oe-config-and-loader-preflight", "status": "passed",
                      "platform": args.platform, "config": str(args.config),
                      "sample_count": count, "first_loaded_sample": first,
                      "compilation": "not-run", "board": "not-run"}))


if __name__ == "__main__":
    main()
