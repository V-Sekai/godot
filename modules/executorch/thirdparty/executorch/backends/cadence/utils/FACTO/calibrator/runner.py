# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from typing import Any, List, Optional, OrderedDict, Tuple

import torch
from facto.inputgen.argtuple.engine import MetaArgTupleEngine
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.specs.model import Spec
from facto.inputgen.utils.config import TensorConfig
from facto.specdb.db import SpecDictDB
from facto.utils.ops import get_op_overload
from torch._ops import OpOverload

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def smt(meta_tuple):
    return str([str(x) for x in meta_tuple])


class SpecRunner:
    def __init__(
        self,
        op: OpOverload,
        spec: Spec,
        *,
        valid: bool = True,
        out: bool = False,
        devices: Tuple[str] = ("cpu",),
        config: Optional[TensorConfig] = None,
    ):
        self.spec = spec
        self.config = config
        self.generator = ArgumentTupleGenerator(self.spec, config=config)
        self.valid = valid
        self.out = out
        self.op_name = op.__name__
        self.op = op
        self.results = {}
        self.devices = devices
        self.results = {}
        for device in self.devices:
            self.results[device] = {}

    def report_device(self, device):
        print(f"Device: {device}\n")
        failures = []
        for meta_tuple in self.results[device]:
            success = self.results[device][meta_tuple]
            if not success:
                failures.append(meta_tuple)
        if len(failures) > 0:
            print("FAILURES\n")
            for meta_tuple in failures:
                print(f"\t{meta_tuple}\n")
        else:
            print("SUCCESS\n")

    def report_inconsistencies(self):
        print(f"Devices: {' '.join(self.devices)}\n")
        meta_tuples = self.results[self.devices[0]].keys()
        inconsistencies = set()
        for meta_tuple in meta_tuples:
            res = self.results[self.devices[0]][meta_tuple]
            for device in self.devices[1:]:
                res ^= self.results[device][meta_tuple]
                if not res:
                    inconsistencies.add(meta_tuple)
        if len(inconsistencies) > 0:
            print("INCONSISTENCIES\n")
            for meta_tuple in inconsistencies:
                res = [self.results[d][meta_tuple] for d in self.devices]
                res_string = " ".join(["x" if r else "o" for r in res])
                print(f"\t{res_string} {meta_tuple}\n")

    def run(self):
        engine = MetaArgTupleEngine(self.spec, out=self.out)
        for meta_tuple in engine.gen(valid=self.valid):
            self.run_meta_tuple(meta_tuple)
        if len(self.devices) > 1:
            self.report_inconsistencies()
        for device in self.devices:
            self.report_device(device)

    def move_to_device(
        self,
        device: str,
        src_posargs: List[Any],
        src_inkwargs: OrderedDict[str, Any],
        src_outargs: OrderedDict[str, Any],
    ):
        if device == ("cpu" if self.config is None else self.config.device):
            return src_posargs, src_inkwargs, src_outargs
        posargs = []
        inkwargs = OrderedDict()
        outargs = OrderedDict()
        for arg in src_posargs:
            new = arg
            if isinstance(arg, torch.Tensor):
                new = arg.to(device=device)
            posargs.append(new)
        for k, v in src_inkwargs.items():
            new = v
            if isinstance(v, torch.Tensor):
                new = v.to(device=device)
            inkwargs[k] = new
        for k, v in src_outargs.items():
            new = v
            if isinstance(v, torch.Tensor):
                new = v.to(device=device)
            outargs[k] = new
        return posargs, inkwargs, outargs

    def run_meta_tuple(
        self, meta_tuple: Tuple[MetaArg]
    ) -> Tuple[bool, Any, List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        print(f"Running op: {self.op_name}, meta_tuple: {[str(x) for x in meta_tuple]}")
        posargs, inkwargs, outargs = self.generator.gen_tuple(meta_tuple, out=self.out)
        for device in self.devices:
            posargs, inkwargs, outargs = self.move_to_device(
                device, posargs, inkwargs, outargs
            )
            success, res, res_posargs, res_inkwargs, res_outargs = self.run_values(
                meta_tuple, posargs, inkwargs, outargs
            )
            if (
                self.valid
                and success
                and device != "cpu"
                and isinstance(res, torch.Tensor)
            ):
                cpu_posargs, cpu_inkwargs, cpu_outargs = self.move_to_device(
                    "cpu", posargs, inkwargs, outargs
                )
                (
                    cpu_success,
                    cpu_res,
                    cpu_res_posargs,
                    cpu_res_inkwargs,
                    cpu_res_outargs,
                ) = self.run_values(meta_tuple, cpu_posargs, cpu_inkwargs, cpu_outargs)
                if cpu_success and cpu_res is not None:
                    if not torch.allclose(cpu_res, res.to("cpu")):
                        logging.warning(
                            f"NOT ALL CLOSE opname: {self.op_name}, meta_tuple: {smt(meta_tuple)}, device: {device}, {(cpu_res.to(torch.float) - res.to('cpu').to(torch.float)).abs().max()}"
                        )
            mt = smt(meta_tuple)
            if mt in self.results[device]:
                logging.warning(f"Repeated meta_tuple {mt}")
                self.results[device][mt] &= success
            else:
                self.results[device][mt] = success

    def run_values(
        self,
        meta_tuple: Tuple[MetaArg],
        posargs: List[Any],
        inkwargs: OrderedDict[str, Any],
        outargs: OrderedDict[str, Any],
    ) -> Tuple[bool, Any, List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        try:
            res = self.op(*posargs, **inkwargs, **outargs)
            if not self.valid:
                logging.warning("unexpected success")
            success = self.valid
            return success, res, posargs, inkwargs, outargs
        except AssertionError as e:
            raise RuntimeError(
                f"opname: {self.op_name}, meta_tuple: {meta_tuple}"
            ) from e
        except Exception as e:
            if self.valid:
                logging.warning(
                    f"opname: {self.op_name}, meta_tuple: {smt(meta_tuple)}, exception: {e}"
                )
            success = not self.valid
            return success, None, posargs, inkwargs, outargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op")
    parser.add_argument(
        "--invalid", action="store_true", help="generate invalid inputs"
    )
    parser.add_argument("--out", action="store_true", help="run out variants")
    parser.add_argument("--devices", nargs="*", default=("cpu",), help="run on devices")
    args = parser.parse_args()

    if args.op not in SpecDictDB:
        raise RuntimeError(f"Op {args.op} not found in SpecDB")

    spec = SpecDictDB[args.op]
    op = get_op_overload(args.op)
    SpecRunner(
        op, spec, valid=not args.invalid, out=args.out, devices=args.devices
    ).run()


if __name__ == "__main__":
    main()
