"""Build hook for the binary wheel.

The distribution is one Python module plus the shared object it drives, so the
wheel is not "pure": setup.py builds `libanubis_solver.so` with the repository's
own Makefile (CPU kernels only - the GPU and TPU backends dlopen their runtimes
at run time, so nothing else has to be present) and copies it into
`_anubis_solver_native/`, where `anubis_solver._candidate_paths()` looks for it.

`has_ext_modules()` is what makes bdist_wheel emit a platform tag instead of
py3-none-any; the object is not a CPython extension module, so the tag stays
`py3-none-<platform>`.
"""

import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:  # wheel >= 0.40 keeps it here, setuptools vendors the older path
    from wheel.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover
    from setuptools.command.bdist_wheel import bdist_wheel

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(HERE, "_anubis_solver_native")
LIB = "libanubis_solver.so"
# GPU/TPU backends are dlopen()ed, so the wheel carries CPU kernels only.
MAKE_FLAGS = ["CUDA=0", "VULKAN=0", "L0=0", "OPENCL=0"]
TREE = ("Makefile", "src", "include", "third_party")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # setuptools hook
        return True


class BuildPy(build_py):
    def run(self):
        # Build in a copy of the tree with the wheel's flags: reusing the
        # developer's .so would ship whatever backends their last `make`
        # happened to enable (and make does not track flag changes).
        with tempfile.TemporaryDirectory() as td:
            for name in TREE:
                src = os.path.join(HERE, name)
                dst = os.path.join(td, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            jobs = str(os.cpu_count() or 1)
            argv = ["make", "-s", "-j" + jobs, LIB] + MAKE_FLAGS
            print("building the solver:", " ".join(argv), file=sys.stderr)
            subprocess.run(argv, cwd=td, check=True)
            lib = os.path.join(td, LIB)
            if not os.path.exists(lib):
                raise SystemExit("make did not produce " + LIB)
            os.makedirs(NATIVE, exist_ok=True)
            shutil.copy2(lib, os.path.join(NATIVE, LIB))
        super().run()
        # build_py only copies package data it knows about; make sure the
        # object lands even when the package directory is rebuilt from a
        # stale build/lib tree.
        target = os.path.join(self.build_lib, "_anubis_solver_native", LIB)
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(NATIVE, LIB), target)


class BDistWheel(bdist_wheel):
    """py3-none-<platform>: the object is not a CPython extension module."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        # a wheel built here needs this host's glibc, so say so: a
        # `linux_x86_64` tag would install on systems where the object cannot
        # load. ANUBIS_WHEEL_PLAT=manylinux_2_34_x86_64 when built against the
        # bookworm sysroots in RELEASING.md.
        plat = os.environ.get("ANUBIS_WHEEL_PLAT", plat)
        return "py3", "none", plat


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy, "bdist_wheel": BDistWheel},
)
