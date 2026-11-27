"""Fits and verifies the antipodal Gaussian VQ3D/VQ4D codebooks.

Kidi rotates each weight matrix with block Hadamard transforms on both sides
and normalizes every group by its RMS. That makes groups of 3 or 4 weights look
like samples of N(0, I), so one universal codebook per dimension serves every
model, like the FP4 levels. Each code byte holds a sign bit and a 7-bit index
into 128 centroids c_k, so the 256 codewords are the antipodal pairs
{+c_k, -c_k}.

The centroids minimize mean squared error on N(0, I) samples with Lloyd's
algorithm restricted to antipodal codebooks. A sample x joins the pair that
maximizes |x . c_k| - |c_k|^2 / 2, and c_k becomes the mean of sign * x over
the samples that joined it.

The tables in vq_linear.cc are part of the packed format: changing them changes
every VQ weight and invalidates prepared checkpoints. The run that produced
them did not record its seed or sample count, so `fit` reproduces the method
rather than those exact digits, and `verify` checks that the shipped tables are
converged Lloyd solutions.

Prior work:
  S. P. Lloyd, "Least squares quantization in PCM", IEEE Transactions on
    Information Theory 28(2), 1982 (Bell Labs memo, 1957).
  Y. Linde, A. Buzo and R. M. Gray, "An Algorithm for Vector Quantizer Design",
    IEEE Transactions on Communications 28(1), 1980.
  A. Tseng, J. Chee, Q. Sun, V. Kuleshov and C. De Sa, "QuIP#: Even Better LLM
    Quantization with Hadamard Incoherence and Lattice Codebooks", ICML 2024,
    arXiv:2402.04396.
  A. Tseng, Q. Sun, D. Hou and C. De Sa, "QTIP: Quantization with Trellises and
    Incoherence Processing", NeurIPS 2024, arXiv:2406.11235.

Motivation: thanks to kaitchup's chimachima quantization of Inkling-2-Mini
(https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ), whose
trellis-coded expert weights motivated moving from scalar formats to vector
codes.

Usage (requires NumPy):
  python3 vq_codebook.py verify
  python3 vq_codebook.py fit --dim 3 --seed 0 > vq3d_codebook.inc
"""

import argparse
import pathlib
import re
import sys

import numpy as np

CENTROIDS = 128
TABLES = {3: "kVq3dCodebook", 4: "kVq4dCodebook"}
SOURCE = pathlib.Path(__file__).with_name("vq_linear.cc")
CHUNK = 1 << 18


class Statistics:
  """Antipodal Voronoi-cell sums for one codebook over a set of samples."""

  def __init__(self, half: np.ndarray):
    self.half = half
    dim = half.shape[1]
    self.count = np.zeros(CENTROIDS)
    self.sum = np.zeros((CENTROIDS, dim))
    self.error = 0.0
    self.energy = 0.0

  def add(self, x: np.ndarray) -> None:
    norms = (self.half * self.half).sum(1)
    dots = x @ self.half.T
    index = np.argmax(np.abs(dots) - 0.5 * norms, axis=1)
    dot = dots[np.arange(len(x)), index]
    folded = np.where(dot < 0, -1.0, 1.0)[:, None] * x
    self.count += np.bincount(index, minlength=CENTROIDS)
    for axis in range(x.shape[1]):
      self.sum[:, axis] += np.bincount(index, folded[:, axis], CENTROIDS)
    energy = (x * x).sum()
    self.energy += energy
    self.error += energy - 2 * np.abs(dot).sum() + norms[index].sum()

  def relative_error(self) -> float:
    """Root of the quantization MSE relative to the signal energy."""
    return float(np.sqrt(self.error / self.energy))

  def means(self) -> np.ndarray:
    return self.sum / np.maximum(self.count, 1)[:, None]


def gaussian_chunks(rng, dim: int, samples: int):
  for start in range(0, samples, CHUNK):
    yield rng.standard_normal((min(CHUNK, samples - start), dim))


def measure(half: np.ndarray, dim: int, samples: int, seed: int) -> Statistics:
  stats = Statistics(half)
  for x in gaussian_chunks(np.random.default_rng(seed), dim, samples):
    stats.add(x)
  return stats


def fit(dim: int, seed: int, samples: int, iterations: int,
        tolerance: float) -> np.ndarray:
  """Antipodal Lloyd (K-means) iterations on a fixed N(0, I) training set."""
  rng = np.random.default_rng(seed)
  data = rng.standard_normal((samples, dim))
  half = data[rng.choice(samples, CENTROIDS, replace=False)].copy()
  previous = np.inf
  for iteration in range(iterations):
    stats = Statistics(half)
    for start in range(0, samples, CHUNK):
      stats.add(data[start:start + CHUNK])
    error = stats.relative_error()
    print(f"iteration {iteration}: relative error {100 * error:.3f}%",
          file=sys.stderr)
    empty = stats.count == 0
    half = stats.means()
    half[empty] = data[rng.choice(samples, int(empty.sum()), replace=False)]
    if previous - error < tolerance * error:
      break
    previous = error
  order = np.argsort((half * half).sum(1), kind="stable")
  half = half[order]
  # Keep the member of each antipodal pair whose first nonzero entry is > 0.
  lead = half[np.arange(CENTROIDS), np.argmax(half != 0, axis=1)]
  return np.where(lead < 0, -1.0, 1.0)[:, None] * half


def format_table(dim: int, half: np.ndarray) -> str:
  padded = np.zeros((CENTROIDS, 4))
  padded[:, :dim] = half
  entries = [
      "{" + ", ".join(f"{value:.7f}F" for value in row) + "}"
      for row in padded
  ]
  lines = [
      "    " + ", ".join(entries[i:i + 2]) + ","
      for i in range(0, CENTROIDS, 2)
  ]
  name = TABLES[dim]
  return "\n".join(
      [f"alignas(64) constexpr float {name}[{CENTROIDS}][4] = {{"]
      + lines + ["};"])


def read_table(source: pathlib.Path, dim: int) -> np.ndarray:
  text = source.read_text()
  name = TABLES[dim]
  start = text.index(name)
  body = text[text.index("{", start) + 1:text.index("};", start)]
  values = [float(v) for v in re.findall(r"-?\d+\.\d+(?=F)", body)]
  table = np.array(values).reshape(CENTROIDS, 4)
  if dim == 3 and np.any(table[:, 3]):
    raise ValueError(f"{name} must pad lane 3 with zeros")
  return table[:, :dim]


def verify(source: pathlib.Path, samples: int, seed: int,
           max_shift: float, max_gain: float) -> bool:
  """Checks each table is a converged antipodal Lloyd solution on N(0, I)."""
  ok = True
  for dim in TABLES:
    half = read_table(source, dim)
    points = np.concatenate([half, -half])
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
    spacing = gaps[np.triu_indices(len(points), 1)].min()
    stats = measure(half, dim, samples, seed)
    refit = measure(stats.means(), dim, samples, seed + 1)
    error = stats.relative_error()
    shift = np.abs(stats.means() - half).max()
    gain = 1 - refit.relative_error() / error
    passed = (spacing > 1e-3 and shift <= max_shift * spacing and
              gain <= max_gain)
    ok &= passed
    print(f"{TABLES[dim]}: relative error {100 * error:.2f}%, "
          f"largest centroid-to-cell-mean shift {shift:.3f} "
          f"({100 * shift / spacing:.1f}% of the closest codeword spacing "
          f"{spacing:.3f}), one more Lloyd step gains {100 * gain:.3f}%: "
          f"{'ok' if passed else 'FAILED'}")
  return ok


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  commands = parser.add_subparsers(dest="command", required=True)
  fit_parser = commands.add_parser("fit", help="fit a new table")
  fit_parser.add_argument("--dim", type=int, choices=sorted(TABLES),
                          required=True)
  fit_parser.add_argument("--seed", type=int, default=0)
  fit_parser.add_argument("--samples", type=int, default=2_000_000)
  fit_parser.add_argument("--iterations", type=int, default=200)
  fit_parser.add_argument("--tolerance", type=float, default=1e-6,
                          help="stop below this relative error improvement")
  verify_parser = commands.add_parser("verify",
                                      help="check the shipped tables")
  verify_parser.add_argument("--source", type=pathlib.Path, default=SOURCE)
  verify_parser.add_argument("--samples", type=int, default=2_000_000)
  verify_parser.add_argument("--seed", type=int, default=1)
  verify_parser.add_argument("--max-shift", type=float, default=0.1,
                             help="largest centroid-to-cell-mean shift as a "
                             "fraction of the closest codeword spacing")
  verify_parser.add_argument("--max-gain", type=float, default=0.002,
                             help="largest relative error reduction from one "
                             "more Lloyd step")
  args = parser.parse_args()
  if args.command == "fit":
    half = fit(args.dim, args.seed, args.samples, args.iterations,
               args.tolerance)
    print(format_table(args.dim, half))
    return 0
  return 0 if verify(args.source, args.samples, args.seed, args.max_shift,
                     args.max_gain) else 1


if __name__ == "__main__":
  sys.exit(main())
