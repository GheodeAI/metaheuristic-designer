# tests/benchmarks/test_image_benchmarks.py
import numpy as np
import pytest
from PIL import Image

from metaheuristic_designer.benchmarks import ImgApprox, ImgEntropy
from metaheuristic_designer.encodings import ImageEncoding
from metaheuristic_designer.initializers import UniformInitializer


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def _random_reference_img(size=(8, 8)):
    """Return a PIL Image of the given size with random RGB pixels."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------- ImgApprox with different diff functions ----------
@pytest.mark.parametrize("img_size", [(8, 8), (10, 10)])
@pytest.mark.parametrize("diff_func", ["MAE", "SSIM", "NMI"])
def test_img_approx_diff_funcs(img_size, diff_func):
    ref = _random_reference_img(img_size)
    objfunc = ImgApprox(img_dim=img_size, reference=ref, diff_func=diff_func, mode="min")
    vecsize = objfunc.vecsize
    encoding = ImageEncoding(img_size, color=True)
    init = UniformInitializer(vecsize, 0, 255, pop_size=3, dtype=float, encoding=encoding)
    pop = init.generate_population(objfunc)
    decoded = pop.decode()
    result = objfunc.objective(decoded)
    assert result.shape == (3,)


# ---------- Batch vs single consistency ----------
@pytest.mark.parametrize("img_size", [(8, 8), (10, 10)])
def test_img_approx_single_vs_batch(img_size):
    ref = _random_reference_img(img_size)
    objfunc = ImgApprox(img_dim=img_size, reference=ref, diff_func="MSE", mode="min")
    vecsize = objfunc.vecsize
    encoding = ImageEncoding(img_size, color=True)
    init = UniformInitializer(vecsize, 0, 255, pop_size=2, dtype=float, encoding=encoding)
    pop = init.generate_population(objfunc)
    decoded = pop.decode()  # (2, H, W, 3)
    batch = objfunc.objective(decoded)
    single0 = objfunc.objective(decoded[0:1])
    single1 = objfunc.objective(decoded[1:2])
    assert batch[0] == pytest.approx(single0.item())
    assert batch[1] == pytest.approx(single1.item())

# ---------- Repair preserves already valid values ----------
@pytest.mark.parametrize("img_size", [(8, 8), (10, 10)])
def test_repair_preserves_valid_values(img_size):
    ref = _random_reference_img(img_size)
    objfunc = ImgApprox(img_dim=img_size, reference=ref, diff_func="MSE")
    vecsize = objfunc.vecsize
    valid = np.random.default_rng(42).uniform(0, 255, vecsize)
    repaired = objfunc.repair_solution(valid)
    np.testing.assert_array_almost_equal(repaired, valid)


# ---------- ImgEntropy with different nbins ----------
@pytest.mark.parametrize("img_size", [(8, 8), (10, 10)])
@pytest.mark.parametrize("nbins", [5, 20, 100])
def test_img_entropy_nbins(img_size, nbins):
    objfunc = ImgEntropy(img_dim=img_size, nbins=nbins, mode="max")
    vecsize = objfunc.vecsize
    encoding = ImageEncoding(img_size, color=True)
    init = UniformInitializer(vecsize, 0, 255, pop_size=2, dtype=float, encoding=encoding)
    pop = init.generate_population(objfunc)
    decoded = pop.decode()  # shape (2, H, W, 3) – two images

    # ImgEntropy is not vectorised – evaluate one by one
    results = [objfunc.objective(decoded[i]) for i in range(2)]

    # Each result should be a scalar (non‑vectorised output)
    for r in results:
        assert np.isscalar(r) and r >= 0
