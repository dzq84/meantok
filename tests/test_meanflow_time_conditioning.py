"""CPU regression tests for the interval-conditioned MeanFlow velocity field."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from speechflow.flow.meanflow import MeanFlowWrapper, adaptive_l2_loss, differentiable_sample
from speechflow.modules.dit1d import DiT1D


def tiny_dit():
    torch.manual_seed(12)
    model = DiT1D(
        input_size=4, patch_size=2, in_channels=1, feat_dim=2,
        hidden_size=16, depth=1, num_heads=2, num_tokens=8,
        dropout_prob=0, spk_dim=3,
    ).eval()
    # adaLN-Zero and the new interval branch initially hide all time dependence.
    projections = [model.dt_embedder.mlp[-1], model.final_layer.linear,
                   model.final_layer.adaLN_modulation[-1]]
    projections += [block.adaLN_modulation[-1] for block in model.blocks]
    for layer in projections:
        torch.nn.init.normal_(layer.weight, std=0.2)
    return model


def inputs():
    return (torch.randn(2, 1, 4, 2), torch.tensor([0.8, 0.7]),
            torch.randint(0, 8, (2, 4, 1)), torch.randn(2, 1, 3))


def test_dit_uses_interval_and_preserves_default_r_zero():
    model = tiny_dit()
    x, t, tokens, spk = inputs()
    at_zero = model(x, t, tokens, spk, r=torch.zeros_like(t))
    at_r = model(x, t, tokens, spk, r=torch.tensor([0.3, 0.2]))
    torch.testing.assert_close(model(x, t, tokens, spk), at_zero)
    assert (at_zero - at_r).abs().max() > 1e-5
    at_r.square().mean().backward()
    grad = model.dt_embedder.mlp[-1].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("api", ["autograd", "funtorch"])
def test_dit_jvp_matches_finite_difference_with_r_fixed(api):
    model = tiny_dit()
    x, t, tokens, spk = inputs()
    r = torch.tensor([0.3, 0.2])
    velocity = torch.randn_like(x)

    def field(z, current_t, start_r):
        return model(z, current_t, tokens, spk, r=start_r)

    jvp = (torch.func.jvp if api == "funtorch"
           else torch.autograd.functional.jvp)
    _, derivative = jvp(field, (x, t, r),
                        (velocity, torch.ones_like(t), torch.zeros_like(r)))
    h = 1e-3
    finite_difference = (field(x + h * velocity, t + h, r)
                         - field(x - h * velocity, t - h, r)) / (2 * h)
    torch.testing.assert_close(derivative, finite_difference, rtol=2e-2, atol=1e-3)


class PolynomialVelocity(torch.nn.Module):
    """Analytic velocity makes the wrapper's fixed-r derivative independently testable."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.4))
        self.interval_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.times = []

    def forward(self, x, t, y, spk=None, r=None):
        assert r is not None, "MeanFlow must explicitly pass its sampled r"
        self.times.append((t, r))
        return (self.scale * x * t[:, None, None, None]
                + self.interval_scale * (t - r)[:, None, None, None].square())


@pytest.mark.parametrize("api", ["autograd", "funtorch"])
def test_loss_passes_r_and_uses_the_fixed_r_derivative(api, monkeypatch):
    model = PolynomialVelocity()
    wrapper = MeanFlowWrapper(jvp_api=api)
    x = torch.tensor([[[[0.2, -0.4]]], [[[0.6, 0.1]]]])
    noise = torch.tensor([[[[-0.3, 0.8]]], [[[0.2, -0.7]]]])
    t, r = torch.tensor([0.8, 0.6]), torch.tensor([0.2, 0.6])
    monkeypatch.setattr(wrapper, "sample_t_r", lambda batch, device: (t, r))
    monkeypatch.setattr(torch, "randn_like", lambda _: noise)
    loss, mse = wrapper.loss(model, x, c=None)

    t4, dt = t[:, None, None, None], (t - r)[:, None, None, None]
    z, velocity = (1 - t4) * x + t4 * noise, noise - x
    prediction = model.scale * z * t4 + model.interval_scale * dt.square()
    derivative = model.scale * (velocity * t4 + z) + 2 * model.interval_scale * dt
    error = prediction - (velocity - dt * derivative).detach()
    expected = adaptive_l2_loss(error)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(mse, error.detach().square().mean())
    actual_grads = torch.autograd.grad(loss, tuple(model.parameters()))
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0


def test_both_samplers_use_full_interval_and_differentiable_sample_backpropagates():
    wrapper = MeanFlowWrapper(channels=1, image_size=4)
    model = PolynomialVelocity()
    torch.manual_seed(23)
    noise = torch.randn(2, 1, 4, 2)
    expected = noise - (model.scale * noise + model.interval_scale)
    torch.manual_seed(23)
    sampled = wrapper.sample(model, 2, None, 2, device="cpu")
    torch.testing.assert_close(sampled, expected)
    assert not sampled.requires_grad
    torch.manual_seed(23)
    differentiable = differentiable_sample(model, wrapper, 2, None, 2, device="cpu")
    torch.testing.assert_close(differentiable, expected)
    differentiable.square().mean().backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    for t, r in model.times:
        torch.testing.assert_close(t, torch.ones(2))
        torch.testing.assert_close(r, torch.zeros(2))


def test_trainer_forwards_r_without_loading_optional_training_dependencies():
    # Execute the actual small adapter, without importing Lightning/audio datasets.
    path = CODE_DIR / "speechflow" / "train_meanflow.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "MeanFlowTrainer")
    forward = next(node for node in trainer.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    adapter = namespace["forward"]
    trainer_stub = SimpleNamespace(model=lambda **kwargs: kwargs)
    x, t, tokens, spk = inputs()
    r = torch.tensor([0.3, 0.2])
    passed = adapter(trainer_stub, x, t, tokens, spk, r=r)
    for key, value in {"x": x, "t": t, "y": tokens, "spk": spk, "r": r}.items():
        assert passed[key] is value
    assert adapter(trainer_stub, x, t, tokens, spk)["r"] is None
