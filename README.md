# Pareto/GGG Python Package

This library implements the Pareto/GGG model for customer transaction modeling and lifetime inference, using a pure Python MCMC engine with slice sampling.

## 📦 Features

- Individual-level estimation of interpurchase regularity (`k`), transaction rate (`lambda`), and dropout rate (`mu`)
- Population-level (hierarchical) estimation via Gamma priors
- Slice sampling implementation in NumPy
- Fast simulation and inference tools
- Compatible with editable install (`pip install -e .`)

## 📁 Package Structure

```
pareto_ggg/
├── model.py         # Core sampling logic
├── mcmc.py          # MCMC engine
├── slice_sampling.py # Slice sampler and posteriors
├── utils.py         # Simulation utilities
├── main.py          # Example entry point
├── tests/           # Unit tests
```

## 🚀 Quick Start

```bash
pip install -e .
python -m pareto_ggg.main
```

## 🧪 Testing

```bash
pytest pareto_ggg/tests
```

## 📊 Output

The main script prints:
- Estimated population-level parameters (`t`, `gamma`, `r`, `alpha`, `s`, `beta`)
- Posterior draws for individual customer parameters (`k`, `lambda`, `mu`, `tau`, `z`)

## 🛠 Requirements
- Python 3.8+
- `numpy`, `scipy`, `pandas`, `pytest` (for testing)

## 📚 Reference
Based on the Pareto/GGG framework presented in:
> Platzer & Reutterer (2016). *Ticking away the moments: Timing regularity helps to better predict customer activity.* Marketing Science.

## 📝 License
MIT
