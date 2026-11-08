"""BW1000 Flux-design port. Native CUDA Flux binaries are not used here."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "FluxDCU":
        from .runtime import FluxDCU
        return FluxDCU
    raise AttributeError(name)
