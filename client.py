"""
Convenience client wrapper. Agents use EnvClient from OpenEnv directly;
this module re-exports for discoverability.
"""
try:
    from openenv.core.env_client import EnvClient
except ImportError:
    try:
        from openenv.env_client import EnvClient
    except ImportError:
        from openenv_core.env_client import EnvClient  # type: ignore


__all__ = ["EnvClient"]
