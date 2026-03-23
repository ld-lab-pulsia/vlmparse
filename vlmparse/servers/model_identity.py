"""Model identity mixin for consistent model name handling between server and client configs."""

from pydantic import BaseModel, Field


class ModelIdentityMixin(BaseModel):
    """Mixin providing model identity fields with validation.

    This mixin ensures that model_name and default_model_name are consistently
    passed from server configs to client configs.
    """

    model_name: str
    default_model_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    def get_effective_model_name(self) -> str:
        """Returns the model name to use for API calls."""
        return self.default_model_name if self.default_model_name else self.model_name

    def _create_client_kwargs(self, base_url: str) -> dict:
        """Generate kwargs for client config with model identity.

        Returns a dict with ``model_name`` and ``endpoint`` (a
        :class:`ModelEndpointConfig`) ready to be splatted into a
        ``ConverterConfig`` constructor.
        """
        from vlmparse.model_endpoint_config import ModelEndpointConfig

        return {
            "model_name": self.model_name,
            "endpoint": ModelEndpointConfig(
                base_url=base_url,
                model_name=self.get_effective_model_name(),
            ),
        }

    def get_all_names(self) -> list[str]:
        """Get all names this model can be referenced by.

        Returns:
            List containing model_name, aliases, and short name (after last /).
        """
        names = [self.model_name] + self.aliases
        if "/" in self.model_name:
            names.append(self.model_name.split("/")[-1])
        return [n for n in names if isinstance(n, str)]
