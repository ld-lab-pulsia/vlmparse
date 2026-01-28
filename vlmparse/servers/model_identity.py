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

        Use this method in server configs to ensure consistent passing
        of model_name and default_model_name to client configs.

        Args:
            base_url: The base URL for the client to connect to.

        Returns:
            Dictionary with base_url, model_name, and default_model_name.
        """
        return {
            "base_url": base_url,
            "model_name": self.model_name,
            "default_model_name": self.get_effective_model_name(),
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
