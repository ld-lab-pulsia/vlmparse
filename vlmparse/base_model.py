from pydantic import BaseModel, ConfigDict


class VLMParseBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=True,
        extra="allow",
    )

    def __repr__(self):
        from devtools import PrettyFormat

        pformat = PrettyFormat()
        return pformat(self, highlight=False)
