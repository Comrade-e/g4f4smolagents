from typing import Optional, List, Dict

from smolagents import ApiModel, Tool, ChatMessage

#g4f API class
class G4fModel(ApiModel):

    def __init__(
        self,
        model_id: Optional[str] = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        api_base=None,
        provider=None,
        **kwargs,
    ):

        self.model_id = model_id
        self.api_base = api_base
        self.provider = provider
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """Create the g4f client."""
        try:
            import g4f
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'g4f' extra to use g4f4smolagents: `pip install g4f"
            ) from e

        return g4f.client.Client(provider=self.provider)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        response = self.client.chat.completions.create(**completion_kwargs) # responce from g4f

        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
        )
        return self.postprocess_message(first_message, tools_to_call_from)
