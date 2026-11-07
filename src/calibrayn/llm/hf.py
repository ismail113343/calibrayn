from calibrayn.llm.base import LLMError, require_text


class HuggingFaceInferenceClient:
    hosted = True

    def __init__(self, model, token, client=None):
        self.model = model
        self.name = f"hf_inference:{model}"
        if client is None:
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=model, token=token)
        self._client = client

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = self._client.chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            text = reply.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise LLMError("Hugging Face sent a reply in an unexpected shape.") from error
        except (OSError, RuntimeError, ValueError) as error:
            raise LLMError(f"Hugging Face could not answer: {error}") from error
        return require_text(text, "Hugging Face")


class HuggingFaceLocalClient:
    hosted = False

    def __init__(self, model, pipeline=None, token=None):
        self.model = model
        self.name = f"hf_local:{model}"
        self._pipeline = pipeline
        self._token = token or None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as error:
                raise LLMError(
                    "hf_local needs the transformers package. "
                    "Run: pip install -r requirements-ml.txt"
                ) from error
            self._pipeline = pipeline("text-generation", model=self.model, token=self._token)
        return self._pipeline

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            output = self._load()(
                messages,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
            )
            text = output[0]["generated_text"][-1]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise LLMError("The local model sent a reply in an unexpected shape.") from error
        except (OSError, RuntimeError, ValueError) as error:
            raise LLMError(f"The local model failed: {error}") from error
        return require_text(text, "The local model")
