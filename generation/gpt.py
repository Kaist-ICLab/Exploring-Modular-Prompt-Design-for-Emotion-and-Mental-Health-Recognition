from ollama import Client
import json
import time
import openai
import anthropic
import google.generativeai as genai
import concurrent.futures
import transformers
import torch


class BaseModel:
    def __init__(self, api_key, args):
        self.api_key = api_key
        self.args = args
        self.general_prompts = self.extract_prompt_template()

    def extract_prompt_template(self):
        general_prompts = json.load(open('../prompt_template/classification_general_prompt.json', 'r'))
        if self.args.shot > 0:
            general_prompts = general_prompts['few_shot']
        else:
            general_prompts = general_prompts['zero_shot']
        return general_prompts

    def split_prompt(self, prompt: dict, mode='basic'):
        if mode == 'basic':
            system_prompt = self.general_prompts['system'].format(system_instruction=prompt['system_instruction'],
                                                                  task_query=prompt['task_query'])
            user_prompt = self.general_prompts['user'].format(few_shot=prompt['few_shot'],
                                                              prompt_strategy=prompt['prompt_strategy'],
                                                              context=prompt['context'], label_def=prompt['label_def'],
                                                              output_indicator=prompt['output_indicator'])
        else:
            system_prompt = prompt['system_instruction'] + prompt['task_query']
            user_prompt = (prompt['few_shot'] + prompt['prompt_strategy'] +
                           prompt['context'] + prompt['output_indicator'])

        return system_prompt, user_prompt

    def response(self, prompt: dict):
        system_prompt, user_prompt = self.split_prompt(prompt, mode='basic')
        return system_prompt, user_prompt


class GPT(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        self.client = openai.OpenAI(api_key=self.api_key)

    def _get_response(self, system_prompt, user_prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during GPT response: {e}")
            return None

    def response(self, text: dict, timeout=10, retries=5):
        system_prompt, user_prompt = super().response(text)

        for attempt in range(retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._get_response, system_prompt, user_prompt)
                    response = future.result(timeout=timeout)
                    if response:
                        return system_prompt + user_prompt, response
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1}: Timeout. Retrying...")
            except Exception as e:
                print(f"An error occurred: {e}")

        return system_prompt + user_prompt, "Failed to get a response"


class Claude(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _get_response(self, system_prompt, user_prompt):
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error during Claude response: {e}")
            return None

    def response(self, text: dict, timeout=10, retries=5):
        system_prompt, user_prompt = super().response(text)

        for attempt in range(retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._get_response, system_prompt, user_prompt)
                    response = future.result(timeout=timeout)
                    if response:
                        return system_prompt + user_prompt, response
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1}: Timeout. Retrying...")
            except Exception as e:
                print(f"An error occurred: {e}")

        return system_prompt + user_prompt, "Failed to get a response"


class Gemini(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "candidate_count": 1,
            "max_output_tokens": 256,
            "temperature": 0.0,
            "top_p": 0.9,
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.client = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            safety_settings=self.safety_settings
        )

    def _get_response(self, system_prompt, user_prompt):
        try:
            response = self.client.generate_content(system_prompt + user_prompt)
            return response.text
        except genai.types.generation_types.BlockedPromptException as e:
            print(f"Prompt blocked due to: {e}")
            return None
        except ValueError as e:
            print(f"ValueError encountered: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        finally:
            time.sleep(0.25)

    def response(self, text: dict, timeout=10, retries=5):
        system_prompt, user_prompt = super().response(text)

        for attempt in range(retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._get_response, system_prompt, user_prompt)
                    response = future.result(timeout=timeout)
                    if response:
                        return system_prompt + user_prompt, response
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1}: Timeout. Retrying...")
            except Exception as e:
                print(f"An error occurred: {e}")

        return system_prompt + user_prompt, "Failed to get a response"


class Llama(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        model_id = "alokabhishek/Meta-Llama-3-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

    def _get_response(self, system_prompt, user_prompt):
        inputs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.pipeline(inputs, max_new_tokens=512, top_p=0.9, temperature=0.1)
        return response[0]["generated_text"][-1]['content']

    def response(self, text: dict):
        system_prompt, user_prompt = super().response(text)
        response = self._get_response(system_prompt, user_prompt)
        return system_prompt + user_prompt, response


class Qwen(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

    def _get_response(self, system_prompt, user_prompt):
        inputs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.pipeline(inputs, max_new_tokens=512, top_p=0.9, temperature=0.1)
        return response[0]["generated_text"][-1]['content']

    def response(self, text: dict):
        system_prompt, user_prompt = super().response(text)
        response = self._get_response(system_prompt, user_prompt)
        return system_prompt + user_prompt, response


class Gemma(BaseModel):
    def __init__(self, api_key, args):
        super().__init__(api_key, args)
        model_id = "google/gemma-2-9b-it"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

    def _get_response(self, system_prompt, user_prompt):
        inputs = [
            {"role": "user", "content": system_prompt + user_prompt},
        ]
        response = self.pipeline(inputs, max_new_tokens=512, top_p=0.9, temperature=0.1)
        return response[0]["generated_text"][-1]['content']

    def response(self, text: dict):
        system_prompt, user_prompt = super().response(text)
        response = self._get_response(system_prompt, user_prompt)
        return system_prompt + user_prompt, response


# Ollama 모델 추가
class OllamaBase(BaseModel):
    def __init__(self, api_key, args, model_name):
        super().__init__(api_key, args)
        self.client = Client(host='http://localhost:11434')
        self.model_name = model_name

    def _get_response(self, system_prompt, user_prompt):
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],

                options={"temperature": 0.0, "num_predict":128},
            )

            return response['message']['content']
        except Exception as e:
            print(f"Error during Ollama response: {e}")
            return None

    def response(self, text: dict):
        system_prompt, user_prompt = super().response(text)
        response = self._get_response(system_prompt, user_prompt)
        return system_prompt + user_prompt, response

class OllamaLlama(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='llama3.1:8b-instruct')


class OllamaQwen(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='qwen2:7b-instruct')


class OllamaGemma(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='gemma2:9b-instruct')


class OllamaMistral(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='mistral:7b-instruct-v0.3')
class OllamaPhi(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='phi3:14b-medium-4k-instruct')
class OllamaQwen32B(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='qwen:32b-chat-v1.5')
class OllamaPhi3_5(OllamaBase):
    def __init__(self, api_key, args):
        super().__init__(api_key, args, model_name='phi3.5:3.8b-mini-instruct')

def load_api_keys():
    with open('api_keys.json', 'r') as file:
        return json.load(file)

def load_model(model_name, args):
    api_keys = load_api_keys()

    if model_name == 'Gemini':
        return Gemini(api_key=api_keys['Gemini'], args=args)
    elif model_name == 'Sonnet':
        return Claude(api_key=api_keys['Sonnet'], args=args)
    elif model_name == 'GPT4o':
        return GPT(api_key=api_keys['GPT4o'], args=args)
    elif model_name == 'Ollama_Llama':
        return OllamaLlama(api_key=api_keys['Ollama_Llama'], args=args)
    elif model_name == 'Ollama_Qwen':
        return OllamaQwen(api_key=api_keys['Ollama_Qwen'], args=args)
    elif model_name == 'Ollama_Gemma':
        return OllamaGemma(api_key=api_keys['Ollama_Gemma'], args=args)
    elif model_name == 'Ollama_Mistral':
        return OllamaMistral(api_key=api_keys['Ollama_Mistral'], args=args)
    elif model_name == 'Ollama_Phi':
        return OllamaPhi(api_key=api_keys['Ollama_Phi'], args=args)
    elif model_name == 'Ollama_Qwen32B':
        return OllamaQwen32B(api_key=api_keys['Ollama_Qwen32B'], args=args)
    elif model_name == 'OllamaPhi3_5':
        return OllamaPhi3_5(api_key=api_keys['OllamaPhi3_5'], args=args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



# 요청 예시
if __name__ == "__main__":
    args = type('Args', (object,), {"shot": 0})  # Args 객체 예시
    model = load_model('OllamaLlama', args)
    prompt = {
        "system_instruction": "You are an assistant.",
        "task_query": "Why is the sky blue?",
        "few_shot": "",
        "prompt_strategy": "",
        "context": "",
        "label_def": "",
        "output_indicator": ""
    }
    response = model.response(prompt)
    print(response)