import streamlit as st

# Import Transformer classes for generation
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# import torch for datatype Attributes 
import torch

#Import the prompt wraper but for the llama index
from llama_index.prompts.prompts import SimpleInputPrompt
#create a system prompt
# imort the llama index HF wrapper
from llama_index.llms import HuggingFaceLLM
#bring in embeddings wraer 
from llama_index.embeddings import LangchainEmbedding
#bring in HF embeddings-need these to represent documents chunks
from langchain.embeddings.huggingface import  HuggingFaceBgeEmbeddings
# bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
from pathlib import Path

# set meta llma variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-13b-chat-hf"
#set auth tokenizer variable from hugging face
auth_token = "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"

# creating autotokenizer
tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = './models/', use_auth_token = auth_token)
model = AutoModelForCausalLM.from_pretrained(name,cache_dir = './models/', use_auth_token= auth_token, torch_dtype = torch.float32, rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=False)

#create a system prompt
system_promt = """<s>[INST] <<SYS>> you are a helpful, resecful and honest assistant <</SYS>> """
# throw together the quesry wrapper 
query_wrapper_promt = SimpleInputPrompt("{query_str}[/INST]")

# imort the llama index HF wrapper
from llama_index.llms import HuggingFaceLLM
# create a HF LLM using the llama index wraer
llm = HuggingFaceLLM(context_window=4096,
                     max_new_tokens=256,
                     system_prompt=system_promt,
                     query_wrapper_prompt=query_wrapper_promt,
                     model=model,
                     tokenizer=tokenizer)

# create and dl embeddings instance 
embneddings = LangchainEmbedding(
    HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
)

# create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embneddings
)
# and set the service context actually passing service_context 
set_global_service_context(service_context)

prompt = st.text_input("Please enter your question ")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt= True, skip_special_tokens = True)

if 'token_type_ids' in inputs:# Check if 'token_type_ids' is present and remove it
    del inputs['token_type_ids']

model = model.to(torch.float32)# Convert the model and inputs to 32-bit precision

inputs = inputs.to(torch.float32)# Convert the model and inputs to 32-bit precision

output = model.generate(**inputs, streamer=streamer,use_cache=True, max_new_tokens=float('inf')) 

st.title('Shinzou Sasageyou [write your things]')


if prompt:
    response = query_engine.query(prompt)
    st.write(response)