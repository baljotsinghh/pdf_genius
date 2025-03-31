import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Hugging Face Model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 100},
    huggingfacehub_api_token=os.getenv("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")  # Load from environment
)

# Define the prompt template
prompt = PromptTemplate(template="What is {subject}?", input_variables=["subject"])

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with user input
response = chain.run("what is data analytics")

# Print the response
print(response)

