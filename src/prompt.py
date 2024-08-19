## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
"""
Please answer the questions accurately based on the following context only.
<context>
{context}
<context>
Questions:{input}

Please format the answer into paragraphs that are logical and increase readability.
"""
)