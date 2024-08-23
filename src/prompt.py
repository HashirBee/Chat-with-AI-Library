## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
"""
Please answer the questions accurately based on the following context only.
<context>
{context}
<context>
Questions:{input}

Please format the answer into well-structured html. Use appropriate tags
for paragraphs like <p> and tags like <h4> or <h5> or <h6> for headings) 
instead of plain text.
"""
)