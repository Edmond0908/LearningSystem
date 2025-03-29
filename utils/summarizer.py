from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def summarize_with_palm(text: str) -> str:
    # Use Gemini Pro 1.5 for summarization
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-001",
        temperature=0.7,
        max_output_tokens=512,
        timeout=60,
        max_retries=3,
    )

    prompt = PromptTemplate(
        template="請將以下內容做摘要：\n\n{text}\n\n摘要:", input_variables=["text"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text)
