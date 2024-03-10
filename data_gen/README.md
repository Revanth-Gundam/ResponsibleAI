# Data Generation Pipeline

- Natural Questions Dataset:
    - The Natural Questions (NQ) dataset is a comprehensive question answering dataset developed by Google AI. It comprises real user questions from Google search paired with answers extracted from Wikipedia. With over 300,000 training examples, this dataset serves as a benchmark for training and evaluating question answering systems.

- How Data Generation works:
    - We hosted three models, mistral.py, alpaca-7b, and chatglm3_6b locally and generated the results that way. 
    - Additionally we prompted several SOTA models, such as Llama-70b Chat, Claude-2 and ChatGPT (manually without API key), and prepared a dataset that way. 
    - Prompt Used:
        closed_qa_prompt = 
        ```
        """Instruction: Provide a well-formed answer to the question using information from the given context.
                    Question: {question}
                    Context: {context}
                    """
        ``` 
    - 
