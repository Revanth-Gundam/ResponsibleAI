from transformers import AutoModel, AutoTokenizer
from data_gen.collector_base import ResponseCollectorBase

class ChatAlpaca7B(ResponseCollectorBase):
    def __init__(
        self,
        mname,
        device='cuda'
    ) -> None:
        super().__init__(mname, device)
        
        self.model = AutoModel.from_pretrained("EleutherAI/alpaca-7.1B", trust_remote_code=True).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/alpaca-7.1B", trust_remote_code=True)

        self.max_context_length = 4096
            
    def tokenizer_encode(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False)
    
    def tokenizer_decode(self, encoded):
        return self.tokenizer.decode(encoded)
    
    def get_response(self, prompt):        
        res = self.model.generate(input_ids=self.tokenizer.encode(prompt, return_tensors='pt').to(self.device), max_length=100)
        res = self.tokenizer.decode(res[0], skip_special_tokens=True)
        return res