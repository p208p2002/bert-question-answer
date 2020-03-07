# Bert QA
BERT QA 問答模組
## Usage
- 安裝相依套件: `pip install -r requirements.txt`
- 至release下載預訓練模型
- 詳細用法見example.py
```python
# init BertQA
bertQA = BertQA(model = model, tokenizer = tokenizer, device = device)
context="大同國小有三個職員，王大明是校長，張小美是秘書，陳小玉是總務長"

question = "誰是校長"
answer_results,input_decode = bertQA.ask(context,question)
# score:2.17795 start_index:11(1.07034) end_index:13(1.10761) answer:王大明

question = "陳小玉擔的工作是什麼"
answer_results,input_decode = bertQA.ask(context,question)
# score:2.07151 start_index:29(1.84568) end_index:31(0.22583) answer:總務長
```