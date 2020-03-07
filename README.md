# BERT QA
BERT QA 問答模組
```python
# init BertQA
bertQA = BertQA(model = model, tokenizer = tokenizer, device = device)
context="大同國小有三個職員，王大明是校長，張小美是秘書，陳小玉是總務長"

question = "誰是校長"
answer_results,input_decode = bertQA.ask(context,question)
# score:2.17795 start_index:11(1.07034) end_index:13(1.10761) answer:王大明

question = "陳小玉的工作是什麼"
answer_results,input_decode = bertQA.ask(context,question)
# score:2.07151 start_index:29(1.84568) end_index:31(0.22583) answer:總務長
```
## Usage
0. 確保已經安裝pytorch1.4+
1. 安裝相依套件: `pip install -r requirements.txt`
2. 至[release](https://github.com/p208p2002/bert-question-answer/releases)下載預訓練模型
3. 將下載的模型放到`trained_model/`下
4. 詳細用法見`example.py`
