from bert_qa import BertQA
from bert_qa.core import use_model
import os
import logging
import torch
if __name__ == "__main__":    
    # close transformers logging
    logging.getLogger("transformers.file_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

    # init model
    model,tokenizer = use_model(model_name="bert",model_path="trained_model/")

    # env and device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init BertQA
    bertQA = BertQA(model = model, tokenizer = tokenizer, device = device)

    # ask
    context="大同國小有三個職員，王大明是校長，張小美是秘書，陳小玉是總務長"
    question = "陳小玉是什麼"
    answer_results = bertQA.ask(context,question)
    for answer_result in answer_results:
        # print('input_decode',answer_result[5])
        print("score:%3.5f start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"\
            %(answer_result[1]*answer_result[3],answer_result[0],answer_result[1],answer_result[2],answer_result[3],answer_result[4]))
