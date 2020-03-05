from core import use_model
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"trained_model/config.json", 
        "model_file_path":"trained_model/pytorch_model.bin", 
        "vocab_file_path":"trained_model/vocab.txt"
    }    
    model,tokenizer = use_model(**model_setting)

    context = "王大明在文德國小擔任校長"
    question = "誰擔任校長"