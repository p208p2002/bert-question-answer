def use_model(model_name, config_file_path, model_file_path, vocab_file_path, num_labels):
    # 選擇模型並加載設定
    if(model_name == 'bert'):
        from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
        model_config, model_class, model_tokenizer = (BertConfig, BertForQuestionAnswering, BertTokenizer)
        config = model_config.from_pretrained(config_file_path,num_labels = num_labels)
        model = model_class.from_pretrained(model_file_path, from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
        tokenizer = model_tokenizer(vocab_file=vocab_file_path)
        return model, tokenizer