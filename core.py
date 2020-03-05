import logging
logger = logging.getLogger(__name__)
def use_model(model_name, config_file_path, model_file_path, vocab_file_path):
    # 選擇模型並加載設定
    if(model_name == 'bert'):
        from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
        model_config, model_class, model_tokenizer = (BertConfig, BertForQuestionAnswering, BertTokenizer)
        config = model_config.from_pretrained(config_file_path)
        model = model_class.from_pretrained(model_file_path, from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
        tokenizer = model_tokenizer(vocab_file=vocab_file_path)
        return model, tokenizer


def convert_data_to_feature(context,question,tokenizer,doc_strike=128):
    """convert string data to bert input, also deal with long context."""
    def convert_text_to_ids(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    
    # def
    len_limit_remain = 512 # bert input length limit
    special_token_length = 2
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    #
    question = question[:100] # limit question length
    len_limit_remain -= len(question)
    context_ids = convert_text_to_ids(context)

    logger.debug(context_ids)
    logger.debug(tokenizer.decode(context_ids))
    logger.debug(len_limit_remain)

    index = 0
    window_size = len_limit_remain - special_token_length
    logger.debug("context length:%d"%(len(context)))
    while(len(context) > window_size):        
        logger.debug("start_index:%d"%(index))
        input_context = context[index:index+window_size]
        input_context = cls_token + input_context + sep_token
        logger.debug(input_context)

        #
        index += window_size
        context = context[index:]


