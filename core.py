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
    bert_input_len_limit = 512 # bert input length limit
    special_token_length = 3 # [CLS]A[SEP]B[SEP]
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    #
    question = question[:100] # limit question length
    len_limit_remain = bert_input_len_limit - len(question)
    context_ids = convert_text_to_ids(context)

    logger.debug(context_ids)
    logger.debug(tokenizer.decode(context_ids))
    logger.debug(len_limit_remain)

    next_index = 0
    index = next_index
    window_size = len_limit_remain - special_token_length
    logger.debug("context length:%d"%(len(context)))
    token_embeddings_list = []
    while(True):
        next_index = index+doc_strike
        logger.debug("start_index:%d window_size:%d next_index:%d"%(index, window_size, next_index))
        
        input_context = context[index:index+window_size]
        logger.debug("input context len:%d"%(len(input_context)))

        full_input = cls_token + input_context + sep_token + question + sep_token        
        logger.debug(full_input)
        
        token_embeddings = convert_text_to_ids(full_input)
        assert len(token_embeddings) <= bert_input_len_limit
        token_embeddings_list .append(token_embeddings)

        logger.debug('input token length:%d\n',len(token_embeddings))

        # already process the end of input context
        if(len(input_context) < window_size):
            break

        #
        index = next_index
    #
    return token_embeddings_list

