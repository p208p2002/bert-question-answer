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

    #
    len_limit_remain = bert_input_len_limit - len(question)
    context_ids = convert_text_to_ids(context)
    question_ids = convert_text_to_ids(question)
    question_ids = question_ids[:100] # limit question length

    logger.debug("len_limit_remain:%d"%(len_limit_remain))

    next_index = 0
    index = next_index
    window_size = len_limit_remain - special_token_length
    logger.debug("context length:%d"%(len(context_ids)))

    # bert inputs
    token_embeddings_list = []
    segment_embeddings_lsit = []
    attention_embeddings_list = []
    logger.debug("convert to feature and process doc strike\n")
    while(True):
        next_index = index+doc_strike
        logger.debug("start_index:%d window_size:%d next_index:%d"%(index, window_size, next_index))
        
        input_context_ids = context_ids[index:index+window_size]
        logger.debug("input context len:%d"%(len(input_context_ids)))

        token_embeddings = tokenizer.build_inputs_with_special_tokens(input_context_ids,question_ids)
        segment_embeddings = [0]*(len(input_context_ids)+2) + [1]*(len(question_ids)+1)
        attention_embeddings = [1]*len(token_embeddings)
        logger.debug('input token length:%d',len(token_embeddings))

        # padding
        padding_length = bert_input_len_limit - len(token_embeddings)
        logger.debug("padding_length:%d\n"%(padding_length))
        token_embeddings = token_embeddings + [0]*padding_length
        segment_embeddings = segment_embeddings + [0]*padding_length
        attention_embeddings = attention_embeddings + [0]*padding_length

        assert len(token_embeddings) == bert_input_len_limit
        assert len(token_embeddings) == len(segment_embeddings) == len(attention_embeddings)

        # already process the end of input context
        if(len(input_context_ids) < window_size):
            break

        #
        index = next_index
    
    return token_embeddings_list

