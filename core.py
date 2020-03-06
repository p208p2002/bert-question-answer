import logging
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

logger = logging.getLogger(__name__)
def use_model(model_name, config_file_path, model_file_path, vocab_file_path):
    if(model_name == 'bert'):
        from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
        model_config, model_class, model_tokenizer = (BertConfig, BertForQuestionAnswering, BertTokenizer)
        config = model_config.from_pretrained(config_file_path)
        model = model_class.from_pretrained(model_file_path, from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
        tokenizer = model_tokenizer(vocab_file=vocab_file_path,do_lower_case=False)
        return model, tokenizer

def make_torch_dataset(*features):
    tensor_features = []
    for feature in features:
        tensor_feature = torch.tensor([f for f in feature])
        tensor_features.append(tensor_feature)
    return TensorDataset(*tensor_features)

def make_torch_data_loader(torch_dataset,**options):
    # options: batch_size=int,shuffle=bool
    return DataLoader(torch_dataset,**options)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def _check_has_skip_token(check_tokens,skip_tokens):
    for check_token in check_tokens:
        for skip_token in skip_tokens:
            if check_token == skip_token:
                return True
    return False

def _check_segment_type_is_b(start_index,end_index,segment_embeddings):
    tag_segment_embeddings = segment_embeddings[start_index]
    if 1 in tag_segment_embeddings:
        return True
    return False

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def convert_single_data_to_feature(context,question,tokenizer,doc_strike=128):
    """convert single string data to bert input, also deal with long context."""
    def convert_text_to_ids(text,require_str_token = False):
        str_token = tokenizer.tokenize(text)
        ids_token = tokenizer.convert_tokens_to_ids(str_token)
        if(require_str_token):
            return ids_token,str_token
        return ids_token
    
    # def
    bert_input_len_limit = 512 # bert input length limit
    special_token_length = 3 # [CLS]A[SEP]B[SEP]

    #
    len_limit_remain = bert_input_len_limit - len(question)
    context_ids,context_str_tokens = convert_text_to_ids(context,require_str_token=True)
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
    raw_input_token_list = []
    logger.debug("convert to feature and process doc strike\n")
    while(True):
        next_index = index+doc_strike
        logger.debug("start_index:%d window_size:%d next_index:%d"%(index, window_size, next_index))
        
        input_context_ids = context_ids[index:index+window_size]
        input_context_strs = context_str_tokens[index:index+window_size]
        logger.debug("input context len:%d"%(len(input_context_ids)))

        token_embeddings = tokenizer.build_inputs_with_special_tokens(input_context_ids,question_ids)
        segment_embeddings = [0]*(len(input_context_ids)+2) + [1]*(len(question_ids)+1)
        attention_embeddings = [1]*(len(input_context_ids)+2) + [1]*(len(question_ids)+1)
        logger.debug('input token length:%d',len(token_embeddings))

        # padding
        padding_length = bert_input_len_limit - len(token_embeddings)
        logger.debug("padding_length:%d\n"%(padding_length))
        token_embeddings = token_embeddings + [0]*padding_length
        segment_embeddings = segment_embeddings + [0]*padding_length
        attention_embeddings = attention_embeddings + [0]*padding_length

        # assert len(token_embeddings) == bert_input_len_limit
        assert len(token_embeddings) == len(segment_embeddings) == len(attention_embeddings)

        # save to list
        token_embeddings_list.append(token_embeddings)
        segment_embeddings_lsit.append(segment_embeddings)
        attention_embeddings_list.append(attention_embeddings)
        raw_input_token_list.append(input_context_strs)

        # already process the end of input context
        if(len(input_context_ids) < window_size):
            break

        #
        index = next_index
    
    return (
        token_embeddings_list,
        segment_embeddings_lsit,
        attention_embeddings_list,
        raw_input_token_list
    )

