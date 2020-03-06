from core import use_model,convert_single_data_to_feature,make_torch_data_loader,make_torch_dataset,to_list, \
    _get_best_indexes,_check_has_skip_token,_check_segment_type_is_b
import os
import logging
import torch
if __name__ == "__main__":    
    # log & env setting
    logging.getLogger('core').setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # init model
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"trained_model/", 
        "model_file_path":"trained_model/", 
        "vocab_file_path":"trained_model/vocab.txt"
    }    
    model,tokenizer = use_model(**model_setting)

    context = "試駕車未選配PCCB陶瓷複合煞車系統，而是採用標配設定的前六活塞、後四活塞煞車卡鉗與380mm打孔通風碟盤，在這次試駕過程中我對於制動性能表現已覺得相當夠用，但如果有上賽道的需求或是想升級的車主，亦可加價選配PCCB陶瓷複合煞車系統，除了碟盤尺寸加大至前420mm、後390mm之外，陶瓷複合材質亦具備輕量化的效果，可藉此提升操控與制動效果極限"
    question = "可以選配什麼"

    token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list = convert_single_data_to_feature(context,question,tokenizer,doc_strike=128)
    qc_dataset = make_torch_dataset(token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list)
    qc_data_loader = make_torch_data_loader(qc_dataset,batch_size=1)

    for index,batch in enumerate(qc_data_loader):
        start_scores, end_scores = model(input_ids=batch[0],token_type_ids=batch[1],attention_mask=batch[2])

        start_scores = to_list(start_scores.squeeze(0))
        end_scores = to_list(end_scores.squeeze(0))

        start_indexs = _get_best_indexes(start_scores,n_best_size=10)
        end_indexs = _get_best_indexes(end_scores,n_best_size=10)
        input_decode = tokenizer.convert_ids_to_tokens(batch[0].squeeze(0))
        
        logger.debug(input_decode)
        logger.debug(to_list(batch[0].squeeze(0)))
        logger.debug(start_indexs)
        logger.debug(end_indexs)

        for start_index in start_indexs:
            for end_index in end_indexs:
                end_index += 1
                answer_token = input_decode[start_index:end_index]
                if(len(answer_token) == 0):
                    continue
                elif(_check_has_skip_token(check_tokens = answer_token, skip_tokens = ['[CLS]','[SEP]','[PAD]','[UNK]'])):
                    continue
                elif(_check_segment_type_is_b(start_index,end_index,batch[1].squeeze(0))):
                    continue
                answer = "".join(answer_token)
                logger.debug("batch_index:%d"%(index))
                logger.debug("start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"%(start_index,start_scores[start_index],end_index,end_scores[end_index],answer[:16]))