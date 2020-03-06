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

    # context = "試駕車未選配PCCB陶瓷複合煞車系統，而是採用標配設定的前六活塞、後四活塞煞車卡鉗與380mm打孔通風碟盤，在這次試駕過程中我對於制動性能表現已覺得相當夠用，但如果有上賽道的需求或是想升級的車主，亦可加價選配PCCB陶瓷複合煞車系統，除了碟盤尺寸加大至前420mm、後390mm之外，陶瓷複合材質亦具備輕量化的效果，可藉此提升操控與制動效果極限。車主在購買911 GT3時，可以另外選配Clubsport套件，而這組套件是不需額外付費的，套件內容包括一組座艙後方的防滾籠、紅色六點式安全帶以及一組安裝於副手座前方的滅火器，雖然會犧牲不少車內空間，但對於增進戰鬥氛圍具有加倍的效果。方向盤採用麂皮材質包覆，握感與止滑效果都很棒，另外可以發現方向盤右下方並未配置991.2世代所追加的駕駛模式選擇轉盤，正因為本車採用自然進氣引擎設定，未提供Sport Response功能；但其實開過GT3就知道，本車是直接預設Sport模式起跳。動力升級之後的991.2 GT3，加速表現也較前期型略有進步，0~100km/h加速成績為3.4秒（前期型GT3為3.5秒）、極速為318km/h（前期型為315km/h）；雖然加速表現的進步幅度不算明顯，但實際駕馭可說是相當有感，先前試駕前期型911 GT3時，可以感覺到變速箱的動力銜接在起步時不夠明快，除非是以Launch control模式彈射起步，否則就會覺得有細微的動力空窗期產生，這次試駕的991.2 GT3針對此點可說是徹底改善，起步時的動力銜接反應順暢無比，車速攀升的步調也很有效率，讓人感覺到未有一分一毫的動力被浪費掉。"
    # question = "預設什麼模式"
    context="大同國小有三個職員，王大明是校長，張小美是秘書，陳小玉是總務長"
    question = "陳小玉是什麼"

    token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list = convert_single_data_to_feature(context,question,tokenizer,doc_strike=128)
    qc_dataset = make_torch_dataset(token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list)
    qc_data_loader = make_torch_data_loader(qc_dataset,batch_size=1)

    model.eval()
    for index,batch in enumerate(qc_data_loader):
        start_scores, end_scores = model(input_ids=batch[0],token_type_ids=batch[1],attention_mask=batch[2])

        start_scores = to_list(start_scores.squeeze(0)) ##
        end_scores = to_list(end_scores.squeeze(0))

        start_indexs = _get_best_indexes(start_scores,n_best_size=20)
        end_indexs = _get_best_indexes(end_scores,n_best_size=20)
        input_decode = tokenizer.convert_ids_to_tokens(batch[0].squeeze(0))
        
        # logger.debug(input_decode)
        # logger.debug(to_list(batch[0].squeeze(0)))
        logger.debug(start_indexs)
        logger.debug(end_indexs)

        answer_results = []
        for start_index in start_indexs:
            for end_index in end_indexs:
                # end_index += 1
                answer_token = input_decode[start_index:end_index+1]
                if(len(answer_token) == 0 or len(answer_token)>30):
                    continue
                elif(_check_has_skip_token(check_tokens = answer_token, skip_tokens = ['[CLS]','[SEP]','[PAD]'])):
                    continue
                elif(_check_segment_type_is_b(start_index,end_index,batch[1].squeeze(0))):
                    continue
                answer = "".join(answer_token)
                answer_result = (start_index,start_scores[start_index],end_index,end_scores[end_index],answer)
                answer_results.append(answer_result)
                logger.debug("batch_index:%d"%(index))
                # logger.debug("start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"%(answer_result[0],answer_result[1],answer_result[2],answer_result[3],answer_result[4]))
        
        answer_results = sorted(answer_results,key=lambda answer_result:answer_result[1]+answer_result[3],reverse=True)
        for answer_result in answer_results:
            logger.debug("score:%3.5f start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"\
                %(answer_result[1]+answer_result[3],answer_result[0],answer_result[1],answer_result[2],answer_result[3],answer_result[4]))