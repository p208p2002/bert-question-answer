from .core import use_model,convert_single_data_to_feature,make_torch_data_loader,make_torch_dataset,to_list, \
    _get_best_indexes,_check_has_skip_token,_check_segment_type_is_b
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
class BertQA():
    def __init__(self,model,tokenizer,device,logging_level=logging.WARNING):
        #
        logger.setLevel(logging_level)
        #
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        #
        self.model.to(device)
        self.model.eval()
    
    def ask(self,context,question,n_best_size=20,batch_size=4):
        '''return format: [(start_index,start_scores,end_index,end_scores,answer,input_decode)]'''
        #
        model,tokenizer,device = self.model, self.tokenizer, self.device
        
        #
        token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list = convert_single_data_to_feature(context,question,tokenizer,doc_strike=128)
        qc_dataset = make_torch_dataset(token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list)
        qc_data_loader = make_torch_data_loader(qc_dataset,batch_size=batch_size)
        answer_results = []
        pbar = tqdm(total=len(qc_data_loader))
        for index,batch in enumerate(qc_data_loader):
            batch = [x.to(device) for x in batch]
            start_scores, end_scores = model(input_ids=batch[0],token_type_ids=batch[1],attention_mask=batch[2])

            batch_start_scores = to_list(start_scores) ##
            batch_end_scores = to_list(end_scores)

            for i,(start_scores,end_scores) in enumerate(zip(batch_start_scores,batch_end_scores)):
                start_indexs = _get_best_indexes(start_scores,n_best_size=n_best_size)
                end_indexs = _get_best_indexes(end_scores,n_best_size=n_best_size)
                input_decode = tokenizer.convert_ids_to_tokens(batch[0][i])
            
                logger.debug(start_indexs)
                logger.debug(end_indexs)

                for start_index in start_indexs:
                    for end_index in end_indexs:
                        # end_index += 1
                        answer_token = input_decode[start_index:end_index+1]
                        if(len(answer_token) == 0 or len(answer_token)>30):
                            continue
                        elif(_check_has_skip_token(check_tokens = answer_token, skip_tokens = ['[CLS]','[SEP]','[PAD]'])):
                            continue
                        elif(_check_segment_type_is_b(start_index,end_index,batch[1][i])):
                            continue
                        answer = "".join(answer_token)
                        answer_result = (start_index,start_scores[start_index],end_index,end_scores[end_index],answer,input_decode)
                        answer_results.append(answer_result)
                        logger.debug("batch_index:%d"%(index))
                        # logger.debug("start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"%(answer_result[0],answer_result[1],answer_result[2],answer_result[3],answer_result[4]))
            pbar.update()
                
        answer_records = []
        answer_results = sorted(answer_results,key=lambda answer_result:answer_result[1]+answer_result[3],reverse=True)

        pbar = tqdm(total=n_best_size)
        n_best_answer_results = []
        for answer_result in answer_results[:]:
            answer_tag = answer_result[4]
            if(answer_tag not in answer_records):
                answer_records.append(answer_tag)
                n_best_answer_results.append(answer_result)
            else:
                # answer_results.remove(answer_result)
                continue
            logger.debug("score:%3.5f start_index:%d(%3.5f) end_index:%d(%3.5f) answer:%s"\
                %(answer_result[1]+answer_result[3],answer_result[0],answer_result[1],answer_result[2],answer_result[3],answer_result[4]))

            pbar.update()            
            if(len(n_best_answer_results) >= n_best_size):
                return n_best_answer_results
        return n_best_answer_results
        
        