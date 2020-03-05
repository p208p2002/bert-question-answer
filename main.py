from core import use_model,convert_single_data_to_feature,make_torch_data_loader,make_torch_dataset
import os
import logging
logging.getLogger('core').setLevel(logging.DEBUG)

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"trained_model/config.json", 
        "model_file_path":"trained_model/pytorch_model.bin", 
        "vocab_file_path":"trained_model/vocab.txt"
    }    
    model,tokenizer = use_model(**model_setting)

    context = "「部落是個笑話！」 希瓦娜斯・風行者在撂下這樣的狠話之後，便背棄自己曾經對部落立下的誓言抽身離去。如今，黑暗女王與她的部下正暗中密謀著，而部落與聯盟則是努力想要追查出她的下一步行動，就連她的親姊姊艾蘭里亞也參與其中。身為領袖的安杜因國王身負重責大任且疲於奔命，因此，他決定委託這位虛無精靈以及大主教圖拉揚去調查希瓦娜斯的下落。 部落現在正處在命運的十字路口。各個陣營決定組成聯合議會，不再延續設立大酋長的傳統。索爾、洛索瑪・塞隆、貝恩・血蹄、首席秘法師薩莉瑟拉和許多大家所熟識的角色都紛紛挺身而出，想要貢獻一己之力。但是無數的威脅正籠罩著部落，眾人彼此之間也充滿了猜忌與懷疑。 在塔蘭姬（身為贊達拉女王的她是一名不可或缺的盟友）險些遭到刺殺之後，議會眼看就要土崩瓦解，索爾與其他部落的領袖因此被迫採取行動。他們提拔至今仍無法忘懷瓦洛克・薩魯法爾之死的澤坎，並賦予他一件重要的任務，讓他去協助塔蘭姬，看看究竟是什麼樣的威脅正在向她逼近。 與此同時，納薩諾斯・凋零者和西拉・月守接獲黑暗女王的命令，準備採取一次極為大膽的行動，那就是殺害食人妖的死亡羅亞「伯昂撒姆第」。 澤坎和塔蘭姬出發拯救伯昂撒姆第的這趟旅途，將決定部落是否能成長茁壯並對抗即將襲來的黑暗，也將幫助他們兩個重新認識自己並且不再迷惘。如果他們無法成功拯救部落的盟友和這位狡詐的神靈，後果勢必不堪設想；但倘若此次的計畫成功，將可以幫助部落重拾那段曾經堅強的過往。「部落是個笑話！」 希瓦娜斯・風行者在撂下這樣的狠話之後，便背棄自己曾經對部落立下的誓言抽身離去。如今，黑暗女王與她的部下正暗中密謀著，而部落與聯盟則是努力想要追查出她的下一步行動，就連她的親姊姊艾蘭里亞也參與其中。身為領袖的安杜因國王身負重責大任且疲於奔命，因此，他決定委託這位虛無精靈以及大主教圖拉揚去調查希瓦娜斯的下落。 部落現在正處在命運的十字路口。各個陣營決定組成聯合議會，不再延續設立大酋長的傳統。索爾、洛索瑪・塞隆、貝恩・血蹄、首席秘法師薩莉瑟拉和許多大家所熟識的角色都紛紛挺身而出，想要貢獻一己之力。但是無數的威脅正籠罩著部落，眾人彼此之間也充滿了猜忌與懷疑。 在塔蘭姬（身為贊達拉女王的她是一名不可或缺的盟友）險些遭到刺殺之後，議會眼看就要土崩瓦解，索爾與其他部落的領袖因此被迫採取行動。他們提拔至今仍無法忘懷瓦洛克・薩魯法爾之死的澤坎，並賦予他一件重要的任務，讓他去協助塔蘭姬，看看究竟是什麼樣的威脅正在向她逼近。 與此同時，納薩諾斯・凋零者和西拉・月守接獲黑暗女王的命令，準備採取一次極為大膽的行動，那就是殺害食人妖的死亡羅亞「伯昂撒姆第」。 澤坎和塔蘭姬出發拯救伯昂撒姆第的這趟旅途，將決定部落是否能成長茁壯並對抗即將襲來的黑暗，也將幫助他們兩個重新認識自己並且不再迷惘。如果他們無法成功拯救部落的盟友和這位狡詐的神靈，後果勢必不堪設想；但倘若此次的計畫成功，將可以幫助部落重拾那段曾經堅強的過往。"
    question = "什麼是笑話"

    token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list = convert_single_data_to_feature(context,question,tokenizer,doc_strike=128)
    qc_dataset = make_torch_dataset(token_embeddings_list, segment_embeddings_lsit, attention_embeddings_list)
    qc_data_loader = make_torch_data_loader(qc_dataset,batch_size=3)