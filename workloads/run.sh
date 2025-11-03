# CUDA_VISIBLE_DEVICES=0 python BERT_base_on_wiki.py > BERT_base_on_wiki_out.txt
# CUDA_VISIBLE_DEVICES=0 python BERT_large_on_wiki.py > BERT_large_on_wiki_out.txt

CUDA_VISIBLE_DEVICES=0,1 python gpt2_large_on_wiki.py > gpt2_large_on_wiki_two_gpu_out.txt
# CUDA_VISIBLE_DEVICES=0 python gpt2_xl_on_wiki.py > gpt2_xl_on_wiki_out.txt

CUDA_VISIBLE_DEVICES=0,1 python xlnet_base_cased.py > xlnet_base_cased_two_GPU_out.txt
CUDA_VISIBLE_DEVICES=0,1 python xlnet_large_cased.py > xlnet_large_cased_two_GPU_out.txt