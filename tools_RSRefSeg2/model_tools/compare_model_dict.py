import torch
import sys
sys.path.append(sys.path[0] + '/../..')

def compare_models(dict1, dict2):

	# 检查键是否一致
	if dict1.keys() != dict2.keys():
		raise ValueError("模型结构不同，无法比较")

	same_parents = set()

	for key in dict1:
		if not torch.equal(dict1[key], dict2[key]):
			if 'lora' in key or 'prompter' in key or 'sam_mask_decoder' in key:
				continue
			parts = key.split('.')
			same_parents.add('.'.join(parts))

	return same_parents


if __name__ == "__main__":
	model1 = torch.load("work_dirs/mp_rank_00_model_states.pt")['module']
	model2 = torch.load("work_dirs/epoch_200.pth/mp_rank_00_model_states.pt")['module']

	same_parents = compare_models(model1, model2)

	print("相同的父模块:")
	for parent in same_parents:
		print(parent)