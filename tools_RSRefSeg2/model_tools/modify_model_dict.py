import torch
import sys
sys.path.append(sys.path[0] + '/../..')


if __name__ == "__main__":
	model = torch.load("../refseg/work_dirs/rrsisd_6917.pth")
	new_model = {'state_dict': model['state_dict']}
	torch.save(new_model, "../refseg/work_dirs/rrsisd.pth")