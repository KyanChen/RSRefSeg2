import torch
import sys
sys.path.append(sys.path[0] + '/../..')


if __name__ == "__main__":
	model = torch.load("work_dirs/refsegrs/best_RefSeg_gIoU_1_epoch_110.pth")
	import ipdb; ipdb.set_trace()
	torch.save(model, "work_dirs/refsegrs/best_RefSeg_gIoU_1_epoch_110.pth")