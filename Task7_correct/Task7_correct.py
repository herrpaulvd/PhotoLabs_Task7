from mmdet.apis import init_detector, inference_detector

inputFileName = input("Input file name:")

config_file = './configs/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
model.show_result(inputFileName, inference_detector(model, inputFileName), out_file=inputFileName + ".output.png")
print('Successful')
