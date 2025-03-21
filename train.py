from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8-modified.yaml")
    # 训练模型
    # YOLOv8n-p2p.yaml
    # yolov8-modified.yaml
    results = model.train(data="ultralytics/cfg/datasets/SIRST-UAVB.yaml",
                          resume=True,
                          epochs=700,
                          patience=80,
                          name='pconv-yolov8-200',
                          amp=False,
                          device=[4, 5, 6, 7],
                          batch=16,)
