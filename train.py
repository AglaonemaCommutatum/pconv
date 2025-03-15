from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8-pconv.yaml")
    # 训练模型
    results = model.train(data="ultralytics/cfg/datasets/SIRST-UAVB.yaml",
                          resume=True,
                          epochs=200,
                          patience=30,
                          name='pconv-yolov8-200',
                          amp=False,
                          device=[0, 1, 2, 3])
