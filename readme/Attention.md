1. 本代码要求的是数据格式为 x,y,w,h
2. 数据生成
    - 在终端输入gpu_server_2登录服务器，进入yolo_ks_frac目录
    ~~~
    sh train.sh
    ~~~
    - 将生成的训练和测试ground truth复制到CenterNet_xing/data/ks文件夹下
    ~~~
    cp data_lib/dataset/ks_t* ../CenterNet_xing/data/ks/
    ~~~
    - 进入CenterNet_xing/，删除原有data/ks/*.json文件
    ~~~
    cd ../CenterNet_xing
    rm -rf data/ks/*.json
    ~~~
    - 运行如下代码重新生成.json文件
    ~~~
    python src/tools/convert_ks_to_coco.py
    ~~~
3. 模型训练
    - 重开一个终端，输入CenterNet_xing登录服务器
    ~~~
    tmux attach -t CenterNet
    ~~~
    - 开始训练
    ~~~
    python main.py ctdet --exp_id coco_dla --batch_size 32 --master_batch_size 15 --lr 1.25e-4  --gpus 0,1,2,3,4,5,6,7
    ~~~
    > 如果CenterNet session不存在，则在开始训练前执行以下命令
    ~~~
    tmux new -s CenterNet
    conda activate CenterNet
    ~~~
4. 模型测试
    - python demo.py ctdet --demo ./ --load_model ../exp/ctdet/coco_dla/model_best.pth
