# tf-pvanet
Implementation of [PVANet](https://arxiv.org/abs/1611.08588) in Tensorflow.
* When repeating the network structure in [caffe pt](https://github.com/sanghoon/pva-faster-rcnn/blob/master/models/pvanet/pva9.1/faster_rcnn_train_test_21cls.pt), the original layer names are carefully reserved. 
* The `caffe_to_tensorflow.py` converts pva model in caffe  to tensorflow ckpt according to the layer names. Caffe models can be downloaded via [this script](https://github.com/sanghoon/pva-faster-rcnn/blob/master/models/pvanet/download_all_models.sh).
* Only the conv stages from `conv1_1` to `conv5_4` are implemented, as an base network archietecture for any other CNN-based algorithms, more than faster-rcnn.
* The `util` module come from [pylib](https://github.com/dengdan/pylib). Add the path of its `src` directory to `PYTHONPATH` when using.
