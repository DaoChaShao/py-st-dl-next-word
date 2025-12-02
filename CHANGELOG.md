<!-- insertion marker -->
<a name="0.1.0"></a>

## [0.1.0](https://github.com///compare/ce6dff8cae32df74d3531de065508043f04da989...0.1.0) (2025-12-03)

### Features

- add raw data synthesized_.jsonl ([52f7fe9](https://github.com///commit/52f7fe95d2e399fbfc815c5be6a83aedfb832fde))
- add uv.lock ([ecd389c](https://github.com///commit/ecd389c1e129e49cdb1f30a52f7ed9405ecdb504))
- add unet_sem_seg.py for binary segmentation with IoU evaluation and training management ([e4d0ed6](https://github.com///commit/e4d0ed67a547092e1e28546450a95d21a93bbc3c))
- add unet_focal.py for Focal Loss implementation in binary segmentation ([87e722a](https://github.com///commit/87e722a5d92bb065c64412945423979a8e52fd6f))
- add unet_edge.py for Edge Aware Loss implementation ([c3e5b97](https://github.com///commit/c3e5b9772c274d7fc124f1be2a196f94fc0354b0))
- add unet_dnf.py for combined Dice, BCE, and Focal loss implementation ([ade78e1](https://github.com///commit/ade78e1e51f6edd651126fd6b66541cd92c8a2f6))
- add unet_dice.py for Dice loss implementation combining BCE and Dice coefficients ([ff1d252](https://github.com///commit/ff1d2523fc9f2a6e9f689fc92c30234af8f54609))
- add unet_5layers_sem_seg.py for 5-layer UNet model implementation ([6a9570f](https://github.com///commit/6a9570f880cf516da2a274c676901beea37699b4))
- add unet_4layers_sem_seg.py for 4-layer UNet model implementation ([b7b7c06](https://github.com///commit/b7b7c06caf8bdd39efe3e0eb065352ffde810d05))
- add THU.py for text segmentation and POS tagging using THULAC ([79661ba](https://github.com///commit/79661ba4ae4b0bdb5019af034bc9f9808aba38a9))
- add stats.py for data handling and preprocessing utilities ([7eada72](https://github.com///commit/7eada72464d69f7f6e263826f0187664f1a17913))
- add TorchDataset4SeqPredictionNextStep class for handling sequential features and labels ([9fe3100](https://github.com///commit/9fe3100665d7cb9cd8b35221ee8753c2304b2fec))
- add TorchDataset4Seq2Classification class for handling sequential features and labels ([5bdecf6](https://github.com///commit/5bdecf6294948604941c5d03a6f8344339f0dcf0))
- add TorchDataset4SemanticSegmentation class for custom dataset handling in UNet model ([2521aaf](https://github.com///commit/2521aaf8a37315f695265d6a2dffb5e6ea6c1fd1))
- add TorchTrainer4Seq2Classification class for training and validating sequence classification models ([c52da0e](https://github.com///commit/c52da0ec13b3517d870b07a4aa72d77ec96064d7))
- add LSTMRNNForClassification model for multi-class classification tasks ([90230fd](https://github.com///commit/90230fd84984c0ef34707d8f9974375f16d798f5))
- add reshaper.py module for reshaping flattened tensors to grayscale ([b29d045](https://github.com///commit/b29d0458f4477cc862b9630c39953ad1546c0c47))
- add Chinese README.md with project overview, features, privacy notice, and environment setup instructions ([bcc524a](https://github.com///commit/bcc524aa83ab42fea981f4656c1ba0d122836022))
- add comprehensive README.md with project introduction, features, privacy notice, and environment setup instructions ([fbee717](https://github.com///commit/fbee71757c3ce4dfdf634c89b196216921a713cd))
- update pyproject.toml with new dependencies and configuration for git-changelog ([e5a8f70](https://github.com///commit/e5a8f70abed2eb9625f9bbb57526bc3d6b2edf73))
- add PT.py module for managing random seeds and device checks in PyTorch ([47ff939](https://github.com///commit/47ff939f5bba8a3910ea78eaf7b2f21dd426fbbc))
- add preprocessor.py module with main function placeholder ([327bf82](https://github.com///commit/327bf82fa4bc1f7a03dead5fb2f1f51c86f911dc))
- add preparer.py module with main function placeholder ([10b4a5a](https://github.com///commit/10b4a5a7dbc744930073ad33f6666829a3604910))
- add predictor.py module with main function placeholder ([aaa53fa](https://github.com///commit/aaa53fa5fc94c03a876a1c85905ab85f1fb67638))
- add nlp.py module for processing Chinese and English text with various NLP functions ([a082cb2](https://github.com///commit/a082cb2df38d62f1f914c1612835a6e556d3b1e3))
- add TorchTrainer4Regression class for training and validating regression models ([6583d3f](https://github.com///commit/6583d3f662e64cf91985816f24f381a0e6872be9))
- add regression_log_mse function for Log Mean Squared Error loss calculation ([62f5886](https://github.com///commit/62f5886b170108a88aeff83720326b3bbffb4d2a))
- add mask_mapper module for converting mask images to class indices ([6d6380c](https://github.com///commit/6d6380c75fa694b779a56cae7b0101ffea895c43))
- add logger module for logging training metrics and events ([8735310](https://github.com///commit/87353108c5bb9f7c4c643e04b4ec08b19cb02fa1))
- add custom PyTorch Dataset class for label classification ([a509163](https://github.com///commit/a509163526a96dd8002a1c2d8dbb8da70aa0e09d))
- add highlighter module with text formatting functions ([29203a6](https://github.com///commit/29203a63e854b3de8fdec137227fa978595af9db))
- add helper module with context managers for beautification, timing, and random seed management ([3726ec1](https://github.com///commit/3726ec17009715baa38efbb6f31ca190dae7c5fb))
- add evaluator module with main function stub ([c9b05a6](https://github.com///commit/c9b05a6bcfc328ae4dd6622465e6122af00867bb))
- add decorators for function output beautification and timing ([437d2f2](https://github.com///commit/437d2f28cabaf8e8b902a63db92a48c75544ced3))
- add custom PyTorch DataLoader class for dataset handling ([41a302c](https://github.com///commit/41a302c2a01ddb5766195f51d5921cbfe539f45f))
- add configuration module for UNet parameters and preprocessing ([54844c6](https://github.com///commit/54844c6186ca3df3559aa986cafbf827bf0847da))
- add configuration module for RNN parameters and preprocessing ([ec7e678](https://github.com///commit/ec7e678f3b25948d652f8f42e1c9b335a1af292b))
- add configuration module for MLP parameters and preprocessing ([27a8bd1](https://github.com///commit/27a8bd11eae893f5aeb9f4f577111700aff9a0ba))
- add configuration module for CNN parameters and preprocessing ([7d1ab4d](https://github.com///commit/7d1ab4d9573985f6fb23a69213cf70e383d2ffc1))
- add configuration module for data preprocessing and hyperparameters ([b38451c](https://github.com///commit/b38451ca70d1adc5ebb19f3610c0caf04a26b694))
- add configuration module with file paths and settings ([caeb796](https://github.com///commit/caeb7967bfeac7d47faa4bbfcfd8d7561a8c161d))
- add API wrappers for OpenAI text and image completion services ([f0222bc](https://github.com///commit/f0222bc616fb5def7b4371fd79dddde044830640))
- add Trainers module for specialized PyTorch training frameworks ([9dabb8f](https://github.com///commit/9dabb8f026ddf42971174ac0f128c006b4d93950))
- add ML/Data Processing Utility Module with comprehensive toolkit ([edeae2a](https://github.com///commit/edeae2a0d1665070544a20a613c91e5d1c840c15))
- add Trainers module with specialized PyTorch training implementations ([bc97431](https://github.com///commit/bc97431380b1ae038530f961e30a16d359084471))
- add neural network module with various architectures for ML tasks ([78da77a](https://github.com///commit/78da77a35cb5aba337ca60ae721fc6b89d9957cf))
- add Trainer module for neural network training implementations ([0a0bce2](https://github.com///commit/0a0bce2e5722a054d091e0d3a3155e0273aeaa08))
- add Dataloader module with PyTorch DataLoader wrapper ([7a78532](https://github.com///commit/7a785320039dbacc1160740f96b42245a9302df6))
- add criterion module with specialized PyTorch loss functions ([6d39e08](https://github.com///commit/6d39e088683537134eb2339c3d79fb87f4c325de))
- add initial configuration module for ML/Data Processing ([449baaa](https://github.com///commit/449baaa661870d5224a3414b25eedeb5aa53a978))

### Bug Fixes

- ensure main function is called and add newline at end of file ([536edf7](https://github.com///commit/536edf76390d9173a57def94cd9e459bfa3999e7))

### Chore

- add .gitignore file to exclude Python and IDE-specific files ([f5d9a90](https://github.com///commit/f5d9a903d79f2ae3556a2d075bbcc09942944da8))

