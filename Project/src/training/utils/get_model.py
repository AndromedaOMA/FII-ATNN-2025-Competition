import timm
import sys
import detectors


def get_model(name, config):
    name = name.lower()
    if name == 'resnet50':
        # model = timm.create_model("hf_hub:anonauthors/cifar100-timm-resnet50", pretrained=config['model']['pretrained'])  # https://huggingface.co/anonauthors/cifar100-timm-resnet50
        model = timm.create_model("resnet50_cifar100", pretrained=True)     # https://huggingface.co/edadaltocg/resnet50_cifar100
        print('Model resnet50_cifar100 loaded!')
        return model
    elif name == 'resnet18':
        model = timm.create_model("resnet18_cifar100", pretrained=config['model']['pretrained'])    # https://huggingface.co/edadaltocg/resnet18_cifar100
        print('Model resnet18_cifar100 loaded!')
        return model
    elif name == 'resnest14d':
        model = timm.create_model("hf_hub:timm/resnest14d.gluon_in1k", pretrained=config['model']['pretrained'])
        print('Model resnest14d.gluon_in1k loaded!')
        return model
    elif name == 'resnest26d':
        model = timm.create_model("hf_hub:timm/resnest26d.gluon_in1k", pretrained=config['model']['pretrained'])
        print('Model resnest26d.gluon_in1k loaded!')
        return model
    elif name == 'efficientnet_b0' or name == 'efficientnetb0':
        drop_rate = config.get('model', {}).get('drop_rate', 0.0)
        model = timm.create_model(
            "efficientnet_b0", 
            pretrained=config['model']['pretrained'],
            num_classes=config['model']['num_classes'],
            drop_rate=drop_rate
        )
        print(f'Model efficientnet_b0 loaded with drop_rate={drop_rate}!')
        return model
    elif name == 'efficientnet_b0_ns' or name == 'efficientnetb0_ns' or name == 'tf_efficientnet_b0_ns':
        model = timm.create_model(
            "tf_efficientnet_b0_ns", 
            pretrained=config['model']['pretrained'],
            num_classes=config['model']['num_classes']
        )
        print('Model tf_efficientnet_b0_ns (Noisy Student) loaded!')
        return model
    elif name == 'MLP':
        pass
    else:
        print('The network name you have entered is not supported!')
        sys.exit()
