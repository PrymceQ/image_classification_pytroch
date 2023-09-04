import torch
import argparse
import onnxruntime
import warnings
# Filter out the specific warning about "Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied."
warnings.filterwarnings("ignore", category=Warning, message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.")



def get_args():
    parser = argparse.ArgumentParser('Convert pth -> onnx')
    parser.add_argument('--model_path', type=str, default='weights/test/best.pth', help='model_path')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--saved_path', type=str, default='weights/test/model.onnx')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

if __name__ == '__main__':
    opt = get_args()

    # Load model
    model = load_checkpoint(opt.model_path)
    if torch.cuda.is_available():
        model.cuda()
    print('..... Finished loading model! ......')

    # Create dummy input
    dummy_input = torch.randn([1, 3, opt.img_size, opt.img_size])
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    print('..... Finished create dummy input {}! ......'.format(dummy_input.shape))

    print(f'Model output: {model(dummy_input).detach().cpu().numpy()}')

    # Save onnx
    torch.onnx.export(
        model,
        dummy_input,
        opt.saved_path,
        # do_constant_folding=False,
        input_names=["input"],
        output_names=['output'],
    )

    # Inference and check
    session = onnxruntime.InferenceSession(opt.saved_path, providers=['CPUExecutionProvider'])
    input_names = list(map(lambda x: x.name, session.get_inputs()))
    output_names = list(map(lambda x: x.name, session.get_outputs()))
    res = session.run(output_names, {input_names[0]: dummy_input.cpu().numpy()})
    print(f'Onnx model output (fp32): {res}')
