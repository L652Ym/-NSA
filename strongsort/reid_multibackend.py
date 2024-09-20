from collections import OrderedDict, namedtuple
from os.path import exists as file_exists
from pathlib import Path

import cv2
import gdown
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from yolov5.utils.general import LOGGER, check_version, check_requirements
from torchreid.utils import FeatureExtractor

from strongsort.deep.models import build_model
from strongsort.deep.reid_model_factory import (
    get_model_name,
    get_model_url,
    load_pretrained_weights,
    show_downloadeable_models,
)


# kadirnar: I added export_formats to the function
def export_formats():
    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights="osnet_x0_25_msmt17.pt", device=torch.device("cpu"), fp16=False):
        super().__init__()
        
        if not isinstance(device, torch.device):
            device = torch.device(device)
        
        w = str(weights[0] if isinstance(weights, list) else weights)
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.coreml,
            self.saved_model,
            self.pb,
            self.tflite,
            self.edgetpu,
            self.tfjs,
        ) = self.model_type(
            w
        )  # get backend
        self.fp16 = fp16
        self.fp16 &= (self.pt or self.jit or self.onnx or self.engine) and device.type != 'cpu'  # FP16

        # Build transform functions
        self.device = device
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()
        model_name = get_model_name(w)
        if self.pt:  # PyTorch
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            elif model_url is None:
                print(f"No URL associated to the chosen StrongSORT weights ({w}). Choose between:")
                show_downloadeable_models()
                exit()

            self.extractor = FeatureExtractor(
                # get rid of dataset information DeepSort model name
                model_name=model_name,
                model_path=weights,
                device=str(device)
            )
            
            self.extractor.model.half() if fp16 else  self.extractor.model.float()
        elif self.jit:
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            cuda = torch.cuda.is_available() and device.type != "cpu"
            # check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False  # default updated below
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            batch_size = self.bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif self.xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements(("openvino",))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCWH"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            self.executable_network = ie.compile_model(
                network, device_name="CPU"
            )  # device_name="MYRIAD" for Intel NCS2
            self.output_layer = next(iter(self.executable_network.outputs))

        elif self.tflite:
            LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            self.interpreter = tf.lite.Interpreter(model_path=w)
            self.interpreter.allocate_tensors()
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Test model on random input data.
            input_data = np.array(np.random.random_sample((1, 256, 128, 3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        else:
            print("This model framework is not supported yet!")
            exit()

    @staticmethod
    def model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = list(export_formats().Suffix) + [".xml"]  # export suffixes
        check_suffix(p, suffixes + [".pth"])  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2, _ = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

    def preprocess(self, im_batch):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32), size)

        im = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        im = im.float().to(device=self.device)
        return im

    def forward(self, im_batch):

        # preprocess batch
        im_batch = self.preprocess(im_batch)
        b, ch, h, w = im_batch.shape  # batch, channel, height, width
        features = []
        for i in range(0, im_batch.shape[0]):
            im = im_batch[i, :, :, :].unsqueeze(0)
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.pt:  # PyTorch
                y = self.extractor.model(im)[0]
            elif self.jit:  # TorchScript
                y = self.model(im)[0]
            elif self.onnx:  # ONNX Runtime
                im = im.permute(0, 1, 3, 2).cpu().numpy()  # torch to numpy
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
            elif self.xml:  # OpenVINO
                im = im.cpu().numpy()  # FP32
                y = self.executable_network([im])[self.output_layer]
            elif self.engine:  # TensorRT
                im = im.permute(0, 1, 3, 2)
                if self.dynamic and im.shape != self.bindings['images'].shape:
                    i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
                    self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
                    self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                    self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
                s = self.bindings['images'].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs['images'] = int(im.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = self.bindings['output'].data
            else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
                im = im.permute(0, 3, 2, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = torch.tensor(self.interpreter.get_tensor(output['index']))
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, device=self.device)
            features.append(y.squeeze())
    
        return features

    def warmup(self, imgsz=(1, 256, 128, 3)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb
        if any(warmup_types) and self.device.type != "cpu":
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            im = im.cpu().numpy()
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup
